use crate::authority::{
    ensure_private_directory, ensure_private_parent, existing_regular_file_under_root,
    root_relative_path,
};
pub use crate::digest::sha256_file;
use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::security::{secure_dir, secure_file};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: &[u8; 8] = b"TIDEXD01";
const F64_MAGIC: &[u8; 8] = b"TIDEXF64";
const HEADER_BYTES: u64 = 16;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeltaArtifactRef {
    pub path: PathBuf,
    pub sha256: Sha256Digest,
    pub parameter_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct F64ArtifactRef {
    pub path: PathBuf,
    pub sha256: Sha256Digest,
    pub element_count: u64,
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

/// Validate the caller-supplied artifact root before any path is joined below
/// it.  The authority helpers protect every child component; this closes the
/// remaining root-level indirection, which would otherwise let a symlinked
/// root redirect the whole artifact store outside its declared authority.
fn verified_artifact_root(root: &Path) -> BrainResult<PathBuf> {
    if !root.is_absolute() {
        return Err(BrainError::Invalid("artifact_root_must_be_absolute".into()));
    }
    let metadata = fs::symlink_metadata(root)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err(BrainError::Integrity("artifact_root_invalid".into()));
    }
    Ok(root.canonicalize()?)
}

fn writable_artifact_root(root: &Path) -> BrainResult<PathBuf> {
    let root = verified_artifact_root(root)?;
    secure_dir(&root)?;
    Ok(root)
}

fn component_name(path: &Path) -> Option<&str> {
    path.file_name()?.to_str()
}

fn root_for_dvec_path(path: &Path) -> BrainResult<PathBuf> {
    if !path.is_absolute()
        || path.extension().and_then(|extension| extension.to_str()) != Some("dvec")
    {
        return Err(BrainError::Invalid("artifact_path_invalid".into()));
    }
    let parent = path
        .parent()
        .ok_or_else(|| BrainError::Invalid("artifact_path_parent_missing".into()))?;
    let root = if component_name(parent) == Some("by-sha") {
        let deltas = parent
            .parent()
            .ok_or_else(|| BrainError::Invalid("artifact_path_layout_invalid".into()))?;
        let artifacts = deltas
            .parent()
            .ok_or_else(|| BrainError::Invalid("artifact_path_layout_invalid".into()))?;
        if component_name(deltas) != Some("deltas")
            || component_name(artifacts) != Some("artifacts")
        {
            return Err(BrainError::Integrity("artifact_path_layout_invalid".into()));
        }
        artifacts
            .parent()
            .ok_or_else(|| BrainError::Invalid("artifact_root_missing".into()))?
    } else {
        let artifacts = parent
            .parent()
            .ok_or_else(|| BrainError::Invalid("artifact_path_layout_invalid".into()))?;
        if component_name(parent) != Some("deltas")
            || component_name(artifacts) != Some("artifacts")
        {
            return Err(BrainError::Integrity("artifact_path_layout_invalid".into()));
        }
        artifacts
            .parent()
            .ok_or_else(|| BrainError::Invalid("artifact_root_missing".into()))?
    };
    let root = verified_artifact_root(root)?;
    root_relative_path(&root, path)?;
    Ok(root)
}

fn root_for_f64_path(path: &Path) -> BrainResult<PathBuf> {
    if !path.is_absolute()
        || path.extension().and_then(|extension| extension.to_str()) != Some("f64bin")
    {
        return Err(BrainError::Invalid("f64_artifact_path_invalid".into()));
    }
    let by_sha = path
        .parent()
        .ok_or_else(|| BrainError::Invalid("f64_artifact_path_parent_missing".into()))?;
    let f64_dir = by_sha
        .parent()
        .ok_or_else(|| BrainError::Invalid("f64_artifact_path_layout_invalid".into()))?;
    let artifacts = f64_dir
        .parent()
        .ok_or_else(|| BrainError::Invalid("f64_artifact_path_layout_invalid".into()))?;
    if component_name(by_sha) != Some("by-sha")
        || component_name(f64_dir) != Some("f64")
        || component_name(artifacts) != Some("artifacts")
    {
        return Err(BrainError::Integrity(
            "f64_artifact_path_layout_invalid".into(),
        ));
    }
    let root = artifacts
        .parent()
        .ok_or_else(|| BrainError::Invalid("artifact_root_missing".into()))?;
    let root = verified_artifact_root(root)?;
    root_relative_path(&root, path)?;
    Ok(root)
}

fn open_verified_read(path: &Path) -> BrainResult<File> {
    // Authority validates every component before this open.  Linux's
    // O_NOFOLLOW closes the remaining leaf replacement window between that
    // validation and the descriptor acquisition; writes use create_new.
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    options.custom_flags(0o400000); // O_NOFOLLOW
    Ok(options.open(path)?)
}

fn temp_path(directory: &Path, label: &str) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    directory.join(format!(".{label}.{}.{}.tmp", std::process::id(), sequence))
}

/// Atomically install a newly-created immutable file without ever replacing an
/// existing destination.  `rename` is intentionally not used: on Unix it can
/// overwrite a digest-addressed object.  A hard link either installs this
/// exact inode or reports that a competing immutable object is already there.
fn install_new_immutable_file(
    root: &Path,
    temporary: &Path,
    destination: &Path,
) -> BrainResult<bool> {
    ensure_private_parent(root, destination)?;
    let temporary = existing_regular_file_under_root(root, temporary)?;
    match fs::hard_link(&temporary, destination) {
        Ok(()) => {
            secure_file(destination)?;
            fs::remove_file(&temporary)?;
            existing_regular_file_under_root(root, destination)?;
            Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            // Reject a link/directory/etc. at the collision target instead of
            // treating its mere existence as a valid immutable object.
            existing_regular_file_under_root(root, destination)?;
            fs::remove_file(&temporary)?;
            Ok(false)
        }
        Err(error) => Err(error.into()),
    }
}

fn existing_regular_file_if_present(root: &Path, path: &Path) -> BrainResult<Option<PathBuf>> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(Some(existing_regular_file_under_root(root, path)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn sha256_verified_file_under_root(root: &Path, path: &Path) -> BrainResult<Sha256Digest> {
    let path = existing_regular_file_under_root(root, path)?;
    let mut reader = BufReader::with_capacity(1 << 20, open_verified_read(&path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1 << 20];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Sha256Digest::parse(format!("{:x}", hasher.finalize()))
}

fn artifact_dir(root: &Path) -> PathBuf {
    root.join("artifacts").join("deltas")
}
fn content_artifact_dir(root: &Path) -> PathBuf {
    artifact_dir(root).join("by-sha")
}
fn content_output_path(root: &Path, digest: &Sha256Digest) -> PathBuf {
    content_artifact_dir(root).join(format!("{digest}.dvec"))
}
fn output_path(root: &Path, id: &str) -> BrainResult<PathBuf> {
    if !valid_id(id) {
        return Err(BrainError::Invalid("artifact_id_invalid".into()));
    }
    Ok(artifact_dir(root).join(format!("{id}.dvec")))
}
fn read_header<R: Read>(r: &mut R) -> BrainResult<u64> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(BrainError::Integrity("delta_artifact_magic_invalid".into()));
    }
    let mut c = [0u8; 8];
    r.read_exact(&mut c)?;
    Ok(u64::from_le_bytes(c))
}
fn write_header<W: Write>(w: &mut W, count: u64) -> BrainResult<()> {
    w.write_all(MAGIC)?;
    w.write_all(&count.to_le_bytes())?;
    Ok(())
}

fn content_dvec_digest_from_path(path: &Path) -> BrainResult<Option<Sha256Digest>> {
    if component_name(path.parent().unwrap_or_else(|| Path::new(""))) != Some("by-sha") {
        return Ok(None);
    }
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| BrainError::Integrity("content_addressed_artifact_name_invalid".into()))?;
    Ok(Some(Sha256Digest::parse(stem)?))
}

fn verified_dvec_path_under_root(root: &Path, path: &Path) -> BrainResult<PathBuf> {
    let root = verified_artifact_root(root)?;
    if root_for_dvec_path(path)? != root {
        return Err(BrainError::Integrity("artifact_path_root_mismatch".into()));
    }
    existing_regular_file_under_root(&root, path)
}

fn inspect_dvec_under_root(root: &Path, path: &Path) -> BrainResult<DeltaArtifactRef> {
    let path = verified_dvec_path_under_root(root, path)?;
    let file = open_verified_read(&path)?;
    let size = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let count = read_header(&mut reader)?;
    let expected = HEADER_BYTES
        .checked_add(
            count
                .checked_mul(4)
                .ok_or_else(|| BrainError::Invalid("artifact_size_overflow".into()))?,
        )
        .ok_or_else(|| BrainError::Invalid("artifact_size_overflow".into()))?;
    if size != expected {
        return Err(BrainError::Integrity(format!(
            "artifact_size_mismatch:{size}:{expected}"
        )));
    }
    reader.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1 << 20];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let sha256 = Sha256Digest::parse(format!("{:x}", hasher.finalize()))?;
    if let Some(named_digest) = content_dvec_digest_from_path(&path)? {
        if named_digest != sha256 {
            return Err(BrainError::Integrity(
                "content_addressed_artifact_name_digest_mismatch".into(),
            ));
        }
    }
    Ok(DeltaArtifactRef {
        path,
        sha256,
        parameter_count: count,
    })
}

fn verified_delta_reference_under_root(
    root: &Path,
    reference: &DeltaArtifactRef,
) -> BrainResult<PathBuf> {
    let inspected = inspect_dvec_under_root(root, &reference.path)?;
    if inspected.sha256 != reference.sha256
        || inspected.parameter_count != reference.parameter_count
    {
        return Err(BrainError::Integrity("artifact_reference_mismatch".into()));
    }
    if content_dvec_digest_from_path(&inspected.path)?.is_some()
        && inspected.path != content_output_path(root, &reference.sha256)
    {
        return Err(BrainError::Integrity(
            "content_addressed_artifact_reference_path_mismatch".into(),
        ));
    }
    Ok(inspected.path)
}

fn write_dvec_values(path: &Path, values: &[f32]) -> BrainResult<()> {
    let mut writer = BufWriter::with_capacity(
        1 << 20,
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?,
    );
    write_header(&mut writer, values.len() as u64)?;
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    secure_file(path)
}

fn dvec_matches_values(root: &Path, path: &Path, values: &[f32]) -> BrainResult<bool> {
    let inspected = inspect_dvec_under_root(root, path)?;
    if inspected.parameter_count != values.len() as u64 {
        return Ok(false);
    }
    let path = verified_dvec_path_under_root(root, path)?;
    let mut reader = BufReader::with_capacity(1 << 20, open_verified_read(&path)?);
    if read_header(&mut reader)? != values.len() as u64 {
        return Ok(false);
    }
    let mut raw = [0u8; 4];
    for value in values {
        reader.read_exact(&mut raw)?;
        if raw != value.to_le_bytes() {
            return Ok(false);
        }
    }
    Ok(true)
}

pub fn create_dvec(root: &Path, id: &str, values: &[f32]) -> BrainResult<DeltaArtifactRef> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(BrainError::Invalid("artifact_value_non_finite".into()));
    }
    let root = writable_artifact_root(root)?;
    let dir = ensure_private_directory(&root, &artifact_dir(&root))?;
    let path = output_path(&root, id)?;
    if existing_regular_file_if_present(&root, &path)?.is_some() {
        return Err(BrainError::Integrity("artifact_already_exists".into()));
    }
    let temporary = temp_path(&dir, id);
    write_dvec_values(&temporary, values)?;
    if !install_new_immutable_file(&root, &temporary, &path)? {
        return Err(BrainError::Integrity("artifact_already_exists".into()));
    }
    inspect_dvec_under_root(&root, &path)
}

pub fn read_dvec_f32(reference: &DeltaArtifactRef) -> BrainResult<Vec<f32>> {
    let root = root_for_dvec_path(&reference.path)?;
    let path = verified_delta_reference_under_root(&root, reference)?;
    let mut reader = BufReader::with_capacity(1 << 20, open_verified_read(&path)?);
    let count = read_header(&mut reader)?;
    let mut values = Vec::with_capacity(count as usize);
    let mut raw = [0u8; 4];
    for _ in 0..count {
        reader.read_exact(&mut raw)?;
        let value = f32::from_le_bytes(raw);
        if !value.is_finite() {
            return Err(BrainError::Integrity("artifact_non_finite".into()));
        }
        values.push(value);
    }
    Ok(values)
}

pub fn create_content_addressed_dvec(root: &Path, values: &[f32]) -> BrainResult<DeltaArtifactRef> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(BrainError::Invalid("artifact_values_invalid".into()));
    }
    let mut hasher = Sha256::new();
    hasher.update(MAGIC);
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    let digest = Sha256Digest::parse(format!("{:x}", hasher.finalize()))?;
    let root = writable_artifact_root(root)?;
    let dir = ensure_private_directory(&root, &content_artifact_dir(&root))?;
    let path = content_output_path(&root, &digest);
    if existing_regular_file_if_present(&root, &path)?.is_some() {
        let existing = inspect_dvec_under_root(&root, &path)?;
        if existing.sha256 != digest
            || existing.parameter_count != values.len() as u64
            || !dvec_matches_values(&root, &path, values)?
        {
            return Err(BrainError::Integrity(
                "content_addressed_artifact_collision".into(),
            ));
        }
        return Ok(existing);
    }
    let temporary = temp_path(&dir, digest.as_str());
    write_dvec_values(&temporary, values)?;
    if !install_new_immutable_file(&root, &temporary, &path)? {
        let existing = inspect_dvec_under_root(&root, &path)?;
        if existing.sha256 != digest
            || existing.parameter_count != values.len() as u64
            || !dvec_matches_values(&root, &path, values)?
        {
            return Err(BrainError::Integrity(
                "content_addressed_artifact_collision".into(),
            ));
        }
        return Ok(existing);
    }
    let inspected = inspect_dvec_under_root(&root, &path)?;
    if inspected.sha256 != digest
        || inspected.parameter_count != values.len() as u64
        || !dvec_matches_values(&root, &path, values)?
    {
        return Err(BrainError::Integrity(
            "content_addressed_artifact_write_mismatch".into(),
        ));
    }
    Ok(inspected)
}

pub fn inspect_dvec(path: &Path) -> BrainResult<DeltaArtifactRef> {
    let root = root_for_dvec_path(path)?;
    inspect_dvec_under_root(&root, path)
}

fn f64_artifact_dir(root: &Path) -> PathBuf {
    root.join("artifacts").join("f64").join("by-sha")
}

fn f64_content_output_path(root: &Path, digest: &Sha256Digest) -> PathBuf {
    f64_artifact_dir(root).join(format!("{digest}.f64bin"))
}

fn read_f64_header<R: Read>(reader: &mut R) -> BrainResult<u64> {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != F64_MAGIC {
        return Err(BrainError::Integrity("f64_artifact_magic_invalid".into()));
    }
    let mut count = [0u8; 8];
    reader.read_exact(&mut count)?;
    Ok(u64::from_le_bytes(count))
}

fn f64_digest_from_path(path: &Path) -> BrainResult<Sha256Digest> {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| BrainError::Integrity("f64_content_addressed_name_invalid".into()))?;
    Sha256Digest::parse(stem)
}

fn verified_f64_path_under_root(root: &Path, path: &Path) -> BrainResult<PathBuf> {
    let root = verified_artifact_root(root)?;
    if root_for_f64_path(path)? != root {
        return Err(BrainError::Integrity(
            "f64_artifact_path_root_mismatch".into(),
        ));
    }
    existing_regular_file_under_root(&root, path)
}

fn inspect_f64_artifact_under_root(root: &Path, path: &Path) -> BrainResult<F64ArtifactRef> {
    let path = verified_f64_path_under_root(root, path)?;
    let file = open_verified_read(&path)?;
    let size = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let count = read_f64_header(&mut reader)?;
    let expected = HEADER_BYTES
        .checked_add(
            count
                .checked_mul(8)
                .ok_or_else(|| BrainError::Invalid("f64_artifact_size_overflow".into()))?,
        )
        .ok_or_else(|| BrainError::Invalid("f64_artifact_size_overflow".into()))?;
    if size != expected {
        return Err(BrainError::Integrity(format!(
            "f64_artifact_size_mismatch:{size}:{expected}"
        )));
    }
    reader.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1 << 20];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let sha256 = Sha256Digest::parse(format!("{:x}", hasher.finalize()))?;
    if f64_digest_from_path(&path)? != sha256 {
        return Err(BrainError::Integrity(
            "f64_content_addressed_name_digest_mismatch".into(),
        ));
    }
    Ok(F64ArtifactRef {
        path,
        sha256,
        element_count: count,
    })
}

fn verified_f64_reference_under_root(
    root: &Path,
    reference: &F64ArtifactRef,
) -> BrainResult<PathBuf> {
    let inspected = inspect_f64_artifact_under_root(root, &reference.path)?;
    if inspected.sha256 != reference.sha256 || inspected.element_count != reference.element_count {
        return Err(BrainError::Integrity(
            "f64_artifact_reference_mismatch".into(),
        ));
    }
    if inspected.path != f64_content_output_path(root, &reference.sha256) {
        return Err(BrainError::Integrity(
            "f64_content_addressed_reference_path_mismatch".into(),
        ));
    }
    Ok(inspected.path)
}

fn write_f64_values(path: &Path, values: &[f64]) -> BrainResult<()> {
    let mut writer = BufWriter::with_capacity(
        1 << 20,
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?,
    );
    writer.write_all(F64_MAGIC)?;
    writer.write_all(&(values.len() as u64).to_le_bytes())?;
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    secure_file(path)
}

fn f64_matches_values(root: &Path, path: &Path, values: &[f64]) -> BrainResult<bool> {
    let inspected = inspect_f64_artifact_under_root(root, path)?;
    if inspected.element_count != values.len() as u64 {
        return Ok(false);
    }
    let path = verified_f64_path_under_root(root, path)?;
    let mut reader = BufReader::with_capacity(1 << 20, open_verified_read(&path)?);
    if read_f64_header(&mut reader)? != values.len() as u64 {
        return Ok(false);
    }
    let mut raw = [0u8; 8];
    for value in values {
        reader.read_exact(&mut raw)?;
        if raw != value.to_le_bytes() {
            return Ok(false);
        }
    }
    Ok(true)
}

pub fn inspect_f64_artifact(path: &Path) -> BrainResult<F64ArtifactRef> {
    let root = root_for_f64_path(path)?;
    inspect_f64_artifact_under_root(&root, path)
}

pub fn create_content_addressed_f64(root: &Path, values: &[f64]) -> BrainResult<F64ArtifactRef> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(BrainError::Invalid("f64_artifact_values_invalid".into()));
    }
    let mut hasher = Sha256::new();
    hasher.update(F64_MAGIC);
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    let digest = Sha256Digest::parse(format!("{:x}", hasher.finalize()))?;
    let root = writable_artifact_root(root)?;
    let dir = ensure_private_directory(&root, &f64_artifact_dir(&root))?;
    let path = f64_content_output_path(&root, &digest);
    if existing_regular_file_if_present(&root, &path)?.is_some() {
        let existing = inspect_f64_artifact_under_root(&root, &path)?;
        if existing.sha256 != digest
            || existing.element_count != values.len() as u64
            || !f64_matches_values(&root, &path, values)?
        {
            return Err(BrainError::Integrity("f64_artifact_collision".into()));
        }
        return Ok(existing);
    }
    let temporary = temp_path(&dir, digest.as_str());
    write_f64_values(&temporary, values)?;
    if !install_new_immutable_file(&root, &temporary, &path)? {
        let existing = inspect_f64_artifact_under_root(&root, &path)?;
        if existing.sha256 != digest
            || existing.element_count != values.len() as u64
            || !f64_matches_values(&root, &path, values)?
        {
            return Err(BrainError::Integrity("f64_artifact_collision".into()));
        }
        return Ok(existing);
    }
    let reference = inspect_f64_artifact_under_root(&root, &path)?;
    if reference.sha256 != digest
        || reference.element_count != values.len() as u64
        || !f64_matches_values(&root, &path, values)?
    {
        return Err(BrainError::Integrity("f64_artifact_write_mismatch".into()));
    }
    Ok(reference)
}

pub fn read_f64_artifact(reference: &F64ArtifactRef) -> BrainResult<Vec<f64>> {
    let root = root_for_f64_path(&reference.path)?;
    let path = verified_f64_reference_under_root(&root, reference)?;
    let mut reader = BufReader::with_capacity(1 << 20, open_verified_read(&path)?);
    let count = read_f64_header(&mut reader)?;
    let mut values = Vec::with_capacity(count as usize);
    let mut raw = [0u8; 8];
    for _ in 0..count {
        reader.read_exact(&mut raw)?;
        let value = f64::from_le_bytes(raw);
        if !value.is_finite() {
            return Err(BrainError::Integrity("f64_artifact_non_finite".into()));
        }
        values.push(value);
    }
    Ok(values)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Streaming CountSketch. Memory is O(sketch_dim), independent of model size.
pub fn sketch_dvec(path: &Path, sketch_dim: usize, seed: u64) -> BrainResult<Vec<f64>> {
    if sketch_dim < 16 {
        return Err(BrainError::Invalid("sketch_dimension_too_small".into()));
    }
    let root = root_for_dvec_path(path)?;
    let path = verified_dvec_path_under_root(&root, path)?;
    // Validate the complete immutable object before the streaming pass.
    inspect_dvec_under_root(&root, &path)?;
    let mut r = BufReader::with_capacity(1 << 20, open_verified_read(&path)?);
    let count = read_header(&mut r)?;
    let mut out = vec![0.0; sketch_dim];
    let mut raw = [0u8; 4];
    for i in 0..count {
        r.read_exact(&mut raw)?;
        let v = f32::from_le_bytes(raw) as f64;
        if !v.is_finite() {
            return Err(BrainError::Integrity("artifact_non_finite".into()));
        }
        let h = splitmix64(i ^ seed);
        let bucket = (h as usize) % sketch_dim;
        let sign = if h & 1 == 0 { 1.0 } else { -1.0 };
        out[bucket] += sign * v;
    }
    Ok(out)
}

/// Read a bounded contiguous parameter range from an immutable delta artifact.
/// This enables tensor/block tomography with memory proportional to one model
/// block rather than the full parameter vector.
pub fn read_dvec_range(path: &Path, start: u64, len: usize) -> BrainResult<Vec<f64>> {
    let root = root_for_dvec_path(path)?;
    let path = verified_dvec_path_under_root(&root, path)?;
    let inspected = inspect_dvec_under_root(&root, &path)?;
    let end = start
        .checked_add(len as u64)
        .ok_or_else(|| BrainError::Invalid("artifact_range_overflow".into()))?;
    if end > inspected.parameter_count {
        return Err(BrainError::Invalid("artifact_range_out_of_bounds".into()));
    }
    let mut file = open_verified_read(&path)?;
    let byte_offset = HEADER_BYTES
        .checked_add(
            start
                .checked_mul(4)
                .ok_or_else(|| BrainError::Invalid("artifact_range_overflow".into()))?,
        )
        .ok_or_else(|| BrainError::Invalid("artifact_range_overflow".into()))?;
    file.seek(SeekFrom::Start(byte_offset))?;
    let mut reader = BufReader::with_capacity((len * 4).clamp(4096, 1 << 20), file);
    let mut values = Vec::with_capacity(len);
    let mut raw = [0u8; 4];
    for _ in 0..len {
        reader.read_exact(&mut raw)?;
        let value = f32::from_le_bytes(raw) as f64;
        if !value.is_finite() {
            return Err(BrainError::Integrity("artifact_non_finite".into()));
        }
        values.push(value);
    }
    Ok(values)
}

fn combination_readers(
    root: &Path,
    sources: &[(DeltaArtifactRef, f64)],
) -> BrainResult<(Vec<BufReader<File>>, u64)> {
    if sources.is_empty() {
        return Err(BrainError::Invalid("artifact_combine_empty".into()));
    }
    if sources
        .iter()
        .any(|(_, coefficient)| !coefficient.is_finite())
    {
        return Err(BrainError::Invalid(
            "artifact_combine_coefficient_non_finite".into(),
        ));
    }
    let mut readers = Vec::with_capacity(sources.len());
    let mut count = None;
    for (reference, _) in sources {
        let path = verified_delta_reference_under_root(root, reference)?;
        let inspected = inspect_dvec_under_root(root, &path)?;
        if let Some(expected) = count {
            if expected != inspected.parameter_count {
                return Err(BrainError::Invalid(
                    "artifact_combine_parameter_count_mismatch".into(),
                ));
            }
        } else {
            count = Some(inspected.parameter_count);
        }
        let mut reader = BufReader::with_capacity(1 << 20, open_verified_read(&path)?);
        let header_count = read_header(&mut reader)?;
        if header_count != inspected.parameter_count {
            return Err(BrainError::Integrity(
                "artifact_header_reference_count_mismatch".into(),
            ));
        }
        readers.push(reader);
    }
    Ok((readers, count.unwrap()))
}

/// Stream exactly the f32 payload bytes of a linear combination.
///
/// Both materialization and verification use this one arithmetic path so a
/// verifier cannot silently accept a result generated with different rounding
/// or source-validation semantics.
fn stream_linear_combination_bytes<F>(
    readers: &mut [BufReader<File>],
    count: u64,
    sources: &[(DeltaArtifactRef, f64)],
    mut consume: F,
) -> BrainResult<()>
where
    F: FnMut([u8; 4]) -> BrainResult<()>,
{
    if readers.len() != sources.len() {
        return Err(BrainError::Integrity(
            "artifact_combine_reader_source_mismatch".into(),
        ));
    }
    let mut raw = vec![[0u8; 4]; readers.len()];
    for _ in 0..count {
        let mut sum = 0.0f64;
        for index in 0..readers.len() {
            readers[index].read_exact(&mut raw[index])?;
            let value = f32::from_le_bytes(raw[index]);
            if !value.is_finite() {
                return Err(BrainError::Integrity("artifact_non_finite".into()));
            }
            sum += sources[index].1 * f64::from(value);
        }
        if !sum.is_finite() || sum.abs() > f32::MAX as f64 {
            return Err(BrainError::Numerical(
                "artifact_combine_nonfinite_or_overflow".into(),
            ));
        }
        consume((sum as f32).to_le_bytes())?;
    }
    Ok(())
}

fn write_linear_combination(
    root: &Path,
    path: &Path,
    sources: &[(DeltaArtifactRef, f64)],
) -> BrainResult<u64> {
    let (mut readers, count) = combination_readers(root, sources)?;
    let mut writer = BufWriter::with_capacity(
        1 << 20,
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?,
    );
    write_header(&mut writer, count)?;
    stream_linear_combination_bytes(&mut readers, count, sources, |bytes| {
        writer.write_all(&bytes)?;
        Ok(())
    })?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    secure_file(path)?;
    Ok(count)
}

fn linear_combination_matches(
    root: &Path,
    path: &Path,
    sources: &[(DeltaArtifactRef, f64)],
    expected_count: u64,
) -> BrainResult<bool> {
    let (mut readers, count) = combination_readers(root, sources)?;
    if count != expected_count {
        return Ok(false);
    }
    let inspected = inspect_dvec_under_root(root, path)?;
    if inspected.parameter_count != count {
        return Ok(false);
    }
    let path = verified_dvec_path_under_root(root, path)?;
    let mut target = BufReader::with_capacity(1 << 20, open_verified_read(&path)?);
    if read_header(&mut target)? != count {
        return Ok(false);
    }
    let mut matches = true;
    stream_linear_combination_bytes(&mut readers, count, sources, |expected| {
        let mut actual = [0u8; 4];
        target.read_exact(&mut actual)?;
        if actual != expected {
            matches = false;
        }
        Ok(())
    })?;
    Ok(matches)
}

/// Derive the immutable content-addressed reference for a dense linear
/// combination without creating, changing, or deleting any filesystem entry.
///
/// The source artifacts are read and validated exactly as they are for
/// [`combine_content_addressed_dvec`]. The returned digest is calculated over
/// the same dvec header and f32 payload bytes that materialization would
/// write, including its f64 accumulation and final f32 rounding.
pub fn derive_content_addressed_dvec_combination(
    root: &Path,
    sources: &[(DeltaArtifactRef, f64)],
) -> BrainResult<DeltaArtifactRef> {
    let root = verified_artifact_root(root)?;
    let (mut readers, parameter_count) = combination_readers(&root, sources)?;
    let mut hasher = Sha256::new();
    hasher.update(MAGIC);
    hasher.update(parameter_count.to_le_bytes());
    stream_linear_combination_bytes(&mut readers, parameter_count, sources, |bytes| {
        hasher.update(bytes);
        Ok(())
    })?;
    let sha256 = Sha256Digest::parse(format!("{:x}", hasher.finalize()))?;
    Ok(DeltaArtifactRef {
        path: content_output_path(&root, &sha256),
        sha256,
        parameter_count,
    })
}

pub fn combine_content_addressed_dvec(
    root: &Path,
    sources: &[(DeltaArtifactRef, f64)],
) -> BrainResult<DeltaArtifactRef> {
    let root = writable_artifact_root(root)?;
    let dir = ensure_private_directory(&root, &content_artifact_dir(&root))?;
    let descriptor = {
        let mut hasher = Sha256::new();
        hasher.update(b"CEREBRO:TIDEX:LINEAR-COMBINATION-TEMP:v1\0");
        for (reference, coefficient) in sources {
            hasher.update(reference.sha256.as_bytes());
            hasher.update(reference.parameter_count.to_be_bytes());
            hasher.update(coefficient.to_bits().to_be_bytes());
        }
        format!("{:x}", hasher.finalize())
    };
    let temporary = temp_path(&dir, &descriptor);
    let count = write_linear_combination(&root, &temporary, sources)?;
    let digest = sha256_verified_file_under_root(&root, &temporary)?;
    let final_path = content_output_path(&root, &digest);
    if existing_regular_file_if_present(&root, &final_path)?.is_some() {
        let existing = inspect_dvec_under_root(&root, &final_path)?;
        if existing.sha256 != digest
            || existing.parameter_count != count
            || !linear_combination_matches(&root, &final_path, sources, count)?
        {
            fs::remove_file(&temporary)?;
            return Err(BrainError::Integrity(
                "content_addressed_combination_collision".into(),
            ));
        }
        fs::remove_file(&temporary)?;
        return Ok(existing);
    }
    if !install_new_immutable_file(&root, &temporary, &final_path)? {
        let existing = inspect_dvec_under_root(&root, &final_path)?;
        if existing.sha256 != digest
            || existing.parameter_count != count
            || !linear_combination_matches(&root, &final_path, sources, count)?
        {
            return Err(BrainError::Integrity(
                "content_addressed_combination_collision".into(),
            ));
        }
        return Ok(existing);
    }
    let result = inspect_dvec_under_root(&root, &final_path)?;
    if result.sha256 != digest
        || result.parameter_count != count
        || !linear_combination_matches(&root, &final_path, sources, count)?
    {
        return Err(BrainError::Integrity(
            "content_addressed_combination_write_mismatch".into(),
        ));
    }
    Ok(result)
}

pub fn combine_dvec(
    root: &Path,
    id: &str,
    sources: &[(DeltaArtifactRef, f64)],
) -> BrainResult<DeltaArtifactRef> {
    let root = writable_artifact_root(root)?;
    let dir = ensure_private_directory(&root, &artifact_dir(&root))?;
    let path = output_path(&root, id)?;
    if existing_regular_file_if_present(&root, &path)?.is_some() {
        return Err(BrainError::Integrity("artifact_already_exists".into()));
    }
    let temporary = temp_path(&dir, id);
    let count = write_linear_combination(&root, &temporary, sources)?;
    if !install_new_immutable_file(&root, &temporary, &path)? {
        return Err(BrainError::Integrity("artifact_already_exists".into()));
    }
    let result = inspect_dvec_under_root(&root, &path)?;
    if result.parameter_count != count {
        return Err(BrainError::Integrity(
            "artifact_combine_written_count_mismatch".into(),
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn isolated_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "cerebro-artifact-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock before epoch")
                .as_nanos()
        ))
    }

    #[test]
    fn derives_exact_content_addressed_combination_without_materializing() {
        let root = isolated_root("derive-combination");
        fs::create_dir_all(&root).unwrap();
        let first = create_dvec(&root, "first", &[0.1, -3.0, 7.25]).unwrap();
        let second = create_dvec(&root, "second", &[2.0, 0.125, -0.5]).unwrap();
        let sources = vec![(first, 1.0 / 3.0), (second, -1.75)];
        let content_dir = root.join("artifacts/deltas/by-sha");

        assert!(!content_dir.exists());
        let derived = derive_content_addressed_dvec_combination(&root, &sources).unwrap();
        assert!(!content_dir.exists());
        assert!(!derived.path.exists());

        let materialized = combine_content_addressed_dvec(&root, &sources).unwrap();
        assert_eq!(derived, materialized);
        assert_eq!(read_dvec_f32(&materialized).unwrap().len() as u64, 3);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn content_addressed_dvec_rejects_parent_and_leaf_symlink_traversal() {
        let root = isolated_root("dvec-symlink");
        let outside = isolated_root("dvec-outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();

        symlink(&outside, root.join("artifacts")).unwrap();
        assert!(create_content_addressed_dvec(&root, &[1.0]).is_err());
        assert!(fs::read_dir(&outside).unwrap().next().is_none());
        fs::remove_file(root.join("artifacts")).unwrap();

        let artifact = create_content_addressed_dvec(&root, &[1.0, -2.5]).unwrap();
        let real_artifacts = root.join("real-artifacts");
        fs::rename(root.join("artifacts"), &real_artifacts).unwrap();
        symlink(&real_artifacts, root.join("artifacts")).unwrap();
        assert!(read_dvec_f32(&artifact).is_err());

        fs::remove_file(root.join("artifacts")).unwrap();
        fs::rename(&real_artifacts, root.join("artifacts")).unwrap();
        let outside_leaf = outside.join("outside.dvec");
        fs::write(&outside_leaf, b"not-an-artifact").unwrap();
        fs::remove_file(&artifact.path).unwrap();
        symlink(&outside_leaf, &artifact.path).unwrap();
        assert!(read_dvec_f32(&artifact).is_err());

        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn content_addressed_artifacts_never_replace_nonidentical_existing_bytes() {
        let root = isolated_root("content-collision");
        fs::create_dir_all(&root).unwrap();

        let dvec = create_content_addressed_dvec(&root, &[0.25, -3.0]).unwrap();
        assert_eq!(
            create_content_addressed_dvec(&root, &[0.25, -3.0]).unwrap(),
            dvec
        );
        fs::write(&dvec.path, b"different bytes").unwrap();
        let dvec_before = fs::read(&dvec.path).unwrap();
        assert!(create_content_addressed_dvec(&root, &[0.25, -3.0]).is_err());
        assert_eq!(fs::read(&dvec.path).unwrap(), dvec_before);

        let f64 = create_content_addressed_f64(&root, &[0.25, -3.0]).unwrap();
        assert_eq!(
            create_content_addressed_f64(&root, &[0.25, -3.0]).unwrap(),
            f64
        );
        fs::write(&f64.path, b"different f64 bytes").unwrap();
        let f64_before = fs::read(&f64.path).unwrap();
        assert!(create_content_addressed_f64(&root, &[0.25, -3.0]).is_err());
        assert_eq!(fs::read(&f64.path).unwrap(), f64_before);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn content_addressed_combination_refuses_to_replace_a_tampered_object() {
        let root = isolated_root("combination-collision");
        fs::create_dir_all(&root).unwrap();
        let source = create_dvec(&root, "source", &[1.5, -2.0]).unwrap();
        let sources = vec![(source, 0.5)];
        let combined = combine_content_addressed_dvec(&root, &sources).unwrap();
        fs::write(&combined.path, b"different combination bytes").unwrap();
        let before = fs::read(&combined.path).unwrap();

        assert!(combine_content_addressed_dvec(&root, &sources).is_err());
        assert_eq!(fs::read(&combined.path).unwrap(), before);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn content_addressed_f64_rejects_parent_and_leaf_symlink_traversal() {
        let root = isolated_root("f64-symlink");
        let outside = isolated_root("f64-outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();

        let artifact = create_content_addressed_f64(&root, &[1.0, -2.5]).unwrap();
        let f64_parent = root.join("artifacts/f64");
        let real_f64_parent = root.join("real-f64");
        fs::rename(&f64_parent, &real_f64_parent).unwrap();
        symlink(&real_f64_parent, &f64_parent).unwrap();
        assert!(read_f64_artifact(&artifact).is_err());

        fs::remove_file(&f64_parent).unwrap();
        fs::rename(&real_f64_parent, &f64_parent).unwrap();
        let outside_leaf = outside.join("outside.f64bin");
        fs::write(&outside_leaf, b"not-an-artifact").unwrap();
        fs::remove_file(&artifact.path).unwrap();
        symlink(&outside_leaf, &artifact.path).unwrap();
        assert!(read_f64_artifact(&artifact).is_err());

        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }
}
