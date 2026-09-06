use crate::digest::{sha256_file, Sha256Digest};
use crate::error::{BrainError, BrainResult};
use crate::security::{secure_dir, secure_file};
use rustix::fs::{
    flock, fstat, open, openat2, renameat_with, FileType, FlockOperation, Mode, OFlags,
    RenameFlags, ResolveFlags, Stat,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_IMMUTABLE_TEMPORARY: AtomicU64 = AtomicU64::new(0);
const PRIVATE_READ_BUFFER_BYTES: usize = 64 * 1024;
const PRIVATE_RESOLUTION: ResolveFlags = ResolveFlags::BENEATH
    .union(ResolveFlags::NO_SYMLINKS)
    .union(ResolveFlags::NO_MAGICLINKS);

/// Reserve a unique private staging file beside an immutable destination.
///
/// The returned descriptor is already opened with `create_new` and mode 0600,
/// so streaming producers can write directly into it without a path-open race.
/// [`install_private_immutable_file`] performs the final sync, digest check and
/// no-overwrite installation.
pub fn create_private_staging_file(
    root: &Path,
    destination: &Path,
) -> BrainResult<(PathBuf, File)> {
    ensure_private_parent(root, destination)?;
    let parent = destination
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?;
    let parent = existing_directory_under_root(root, parent)?;
    let file_name = destination
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| BrainError::Invalid("private_file_name_invalid".into()))?;
    loop {
        let sequence = NEXT_IMMUTABLE_TEMPORARY.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{file_name}.{}.{}.immutable.tmp",
            std::process::id(),
            sequence
        ));
        root_relative_path(root, &temporary)?;
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)
        {
            Ok(file) => return Ok((temporary, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
}

/// Produce a complete private staging file through a streaming callback.
/// Failed producers never leave a partial staging object behind.
pub fn stage_private_file<F>(
    root: &Path,
    destination: &Path,
    produce: F,
) -> BrainResult<(PathBuf, Sha256Digest)>
where
    F: FnOnce(&mut File) -> BrainResult<()>,
{
    let (temporary, mut file) = create_private_staging_file(root, destination)?;
    let result = produce(&mut file)
        .and_then(|()| secure_file(&temporary))
        .and_then(|()| file.sync_all().map_err(BrainError::from));
    drop(file);
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        if let Some(parent) = temporary.parent() {
            let _ = sync_private_directory(root, parent);
        }
        return Err(error);
    }
    match sha256_file(&temporary) {
        Ok(digest) => Ok((temporary, digest)),
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            if let Some(parent) = temporary.parent() {
                let _ = sync_private_directory(root, parent);
            }
            Err(error)
        }
    }
}

/// Content identity of an immutable file under a verified CEREBRO private root.
///
/// This is deliberately generic only over file identity. It does not erase the
/// semantic type of the payload: Delta/F64 artifacts, receipts and JSON records
/// keep their own domain structures on top of this reference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrivateFileReference {
    pub path: PathBuf,
    pub sha256: Sha256Digest,
}

/// Path-independent identity of a complete private directory tree.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrivateDirectoryIdentity {
    pub tree_sha256: Sha256Digest,
    pub entry_count: u64,
    pub regular_file_count: u64,
    pub total_file_bytes: u64,
}

#[derive(Debug)]
struct PrivateTreeEntry {
    relative_path: PathBuf,
    kind: u8,
    file_size: u64,
    file_sha256: Option<Sha256Digest>,
}

impl PrivateFileReference {
    pub fn new(path: impl Into<PathBuf>, sha256: Sha256Digest) -> Self {
        Self {
            path: path.into(),
            sha256,
        }
    }

    /// Verify path confinement and exact current file content through one
    /// descriptor. The returned path is only the confined lexical name; it is
    /// not a capability and must not be reopened by a new security boundary.
    pub fn verify(&self, root: &Path) -> BrainResult<PathBuf> {
        let opened = open_private_reference(root, &self.path)?;
        verify_opened_private_file(opened.file, &opened.stat, &self.sha256, None, false)
            .map(|_| opened.path)
    }

    /// Legacy compatibility reader for payloads whose size is already bounded
    /// by an authenticated enclosing contract. New authority boundaries must
    /// use [`Self::read_verified_bounded`] with their protocol-specific limit.
    pub fn read_verified(&self, root: &Path) -> BrainResult<Vec<u8>> {
        Ok(self.read_verified_with_path(root)?.1)
    }

    /// Read and authenticate at most `max_bytes`, consuming at most one byte
    /// beyond the limit as an oversize sentinel. Resolution, metadata checks,
    /// hashing and reading all use the same descriptor.
    pub fn read_verified_bounded(&self, root: &Path, max_bytes: u64) -> BrainResult<Vec<u8>> {
        self.read_verified_bounded_after_open(root, max_bytes, || {})
    }

    /// Legacy unbounded counterpart returning the confined lexical path and
    /// exact bytes from one descriptor. The path must not be reopened as an
    /// authority capability; new boundaries should retain the bytes returned
    /// by [`Self::read_verified_bounded`] instead.
    pub fn read_verified_with_path(&self, root: &Path) -> BrainResult<(PathBuf, Vec<u8>)> {
        let opened = open_private_reference(root, &self.path)?;
        let path = opened.path;
        let bytes =
            verify_opened_private_file(opened.file, &opened.stat, &self.sha256, None, true)?;
        Ok((path, bytes))
    }

    fn read_verified_bounded_after_open<F>(
        &self,
        root: &Path,
        max_bytes: u64,
        after_open: F,
    ) -> BrainResult<Vec<u8>>
    where
        F: FnOnce(),
    {
        let opened = open_private_reference(root, &self.path)?;
        after_open();
        verify_opened_private_file(
            opened.file,
            &opened.stat,
            &self.sha256,
            Some(max_bytes),
            true,
        )
    }
}

/// Read an untrusted command input from the configured private authority.
///
/// Unlike [`PrivateFileReference`], this boundary deliberately has no
/// pre-declared content digest: command input is untrusted until its caller
/// parses and validates it.  It still receives the same descriptor-bound
/// confinement, regular-file, ownership, stability, and byte-limit checks as
/// authenticated private files.  This is the only appropriate reader for a
/// bounded, ephemeral CLI input that has not yet been admitted as an
/// immutable artifact.
pub fn read_untrusted_private_file_bounded(
    root: &Path,
    raw: &Path,
    max_bytes: u64,
) -> BrainResult<Vec<u8>> {
    let opened = open_private_reference(root, raw)?;
    read_opened_private_file_bounded(opened.file, &opened.stat, max_bytes)
}

struct OpenedPrivateFile {
    path: PathBuf,
    file: File,
    stat: Stat,
}

fn open_private_reference(root: &Path, raw: &Path) -> BrainResult<OpenedPrivateFile> {
    let relative = root_relative_path(root, raw)?;
    if root == Path::new("/")
        || root
            .components()
            .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        return Err(BrainError::Invalid("private_root_path_invalid".into()));
    }
    let filesystem_root = open(
        Path::new("/"),
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(rustix_error)?;
    let relative_root = root
        .strip_prefix(Path::new("/"))
        .map_err(|_| BrainError::Invalid("private_root_path_invalid".into()))?;
    let root_fd = openat2(
        &filesystem_root,
        relative_root,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
        PRIVATE_RESOLUTION,
    )
    .map_err(rustix_error)?;
    let root_stat = fstat(&root_fd).map_err(rustix_error)?;
    validate_private_root_stat(&root_stat)?;

    // NONBLOCK ensures that a type race to a FIFO or device cannot hang this
    // authority before fstat rejects it.
    let fd = openat2(
        &root_fd,
        &relative,
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
        PRIVATE_RESOLUTION,
    )
    .map_err(rustix_error)?;
    let stat = fstat(&fd).map_err(rustix_error)?;
    validate_private_file_stat(&root_stat, &stat)?;
    Ok(OpenedPrivateFile {
        path: root.join(relative),
        file: File::from(fd),
        stat,
    })
}

fn validate_private_root_stat(stat: &Stat) -> BrainResult<()> {
    if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
        return Err(BrainError::Integrity(
            "private_root_descriptor_invalid".into(),
        ));
    }
    Ok(())
}

fn validate_private_file_stat(root: &Stat, file: &Stat) -> BrainResult<()> {
    if FileType::from_raw_mode(file.st_mode) != FileType::RegularFile {
        return Err(BrainError::Integrity("private_file_not_regular".into()));
    }
    // Files must share the root owner, cannot carry special bits or be
    // world-writable. Group-write is accepted only when the root itself denies
    // every group permission, so it remains unreachable through this tree.
    let group_write_exposed = file.st_mode & 0o020 != 0 && root.st_mode & 0o070 != 0;
    if file.st_uid != root.st_uid || file.st_mode & 0o7002 != 0 || group_write_exposed {
        return Err(BrainError::Integrity(
            "private_file_permissions_or_owner_invalid".into(),
        ));
    }
    Ok(())
}

fn verify_opened_private_file(
    mut file: File,
    opened: &Stat,
    expected: &Sha256Digest,
    max_bytes: Option<u64>,
    retain_bytes: bool,
) -> BrainResult<Vec<u8>> {
    let opened_len = u64::try_from(opened.st_size)
        .map_err(|_| BrainError::Integrity("private_file_size_invalid".into()))?;
    if max_bytes.is_some_and(|limit| opened_len > limit) {
        return Err(BrainError::Invalid(
            "private_file_read_limit_exceeded".into(),
        ));
    }
    let capacity = if retain_bytes {
        usize::try_from(opened_len)
            .map_err(|_| BrainError::Invalid("private_file_read_limit_exceeded".into()))?
    } else {
        0
    };
    let mut bytes = Vec::with_capacity(capacity);
    let mut hasher = Sha256::new();
    let mut actual_len = 0_u64;
    let mut buffer = [0_u8; PRIVATE_READ_BUFFER_BYTES];
    loop {
        let read_limit = max_bytes
            .map(|limit| {
                limit
                    .saturating_sub(actual_len)
                    .saturating_add(1)
                    .min(PRIVATE_READ_BUFFER_BYTES as u64) as usize
            })
            .unwrap_or(PRIVATE_READ_BUFFER_BYTES);
        let count = file.read(&mut buffer[..read_limit])?;
        if count == 0 {
            break;
        }
        actual_len = actual_len
            .checked_add(count as u64)
            .ok_or_else(|| BrainError::Integrity("private_file_size_overflow".into()))?;
        if max_bytes.is_some_and(|limit| actual_len > limit) {
            return Err(BrainError::Invalid(
                "private_file_read_limit_exceeded".into(),
            ));
        }
        hasher.update(&buffer[..count]);
        if retain_bytes {
            bytes.extend_from_slice(&buffer[..count]);
        }
    }
    let final_stat = fstat(&file).map_err(rustix_error)?;
    ensure_private_file_stable(opened, &final_stat)?;
    if actual_len != opened_len {
        return Err(BrainError::Integrity(
            "private_file_mutated_during_read".into(),
        ));
    }
    let actual = Sha256Digest::parse(format!("{:x}", hasher.finalize()))?;
    if actual != *expected {
        return Err(BrainError::Integrity(
            "private_file_reference_digest_mismatch".into(),
        ));
    }
    Ok(bytes)
}

fn read_opened_private_file_bounded(
    mut file: File,
    opened: &Stat,
    max_bytes: u64,
) -> BrainResult<Vec<u8>> {
    let opened_len = u64::try_from(opened.st_size)
        .map_err(|_| BrainError::Integrity("private_file_size_invalid".into()))?;
    if opened_len > max_bytes {
        return Err(BrainError::Invalid(
            "private_file_read_limit_exceeded".into(),
        ));
    }
    let capacity = usize::try_from(opened_len)
        .map_err(|_| BrainError::Invalid("private_file_read_limit_exceeded".into()))?;
    let mut bytes = Vec::with_capacity(capacity);
    let mut actual_len = 0_u64;
    let mut buffer = [0_u8; PRIVATE_READ_BUFFER_BYTES];
    loop {
        let remaining_plus_sentinel = max_bytes
            .saturating_sub(actual_len)
            .saturating_add(1)
            .min(PRIVATE_READ_BUFFER_BYTES as u64) as usize;
        let count = file.read(&mut buffer[..remaining_plus_sentinel])?;
        if count == 0 {
            break;
        }
        actual_len = actual_len
            .checked_add(count as u64)
            .ok_or_else(|| BrainError::Integrity("private_file_size_overflow".into()))?;
        if actual_len > max_bytes {
            return Err(BrainError::Invalid(
                "private_file_read_limit_exceeded".into(),
            ));
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    let final_stat = fstat(&file).map_err(rustix_error)?;
    ensure_private_file_stable(opened, &final_stat)?;
    if actual_len != opened_len {
        return Err(BrainError::Integrity(
            "private_file_mutated_during_read".into(),
        ));
    }
    Ok(bytes)
}

fn ensure_private_file_stable(before: &Stat, after: &Stat) -> BrainResult<()> {
    if FileType::from_raw_mode(after.st_mode) != FileType::RegularFile
        || before.st_dev != after.st_dev
        || before.st_ino != after.st_ino
        || before.st_mode != after.st_mode
        || before.st_size != after.st_size
        || before.st_mtime != after.st_mtime
        || before.st_mtime_nsec != after.st_mtime_nsec
        || before.st_ctime != after.st_ctime
        || before.st_ctime_nsec != after.st_ctime_nsec
    {
        return Err(BrainError::Integrity(
            "private_file_mutated_during_read".into(),
        ));
    }
    Ok(())
}

fn rustix_error(error: rustix::io::Errno) -> BrainError {
    BrainError::Io(std::io::Error::from_raw_os_error(error.raw_os_error()))
}

/// Convert an absolute private path into a component-safe relative path.
/// Parent components, root prefixes and other non-normal components are never
/// accepted, so later joins cannot escape the already verified root.
pub fn root_relative_path(root: &Path, raw: &Path) -> BrainResult<PathBuf> {
    if !raw.is_absolute() {
        return Err(BrainError::Invalid(
            "private_file_path_must_be_absolute".into(),
        ));
    }
    let relative = raw
        .strip_prefix(root)
        .map_err(|_| BrainError::Integrity("private_file_path_outside_root".into()))?;
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(BrainError::Invalid("private_file_path_not_confined".into()));
    }
    Ok(relative.to_path_buf())
}

/// Reject symlinks at every existing path component, not only at the leaf.
/// This is the stronger path algorithm already used by representation evidence.
fn assert_existing_components_not_symlinks(root: &Path, relative: &Path) -> BrainResult<()> {
    let mut cursor = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(name) = component else {
            return Err(BrainError::Invalid("private_file_path_not_confined".into()));
        };
        cursor.push(name);
        let metadata = fs::symlink_metadata(&cursor)?;
        if metadata.file_type().is_symlink() {
            return Err(BrainError::Integrity(
                "private_file_symlink_forbidden".into(),
            ));
        }
    }
    Ok(())
}

/// Check every existing component of a prospective private path.  Unlike the
/// strict resolver above, a missing suffix is allowed so callers can safely
/// distinguish a vacant target from a hostile symlink or non-directory that
/// already exists on its route.
fn assert_existing_prefix_not_symlinks(root: &Path, relative: &Path) -> BrainResult<()> {
    let mut cursor = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(name) = component else {
            return Err(BrainError::Invalid("private_file_path_not_confined".into()));
        };
        cursor.push(name);
        match fs::symlink_metadata(&cursor) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(BrainError::Integrity(
                    "private_file_symlink_forbidden".into(),
                ));
            }
            Ok(metadata) if !metadata.file_type().is_file() && !metadata.file_type().is_dir() => {
                return Err(BrainError::Integrity(
                    "private_file_path_component_invalid".into(),
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

/// Resolve an existing regular file without permitting symlink traversal or
/// lexical escape from the supplied, already trusted private root.
pub fn verify_existing_file(
    root: &Path,
    raw: &Path,
    expected_sha256: Option<&Sha256Digest>,
) -> BrainResult<PathBuf> {
    let path = existing_regular_file_under_root(root, raw)?;
    if let Some(expected) = expected_sha256 {
        if sha256_file(&path)? != *expected {
            return Err(BrainError::Integrity(
                "private_file_reference_digest_mismatch".into(),
            ));
        }
    }
    Ok(path)
}

pub fn existing_regular_file_under_root(root: &Path, raw: &Path) -> BrainResult<PathBuf> {
    let relative = root_relative_path(root, raw)?;
    assert_existing_components_not_symlinks(root, &relative)?;
    let path = root.join(&relative);
    let metadata = fs::symlink_metadata(&path)?;
    if !metadata.file_type().is_file() {
        return Err(BrainError::Integrity("private_file_not_regular".into()));
    }
    let canonical = path.canonicalize()?;
    let canonical_root = root.canonicalize()?;
    if !canonical.starts_with(&canonical_root) {
        return Err(BrainError::Integrity(
            "private_file_path_outside_root".into(),
        ));
    }
    Ok(path)
}

/// Resolve a regular file when present without creating any filesystem state.
/// A missing leaf is a normal `None`; every present component is still checked
/// for symlink traversal before that result is returned.
pub fn existing_regular_file_if_present(root: &Path, raw: &Path) -> BrainResult<Option<PathBuf>> {
    let relative = root_relative_path(root, raw)?;
    assert_existing_prefix_not_symlinks(root, &relative)?;
    let path = root.join(&relative);
    match fs::symlink_metadata(&path) {
        Ok(_) => existing_regular_file_under_root(root, &path).map(Some),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Resolve an existing private directory without permitting a symlink at any
/// component.  Authority code must use this rather than `is_dir()` followed by
/// a later read/rename: the latter can silently traverse a replaced parent.
pub fn existing_directory_under_root(root: &Path, raw: &Path) -> BrainResult<PathBuf> {
    let relative = root_relative_path(root, raw)?;
    assert_existing_components_not_symlinks(root, &relative)?;
    let path = root.join(&relative);
    let metadata = fs::symlink_metadata(&path)?;
    if !metadata.file_type().is_dir() {
        return Err(BrainError::Integrity(
            "private_directory_not_directory".into(),
        ));
    }
    let canonical = path.canonicalize()?;
    let canonical_root = root.canonicalize()?;
    if !canonical.starts_with(&canonical_root) {
        return Err(BrainError::Integrity(
            "private_directory_path_outside_root".into(),
        ));
    }
    Ok(path)
}

/// Resolve a private directory when present without creating it.  This is the
/// directory counterpart to `existing_regular_file_if_present` and is used for
/// topology preflights before a transaction mutates any sibling.
pub fn existing_directory_if_present(root: &Path, raw: &Path) -> BrainResult<Option<PathBuf>> {
    let relative = root_relative_path(root, raw)?;
    assert_existing_prefix_not_symlinks(root, &relative)?;
    let path = root.join(&relative);
    match fs::symlink_metadata(&path) {
        Ok(_) => existing_directory_under_root(root, &path).map(Some),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn collect_private_tree_entries(
    root: &Path,
    base: &Path,
    directory: &Path,
    entries: &mut Vec<PrivateTreeEntry>,
) -> BrainResult<()> {
    let directory = existing_directory_under_root(root, directory)?;
    let mut children = fs::read_dir(&directory)?
        .map(|entry| entry.map(|value| value.path()))
        .collect::<Result<Vec<_>, _>>()?;
    children.sort_by(|left, right| {
        left.as_os_str()
            .as_bytes()
            .cmp(right.as_os_str().as_bytes())
    });
    for child in children {
        let metadata = fs::symlink_metadata(&child)?;
        if metadata.file_type().is_symlink() {
            return Err(BrainError::Integrity(
                "private_directory_tree_symlink_forbidden".into(),
            ));
        }
        let relative_path = child
            .strip_prefix(base)
            .map_err(|_| BrainError::Integrity("private_directory_tree_path_escape".into()))?
            .to_path_buf();
        if metadata.file_type().is_dir() {
            existing_directory_under_root(root, &child)?;
            entries.push(PrivateTreeEntry {
                relative_path,
                kind: b'd',
                file_size: 0,
                file_sha256: None,
            });
            collect_private_tree_entries(root, base, &child, entries)?;
        } else if metadata.file_type().is_file() {
            let child = existing_regular_file_under_root(root, &child)?;
            entries.push(PrivateTreeEntry {
                relative_path,
                kind: b'f',
                file_size: metadata.len(),
                file_sha256: Some(sha256_file(&child)?),
            });
        } else {
            return Err(BrainError::Integrity(
                "private_directory_tree_special_file_forbidden".into(),
            ));
        }
    }
    Ok(())
}

/// Authenticate the complete contents and topology of a private directory.
/// The digest excludes the directory's own name, so it remains stable across
/// an authorized move from an inflight location to an archive location.
pub fn inspect_private_directory(
    root: &Path,
    directory: &Path,
) -> BrainResult<PrivateDirectoryIdentity> {
    let directory = existing_directory_under_root(root, directory)?;
    let mut entries = Vec::new();
    collect_private_tree_entries(root, &directory, &directory, &mut entries)?;
    entries.sort_by(|left, right| {
        left.relative_path
            .as_os_str()
            .as_bytes()
            .cmp(right.relative_path.as_os_str().as_bytes())
    });
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:PRIVATE-DIRECTORY-TREE:v1\0");
    let mut regular_file_count = 0u64;
    let mut total_file_bytes = 0u64;
    for entry in &entries {
        let path = entry.relative_path.as_os_str().as_bytes();
        hasher.update([entry.kind]);
        hasher.update((path.len() as u64).to_be_bytes());
        hasher.update(path);
        hasher.update(entry.file_size.to_be_bytes());
        if let Some(digest) = &entry.file_sha256 {
            hasher.update(digest.as_bytes());
            regular_file_count = regular_file_count.checked_add(1).ok_or_else(|| {
                BrainError::Invalid("private_directory_tree_count_overflow".into())
            })?;
            total_file_bytes = total_file_bytes
                .checked_add(entry.file_size)
                .ok_or_else(|| {
                    BrainError::Invalid("private_directory_tree_size_overflow".into())
                })?;
        }
    }
    Ok(PrivateDirectoryIdentity {
        tree_sha256: Sha256Digest::parse(format!("{:x}", hasher.finalize()))?,
        entry_count: entries.len() as u64,
        regular_file_count,
        total_file_bytes,
    })
}

/// Atomically move a complete private directory to a vacant destination.
///
/// Linux `renameat2(RENAME_NOREPLACE)` closes the preflight/rename overwrite
/// race. The complete tree identity is checked before and after the move, and
/// the moved directory plus both parents are synced before success. A retry
/// after a completed rename is accepted only when the source is absent and the
/// destination still has the exact expected identity.
pub fn move_private_directory_transactional(
    root: &Path,
    source: &Path,
    destination: &Path,
    expected: &PrivateDirectoryIdentity,
) -> BrainResult<()> {
    if source == destination {
        return Err(BrainError::Invalid(
            "private_directory_move_source_equals_destination".into(),
        ));
    }
    let Some(source) = existing_directory_if_present(root, source)? else {
        let destination_identity = inspect_private_directory(root, destination)?;
        if destination_identity != *expected {
            return Err(BrainError::Integrity(
                "private_directory_move_completed_identity_mismatch".into(),
            ));
        }
        return Ok(());
    };
    if inspect_private_directory(root, &source)? != *expected {
        return Err(BrainError::Integrity(
            "private_directory_move_source_identity_mismatch".into(),
        ));
    }
    if existing_directory_if_present(root, destination)?.is_some() {
        return Err(BrainError::Integrity(
            "private_directory_move_destination_already_exists".into(),
        ));
    }
    ensure_private_parent(root, destination)?;
    let source_parent = source.parent().ok_or_else(|| {
        BrainError::Invalid("private_directory_move_source_parent_missing".into())
    })?;
    let destination_parent = destination.parent().ok_or_else(|| {
        BrainError::Invalid("private_directory_move_destination_parent_missing".into())
    })?;
    let source_parent = existing_directory_under_root(root, source_parent)?;
    let destination_parent = existing_directory_under_root(root, destination_parent)?;
    let source_name = source
        .file_name()
        .ok_or_else(|| BrainError::Invalid("private_directory_move_source_name_missing".into()))?;
    let destination_name = destination.file_name().ok_or_else(|| {
        BrainError::Invalid("private_directory_move_destination_name_missing".into())
    })?;
    let source_parent_file = File::open(&source_parent)?;
    let destination_parent_file = File::open(&destination_parent)?;
    renameat_with(
        &source_parent_file,
        source_name,
        &destination_parent_file,
        destination_name,
        RenameFlags::NOREPLACE,
    )
    .map_err(|error| std::io::Error::from_raw_os_error(error.raw_os_error()))?;
    secure_dir(destination)?;
    File::open(destination)?.sync_all()?;
    destination_parent_file.sync_all()?;
    if source_parent != destination_parent {
        source_parent_file.sync_all()?;
    }
    let installed = inspect_private_directory(root, destination)?;
    if installed != *expected {
        return Err(BrainError::Integrity(
            "private_directory_move_postinstall_identity_mismatch".into(),
        ));
    }
    Ok(())
}

/// Securely create only missing directory components below the private root.
/// Existing symlinks or non-directory components fail closed.
pub fn ensure_private_parent(root: &Path, destination: &Path) -> BrainResult<()> {
    let relative = root_relative_path(root, destination)?;
    let parent = relative
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?;
    let mut cursor = root.to_path_buf();
    for component in parent.components() {
        let Component::Normal(name) = component else {
            return Err(BrainError::Invalid("private_file_path_not_confined".into()));
        };
        cursor.push(name);
        match fs::symlink_metadata(&cursor) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() {
                    return Err(BrainError::Integrity(
                        "private_file_symlink_forbidden".into(),
                    ));
                }
                if !metadata.file_type().is_dir() {
                    return Err(BrainError::Integrity(
                        "private_file_parent_not_directory".into(),
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match fs::create_dir(&cursor) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        let metadata = fs::symlink_metadata(&cursor)?;
                        if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
                            return Err(BrainError::Integrity(
                                "private_file_parent_not_directory".into(),
                            ));
                        }
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            Err(error) => return Err(error.into()),
        }
        secure_dir(&cursor)?;
    }
    Ok(())
}

/// Create a private directory tree only through non-symlink components, then
/// return the exact directory.  This is for transaction roots and immutable
/// stores; callers must not use unrestricted `create_dir_all` below authority
/// state because a pre-existing parent can redirect writes.
pub fn ensure_private_directory(root: &Path, directory: &Path) -> BrainResult<PathBuf> {
    let relative = root_relative_path(root, directory)?;
    let mut cursor = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(name) = component else {
            return Err(BrainError::Invalid(
                "private_directory_path_not_confined".into(),
            ));
        };
        cursor.push(name);
        match fs::symlink_metadata(&cursor) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
                    return Err(BrainError::Integrity(
                        "private_directory_target_invalid".into(),
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match fs::create_dir(&cursor) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        let metadata = fs::symlink_metadata(&cursor)?;
                        if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
                            return Err(BrainError::Integrity(
                                "private_directory_target_invalid".into(),
                            ));
                        }
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            Err(error) => return Err(error.into()),
        }
        secure_dir(&cursor)?;
    }
    existing_directory_under_root(root, directory)
}

/// Install immutable bytes or verify the exact existing content. No overwrite
/// path exists: a content mismatch is an integrity failure.
pub fn write_or_verify_immutable(
    root: &Path,
    path: &Path,
    bytes: &[u8],
) -> BrainResult<Sha256Digest> {
    let expected = Sha256Digest::digest_bytes(bytes);
    if existing_regular_file_if_present(root, path)?.is_some() {
        verify_immutable_content(root, path, &expected)?;
        return Ok(expected);
    }
    let temporary = stage_private_bytes(root, path, bytes, &expected)?;
    if !install_private_immutable_file(root, &temporary, path, &expected)? {
        verify_immutable_content(root, path, &expected)?;
    }
    Ok(expected)
}

/// Create a new immutable file.  Unlike `write_or_verify_immutable`, any
/// existing leaf is an integrity error even when its bytes happen to match.
/// This prevents callers that reserve a fresh receipt or evidence identity
/// from silently accepting a pre-existing authority object.
pub fn create_private_immutable(
    root: &Path,
    path: &Path,
    bytes: &[u8],
) -> BrainResult<Sha256Digest> {
    if existing_regular_file_if_present(root, path)?.is_some() {
        return Err(BrainError::Integrity(
            "private_file_immutable_target_already_exists".into(),
        ));
    }
    let expected = Sha256Digest::digest_bytes(bytes);
    let temporary = stage_private_bytes(root, path, bytes, &expected)?;
    if install_private_immutable_file(root, &temporary, path, &expected)? {
        Ok(expected)
    } else {
        Err(BrainError::Integrity(
            "private_file_immutable_target_already_exists".into(),
        ))
    }
}

/// Durably install a fully-written private staging file as an immutable object.
///
/// The staging file is synced and checked against `expected` before it becomes
/// visible. Installation uses a hard link, which is atomic and cannot replace
/// an existing leaf. The destination file and affected parent directory are
/// synced before success is returned, then the installed bytes are verified
/// again. `Ok(false)` means that another object already occupied the target;
/// the caller must decide whether that existing object is an idempotent match
/// or a collision. The staging file is removed on every terminal outcome.
pub fn install_private_immutable_file(
    root: &Path,
    temporary: &Path,
    destination: &Path,
    expected: &Sha256Digest,
) -> BrainResult<bool> {
    if temporary == destination {
        return Err(BrainError::Invalid(
            "private_file_staging_equals_destination".into(),
        ));
    }
    ensure_private_parent(root, destination)?;
    let temporary = existing_regular_file_under_root(root, temporary)?;
    let temporary_parent = temporary
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?
        .to_path_buf();
    let destination_parent = destination
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?
        .to_path_buf();

    let prepared = (|| -> BrainResult<()> {
        secure_file(&temporary)?;
        OpenOptions::new()
            .read(true)
            .write(true)
            .open(&temporary)?
            .sync_all()?;
        if sha256_file(&temporary)? != *expected {
            return Err(BrainError::Integrity(
                "private_file_staged_content_mismatch".into(),
            ));
        }
        Ok(())
    })();
    if let Err(error) = prepared {
        let _ = fs::remove_file(&temporary);
        let _ = sync_private_directory(root, &temporary_parent);
        return Err(error);
    }

    let installed = match fs::hard_link(&temporary, destination) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            // A hostile symlink or non-file collision is never a benign race.
            if let Err(validation_error) = existing_regular_file_under_root(root, destination) {
                let _ = fs::remove_file(&temporary);
                let _ = sync_private_directory(root, &temporary_parent);
                return Err(validation_error);
            }
            false
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            let _ = sync_private_directory(root, &temporary_parent);
            return Err(error.into());
        }
    };

    let finalize = (|| -> BrainResult<()> {
        if installed {
            secure_file(destination)?;
            let mut options = OpenOptions::new();
            options.read(true).write(true);
            #[cfg(any(target_os = "linux", target_os = "android"))]
            options.custom_flags(0o400000); // O_NOFOLLOW
            options.open(destination)?.sync_all()?;
        }
        Ok(())
    })();
    let cleanup = (|| -> BrainResult<()> {
        fs::remove_file(&temporary)?;
        sync_private_directory(root, &temporary_parent)?;
        if destination_parent != temporary_parent {
            sync_private_directory(root, &destination_parent)?;
        }
        Ok(())
    })();
    finalize?;
    cleanup?;
    if installed {
        verify_immutable_content(root, destination, expected)?;
    }
    Ok(installed)
}

/// Move a private regular file to a vacant private destination without an
/// overwrite window, with crash-recoverable link-then-unlink semantics.
///
/// If a crash leaves both names visible, a retry completes the transaction
/// only when both names identify the same inode. A distinct pre-existing
/// destination is always an integrity failure, even when its bytes match.
pub fn move_private_file_transactional(
    root: &Path,
    source: &Path,
    destination: &Path,
    expected: &Sha256Digest,
) -> BrainResult<()> {
    if source == destination {
        return Err(BrainError::Invalid(
            "private_file_move_source_equals_destination".into(),
        ));
    }
    let Some(source) = existing_regular_file_if_present(root, source)? else {
        // Completed crash/retry state: the source name was already durably
        // removed. Exact destination identity is the only accepted witness.
        verify_immutable_content(root, destination, expected)?;
        return Ok(());
    };
    if sha256_file(&source)? != *expected {
        return Err(BrainError::Integrity(
            "private_file_move_source_digest_mismatch".into(),
        ));
    }
    ensure_private_parent(root, destination)?;
    let source_parent = source
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?
        .to_path_buf();
    let destination_parent = destination
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?
        .to_path_buf();

    let source_metadata = fs::metadata(&source)?;
    match fs::hard_link(&source, destination) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let destination = existing_regular_file_under_root(root, destination)?;
            let destination_metadata = fs::metadata(destination)?;
            if source_metadata.dev() != destination_metadata.dev()
                || source_metadata.ino() != destination_metadata.ino()
            {
                return Err(BrainError::Integrity(
                    "private_file_move_destination_already_exists".into(),
                ));
            }
        }
        Err(error) => return Err(error.into()),
    }
    secure_file(destination)?;
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    options.custom_flags(0o400000); // O_NOFOLLOW
    options.open(destination)?.sync_all()?;
    sync_private_directory(root, &destination_parent)?;
    fs::remove_file(&source)?;
    sync_private_directory(root, &source_parent)?;
    verify_immutable_content(root, destination, expected)
}

fn stage_private_bytes(
    root: &Path,
    path: &Path,
    bytes: &[u8],
    expected: &Sha256Digest,
) -> BrainResult<PathBuf> {
    let (temporary, actual) = stage_private_file(root, path, |file| {
        file.write_all(bytes)?;
        Ok(())
    })?;
    if actual != *expected {
        let _ = fs::remove_file(&temporary);
        if let Some(parent) = temporary.parent() {
            let _ = sync_private_directory(root, parent);
        }
        return Err(BrainError::Integrity(
            "private_file_staged_content_mismatch".into(),
        ));
    }
    Ok(temporary)
}

fn verify_immutable_content(root: &Path, path: &Path, expected: &Sha256Digest) -> BrainResult<()> {
    let verified = existing_regular_file_under_root(root, path)?;
    if sha256_file(&verified)? != *expected {
        return Err(BrainError::Integrity(
            "private_file_immutable_content_mismatch".into(),
        ));
    }
    secure_file(&verified)
}

fn sync_private_directory(root: &Path, directory: &Path) -> BrainResult<()> {
    let directory = existing_directory_under_root(root, directory)?;
    fs::File::open(directory)?.sync_all()?;
    Ok(())
}

/// Atomically replace a mutable private file after validating its entire path.
/// Immutable receipts and artifacts must use an immutable writer instead.  The
/// caller remains responsible for deciding which pointer paths are mutable.
pub fn replace_private_file_atomic(
    root: &Path,
    path: &Path,
    bytes: &[u8],
    expected_sha256: Option<&Sha256Digest>,
) -> BrainResult<Sha256Digest> {
    let digest = Sha256Digest::digest_bytes(bytes);
    if expected_sha256.is_some_and(|expected| expected != &digest) {
        return Err(BrainError::Integrity(
            "private_file_atomic_expected_digest_mismatch".into(),
        ));
    }
    let _ = existing_regular_file_if_present(root, path)?;
    ensure_private_parent(root, path)?;
    let parent = path
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?;
    let parent = existing_directory_under_root(root, parent)?;
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| BrainError::Invalid("private_file_name_invalid".into()))?;
    let temporary = parent.join(format!(".{file_name}.{}.tmp", std::process::id()));
    root_relative_path(root, &temporary)?;
    match fs::symlink_metadata(&temporary) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Ok(_) => {
            return Err(BrainError::Integrity(
                "private_file_atomic_temporary_exists".into(),
            ));
        }
        Err(error) => return Err(error.into()),
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path)?;
    secure_file(path)?;
    let installed = existing_regular_file_under_root(root, path)?;
    if sha256_file(&installed)? != digest {
        return Err(BrainError::Integrity(
            "private_file_atomic_postwrite_mismatch".into(),
        ));
    }
    fs::File::open(parent)?.sync_all()?;
    Ok(digest)
}

/// Execute a critical section guarded by a descriptor-bound, process-safe lock
/// under the private authority root.  The lock object is immutable and carries
/// no semantic state; the advisory lock is released automatically if a process
/// terminates, so a crash cannot leave a stale ownership marker behind.
///
/// This primitive is intentionally separate from mutable payload writes.  A
/// caller must still read and compare the current authoritative value while
/// holding the lock, then use [`replace_private_file_atomic`] to publish it.
pub fn with_private_authority_lock<T, F>(
    root: &Path,
    lock_path: &Path,
    operation: F,
) -> BrainResult<T>
where
    F: FnOnce() -> BrainResult<T>,
{
    write_or_verify_immutable(root, lock_path, b"")?;
    let opened = open_private_reference(root, lock_path)?;
    flock(&opened.file, FlockOperation::LockExclusive).map_err(rustix_error)?;
    let result = operation();
    // Explicit release makes failures observable on platforms where close is
    // delayed.  The descriptor's drop remains the crash-safe fallback.
    let unlock = flock(&opened.file, FlockOperation::Unlock).map_err(rustix_error);
    match (result, unlock) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cerebro-authority-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        secure_dir(&root).unwrap();
        root
    }

    #[test]
    fn private_reference_verifies_exact_bytes() {
        let root = root("reference");
        let path = root.join("state/evidence.bin");
        let digest = write_or_verify_immutable(&root, &path, b"evidence").unwrap();
        let reference = PrivateFileReference::new(path.clone(), digest);
        assert_eq!(reference.read_verified(&root).unwrap(), b"evidence");
        fs::write(&path, b"tampered").unwrap();
        assert!(reference.read_verified(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bounded_reference_read_rejects_oversize_without_allocating_the_payload() {
        let root = root("bounded-reference");
        let path = root.join("state/evidence.bin");
        let payload = vec![b'x'; PRIVATE_READ_BUFFER_BYTES + 1];
        let digest = write_or_verify_immutable(&root, &path, &payload).unwrap();
        let reference = PrivateFileReference::new(path, digest);

        assert!(reference
            .read_verified_bounded(&root, PRIVATE_READ_BUFFER_BYTES as u64)
            .is_err());
        assert_eq!(
            reference
                .read_verified_bounded(&root, payload.len() as u64)
                .unwrap(),
            payload
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn descriptor_resolution_rejects_leaf_and_parent_symlinks() {
        let root = root("descriptor-symlinks");
        let target = root.join("state/target.bin");
        let target_digest = write_or_verify_immutable(&root, &target, b"target").unwrap();

        let leaf = root.join("state/leaf.bin");
        symlink(&target, &leaf).unwrap();
        let leaf_reference = PrivateFileReference::new(leaf, target_digest.clone());
        assert!(leaf_reference.read_verified_bounded(&root, 64).is_err());

        let real_parent = root.join("real-parent");
        ensure_private_directory(&root, &real_parent).unwrap();
        let nested = real_parent.join("nested.bin");
        write_or_verify_immutable(&root, &nested, b"nested").unwrap();
        let linked_parent = root.join("linked-parent");
        symlink(&real_parent, &linked_parent).unwrap();
        let parent_reference = PrivateFileReference::new(
            linked_parent.join("nested.bin"),
            Sha256Digest::digest_bytes(b"nested"),
        );
        assert!(parent_reference.read_verified_bounded(&root, 64).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn replacement_after_open_fails_closed_without_accepting_replacement_bytes() {
        let root = root("descriptor-replacement");
        let path = root.join("state/object.bin");
        let replacement = root.join("state/replacement.bin");
        let digest = write_or_verify_immutable(&root, &path, b"original").unwrap();
        write_or_verify_immutable(&root, &replacement, b"hostile!").unwrap();
        let reference = PrivateFileReference::new(path.clone(), digest);

        let result = reference.read_verified_bounded_after_open(&root, 64, || {
            fs::rename(&replacement, &path).unwrap();
        });
        assert!(result.is_err());
        assert_eq!(fs::read(path).unwrap(), b"hostile!");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mutation_race_on_the_open_inode_fails_closed() {
        let root = root("descriptor-mutation");
        let path = root.join("state/object.bin");
        let digest = write_or_verify_immutable(&root, &path, b"original").unwrap();
        let reference = PrivateFileReference::new(path.clone(), digest);

        assert!(reference
            .read_verified_bounded_after_open(&root, 64, || {
                fs::write(&path, b"mutated!").unwrap();
            })
            .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn writable_by_other_users_is_not_private_authority_data() {
        let root = root("descriptor-permissions");
        let path = root.join("state/object.bin");
        let digest = write_or_verify_immutable(&root, &path, b"payload").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o622)).unwrap();
        let reference = PrivateFileReference::new(path, digest);
        assert!(reference.read_verified_bounded(&root, 64).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fresh_immutable_write_never_accepts_a_preexisting_leaf() {
        let root = root("fresh-immutable");
        let path = root.join("state/receipt.json");
        create_private_immutable(&root, &path, b"first").unwrap();
        assert!(create_private_immutable(&root, &path, b"first").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn streamed_staging_install_is_synced_private_verified_and_clean() {
        let root = root("streamed-install");
        let path = root.join("state/objects/payload.bin");
        let (temporary, mut file) = create_private_staging_file(&root, &path).unwrap();
        file.write_all(b"streamed immutable payload").unwrap();
        drop(file);
        let expected = Sha256Digest::digest_bytes(b"streamed immutable payload");

        assert!(install_private_immutable_file(&root, &temporary, &path, &expected).unwrap());
        assert!(!temporary.exists());
        assert_eq!(fs::read(&path).unwrap(), b"streamed immutable payload");
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        verify_immutable_content(&root, &path, &expected).unwrap();
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn staged_digest_mismatch_fails_closed_and_removes_staging() {
        let root = root("staged-mismatch");
        let path = root.join("state/object.bin");
        let (temporary, mut file) = create_private_staging_file(&root, &path).unwrap();
        file.write_all(b"actual").unwrap();
        drop(file);
        let wrong = Sha256Digest::digest_bytes(b"expected");

        assert!(install_private_immutable_file(&root, &temporary, &path, &wrong).is_err());
        assert!(!temporary.exists());
        assert!(!path.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_streaming_producer_leaves_no_partial_file() {
        let root = root("streaming-failure");
        let path = root.join("state/object.bin");
        let result = stage_private_file(&root, &path, |file| {
            file.write_all(b"partial")?;
            Err(BrainError::Integrity("injected_stream_failure".into()))
        });
        assert!(result.is_err());
        assert!(!path.exists());
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn staging_installer_rejects_symlink_collision_without_touching_target() {
        let root = root("installer-symlink");
        let outside = root.join("outside.bin");
        fs::write(&outside, b"outside").unwrap();
        let destination = root.join("state/object.bin");
        let (temporary, mut file) = create_private_staging_file(&root, &destination).unwrap();
        file.write_all(b"inside").unwrap();
        drop(file);
        symlink(&outside, &destination).unwrap();
        let expected = Sha256Digest::digest_bytes(b"inside");

        assert!(
            install_private_immutable_file(&root, &temporary, &destination, &expected).is_err()
        );
        assert!(!temporary.exists());
        assert_eq!(fs::read(outside).unwrap(), b"outside");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn transactional_move_never_overwrites_and_recovers_its_own_link() {
        let root = root("transactional-move");
        let source = root.join("live/object.bin");
        let destination = root.join("archive/object.bin");
        ensure_private_parent(&root, &source).unwrap();
        fs::write(&source, b"payload").unwrap();
        let expected = Sha256Digest::digest_bytes(b"payload");

        // Reproduce the only intermediate crash state: destination linked and
        // synced, but source name not unlinked yet.
        ensure_private_parent(&root, &destination).unwrap();
        fs::hard_link(&source, &destination).unwrap();
        move_private_file_transactional(&root, &source, &destination, &expected).unwrap();
        assert!(!source.exists());
        assert_eq!(fs::read(&destination).unwrap(), b"payload");
        move_private_file_transactional(&root, &source, &destination, &expected).unwrap();

        let second_source = root.join("live/second.bin");
        fs::write(&second_source, b"other").unwrap();
        let other = Sha256Digest::digest_bytes(b"other");
        assert!(
            move_private_file_transactional(&root, &second_source, &destination, &other).is_err()
        );
        assert_eq!(fs::read(&second_source).unwrap(), b"other");
        assert_eq!(fs::read(&destination).unwrap(), b"payload");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn directory_move_preserves_complete_identity_and_retry_is_idempotent() {
        let root = root("directory-move-retry");
        let source = root.join("inflight/operation");
        let destination = root.join("archive/operation");
        ensure_private_directory(&root, &source.join("nested")).unwrap();
        fs::write(source.join("intent.json"), b"intent").unwrap();
        fs::write(source.join("nested/evidence.bin"), b"evidence").unwrap();
        let expected = inspect_private_directory(&root, &source).unwrap();

        move_private_directory_transactional(&root, &source, &destination, &expected).unwrap();
        assert!(!source.exists());
        assert_eq!(
            inspect_private_directory(&root, &destination).unwrap(),
            expected
        );
        move_private_directory_transactional(&root, &source, &destination, &expected).unwrap();
        assert_eq!(
            fs::read(destination.join("nested/evidence.bin")).unwrap(),
            b"evidence"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn directory_move_rejects_collision_and_tree_tampering_without_mutation() {
        let root = root("directory-move-collision");
        let source = root.join("inflight/operation");
        let destination = root.join("archive/operation");
        ensure_private_directory(&root, &source).unwrap();
        ensure_private_directory(&root, &destination).unwrap();
        fs::write(source.join("intent.json"), b"source").unwrap();
        fs::write(destination.join("intent.json"), b"destination").unwrap();
        let expected = inspect_private_directory(&root, &source).unwrap();

        assert!(
            move_private_directory_transactional(&root, &source, &destination, &expected).is_err()
        );
        assert_eq!(fs::read(source.join("intent.json")).unwrap(), b"source");
        assert_eq!(
            fs::read(destination.join("intent.json")).unwrap(),
            b"destination"
        );

        fs::remove_dir_all(&destination).unwrap();
        fs::write(source.join("intent.json"), b"tampered").unwrap();
        assert!(
            move_private_directory_transactional(&root, &source, &destination, &expected).is_err()
        );
        assert!(source.exists());
        assert!(!destination.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn directory_identity_and_move_reject_nested_symlinks() {
        let root = root("directory-move-symlink");
        let source = root.join("inflight/operation");
        let destination = root.join("archive/operation");
        ensure_private_directory(&root, &source).unwrap();
        let outside = root.join("outside.bin");
        fs::write(&outside, b"outside").unwrap();
        symlink(&outside, source.join("redirect.bin")).unwrap();

        assert!(inspect_private_directory(&root, &source).is_err());
        assert!(!destination.exists());
        assert_eq!(fs::read(outside).unwrap(), b"outside");

        fs::remove_file(source.join("redirect.bin")).unwrap();
        fs::write(source.join("intent.json"), b"intent").unwrap();
        let expected = inspect_private_directory(&root, &source).unwrap();
        let redirected_parent = root.join("redirected-archive");
        fs::create_dir(&redirected_parent).unwrap();
        symlink(&redirected_parent, root.join("archive")).unwrap();
        assert!(
            move_private_directory_transactional(&root, &source, &destination, &expected).is_err()
        );
        assert!(source.exists());
        assert_eq!(fs::read_dir(redirected_parent).unwrap().count(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn concurrent_directory_moves_never_replace_the_winner() {
        let root = Arc::new(root("directory-move-race"));
        let destination = root.join("archive/operation");
        let barrier = Arc::new(Barrier::new(2));
        let mut moves = Vec::new();
        for index in 0..2 {
            let source = root.join(format!("inflight/operation-{index}"));
            ensure_private_directory(&root, &source).unwrap();
            fs::write(source.join("intent.json"), format!("intent-{index}")).unwrap();
            let expected = inspect_private_directory(&root, &source).unwrap();
            let root = Arc::clone(&root);
            let destination = destination.clone();
            let barrier = Arc::clone(&barrier);
            moves.push(thread::spawn(move || {
                barrier.wait();
                move_private_directory_transactional(&root, &source, &destination, &expected)
            }));
        }
        let successes = moves
            .into_iter()
            .map(|operation| usize::from(operation.join().unwrap().is_ok()))
            .sum::<usize>();
        assert_eq!(successes, 1);
        let installed = fs::read(destination.join("intent.json")).unwrap();
        assert!(installed == b"intent-0" || installed == b"intent-1");
        fs::remove_dir_all(root.as_ref()).unwrap();
    }

    #[test]
    fn concurrent_idempotent_writers_install_one_complete_object() {
        let root = Arc::new(root("concurrent-same-content"));
        let path = root.join("state/object.bin");
        let barrier = Arc::new(Barrier::new(8));
        let writers = (0..8)
            .map(|_| {
                let root = Arc::clone(&root);
                let path = path.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    write_or_verify_immutable(&root, &path, b"complete immutable object")
                })
            })
            .collect::<Vec<_>>();
        for writer in writers {
            assert!(writer.join().unwrap().is_ok());
        }
        assert_eq!(fs::read(&path).unwrap(), b"complete immutable object");
        assert_eq!(
            fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1,
            "no staging file may survive a successful race"
        );
        fs::remove_dir_all(root.as_ref()).unwrap();
    }

    #[test]
    fn concurrent_conflicting_writers_never_replace_the_winner() {
        let root = Arc::new(root("concurrent-conflicting-content"));
        let path = root.join("state/object.bin");
        let barrier = Arc::new(Barrier::new(2));
        let writers = [b"first".as_slice(), b"second".as_slice()]
            .into_iter()
            .map(|payload| {
                let root = Arc::clone(&root);
                let path = path.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    write_or_verify_immutable(&root, &path, payload)
                })
            })
            .collect::<Vec<_>>();
        let successes = writers
            .into_iter()
            .map(|writer| usize::from(writer.join().unwrap().is_ok()))
            .sum::<usize>();
        assert_eq!(successes, 1);
        let installed = fs::read(&path).unwrap();
        assert!(installed == b"first" || installed == b"second");
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
        fs::remove_dir_all(root.as_ref()).unwrap();
    }

    #[test]
    fn optional_resolver_rejects_a_symlinked_missing_prefix() {
        let root = root("optional-symlink");
        let outside =
            std::env::temp_dir().join(format!("cerebro-authority-outside-{}", std::process::id()));
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, root.join("state")).unwrap();
        assert!(existing_regular_file_if_present(&root, &root.join("state/future.json")).is_err());
        fs::remove_file(root.join("state")).unwrap();
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn atomic_pointer_replacement_verifies_requested_bytes() {
        let root = root("atomic-pointer");
        let path = root.join("state/current.json");
        let first = Sha256Digest::digest_bytes(b"first");
        replace_private_file_atomic(&root, &path, b"first", Some(&first)).unwrap();
        let second = Sha256Digest::digest_bytes(b"second");
        replace_private_file_atomic(&root, &path, b"second", Some(&second)).unwrap();
        assert_eq!(fs::read(path).unwrap(), b"second");
        assert!(replace_private_file_atomic(
            &root,
            &root.join("state/bad.json"),
            b"x",
            Some(&second)
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
