use crate::digest::{sha256_file, Sha256Digest};
use crate::error::{BrainError, BrainResult};
use crate::security::{secure_dir, secure_file};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};

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

impl PrivateFileReference {
    pub fn new(path: impl Into<PathBuf>, sha256: Sha256Digest) -> Self {
        Self {
            path: path.into(),
            sha256,
        }
    }

    /// Verify path confinement and exact current file content.
    pub fn verify(&self, root: &Path) -> BrainResult<PathBuf> {
        let path = existing_regular_file_under_root(root, &self.path)?;
        if sha256_file(&path)? != self.sha256 {
            return Err(BrainError::Integrity(
                "private_file_reference_digest_mismatch".into(),
            ));
        }
        Ok(path)
    }

    /// Read the exact verified byte sequence in one authority operation.
    pub fn read_verified(&self, root: &Path) -> BrainResult<Vec<u8>> {
        Ok(self.read_verified_with_path(root)?.1)
    }

    /// Resolve confinement and return the path together with the exact bytes
    /// whose digest matches this reference. Consumers that need both should use
    /// this instead of resolving and hashing in separate subsystem helpers.
    pub fn read_verified_with_path(&self, root: &Path) -> BrainResult<(PathBuf, Vec<u8>)> {
        let path = existing_regular_file_under_root(root, &self.path)?;
        let bytes = fs::read(&path)?;
        if Sha256Digest::digest_bytes(&bytes) != self.sha256 {
            return Err(BrainError::Integrity(
                "private_file_reference_digest_mismatch".into(),
            ));
        }
        Ok((path, bytes))
    }
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
                fs::create_dir(&cursor)?;
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
                fs::create_dir(&cursor)?;
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
    ensure_private_parent(root, path)?;
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
                return Err(BrainError::Integrity(
                    "private_file_immutable_target_invalid".into(),
                ));
            }
            let verified = existing_regular_file_under_root(root, path)?;
            if sha256_file(&verified)? != expected {
                return Err(BrainError::Integrity(
                    "private_file_immutable_content_mismatch".into(),
                ));
            }
            secure_file(path)?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(path)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            drop(file);
            secure_file(path)?;
            let verified = existing_regular_file_under_root(root, path)?;
            if sha256_file(&verified)? != expected {
                return Err(BrainError::Integrity(
                    "private_file_written_content_mismatch".into(),
                ));
            }
        }
        Err(error) => return Err(error.into()),
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
    write_or_verify_immutable(root, path, bytes)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
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
    fn fresh_immutable_write_never_accepts_a_preexisting_leaf() {
        let root = root("fresh-immutable");
        let path = root.join("state/receipt.json");
        create_private_immutable(&root, &path, b"first").unwrap();
        assert!(create_private_immutable(&root, &path, b"first").is_err());
        fs::remove_dir_all(root).unwrap();
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
