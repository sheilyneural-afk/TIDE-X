use crate::error::{BrainError, BrainResult};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

/// Resolve the private state root for this installation.
///
/// The path is configuration, not part of TIDE-X's identity.  It must already
/// exist as a private directory: creation belongs to the installer, where
/// ownership and deployment policy can be established explicitly.
pub fn configured_private_root() -> BrainResult<PathBuf> {
    let configured = std::env::var_os("TIDEX_PRIVATE_ROOT")
        .map(PathBuf::from)
        .ok_or_else(|| BrainError::Invalid("private_root_not_configured".into()))?;
    if !configured.is_absolute() {
        return Err(BrainError::Invalid("private_root_must_be_absolute".into()));
    }
    let metadata = fs::symlink_metadata(&configured)
        .map_err(|error| BrainError::Integrity(format!("private_root_unreadable:{error}")))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err(BrainError::Integrity("private_root_invalid".into()));
    }
    let canonical = configured.canonicalize()?;
    verify_private_root_against(&canonical, &canonical)
}

pub fn verify_private_root(root: &Path) -> BrainResult<PathBuf> {
    let expected = configured_private_root()?;
    let actual = fs::canonicalize(root)
        .map_err(|e| BrainError::Integrity(format!("root_unreadable:{e}")))?;
    verify_private_root_against(&actual, &expected)
}

/// Internal authority boundary. Production has exactly the same configured-root
/// requirement as [`verify_private_root`]. Unit tests may instead use their own
/// isolated 0700 root so parallel fixtures do not mutate a process-global
/// environment variable.
pub(crate) fn verify_internal_private_root(root: &Path) -> BrainResult<PathBuf> {
    #[cfg(not(test))]
    {
        verify_private_root(root)
    }
    #[cfg(test)]
    {
        if !root.is_absolute() {
            return Err(BrainError::Invalid("private_root_must_be_absolute".into()));
        }
        let actual = fs::canonicalize(root)
            .map_err(|error| BrainError::Integrity(format!("root_unreadable:{error}")))?;
        verify_private_root_against(&actual, &actual)
    }
}

fn verify_private_root_against(actual: &Path, expected: &Path) -> BrainResult<PathBuf> {
    if actual != expected {
        return Err(BrainError::Integrity(format!(
            "root_must_be_private:{}",
            expected.display()
        )));
    }
    let md = fs::symlink_metadata(actual)?;
    if md.file_type().is_symlink() {
        return Err(BrainError::Integrity(
            "private_root_symlink_forbidden".into(),
        ));
    }
    let mode = md.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(BrainError::Integrity(format!(
            "private_root_permissions_too_open:{mode:o}"
        )));
    }
    Ok(actual.to_path_buf())
}

pub fn secure_file(path: &Path) -> BrainResult<()> {
    let mut p = fs::metadata(path)?.permissions();
    p.set_mode(0o600);
    fs::set_permissions(path, p)?;
    Ok(())
}
pub fn secure_dir(path: &Path) -> BrainResult<()> {
    let mut p = fs::metadata(path)?.permissions();
    p.set_mode(0o700);
    fs::set_permissions(path, p)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_root_policy_is_portable_but_stays_fail_closed() {
        let root = std::env::temp_dir().join(format!("tidex-private-root-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        secure_dir(&root).unwrap();
        assert_eq!(verify_private_root_against(&root, &root).unwrap(), root);
        assert!(
            verify_private_root_against(&root, Path::new("/definitely-not-this-root")).is_err()
        );
        let _ = fs::remove_dir_all(root);
    }
}
