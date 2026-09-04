use crate::error::{BrainError, BrainResult};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

pub const PRIVATE_ROOT: &str = "/home/yo/cerebro";

pub fn verify_private_root(root: &Path) -> BrainResult<PathBuf> {
    let expected = fs::canonicalize(PRIVATE_ROOT)
        .map_err(|e| BrainError::Integrity(format!("private_root_missing:{e}")))?;
    let actual = fs::canonicalize(root)
        .map_err(|e| BrainError::Integrity(format!("root_unreadable:{e}")))?;
    if actual != expected {
        return Err(BrainError::Integrity(format!(
            "root_must_be_private:{}",
            expected.display()
        )));
    }
    let md = fs::symlink_metadata(&actual)?;
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
    if std::env::var("CEREBRO_EXTERNAL_INTEGRATION")
        .ok()
        .is_some_and(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
    {
        return Err(BrainError::Integrity(
            "external_integration_hard_blocked".into(),
        ));
    }
    Ok(actual)
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
