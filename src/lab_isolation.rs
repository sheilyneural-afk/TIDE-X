//! Laboratory-root admission. Experimental state must never share a root with production.

use crate::authority::existing_directory_under_root;
use crate::error::{BrainError, BrainResult};
#[cfg(test)]
use crate::security::verify_internal_private_root;
use crate::security::verify_private_root;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabRoots {
    lab_root: PathBuf,
    state_root: PathBuf,
    artifact_root: PathBuf,
}

impl LabRoots {
    pub fn open(
        lab_root: &Path,
        state_root: &Path,
        artifact_root: &Path,
        production_root: &Path,
    ) -> BrainResult<Self> {
        let lab_root = verify_private_root(lab_root)?;
        Self::admit(lab_root, state_root, artifact_root, production_root)
    }

    #[cfg(test)]
    pub(crate) fn open_for_test(
        lab_root: &Path,
        state_root: &Path,
        artifact_root: &Path,
        production_root: &Path,
    ) -> BrainResult<Self> {
        let lab_root = verify_internal_private_root(lab_root)?;
        Self::admit(lab_root, state_root, artifact_root, production_root)
    }

    fn admit(
        lab_root: PathBuf,
        state_root: &Path,
        artifact_root: &Path,
        production_root: &Path,
    ) -> BrainResult<Self> {
        for (root, label) in [
            (state_root, "laboratory_state_root_invalid"),
            (artifact_root, "laboratory_artifact_root_invalid"),
        ] {
            let metadata =
                fs::symlink_metadata(root).map_err(|_| BrainError::Invalid(label.into()))?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(BrainError::Integrity(label.into()));
            }
        }
        let state_root = state_root
            .canonicalize()
            .map_err(|_| BrainError::Invalid("laboratory_state_root_unavailable".into()))?;
        let artifact_root = artifact_root
            .canonicalize()
            .map_err(|_| BrainError::Invalid("laboratory_artifact_root_unavailable".into()))?;
        let production_root = production_root
            .canonicalize()
            .map_err(|_| BrainError::Invalid("production_root_unavailable".into()))?;
        if !state_root.starts_with(&lab_root)
            || !artifact_root.starts_with(&lab_root)
            || state_root.starts_with(&artifact_root)
            || artifact_root.starts_with(&state_root)
            || state_root.starts_with(&production_root)
            || artifact_root.starts_with(&production_root)
            || production_root.starts_with(&lab_root)
        {
            return Err(BrainError::Integrity(
                "laboratory_roots_not_isolated".into(),
            ));
        }
        existing_directory_under_root(&lab_root, &state_root)?;
        existing_directory_under_root(&lab_root, &artifact_root)?;
        Ok(Self {
            lab_root,
            state_root,
            artifact_root,
        })
    }

    pub fn lab_root(&self) -> &Path {
        &self.lab_root
    }

    pub fn state_root(&self) -> &Path {
        &self.state_root
    }

    pub fn artifact_root(&self) -> &Path {
        &self.artifact_root
    }
}
