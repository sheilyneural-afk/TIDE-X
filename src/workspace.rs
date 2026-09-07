use crate::error::{BrainError, BrainResult};
use crate::security::{secure_dir, secure_file};
use serde::{Deserialize, Serialize};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Component, Path, PathBuf};

const WORKSPACE_SCHEMA: &str = "cerebro.tidex.workspace/v1";
const MODEL_SCHEMA: &str = "cerebro.tidex.model_profile/v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceManifest {
    pub schema: String,
    pub name: String,
    pub target: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelProvider {
    OpenAiCompatible,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelProfile {
    pub schema: String,
    pub name: String,
    pub provider: ModelProvider,
    pub endpoint: String,
    pub model: String,
}

pub fn configured_tidex_home() -> BrainResult<PathBuf> {
    let home = std::env::var_os("TIDEX_HOME")
        .map(PathBuf::from)
        .ok_or_else(|| BrainError::Invalid("tidex_home_not_configured".into()))?;
    verify_private_directory(&home, "tidex_home")
}

pub fn create_workspace(home: &Path, name: &str, target: &Path) -> BrainResult<WorkspaceManifest> {
    let home = verify_private_directory(home, "tidex_home")?;
    validate_name(name, "workspace_name")?;
    let target = verify_target(target)?;
    let workspace_dir = home.join("workspaces").join(name);
    if fs::symlink_metadata(&workspace_dir).is_ok() {
        return Err(BrainError::Integrity("workspace_already_exists".into()));
    }
    fs::create_dir_all(workspace_dir.join("state"))?;
    secure_dir(&home.join("workspaces"))?;
    secure_dir(&workspace_dir)?;
    secure_dir(&workspace_dir.join("state"))?;
    let manifest = WorkspaceManifest {
        schema: WORKSPACE_SCHEMA.into(),
        name: name.into(),
        target,
    };
    write_canonical_json(&workspace_dir.join("workspace.json"), &manifest)?;
    Ok(manifest)
}

pub fn load_workspace(home: &Path, name: &str) -> BrainResult<WorkspaceManifest> {
    let home = verify_private_directory(home, "tidex_home")?;
    validate_name(name, "workspace_name")?;
    let workspace_dir = verify_private_directory(&home.join("workspaces").join(name), "workspace")?;
    let manifest: WorkspaceManifest = read_canonical_json(&workspace_dir.join("workspace.json"))?;
    if manifest.schema != WORKSPACE_SCHEMA || manifest.name != name || verify_target(&manifest.target)? != manifest.target {
        return Err(BrainError::Integrity("workspace_manifest_invalid".into()));
    }
    Ok(manifest)
}

pub fn use_workspace(home: &Path, name: &str) -> BrainResult<()> {
    load_workspace(home, name)?;
    write_canonical_json(&home.join("current-workspace.json"), &serde_json::json!({
        "schema":"cerebro.tidex.current_workspace/v1",
        "name":name
    }))
}

pub fn current_workspace(home: &Path) -> BrainResult<WorkspaceManifest> {
    let value: serde_json::Value = read_canonical_json(&home.join("current-workspace.json"))?;
    let name = value.get("name").and_then(|v| v.as_str()).ok_or_else(|| BrainError::Integrity("current_workspace_invalid".into()))?;
    if value.get("schema").and_then(|v| v.as_str()) != Some("cerebro.tidex.current_workspace/v1") || value.as_object().map(|o| o.len()) != Some(2) {
        return Err(BrainError::Integrity("current_workspace_invalid".into()));
    }
    load_workspace(home, name)
}

pub fn add_model(home: &Path, profile: ModelProfile) -> BrainResult<()> {
    let home = verify_private_directory(home, "tidex_home")?;
    validate_name(&profile.name, "model_name")?;
    validate_model(&profile)?;
    let models = home.join("models");
    fs::create_dir_all(&models)?;
    secure_dir(&models)?;
    let path = models.join(format!("{}.json", profile.name));
    if fs::symlink_metadata(&path).is_ok() {
        return Err(BrainError::Integrity("model_profile_already_exists".into()));
    }
    write_canonical_json(&path, &profile)
}

pub fn use_model(home: &Path, name: &str) -> BrainResult<()> {
    let profile = load_model(home, name)?;
    write_canonical_json(&home.join("current-model.json"), &serde_json::json!({
        "schema":"cerebro.tidex.current_model/v1",
        "name":profile.name
    }))
}

pub fn load_model(home: &Path, name: &str) -> BrainResult<ModelProfile> {
    let home = verify_private_directory(home, "tidex_home")?;
    validate_name(name, "model_name")?;
    let profile: ModelProfile = read_canonical_json(&home.join("models").join(format!("{name}.json")))?;
    validate_model(&profile)?;
    if profile.name != name { return Err(BrainError::Integrity("model_profile_identity_mismatch".into())); }
    Ok(profile)
}

fn validate_model(profile: &ModelProfile) -> BrainResult<()> {
    if profile.schema != MODEL_SCHEMA || profile.model.trim().is_empty() || profile.model.len() > 512 {
        return Err(BrainError::Invalid("model_profile_invalid".into()));
    }
    let endpoint = profile.endpoint.strip_prefix("http://").or_else(|| profile.endpoint.strip_prefix("https://"))
        .ok_or_else(|| BrainError::Invalid("model_endpoint_scheme_invalid".into()))?;
    if endpoint.is_empty() || endpoint.contains(char::is_whitespace) || profile.endpoint.len() > 4096 {
        return Err(BrainError::Invalid("model_endpoint_invalid".into()));
    }
    Ok(())
}

fn verify_target(target: &Path) -> BrainResult<PathBuf> {
    if !target.is_absolute() { return Err(BrainError::Invalid("workspace_target_must_be_absolute".into())); }
    let metadata = fs::symlink_metadata(target).map_err(|e| BrainError::Integrity(format!("workspace_target_unreadable:{e}")))?;
    if metadata.file_type().is_symlink() || !(metadata.is_dir() || metadata.is_file()) {
        return Err(BrainError::Integrity("workspace_target_invalid".into()));
    }
    Ok(target.canonicalize()?)
}

fn verify_private_directory(path: &Path, label: &str) -> BrainResult<PathBuf> {
    if !path.is_absolute() { return Err(BrainError::Invalid(format!("{label}_must_be_absolute"))); }
    let metadata = fs::symlink_metadata(path).map_err(|e| BrainError::Integrity(format!("{label}_unreadable:{e}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() || metadata.permissions().mode() & 0o077 != 0 {
        return Err(BrainError::Integrity(format!("{label}_not_private")));
    }
    Ok(path.canonicalize()?)
}

fn validate_name(name: &str, label: &str) -> BrainResult<()> {
    if name.is_empty() || name.len() > 128 || name == "." || name == ".." || name.chars().any(|c| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))) {
        return Err(BrainError::Invalid(format!("{label}_invalid")));
    }
    Ok(())
}

fn write_canonical_json<T: Serialize>(path: &Path, value: &T) -> BrainResult<()> {
    if path.components().any(|c| matches!(c, Component::ParentDir)) { return Err(BrainError::Invalid("workspace_path_invalid".into())); }
    let bytes = serde_json::to_vec(value)?;
    fs::write(path, bytes)?;
    secure_file(path)?;
    Ok(())
}

fn read_canonical_json<T: serde::de::DeserializeOwned + Serialize>(path: &Path) -> BrainResult<T> {
    let metadata = fs::symlink_metadata(path).map_err(|e| BrainError::Integrity(format!("workspace_record_unreadable:{e}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.permissions().mode() & 0o077 != 0 {
        return Err(BrainError::Integrity("workspace_record_invalid".into()));
    }
    let bytes = fs::read(path)?;
    let value: T = serde_json::from_slice(&bytes)?;
    if serde_json::to_vec(&value)? != bytes { return Err(BrainError::Integrity("workspace_record_noncanonical".into())); }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn roots() -> (PathBuf, PathBuf) {
        let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let base = std::env::temp_dir().join(format!("tidex-workspace-{}-{n}", std::process::id()));
        let home = base.join("home"); let target = base.join("target");
        fs::create_dir_all(&home).unwrap(); fs::create_dir_all(&target).unwrap(); secure_dir(&home).unwrap();
        (home, target)
    }

    #[test]
    fn workspace_and_model_selection_are_persistent_and_separate_from_target() {
        let (home, target) = roots();
        let ws = create_workspace(&home, "demo", &target).unwrap();
        assert_eq!(ws.target, target.canonicalize().unwrap());
        use_workspace(&home, "demo").unwrap();
        assert_eq!(current_workspace(&home).unwrap().name, "demo");
        add_model(&home, ModelProfile { schema: MODEL_SCHEMA.into(), name:"qwen".into(), provider:ModelProvider::OpenAiCompatible, endpoint:"http://127.0.0.1:8080/v1".into(), model:"Qwen".into() }).unwrap();
        use_model(&home, "qwen").unwrap();
        assert_eq!(load_model(&home, "qwen").unwrap().model, "Qwen");
        fs::remove_dir_all(home.parent().unwrap()).unwrap();
    }

    #[test]
    fn rejects_non_private_home_relative_target_and_unsafe_names() {
        let (home, target) = roots();
        fs::set_permissions(&home, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(create_workspace(&home, "demo", &target).is_err());
        fs::set_permissions(&home, fs::Permissions::from_mode(0o700)).unwrap();
        assert!(create_workspace(&home, "../escape", &target).is_err());
        assert!(create_workspace(&home, "demo", Path::new("relative")).is_err());
        fs::remove_dir_all(home.parent().unwrap()).unwrap();
    }
}
