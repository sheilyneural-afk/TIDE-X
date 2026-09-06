//! Produce the single current source-capture authority: request, exact
//! re-verifiable envelope and retained CAS objects are sealed together in one
//! receipt.  This command performs no donor execution and never claims that a
//! captured source tree has been understood or transferred to weights.

use cerebro_tidex::acquisition_contract::{
    AcquisitionBudget, AcquisitionRequest, AcquisitionScope, DeclaredRelativePath, NoisePolicy,
    RequestedResidency,
};
use cerebro_tidex::content_vault::capture_to_vault;
use cerebro_tidex::identity::AcquisitionId;
use cerebro_tidex::security::configured_private_root;
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run(std::env::args_os().skip(1).collect()) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run(arguments: Vec<OsString>) -> Result<(), Box<dyn std::error::Error>> {
    let invocation = parse(arguments)?;
    let private_root = configured_private_root()?;
    let request = AcquisitionRequest::new(
        AcquisitionId::parse(&invocation.acquisition_id)?,
        invocation.scope,
        invocation.residency,
        invocation.noise_policy,
        AcquisitionBudget {
            max_files: invocation.max_files,
            max_total_bytes: invocation.max_total_bytes,
        },
        invocation.exclusions,
    )?;
    let receipt = capture_to_vault(&invocation.source_root, &private_root, &request)?;
    let receipt_reference = receipt.persist(&private_root)?;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "acquisition_id": receipt.request().acquisition_id(),
            "capture_receipt_sha256": receipt.manifest_sha256(),
            "capture_receipt": receipt_reference,
            "system_envelope_sha256": receipt.envelope().manifest_sha256(),
            "completeness": receipt.envelope().completeness(),
            "entries": receipt.envelope().entries().len(),
            "bytes": receipt.total_file_bytes(),
        }))?
    );
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct Invocation {
    source_root: PathBuf,
    acquisition_id: String,
    scope: AcquisitionScope,
    residency: RequestedResidency,
    max_files: u64,
    max_total_bytes: u64,
    exclusions: Vec<PathBuf>,
    noise_policy: NoisePolicy,
}

fn parse(arguments: Vec<OsString>) -> Result<Invocation, String> {
    let mut source_root = None;
    let mut acquisition_id = None;
    let mut scope = None;
    let mut residency = RequestedResidency::BestVerified;
    let mut max_files = 100_000;
    let mut max_total_bytes = 8 * 1024 * 1024 * 1024;
    let mut exclusions = Vec::new();
    let mut explicit_paths = Vec::new();
    // Generated artifacts can contain required binaries, models or oracles.
    // Automatic exclusion is therefore an explicit opt-in, never the default.
    let mut noise_policy = NoisePolicy::ExplicitOnly;
    let mut singleton_flags = BTreeSet::new();
    let mut iterator = arguments.into_iter();
    while let Some(argument) = iterator.next() {
        let flag = argument
            .to_str()
            .ok_or_else(|| "argument_name_not_utf8".to_string())?;
        let mut value = || {
            iterator
                .next()
                .ok_or_else(|| format!("{flag}_value_missing"))
        };
        match flag {
            "--source-root" => {
                mark_singleton(&mut singleton_flags, "source_root")?;
                source_root = Some(PathBuf::from(value()?));
            }
            "--acquisition-id" => {
                mark_singleton(&mut singleton_flags, "acquisition_id")?;
                acquisition_id = Some(utf8_value(value()?, "acquisition_id")?);
            }
            "--scope" => {
                mark_singleton(&mut singleton_flags, "scope")?;
                let scope_value = utf8_value(value()?, "scope")?;
                scope = Some(match scope_value.as_str() {
                    "whole-project" => AcquisitionScope::WholeProject,
                    "declared-paths" => AcquisitionScope::DeclaredPaths { roots: Vec::new() },
                    _ => {
                        let roots: Vec<_> = scope_value
                            .strip_prefix("paths:")
                            .ok_or_else(|| "scope_invalid".to_string())?
                            .split(',')
                            .map(PathBuf::from)
                            .collect();
                        if roots.is_empty() || roots.iter().any(|root| root.as_os_str().is_empty())
                        {
                            return Err("scope_invalid".into());
                        }
                        AcquisitionScope::declared_paths(roots)
                            .map_err(|_| "scope_invalid".to_string())?
                    }
                });
            }
            // Repeatable and lossless: unlike the compatibility `paths:a,b`
            // form, this can represent a Unix filename that contains a comma.
            "--path" => explicit_paths.push(
                DeclaredRelativePath::parse(PathBuf::from(value()?))
                    .map_err(|_| "path_invalid".to_string())?,
            ),
            "--residency" => {
                mark_singleton(&mut singleton_flags, "residency")?;
                residency = match utf8_value(value()?, "residency")?.as_str() {
                    "best-verified" => RequestedResidency::BestVerified,
                    "weights-only" => RequestedResidency::WeightsOnly,
                    "portable-ir-only" => RequestedResidency::PortableIrOnly,
                    _ => return Err("residency_invalid".into()),
                }
            }
            "--max-files" => {
                mark_singleton(&mut singleton_flags, "max_files")?;
                max_files = utf8_value(value()?, "max_files")?
                    .parse()
                    .map_err(|_| "max_files_invalid")?;
            }
            "--max-bytes" => {
                mark_singleton(&mut singleton_flags, "max_bytes")?;
                max_total_bytes = utf8_value(value()?, "max_bytes")?
                    .parse()
                    .map_err(|_| "max_bytes_invalid")?;
            }
            "--exclude" => exclusions.push(PathBuf::from(value()?)),
            "--noise-policy" => {
                mark_singleton(&mut singleton_flags, "noise_policy")?;
                noise_policy = match utf8_value(value()?, "noise_policy")?.as_str() {
                    "conservative" => NoisePolicy::ConservativeGeneratedArtifacts,
                    "explicit-only" => NoisePolicy::ExplicitOnly,
                    _ => return Err("noise_policy_invalid".into()),
                }
            }
            _ => return Err("acquire_system_argument_invalid".into()),
        }
    }
    let scope = match scope.ok_or_else(|| "scope_missing".to_string())? {
        AcquisitionScope::WholeProject if !explicit_paths.is_empty() => {
            return Err("path_with_whole_project_scope".into());
        }
        AcquisitionScope::WholeProject => AcquisitionScope::WholeProject,
        AcquisitionScope::DeclaredPaths { mut roots } => {
            roots.extend(explicit_paths);
            if roots.is_empty() {
                return Err("declared_paths_empty".into());
            }
            AcquisitionScope::declared_paths(
                roots
                    .into_iter()
                    .map(DeclaredRelativePath::into_path_buf)
                    .collect(),
            )
            .map_err(|_| "scope_invalid".to_string())?
        }
    };
    Ok(Invocation {
        source_root: source_root.ok_or_else(|| "source_root_missing".to_string())?,
        acquisition_id: acquisition_id.ok_or_else(|| "acquisition_id_missing".to_string())?,
        scope,
        residency,
        max_files,
        max_total_bytes,
        exclusions,
        noise_policy,
    })
}

fn utf8_value(value: OsString, label: &str) -> Result<String, String> {
    value.into_string().map_err(|_| format!("{label}_not_utf8"))
}

fn mark_singleton(observed: &mut BTreeSet<&'static str>, flag: &'static str) -> Result<(), String> {
    if !observed.insert(flag) {
        return Err(format!("{flag}_duplicate"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_rejects_an_ambiguous_or_incomplete_acquisition() {
        assert!(parse(vec![]).is_err());
        assert!(parse(vec!["--source-root".into(), "/tmp/project".into()]).is_err());
        assert!(parse(vec![
            "--source-root".into(),
            "/tmp/project".into(),
            "--acquisition-id".into(),
            "capture-1".into(),
            "--scope".into(),
            "unknown".into(),
        ])
        .is_err());
    }

    #[test]
    fn parser_requires_a_sealed_scope_and_has_no_provider_default() {
        let invocation = parse(vec![
            "--source-root".into(),
            "/tmp/project".into(),
            "--acquisition-id".into(),
            "capture-1".into(),
            "--scope".into(),
            "paths:src,config".into(),
            "--residency".into(),
            "weights-only".into(),
            "--exclude".into(),
            "target".into(),
        ])
        .unwrap();
        assert_eq!(invocation.residency, RequestedResidency::WeightsOnly);
        assert_eq!(invocation.noise_policy, NoisePolicy::ExplicitOnly);
        assert_eq!(invocation.exclusions, vec![PathBuf::from("target")]);
        assert_eq!(
            invocation.scope,
            AcquisitionScope::declared_paths(vec![PathBuf::from("src"), PathBuf::from("config")])
                .unwrap()
        );
    }

    #[test]
    fn parser_rejects_duplicate_singleton_flags() {
        assert_eq!(
            parse(vec![
                "--source-root".into(),
                "/tmp/first".into(),
                "--source-root".into(),
                "/tmp/second".into(),
                "--acquisition-id".into(),
                "capture-1".into(),
                "--scope".into(),
                "whole-project".into(),
            ]),
            Err("source_root_duplicate".into())
        );
        assert!(parse(vec![
            "--source-root".into(),
            "/tmp/project".into(),
            "--acquisition-id".into(),
            "capture-1".into(),
            "--scope".into(),
            "whole-project".into(),
            "--noise-policy".into(),
            "explicit-only".into(),
            "--noise-policy".into(),
            "conservative".into(),
        ])
        .is_err());
    }

    #[test]
    fn repeated_path_flags_preserve_paths_containing_commas() {
        let invocation = parse(vec![
            "--source-root".into(),
            "/tmp/project".into(),
            "--acquisition-id".into(),
            "capture-1".into(),
            "--scope".into(),
            "declared-paths".into(),
            "--path".into(),
            "src".into(),
            "--path".into(),
            "models/a,b.bin".into(),
        ])
        .unwrap();
        assert_eq!(
            invocation.scope,
            AcquisitionScope::declared_paths(vec![
                PathBuf::from("src"),
                PathBuf::from("models/a,b.bin")
            ])
            .unwrap()
        );
        assert!(parse(vec![
            "--source-root".into(),
            "/tmp/project".into(),
            "--acquisition-id".into(),
            "capture-1".into(),
            "--scope".into(),
            "whole-project".into(),
            "--path".into(),
            "src".into(),
        ])
        .is_err());
    }
}
