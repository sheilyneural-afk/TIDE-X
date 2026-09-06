use cerebro_tidex::authority::read_untrusted_private_file_bounded;
use cerebro_tidex::contracts::{BrainConfig, DeltaObservation};
use cerebro_tidex::engine::BrainEngine;
use cerebro_tidex::security::configured_private_root;
use std::path::Path;

const MAX_ANALYZE_INPUT_BYTES: u64 = 16 * 1024 * 1024;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

/// The default production binary exposes diagnostics plus the one canonical
/// sleep transaction. Learning/finalization mutations enter only through their
/// receipt-bound TIDE-X binaries; `sleep` itself seals a full ledger-bound
/// transaction and cannot accept observations or parameter deltas from CLI.
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let command = args.get(1).map(String::as_str).unwrap_or("status");
    match command {
        "status" => {
            let private_root = configured_private_root()?;
            let engine = BrainEngine::open(&private_root, BrainConfig::default())?;
            println!("{}", serde_json::to_string_pretty(&engine.status()?)?);
        }
        "analyze" => {
            let observation_path = args.get(2).ok_or("observation JSON path required")?;
            let private_root = configured_private_root()?;
            let observations = load_analyze_observations(&private_root, Path::new(observation_path))?;
            let engine = BrainEngine::open(&private_root, BrainConfig::default())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&engine.analyze(&observations)?)?
            );
        }
        "sleep" => {
            let private_root = configured_private_root()?;
            let engine = BrainEngine::open(&private_root, BrainConfig::default())?;
            println!("{}", serde_json::to_string_pretty(&engine.sleep_cycle()?)?);
        }
        "commit" => {
            return Err(
                "retired command:commit; use the receipt-bound TIDE-X learning finalizer".into(),
            )
        }
        "artifact-import-f32" => {
            return Err(
                "retired command:artifact-import-f32; arbitrary raw delta ingestion is not a governed route"
                    .into(),
            )
        }
        "artifact-ties" => {
            return Err(
                "retired command:artifact-ties; arbitrary parameter merging is not a governed route"
                    .into(),
            )
        }
        other => {
            return Err(format!(
                "unknown command:{other}; allowed=status|analyze|sleep; canonical learning and finalization use their receipt-bound binaries"
            )
            .into())
        }
    }
    Ok(())
}

fn load_analyze_observations(
    private_root: &Path,
    observation_path: &Path,
) -> Result<Vec<DeltaObservation>, Box<dyn std::error::Error>> {
    let bytes = read_untrusted_private_file_bounded(
        private_root,
        observation_path,
        MAX_ANALYZE_INPUT_BYTES,
    )?;
    Ok(serde_json::from_slice(&bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerebro_tidex::security::{secure_dir, secure_file};
    use std::fs;

    fn private_root(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "tidex-main-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        secure_dir(&root).unwrap();
        root
    }

    #[test]
    fn analyze_input_is_confined_to_private_root_and_bounded() {
        let root = private_root("input");
        let input = root.join("observations.json");
        fs::write(&input, b"[]").unwrap();
        secure_file(&input).unwrap();
        assert!(load_analyze_observations(&root, &input).unwrap().is_empty());

        let outside = std::env::temp_dir().join(format!(
            "tidex-main-outside-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(&outside, b"[]").unwrap();
        assert!(load_analyze_observations(&root, &outside).is_err());
        let _ = fs::remove_file(outside);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn analyze_input_rejects_oversize_payload_before_json_parsing() {
        let root = private_root("oversize");
        let input = root.join("observations.json");
        fs::write(&input, vec![b' '; MAX_ANALYZE_INPUT_BYTES as usize + 1]).unwrap();
        secure_file(&input).unwrap();
        assert!(load_analyze_observations(&root, &input).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
