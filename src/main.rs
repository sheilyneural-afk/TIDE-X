use cerebro_tidex::contracts::{BrainConfig, DeltaObservation};
use cerebro_tidex::engine::BrainEngine;
use cerebro_tidex::security::PRIVATE_ROOT;
use std::fs;
use std::path::Path;

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
            let engine = BrainEngine::open(PRIVATE_ROOT, BrainConfig::default())?;
            println!("{}", serde_json::to_string_pretty(&engine.status()?)?);
        }
        "analyze" => {
            let observation_path = args.get(2).ok_or("observation JSON path required")?;
            let observations: Vec<DeltaObservation> =
                serde_json::from_slice(&fs::read(Path::new(observation_path))?)?;
            let engine = BrainEngine::open(PRIVATE_ROOT, BrainConfig::default())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&engine.analyze(&observations)?)?
            );
        }
        "sleep" => {
            let engine = BrainEngine::open(PRIVATE_ROOT, BrainConfig::default())?;
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
