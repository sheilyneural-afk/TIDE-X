use cerebro_tidex::contracts::BrainConfig;
use cerebro_tidex::engine::BrainEngine;
use cerebro_tidex::learning_finalization::prepare_learning_finalization;
use cerebro_tidex::security::PRIVATE_ROOT;
use std::path::Path;

fn parse_arguments(args: &[String]) -> Result<(&str, &str), String> {
    match args {
        [session_id, representation_evidence_receipt]
            if !session_id.trim().is_empty()
                && !representation_evidence_receipt.trim().is_empty() =>
        {
            Ok((session_id, representation_evidence_receipt))
        }
        _ => {
            Err("usage: tidex_finalize <session-id> <representation-evidence-receipt.json>".into())
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    // Argument validation happens before the lifecycle or engine is opened, so
    // an invalid invocation cannot mutate adaptive state, observations, or the
    // active corpus.
    let (session_id, representation_evidence_receipt) = parse_arguments(&args)?;
    let root = Path::new(PRIVATE_ROOT);
    let input = prepare_learning_finalization(root, session_id, representation_evidence_receipt)?;
    let engine = BrainEngine::open(root, BrainConfig::default())?;
    let receipt = engine.commit_finalized_learning_session(&input)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_arguments;

    #[test]
    fn invalid_cli_is_rejected_before_finalization() {
        assert!(parse_arguments(&[]).is_err());
        assert!(parse_arguments(&["session".into()]).is_err());
        assert!(
            parse_arguments(&["session".into(), "receipt.json".into(), "unexpected".into(),])
                .is_err()
        );
        assert!(parse_arguments(&["   ".into(), "receipt.json".into()]).is_err());
        assert!(parse_arguments(&["session".into(), "   ".into()]).is_err());
    }
}
