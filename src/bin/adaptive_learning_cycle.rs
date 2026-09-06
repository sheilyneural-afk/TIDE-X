use cerebro_tidex::authority::read_existing_private_file_bounded;
use cerebro_tidex::contracts::BrainConfig;
use cerebro_tidex::engine::{BrainEngine, ControllerInvocation, RecordedControllerExecution};
use cerebro_tidex::learned_controller::{
    load_persisted_runtime_learned_controller, train_persisted_runtime_learned_controller,
    LearnedControllerBinding, LearnedControllerPolicy,
};
use cerebro_tidex::learning_orchestrator::{
    assimilate_persistent_learning_evidence, issue_next_persistent_learning_aperture,
    load_persistent_adaptive_learning_receipt, start_persistent_adaptive_learning,
    AdaptiveLearningPolicy, LearningTarget,
};
use cerebro_tidex::security::configured_private_root;
use serde_json::json;
use std::error::Error;
use std::fs;
use std::io::Read;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

const MAX_CLI_JSON_BYTES: u64 = 64 * 1024 * 1024;

fn read_confined_invocation(root: &Path, raw_path: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    Ok(read_existing_private_file_bounded(
        root,
        Path::new(raw_path),
        MAX_CLI_JSON_BYTES,
    )?)
}

fn controller_compose(
    invocation_path: &str,
) -> Result<RecordedControllerExecution, Box<dyn Error>> {
    let root = configured_private_root()?;
    let invocation_raw = read_confined_invocation(&root, invocation_path)?;
    let invocation: ControllerInvocation = serde_json::from_slice(&invocation_raw)?;
    let engine = BrainEngine::open(&root, BrainConfig::default())?;
    Ok(engine.compose_current_learned_controller_and_record(&invocation)?)
}

fn usage() -> &'static str {
    "usage:\n  adaptive_learning_cycle start <session-id> <learning-target.json> <policy.json>\n  adaptive_learning_cycle next <session-id>\n  adaptive_learning_cycle assimilate <session-id> <experiment-evidence.json>\n  adaptive_learning_cycle show <session-id>\n  adaptive_learning_cycle controller-train <controller-training-dataset.json> <controller-policy.json> <controller-binding.json>\n  adaptive_learning_cycle controller-show <session-id>\n  adaptive_learning_cycle controller-compose <invocation.json>"
}

fn print_receipt(
    loaded: cerebro_tidex::learning_orchestrator::LoadedAdaptiveLearningReceipt,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.adaptive_learning_cli_output/v1",
            "receipt_sha256":loaded.receipt_sha256,
            "receipt":loaded.receipt,
        }))?
    );
    Ok(())
}

fn print_controller_receipt(
    loaded: cerebro_tidex::learned_controller::LoadedLearnedControllerReceipt,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.learned_controller_cli_output/v1",
            "receipt_sha256":loaded.receipt_sha256,
            "receipt":loaded.receipt,
        }))?
    );
    Ok(())
}

fn print_controller_execution(
    execution: RecordedControllerExecution,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.controller_execution_cli_output/v1",
            "receipt_path":execution.receipt_path,
            "receipt_sha256":execution.receipt_sha256,
            "ledger_event_hash":execution.ledger_event_hash,
            "receipt":execution.receipt,
        }))?
    );
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T, Box<dyn std::error::Error>> {
    let file = fs::File::open(Path::new(path))?;
    let mut bytes = Vec::new();
    file.take(MAX_CLI_JSON_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > MAX_CLI_JSON_BYTES {
        return Err("adaptive_learning_cli_input_too_large".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let command = arguments.first().map(String::as_str).ok_or_else(usage)?;
    let root = configured_private_root()?;
    match command {
        "start" if arguments.len() == 4 => {
            let target: LearningTarget = read_json(&arguments[2])?;
            let policy: AdaptiveLearningPolicy = read_json(&arguments[3])?;
            print_receipt(start_persistent_adaptive_learning(
                &root,
                &arguments[1],
                &target,
                &policy,
            )?)
        }
        "next" if arguments.len() == 2 => print_receipt(issue_next_persistent_learning_aperture(
            &root,
            &arguments[1],
        )?),
        "assimilate" if arguments.len() == 3 => print_receipt(
            assimilate_persistent_learning_evidence(&root, &arguments[1], &arguments[2])?,
        ),
        "show" if arguments.len() == 2 => print_receipt(load_persistent_adaptive_learning_receipt(
            &root,
            &arguments[1],
        )?),
        "controller-train" if arguments.len() == 4 => {
            let policy: LearnedControllerPolicy = read_json(&arguments[2])?;
            let binding: LearnedControllerBinding = read_json(&arguments[3])?;
            print_controller_receipt(train_persisted_runtime_learned_controller(
                &root,
                &arguments[1],
                &policy,
                &binding,
            )?)
        }
        "controller-show" if arguments.len() == 2 => print_controller_receipt(
            load_persisted_runtime_learned_controller(&root, &arguments[1])?,
        ),
        "controller-compose" if arguments.len() == 2 => {
            print_controller_execution(controller_compose(&arguments[1])?)
        }
        _ => Err(usage().into()),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerebro_tidex::security::secure_dir;
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cerebro-controller-execution-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        secure_dir(&root).unwrap();
        root
    }

    #[test]
    fn invocation_contract_rejects_free_vectors_and_invalid_state() {
        let free_observation = json!({
            "schema":"cerebro.tidex.controller_invocation/v1",
            "session_id":"session-a",
            "state_before":[1.0],
            "observation":[0.5],
            "promoted_observation_semantic_sha256":"a".repeat(64)
        });
        assert!(serde_json::from_value::<ControllerInvocation>(free_observation).is_err());
        let free_coefficients = json!({
            "schema":"cerebro.tidex.controller_invocation/v1",
            "session_id":"session-a",
            "state_before":[1.0],
            "promoted_observation_semantic_sha256":"a".repeat(64),
            "coefficients":[42.0]
        });
        assert!(serde_json::from_value::<ControllerInvocation>(free_coefficients).is_err());

        let invalid = ControllerInvocation {
            schema: "cerebro.tidex.controller_invocation/v1".into(),
            session_id: cerebro_tidex::identity::SessionId::parse("session-a").unwrap(),
            state_before: Vec::new(),
            promoted_observation_semantic_sha256: cerebro_tidex::digest::Sha256Digest::parse(
                "a".repeat(64),
            )
            .unwrap(),
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn invocation_deserialization_rejects_invalid_semantic_identities() {
        let invalid_session = json!({
            "schema":"cerebro.tidex.controller_invocation/v1",
            "session_id":"../escape",
            "state_before":[1.0],
            "promoted_observation_semantic_sha256":"a".repeat(64)
        });
        assert!(serde_json::from_value::<ControllerInvocation>(invalid_session).is_err());

        let invalid_digest = json!({
            "schema":"cerebro.tidex.controller_invocation/v1",
            "session_id":"session-a",
            "state_before":[1.0],
            "promoted_observation_semantic_sha256":"not-a-digest"
        });
        assert!(serde_json::from_value::<ControllerInvocation>(invalid_digest).is_err());
    }

    #[test]
    fn invocation_path_rejects_escape_and_symlink() {
        let root = temporary_root("path");
        let inside = root.join("invocation.json");
        fs::write(&inside, b"{}").unwrap();
        assert!(read_confined_invocation(&root, inside.to_str().unwrap()).is_ok());

        let outside = std::env::temp_dir().join(format!(
            "cerebro-controller-execution-outside-{}",
            std::process::id()
        ));
        fs::write(&outside, b"{}").unwrap();
        assert!(read_confined_invocation(&root, outside.to_str().unwrap()).is_err());
        let linked = root.join("linked.json");
        symlink(&outside, &linked).unwrap();
        assert!(read_confined_invocation(&root, linked.to_str().unwrap()).is_err());

        let real_dir = root.join("real");
        fs::create_dir(&real_dir).unwrap();
        let nested = real_dir.join("nested.json");
        fs::write(&nested, b"{}").unwrap();
        let linked_dir = root.join("linked-dir");
        symlink(&real_dir, &linked_dir).unwrap();
        let through_intermediate_link = linked_dir.join("nested.json");
        assert!(
            read_confined_invocation(&root, through_intermediate_link.to_str().unwrap(),).is_err()
        );

        fs::remove_file(&outside).unwrap();
        fs::remove_dir_all(&root).unwrap();
    }
}
