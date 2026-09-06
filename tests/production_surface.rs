use std::process::Command;

#[test]
fn primary_cli_rejects_unguarded_mutation_commands_before_opening_state() {
    let executable = env!("CARGO_BIN_EXE_cerebro-tidex");
    for (command, expected) in [
        (
            "commit",
            "retired command:commit; use the receipt-bound TIDE-X learning finalizer",
        ),
        (
            "artifact-import-f32",
            "retired command:artifact-import-f32; arbitrary raw delta ingestion is not a governed route",
        ),
        (
            "artifact-ties",
            "retired command:artifact-ties; arbitrary parameter merging is not a governed route",
        ),
    ] {
        let output = Command::new(executable)
            .arg(command)
            .output()
            .expect("retired command should execute the primary binary");
        assert_eq!(output.status.code(), Some(2), "command={command}");
        assert_eq!(String::from_utf8(output.stdout).unwrap(), "");
        assert_eq!(
            String::from_utf8(output.stderr).unwrap().trim(),
            expected,
            "command={command}"
        );
    }
}

#[test]
fn autonomous_learning_plan_cli_lifecycle() {
    let executable = env!("CARGO_BIN_EXE_autonomous-learning-plan");

    // 1. Missing arguments -> usage error
    let output = Command::new(executable)
        .output()
        .expect("execute autonomous_learning_plan");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("usage: autonomous_learning_plan"));

    // 2. Non-existent file -> I/O error
    let output = Command::new(executable)
        .arg("/tmp/nonexistent-learning-target.json")
        .output()
        .expect("execute autonomous_learning_plan");
    assert_eq!(output.status.code(), Some(2));

    // 3. Valid learning target -> successfully generates plan
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_target = std::env::temp_dir().join(format!("test-target-{unique}.json"));
    let target_json = serde_json::json!({
        "target_id": "target-cli-test",
        "capability_ids": ["a", "b", "c", "d"],
        "candidate_budget": 64,
        "plan_steps": 12,
        "noise_variance": 0.1,
        "cost_weight": 0.0,
        "risk_weight": 0.0
    });
    std::fs::write(
        &temp_target,
        serde_json::to_vec_pretty(&target_json).unwrap(),
    )
    .unwrap();

    let output = Command::new(executable)
        .arg(&temp_target)
        .output()
        .expect("execute autonomous_learning_plan");
    let _ = std::fs::remove_file(&temp_target);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("cerebro.tidex.autonomous_learning_plan/v1"));
}

#[test]
fn ledger_diagnose_cli_lifecycle() {
    let executable = env!("CARGO_BIN_EXE_ledger-diagnose");

    // 1. Without valid TIDEX_PRIVATE_ROOT -> error
    let output = Command::new(executable)
        .env_remove("TIDEX_PRIVATE_ROOT")
        .output()
        .expect("execute ledger_diagnose");
    assert!(!output.status.success());

    // 2. With valid TIDEX_PRIVATE_ROOT containing an initialized engine/ledger
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_root = std::env::temp_dir().join(format!("test-ledger-diag-{unique}"));
    std::fs::create_dir_all(&temp_root).unwrap();
    cerebro_tidex::security::secure_dir(&temp_root).unwrap();

    let previous = std::env::var("TIDEX_PRIVATE_ROOT").ok();
    std::env::set_var("TIDEX_PRIVATE_ROOT", &temp_root);

    // Open an engine to create the ledger structure
    let _engine = cerebro_tidex::engine::BrainEngine::open(
        &temp_root,
        cerebro_tidex::contracts::BrainConfig::default(),
    )
    .unwrap();

    let output = Command::new(executable)
        .env("TIDEX_PRIVATE_ROOT", &temp_root)
        .output()
        .expect("execute ledger_diagnose");

    match previous {
        Some(ref p) => std::env::set_var("TIDEX_PRIVATE_ROOT", p),
        None => std::env::remove_var("TIDEX_PRIVATE_ROOT"),
    }
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("verified_events="));
    assert!(stdout.contains("verified_head="));

    let _ = std::fs::remove_dir_all(&temp_root);
}

#[test]
fn pure_linear_runner_cli_lifecycle() {
    let executable = env!("CARGO_BIN_EXE_pure-linear-runner");

    // 1. Without TIDEX_INPUT_PATH -> exit 1, tidex_input_path_missing
    let output = Command::new(executable)
        .env_remove("TIDEX_INPUT_PATH")
        .output()
        .expect("execute pure_linear_runner");
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("tidex_input_path_missing"));

    // 2. With invalid TIDEX_INPUT_PATH -> exit 1, tidex_input_path_invalid
    let output = Command::new(executable)
        .env("TIDEX_INPUT_PATH", "/tmp/wrong/path")
        .output()
        .expect("execute pure_linear_runner");
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("tidex_input_path_invalid"));
}

#[test]
fn record_representation_evidence_cli_lifecycle() {
    let executable = env!("CARGO_BIN_EXE_record-representation-evidence");

    // 1. No args -> exit 2 with usage
    let output = Command::new(executable)
        .output()
        .expect("execute record_representation_evidence");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("usage: record_representation_evidence"));

    // 2. Multiple args -> exit 2 with usage
    let output = Command::new(executable)
        .args(["arg1", "arg2"])
        .output()
        .expect("execute record_representation_evidence");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("usage: record_representation_evidence"));

    // 3. Non-existent file -> exit 2 with error
    let output = Command::new(executable)
        .arg("/tmp/nonexistent-rep-evidence.json")
        .output()
        .expect("execute record_representation_evidence");
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn tidex_finalize_cli_lifecycle() {
    let executable = env!("CARGO_BIN_EXE_tidex-finalize");

    // 1. No args -> exit 2 with usage
    let output = Command::new(executable)
        .output()
        .expect("execute tidex_finalize");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("usage: tidex_finalize"));

    // 2. Single arg -> exit 2 with usage
    let output = Command::new(executable)
        .arg("session-1")
        .output()
        .expect("execute tidex_finalize");
    assert_eq!(output.status.code(), Some(2));

    // 3. Three args -> exit 2 with usage
    let output = Command::new(executable)
        .args(["session-1", "receipt.json", "extra"])
        .output()
        .expect("execute tidex_finalize");
    assert_eq!(output.status.code(), Some(2));
}
