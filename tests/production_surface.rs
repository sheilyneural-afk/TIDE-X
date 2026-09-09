use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

fn write_minimal_receiver_checkpoint(path: &Path) {
    let tensors = [
        (
            "model.layers.0.self_attn.q_proj.weight",
            [1.0_f32, 2.0, 3.0, 4.0],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            [-1.0_f32, -2.0, -3.0, -4.0],
        ),
    ];
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_string(),
        serde_json::json!({"format":"pt"}),
    );
    for (name, values) in tensors {
        let start = data.len();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype":"F32",
                "shape":[2,2],
                "data_offsets":[start,data.len()]
            }),
        );
    }
    let mut header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let padding = (8 - header.len() % 8) % 8;
    header.extend(std::iter::repeat_n(b' ', padding));
    let mut output = std::fs::File::create(path).unwrap();
    output
        .write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    output.write_all(&header).unwrap();
    output.write_all(&data).unwrap();
    output.sync_all().unwrap();
}

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
fn tidex_invalid_guarded_routes_fail_before_opening_private_state() {
    let executable = env!("CARGO_BIN_EXE_tidex");
    let routes: &[&[&str]] = &[
        &["adapter-bank"],
        &["adapter-bank", "import"],
        &["adapter-bank", "unknown", "/tmp/input.json"],
        &["adapter-bank", "status", "extra"],
        &["adapter-bank", "materialize"],
        &["adapter-bank", "verify-materialization"],
        &["adapter-bank", "verify-resolution"],
        &["receiver", "profile"],
        &["receiver", "verify-profile"],
        &["receiver", "verify-live-profile"],
        &["receiver", "project-physical-profile"],
        &["materialize", "checkpoint"],
        &["materialize", "verify-checkpoint"],
        &["compile", "universal"],
    ];
    for route in routes {
        let output = Command::new(executable)
            .args(*route)
            .env_remove("TIDEX_PRIVATE_ROOT")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2), "route={route:?}");
        assert!(output.stdout.is_empty(), "route={route:?}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            stderr.contains("usage:"),
            "route={route:?}, stderr={stderr}"
        );
        for command in [
            "receiver verify-profile",
            "receiver verify-live-profile",
            "adapter-bank materialize",
            "adapter-bank verify-materialization",
            "adapter-bank verify-resolution",
        ] {
            assert!(
                stderr.contains(command),
                "command={command}, stderr={stderr}"
            );
        }
        assert!(!stderr.contains("private_root"), "route={route:?}");
    }
}

#[test]
fn tidex_adapter_bank_status_is_valid_on_an_empty_private_authority() {
    let executable = env!("CARGO_BIN_EXE_tidex");
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("test-tidex-empty-adapter-bank-{unique}"));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();

    let output = Command::new(executable)
        .args(["adapter-bank", "status"])
        .env("TIDEX_PRIVATE_ROOT", &root)
        .env_remove("TIDEX_HOME")
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let status: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(status["revision"], 0);
    assert_eq!(status["verified_revision_count"], 0);
    assert_eq!(status["registered_adapter_count"], 0);
    assert_eq!(status["active_adapter_count"], 0);
    assert_eq!(status["revoked_adapter_count"], 0);

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn tidex_profiles_and_reauthenticates_a_physical_receiver_through_the_cli() {
    let executable = env!("CARGO_BIN_EXE_tidex");
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("test-tidex-profile-cli-{unique}"));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
    let checkpoint = root.join("model.safetensors");
    let config = root.join("config.json");
    let tokenizer = root.join("tokenizer.json");
    let request = root.join("profile-request.json");
    let reference = root.join("profile-reference.json");
    write_minimal_receiver_checkpoint(&checkpoint);
    std::fs::write(
        &config,
        serde_json::to_vec(&serde_json::json!({
            "model_type":"tidex-test",
            "architectures":["TidexForCausalLM"],
            "num_hidden_layers":1,
            "hidden_size":2,
            "vocab_size":8,
            "max_position_embeddings":128,
            "tie_word_embeddings":false
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &tokenizer,
        serde_json::to_vec(&serde_json::json!({
            "version":"1.0",
            "model":{"type":"test"}
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &request,
        serde_json::to_vec(&serde_json::json!({
            "schema":"cerebro.tidex.receiver_model_profile_input/v1",
            "model_id":"receiver.cli-test",
            "architecture_id":"tidex.cli-test",
            "checkpoint_path":checkpoint,
            "config_path":config,
            "tokenizer_path":tokenizer
        }))
        .unwrap(),
    )
    .unwrap();

    let profiled = Command::new(executable)
        .args(["receiver", "profile"])
        .arg(&request)
        .env("TIDEX_PRIVATE_ROOT", &root)
        .output()
        .unwrap();
    assert_eq!(
        profiled.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&profiled.stderr)
    );
    let receipt: serde_json::Value = serde_json::from_slice(&profiled.stdout).unwrap();
    assert_eq!(
        receipt["schema"],
        "cerebro.tidex.receiver_model_profile_receipt/v1"
    );
    std::fs::write(
        &reference,
        serde_json::to_vec(&receipt["profile_reference"]).unwrap(),
    )
    .unwrap();

    for command in ["verify-profile", "verify-live-profile"] {
        let verified = Command::new(executable)
            .args(["receiver", command])
            .arg(&reference)
            .env("TIDEX_PRIVATE_ROOT", &root)
            .output()
            .unwrap();
        assert_eq!(
            verified.status.code(),
            Some(0),
            "command={command}, stderr={}",
            String::from_utf8_lossy(&verified.stderr)
        );
        let profile: serde_json::Value = serde_json::from_slice(&verified.stdout).unwrap();
        assert_eq!(profile["model_id"], "receiver.cli-test");
    }

    let projection_request = root.join("projection-request.json");
    std::fs::write(
        &projection_request,
        serde_json::to_vec(&serde_json::json!({
            "schema":"cerebro.tidex.physical_receiver_projection_input/v1",
            "physical_profile":receipt["profile_reference"],
            "tensor_ids":["model.layers.0.self_attn.q_proj.weight"],
            "modalities":["text"]
        }))
        .unwrap(),
    )
    .unwrap();
    let projected = Command::new(executable)
        .args(["receiver", "project-physical-profile"])
        .arg(&projection_request)
        .env("TIDEX_PRIVATE_ROOT", &root)
        .output()
        .unwrap();
    assert!(
        projected.status.success(),
        "{}",
        String::from_utf8_lossy(&projected.stderr)
    );
    let projection: serde_json::Value = serde_json::from_slice(&projected.stdout).unwrap();
    assert_eq!(projection["profile"]["parameter_dimension"], 4);
    assert_eq!(
        projection["snapshot"]["model_snapshot_sha256"],
        receipt["profile"]["checkpoint"]["sha256"]
    );

    let invalid = root.join("invalid-compiled.json");
    std::fs::write(&invalid, b"{}").unwrap();
    for command in ["checkpoint", "verify-checkpoint"] {
        let rejected = Command::new(executable)
            .args(["materialize", command])
            .arg(&invalid)
            .env("TIDEX_PRIVATE_ROOT", &root)
            .output()
            .unwrap();
        assert_eq!(rejected.status.code(), Some(2));
        assert!(!String::from_utf8_lossy(&rejected.stderr).contains("usage:"));
        assert!(rejected.stdout.is_empty());
    }

    std::fs::write(&config, b"{}").unwrap();
    let static_check = Command::new(executable)
        .args(["receiver", "verify-profile"])
        .arg(&reference)
        .env("TIDEX_PRIVATE_ROOT", &root)
        .output()
        .unwrap();
    assert_eq!(static_check.status.code(), Some(0));
    let live_check = Command::new(executable)
        .args(["receiver", "verify-live-profile"])
        .arg(&reference)
        .env("TIDEX_PRIVATE_ROOT", &root)
        .output()
        .unwrap();
    assert_eq!(live_check.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&live_check.stderr)
        .contains("receiver_model_profile_source_changed"));

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn tidex_capabilities_identify_receiver_profile_and_adapter_bank_engines() {
    let executable = env!("CARGO_BIN_EXE_tidex");
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base = std::env::temp_dir().join(format!("test-tidex-bank-capabilities-{unique}"));
    let home = base.join("home");
    let target = base.join("target");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(&target).unwrap();
    std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o700)).unwrap();

    let create = Command::new(executable)
        .args(["workspace", "create", "bank-test", "--target"])
        .arg(&target)
        .env("TIDEX_HOME", &home)
        .output()
        .unwrap();
    assert_eq!(
        create.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&create.stderr)
    );
    let select = Command::new(executable)
        .args(["workspace", "use", "bank-test"])
        .env("TIDEX_HOME", &home)
        .output()
        .unwrap();
    assert_eq!(
        select.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&select.stderr)
    );

    let output = Command::new(executable)
        .arg("capabilities")
        .env("TIDEX_HOME", &home)
        .env_remove("TIDEX_PRIVATE_ROOT")
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["schema"], "cerebro.tidex.capabilities/v1");
    let capabilities = report["capabilities"].as_array().unwrap();
    let ids = capabilities
        .iter()
        .map(|c| c["id"].as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        ids.len(),
        capabilities.len(),
        "capability authorities must not be duplicated"
    );
    for id in [
        "compile.universal_capability",
        "materialize.compiled_checkpoint",
        "adapter_bank.lifecycle",
    ] {
        assert!(ids.contains(id), "missing converged capability {id}");
    }
    let by_id = |id: &str| {
        capabilities
            .iter()
            .find(|capability| capability["id"] == id)
            .unwrap_or_else(|| panic!("missing capability {id}"))
    };
    assert_eq!(
        by_id("receiver.profile.physical")["engine"],
        "model_adaptation::profile_receiver_model"
    );
    assert_eq!(
        by_id("adapter_bank.index.dynamic")["engine"],
        "adapter_bank::AdapterBank::query"
    );
    assert_eq!(
        by_id("adapter_bank.lifecycle")["engine"],
        "adapter_bank::AdapterBank::{authorize_governed_promotion_request,activate,revoke,rollback}"
    );

    std::fs::remove_dir_all(base).unwrap();
}

#[test]
fn tidex_preserves_benchmark_and_generic_json_size_errors() {
    let executable = env!("CARGO_BIN_EXE_tidex");
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let input = std::env::temp_dir().join(format!("test-tidex-oversized-json-{unique}.json"));
    std::fs::File::create(&input)
        .unwrap()
        .set_len(64 * 1024 * 1024 + 1)
        .unwrap();

    let benchmark = Command::new(executable)
        .args(["benchmark", "portability"])
        .arg(&input)
        .output()
        .unwrap();
    assert_eq!(benchmark.status.code(), Some(2));
    assert_eq!(
        String::from_utf8(benchmark.stderr).unwrap().trim(),
        "tidex_benchmark_input_too_large"
    );

    let generic = Command::new(executable)
        .args(["receiver", "describe-readout"])
        .arg(&input)
        .output()
        .unwrap();
    assert_eq!(generic.status.code(), Some(2));
    assert_eq!(
        String::from_utf8(generic.stderr).unwrap().trim(),
        "tidex_cli_json_input_too_large"
    );

    std::fs::remove_file(input).unwrap();
}

#[test]
fn tidex_operator_acquires_the_selected_external_workspace_target() {
    let executable = env!("CARGO_BIN_EXE_tidex");
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base = std::env::temp_dir().join(format!("test-tidex-operator-{unique}"));
    let home = base.join("home");
    let target = base.join("external-project");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(target.join("src")).unwrap();
    std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::write(
        target.join("src/lib.rs"),
        b"pub fn external_capability() {}\n",
    )
    .unwrap();

    let create = Command::new(executable)
        .args(["workspace", "create", "external", "--target"])
        .arg(&target)
        .env("TIDEX_HOME", &home)
        .output()
        .unwrap();
    assert_eq!(
        create.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&create.stderr)
    );
    let selected = Command::new(executable)
        .args(["workspace", "use", "external"])
        .env("TIDEX_HOME", &home)
        .output()
        .unwrap();
    assert_eq!(selected.status.code(), Some(0));

    let acquired = Command::new(executable)
        .arg("acquire")
        .env("TIDEX_HOME", &home)
        .env_remove("TIDEX_PRIVATE_ROOT")
        .output()
        .unwrap();
    assert_eq!(
        acquired.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&acquired.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&acquired.stdout).unwrap();
    assert_eq!(value["schema"], "cerebro.tidex.workspace_acquisition/v1");
    assert_eq!(value["workspace"], "external");
    assert!(value["entries"].as_u64().unwrap() >= 2);
    assert!(home
        .join("workspaces/external/state/state/acquisitions/capture-receipts/by-sha")
        .is_dir());
    assert!(!target.join("state").exists());

    let scoped = Command::new(executable)
        .args(["acquire", "--path", "src"])
        .env("TIDEX_HOME", &home)
        .output()
        .unwrap();
    assert_eq!(
        scoped.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&scoped.stderr)
    );
    let scoped_value: serde_json::Value = serde_json::from_slice(&scoped.stdout).unwrap();
    assert_eq!(scoped_value["completeness"], "declared_scope_only");

    std::fs::remove_dir_all(base).unwrap();
}

#[test]
fn tidex_portability_benchmark_is_workspace_independent_and_leave_one_out() {
    let executable = env!("CARGO_BIN_EXE_tidex");
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let input = std::env::temp_dir().join(format!("tidex-portability-{unique}.json"));
    let functional = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
        [-1.0, 2.0],
        [0.5, 2.0],
    ];
    let cases = functional
        .iter()
        .enumerate()
        .map(|(index, values)| {
            serde_json::json!({
                "skill_id":format!("skill-{index}"),
                "functional_signature":values,
                "direct_receiver_solution":[
                    2.0 * values[0] + values[1] + 0.1,
                    -values[0] + 3.0 * values[1] - 0.2
                ]
            })
        })
        .collect::<Vec<_>>();
    std::fs::write(
        &input,
        serde_json::to_vec(&serde_json::json!({
            "schema":"cerebro.tidex.receiver_portability_benchmark_input/v1",
            "ridge":1e-9,
            "cases":cases
        }))
        .unwrap(),
    )
    .unwrap();
    let output = Command::new(executable)
        .args(["benchmark", "portability"])
        .arg(&input)
        .env_remove("TIDEX_HOME")
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        report["schema"],
        "cerebro.tidex.receiver_portability_benchmark/v1"
    );
    assert_eq!(report["case_count"], 6);
    assert_eq!(report["all_resolved"], true);
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

    let output = Command::new(executable)
        .env("TIDEX_PRIVATE_ROOT", &temp_root)
        .output()
        .expect("execute ledger_diagnose");

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
fn adaptive_learning_cycle_cli_routes_fail_closed_on_missing_authority_inputs() {
    let executable = env!("CARGO_BIN_EXE_adaptive-learning-cycle");
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("test-adaptive-cli-{unique}"));
    std::fs::create_dir_all(&root).unwrap();
    cerebro_tidex::security::secure_dir(&root).unwrap();

    let no_args = Command::new(executable).output().unwrap();
    assert_eq!(no_args.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&no_args.stderr).contains("usage:"));

    let routes: &[&[&str]] = &[
        &[
            "start",
            "missing-session",
            "/tmp/tidex-missing-target.json",
            "/tmp/tidex-missing-policy.json",
        ],
        &["next", "missing-session"],
        &[
            "assimilate",
            "missing-session",
            "/tmp/tidex-missing-evidence.json",
        ],
        &["show", "missing-session"],
        &[
            "controller-train",
            "/tmp/tidex-missing-dataset.json",
            "/tmp/tidex-missing-controller-policy.json",
            "/tmp/tidex-missing-binding.json",
        ],
        &["controller-show", "missing-session"],
        &["controller-compose", "/tmp/tidex-missing-invocation.json"],
        &["unknown-command"],
    ];
    for route in routes {
        let output = Command::new(executable)
            .args(*route)
            .env("TIDEX_PRIVATE_ROOT", &root)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2), "route={route:?}");
    }
    std::fs::remove_dir_all(root).unwrap();
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

    // 4. Correct arity reaches the authority-bound finalization path and fails
    // closed because the receipt does not exist under the isolated child root.
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("test-finalize-cli-{unique}"));
    std::fs::create_dir_all(&root).unwrap();
    cerebro_tidex::security::secure_dir(&root).unwrap();
    let missing = root.join("missing-receipt.json");
    let output = Command::new(executable)
        .args(["session-1", missing.to_str().unwrap()])
        .env("TIDEX_PRIVATE_ROOT", &root)
        .output()
        .expect("execute tidex_finalize valid arity");
    assert_eq!(output.status.code(), Some(2));
    std::fs::remove_dir_all(root).unwrap();
}
