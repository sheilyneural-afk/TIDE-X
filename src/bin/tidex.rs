use cerebro_tidex::acquisition_contract::{
    AcquisitionBudget, AcquisitionRequest, AcquisitionScope, DeclaredRelativePath, NoisePolicy,
    RequestedResidency,
};
use cerebro_tidex::content_vault::capture_to_vault;
use cerebro_tidex::identity::AcquisitionId;
use cerebro_tidex::receiver_compiler::{
    benchmark_receiver_portability_leave_one_out, ReceiverPortabilityBenchmarkInput,
};
use cerebro_tidex::workspace::{
    add_model, configured_tidex_home, create_workspace, current_workspace, load_model, use_model,
    use_workspace, ModelProfile, ModelProvider,
};
use serde_json::json;
use std::fs;
use std::io::Read;
use std::path::Path;

const MAX_BENCHMARK_INPUT_BYTES: u64 = 64 * 1024 * 1024;

fn main() {
    if let Err(error) = run(std::env::args().skip(1).collect()) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    match args.as_slice() {
        [area, command, name, flag, target]
            if area == "workspace" && command == "create" && flag == "--target" =>
        {
            let home = configured_tidex_home()?;
            let manifest = create_workspace(&home, name, Path::new(target))?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        [area, command, name] if area == "workspace" && command == "use" => {
            let home = configured_tidex_home()?;
            use_workspace(&home, name)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&current_workspace(&home)?)?
            );
        }
        [area, command] if area == "workspace" && command == "show" => {
            let home = configured_tidex_home()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&current_workspace(&home)?)?
            );
        }
        [area, command, name, provider_flag, provider, endpoint_flag, endpoint, model_flag, model]
            if area == "model"
                && command == "add"
                && provider_flag == "--provider"
                && endpoint_flag == "--url"
                && model_flag == "--model" =>
        {
            let home = configured_tidex_home()?;
            let provider = match provider.as_str() {
                "openai-compatible" => ModelProvider::OpenAiCompatible,
                _ => return Err("model_provider_invalid".into()),
            };
            add_model(
                &home,
                ModelProfile {
                    schema: "cerebro.tidex.model_profile/v1".into(),
                    name: name.clone(),
                    provider,
                    endpoint: endpoint.clone(),
                    model: model.clone(),
                },
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&load_model(&home, name)?)?
            );
        }
        [area, command, name] if area == "model" && command == "use" => {
            let home = configured_tidex_home()?;
            use_model(&home, name)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&load_model(&home, name)?)?
            );
        }
        [command] if command == "acquire" => {
            let home = configured_tidex_home()?;
            acquire_workspace(&home, None)?
        }
        [command, flag, path] if command == "acquire" && flag == "--path" => {
            let home = configured_tidex_home()?;
            acquire_workspace(&home, Some(Path::new(path)))?
        }
        [area, command, path] if area == "benchmark" && command == "portability" => {
            let input: ReceiverPortabilityBenchmarkInput = read_json_bounded(Path::new(path))?;
            let report = benchmark_receiver_portability_leave_one_out(&input)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        [command] if command == "capabilities" => {
            let home = configured_tidex_home()?;
            let workspace = current_workspace(&home)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "schema":"cerebro.tidex.capabilities/v1",
                    "workspace":workspace.name,
                    "target":workspace.target,
                    "capabilities":[
                        {"id":"acquisition.capture","status":"implemented","engine":"content_vault::capture_to_vault"},
                        {"id":"analysis.skill_fields","status":"implemented","engine":"BrainEngine::analyze"},
                        {"id":"learning.adaptive","status":"implemented","engine":"learning_orchestrator"},
                        {"id":"controller.learned","status":"implemented","engine":"learned_controller"},
                        {"id":"transport.functional","status":"implemented","engine":"transport::learn_functional_transplant"},
                        {"id":"transport.relational","status":"implemented","engine":"transport::learn_relational_transport"},
                        {"id":"compile.skill_fields","status":"implemented","engine":"parametric_program::compile_operator_to_fields"},
                        {"id":"compile.receiver","status":"implemented_experimental","engine":"receiver_compiler::compile_receiver_capability","reason":"held-out receiver compilation, reverse functional verification, wrong-skill separation, protection and trust-region gates are implemented; LLM-scale evidence is not yet established"},
                        {"id":"benchmark.portability","status":"implemented","engine":"receiver_compiler::benchmark_receiver_portability_leave_one_out"},
                        {"id":"runtime.sleep","status":"implemented","engine":"BrainEngine::sleep_cycle"},
                        {"id":"capability_ir.v63.contract","status":"implemented_foundation","engine":"capability_ir::OperationalCapabilityContract","reason":"StateIR anchors, repeated OperatorIR transitions, canonical transition signatures, closure and contraction verification exist; universal receiver compilation evidence remains pending"},
                        {"id":"model.assistance","status":"configured_not_authoritative","reason":"model profiles are selectable; no model call is permitted to create evidence or promotion authority"}
                    ]
                }))?
            );
        }
        _ => return Err(usage().into()),
    }
    Ok(())
}

fn read_json_bounded<T: serde::de::DeserializeOwned>(
    path: &Path,
) -> Result<T, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(MAX_BENCHMARK_INPUT_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > MAX_BENCHMARK_INPUT_BYTES {
        return Err("tidex_benchmark_input_too_large".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn acquire_workspace(
    home: &Path,
    selected: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let workspace = current_workspace(home)?;
    let private_root = workspace.private_root(home);
    let scope = match selected {
        None => AcquisitionScope::WholeProject,
        Some(path) => AcquisitionScope::DeclaredPaths {
            roots: vec![DeclaredRelativePath::parse(path.to_path_buf())?],
        },
    };
    let request = AcquisitionRequest::new(
        AcquisitionId::parse(format!("workspace-{}-capture", workspace.name))?,
        scope,
        RequestedResidency::BestVerified,
        NoisePolicy::ConservativeGeneratedArtifacts,
        AcquisitionBudget {
            max_files: 100_000,
            max_total_bytes: 8 * 1024 * 1024 * 1024,
        },
        vec![],
    )?;
    let receipt = capture_to_vault(&workspace.target, &private_root, &request)?;
    let reference = receipt.persist(&private_root)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.workspace_acquisition/v1",
            "workspace":workspace.name,
            "target":workspace.target,
            "capture_receipt_sha256":receipt.manifest_sha256(),
            "capture_receipt":reference,
            "system_envelope_sha256":receipt.envelope().manifest_sha256(),
            "completeness":receipt.envelope().completeness(),
            "entries":receipt.envelope().entries().len(),
            "bytes":receipt.total_file_bytes()
        }))?
    );
    Ok(())
}

fn usage() -> &'static str {
    "usage:\n  tidex workspace create <name> --target <absolute-path>\n  tidex workspace use <name>\n  tidex workspace show\n  tidex model add <name> --provider openai-compatible --url <endpoint> --model <model>\n  tidex model use <name>\n  tidex acquire [--path <relative-project-path>]\n  tidex benchmark portability <input.json>\n  tidex capabilities"
}
