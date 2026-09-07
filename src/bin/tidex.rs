use cerebro_tidex::workspace::{
    add_model, configured_tidex_home, create_workspace, current_workspace, load_model, use_model,
    use_workspace, ModelProfile, ModelProvider,
};
use serde_json::json;
use std::path::Path;

fn main() {
    if let Err(error) = run(std::env::args().skip(1).collect()) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let home = configured_tidex_home()?;
    match args.as_slice() {
        [area, command, name, flag, target] if area == "workspace" && command == "create" && flag == "--target" => {
            let manifest = create_workspace(&home, name, Path::new(target))?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        [area, command, name] if area == "workspace" && command == "use" => {
            use_workspace(&home, name)?;
            println!("{}", serde_json::to_string_pretty(&current_workspace(&home)?)?);
        }
        [area, command] if area == "workspace" && command == "show" => {
            println!("{}", serde_json::to_string_pretty(&current_workspace(&home)?)?);
        }
        [area, command, name, provider_flag, provider, endpoint_flag, endpoint, model_flag, model]
            if area == "model" && command == "add" && provider_flag == "--provider" && endpoint_flag == "--url" && model_flag == "--model" => {
            let provider = match provider.as_str() {
                "openai-compatible" => ModelProvider::OpenAiCompatible,
                _ => return Err("model_provider_invalid".into()),
            };
            add_model(&home, ModelProfile {
                schema: "cerebro.tidex.model_profile/v1".into(),
                name: name.clone(), provider, endpoint: endpoint.clone(), model: model.clone(),
            })?;
            println!("{}", serde_json::to_string_pretty(&load_model(&home, name)?)?);
        }
        [area, command, name] if area == "model" && command == "use" => {
            use_model(&home, name)?;
            println!("{}", serde_json::to_string_pretty(&load_model(&home, name)?)?);
        }
        [command] if command == "capabilities" => {
            let workspace = current_workspace(&home)?;
            println!("{}", serde_json::to_string_pretty(&json!({
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
                    {"id":"runtime.sleep","status":"implemented","engine":"BrainEngine::sleep_cycle"},
                    {"id":"capability_ir.v63","status":"not_implemented","reason":"StateIR/OperatorIR/interface closure contracts are not present in the current CapabilityIr"},
                    {"id":"model.assistance","status":"configured_not_authoritative","reason":"model profiles are selectable; no model call is permitted to create evidence or promotion authority"}
                ]
            }))?);
        }
        _ => return Err(usage().into()),
    }
    Ok(())
}

fn usage() -> &'static str {
    "usage:\n  tidex workspace create <name> --target <absolute-path>\n  tidex workspace use <name>\n  tidex workspace show\n  tidex model add <name> --provider openai-compatible --url <endpoint> --model <model>\n  tidex model use <name>\n  tidex capabilities"
}
