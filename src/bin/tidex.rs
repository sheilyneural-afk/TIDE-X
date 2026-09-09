use cerebro_tidex::acquisition_contract::{
    AcquisitionBudget, AcquisitionRequest, AcquisitionScope, DeclaredRelativePath, NoisePolicy,
    RequestedResidency,
};
use cerebro_tidex::activation_steering_materializer::{
    materialize_replayed_activation_steering_shadow, ActivationSteeringLayout,
    ActivationSteeringPolicy,
};
use cerebro_tidex::adapter_bank::{
    AdapterActivationRequest, AdapterBank, AdapterBankLookup, AdapterBankQuery,
    AdapterCandidateMaterializationRequest, AdapterCompositionRequest, AdapterExecutionResolution,
    AdapterGovernedPromotionRequest, AdapterImportRequest, AdapterResolutionRequest,
    AdapterRevocationRequest, AdapterRollbackRequest,
};
use cerebro_tidex::authority::PrivateFileReference;
use cerebro_tidex::capability_discovery::CapabilityDiscoveryRequest;
use cerebro_tidex::checkpoint_adapter::{inspect_safetensors_receiver, SafeTensorsReceiverRequest};
use cerebro_tidex::content_vault::capture_to_vault;
use cerebro_tidex::dense_shadow_materializer::materialize_replayed_dense_delta_shadow;
use cerebro_tidex::identity::AcquisitionId;
use cerebro_tidex::isolated_execution::AuthenticatedBytes;
use cerebro_tidex::low_rank_shadow_materializer::{
    materialize_replayed_low_rank_shadow, LowRankShadowPolicy,
};
use cerebro_tidex::materialization_selector::BackendSelectionInput;
use cerebro_tidex::model_adaptation::{
    authenticate_live_receiver_model_profile, authenticate_receiver_model_profile,
    profile_receiver_model, ReceiverModelProfileInput,
};
use cerebro_tidex::receiver_compiler::{
    benchmark_receiver_portability_leave_one_out, ReceiverPortabilityBenchmarkInput,
};
use cerebro_tidex::receiver_layout::ReceiverMaterializationLayout;
use cerebro_tidex::receiver_weight_binding::{
    assemble_distributed_lora_basis, authenticate_receiver_weight_candidate,
    materialize_receiver_weight_candidate, prepare_receiver_weight_candidate,
    DistributedLoraBasisInput,
};
use cerebro_tidex::security::configured_private_root;
use cerebro_tidex::shadow_evaluation::{run_shadow_evaluation, ShadowEvaluationInput};
use cerebro_tidex::sparse_shadow_materializer::{
    materialize_replayed_sparse_shadow, SparseShadowPolicy,
};
use cerebro_tidex::universal_capability_compiler::{
    execute_experimental_universal_capability_request, execute_universal_capability_shadow_plan,
    replay_experimental_universal_capability_request, replay_universal_capability_shadow_plan,
    UniversalCapabilityCompilationReceipt, UniversalCapabilityCompilationRequest,
    UniversalCapabilityPlanningRequest, UniversalCapabilityShadowPlanReceipt,
};
use cerebro_tidex::universal_promotion_gate::{
    evaluate_universal_promotion_gate, UniversalPromotionGateRequest,
};
use cerebro_tidex::universality_evidence::UniversalityEvidenceInput;
use cerebro_tidex::workspace::{
    add_model, configured_tidex_home, create_workspace, current_workspace, load_model, use_model,
    use_workspace, ModelProfile, ModelProvider,
};
use serde_json::json;
use std::fs;
use std::io::Read;
use std::path::Path;

const MAX_CLI_JSON_BYTES: u64 = 64 * 1024 * 1024;

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
        [area, command, path] if area == "receiver" && command == "freeze-compiler" => {
            let input: cerebro_tidex::receiver_compiler::FrozenReceiverCompilerInput =
                read_json_bounded(Path::new(path))?;
            let frozen = cerebro_tidex::receiver_compiler::freeze_receiver_compiler(&input)?;
            println!("{}", serde_json::to_string_pretty(&frozen)?);
        }
        [area, command, path] if area == "receiver" && command == "verify-compiler" => {
            let frozen: cerebro_tidex::receiver_compiler::FrozenReceiverCompiler =
                read_json_bounded(Path::new(path))?;
            frozen.verify()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "schema": "cerebro.tidex.frozen_receiver_compiler_verification/v1",
                    "manifest_sha256": frozen.manifest_sha256(),
                    "calibration_replay_verified": true,
                    "model_execution_verified": false,
                    "authorizes_promotion": false
                }))?
            );
        }
        [area, command, path] if area == "benchmark" && command == "response" => {
            let input: cerebro_tidex::receiver_compiler::ReceiverSignatureBenchmarkInput =
                read_benchmark_json_bounded(Path::new(path))?;
            let report = cerebro_tidex::receiver_compiler::benchmark_receiver_signature(&input)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        [area, command, path] if area == "benchmark" && command == "receiver-basis" => {
            let input: cerebro_tidex::receiver_compiler::ReceiverBasisBenchmarkInput =
                read_benchmark_json_bounded(Path::new(path))?;
            let report = cerebro_tidex::receiver_compiler::benchmark_receiver_basis(&input)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        [area, command, path] if area == "benchmark" && command == "portability" => {
            let input: ReceiverPortabilityBenchmarkInput =
                read_benchmark_json_bounded(Path::new(path))?;
            let report = benchmark_receiver_portability_leave_one_out(&input)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        [area, command, path] if area == "receiver" && command == "describe-readout" => {
            let input: cerebro_tidex::receiver_weight_binding::DescribeLinearReadoutInput =
                read_json_bounded(Path::new(path))?;
            let report = cerebro_tidex::receiver_weight_binding::describe_linear_readout(&input)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        [area, command, path] if area == "receiver" && command == "acquire-readout" => {
            let root = configured_private_root()?;
            let input: cerebro_tidex::receiver_weight_binding::AcquireLinearReadoutInput =
                read_json_bounded(Path::new(path))?;
            let report =
                cerebro_tidex::receiver_weight_binding::acquire_linear_readout(&root, &input)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        [area, command, reference] if area == "receiver" && command == "import-axis" => {
            let root = configured_private_root()?;
            let input: PrivateFileReference = read_json_bounded(Path::new(reference))?;
            let bytes = input.read_verified_bounded(&root, 8 * 1024 * 1024)?;
            let values: Vec<f32> = serde_json::from_slice(&bytes)?;
            let delta = cerebro_tidex::artifact::ArtifactWriteAuthority::open(&root)?
                .create_content_addressed_dvec(&values)?;
            println!("{}", serde_json::to_string_pretty(&delta)?);
        }
        [area, command, path] if area == "receiver" && command == "normalize-sharded" => {
            let root = configured_private_root()?;
            let input: cerebro_tidex::weight_actuator::ShardedSafetensorsNormalizationInput =
                read_json_bounded(Path::new(path))?;
            let receipt =
                cerebro_tidex::weight_actuator::normalize_sharded_safetensors(&root, &input)?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        [area, command, path] if area == "receiver" && command == "import-lora-axis" => {
            let root = configured_private_root()?;
            let input: cerebro_tidex::weight_actuator::LoraAdapterAxisInput =
                read_json_bounded(Path::new(path))?;
            let receipt =
                cerebro_tidex::weight_actuator::import_peft_lora_as_dense_axis(&root, &input)?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        [area, command, path] if area == "receiver" && command == "assemble-lora-basis" => {
            let root = configured_private_root()?;
            let input: DistributedLoraBasisInput = read_json_bounded(Path::new(path))?;
            let basis = assemble_distributed_lora_basis(&root, &input)?;
            println!("{}", serde_json::to_string_pretty(&basis)?);
        }
        [area, command, reference] if area == "receiver" && command == "compile" => {
            let root = configured_private_root()?;
            let input: PrivateFileReference = read_json_bounded(Path::new(reference))?;
            let candidate = prepare_receiver_weight_candidate(&root, &input)?;
            println!("{}", serde_json::to_string_pretty(&candidate)?);
        }
        [area, command, reference] if area == "receiver" && command == "inspect" => {
            let root = configured_private_root()?;
            let input: PrivateFileReference = read_json_bounded(Path::new(reference))?;
            let candidate = authenticate_receiver_weight_candidate(&root, &input)?;
            println!("{}", serde_json::to_string_pretty(&candidate)?);
        }
        [area, command, reference, base_flag, base, output_flag, output]
            if area == "receiver"
                && command == "materialize"
                && base_flag == "--base-model"
                && output_flag == "--output" =>
        {
            let root = configured_private_root()?;
            let input: PrivateFileReference = read_json_bounded(Path::new(reference))?;
            let receipt = materialize_receiver_weight_candidate(
                &root,
                &input,
                Path::new(base),
                Path::new(output),
            )?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        [area, command, path] if area == "receiver" && command == "profile" => {
            let root = configured_private_root()?;
            let input: ReceiverModelProfileInput = read_json_bounded(Path::new(path))?;
            let receipt = profile_receiver_model(&root, &input)?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        [area, command, reference] if area == "receiver" && command == "verify-profile" => {
            let root = configured_private_root()?;
            let input: PrivateFileReference = read_json_bounded(Path::new(reference))?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &authenticate_receiver_model_profile(&root, &input,)?
                )?
            );
        }
        [area, command, reference] if area == "receiver" && command == "verify-live-profile" => {
            let root = configured_private_root()?;
            let input: PrivateFileReference = read_json_bounded(Path::new(reference))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&authenticate_live_receiver_model_profile(
                    &root, &input,
                )?)?
            );
        }
        [area, command, path] if area == "adapter-bank" && command == "import" => {
            let root = configured_private_root()?;
            let bank = AdapterBank::open(&root)?;
            let input: AdapterImportRequest = read_json_bounded(Path::new(path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&bank.import_lora(&input)?)?
            );
        }
        [area, command, path] if area == "adapter-bank" && command == "compose" => {
            let root = configured_private_root()?;
            let bank = AdapterBank::open(&root)?;
            let input: AdapterCompositionRequest = read_json_bounded(Path::new(path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&bank.compose_exact(&input)?)?
            );
        }
        [area, command, path] if area == "adapter-bank" && command == "materialize" => {
            let root = configured_private_root()?;
            let bank = AdapterBank::open(&root)?;
            let input: AdapterCandidateMaterializationRequest = read_json_bounded(Path::new(path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&bank.materialize_candidate(&input)?)?
            );
        }
        [area, command, path] if area == "adapter-bank" && command == "authorize" => {
            let root = configured_private_root()?;
            let bank = AdapterBank::open(&root)?;
            let input: AdapterGovernedPromotionRequest = read_json_bounded(Path::new(path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&bank.authorize_governed_promotion_request(&input)?)?
            );
        }
        [area, command, path] if area == "adapter-bank" && command == "verify-materialization" => {
            let root = configured_private_root()?;
            let bank = AdapterBank::open(&root)?;
            let input: PrivateFileReference = read_json_bounded(Path::new(path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &bank.authenticate_candidate_materialization(&input)?,
                )?
            );
        }
        [area, command, path] if area == "adapter-bank" && command == "query" => {
            let root = configured_private_root()?;
            let bank = AdapterBank::open(&root)?;
            let input: AdapterBankQuery = read_json_bounded(Path::new(path))?;
            println!("{}", serde_json::to_string_pretty(&bank.query(&input)?)?);
        }
        [area, command, path] if area == "adapter-bank" && command == "show" => {
            let root = configured_private_root()?;
            let bank = AdapterBank::open(&root)?;
            let input: AdapterBankLookup = read_json_bounded(Path::new(path))?;
            let manifest = bank
                .lookup(&input)?
                .ok_or("adapter_bank_manifest_not_found")?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        [area, command, path] if area == "adapter-bank" && command == "resolve" => {
            let root = configured_private_root()?;
            let bank = AdapterBank::open(&root)?;
            let input: AdapterResolutionRequest = read_json_bounded(Path::new(path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&bank.resolve_active(&input)?)?
            );
        }
        [area, command, path] if area == "adapter-bank" && command == "verify-resolution" => {
            let root = configured_private_root()?;
            let bank = AdapterBank::open(&root)?;
            let input: AdapterExecutionResolution = read_json_bounded(Path::new(path))?;
            bank.authenticate_execution_resolution(&input)?;
            println!("{}", serde_json::to_string_pretty(&input)?);
        }

        [area, command, path] if area == "adapter-bank" && command == "activate" => {
            let root = configured_private_root()?;
            let bank = AdapterBank::open(&root)?;
            let input: AdapterActivationRequest = read_json_bounded(Path::new(path))?;
            println!("{}", serde_json::to_string_pretty(&bank.activate(&input)?)?);
        }
        [area, command, path] if area == "adapter-bank" && command == "revoke" => {
            let root = configured_private_root()?;
            let bank = AdapterBank::open(&root)?;
            let input: AdapterRevocationRequest = read_json_bounded(Path::new(path))?;
            println!("{}", serde_json::to_string_pretty(&bank.revoke(&input)?)?);
        }
        [area, command, path] if area == "adapter-bank" && command == "rollback" => {
            let root = configured_private_root()?;
            let bank = AdapterBank::open(&root)?;
            let input: AdapterRollbackRequest = read_json_bounded(Path::new(path))?;
            println!("{}", serde_json::to_string_pretty(&bank.rollback(&input)?)?);
        }
        [area, command] if area == "adapter-bank" && command == "status" => {
            let root = configured_private_root()?;
            let bank = AdapterBank::open(&root)?;
            println!("{}", serde_json::to_string_pretty(&bank.verify_history()?)?);
        }
        [area, command, path] if area == "compile" && command == "universal" => {
            let request: UniversalCapabilityCompilationRequest =
                read_json_bounded(Path::new(path))?;
            let report = execute_experimental_universal_capability_request(&request)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        [area, command, request_path, receipt_path]
            if area == "compile" && command == "universal-replay" =>
        {
            let request: UniversalCapabilityCompilationRequest =
                read_json_bounded(Path::new(request_path))?;
            let receipt: UniversalCapabilityCompilationReceipt =
                read_json_bounded(Path::new(receipt_path))?;
            replay_experimental_universal_capability_request(&request, &receipt)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "schema":"cerebro.tidex.universal_capability_compilation_replay/v1",
                    "request_sha256":receipt.request_sha256,
                    "replayed":true
                }))?
            );
        }
        [area, command, path] if area == "compile" && command == "universal-plan" => {
            let request: UniversalCapabilityPlanningRequest = read_json_bounded(Path::new(path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&execute_universal_capability_shadow_plan(&request)?)?
            );
        }
        [area, command, request_path, receipt_path]
            if area == "compile" && command == "universal-plan-replay" =>
        {
            let request: UniversalCapabilityPlanningRequest =
                read_json_bounded(Path::new(request_path))?;
            let receipt: UniversalCapabilityShadowPlanReceipt =
                read_json_bounded(Path::new(receipt_path))?;
            replay_universal_capability_shadow_plan(&request, &receipt)?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &json!({"schema":"cerebro.tidex.universal_shadow_plan_replay/v1","planning_request_sha256":receipt.planning_request_sha256,"replayed":true})
                )?
            );
        }
        [area, backend, request_path, receipt_path, layout_path]
            if area == "materialize" && backend == "dense" =>
        {
            let request: UniversalCapabilityPlanningRequest =
                read_json_bounded(Path::new(request_path))?;
            let receipt: UniversalCapabilityShadowPlanReceipt =
                read_json_bounded(Path::new(receipt_path))?;
            let layout: ReceiverMaterializationLayout = read_json_bounded(Path::new(layout_path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&materialize_replayed_dense_delta_shadow(
                    &request, &receipt, &layout
                )?)?
            );
        }
        [area, backend, request_path, receipt_path, layout_path, policy_path]
            if area == "materialize" && backend == "low-rank" =>
        {
            let request: UniversalCapabilityPlanningRequest =
                read_json_bounded(Path::new(request_path))?;
            let receipt: UniversalCapabilityShadowPlanReceipt =
                read_json_bounded(Path::new(receipt_path))?;
            let layout: ReceiverMaterializationLayout = read_json_bounded(Path::new(layout_path))?;
            let policy: LowRankShadowPolicy = read_json_bounded(Path::new(policy_path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&materialize_replayed_low_rank_shadow(
                    &request, &receipt, &layout, &policy
                )?)?
            );
        }
        [area, backend, request_path, receipt_path, layout_path, policy_path]
            if area == "materialize" && backend == "sparse" =>
        {
            let request: UniversalCapabilityPlanningRequest =
                read_json_bounded(Path::new(request_path))?;
            let receipt: UniversalCapabilityShadowPlanReceipt =
                read_json_bounded(Path::new(receipt_path))?;
            let layout: ReceiverMaterializationLayout = read_json_bounded(Path::new(layout_path))?;
            let policy: SparseShadowPolicy = read_json_bounded(Path::new(policy_path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&materialize_replayed_sparse_shadow(
                    &request, &receipt, &layout, &policy
                )?)?
            );
        }
        [area, backend, request_path, receipt_path, receiver_layout_path, steering_layout_path, policy_path]
            if area == "materialize" && backend == "steering" =>
        {
            let request: UniversalCapabilityPlanningRequest =
                read_json_bounded(Path::new(request_path))?;
            let receipt: UniversalCapabilityShadowPlanReceipt =
                read_json_bounded(Path::new(receipt_path))?;
            let receiver_layout: ReceiverMaterializationLayout =
                read_json_bounded(Path::new(receiver_layout_path))?;
            let steering_layout: ActivationSteeringLayout =
                read_json_bounded(Path::new(steering_layout_path))?;
            let policy: ActivationSteeringPolicy = read_json_bounded(Path::new(policy_path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&materialize_replayed_activation_steering_shadow(
                    &request,
                    &receipt,
                    &receiver_layout,
                    &steering_layout,
                    &policy
                )?)?
            );
        }
        [area, command, root, request_path]
            if area == "receiver" && command == "inspect-safetensors" =>
        {
            let request: SafeTensorsReceiverRequest = read_json_bounded(Path::new(request_path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&inspect_safetensors_receiver(
                    Path::new(root),
                    &request
                )?)?
            );
        }
        [area, command, input_path] if area == "select" && command == "backend" => {
            let input: BackendSelectionInput = read_json_bounded(Path::new(input_path))?;
            println!("{}", serde_json::to_string_pretty(&input.execute()?)?);
        }
        [area, command, input_path] if area == "measure" && command == "universality" => {
            let input: UniversalityEvidenceInput = read_json_bounded(Path::new(input_path))?;
            println!("{}", serde_json::to_string_pretty(&input.execute()?)?);
        }
        [area, command, input_path] if area == "discover" && command == "capabilities" => {
            let input: CapabilityDiscoveryRequest = read_json_bounded(Path::new(input_path))?;
            println!("{}", serde_json::to_string_pretty(&input.execute()?)?);
        }
        [area, command, input_path] if area == "gate" && command == "promotion" => {
            let input: UniversalPromotionGateRequest = read_json_bounded(Path::new(input_path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&evaluate_universal_promotion_gate(&input)?)?
            );
        }
        [area, command, runner_path, input_path] if area == "shadow" && command == "run" => {
            let input: ShadowEvaluationInput = read_json_bounded(Path::new(input_path))?;
            if input.schema != "cerebro.tidex.shadow_evaluation_input/v1" {
                return Err("shadow_evaluation_input_schema_invalid".into());
            }
            let runner = AuthenticatedBytes::from_trusted_bytes(read_bytes_bounded(
                Path::new(runner_path),
                64 * 1024 * 1024,
            )?);
            println!(
                "{}",
                serde_json::to_string_pretty(&run_shadow_evaluation(
                    runner,
                    &input.bundle,
                    input.arguments,
                    input.limits,
                    input.requirements
                )?)?
            );
        }
        [area, command, path] if area == "receiver" && command == "project-physical-profile" => {
            let root = configured_private_root()?;
            let input: cerebro_tidex::checkpoint_adapter::PhysicalReceiverProjectionInput =
                read_json_bounded(Path::new(path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &cerebro_tidex::checkpoint_adapter::project_authenticated_receiver(
                        &root, &input
                    )?
                )?
            );
        }
        [area, command, path] if area == "materialize" && command == "checkpoint" => {
            let root = configured_private_root()?;
            let input: cerebro_tidex::checkpoint_adapter::CompiledCheckpointRequest =
                read_json_bounded(Path::new(path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &cerebro_tidex::checkpoint_adapter::materialize_compiled_checkpoint(
                        &root, &input
                    )?
                )?
            );
        }
        [area, command, path] if area == "materialize" && command == "verify-checkpoint" => {
            let root = configured_private_root()?;
            let input: PrivateFileReference = read_json_bounded(Path::new(path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &cerebro_tidex::checkpoint_adapter::authenticate_compiled_checkpoint(
                        &root, &input
                    )?
                )?
            );
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
                        {"id":"compile.receiver","canonical_name":"compile.receiver.operational","status":"implemented_experimental","engine":"receiver_compiler::compile_receiver_capability","evidence_status":"bounded_numeric_tests","reason":"operational IR compilation with a learned inverse predictor, protection and trust-region gates; V66 LoRA evidence does not establish this Rust compiler on an LLM"},
                        {"id":"materialize.receiver.verified_behavior","status":"implemented_experimental","engine":"quality/experiments/v66_mbpp_receiver_compile.py","evidence_status":"bounded_cross_model_experimental","reason":"V66 is a separate verified-behavior LoRA training backend; no evidence of target-training-free functional compilation"},
                        {"id":"materialize.receiver.weights","status":"implemented_experimental","engine":"weight_actuator::materialize_dense_delta_checkpoint","reason":"candidate checkpoint writer; V67 uses a V66-derived delta and is not evidence of independent functional transfer"},
                        {"id":"compile.receiver.distributed_lora_axes","status":"implemented_candidate_backend","engine":"weight_actuator::import_peft_lora_as_dense_axis -> receiver_weight_binding::assemble_distributed_lora_basis -> capability_ir::execute_linear_readout -> receiver_weight_binding::prepare_receiver_weight_candidate","reason":"reconstructs calibration-only PEFT LoRAs as authenticated full multiblock axes; distributed target compilation requires executed CapabilityIR evidence and performs no target receiver training; independent target execution remains mandatory"},
                        {"id":"compile.receiver.capability_ir_readout","status":"experimental_partial_candidate_only","engine":"capability_ir::execute_linear_readout -> receiver_compiler::compile_receiver_readout_capability","reason":"executes an authenticated resident readout fragment before compiling receiver-native coordinates; full task semantics and independent receiver validation remain separate"},
                        {"id":"compile.receiver.measured_weights","status":"experimental_candidate_only","engine":"receiver_weight_binding::prepare_receiver_weight_candidate","reason":"authenticated measured-response calibration -> receiver coordinates -> dense delta -> standalone candidate; inverse predictions are not model execution or capability-transfer evidence"},
                        {"id":"benchmark.portability","canonical_name":"benchmark.receiver_compilation","status":"implemented","engine":"receiver_compiler::benchmark_receiver_portability_leave_one_out","reason":"leave-one-capability-out functional-space benchmark over declared calibration cases; the legacy portability id is retained for compatibility, while the benchmark measures receiver compilation recovery inside that domain and is not a universal cross-model claim"},
                        {"id":"receiver.normalize.sharded_safetensors","status":"implemented","engine":"weight_actuator::normalize_sharded_safetensors","reason":"validates a Hugging Face SafeTensors weight_map against every shard and streams exact tensor payloads into one immutable content-addressed SafeTensors checkpoint; normalization is structural/data-plane evidence, not behavioral equivalence or promotion"},
                        {"id":"receiver.profile.physical","status":"implemented","engine":"model_adaptation::profile_receiver_model","reason":"binds an exact single-file SafeTensors receiver to config, tokenizer, physical topology and authenticated adaptation layouts; standard Hugging Face sharded SafeTensors inputs are first normalized into the same retained single-file authority; unsupported surfaces remain fail-closed"},
                        {"id":"adapter_bank.modular","status":"implemented","engine":"adapter_bank::AdapterBank","reason":"immutable manifests with normalized retained provenance and a transactionally published snapshot chain"},
                        {"id":"adapter_bank.index.dynamic","status":"implemented","engine":"adapter_bank::AdapterBank::query","reason":"capability/model projections are regenerated and authenticated from the primary manifest table"},
                        {"id":"adapter.compose.exact_dense","status":"implemented","engine":"adapter_bank::AdapterBank::compose_exact","reason":"canonical ordered f32 axes multiplied and accumulated in f64 with one final f32 rounding; no SVD, pruning or rank truncation"},
                        {"id":"adapter_bank.lifecycle","status":"implemented_governed","engine":"adapter_bank::AdapterBank::{authorize_governed_promotion_request,activate,revoke,rollback}","reason":"authorization reopens and semantically reauthenticates sealed gate/PETFC/canary witnesses before minting a current-index-bound permit; activation consumes that permit; revocation is sticky and transitive; rollback publishes a new forward revision"},
                        {"id":"compile.universal_capability","status":"implemented_experimental","engine":"universal_capability_compiler::execute_experimental_universal_capability_request","reason":"versioned JSON request binds a verified source envelope, closed CapabilityIR, operational contract, receiver calibration, protection and risk controls. The emitted receipt is request-bound and replayable. A passing result is experimental only and cannot authorize deployment or promotion"},
                        {"id":"receiver.inspect_safetensors","status":"implemented_experimental","engine":"checkpoint_adapter::inspect_safetensors_receiver","reason":"read-only inspection authenticates exact checkpoint, configuration and tokenizer bytes; validates SafeTensors geometry and produces snapshot-bound receiver layout plus architecture-family fingerprint"},
                        {"id":"capability.discover_behavior","status":"implemented_experimental","engine":"capability_discovery::discover_capabilities","reason":"repeated functional probes, seeds, controls, closure and contraction determine whether evidence is ready for CapabilityIR; tensor names alone never declare a capability"},
                        {"id":"materialize.shadow","status":"implemented_experimental","backends":["receiver_coordinates","dense_delta","low_rank","sparse_delta","activation_steering"],"reason":"all backends consume only replayed compiler target deltas and remain inert, layout-bound and non-actuating"},
                        {"id":"materialize.select","status":"implemented_experimental","engine":"materialization_selector::select_materialization_backend","reason":"canonical comparative ranking over supplied metrics; hybrid proposals require joint execution evidence before promotion"},
                        {"id":"measure.universality_n","status":"implemented_experimental","engine":"universality_evidence::measure_universality_n","reason":"N summarizes supplied trials with held-out, zero-step, receiver/family, successful coverage, preservation and control gates; execution provenance and independence must be established separately"},
                        {"id":"gate.universal_promotion","status":"implemented_non_actuating","engine":"universal_promotion_gate::evaluate_universal_promotion_gate","reason":"can establish readiness for a separate promotion authority but always returns authorizes_activation=false"},
                        {"id":"materialize.compiled_checkpoint","status":"implemented_candidate_only","engine":"checkpoint_adapter::materialize_compiled_checkpoint -> weight_actuator::materialize_dense_delta_checkpoint","reason":"replays universal planning and dense/low-rank/sparse/coordinate representations against the live physical profile; no model-execution or promotion claim"},
                        {"id":"runtime.sleep","status":"implemented","engine":"BrainEngine::sleep_cycle"},
                        {"id":"capability_ir.v63.contract","status":"implemented_foundation","engine":"capability_ir::OperationalCapabilityContract","reason":"StateIR anchors, repeated OperatorIR transitions, canonical transition signatures, closure and contraction verification are implemented; evidence is bounded to tested domains and does not establish a universal capability representation across arbitrary models or tasks"},
                        {"id":"model.assistance","status":"configured_not_authoritative","reason":"model profiles are selectable; no model call is permitted to create evidence or promotion authority"}
                    ]
                }))?
            );
        }
        _ => return Err(usage().into()),
    }
    Ok(())
}

fn read_benchmark_json_bounded<T: serde::de::DeserializeOwned>(
    path: &Path,
) -> Result<T, Box<dyn std::error::Error>> {
    read_json_bounded_with_error(path, "tidex_benchmark_input_too_large")
}

fn read_json_bounded<T: serde::de::DeserializeOwned>(
    path: &Path,
) -> Result<T, Box<dyn std::error::Error>> {
    read_json_bounded_with_error(path, "tidex_cli_json_input_too_large")
}

fn read_json_bounded_with_error<T: serde::de::DeserializeOwned>(
    path: &Path,
    too_large_error: &'static str,
) -> Result<T, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(MAX_CLI_JSON_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > MAX_CLI_JSON_BYTES {
        return Err(too_large_error.into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn read_bytes_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        return Err("tidex_binary_input_too_large".into());
    }
    Ok(bytes)
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
    concat!(
        "usage:\n",
        "  tidex workspace create <name> --target <absolute-path>\n",
        "  tidex workspace use <name>\n",
        "  tidex workspace show\n",
        "  tidex model add <name> --provider openai-compatible --url <endpoint> --model <model>\n",
        "  tidex model use <name>\n",
        "  tidex acquire [--path <relative-project-path>]\n",
        "  tidex benchmark portability <input.json>\n",
        "  tidex benchmark response <input.json>\n",
        "  tidex benchmark receiver-basis <input.json>\n",
        "  tidex receiver freeze-compiler <input.json>\n",
        "  tidex receiver verify-compiler <compiler.json>\n",
        "  tidex receiver describe-readout <input.json>\n",
        "  tidex receiver acquire-readout <input.json>\n",
        "  tidex receiver import-axis <values-reference.json>\n",
        "  tidex receiver normalize-sharded <input.json>\n",
        "  tidex receiver import-lora-axis <input.json>\n",
        "  tidex receiver profile <input.json>\n",
        "  tidex receiver verify-profile <reference.json>\n",
        "  tidex receiver verify-live-profile <reference.json>\n",
        "  tidex receiver assemble-lora-basis <input.json>\n",
        "  tidex receiver compile <request-reference.json>\n",
        "  tidex receiver inspect <candidate-reference.json>\n",
        "  tidex receiver materialize <candidate-reference.json> --base-model <model.safetensors> --output <private-root/model.safetensors>\n",
        "  tidex adapter-bank import <input.json>\n",
        "  tidex adapter-bank compose <input.json>\n",
        "  tidex adapter-bank materialize <input.json>\n",
        "  tidex adapter-bank authorize <input.json>\n",
        "  tidex adapter-bank verify-materialization <reference.json>\n",
        "  tidex adapter-bank query <query.json>\n",
        "  tidex adapter-bank show <lookup.json>\n",
        "  tidex adapter-bank resolve <input.json>\n",
        "  tidex adapter-bank verify-resolution <resolution.json>\n",
        "  tidex adapter-bank activate <input.json>\n",
        "  tidex adapter-bank revoke <input.json>\n",
        "  tidex adapter-bank rollback <input.json>\n",
        "  tidex adapter-bank status\n",
        "  tidex compile universal <input.json>\n",
        "  tidex compile universal-replay <input.json> <receipt.json>\n",
        "  tidex compile universal-plan <request.json>\n",
        "  tidex compile universal-plan-replay <request.json> <receipt.json>\n",
        "  tidex materialize dense <request.json> <receipt.json> <layout.json>\n",
        "  tidex materialize low-rank <request.json> <receipt.json> <layout.json> <policy.json>\n",
        "  tidex materialize sparse <request.json> <receipt.json> <layout.json> <policy.json>\n",
        "  tidex materialize steering <request.json> <receipt.json> <receiver-layout.json> <steering-layout.json> <policy.json>\n",
        "  tidex receiver inspect-safetensors <root> <request.json>\n",
        "  tidex discover capabilities <input.json>\n",
        "  tidex shadow run <runner> <input.json>\n",
        "  tidex select backend <input.json>\n",
        "  tidex measure universality <input.json>\n",
        "  tidex gate promotion <input.json>\n",
        "  tidex receiver project-physical-profile <input.json>\n",
        "  tidex materialize checkpoint <input.json>\n",
        "  tidex materialize verify-checkpoint <reference.json>\n",
        "  tidex capabilities"
    )
}
