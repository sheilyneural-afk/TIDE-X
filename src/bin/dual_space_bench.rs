use cerebro_tidex::contracts::{BrainConfig, DeltaObservation};
use cerebro_tidex::dual_space::{
    analyze_dual_space, DualSpaceAnalysisConfig, DualSpaceModel, RepresentationObservation,
};
use cerebro_tidex::engine::ReconstructionReport;
use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct RepresentationPayload {
    schema: String,
    probe_count: usize,
    probe_sha256: String,
    layer_count: usize,
    hidden_dim: usize,
    raw_dimension_per_observation: usize,
    sketch_dim: usize,
    task_labels_used: bool,
    observations: Vec<RepresentationObservation>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 4 {
        return Err("usage: dual_space_bench <observations.json> <tidex_report.json> <representation_observations.json>".into());
    }
    let observations: Vec<DeltaObservation> =
        serde_json::from_slice(&fs::read(Path::new(&args[1]))?)?;
    let report: ReconstructionReport = serde_json::from_slice(&fs::read(Path::new(&args[2]))?)?;
    let representations: RepresentationPayload =
        serde_json::from_slice(&fs::read(Path::new(&args[3]))?)?;
    if !matches!(
        representations.schema.as_str(),
        "cerebro.tidex.representation_observations/v1"
            | "cerebro.tidex.representation_observations/v2"
    ) || representations.task_labels_used
        || representations.observations.len() != observations.len()
    {
        return Err("representation payload contract invalid".into());
    }
    let cfg = BrainConfig::default();
    let model = DualSpaceModel {
        fields: &report.fields,
        field_coefficients: &report.field_coefficients,
        skill_source_mixtures: &report.skill_source_mixtures,
        parameter_inverse_mode: report.inverse_mode,
        parameter_promotable: report.promotion.allowed,
        functional_cv_r2: report.functional_cv_r2,
    };
    let dual = analyze_dual_space(
        &model,
        &observations,
        &representations.observations,
        DualSpaceAnalysisConfig {
            ridge: cfg.ridge,
            minimum_independence_groups: cfg.min_independent_apertures,
            minimum_representation_cv_r2: cfg.min_representation_cv_r2,
            minimum_match_accuracy: cfg.min_representation_match_accuracy,
            minimum_match_margin: cfg.min_representation_match_margin,
        },
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.dual_space_benchmark/v1",
            "probe_count":representations.probe_count,
            "probe_sha256":representations.probe_sha256,
            "layer_count":representations.layer_count,
            "hidden_dim":representations.hidden_dim,
            "raw_dimension_per_observation":representations.raw_dimension_per_observation,
            "sketch_dim":representations.sketch_dim,
            "task_labels_used":representations.task_labels_used,
            "dual_space":dual,
        }))?
    );
    Ok(())
}
