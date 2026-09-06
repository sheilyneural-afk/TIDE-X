use cerebro_tidex::artifact::sha256_file;
use cerebro_tidex::causal_credit::CausalCreditReport;
use cerebro_tidex::cognitive_field::{
    CognitiveFieldConfig, CognitiveFieldDrive, DynamicCognitiveField,
};
use cerebro_tidex::engine::ReconstructionReport;
use cerebro_tidex::identity::SkillId;
use cerebro_tidex::linalg::Matrix;
use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct TrustArtifact {
    schema: String,
    report_sha256: String,
    field_ids: Vec<SkillId>,
    interaction_matrix: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct CausalArtifact {
    schema: String,
    blind_data_accessed: bool,
    replay_sha256: String,
    report_sha256: String,
    plan_sha256: String,
    field_ids: Vec<SkillId>,
    causal_credit: CausalCreditReport,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 4 {
        return Err(
            "usage: cognitive_field_bench <report.json> <trust_region.json> <causal_credit.json>"
                .into(),
        );
    }
    let report_path = Path::new(&args[1]);
    let trust_path = Path::new(&args[2]);
    let causal_path = Path::new(&args[3]);
    let report: ReconstructionReport = serde_json::from_slice(&fs::read(report_path)?)?;
    let trust: TrustArtifact = serde_json::from_slice(&fs::read(trust_path)?)?;
    let causal: CausalArtifact = serde_json::from_slice(&fs::read(causal_path)?)?;
    let report_sha = sha256_file(report_path)?;
    let field_ids = report
        .fields
        .iter()
        .map(|field| field.skill_id.clone())
        .collect::<Vec<_>>();
    if report.schema != "cerebro.tidex.reconstruction/v7"
        || !report.promotion.allowed
        || report.fields.is_empty()
        || trust.schema != "cerebro.tidex.trust_region_benchmark/v3"
        || trust.report_sha256 != report_sha
        || trust.field_ids != field_ids
        || causal.schema != "cerebro.tidex.causal_credit_benchmark/v3"
        || causal.blind_data_accessed
        || causal.report_sha256 != report_sha
        || causal.field_ids != field_ids
        || causal.replay_sha256.len() != 64
        || causal.plan_sha256.len() != 64
    {
        return Err("cognitive field benchmark input contract invalid".into());
    }
    let curvature = Matrix::from_rows(&trust.interaction_matrix)?;
    let config = CognitiveFieldConfig::default();
    let field =
        DynamicCognitiveField::build(&report.fields, &curvature, &causal.causal_credit, config)?;
    let n = field_ids.len();
    let zeros = vec![0.0; n];
    let zero_drive = CognitiveFieldDrive {
        evidence: zeros.clone(),
        prediction_error: zeros.clone(),
        inhibition: zeros.clone(),
        risk: zeros.clone(),
    };
    let spontaneous = field.evolve(&zeros, &zero_drive)?;
    if !spontaneous.converged {
        return Err("cognitive field spontaneous state failed to converge".into());
    }

    let mut impulse_responses = Vec::with_capacity(n);
    for stimulus_index in 0..n {
        let mut evidence = vec![0.0; n];
        evidence[stimulus_index] = 1.0;
        let state = field.evolve(
            &zeros,
            &CognitiveFieldDrive {
                evidence,
                prediction_error: vec![0.0; n],
                inhibition: vec![0.0; n],
                risk: vec![0.0; n],
            },
        )?;
        if !state.converged {
            return Err(format!(
                "cognitive field impulse failed to converge:{}",
                field_ids[stimulus_index]
            )
            .into());
        }
        let route = field.route_top_k(&state, n.min(2), 0.0)?;
        impulse_responses.push(json!({
            "stimulus_skill_id":field_ids[stimulus_index],
            "state":state,
            "top_k_route":route,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.dynamic_cognitive_field_benchmark/v1",
            "task_labels_used_for_dynamics":false,
            "report_sha256":report_sha,
            "trust_region_sha256":sha256_file(trust_path)?,
            "causal_credit_sha256":sha256_file(causal_path)?,
            "field_ids":field_ids,
            "model":field,
            "spontaneous_state":spontaneous,
            "impulse_responses":impulse_responses,
        }))?
    );
    Ok(())
}
