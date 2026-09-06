use cerebro_tidex::artifact::{read_dvec_f32, sha256_file};
use cerebro_tidex::causal_credit::{certified_causal_priority_weights, CausalCreditReport};
use cerebro_tidex::engine::ReconstructionReport;
use cerebro_tidex::identity::SkillId;
use cerebro_tidex::interaction::second_order_interactions;
use cerebro_tidex::protected_map::{load_protected_cortex, ProtectedMapArtifactReport};
use cerebro_tidex::security::configured_private_root;
use cerebro_tidex::trust_region::apply_causal_priority_trust_region;
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct ProtectedWrapper {
    schema: String,
    task_labels_used: bool,
    map: ProtectedMapArtifactReport,
}

#[derive(Debug, Deserialize)]
struct Assignment {
    current_skill_id: SkillId,
    coefficient: f64,
    functional_cosine: f64,
}

#[derive(Debug, Deserialize)]
struct CausalPlan {
    schema: String,
    blind_data_accessed: bool,
    current_report_sha256: String,
    assignments: Vec<Assignment>,
}

#[derive(Debug, Deserialize)]
struct CausalCreditArtifact {
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
    if args.len() != 5 {
        return Err(
            "usage: trust_region_bench <report.json> <protected_map_v2.json> <causal_plan.json> <causal_credit.json>"
                .into(),
        );
    }
    let report_path = Path::new(&args[1]);
    let protected_path = Path::new(&args[2]);
    let plan_path = Path::new(&args[3]);
    let causal_path = Path::new(&args[4]);
    let report: ReconstructionReport = serde_json::from_slice(&fs::read(report_path)?)?;
    let protected: ProtectedWrapper = serde_json::from_slice(&fs::read(protected_path)?)?;
    let plan: CausalPlan = serde_json::from_slice(&fs::read(plan_path)?)?;
    let causal: CausalCreditArtifact = serde_json::from_slice(&fs::read(causal_path)?)?;
    let report_sha256 = sha256_file(report_path)?;
    let plan_sha256 = sha256_file(plan_path)?;
    let causal_sha256 = sha256_file(causal_path)?;
    let field_ids = report
        .fields
        .iter()
        .map(|field| field.skill_id.clone())
        .collect::<Vec<_>>();
    if protected.schema != "cerebro.tidex.protected_map_benchmark/v2"
        || protected.task_labels_used
        || plan.schema != "cerebro.tidex.causal_replay_plan/v2"
        || plan.blind_data_accessed
        || plan.current_report_sha256 != report_sha256
        || !report.promotion.allowed
        || report.fields.is_empty()
        || causal.schema != "cerebro.tidex.causal_credit_benchmark/v3"
        || causal.blind_data_accessed
        || causal.replay_sha256.len() != 64
        || causal.report_sha256 != report_sha256
        || causal.plan_sha256 != plan_sha256
        || causal.field_ids != field_ids
    {
        return Err("current trust-region input contract invalid".into());
    }
    let root = configured_private_root()?;
    let cortex = load_protected_cortex(&root, &protected.map)?;
    let dense_dim = cortex.parameter_importance.len();
    if dense_dim == 0 || protected.map.parameter_dimension != dense_dim {
        return Err("protected cortex dimension invalid".into());
    }
    let assignment_count = plan.assignments.len();
    let assignment_by_id = plan
        .assignments
        .into_iter()
        .map(|assignment| (assignment.current_skill_id.clone(), assignment))
        .collect::<BTreeMap<_, _>>();
    if assignment_by_id.len() != report.fields.len() || assignment_by_id.len() != assignment_count {
        return Err("trust coefficient identity count mismatch".into());
    }
    let mut dense_fields = Vec::with_capacity(report.fields.len());
    let mut proposed = Vec::with_capacity(report.fields.len());
    let numerical_tolerance = f64::EPSILON.sqrt() * 32.0;
    for field in &report.fields {
        let reference = field
            .dense_materialization
            .as_ref()
            .ok_or("trust field dense materialization missing")?;
        if reference.parameter_count != dense_dim as u64 {
            return Err("trust field dense dimension mismatch".into());
        }
        let dense = read_dvec_f32(&root, reference)?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>();
        dense_fields.push(dense);
        let assignment = assignment_by_id
            .get(&field.skill_id)
            .ok_or("trust field coefficient identity missing")?;
        if !assignment.coefficient.is_finite()
            || !assignment.functional_cosine.is_finite()
            || 1.0 - assignment.functional_cosine > numerical_tolerance
        {
            return Err("trust coefficient functional identity unresolved".into());
        }
        proposed.push(assignment.coefficient);
    }
    let interactions = second_order_interactions(&dense_fields, &cortex.parameter_importance)?;
    let diagonal_budget = (0..proposed.len())
        .map(|index| proposed[index] * proposed[index] * interactions.get(index, index))
        .sum::<f64>()
        .max(0.0);
    let causal_priority_weights =
        certified_causal_priority_weights(&causal.causal_credit, &field_ids)?;
    let result = apply_causal_priority_trust_region(
        &interactions,
        &proposed,
        diagonal_budget,
        &causal_priority_weights,
    )?;
    let matrix = (0..interactions.row_count())
        .map(|row| interactions.row_vec(row))
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.trust_region_benchmark/v3",
            "report_sha256":report_sha256,
            "protected_map_sha256":sha256_file(protected_path)?,
            "causal_plan_sha256":plan_sha256,
            "causal_credit_sha256":causal_sha256,
            "causal_priority_weight_kind":"causal_lower_confidence_bound_95",
            "field_ids":field_ids,
            "dense_parameter_dimension":dense_dim,
            "curvature_source":"certified ProtectedCortex f64 parameter importance",
            "interaction_matrix":matrix,
            "diagonal_budget":diagonal_budget,
            "off_diagonal_cost":result.proposed_quadratic_cost-diagonal_budget,
            "trust_region":result,
        }))?
    );
    Ok(())
}
