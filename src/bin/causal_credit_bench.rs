use cerebro_tidex::artifact::sha256_file;
use cerebro_tidex::causal_credit::{estimate_causal_credit, CounterfactualEvaluation};
use cerebro_tidex::identity::SkillId;
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct ReplayPayload {
    schema: String,
    report_sha256: String,
    plan_sha256: String,
    field_ids: Vec<SkillId>,
    blind_data_accessed: bool,
    evaluations: Vec<CounterfactualEvaluation>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: causal_credit_bench <replay.json>")?;
    let replay_path = Path::new(&path);
    let payload: ReplayPayload = serde_json::from_slice(&fs::read(replay_path)?)?;
    if !matches!(
        payload.schema.as_str(),
        "cerebro.tidex.counterfactual_replay/v3"
    ) || payload.blind_data_accessed
        || payload.report_sha256.len() != 64
        || payload.plan_sha256.len() != 64
        || payload.field_ids.is_empty()
        || payload.field_ids.iter().collect::<BTreeSet<_>>().len() != payload.field_ids.len()
    {
        return Err("counterfactual replay contract invalid".into());
    }
    let report = estimate_causal_credit(&payload.evaluations)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.causal_credit_benchmark/v3",
            "blind_data_accessed":payload.blind_data_accessed,
            "replay_sha256":sha256_file(replay_path)?,
            "report_sha256":payload.report_sha256,
            "plan_sha256":payload.plan_sha256,
            "field_ids":payload.field_ids,
            "causal_credit":report,
        }))?
    );
    Ok(())
}
