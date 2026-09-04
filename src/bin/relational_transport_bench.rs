use cerebro_tidex::transport::learn_relational_transport;
use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Input {
    schema: String,
    source_anchors: Vec<Vec<f64>>,
    target_anchors: Vec<Vec<f64>>,
    holdout_source_signature: Vec<f64>,
    ridge: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: relational_transport_bench <input.json>")?;
    let input: Input = serde_json::from_slice(&fs::read(Path::new(&path))?)?;
    if input.schema != "cerebro.tidex.relational_transport_input/v1" {
        return Err("relational transport input schema invalid".into());
    }
    let map =
        learn_relational_transport(&input.source_anchors, &input.target_anchors, input.ridge)?;
    let transplant = map.transplant(&input.holdout_source_signature)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.relational_transport_benchmark/v1",
            "anchor_count":map.anchor_count,
            "source_signature_dim":map.source_signature_dim,
            "target_signature_dim":map.target_signature_dim,
            "ridge":map.ridge,
            "loo_source_cosines":map.loo_source_cosines,
            "loo_target_cosines":map.loo_target_cosines,
            "loo_coefficient_norms":map.loo_coefficient_norms,
            "min_loo_source_cosine":map.min_loo_source_cosine,
            "min_loo_target_cosine":map.min_loo_target_cosine,
            "mean_loo_source_cosine":map.mean_loo_source_cosine,
            "mean_loo_target_cosine":map.mean_loo_target_cosine,
            "max_loo_coefficient_norm":map.max_loo_coefficient_norm,
            "map_resolved":map.resolved,
            "holdout":{
                "target_coefficients":transplant.target_coefficients,
                "predicted_target_signature":transplant.predicted_target_signature,
                "source_projection_cosine":transplant.source_projection_cosine,
                "coefficient_norm":transplant.coefficient_norm,
                "within_training_support":transplant.within_training_support,
                "resolved":transplant.resolved,
            }
        }))?
    );
    Ok(())
}
