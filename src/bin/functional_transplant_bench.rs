use cerebro_tidex::transport::learn_functional_transplant;
use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Input {
    schema: String,
    source_functional_anchors: Vec<Vec<f64>>,
    target_coordinate_anchors: Vec<Vec<f64>>,
    holdout_source_functional_signature: Vec<f64>,
    ridge: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: functional_transplant_bench <input.json>")?;
    let input: Input = serde_json::from_slice(&fs::read(Path::new(&path))?)?;
    if input.schema != "cerebro.tidex.functional_transplant_input/v1" {
        return Err("functional transplant input schema invalid".into());
    }
    let map = learn_functional_transplant(
        &input.source_functional_anchors,
        &input.target_coordinate_anchors,
        input.ridge,
    )?;
    let transplanted = map.transplant(&input.holdout_source_functional_signature)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.functional_transplant_benchmark/v1",
            "functional_dim":map.functional_dim,
            "target_dim":map.target_dim,
            "anchor_count":map.anchor_count,
            "loo_cv_r2":map.loo_cv_r2,
            "mean_loo_cosine":map.mean_loo_cosine,
            "min_loo_cosine":map.min_loo_cosine,
            "resolved":map.resolved,
            "holdout_target_coefficients":transplanted.target_vector,
            "transport_resolved":transplanted.transport_resolved,
        }))?
    );
    Ok(())
}
