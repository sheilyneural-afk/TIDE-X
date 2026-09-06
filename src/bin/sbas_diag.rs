use cerebro_tidex::contracts::DeltaObservation;
use cerebro_tidex::linalg::{norm, sub};
use cerebro_tidex::sbas::reconstruct_trajectory;
use serde_json::json;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: sbas_diag <observations.json>")?;
    let obs: Vec<DeltaObservation> = serde_json::from_slice(&fs::read(Path::new(&path))?)?;
    let out = reconstruct_trajectory(&obs, 1e-6)?;
    let first = obs
        .first()
        .ok_or("sbas_diag requires at least one observation")?;
    let i = out
        .checkpoint_order
        .iter()
        .position(|x| x == &first.from_checkpoint)
        .ok_or("reconstruction omitted the first source checkpoint")?;
    let j = out
        .checkpoint_order
        .iter()
        .position(|x| x == &first.to_checkpoint)
        .ok_or("reconstruction omitted the first destination checkpoint")?;
    let predicted = sub(&out.potentials[j], &out.potentials[i])?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "cycle_rms":out.cycle_rms,
            "max_edge_residual":out.max_edge_residual,
            "first":{
                "from_index":i,"to_index":j,
                "delta_norm":norm(&first.delta)?,
                "predicted_norm":norm(&predicted)?,
                "residual_norm":norm(&sub(&predicted,&first.delta)?)?,
                "delta_head":&first.delta[..8.min(first.delta.len())],
                "predicted_head":&predicted[..8.min(predicted.len())],
            }
        }))?
    );
    Ok(())
}
