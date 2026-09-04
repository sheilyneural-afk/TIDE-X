use cerebro_tidex::active::plan_active_apertures;
use cerebro_tidex::contracts::ApertureCandidate;
use cerebro_tidex::engine::ReconstructionReport;
use cerebro_tidex::linalg::{norm, Matrix};
use serde_json::json;
use std::fs;
use std::path::Path;

fn normalize(values: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n = norm(values)?.max(1e-15);
    Ok(values.iter().map(|value| value / n).collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: active_aperture_bench <tidex_report.json>")?;
    let report: ReconstructionReport = serde_json::from_slice(&fs::read(Path::new(&path))?)?;
    let covariance = Matrix::from_rows(&report.resolution_map.posterior_covariance)?;
    if covariance.rows != report.fields.len()
        || report.field_coefficients.len() != report.observation_count
    {
        return Err("report posterior/field coefficient contract invalid".into());
    }
    let mean_posterior_variance = (0..covariance.rows)
        .map(|index| covariance.get(index, index))
        .sum::<f64>()
        / covariance.rows as f64;
    let noise_variance = mean_posterior_variance.max(f64::EPSILON);

    let mut candidates = Vec::new();
    for field_index in 0..report.fields.len() {
        let mut direction = vec![0.0; report.fields.len()];
        direction[field_index] = 1.0;
        candidates.push(ApertureCandidate {
            aperture_id: format!("isolate-field-{field_index}"),
            sensing_vector: direction,
            noise_variance,
            cost: 0.0,
            risk: 0.0,
        });
    }
    for (index, row) in report.field_coefficients.iter().enumerate() {
        candidates.push(ApertureCandidate {
            aperture_id: format!("repeat-observed-direction-{index}"),
            sensing_vector: normalize(row)?,
            noise_variance,
            cost: 0.0,
            risk: 0.0,
        });
    }
    let plan = plan_active_apertures(&candidates, &covariance, 0.0, 0.0, 3, false)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.active_aperture_real_posterior_benchmark/v1",
            "source_report":path,
            "field_count":report.fields.len(),
            "candidate_count":candidates.len(),
            "noise_model":"mean posterior marginal variance from real ResolutionMap",
            "cost_risk_mode":"information-only; cost and risk weights zero because no measured execution cost/risk supplied",
            "blind_data_accessed":false,
            "actual_experiment_outcomes_fabricated":false,
            "plan":plan,
        }))?
    );
    Ok(())
}
