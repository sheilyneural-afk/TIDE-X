use cerebro_tidex::learning_orchestrator::{plan_autonomous_learning, LearningTarget};
use std::fs;
use std::io::Read;
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

const MAX_LEARNING_TARGET_BYTES: u64 = 64 * 1024 * 1024;

fn run_from_path(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(MAX_LEARNING_TARGET_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > MAX_LEARNING_TARGET_BYTES {
        return Err("autonomous_learning_target_too_large".into());
    }
    let target: LearningTarget = serde_json::from_slice(&bytes)?;
    let plan = plan_autonomous_learning(&target)?;
    Ok(serde_json::to_string_pretty(&plan)?)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: autonomous_learning_plan <learning-target.json>")?;
    let output = run_from_path(Path::new(&path))?;
    println!("{output}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_from_path_generates_valid_plan_or_errors() {
        assert!(run_from_path(Path::new("/tmp/nonexistent-learning-target.json")).is_err());

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_target = std::env::temp_dir().join(format!("test-bin-target-{unique}.json"));
        let target_json = serde_json::json!({
            "target_id": "target-bin-test",
            "capability_ids": ["a", "b", "c", "d"],
            "candidate_budget": 64,
            "plan_steps": 12,
            "noise_variance": 0.1,
            "cost_weight": 0.0,
            "risk_weight": 0.0
        });
        std::fs::write(
            &temp_target,
            serde_json::to_vec_pretty(&target_json).unwrap(),
        )
        .unwrap();

        let output = run_from_path(&temp_target).unwrap();
        let _ = std::fs::remove_file(&temp_target);
        assert!(output.contains("cerebro.tidex.autonomous_learning_plan/v1"));
    }
}
