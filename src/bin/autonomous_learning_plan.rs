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

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: autonomous_learning_plan <learning-target.json>")?;
    let file = fs::File::open(Path::new(&path))?;
    let mut bytes = Vec::new();
    file.take(MAX_LEARNING_TARGET_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > MAX_LEARNING_TARGET_BYTES {
        return Err("autonomous_learning_target_too_large".into());
    }
    let target: LearningTarget = serde_json::from_slice(&bytes)?;
    let plan = plan_autonomous_learning(&target)?;
    println!("{}", serde_json::to_string_pretty(&plan)?);
    Ok(())
}
