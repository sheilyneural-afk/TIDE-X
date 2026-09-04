use cerebro_tidex::learning_orchestrator::{plan_autonomous_learning, LearningTarget};
use std::fs;
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: autonomous_learning_plan <learning-target.json>")?;
    let target: LearningTarget = serde_json::from_slice(&fs::read(Path::new(&path))?)?;
    let plan = plan_autonomous_learning(&target)?;
    println!("{}", serde_json::to_string_pretty(&plan)?);
    Ok(())
}
