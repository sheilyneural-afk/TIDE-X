use cerebro_tidex::representation_evidence::record_representation_evidence;
use cerebro_tidex::security::configured_private_root;
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let source_payload = std::env::args()
        .nth(1)
        .ok_or("usage: record_representation_evidence <sealed-install-request.json>")?;
    if std::env::args().nth(2).is_some() {
        return Err("usage: record_representation_evidence <sealed-install-request.json>".into());
    }
    let private_root = configured_private_root()?;
    let receipt = record_representation_evidence(&private_root, Path::new(&source_payload))?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
