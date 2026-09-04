use cerebro_tidex::representation_evidence::record_representation_evidence;
use cerebro_tidex::security::PRIVATE_ROOT;
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
    let receipt =
        record_representation_evidence(Path::new(PRIVATE_ROOT), Path::new(&source_payload))?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
