use cerebro_tidex::ledger;
use cerebro_tidex::security::configured_private_root;
use sha2::{Digest, Sha256};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = configured_private_root()?;
    run_diagnose(&root)
}

fn run_diagnose(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let (status, events) = ledger::verified_v2_snapshot(root)?;
    for (line_index, event) in events.iter().enumerate() {
        let payload = event.payload()?;
        let payload_sha256 = format!("{:x}", Sha256::digest(event.payload_json.as_bytes()));
        println!(
            "line={} seq={} schema={} kind={} event_hash={} payload_sha256={} payload={}",
            line_index + 1,
            event.seq,
            event.schema,
            event.kind,
            event.event_hash,
            payload_sha256,
            serde_json::to_string(&payload)?
        );
    }
    println!("verified_events={}", status.events);
    println!("verified_head={}", status.head);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_diagnose_on_initialized_engine_root() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("test-bin-diag-{unique}"));
        std::fs::create_dir_all(&root).unwrap();
        cerebro_tidex::security::secure_dir(&root).unwrap();

        assert!(run_diagnose(&root).is_ok());
        let _ = std::fs::remove_dir_all(&root);
    }
}
