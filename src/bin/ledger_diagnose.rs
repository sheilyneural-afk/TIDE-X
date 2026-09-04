use cerebro_tidex::ledger::{self, LedgerEvent};
use cerebro_tidex::security::PRIVATE_ROOT;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(PRIVATE_ROOT);
    let status = ledger::verify(root)?;
    let path = root.join("state/ledger.jsonl");
    let file = fs::File::open(&path)?;
    let mut parsed = 0u64;
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let raw = line?;
        if raw.trim().is_empty() {
            continue;
        }
        let event: LedgerEvent = serde_json::from_str(&raw)?;
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
        parsed += 1;
    }
    if parsed != status.events {
        return Err(format!(
            "ledger parsed event count mismatch:{}:{}",
            parsed, status.events
        )
        .into());
    }
    println!("verified_events={}", status.events);
    println!("verified_head={}", status.head);
    Ok(())
}
