use cerebro_tidex::ledger;
use cerebro_tidex::security::configured_private_root;
use sha2::{Digest, Sha256};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = configured_private_root()?;
    let (status, events) = ledger::verified_v2_snapshot(&root)?;
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
