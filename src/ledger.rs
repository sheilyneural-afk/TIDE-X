use crate::authority::{
    read_existing_private_file_bounded, replace_private_file_atomic, with_private_authority_lock,
};
use crate::error::{BrainError, BrainResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

const EVENT_SCHEMA_V2: &str = "cerebro.tidex.ledger_event/v2";
const DOMAIN_V1: &[u8] = b"CEREBRO:TIDEX:LEDGER:v1\0";
const DOMAIN_V2: &[u8] = b"CEREBRO:TIDEX:LEDGER:v2\0";
const MAX_LEDGER_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LedgerEvent {
    pub schema: String,
    pub seq: u64,
    pub prev_hash: String,
    pub kind: String,
    /// Exact canonical JSON bytes, represented as a JSON string in the outer
    /// ledger record. The event hash is over these exact UTF-8 bytes. We never
    /// parse and reserialize them to verify the chain, avoiding f64 ULP drift.
    pub payload_json: String,
    pub event_hash: String,
}

impl LedgerEvent {
    pub fn payload(&self) -> BrainResult<Value> {
        Ok(serde_json::from_str(&self.payload_json)?)
    }
}

#[derive(Debug, Clone)]
pub struct LedgerStatus {
    pub events: u64,
    pub head: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyLedgerEventV1 {
    seq: u64,
    prev_hash: String,
    kind: String,
    payload: Value,
    event_hash: String,
}

fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut output = serde_json::Map::new();
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            for key in keys {
                output.insert(key.clone(), canonical(&map[key]));
            }
            Value::Object(output)
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical).collect()),
        _ => value.clone(),
    }
}

fn canonical_payload_json(payload: &Value) -> BrainResult<String> {
    Ok(serde_json::to_string(&canonical(payload))?)
}

fn hash_event_v1(seq: u64, prev: &str, kind: &str, payload: &Value) -> BrainResult<String> {
    let bytes = serde_json::to_vec(&canonical(payload))?;
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_V1);
    hasher.update(seq.to_be_bytes());
    hasher.update(prev.as_bytes());
    hasher.update(kind.as_bytes());
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_event_v2(seq: u64, prev: &str, kind: &str, payload_json: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_V2);
    hasher.update(seq.to_be_bytes());
    hasher.update(prev.as_bytes());
    hasher.update(kind.as_bytes());
    hasher.update(payload_json.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn ledger_path(root: &Path) -> PathBuf {
    root.join("state").join("ledger.jsonl")
}

fn lock_path(root: &Path) -> PathBuf {
    root.join("state").join(".ledger.lock")
}

fn read_ledger_bytes(root: &Path) -> BrainResult<Option<Vec<u8>>> {
    let path = ledger_path(root);
    match read_existing_private_file_bounded(root, &path, MAX_LEDGER_BYTES) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(BrainError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn verify_v2_event(event: &LedgerEvent, seq: u64, prev: &str) -> BrainResult<()> {
    if event.schema != EVENT_SCHEMA_V2 || event.seq != seq || event.prev_hash != prev {
        return Err(BrainError::Integrity(
            "ledger_chain_sequence_or_parent_mismatch".into(),
        ));
    }
    let _: Value = serde_json::from_str(&event.payload_json)?;
    let expected = hash_event_v2(
        event.seq,
        &event.prev_hash,
        &event.kind,
        &event.payload_json,
    );
    if expected != event.event_hash {
        return Err(BrainError::Integrity("ledger_event_hash_mismatch".into()));
    }
    Ok(())
}

fn verify_v1_event(event: &LegacyLedgerEventV1, seq: u64, prev: &str) -> BrainResult<()> {
    if event.seq != seq || event.prev_hash != prev {
        return Err(BrainError::Integrity(
            "ledger_chain_sequence_or_parent_mismatch".into(),
        ));
    }
    let expected = hash_event_v1(event.seq, &event.prev_hash, &event.kind, &event.payload)?;
    if expected != event.event_hash {
        return Err(BrainError::Integrity(
            "ledger_v1_event_hash_mismatch_requires_migration".into(),
        ));
    }
    Ok(())
}

fn verify_ledger_bytes(bytes: &[u8]) -> BrainResult<LedgerStatus> {
    let mut prev = "0".repeat(64);
    let mut seq = 0u64;
    for line in BufReader::new(bytes).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        seq = seq
            .checked_add(1)
            .ok_or_else(|| BrainError::Integrity("ledger_sequence_overflow".into()))?;
        let raw: Value = serde_json::from_str(&line)?;
        let is_v2 = raw
            .get("schema")
            .and_then(Value::as_str)
            .is_some_and(|schema| schema == EVENT_SCHEMA_V2);
        if is_v2 {
            let event: LedgerEvent = serde_json::from_value(raw)?;
            verify_v2_event(&event, seq, &prev)?;
            prev = event.event_hash;
        } else {
            let event: LegacyLedgerEventV1 = serde_json::from_value(raw)?;
            verify_v1_event(&event, seq, &prev)?;
            prev = event.event_hash;
        }
    }
    Ok(LedgerStatus {
        events: seq,
        head: prev,
    })
}

pub fn verify(root: &Path) -> BrainResult<LedgerStatus> {
    match read_ledger_bytes(root)? {
        Some(bytes) => verify_ledger_bytes(&bytes),
        None => Ok(LedgerStatus {
            events: 0,
            head: "0".repeat(64),
        }),
    }
}

/// Return one verified V2 ledger snapshot for diagnostics without reopening the
/// ledger after chain verification. Legacy V1 records are deliberately rejected
/// here because `LedgerEvent` represents only the current wire schema.
pub fn verified_v2_snapshot(root: &Path) -> BrainResult<(LedgerStatus, Vec<LedgerEvent>)> {
    let Some(bytes) = read_ledger_bytes(root)? else {
        return Ok((
            LedgerStatus {
                events: 0,
                head: "0".repeat(64),
            },
            Vec::new(),
        ));
    };
    let status = verify_ledger_bytes(&bytes)?;
    let mut events = Vec::new();
    for line in BufReader::new(bytes.as_slice()).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let raw: Value = serde_json::from_str(&line)?;
        if raw.get("schema").and_then(Value::as_str) != Some(EVENT_SCHEMA_V2) {
            return Err(BrainError::Integrity(
                "ledger_diagnostic_snapshot_contains_legacy_event".into(),
            ));
        }
        events.push(serde_json::from_value(raw)?);
    }
    if events.len() as u64 != status.events {
        return Err(BrainError::Integrity(
            "ledger_diagnostic_snapshot_count_mismatch".into(),
        ));
    }
    Ok((status, events))
}

/// Append is intentionally crate-private. A ledger update is performed as one
/// descriptor-bound, process-locked atomic replacement rather than an in-place
/// append. This makes verification and publication operate on one authenticated
/// snapshot and prevents partial-line crash states or verify→reopen races.
pub(crate) fn append(root: &Path, kind: &str, payload: Value) -> BrainResult<LedgerEvent> {
    if kind.trim().is_empty() {
        return Err(BrainError::Invalid("ledger_kind_empty".into()));
    }
    let lock = lock_path(root);
    with_private_authority_lock(root, &lock, || {
        let mut current = read_ledger_bytes(root)?.unwrap_or_default();
        let status = verify_ledger_bytes(&current)?;
        let seq = status
            .events
            .checked_add(1)
            .ok_or_else(|| BrainError::Integrity("ledger_sequence_overflow".into()))?;
        let payload_json = canonical_payload_json(&payload)?;
        let _: Value = serde_json::from_str(&payload_json)?;
        let event_hash = hash_event_v2(seq, &status.head, kind, &payload_json);
        let event = LedgerEvent {
            schema: EVENT_SCHEMA_V2.into(),
            seq,
            prev_hash: status.head,
            kind: kind.to_string(),
            payload_json,
            event_hash,
        };
        let mut line = serde_json::to_vec(&event)?;
        line.push(b'\n');
        let next_len = current
            .len()
            .checked_add(line.len())
            .ok_or_else(|| BrainError::Invalid("ledger_size_overflow".into()))?;
        if u64::try_from(next_len)
            .map_err(|_| BrainError::Invalid("ledger_size_overflow".into()))?
            > MAX_LEDGER_BYTES
        {
            return Err(BrainError::Invalid("ledger_size_limit_exceeded".into()));
        }
        current.extend_from_slice(&line);
        replace_private_file_atomic(root, &ledger_path(root), &current, None)?;
        let persisted = read_ledger_bytes(root)?
            .ok_or_else(|| BrainError::Integrity("ledger_missing_after_atomic_replace".into()))?;
        if persisted != current {
            return Err(BrainError::Integrity(
                "ledger_atomic_replace_content_mismatch".into(),
            ));
        }
        let persisted_status = verify_ledger_bytes(&persisted)?;
        if persisted_status.events != seq || persisted_status.head != event.event_hash {
            return Err(BrainError::Integrity(
                "ledger_atomic_replace_chain_mismatch".into(),
            ));
        }
        Ok(event)
    })
}

pub fn contains_event_hash(root: &Path, target_hash: &str) -> BrainResult<bool> {
    let Some(bytes) = read_ledger_bytes(root)? else {
        return Ok(false);
    };
    let _ = verify_ledger_bytes(&bytes)?;
    for line in BufReader::new(bytes.as_slice()).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let raw: Value = serde_json::from_str(&line)?;
        if raw.get("event_hash").and_then(Value::as_str) == Some(target_hash) {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn find_v2_event_by_payload_string(
    root: &Path,
    kind: &str,
    key: &str,
    expected: &str,
) -> BrainResult<Option<LedgerEvent>> {
    let Some(bytes) = read_ledger_bytes(root)? else {
        return Ok(None);
    };
    let _ = verify_ledger_bytes(&bytes)?;
    let mut matched = None;
    for line in BufReader::new(bytes.as_slice()).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let raw: Value = serde_json::from_str(&line)?;
        if raw.get("schema").and_then(Value::as_str) != Some(EVENT_SCHEMA_V2) {
            continue;
        }
        let event: LedgerEvent = serde_json::from_value(raw)?;
        if event.kind != kind {
            continue;
        }
        let payload = event.payload()?;
        if payload.get(key).and_then(Value::as_str) == Some(expected)
            && matched.replace(event).is_some()
        {
            return Err(BrainError::Integrity(
                "ledger_v2_event_payload_match_ambiguous".into(),
            ));
        }
    }
    Ok(matched)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cerebro-ledger-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        root
    }

    #[test]
    fn v2_hash_survives_values_that_drift_under_json_f64_roundtrip() {
        let payload = json!({
            "a": 0.9376332445278925_f64,
            "b": [9.262570687188195_f64, 0.012341817216899103_f64],
        });
        let payload_json = canonical_payload_json(&payload).unwrap();
        let parsed: Value = serde_json::from_str(&payload_json).unwrap();
        // This is the exact class of issue observed in the real reconstruction
        // event: serde_json can return the neighboring f64 for decimal input.
        assert_ne!(payload, parsed);
        let hash = hash_event_v2(1, &"0".repeat(64), "x", &payload_json);
        let event = LedgerEvent {
            schema: EVENT_SCHEMA_V2.into(),
            seq: 1,
            prev_hash: "0".repeat(64),
            kind: "x".into(),
            payload_json,
            event_hash: hash,
        };
        verify_v2_event(&event, 1, &"0".repeat(64)).unwrap();
    }

    #[test]
    fn duplicate_v2_payload_key_is_ambiguous_not_first_match() {
        let root = temporary_root("duplicate");
        append(&root, "operation", json!({"operation_key":"same"})).unwrap();
        append(&root, "operation", json!({"operation_key":"same"})).unwrap();

        assert!(
            find_v2_event_by_payload_string(&root, "operation", "operation_key", "same").is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ledger_read_and_append_reject_symlinked_state_parent() {
        let root = temporary_root("state-symlink");
        let outside = temporary_root("state-symlink-outside");
        symlink(&outside, root.join("state")).unwrap();

        assert!(verify(&root).is_err());
        assert!(append(&root, "operation", json!({"operation_key":"one"})).is_err());

        fs::remove_file(root.join("state")).unwrap();
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn append_rejects_symlinked_lock_leaf() {
        let root = temporary_root("lock-symlink");
        let state = root.join("state");
        fs::create_dir(&state).unwrap();
        let outside = std::env::temp_dir().join(format!(
            "cerebro-ledger-lock-outside-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(&outside, b"not-a-lock").unwrap();
        symlink(&outside, state.join(".ledger.lock")).unwrap();

        assert!(append(&root, "operation", json!({"operation_key":"one"})).is_err());

        fs::remove_file(state.join(".ledger.lock")).unwrap();
        fs::remove_file(&outside).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn private_append_chain_detects_tampering() {
        let root = temporary_root("chain-tamper");
        append(&root, "a", json!({"x": 1})).unwrap();
        append(&root, "b", json!({"y": 2})).unwrap();
        assert_eq!(verify(&root).unwrap().events, 2);

        let path = root.join("state/ledger.jsonl");
        let raw = fs::read_to_string(&path).unwrap();
        let mut lines = raw.lines().map(str::to_string).collect::<Vec<_>>();
        let mut first: Value = serde_json::from_str(&lines[0]).unwrap();
        first["payload_json"] = Value::String("{\"x\":9}".into());
        lines[0] = serde_json::to_string(&first).unwrap();
        fs::write(&path, format!("{}\n", lines.join("\n"))).unwrap();

        assert!(verify(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
