use crate::authority::{
    ensure_private_parent, existing_regular_file_under_root, root_relative_path,
};
use crate::error::{BrainError, BrainResult};
use crate::security::secure_file;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

const EVENT_SCHEMA_V2: &str = "cerebro.tidex.ledger_event/v2";
const DOMAIN_V1: &[u8] = b"CEREBRO:TIDEX:LEDGER:v1\0";
const DOMAIN_V2: &[u8] = b"CEREBRO:TIDEX:LEDGER:v2\0";

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

/// The ledger is an authority boundary, so even a read must never follow an
/// untrusted root or `state/` symlink.  The caller's private root is expected
/// to already be authenticated by its subsystem; this verifies the local path
/// shape needed before the shared authority helpers walk children beneath it.
fn validate_ledger_root(root: &Path) -> BrainResult<()> {
    let metadata = fs::symlink_metadata(root)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err(BrainError::Integrity("ledger_private_root_invalid".into()));
    }
    Ok(())
}

/// Validate the ledger's existing parent without creating it. Read-only ledger
/// operations must remain read-only when no ledger has been initialized, but
/// must fail if an existing parent is a link or a non-directory.
fn validate_existing_ledger_parent(root: &Path, path: &Path) -> BrainResult<()> {
    validate_ledger_root(root)?;
    let relative = root_relative_path(root, path)?;
    let parent = relative
        .parent()
        .ok_or_else(|| BrainError::Integrity("ledger_parent_missing".into()))?;
    let parent_path = root.join(parent);
    match fs::symlink_metadata(&parent_path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
                return Err(BrainError::Integrity("ledger_parent_invalid".into()));
            }
            let canonical_root = root.canonicalize()?;
            let canonical_parent = parent_path.canonicalize()?;
            if !canonical_parent.starts_with(&canonical_root) {
                return Err(BrainError::Integrity(
                    "ledger_parent_outside_private_root".into(),
                ));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

/// Return the ledger only if it is an existing, regular private file.  This
/// rejects both a symlinked ledger leaf and a symlinked intermediate `state/`
/// directory before any ledger bytes are opened.
fn existing_ledger_file(root: &Path) -> BrainResult<Option<PathBuf>> {
    let path = ledger_path(root);
    validate_existing_ledger_parent(root, &path)?;
    match fs::symlink_metadata(&path) {
        Ok(_) => Ok(Some(existing_regular_file_under_root(root, &path)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

struct Lock {
    root: PathBuf,
    path: PathBuf,
}
impl Drop for Lock {
    fn drop(&mut self) {
        // Never follow a replacement symlink while releasing a lock. If the
        // lock path was tampered with, leave it in place and make the next
        // operation fail closed rather than deleting an arbitrary file.
        if existing_regular_file_under_root(&self.root, &self.path).is_ok() {
            let _ = fs::remove_file(&self.path);
        }
    }
}
fn acquire(root: &Path) -> BrainResult<Lock> {
    let path = lock_path(root);
    validate_ledger_root(root)?;
    ensure_private_parent(root, &path)?;
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(BrainError::Integrity("ledger_lock_path_invalid".into()));
        }
        Ok(_) => {
            return Err(BrainError::Integrity(
                "ledger_locked_or_unavailable:lock_exists".into(),
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let _ = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)
        .map_err(|error| BrainError::Integrity(format!("ledger_locked_or_unavailable:{error}")))?;
    let verified = existing_regular_file_under_root(root, &path)?;
    secure_file(&verified)?;
    Ok(Lock {
        root: root.to_path_buf(),
        path,
    })
}

fn verify_v2_event(event: &LedgerEvent, seq: u64, prev: &str) -> BrainResult<()> {
    if event.schema != EVENT_SCHEMA_V2 || event.seq != seq || event.prev_hash != prev {
        return Err(BrainError::Integrity(
            "ledger_chain_sequence_or_parent_mismatch".into(),
        ));
    }
    // Payload must remain valid JSON, but chain integrity is over the exact
    // stored string bytes, not a floating-point round trip.
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

pub fn verify(root: &Path) -> BrainResult<LedgerStatus> {
    let Some(path) = existing_ledger_file(root)? else {
        return Ok(LedgerStatus {
            events: 0,
            head: "0".repeat(64),
        });
    };
    let file = OpenOptions::new().read(true).open(&path)?;
    let mut prev = "0".repeat(64);
    let mut seq = 0u64;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        seq += 1;
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

/// Append is intentionally crate-private.  A valid hash chain is not, by
/// itself, authority to manufacture a TIDE-X event: public runtime entry
/// points must first validate the specific receipt-backed transaction they
/// are recording.
pub(crate) fn append(root: &Path, kind: &str, payload: Value) -> BrainResult<LedgerEvent> {
    if kind.trim().is_empty() {
        return Err(BrainError::Invalid("ledger_kind_empty".into()));
    }
    let _guard = acquire(root)?;
    let status = verify(root)?;
    let seq = status.events + 1;
    let payload_json = canonical_payload_json(&payload)?;
    // Prove before committing that the exact string can be parsed, while never
    // using the parsed value to derive the event hash.
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
    let path = ledger_path(root);
    let mut file = match fs::symlink_metadata(&path) {
        Ok(_) => {
            let verified = existing_regular_file_under_root(root, &path)?;
            OpenOptions::new().append(true).mode(0o600).open(verified)?
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)?,
        Err(error) => return Err(error.into()),
    };
    serde_json::to_writer(&mut file, &event)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    let verified = existing_regular_file_under_root(root, &path)?;
    secure_file(&verified)?;
    Ok(event)
}

pub fn contains_event_hash(root: &Path, target_hash: &str) -> BrainResult<bool> {
    let _ = verify(root)?;
    let Some(path) = existing_ledger_file(root)? else {
        return Ok(false);
    };
    let file = OpenOptions::new().read(true).open(path)?;
    for line in BufReader::new(file).lines() {
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
    let _ = verify(root)?;
    let Some(path) = existing_ledger_file(root)? else {
        return Ok(None);
    };
    let file = OpenOptions::new().read(true).open(path)?;
    let mut matched = None;
    for line in BufReader::new(file).lines() {
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
