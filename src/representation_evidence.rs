//! Immutable installation of sealed, task-agnostic representation evidence.
//!
//! This module deliberately does not update `state/observations`.  It turns a
//! sealed capture plus fresh, staged `DeltaObservation` records into immutable
//! F64 artifacts and new observation records in a dedicated staging namespace.
//! A later authoritative engine operation must explicitly consume those output
//! records before they can influence a skill bank.

use crate::artifact::{read_f64_artifact, ArtifactWriteAuthority, F64ArtifactRef};
use crate::authority::{
    ensure_private_parent, existing_regular_file_under_root, read_existing_private_file_bounded,
    root_relative_path, write_or_verify_immutable, PrivateFileReference,
};
use crate::contracts::DeltaObservation;
use crate::digest::{
    ObservationRecordDigest, RepresentationProtocolDigest, RepresentationRequestDigest,
    Sha256Digest,
};
use crate::error::{BrainError, BrainResult};
use crate::identity::ObservationId;
use crate::ledger;
use crate::security::verify_private_root;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use std::fs;
use std::path::{Path, PathBuf};

pub const REPRESENTATION_EVIDENCE_REQUEST_SCHEMA: &str =
    "cerebro.tidex.representation_evidence_install_request/v1";
pub const REPRESENTATION_CAPTURE_SCHEMA: &str = "cerebro.tidex.representation_capture/v1";
pub const REPRESENTATION_PROTOCOL_SCHEMA: &str = "cerebro.tidex.representation_protocol/v1";
pub const REPRESENTATION_EVIDENCE_RECEIPT_SCHEMA: &str =
    "cerebro.tidex.representation_evidence_receipt/v1";

const INSTALLED_OBSERVATIONS_RELATIVE: &str =
    "state/representation_evidence/installed-observations";
const RECEIPTS_RELATIVE: &str = "state/representation_evidence/receipts/by-request-sha";
const PROTOCOLS_RELATIVE: &str = "state/representation_protocols/by-sha";
const LEDGER_KIND: &str = "representation_evidence_recorded";
const MAX_REPRESENTATION_AUTHORITY_JSON_BYTES: u64 = 64 * 1024 * 1024;

/// Engine-compatible description of a sealed generic probe protocol.
///
/// `source_representation_sha256` is the digest of the canonical
/// [`RepresentationCapture`], not a path-dependent digest.  This makes the
/// protocol bind the measured shifts even after a staging directory is moved.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SealedRepresentationProtocol {
    pub schema: String,
    pub source_representation_sha256: Sha256Digest,
    pub probe_sha256: Sha256Digest,
    pub probe_text_sha256: Sha256Digest,
    pub forbidden_vocabulary_sha256: Sha256Digest,
    pub task_labels_used: bool,
    pub probe_vocabulary_overlap: Vec<String>,
    pub probe_count: u64,
    pub layer_count: u64,
    pub hidden_dim: u64,
    pub raw_dimension_per_observation: u64,
    pub sketch_dim: u64,
    pub sketch_seed: u64,
}

/// A representation shift produced by the sealed probe protocol for one
/// aperture observation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RepresentationShift {
    pub observation_id: ObservationId,
    pub raw_dimension: u64,
    pub shift: Vec<f64>,
}

/// The path-independent portion of the source payload that the protocol
/// authenticates.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RepresentationCapture {
    pub schema: String,
    pub observations: Vec<RepresentationShift>,
}

/// A source observation and its immutable, non-active destination record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RepresentationEvidenceInstallTarget {
    pub observation_id: ObservationId,
    /// Absolute path under the CEREBRO private root.
    pub source_observation_path: String,
    /// SHA-256 of the exact source observation JSON bytes.
    pub source_observation_sha256: ObservationRecordDigest,
    /// Absolute path under `state/representation_evidence/installed-observations`.
    /// The active `state/observations` namespace is never a legal target.
    pub destination_observation_path: String,
}

/// Strict source payload accepted by `record_representation_evidence`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RepresentationEvidenceInstallRequest {
    pub schema: String,
    pub protocol: SealedRepresentationProtocol,
    pub capture: RepresentationCapture,
    pub installations: Vec<RepresentationEvidenceInstallTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InstalledRepresentationEvidence {
    pub observation_id: ObservationId,
    pub source_observation_path: String,
    pub source_observation_sha256: ObservationRecordDigest,
    pub destination_observation_path: String,
    pub destination_observation_sha256: ObservationRecordDigest,
    pub representation_artifact: F64ArtifactRef,
}

/// Immutable receipt, linked to the ledger and suitable as direct provenance
/// input to a controlled TIDE-X learning finalizer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RepresentationEvidenceReceipt {
    pub schema: String,
    pub source_tree_digest: Sha256Digest,
    pub request_sha256: RepresentationRequestDigest,
    pub source_representation_sha256: Sha256Digest,
    pub representation_protocol_sha256: RepresentationProtocolDigest,
    pub installations_sha256: Sha256Digest,
    pub observation_count: usize,
    pub installations: Vec<InstalledRepresentationEvidence>,
    pub ledger_event_hash: String,
}

fn invalid<T>(message: impl Into<String>) -> BrainResult<T> {
    Err(BrainError::Invalid(message.into()))
}

fn integrity<T>(message: impl Into<String>) -> BrainResult<T> {
    Err(BrainError::Integrity(message.into()))
}

fn sha256_bytes(bytes: &[u8]) -> Sha256Digest {
    Sha256Digest::digest_bytes(bytes)
}

fn json_bytes<T: Serialize + ?Sized>(value: &T) -> BrainResult<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}

fn pretty_json_line<T: Serialize>(value: &T) -> BrainResult<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Digest the canonical representation capture that a protocol must bind.
pub fn representation_capture_sha256(capture: &RepresentationCapture) -> BrainResult<Sha256Digest> {
    Ok(sha256_bytes(&json_bytes(capture)?))
}

/// Validate the generic, sealed representation contract independently of any
/// particular training task or model vocabulary.
pub fn validate_sealed_representation_protocol(
    protocol: &SealedRepresentationProtocol,
) -> BrainResult<()> {
    if protocol.schema != REPRESENTATION_PROTOCOL_SCHEMA
        || protocol.task_labels_used
        || !protocol.probe_vocabulary_overlap.is_empty()
        || protocol.probe_count == 0
        || protocol.layer_count == 0
        || protocol.hidden_dim == 0
        || protocol.raw_dimension_per_observation == 0
        || protocol.sketch_dim == 0
    {
        return invalid("representation_protocol_contract_invalid");
    }
    if protocol.probe_sha256 != protocol.probe_text_sha256 {
        return invalid("representation_protocol_probe_text_digest_mismatch");
    }
    let expected_raw_dimension = protocol
        .probe_count
        .checked_mul(protocol.layer_count)
        .and_then(|value| value.checked_mul(protocol.hidden_dim))
        .ok_or_else(|| BrainError::Invalid("representation_protocol_dimension_overflow".into()))?;
    if expected_raw_dimension != protocol.raw_dimension_per_observation {
        return invalid("representation_protocol_raw_dimension_mismatch");
    }
    Ok(())
}

fn validate_request_contract(
    request: &RepresentationEvidenceInstallRequest,
) -> BrainResult<BTreeMap<ObservationId, RepresentationShift>> {
    if request.schema != REPRESENTATION_EVIDENCE_REQUEST_SCHEMA
        || request.capture.schema != REPRESENTATION_CAPTURE_SCHEMA
        || request.capture.observations.is_empty()
        || request.installations.is_empty()
    {
        return invalid("representation_evidence_request_contract_invalid");
    }
    validate_sealed_representation_protocol(&request.protocol)?;
    let capture_sha = representation_capture_sha256(&request.capture)?;
    if request.protocol.source_representation_sha256 != capture_sha {
        return integrity("representation_capture_protocol_digest_mismatch");
    }

    let mut shifts = BTreeMap::new();
    for shift in &request.capture.observations {
        if shift.raw_dimension != request.protocol.raw_dimension_per_observation
            || shift.shift.len() as u64 != request.protocol.sketch_dim
            || shift.shift.iter().any(|value| !value.is_finite())
            || shifts
                .insert(shift.observation_id.clone(), shift.clone())
                .is_some()
        {
            return invalid("representation_capture_observation_invalid");
        }
    }

    let mut target_ids = BTreeSet::new();
    let mut source_paths = BTreeSet::new();
    let mut destination_paths = BTreeSet::new();
    for target in &request.installations {
        if target.source_observation_path.trim().is_empty()
            || target.destination_observation_path.trim().is_empty()
            || !target_ids.insert(target.observation_id.clone())
            || !source_paths.insert(target.source_observation_path.clone())
            || !destination_paths.insert(target.destination_observation_path.clone())
        {
            return invalid("representation_installation_target_invalid");
        }
    }
    if target_ids != shifts.keys().cloned().collect::<BTreeSet<_>>() {
        return invalid("representation_capture_target_id_set_mismatch");
    }
    Ok(shifts)
}

fn installed_destination_under_root(root: &Path, raw: &Path) -> BrainResult<PathBuf> {
    let relative = root_relative_path(root, raw)?;
    let path = root.join(relative);
    let permitted = root.join(INSTALLED_OBSERVATIONS_RELATIVE);
    let active_observations = root.join("state/observations");
    if !path.starts_with(&permitted)
        || path.starts_with(&active_observations)
        || path.extension().and_then(|extension| extension.to_str()) != Some("json")
    {
        return invalid("representation_evidence_destination_not_staged_observation_json");
    }
    Ok(path)
}

fn protocol_bytes(protocol: &SealedRepresentationProtocol) -> BrainResult<Vec<u8>> {
    // The protocol digest is over the exact immutable JSON bytes written to
    // `state/representation_protocols/by-sha`, which is what the engine loads.
    json_bytes(protocol)
}

fn persist_protocol(
    root: &Path,
    protocol: &SealedRepresentationProtocol,
) -> BrainResult<RepresentationProtocolDigest> {
    let bytes = protocol_bytes(protocol)?;
    let digest = RepresentationProtocolDigest::from(sha256_bytes(&bytes));
    let directory = root.join(PROTOCOLS_RELATIVE);
    // `ensure_private_parent` takes a prospective child path; this concrete
    // protocol filename makes that intent explicit without introducing a
    // fictitious artifact into the persistence protocol.
    ensure_private_parent(root, &directory.join("protocol.json"))?;
    let path = directory.join(format!("{digest}.json"));
    write_or_verify_immutable(root, &path, &bytes)?;
    Ok(digest)
}

fn destination_observation_bytes(
    before: &DeltaObservation,
    artifact: F64ArtifactRef,
    protocol_sha256: &RepresentationProtocolDigest,
) -> BrainResult<Vec<u8>> {
    if before.representation_artifact.is_some() || before.representation_protocol_sha256.is_some() {
        return integrity("representation_evidence_already_installed_on_source");
    }
    let mut after = before.clone();
    after.representation_artifact = Some(artifact);
    after.representation_protocol_sha256 = Some(protocol_sha256.clone());
    let mut semantic_check = after.clone();
    semantic_check.representation_artifact = None;
    semantic_check.representation_protocol_sha256 = None;
    if semantic_check != *before {
        return integrity("representation_evidence_changed_nonrepresentation_semantics");
    }
    pretty_json_line(&after)
}

fn installations_sha256(
    installations: &[InstalledRepresentationEvidence],
) -> BrainResult<Sha256Digest> {
    Ok(sha256_bytes(&json_bytes(installations)?))
}

fn receipt_path(root: &Path, request_sha256: &RepresentationRequestDigest) -> PathBuf {
    root.join(RECEIPTS_RELATIVE)
        .join(format!("{request_sha256}.json"))
}

fn ledger_payload(receipt: &RepresentationEvidenceReceipt) -> Value {
    json!({
        "schema": "cerebro.tidex.representation_evidence_recorded/v1",
        "source_tree_digest": receipt.source_tree_digest,
        "request_sha256": receipt.request_sha256,
        "source_representation_sha256": receipt.source_representation_sha256,
        "representation_protocol_sha256": receipt.representation_protocol_sha256,
        "installations_sha256": receipt.installations_sha256,
        "observation_count": receipt.observation_count,
    })
}

fn verify_reference_under_root(root: &Path, reference: &F64ArtifactRef) -> BrainResult<()> {
    let _ = existing_regular_file_under_root(root, Path::new(&reference.path))?;
    let value = read_f64_artifact(root, reference)?;
    if value.len() as u64 != reference.element_count {
        return integrity("representation_evidence_artifact_reference_mismatch");
    }
    Ok(())
}

fn verify_existing_receipt(
    root: &Path,
    receipt: &RepresentationEvidenceReceipt,
    request: &RepresentationEvidenceInstallRequest,
    request_sha256: &RepresentationRequestDigest,
    capture_sha256: &Sha256Digest,
    protocol_sha256: &RepresentationProtocolDigest,
) -> BrainResult<()> {
    if receipt.schema != REPRESENTATION_EVIDENCE_RECEIPT_SCHEMA
        || receipt.source_tree_digest.as_str() != env!("TIDEX_SOURCE_TREE_DIGEST")
        || receipt.request_sha256 != *request_sha256
        || receipt.source_representation_sha256 != *capture_sha256
        || receipt.representation_protocol_sha256 != *protocol_sha256
        || receipt.observation_count != request.installations.len()
        || receipt.installations.len() != request.installations.len()
        || receipt.installations_sha256 != installations_sha256(&receipt.installations)?
    {
        return integrity("representation_evidence_receipt_contract_mismatch");
    }
    let targets = request
        .installations
        .iter()
        .map(|target| (target.observation_id.as_str(), target))
        .collect::<BTreeMap<_, _>>();
    for installed in &receipt.installations {
        let target = targets
            .get(installed.observation_id.as_str())
            .ok_or_else(|| {
                BrainError::Integrity("representation_evidence_receipt_unknown_id".into())
            })?;
        if installed.source_observation_path != target.source_observation_path
            || installed.source_observation_sha256 != target.source_observation_sha256
            || installed.destination_observation_path != target.destination_observation_path
        {
            return integrity("representation_evidence_receipt_target_mismatch");
        }
        let destination = installed_destination_under_root(
            root,
            Path::new(&installed.destination_observation_path),
        )?;
        PrivateFileReference::new(
            destination,
            installed.destination_observation_sha256.as_digest().clone(),
        )
        .read_verified_bounded(root, MAX_REPRESENTATION_AUTHORITY_JSON_BYTES)
        .map_err(|_| {
            BrainError::Integrity("representation_evidence_receipt_destination_mismatch".into())
        })?;
        verify_reference_under_root(root, &installed.representation_artifact)?;
    }
    let protocol_path = root
        .join(PROTOCOLS_RELATIVE)
        .join(format!("{protocol_sha256}.json"));
    PrivateFileReference::new(protocol_path, protocol_sha256.as_digest().clone())
        .read_verified_bounded(root, MAX_REPRESENTATION_AUTHORITY_JSON_BYTES)
        .map_err(|_| {
            BrainError::Integrity("representation_evidence_receipt_protocol_mismatch".into())
        })?;
    let event = ledger::find_v2_event_by_payload_string(
        root,
        LEDGER_KIND,
        "request_sha256",
        request_sha256.as_str(),
    )?
    .ok_or_else(|| {
        BrainError::Integrity("representation_evidence_receipt_ledger_missing".into())
    })?;
    if event.event_hash != receipt.ledger_event_hash || event.payload()? != ledger_payload(receipt)
    {
        return integrity("representation_evidence_receipt_ledger_mismatch");
    }
    Ok(())
}

fn record_from_payload_path_at_root(
    root: &Path,
    source_payload_path: &Path,
) -> BrainResult<RepresentationEvidenceReceipt> {
    let payload_bytes = read_existing_private_file_bounded(
        root,
        source_payload_path,
        MAX_REPRESENTATION_AUTHORITY_JSON_BYTES,
    )?;
    let request_sha256 = RepresentationRequestDigest::from(sha256_bytes(&payload_bytes));
    let request: RepresentationEvidenceInstallRequest = serde_json::from_slice(&payload_bytes)?;
    let shifts = validate_request_contract(&request)?;
    let capture_sha256 = representation_capture_sha256(&request.capture)?;
    let protocol_sha256 =
        RepresentationProtocolDigest::from(sha256_bytes(&protocol_bytes(&request.protocol)?));

    let receipt_path = receipt_path(root, &request_sha256);
    if receipt_path.exists() {
        let receipt_raw = read_existing_private_file_bounded(
            root,
            &receipt_path,
            MAX_REPRESENTATION_AUTHORITY_JSON_BYTES,
        )?;
        let receipt: RepresentationEvidenceReceipt = serde_json::from_slice(&receipt_raw)?;
        verify_existing_receipt(
            root,
            &receipt,
            &request,
            &request_sha256,
            &capture_sha256,
            &protocol_sha256,
        )?;
        return Ok(receipt);
    }

    let persisted_protocol_sha256 = persist_protocol(root, &request.protocol)?;
    if persisted_protocol_sha256 != protocol_sha256 {
        return integrity("representation_evidence_protocol_persistence_digest_mismatch");
    }

    let mut installed = Vec::with_capacity(request.installations.len());
    for target in &request.installations {
        let source_reference = PrivateFileReference::new(
            PathBuf::from(&target.source_observation_path),
            target.source_observation_sha256.as_digest().clone(),
        );
        let source_raw = source_reference
            .read_verified_bounded(root, MAX_REPRESENTATION_AUTHORITY_JSON_BYTES)
            .map_err(|_| {
                BrainError::Integrity(
                    "representation_evidence_source_observation_digest_mismatch".into(),
                )
            })?;
        let before: DeltaObservation = serde_json::from_slice(&source_raw)?;
        if before.observation_id != target.observation_id {
            return integrity("representation_evidence_source_observation_id_mismatch");
        }
        let shift = shifts
            .get(&target.observation_id)
            .ok_or_else(|| BrainError::Integrity("representation_evidence_shift_missing".into()))?;
        let artifact = ArtifactWriteAuthority::for_internal_root(root)?
            .create_content_addressed_f64(&shift.shift)?;
        if artifact.element_count != request.protocol.sketch_dim {
            return integrity("representation_evidence_artifact_dimension_mismatch");
        }
        let destination = installed_destination_under_root(
            root,
            Path::new(&target.destination_observation_path),
        )?;
        ensure_private_parent(root, &destination)?;
        let destination_bytes =
            destination_observation_bytes(&before, artifact.clone(), &protocol_sha256)?;
        write_or_verify_immutable(root, &destination, &destination_bytes)?;
        installed.push(InstalledRepresentationEvidence {
            observation_id: target.observation_id.clone(),
            source_observation_path: target.source_observation_path.clone(),
            source_observation_sha256: target.source_observation_sha256.clone(),
            destination_observation_path: target.destination_observation_path.clone(),
            destination_observation_sha256: ObservationRecordDigest::from(sha256_bytes(
                &destination_bytes,
            )),
            representation_artifact: artifact,
        });
    }
    installed.sort_by(|left, right| left.observation_id.cmp(&right.observation_id));
    let installations_sha256 = installations_sha256(&installed)?;
    let mut receipt = RepresentationEvidenceReceipt {
        schema: REPRESENTATION_EVIDENCE_RECEIPT_SCHEMA.to_string(),
        source_tree_digest: Sha256Digest::parse(env!("TIDEX_SOURCE_TREE_DIGEST"))?,
        request_sha256: request_sha256.clone(),
        source_representation_sha256: capture_sha256,
        representation_protocol_sha256: protocol_sha256,
        installations_sha256,
        observation_count: installed.len(),
        installations: installed,
        ledger_event_hash: String::new(),
    };
    let expected_payload = ledger_payload(&receipt);
    let event = if let Some(existing) = ledger::find_v2_event_by_payload_string(
        root,
        LEDGER_KIND,
        "request_sha256",
        request_sha256.as_str(),
    )? {
        if existing.payload()? != expected_payload {
            return integrity("representation_evidence_ledger_request_collision");
        }
        existing
    } else {
        ledger::append(root, LEDGER_KIND, expected_payload)?
    };
    receipt.ledger_event_hash = event.event_hash;
    ensure_private_parent(root, &receipt_path)?;
    write_or_verify_immutable(root, &receipt_path, &pretty_json_line(&receipt)?)?;
    verify_existing_receipt(
        root,
        &receipt,
        &request,
        &request_sha256,
        &receipt.source_representation_sha256,
        &receipt.representation_protocol_sha256,
    )?;
    Ok(receipt)
}

/// Record immutable representation evidence from a strict payload under the
/// private TIDE-X root. This is the public entry point for verified aperture
/// producers; it intentionally refuses a root other than the configured
/// private authority.
pub fn record_representation_evidence(
    root: impl AsRef<Path>,
    source_payload_path: impl AsRef<Path>,
) -> BrainResult<RepresentationEvidenceReceipt> {
    let root = verify_private_root(root.as_ref())?;
    record_from_payload_path_at_root(&root, source_payload_path.as_ref())
}

/// Load a representation-evidence receipt as a reusable finalization input.
///
/// The source install request need not remain as a mutable authority after it
/// has been recorded.  This loader instead verifies the immutable receipt,
/// its ledger event, protocol, every source/destination observation pair, and
/// every representation artifact. It is therefore safe for a later TIDE-X
/// finalizer to consume only the receipt path.
fn load_verified_representation_evidence_receipt_at_root(
    root: &Path,
    raw_receipt_path: &Path,
) -> BrainResult<RepresentationEvidenceReceipt> {
    let receipt_raw = read_existing_private_file_bounded(
        root,
        raw_receipt_path,
        MAX_REPRESENTATION_AUTHORITY_JSON_BYTES,
    )?;
    let loaded_receipt_path = raw_receipt_path.to_path_buf();
    let receipt: RepresentationEvidenceReceipt = serde_json::from_slice(&receipt_raw)?;
    if receipt.schema != REPRESENTATION_EVIDENCE_RECEIPT_SCHEMA
        || receipt.source_tree_digest.as_str() != env!("TIDEX_SOURCE_TREE_DIGEST")
        || receipt.observation_count == 0
        || receipt.observation_count != receipt.installations.len()
        || receipt.installations_sha256 != installations_sha256(&receipt.installations)?
        || loaded_receipt_path != receipt_path(root, &receipt.request_sha256)
    {
        return integrity("representation_evidence_finalization_receipt_contract_invalid");
    }

    let protocol_path = root
        .join(PROTOCOLS_RELATIVE)
        .join(format!("{}.json", receipt.representation_protocol_sha256));
    let protocol_raw = PrivateFileReference::new(
        protocol_path,
        receipt.representation_protocol_sha256.as_digest().clone(),
    )
    .read_verified_bounded(root, MAX_REPRESENTATION_AUTHORITY_JSON_BYTES)
    .map_err(|_| {
        BrainError::Integrity(
            "representation_evidence_finalization_protocol_digest_mismatch".into(),
        )
    })?;
    let protocol: SealedRepresentationProtocol = serde_json::from_slice(&protocol_raw)?;
    validate_sealed_representation_protocol(&protocol)?;
    if protocol.source_representation_sha256 != receipt.source_representation_sha256
        || sha256_bytes(&protocol_bytes(&protocol)?)
            != *receipt.representation_protocol_sha256.as_digest()
    {
        return integrity("representation_evidence_finalization_protocol_contract_mismatch");
    }

    let mut observation_ids = BTreeSet::new();
    let mut source_paths = BTreeSet::new();
    let mut destination_paths = BTreeSet::new();
    let mut previous_id = None::<ObservationId>;
    for installation in &receipt.installations {
        if !observation_ids.insert(installation.observation_id.clone())
            || !source_paths.insert(installation.source_observation_path.clone())
            || !destination_paths.insert(installation.destination_observation_path.clone())
            || previous_id
                .as_ref()
                .is_some_and(|previous| previous >= &installation.observation_id)
        {
            return integrity("representation_evidence_finalization_installation_identity_invalid");
        }
        previous_id = Some(installation.observation_id.clone());

        let source_reference = PrivateFileReference::new(
            PathBuf::from(&installation.source_observation_path),
            installation.source_observation_sha256.as_digest().clone(),
        );
        let source_raw = source_reference
            .read_verified_bounded(root, MAX_REPRESENTATION_AUTHORITY_JSON_BYTES)
            .map_err(|_| {
                BrainError::Integrity(
                    "representation_evidence_finalization_observation_digest_mismatch".into(),
                )
            })?;
        let destination_path = installed_destination_under_root(
            root,
            Path::new(&installation.destination_observation_path),
        )?;
        let destination_raw = PrivateFileReference::new(
            destination_path,
            installation
                .destination_observation_sha256
                .as_digest()
                .clone(),
        )
        .read_verified_bounded(root, MAX_REPRESENTATION_AUTHORITY_JSON_BYTES)
        .map_err(|_| {
            BrainError::Integrity(
                "representation_evidence_finalization_observation_digest_mismatch".into(),
            )
        })?;
        let source: DeltaObservation = serde_json::from_slice(&source_raw)?;
        let destination: DeltaObservation = serde_json::from_slice(&destination_raw)?;
        if source.observation_id != installation.observation_id
            || destination.observation_id != installation.observation_id
            || destination.representation_artifact.as_ref()
                != Some(&installation.representation_artifact)
            || destination.representation_protocol_sha256.as_deref()
                != Some(receipt.representation_protocol_sha256.as_str())
        {
            return integrity("representation_evidence_finalization_observation_identity_mismatch");
        }
        verify_reference_under_root(root, &installation.representation_artifact)?;
        let mut without_representation = destination;
        without_representation.representation_artifact = None;
        without_representation.representation_protocol_sha256 = None;
        if without_representation != source {
            return integrity("representation_evidence_finalization_nonrepresentation_mutation");
        }
    }

    let event = ledger::find_v2_event_by_payload_string(
        root,
        LEDGER_KIND,
        "request_sha256",
        receipt.request_sha256.as_str(),
    )?
    .ok_or_else(|| {
        BrainError::Integrity("representation_evidence_finalization_ledger_missing".into())
    })?;
    if event.event_hash != receipt.ledger_event_hash || event.payload()? != ledger_payload(&receipt)
    {
        return integrity("representation_evidence_finalization_ledger_mismatch");
    }
    Ok(receipt)
}

/// Public fail-closed receipt loader for later authoritative consumers such as
/// TIDE-X finalization. The receipt path must be absolute and confined to the
/// CEREBRO private root.
pub fn load_verified_representation_evidence_receipt(
    root: impl AsRef<Path>,
    receipt_path: impl AsRef<Path>,
) -> BrainResult<RepresentationEvidenceReceipt> {
    let root = verify_private_root(root.as_ref())?;
    load_verified_representation_evidence_receipt_at_root(&root, receipt_path.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::secure_dir;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn digest(label: &str) -> Sha256Digest {
        sha256_bytes(label.as_bytes())
    }

    fn observation_digest(label: &str) -> ObservationRecordDigest {
        ObservationRecordDigest::from(digest(label))
    }

    fn protocol(capture: &RepresentationCapture) -> SealedRepresentationProtocol {
        SealedRepresentationProtocol {
            schema: REPRESENTATION_PROTOCOL_SCHEMA.to_string(),
            source_representation_sha256: representation_capture_sha256(capture).unwrap(),
            probe_sha256: digest("sealed-probe"),
            probe_text_sha256: digest("sealed-probe"),
            forbidden_vocabulary_sha256: digest("forbidden-vocabulary"),
            task_labels_used: false,
            probe_vocabulary_overlap: Vec::new(),
            probe_count: 2,
            layer_count: 2,
            hidden_dim: 2,
            raw_dimension_per_observation: 8,
            sketch_dim: 3,
            sketch_seed: 17,
        }
    }

    fn observation(id: &str) -> DeltaObservation {
        DeltaObservation {
            observation_id: ObservationId::parse(id).unwrap(),
            from_checkpoint: "base".to_string(),
            to_checkpoint: format!("checkpoint-{id}"),
            generation: 1,
            delta: vec![0.1, -0.2, 0.3],
            functional_response: vec![0.2],
            confounders: Vec::new(),
            reliability: 0.9,
            independence_group: "aperture-a".to_string(),
            experiment_lineage: Default::default(),
            dense_artifact: None,
            parameter_layout_sha256: None,
            representation_artifact: None,
            representation_protocol_sha256: None,
            provenance_digest: crate::digest::ProvenanceDigest::from(digest(id)),
        }
    }

    #[test]
    fn sealed_protocol_rejects_labels_and_raw_dimension_mismatch() {
        let capture = RepresentationCapture {
            schema: REPRESENTATION_CAPTURE_SCHEMA.to_string(),
            observations: vec![RepresentationShift {
                observation_id: ObservationId::parse("obs-a").unwrap(),
                raw_dimension: 8,
                shift: vec![0.1, 0.2, 0.3],
            }],
        };
        let mut sealed = protocol(&capture);
        sealed.task_labels_used = true;
        assert!(validate_sealed_representation_protocol(&sealed).is_err());
        sealed.task_labels_used = false;
        sealed.raw_dimension_per_observation = 7;
        assert!(validate_sealed_representation_protocol(&sealed).is_err());
    }

    #[test]
    fn request_contract_rejects_capture_protocol_digest_mismatch() {
        let capture = RepresentationCapture {
            schema: REPRESENTATION_CAPTURE_SCHEMA.to_string(),
            observations: vec![RepresentationShift {
                observation_id: ObservationId::parse("obs-a").unwrap(),
                raw_dimension: 8,
                shift: vec![0.1, 0.2, 0.3],
            }],
        };
        let mut sealed = protocol(&capture);
        sealed.source_representation_sha256 = digest("wrong-capture");
        let fixture_root = std::env::temp_dir().join(format!(
            "tidex-representation-contract-fixture-{}",
            std::process::id()
        ));
        let request = RepresentationEvidenceInstallRequest {
            schema: REPRESENTATION_EVIDENCE_REQUEST_SCHEMA.to_string(),
            protocol: sealed,
            capture,
            installations: vec![RepresentationEvidenceInstallTarget {
                observation_id: ObservationId::parse("obs-a").unwrap(),
                source_observation_path: fixture_root
                    .join("source/x.json")
                    .to_string_lossy()
                    .into_owned(),
                source_observation_sha256: observation_digest("source"),
                destination_observation_path: fixture_root
                    .join(INSTALLED_OBSERVATIONS_RELATIVE)
                    .join("obs-a.json")
                    .to_string_lossy()
                    .into_owned(),
            }],
        };
        assert!(validate_request_contract(&request).is_err());
    }

    #[test]
    fn records_staged_observation_without_touching_source_or_active_state() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cerebro-representation-evidence-test-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        secure_dir(&root).unwrap();
        let source_dir = root.join("state/staged-input");
        fs::create_dir_all(&source_dir).unwrap();
        secure_dir(&root.join("state")).unwrap();
        secure_dir(&source_dir).unwrap();
        let source_path = source_dir.join("obs-a.json");
        let before = observation("obs-a");
        let source_bytes = pretty_json_line(&before).unwrap();
        write_or_verify_immutable(&root, &source_path, &source_bytes).unwrap();
        let capture = RepresentationCapture {
            schema: REPRESENTATION_CAPTURE_SCHEMA.to_string(),
            observations: vec![RepresentationShift {
                observation_id: ObservationId::parse("obs-a").unwrap(),
                raw_dimension: 8,
                shift: vec![0.1, -0.2, 0.3],
            }],
        };
        let destination = root
            .join(INSTALLED_OBSERVATIONS_RELATIVE)
            .join("obs-a-installed.json");
        let request = RepresentationEvidenceInstallRequest {
            schema: REPRESENTATION_EVIDENCE_REQUEST_SCHEMA.to_string(),
            protocol: protocol(&capture),
            capture,
            installations: vec![RepresentationEvidenceInstallTarget {
                observation_id: ObservationId::parse("obs-a").unwrap(),
                source_observation_path: source_path.to_string_lossy().into_owned(),
                source_observation_sha256: ObservationRecordDigest::from(sha256_bytes(
                    &source_bytes,
                )),
                destination_observation_path: destination.to_string_lossy().into_owned(),
            }],
        };
        let payload_path = root.join("state/install-request.json");
        let payload_bytes = pretty_json_line(&request).unwrap();
        write_or_verify_immutable(&root, &payload_path, &payload_bytes).unwrap();

        let receipt = record_from_payload_path_at_root(&root, &payload_path).unwrap();
        assert_eq!(receipt.observation_count, 1);
        assert_eq!(fs::read(&source_path).unwrap(), source_bytes);
        assert!(!root.join("state/observations").exists());
        let installed: DeltaObservation =
            serde_json::from_slice(&fs::read(&destination).unwrap()).unwrap();
        assert_eq!(installed.observation_id, before.observation_id);
        assert!(installed.representation_artifact.is_some());
        assert_eq!(
            installed.representation_protocol_sha256.as_deref(),
            Some(receipt.representation_protocol_sha256.as_str())
        );
        assert!(ledger::contains_event_hash(&root, &receipt.ledger_event_hash).unwrap());
        assert_eq!(
            record_from_payload_path_at_root(&root, &payload_path).unwrap(),
            receipt
        );
        assert_eq!(
            load_verified_representation_evidence_receipt_at_root(
                &root,
                &receipt_path(&root, &receipt.request_sha256),
            )
            .unwrap(),
            receipt
        );

        // A finalizer must never trust an on-disk receipt merely because it
        // occupies the canonical filename: its own contents and ledger
        // binding are re-verified on every load.
        let canonical_receipt = receipt_path(&root, &receipt.request_sha256);
        let mut tampered: RepresentationEvidenceReceipt =
            serde_json::from_slice(&fs::read(&canonical_receipt).unwrap()).unwrap();
        tampered.observation_count += 1;
        fs::write(&canonical_receipt, pretty_json_line(&tampered).unwrap()).unwrap();
        assert!(
            load_verified_representation_evidence_receipt_at_root(&root, &canonical_receipt,)
                .is_err()
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn finalization_loader_rejects_receipt_outside_root() {
        let root = std::env::temp_dir().join(format!(
            "cerebro-representation-evidence-root-test-{}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        let outside = std::env::temp_dir().join(format!(
            "cerebro-representation-evidence-outside-test-{}",
            std::process::id()
        ));
        fs::write(&outside, b"{}\n").unwrap();
        assert!(load_verified_representation_evidence_receipt_at_root(&root, &outside).is_err());
        fs::remove_file(&outside).unwrap();
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn strict_wire_types_reject_path_ids_raw_digests_and_unknown_fields() {
        assert!(serde_json::from_value::<RepresentationShift>(json!({
            "observation_id": "../obs-a",
            "raw_dimension": 1,
            "shift": [0.0]
        }))
        .is_err());
        assert!(
            serde_json::from_value::<SealedRepresentationProtocol>(json!({
                "schema": REPRESENTATION_PROTOCOL_SCHEMA,
                "source_representation_sha256": "not-a-digest",
                "probe_sha256": "0".repeat(64),
                "probe_text_sha256": "0".repeat(64),
                "forbidden_vocabulary_sha256": "0".repeat(64),
                "task_labels_used": false,
                "probe_vocabulary_overlap": [],
                "probe_count": 1,
                "layer_count": 1,
                "hidden_dim": 1,
                "raw_dimension_per_observation": 1,
                "sketch_dim": 1,
                "sketch_seed": 1
            }))
            .is_err()
        );
        assert!(serde_json::from_value::<RepresentationCapture>(json!({
            "schema": REPRESENTATION_CAPTURE_SCHEMA,
            "observations": [],
            "unexpected": true
        }))
        .is_err());
    }
}
