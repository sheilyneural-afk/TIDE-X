//! Shared engine support: confined IO, digests, keys and receipt loaders.

use super::*;

pub(super) fn sha256_bytes(bytes: &[u8]) -> String {
    Sha256Digest::digest_bytes(bytes).into_string()
}

pub(super) fn serialize_pretty_line<T: Serialize>(value: &T) -> BrainResult<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub(super) fn read_private_json<T: DeserializeOwned>(
    root: &Path,
    path: &Path,
    max_bytes: u64,
) -> BrainResult<T> {
    let bytes = read_untrusted_private_file_bounded(root, path, max_bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub(super) fn file_sha256(path: &Path) -> BrainResult<String> {
    crate::artifact::sha256_file(path).map(Into::into)
}

/// Boolean health predicate for a content-addressed private artifact. Any
/// path, symlink, read, or digest failure is deliberately an unhealthy value.
pub(super) fn private_file_digest_matches(root: &Path, path: &Path, expected_sha256: &str) -> bool {
    existing_regular_file_under_root(root, path)
        .and_then(|verified| {
            if file_sha256(&verified)? == expected_sha256 {
                Ok(())
            } else {
                Err(BrainError::Integrity("private_file_digest_mismatch".into()))
            }
        })
        .is_ok()
}

pub(super) fn write_new_private(root: &Path, path: &Path, bytes: &[u8]) -> BrainResult<()> {
    let expected = Sha256Digest::digest_bytes(bytes);
    if write_or_verify_immutable(root, path, bytes)? != expected {
        return Err(BrainError::Integrity(
            "private_immutable_write_digest_mismatch".into(),
        ));
    }
    Ok(())
}

pub(super) fn write_immutable_exact(
    root: &Path,
    path: &Path,
    bytes: &[u8],
    expected_sha256: &str,
) -> BrainResult<()> {
    if sha256_bytes(bytes) != expected_sha256 {
        return Err(BrainError::Integrity(
            "private_immutable_expected_digest_mismatch".into(),
        ));
    }
    write_new_private(root, path, bytes)
}

/// Replace only an explicitly mutable current pointer. Historical
/// content-addressed artifacts use `write_new_private` and are never
/// overwritten once their identity is assigned.
pub(super) fn replace_private_pointer_exact(
    root: &Path,
    path: &Path,
    bytes: &[u8],
    expected_sha: &str,
) -> BrainResult<()> {
    let relative = root_relative_path(root, path)?;
    let permitted = [
        Path::new("state/memory/current.json"),
        Path::new("state/shadow_skill_bank.json"),
        Path::new("state/skill_bank.json"),
        Path::new("state/sleep_state.json"),
    ];
    if !permitted.contains(&relative.as_path()) {
        return Err(BrainError::Integrity(
            "transaction_pointer_target_not_allowlisted".into(),
        ));
    }
    let expected = Sha256Digest::parse(expected_sha)?;
    let installed = replace_private_file_atomic(root, path, bytes, Some(&expected))?;
    if installed != expected {
        return Err(BrainError::Integrity(
            "transaction_installed_digest_mismatch".into(),
        ));
    }
    Ok(())
}

pub(super) fn valid_digest(value: &str) -> bool {
    Sha256Digest::is_valid_str(value)
}

pub(super) fn persistent_not_applicable_reason(error: &BrainError) -> Option<String> {
    let BrainError::Numerical(reason) = error else {
        return None;
    };
    match reason.as_str() {
        "persistent_coherence_structure_not_identifiable"
        | "persistent_cluster_weight_zero"
        | "persistent_cluster_centroid_degenerate"
        | "persistent_functional_weight_zero"
        | "persistent_cv_no_cluster_assignment"
        | "persistent_total_energy_degenerate"
        | "persistent_cluster_energy_degenerate"
        | "persistent_field_geometry_rank_deficient"
        | "persistent_field_geometry_empty" => Some(reason.clone()),
        _ => None,
    }
}

impl ControllerInvocation {
    /// Validate the caller-supplied portion of a controller action. The
    /// controller coefficients and functional response are intentionally not
    /// part of this wire contract: both are rederived under engine authority.
    pub fn validate(&self) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.controller_invocation/v1"
            || self.state_before.is_empty()
            || self.state_before.iter().any(|value| !value.is_finite())
        {
            return Err(BrainError::Invalid(
                "controller_invocation_contract_invalid".into(),
            ));
        }
        Ok(())
    }
}

pub(super) fn controller_execution_receipt_path(root: &Path, receipt_sha256: &str) -> PathBuf {
    root.join("state/controller_executions/by-sha")
        .join(format!("{receipt_sha256}.json"))
}

pub(super) fn controller_execution_ledger_binding(
    root: &Path,
    receipt_sha256: &str,
    receipt: &ControllerExecutionReceipt,
) -> BrainResult<String> {
    let event = ledger::find_v2_event_by_payload_string(
        root,
        "controller_execution_receipt",
        "receipt_sha256",
        receipt_sha256,
    )?
    .ok_or_else(|| BrainError::Integrity("controller_execution_receipt_ledger_missing".into()))?;
    let payload = event.payload()?;
    if payload.get("schema").and_then(Value::as_str)
        != Some("cerebro.tidex.controller_execution_ledger_binding/v1")
        || payload.get("receipt_sha256").and_then(Value::as_str) != Some(receipt_sha256)
        || payload.get("session_id").and_then(Value::as_str) != Some(receipt.session_id.as_str())
        || payload.get("invocation_sha256").and_then(Value::as_str)
            != Some(receipt.invocation_sha256.as_str())
        || payload
            .get("controller_receipt_sha256")
            .and_then(Value::as_str)
            != Some(receipt.controller_receipt_sha256.as_str())
        || payload.get("state_before_sha256").and_then(Value::as_str)
            != Some(receipt.state_before_sha256.as_str())
        || payload
            .get("promoted_observation_semantic_sha256")
            .and_then(Value::as_str)
            != Some(receipt.promoted_observation_semantic_sha256.as_str())
        || payload
            .get("governed_composition_receipt_sha256")
            .and_then(Value::as_str)
            != Some(receipt.governed_composition_receipt_sha256.as_str())
    {
        return Err(BrainError::Integrity(
            "controller_execution_receipt_ledger_payload_mismatch".into(),
        ));
    }
    Ok(event.event_hash)
}

pub(super) fn persist_controller_execution(
    root: &Path,
    receipt: ControllerExecutionReceipt,
) -> BrainResult<RecordedControllerExecution> {
    let by_sha = root.join("state/controller_executions/by-sha");
    ensure_private_directory(root, &by_sha)?;
    let bytes = serialize_pretty_line(&receipt)?;
    let receipt_sha256 = sha256_bytes(&bytes);
    let receipt_path = controller_execution_receipt_path(root, &receipt_sha256);
    match fs::symlink_metadata(&receipt_path) {
        Ok(_) => {
            let path = existing_regular_file_under_root(root, &receipt_path)?;
            let bytes = fs::read(path)?;
            if sha256_bytes(&bytes) != receipt_sha256 {
                return Err(BrainError::Integrity(
                    "controller_execution_receipt_artifact_invalid".into(),
                ));
            }
            let existing: ControllerExecutionReceipt = serde_json::from_slice(&bytes)?;
            if existing != receipt {
                return Err(BrainError::Integrity(
                    "controller_execution_receipt_digest_collision".into(),
                ));
            }
            let ledger_event_hash =
                controller_execution_ledger_binding(root, &receipt_sha256, &existing)?;
            Ok(RecordedControllerExecution {
                receipt_path: receipt_path.to_string_lossy().into_owned(),
                receipt_sha256,
                ledger_event_hash,
                receipt: existing,
            })
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            write_new_private(root, &receipt_path, &bytes)?;
            let event = if let Some(existing) = ledger::find_v2_event_by_payload_string(
                root,
                "controller_execution_receipt",
                "receipt_sha256",
                &receipt_sha256,
            )? {
                let existing_hash =
                    controller_execution_ledger_binding(root, &receipt_sha256, &receipt)?;
                if existing.event_hash != existing_hash {
                    return Err(BrainError::Integrity(
                        "controller_execution_receipt_ledger_changed".into(),
                    ));
                }
                existing
            } else {
                ledger::append(
                    root,
                    "controller_execution_receipt",
                    json!({
                        "schema":"cerebro.tidex.controller_execution_ledger_binding/v1",
                        "receipt_sha256":&receipt_sha256,
                        "session_id":&receipt.session_id,
                        "invocation_sha256":&receipt.invocation_sha256,
                        "controller_receipt_sha256":&receipt.controller_receipt_sha256,
                        "state_before_sha256":&receipt.state_before_sha256,
                        "promoted_observation_semantic_sha256":&receipt.promoted_observation_semantic_sha256,
                        "governed_composition_receipt_sha256":&receipt.governed_composition_receipt_sha256,
                    }),
                )?
            };
            let ledger_event_hash =
                controller_execution_ledger_binding(root, &receipt_sha256, &receipt)?;
            if event.event_hash != ledger_event_hash {
                return Err(BrainError::Integrity(
                    "controller_execution_receipt_ledger_event_changed".into(),
                ));
            }
            Ok(RecordedControllerExecution {
                receipt_path: receipt_path.to_string_lossy().into_owned(),
                receipt_sha256,
                ledger_event_hash,
                receipt,
            })
        }
        Err(error) => Err(error.into()),
    }
}

pub(super) fn verify_controller_execution_receipt(
    root: &Path,
    recorded: &RecordedControllerExecution,
) -> BrainResult<()> {
    let invocation = &recorded.receipt.invocation;
    invocation.validate()?;
    if !valid_digest(&recorded.receipt_sha256)
        || !valid_digest(&recorded.ledger_event_hash)
        || recorded.receipt.schema != "cerebro.tidex.controller_execution_receipt/v1"
        || recorded.receipt_path
            != controller_execution_receipt_path(root, &recorded.receipt_sha256).to_string_lossy()
        || recorded.receipt.session_id != invocation.session_id
        || recorded.receipt.invocation_sha256 != digest_json(invocation)?
        || recorded.receipt.state_before_sha256 != digest_json(&invocation.state_before)?
        || recorded.receipt.promoted_observation_semantic_sha256
            != invocation.promoted_observation_semantic_sha256
    {
        return Err(BrainError::Integrity(
            "controller_execution_receipt_contract_invalid".into(),
        ));
    }
    let receipt_path = existing_regular_file_under_root(
        root,
        &controller_execution_receipt_path(root, &recorded.receipt_sha256),
    )?;
    let receipt_bytes = fs::read(&receipt_path)?;
    if sha256_bytes(&receipt_bytes) != recorded.receipt_sha256
        || serde_json::from_slice::<ControllerExecutionReceipt>(&receipt_bytes)? != recorded.receipt
    {
        return Err(BrainError::Integrity(
            "controller_execution_receipt_bytes_mismatch".into(),
        ));
    }
    let current_controller =
        load_persisted_runtime_learned_controller(root, invocation.session_id.as_str())?;
    if current_controller.receipt_sha256.as_str()
        != recorded.receipt.controller_receipt_sha256.as_str()
    {
        return Err(BrainError::Integrity(
            "controller_execution_controller_receipt_stale".into(),
        ));
    }
    let governed_path = root
        .join("state/governed_compositions/by-sha")
        .join(format!(
            "{}.json",
            recorded.receipt.governed_composition_receipt_sha256
        ));
    let governed = load_verified_governed_composition_receipt(
        root,
        &governed_path,
        &recorded.receipt.governed_composition_receipt_sha256,
    )?;
    let engine = BrainEngine::open(root, BrainConfig::default())?;
    engine.require_canonical_runtime_config()?;
    let source = engine.load_current_observation_by_semantic_sha256(
        &invocation.promoted_observation_semantic_sha256,
    )?;
    let activation = engine.learned_controller_activation(
        &current_controller.receipt.runtime_controller,
        &invocation.state_before,
        &source.functional_response,
    )?;
    if governed.source_observation_sha256 != invocation.promoted_observation_semantic_sha256
        || governed.requested_activation != activation
    {
        return Err(BrainError::Integrity(
            "controller_execution_governed_composition_binding_invalid".into(),
        ));
    }
    let ledger_event_hash =
        controller_execution_ledger_binding(root, &recorded.receipt_sha256, &recorded.receipt)?;
    if ledger_event_hash != recorded.ledger_event_hash {
        return Err(BrainError::Integrity(
            "controller_execution_receipt_ledger_hash_mismatch".into(),
        ));
    }
    Ok(())
}

pub(super) fn digest_json<T: Serialize>(v: &T) -> BrainResult<String> {
    let bytes = serde_json::to_vec(v)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub(super) fn canonical_observations(observations: &[DeltaObservation]) -> Vec<DeltaObservation> {
    let mut canonical = observations.to_vec();
    canonical.sort_by(|left, right| {
        left.observation_id
            .cmp(&right.observation_id)
            .then_with(|| left.provenance_digest.cmp(&right.provenance_digest))
    });
    canonical
}

pub(super) fn observation_set_digest(
    observations: &[DeltaObservation],
) -> BrainResult<CorpusDigest> {
    let mut digests = observations
        .iter()
        .map(digest_json)
        .collect::<BrainResult<Vec<_>>>()?;
    digests.sort();
    digests.dedup();
    Ok(CorpusDigest::from(Sha256Digest::parse(digest_json(
        &digests,
    )?)?))
}

pub(super) fn attach_evidence_support(
    fields: &mut [SkillField],
    source_mixtures: &[Vec<f64>],
    observations: &[DeltaObservation],
) -> BrainResult<()> {
    if fields.len() != source_mixtures.len() {
        return Err(BrainError::Integrity(
            "field_evidence_support_count_mismatch".into(),
        ));
    }
    for (field, mixture) in fields.iter_mut().zip(source_mixtures) {
        if mixture.len() != observations.len() {
            return Err(BrainError::Integrity(format!(
                "field_evidence_mixture_shape:{}",
                field.skill_id
            )));
        }
        let indices = source_support_indices(mixture)?;
        if indices.is_empty() {
            return Err(BrainError::Integrity(format!(
                "field_evidence_support_empty:{}",
                field.skill_id
            )));
        }
        let mut digests = indices
            .into_iter()
            .map(|index| {
                Ok(ObservationRecordDigest::from(Sha256Digest::parse(
                    digest_json(&observations[index])?,
                )?))
            })
            .collect::<BrainResult<Vec<_>>>()?;
        digests.sort();
        digests.dedup();
        if digests.is_empty() {
            return Err(BrainError::Integrity(format!(
                "field_evidence_support_invalid:{}",
                field.skill_id
            )));
        }
        field.support = digests.len();
        field.evidence_support_digests = digests;
    }
    Ok(())
}

/// Identity for a TIDE-X reconstruction. Experimental producers may create
/// evidence, but their scripts and implementation details are not part of the
/// brain's algorithmic authority.
pub(super) fn analysis_identity(
    config: &BrainConfig,
) -> BrainResult<(SourceTreeDigest, ConfigDigest, AnalysisVersionDigest)> {
    let source_tree_digest = Sha256Digest::parse(env!("TIDEX_SOURCE_TREE_DIGEST"))?;
    if !valid_digest(&source_tree_digest) {
        return Err(BrainError::Integrity("source_tree_digest_invalid".into()));
    }
    let config_digest = Sha256Digest::parse(digest_json(config)?)?;
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:ANALYSIS:v6\0");
    hasher.update(source_tree_digest.as_bytes());
    hasher.update(config_digest.as_bytes());
    hasher.update(b"reconstruction/v7");
    let analysis_version_digest = Sha256Digest::parse(format!("{:x}", hasher.finalize()))?;
    Ok((
        SourceTreeDigest::from(source_tree_digest),
        ConfigDigest::from(config_digest),
        AnalysisVersionDigest::from(analysis_version_digest),
    ))
}

pub(super) fn commit_operation_key(
    batch_digest: &str,
    report: &ReconstructionReport,
    report_sha256: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:COMMIT-OPERATION:v2\0");
    hasher.update(batch_digest.as_bytes());
    hasher.update(report.analysis_version_digest.as_bytes());
    hasher.update(report.config_digest.as_bytes());
    hasher.update(report_sha256.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub(super) fn commit_transaction_payload(
    intent: &CommitTransactionIntent,
    observation_count: usize,
    report: &ReconstructionReport,
) -> Value {
    json!({
        "schema":"cerebro.tidex.commit_transaction/v2",
        "operation_key":intent.operation_key,
        "batch_digest":intent.batch_digest,
        "observation_count":observation_count,
        "observation_digests":intent.observation_digests,
        "report_sha256":intent.report_sha256,
        "report_promotable":intent.report_promotable,
        "generation":intent.generation,
        "memory_sha256":intent.memory_sha256,
        "shadow_bank_sha256":intent.shadow_bank_sha256,
        "prior_shadow_bank_sha256":intent.prior_shadow_bank_sha256,
        "summary":{
            "inverse_mode":report.inverse_mode,
            "selected_rank":report.selected_rank,
            "functional_cv_r2":report.functional_cv_r2,
            "promotion_allowed":report.promotion.allowed,
        }
    })
}

pub(super) fn verify_commit_transaction_ledger_binding(
    event: &ledger::LedgerEvent,
    intent: &CommitTransactionIntent,
    observation_count: usize,
    report: &ReconstructionReport,
) -> BrainResult<()> {
    if event.payload()? != commit_transaction_payload(intent, observation_count, report) {
        return Err(BrainError::Integrity(
            "commit_transaction_ledger_payload_mismatch".into(),
        ));
    }
    Ok(())
}

pub(super) fn sleep_analysis_key(corpus_digest: &str, report: &ReconstructionReport) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:SLEEP-ANALYSIS:v1\0");
    hasher.update(corpus_digest.as_bytes());
    hasher.update(report.analysis_version_digest.as_bytes());
    hasher.update(report.config_digest.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub(super) fn sleep_operation_key(
    analysis_key: &str,
    evidence_sha: Option<&str>,
    certification_status: CertificationStatus,
    active_bank_sha: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:SLEEP-OPERATION:v1\0");
    hasher.update(analysis_key.as_bytes());
    hasher.update(evidence_sha.unwrap_or("none").as_bytes());
    hasher.update(certification_status.as_str().as_bytes());
    hasher.update(active_bank_sha.unwrap_or("none").as_bytes());
    format!("{:x}", hasher.finalize())
}

pub(super) fn required_sleep_state_string<'a>(
    state: &'a Value,
    field: &str,
) -> BrainResult<&'a str> {
    state
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| BrainError::Integrity(format!("sleep_state_{field}_missing_or_invalid")))
}

/// Prove that a sleep receipt is bound to its own immutable ledger event and
/// to the exact state it claims to certify.  A bare ledger hash is never
/// sufficient: it could otherwise name an unrelated valid event.
pub(super) fn verify_sleep_receipt_ledger_binding(
    root: &Path,
    receipt: &SleepReceipt,
    state: &Value,
    current_state_sha256: &str,
) -> BrainResult<()> {
    if receipt.schema != "cerebro.tidex.sleep_receipt/v1"
        || required_sleep_state_string(state, "schema")? != "cerebro.tidex.sleep_state/v5"
        || required_sleep_state_string(state, "operation_key")? != receipt.operation_key
        || required_sleep_state_string(state, "analysis_key")? != receipt.analysis_key
        || required_sleep_state_string(state, "report_sha256")? != receipt.report_sha256.as_str()
        || required_sleep_state_string(state, "memory_digest")? != receipt.memory_sha256.as_str()
        || current_state_sha256 != receipt.sleep_state_sha256
    {
        return Err(BrainError::Integrity(
            "sleep_receipt_state_binding_invalid".into(),
        ));
    }
    for field in [
        "corpus_digest",
        "source_tree_digest",
        "config_digest",
        "analysis_version_digest",
    ] {
        let _ = required_sleep_state_string(state, field)?;
    }
    let expected_bank = serde_json::to_value(&receipt.active_bank_sha256)?;
    let expected_evidence = serde_json::to_value(&receipt.evidence_bundle_sha256)?;
    if state.get("active_bank_sha256") != Some(&expected_bank)
        || state.get("evidence_bundle_sha256") != Some(&expected_evidence)
    {
        return Err(BrainError::Integrity(
            "sleep_receipt_optional_state_binding_invalid".into(),
        ));
    }
    let certification_status =
        CertificationStatus::parse(required_sleep_state_string(state, "certification_status")?)
            .ok_or_else(|| {
                BrainError::Integrity("sleep_state_certification_status_invalid".into())
            })?;
    let evidence_verified = state
        .get("evidence_verification")
        .and_then(|value| value.get("verified"))
        .and_then(Value::as_bool)
        .ok_or_else(|| BrainError::Integrity("sleep_state_evidence_verification_invalid".into()))?;
    let promoted = state
        .get("promoted")
        .and_then(Value::as_bool)
        .ok_or_else(|| BrainError::Integrity("sleep_state_promoted_invalid".into()))?;
    let expected_operation_key = sleep_operation_key(
        &receipt.analysis_key,
        receipt.evidence_bundle_sha256.as_deref(),
        certification_status,
        receipt.active_bank_sha256.as_deref(),
    );
    if receipt.operation_key != expected_operation_key {
        return Err(BrainError::Integrity(
            "sleep_receipt_operation_key_invalid".into(),
        ));
    }
    let event = ledger::find_v2_event_by_payload_string(
        root,
        "sleep_transaction",
        "operation_key",
        &receipt.operation_key,
    )?
    .ok_or_else(|| BrainError::Integrity("sleep_receipt_ledger_event_missing".into()))?;
    let payload = event.payload()?;
    if event.event_hash != receipt.ledger_event_hash
        || payload.get("schema").and_then(Value::as_str)
            != Some("cerebro.tidex.sleep_transaction/v1")
        || payload.get("operation_key").and_then(Value::as_str)
            != Some(receipt.operation_key.as_str())
        || payload.get("analysis_key").and_then(Value::as_str)
            != Some(receipt.analysis_key.as_str())
        || payload.get("report_sha256").and_then(Value::as_str)
            != Some(receipt.report_sha256.as_str())
        || payload.get("memory_sha256").and_then(Value::as_str)
            != Some(receipt.memory_sha256.as_str())
        || payload.get("sleep_state_sha256").and_then(Value::as_str)
            != Some(receipt.sleep_state_sha256.as_str())
        || payload.get("active_bank_sha256") != Some(&expected_bank)
        || payload.get("evidence_bundle_sha256") != Some(&expected_evidence)
        || payload.get("evidence_verified").and_then(Value::as_bool) != Some(evidence_verified)
        || payload.get("certification_status").and_then(Value::as_str)
            != Some(certification_status.as_str())
        || payload.get("promoted").and_then(Value::as_bool) != Some(promoted)
    {
        return Err(BrainError::Integrity(
            "sleep_receipt_ledger_payload_invalid".into(),
        ));
    }
    for field in ["corpus_digest", "analysis_version_digest", "config_digest"] {
        if payload.get(field) != state.get(field) {
            return Err(BrainError::Integrity(format!(
                "sleep_receipt_ledger_{field}_mismatch"
            )));
        }
    }
    Ok(())
}

/// The transaction event is the one append-only proof that authorizes the
/// staged sleep artifacts to become current pointers.  Reusing a same-key
/// event with a different payload would otherwise mutate state before later
/// health checks notice the inconsistency.
pub(super) fn verify_sleep_transaction_ledger_binding(
    event: &ledger::LedgerEvent,
    intent: &SleepTransactionIntent,
) -> BrainResult<()> {
    let expected = json!({
        "schema":"cerebro.tidex.sleep_transaction/v1",
        "operation_key":intent.operation_key,
        "analysis_key":intent.analysis_key,
        "corpus_digest":intent.corpus_digest,
        "analysis_version_digest":intent.analysis_version_digest,
        "config_digest":intent.config_digest,
        "report_sha256":intent.report_sha256,
        "memory_sha256":intent.memory_sha256,
        "active_bank_sha256":intent.active_bank_sha256,
        "sleep_state_sha256":intent.sleep_state_sha256,
        "evidence_bundle_sha256":intent.evidence_bundle_sha256,
        "evidence_verified":intent.evidence_verified,
        "certification_status":intent.certification_status,
        "promoted":intent.promoted,
    });
    if event.payload()? != expected {
        return Err(BrainError::Integrity(
            "sleep_transaction_ledger_payload_mismatch".into(),
        ));
    }
    Ok(())
}

pub(super) fn learning_finalization_operation_key(
    learning_finalization_input_sha256: &Sha256Digest,
    prior_corpus_digest: &Sha256Digest,
    new_corpus_digest: &Sha256Digest,
) -> BrainResult<Sha256Digest> {
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:LEARNING-FINALIZATION:v1\0");
    hasher.update(learning_finalization_input_sha256.as_str().as_bytes());
    hasher.update(prior_corpus_digest.as_str().as_bytes());
    hasher.update(new_corpus_digest.as_str().as_bytes());
    Sha256Digest::parse(format!("{:x}", hasher.finalize()))
}

pub(super) struct GovernedCompositionOperation<'a> {
    pub(super) report_sha256: &'a str,
    pub(super) active_bank_sha256: &'a str,
    pub(super) evidence_bundle_sha256: &'a str,
    pub(super) causal_credit_sha256: &'a str,
    pub(super) parameter_layout_artifact_sha256: &'a Sha256Digest,
    pub(super) parameter_layout_sha256: &'a ParameterLayoutDigest,
    pub(super) activation: &'a BTreeMap<SkillId, f64>,
    pub(super) projected_delta_sha256: &'a str,
    pub(super) source_observation_sha256: &'a str,
}

pub(super) fn governed_composition_operation_key(
    operation: &GovernedCompositionOperation<'_>,
) -> BrainResult<String> {
    let activation_bytes = serde_json::to_vec(operation.activation)?;
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:GOVERNED-COMPOSITION:v2\0");
    for value in [
        operation.report_sha256,
        operation.active_bank_sha256,
        operation.evidence_bundle_sha256,
        operation.causal_credit_sha256,
        operation.parameter_layout_artifact_sha256.as_str(),
        operation.parameter_layout_sha256.as_digest().as_str(),
        operation.projected_delta_sha256,
        operation.source_observation_sha256,
    ] {
        hasher.update(value.as_bytes());
        hasher.update([0]);
    }
    hasher.update((activation_bytes.len() as u64).to_be_bytes());
    hasher.update(activation_bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

pub(super) const LEARNING_FINALIZATION_ARCHIVE_LABELS: &[&str] = &[
    "observations_manifest.json",
    "skill_bank.json",
    "shadow_skill_bank.json",
    "memory_current.json",
    "sleep_state.json",
    "sleep_evidence_current.json",
];

pub(super) fn learning_finalization_receipt_path(
    root: &Path,
    operation_key: &Sha256Digest,
) -> PathBuf {
    root.join("state/learning_finalizations")
        .join(format!("{operation_key}.json"))
}

pub(super) fn learning_finalization_archive_dir(
    root: &Path,
    operation_key: &Sha256Digest,
) -> PathBuf {
    root.join("state/corpus_transitions/by-operation")
        .join(operation_key.as_str())
}

pub(super) fn archive_label_is_allowed(label: &str) -> bool {
    LEARNING_FINALIZATION_ARCHIVE_LABELS.contains(&label)
}

pub(super) fn archive_private_state_file(
    root: &Path,
    source: &Path,
    archive: &Path,
    label: &str,
    expected_sha256: Option<&Sha256Digest>,
    archived: &mut BTreeMap<String, Sha256Digest>,
) -> BrainResult<()> {
    let source_parent = source.parent().ok_or_else(|| {
        BrainError::Invalid("learning_corpus_transition_source_parent_missing".into())
    })?;
    existing_directory_under_root(root, source_parent)?;
    if !archive_label_is_allowed(label) {
        return Err(BrainError::Integrity(format!(
            "learning_corpus_transition_state_file_invalid:{label}"
        )));
    }
    match fs::symlink_metadata(source) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && expected_sha256.is_none() => {
            return Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if let Some(expected) = expected_sha256 {
                move_private_file_transactional(root, source, archive, expected)?;
                archived.insert(label.to_string(), expected.clone());
                return Ok(());
            }
            return Err(BrainError::Integrity(format!(
                "learning_corpus_transition_required_state_file_missing:{label}"
            )));
        }
        Err(error) => return Err(error.into()),
        Ok(_) => {}
    }
    let source = existing_regular_file_under_root(root, source)?;
    ensure_private_parent(root, archive)?;
    let digest = crate::artifact::sha256_file(&source)?;
    if expected_sha256.is_some_and(|expected| digest != *expected) {
        return Err(BrainError::Integrity(format!(
            "learning_corpus_transition_prior_state_file_changed:{label}"
        )));
    }
    move_private_file_transactional(root, &source, archive, &digest)?;
    archived.insert(label.to_string(), digest);
    Ok(())
}

/// Read a persisted observation corpus from a confined directory, retaining the
/// filename-to-semantic-identity check for both the live and archived corpus.
/// This is deliberately shared so archive verification cannot be weaker than
/// runtime loading.
pub(super) fn load_observations_from_private_directory(
    root: &Path,
    directory: &Path,
    missing_is_empty: bool,
) -> BrainResult<Vec<DeltaObservation>> {
    match fs::symlink_metadata(directory) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && missing_is_empty => {
            return Ok(Vec::new())
        }
        Err(error) => return Err(error.into()),
        Ok(_) => {}
    }
    let directory = existing_directory_under_root(root, directory)?;
    let mut paths = Vec::new();
    for entry in fs::read_dir(&directory)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            return Err(BrainError::Integrity(
                "persisted_observations_entry_invalid".into(),
            ));
        }
        paths.push(existing_regular_file_under_root(root, &path)?);
    }
    paths.sort();
    let mut out = Vec::new();
    let mut ids = BTreeSet::new();
    let mut digests = BTreeSet::new();
    for path in paths {
        let bytes = read_untrusted_private_file_bounded(root, &path, MAX_OBSERVATION_RECORD_BYTES)?;
        let observation: DeltaObservation = serde_json::from_slice(&bytes)?;
        let digest = digest_json(&observation)?;
        let expected_name = format!("{}-{}.json", observation.observation_id, &digest[..16]);
        if path.file_name().and_then(|value| value.to_str()) != Some(expected_name.as_str())
            || !ids.insert(observation.observation_id.clone())
            || !digests.insert(digest)
        {
            return Err(BrainError::Integrity(
                "persisted_observation_identity_invalid".into(),
            ));
        }
        out.push(observation);
    }
    Ok(out)
}

pub(super) fn canonical_observation_digests(
    observations: &[DeltaObservation],
) -> BrainResult<Vec<String>> {
    let mut digests = observations
        .iter()
        .map(digest_json)
        .collect::<BrainResult<Vec<_>>>()?;
    digests.sort();
    digests.dedup();
    if digests.len() != observations.len() {
        return Err(BrainError::Integrity(
            "learning_finalization_observation_digest_duplicate".into(),
        ));
    }
    Ok(digests)
}

pub(super) fn verify_learning_finalization_commit_binding(
    root: &Path,
    receipt: &LearningFinalizationReceipt,
    commit: &CommitReceipt,
    report: &ReconstructionReport,
    observations: &[DeltaObservation],
) -> BrainResult<()> {
    let observation_digests = canonical_observation_digests(observations)?;
    let expected_batch_digest = digest_json(&observation_digests)?;
    let generation = observations
        .iter()
        .map(|observation| observation.generation)
        .max()
        .unwrap_or(0);
    let memory = build_memory_snapshot(
        observations,
        generation,
        true,
        &report.fields,
        &report.skill_source_mixtures,
    )?;
    let memory_sha = Sha256Digest::digest_bytes(&serialize_pretty_line(&memory)?);
    let mut shadow_bank = SkillBank::default();
    assimilate_bank(
        &mut shadow_bank,
        &report.fields,
        BrainConfig::default().skill_match_cosine,
    )?;
    let shadow_bank_sha = Sha256Digest::digest_bytes(&serialize_pretty_line(&shadow_bank)?);
    let expected_operation_key = commit_operation_key(
        &expected_batch_digest,
        report,
        receipt.report_sha256.as_str(),
    );
    if expected_batch_digest != receipt.new_corpus_digest.as_str()
        || commit.schema != "cerebro.tidex.commit_receipt/v2"
        || commit.legacy_recovery
        || commit.operation_key != receipt.commit_operation_key.as_str()
        || commit.operation_key != expected_operation_key
        || commit.batch_digest != expected_batch_digest
        || commit.report_sha256 != receipt.report_sha256.as_str()
        || commit.memory_sha256 != memory_sha.as_str()
        || commit.shadow_bank_sha256.as_deref() != Some(shadow_bank_sha.as_str())
    {
        return Err(BrainError::Integrity(
            "learning_finalization_commit_receipt_contract_invalid".into(),
        ));
    }
    PrivateFileReference::new(
        memory_artifact_path(root, &commit.memory_sha256),
        memory_sha,
    )
    .verify(root)?;
    PrivateFileReference::new(
        root.join("state/skill_banks/by-sha")
            .join(format!("{shadow_bank_sha}.json")),
        shadow_bank_sha.clone(),
    )
    .verify(root)?;
    let event = ledger::find_v2_event_by_payload_string(
        root,
        "commit_transaction",
        "operation_key",
        &commit.operation_key,
    )?
    .ok_or_else(|| BrainError::Integrity("learning_finalization_commit_ledger_missing".into()))?;
    let expected_intent = CommitTransactionIntent {
        schema: "cerebro.tidex.commit_transaction_intent/v2".into(),
        operation_key: expected_operation_key,
        batch_digest: expected_batch_digest,
        observation_digests,
        report_sha256: ReportDigest::from(receipt.report_sha256.clone()),
        report_promotable: true,
        generation,
        memory_sha256: commit.memory_sha256.clone(),
        shadow_bank_sha256: Some(SkillBankDigest::from(shadow_bank_sha)),
        prior_shadow_bank_sha256: None,
    };
    if event.event_hash != commit.ledger_event_hash {
        return Err(BrainError::Integrity(
            "learning_finalization_commit_ledger_binding_invalid".into(),
        ));
    }
    verify_commit_transaction_ledger_binding(&event, &expected_intent, observations.len(), report)?;
    Ok(())
}

pub(super) fn verify_learning_finalization_archive(
    root: &Path,
    receipt: &LearningFinalizationReceipt,
    transition_intent_dir: &Path,
) -> BrainResult<()> {
    // These are live authority pointers in every authenticated prior runtime
    // state. A finalization may not silently omit one after deciding that the
    // prior corpus is revoked; optional shadow state remains optional because
    // a prior non-promoting sleep legitimately has none.
    let required = [
        "observations_manifest.json",
        "skill_bank.json",
        "memory_current.json",
        "sleep_state.json",
        "sleep_evidence_current.json",
    ];
    if required
        .iter()
        .any(|label| !receipt.archived_artifact_sha256.contains_key(*label))
        || receipt
            .archived_artifact_sha256
            .keys()
            .any(|label| !archive_label_is_allowed(label))
    {
        return Err(BrainError::Integrity(
            "learning_finalization_archive_manifest_contract_invalid".into(),
        ));
    }
    let archive_dir = learning_finalization_archive_dir(root, &receipt.operation_key);
    let archive_dir = existing_directory_under_root(root, &archive_dir)?;
    let allowed_entries = receipt
        .archived_artifact_sha256
        .keys()
        .map(String::as_str)
        .chain(["observations", "transition_intent"])
        .collect::<BTreeSet<_>>();
    for entry in fs::read_dir(&archive_dir)? {
        let entry = entry?;
        let name = entry.file_name().into_string().map_err(|_| {
            BrainError::Integrity("learning_finalization_archive_name_invalid".into())
        })?;
        if !allowed_entries.contains(name.as_str()) {
            return Err(BrainError::Integrity(
                "learning_finalization_archive_unexpected_entry".into(),
            ));
        }
    }

    let mut manifest = None;
    for (label, digest) in &receipt.archived_artifact_sha256 {
        let bytes = PrivateFileReference::new(archive_dir.join(label), digest.clone())
            .read_verified(root)?;
        if label == "observations_manifest.json" {
            manifest = Some(serde_json::from_slice::<Vec<DeltaObservation>>(&bytes)?);
        }
    }
    let manifest = manifest.ok_or_else(|| {
        BrainError::Integrity("learning_finalization_archive_manifest_missing".into())
    })?;
    let manifest = canonical_observations(&manifest);
    let archived = canonical_observations(&load_observations_from_private_directory(
        root,
        &archive_dir.join("observations"),
        false,
    )?);
    if manifest != archived
        || archived.len() != receipt.prior_observation_count
        || observation_set_digest(&archived)? != receipt.prior_corpus_digest.as_str()
    {
        return Err(BrainError::Integrity(
            "learning_finalization_archive_observation_binding_invalid".into(),
        ));
    }

    verify_learning_finalization_transition_intent(
        root,
        transition_intent_dir,
        receipt,
        &archive_dir,
    )?;
    Ok(())
}

pub(super) fn expected_learning_finalization_transition_intent(
    receipt: &LearningFinalizationReceipt,
    archive_dir: &Path,
) -> LearningCorpusTransitionIntent {
    LearningCorpusTransitionIntent {
        schema: "cerebro.tidex.learning_corpus_transition_intent/v1".into(),
        operation_key: receipt.operation_key.clone(),
        session_id: receipt.session_id.clone(),
        adaptive_receipt_sha256: receipt.adaptive_receipt_sha256.clone(),
        learning_finalization_input_sha256: receipt.learning_finalization_input_sha256.clone(),
        representation_evidence_receipt: receipt.representation_evidence_receipt.clone(),
        representation_protocol_sha256: receipt.representation_protocol_sha256.clone(),
        representation_observation_bindings_sha256: receipt
            .representation_observation_bindings_sha256
            .clone(),
        prior_corpus_digest: receipt.prior_corpus_digest.clone(),
        prior_observation_count: receipt.prior_observation_count,
        new_corpus_digest: receipt.new_corpus_digest.clone(),
        new_observation_count: receipt.new_observation_count,
        archive_dir: archive_dir.to_path_buf(),
    }
}

pub(super) fn verify_learning_finalization_transition_intent(
    root: &Path,
    intent_dir: &Path,
    receipt: &LearningFinalizationReceipt,
    archive_dir: &Path,
) -> BrainResult<()> {
    let intent_dir = existing_directory_under_root(root, intent_dir)?;
    let mut entries = fs::read_dir(&intent_dir)?;
    let entry = entries.next().transpose()?.ok_or_else(|| {
        BrainError::Integrity("learning_finalization_transition_intent_missing".into())
    })?;
    if entries.next().transpose()?.is_some()
        || entry.file_name().as_encoded_bytes() != b"intent.json"
    {
        return Err(BrainError::Integrity(
            "learning_finalization_transition_intent_directory_invalid".into(),
        ));
    }
    let intent_path = existing_regular_file_under_root(root, &entry.path())?;
    let intent: LearningCorpusTransitionIntent = serde_json::from_slice(&fs::read(intent_path)?)?;
    if intent != expected_learning_finalization_transition_intent(receipt, archive_dir) {
        return Err(BrainError::Integrity(
            "learning_finalization_archive_intent_binding_invalid".into(),
        ));
    }
    Ok(())
}

pub(super) fn close_verified_learning_finalization_inflight(
    root: &Path,
    receipt: &LearningFinalizationReceipt,
    reference: &PrivateFileReference,
) -> BrainResult<()> {
    let inflight_root = root.join("state/corpus_transitions/inflight");
    match fs::symlink_metadata(&inflight_root) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
        Ok(_) => {}
    }
    let inflight_root = existing_directory_under_root(root, &inflight_root)?;
    let mut matching_inflight = None;
    for entry in fs::read_dir(&inflight_root)? {
        let entry = entry?;
        let name = entry.file_name().into_string().map_err(|_| {
            BrainError::Integrity("learning_finalization_inflight_name_invalid".into())
        })?;
        if name != receipt.operation_key.as_str() {
            return Err(BrainError::Integrity(
                "learning_finalization_unrelated_inflight_transition".into(),
            ));
        }
        if matching_inflight.replace(entry.path()).is_some() {
            return Err(BrainError::Integrity(
                "learning_finalization_inflight_transition_ambiguous".into(),
            ));
        }
    }
    let Some(inflight) = matching_inflight else {
        return Ok(());
    };
    let archive_dir = learning_finalization_archive_dir(root, &receipt.operation_key);
    let archive_dir = existing_directory_under_root(root, &archive_dir)?;
    verify_learning_finalization_transition_intent(root, &inflight, receipt, &archive_dir)?;
    // Before closing the only visible fail-closed marker, replay every other
    // authority binding using this exact inflight intent.  A syntactically
    // plausible receipt must never make an invalid half-transition disappear.
    let _ = load_verified_learning_finalization_receipt_under_root(
        root,
        reference,
        Some(&receipt.operation_key),
    )?;
    let closed_intent = archive_dir.join("transition_intent");
    let inflight_identity = inspect_private_directory(root, &inflight)?;
    move_private_directory_transactional(root, &inflight, &closed_intent, &inflight_identity)?;
    verify_learning_finalization_transition_intent(root, &closed_intent, receipt, &archive_dir)
}

pub(super) fn verify_learning_finalization_ledger_binding(
    root: &Path,
    receipt: &LearningFinalizationReceipt,
) -> BrainResult<()> {
    let event = ledger::find_v2_event_by_payload_string(
        root,
        "learning_corpus_transition",
        "operation_key",
        receipt.operation_key.as_str(),
    )?
    .ok_or_else(|| BrainError::Integrity("learning_finalization_receipt_ledger_missing".into()))?;
    let payload = event.payload()?;
    let representation_path = serde_json::to_value(&receipt.representation_evidence_receipt.path)?;
    let archived_artifacts = serde_json::to_value(&receipt.archived_artifact_sha256)?;
    if event.event_hash != receipt.ledger_event_hash.as_str()
        || payload.get("schema").and_then(Value::as_str)
            != Some("cerebro.tidex.learning_corpus_transition/v1")
        || payload.get("operation_key").and_then(Value::as_str)
            != Some(receipt.operation_key.as_str())
        || payload.get("session_id").and_then(Value::as_str) != Some(receipt.session_id.as_str())
        || payload
            .get("adaptive_receipt_sha256")
            .and_then(Value::as_str)
            != Some(receipt.adaptive_receipt_sha256.as_str())
        || payload
            .get("learning_finalization_input_sha256")
            .and_then(Value::as_str)
            != Some(receipt.learning_finalization_input_sha256.as_str())
        || payload.get("representation_evidence_receipt_path") != Some(&representation_path)
        || payload
            .get("representation_evidence_receipt_sha256")
            .and_then(Value::as_str)
            != Some(receipt.representation_evidence_receipt.sha256.as_str())
        || payload
            .get("representation_protocol_sha256")
            .and_then(Value::as_str)
            != Some(receipt.representation_protocol_sha256.as_str())
        || payload
            .get("representation_observation_bindings_sha256")
            .and_then(Value::as_str)
            != Some(receipt.representation_observation_bindings_sha256.as_str())
        || payload.get("prior_corpus_digest").and_then(Value::as_str)
            != Some(receipt.prior_corpus_digest.as_str())
        || payload
            .get("prior_observation_count")
            .and_then(Value::as_u64)
            != Some(receipt.prior_observation_count as u64)
        || payload.get("new_corpus_digest").and_then(Value::as_str)
            != Some(receipt.new_corpus_digest.as_str())
        || payload.get("new_observation_count").and_then(Value::as_u64)
            != Some(receipt.new_observation_count as u64)
        || payload.get("report_sha256").and_then(Value::as_str)
            != Some(receipt.report_sha256.as_str())
        || payload.get("commit_operation_key").and_then(Value::as_str)
            != Some(receipt.commit_operation_key.as_str())
        || payload.get("commit_receipt_sha256").and_then(Value::as_str)
            != Some(receipt.commit_receipt_sha256.as_str())
        || payload.get("archived_artifact_sha256") != Some(&archived_artifacts)
    {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_ledger_binding_invalid".into(),
        ));
    }
    Ok(())
}

/// Load a content-addressed governed-composition receipt and rebind it to the
/// currently active certified corpus. This is public for controller
/// supervision, but it never trusts a caller-provided runtime controller or
/// coefficient vector.
pub fn load_verified_governed_composition_receipt(
    root: impl AsRef<Path>,
    receipt_path: impl AsRef<Path>,
    receipt_sha256: &str,
) -> BrainResult<GovernedCompositionReceipt> {
    let root = verify_private_root(root.as_ref())?;
    if !valid_digest(receipt_sha256) {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_digest_invalid".into(),
        ));
    }
    let expected_receipt_path = root
        .join("state/governed_compositions/by-sha")
        .join(format!("{}.json", receipt_sha256.to_ascii_lowercase()));
    if receipt_path.as_ref() != expected_receipt_path {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_identity_invalid".into(),
        ));
    }
    let receipt_reference =
        PrivateFileReference::new(expected_receipt_path, Sha256Digest::parse(receipt_sha256)?);
    let (_canonical, receipt_bytes) = receipt_reference.read_verified_with_path(&root)?;
    let receipt_value: Value = serde_json::from_slice(&receipt_bytes)?;
    match receipt_value.get("schema").and_then(Value::as_str) {
        Some("cerebro.tidex.governed_composition_receipt/v2") => {}
        Some("cerebro.tidex.governed_composition_receipt/v1") => {
            return Err(BrainError::Integrity(
                "governed_composition_receipt_v1_historical_only".into(),
            ));
        }
        _ => {
            return Err(BrainError::Integrity(
                "governed_composition_receipt_schema_invalid".into(),
            ));
        }
    }
    let receipt: GovernedCompositionReceipt = serde_json::from_value(receipt_value)?;
    if receipt.schema != "cerebro.tidex.governed_composition_receipt/v2"
        || !valid_digest(&receipt.report_sha256)
        || !valid_digest(&receipt.active_bank_sha256)
        || !valid_digest(&receipt.evidence_bundle_sha256)
        || !valid_digest(&receipt.causal_credit_sha256)
        || receipt.field_ids.is_empty()
        || receipt.field_ids.iter().any(|id| id.trim().is_empty())
        || receipt.field_ids.iter().collect::<BTreeSet<_>>().len() != receipt.field_ids.len()
        || receipt.accepted_coefficients.len() != receipt.field_ids.len()
        || receipt
            .accepted_coefficients
            .iter()
            .any(|value| !value.is_finite())
        || receipt.trust_region.accepted_coefficients != receipt.accepted_coefficients
        || receipt.trust_region.allocation_policy
            != TrustRegionAllocationPolicy::CausalPriorityContractionV1
        || receipt
            .trust_region
            .causal_priority_weights
            .as_ref()
            .is_none_or(|weights| {
                weights.len() != receipt.field_ids.len()
                    || weights
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0)
            })
        || receipt.trust_region.component_retention.len() != receipt.field_ids.len()
        || receipt
            .trust_region
            .component_retention
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || !receipt.protection.allowed
        || !receipt.protection.damage_ratio.is_finite()
        || !receipt.protection.removed_energy.is_finite()
        || !receipt.protection.max_weighted_residual.is_finite()
        || !valid_digest(&receipt.source_observation_sha256)
    {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_contract_invalid".into(),
        ));
    }
    ParameterLayoutAuthority::open(&root)?.authenticate_canonical_binding(
        receipt.parameter_layout_artifact.clone(),
        &receipt.parameter_layout_sha256,
        receipt.projected_delta.parameter_count,
    )?;
    let expected_projected_path = root.join("artifacts/deltas/by-sha").join(format!(
        "{}.dvec",
        receipt.projected_delta.sha256.to_ascii_lowercase()
    ));
    if receipt.projected_delta.path != expected_projected_path {
        return Err(BrainError::Integrity(
            "governed_composition_projected_delta_path_invalid".into(),
        ));
    }
    let projected = existing_regular_file_under_root(&root, &expected_projected_path)?;
    let inspected = inspect_dvec(&root, &projected)?;
    if inspected.sha256 != receipt.projected_delta.sha256.to_ascii_lowercase()
        || inspected.parameter_count != receipt.projected_delta.parameter_count
    {
        return Err(BrainError::Integrity(
            "governed_composition_projected_delta_identity_invalid".into(),
        ));
    }
    let event = ledger::find_v2_event_by_payload_string(
        &root,
        "governed_composition_receipt",
        "receipt_sha256",
        receipt_sha256,
    )?
    .ok_or_else(|| BrainError::Integrity("governed_composition_receipt_ledger_missing".into()))?;
    let payload = event.payload()?;
    if payload.get("schema").and_then(serde_json::Value::as_str)
        != Some("cerebro.tidex.governed_composition_ledger_binding/v2")
        || payload
            .get("receipt_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt_sha256)
        || payload
            .get("operation_key")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.operation_key.as_str())
        || payload
            .get("report_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.report_sha256.as_str())
        || payload
            .get("active_bank_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.active_bank_sha256.as_str())
        || payload
            .get("evidence_bundle_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.evidence_bundle_sha256.as_str())
        || payload
            .get("causal_credit_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.causal_credit_sha256.as_str())
        || payload.get("parameter_layout_artifact_path")
            != Some(&Value::String(
                receipt
                    .parameter_layout_artifact
                    .path
                    .to_string_lossy()
                    .into_owned(),
            ))
        || payload
            .get("parameter_layout_artifact_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.parameter_layout_artifact.sha256.as_str())
        || payload
            .get("parameter_layout_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.parameter_layout_sha256.as_digest().as_str())
        || payload
            .get("source_observation_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.source_observation_sha256.as_str())
    {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_ledger_binding_invalid".into(),
        ));
    }
    let bank_path = root.join("state/skill_bank.json");
    let bank_path = existing_regular_file_under_root(&root, &bank_path)?;
    if file_sha256(&bank_path)? != receipt.active_bank_sha256 {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_active_bank_invalid".into(),
        ));
    }
    let bank: SkillBank = serde_json::from_slice(&fs::read(&bank_path)?)?;
    if bank
        .fields
        .iter()
        .map(|field| field.skill_id.clone())
        .collect::<Vec<_>>()
        != receipt.field_ids
    {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_field_identity_invalid".into(),
        ));
    }
    let sleep_path = existing_regular_file_under_root(&root, &root.join("state/sleep_state.json"))?;
    let sleep_state: serde_json::Value = serde_json::from_slice(&fs::read(sleep_path)?)?;
    if sleep_state
        .get("certification_status")
        .and_then(serde_json::Value::as_str)
        != Some("certified")
        || sleep_state
            .get("report_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.report_sha256.as_str())
        || sleep_state
            .get("active_bank_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.active_bank_sha256.as_str())
        || sleep_state
            .get("evidence_bundle_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.evidence_bundle_sha256.as_str())
    {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_sleep_binding_invalid".into(),
        ));
    }
    let evidence = load_sleep_evidence(&root)?;
    if evidence.causal_credit.credit_source_sha256 != receipt.causal_credit_sha256 {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_causal_binding_invalid".into(),
        ));
    }

    // A receipt is evidence of a prior decision, not authority by itself.
    // Reconstruct the decision under the *current* certified CEREBRO state and
    // compare every executable output before a controller may use it.
    let engine = BrainEngine::open(&root, BrainConfig::default())?;
    engine.require_no_incomplete_corpus_transition()?;
    engine.require_current_certification()?;
    let source = receipt.source_observation_sha256.as_str();
    engine.require_current_observation_digest(source)?;
    let recomposed = engine.compose(&receipt.requested_activation)?;
    let expected_protection = GovernedCompositionProtection {
        damage_ratio: recomposed.protection.damage_ratio,
        allowed: recomposed.protection.allowed,
        removed_energy: recomposed.protection.removed_energy,
        protected_rank: recomposed.protection.protected_rank,
        max_weighted_residual: recomposed.protection.max_weighted_residual,
    };
    if recomposed.trust_region != receipt.trust_region || expected_protection != receipt.protection
    {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_recomposition_mismatch".into(),
        ));
    }
    let expected_operation_key =
        governed_composition_operation_key(&GovernedCompositionOperation {
            report_sha256: &receipt.report_sha256,
            active_bank_sha256: &receipt.active_bank_sha256,
            evidence_bundle_sha256: &receipt.evidence_bundle_sha256,
            causal_credit_sha256: &receipt.causal_credit_sha256,
            parameter_layout_artifact_sha256: &receipt.parameter_layout_artifact.sha256,
            parameter_layout_sha256: &receipt.parameter_layout_sha256,
            activation: &receipt.requested_activation,
            projected_delta_sha256: &receipt.projected_delta.sha256,
            source_observation_sha256: source,
        })?;
    if expected_operation_key != receipt.operation_key {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_operation_key_mismatch".into(),
        ));
    }
    let expected_values = recomposed
        .delta
        .iter()
        .map(|value| {
            if !value.is_finite() || value.abs() > f32::MAX as f64 {
                return Err(BrainError::Numerical(
                    "governed_composition_projected_delta_nonrepresentable".into(),
                ));
            }
            Ok(*value as f32)
        })
        .collect::<BrainResult<Vec<_>>>()?;
    let recorded_values = read_dvec_f32(&root, &receipt.projected_delta)?;
    if recorded_values.len() != expected_values.len()
        || recorded_values
            .iter()
            .zip(&expected_values)
            .any(|(recorded, expected)| recorded.to_bits() != expected.to_bits())
    {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_projected_delta_mismatch".into(),
        ));
    }
    Ok(receipt)
}

/// Load and replay the receipt that made a completed adaptive-learning session
/// the active corpus. The producer is irrelevant to this authority boundary:
/// only the immutable learning/representation evidence chain is trusted.
pub fn load_verified_learning_finalization_receipt(
    root: impl AsRef<Path>,
    reference: &PrivateFileReference,
) -> BrainResult<LearningFinalizationReceipt> {
    let root = verify_private_root(root.as_ref())?;
    load_verified_learning_finalization_receipt_under_root(&root, reference, None)
}

/// Internal replay variant used only to authenticate the one matching inflight
/// intent before it is moved into an immutable archive after a crash.  Normal
/// callers always pass `None`, which requires no inflight transition at all.
pub(super) fn load_verified_learning_finalization_receipt_under_root(
    root: &Path,
    reference: &PrivateFileReference,
    permitted_inflight_operation: Option<&Sha256Digest>,
) -> BrainResult<LearningFinalizationReceipt> {
    let (canonical, receipt_bytes) = reference.read_verified_with_path(root)?;
    let receipt: LearningFinalizationReceipt = serde_json::from_slice(&receipt_bytes)?;
    let expected = learning_finalization_receipt_path(root, &receipt.operation_key);
    if canonical != expected
        || receipt.schema != "cerebro.tidex.learning_finalization_receipt/v1"
        || receipt.representation_observation_bindings.is_empty()
    {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_contract_invalid".into(),
        ));
    }

    let input = prepare_learning_finalization(
        root,
        receipt.session_id.as_str(),
        &receipt.representation_evidence_receipt.path,
    )?;
    let bindings_digest = Sha256Digest::digest_bytes(&serde_json::to_vec(
        &input.representation_observation_bindings,
    )?);
    if learning_finalization_input_sha256(&input)? != receipt.learning_finalization_input_sha256
        || input.adaptive_receipt_sha256 != receipt.adaptive_receipt_sha256
        || input.representation_evidence_receipt != receipt.representation_evidence_receipt
        || input.representation_protocol_sha256 != receipt.representation_protocol_sha256
        || bindings_digest != receipt.representation_observation_bindings_sha256
        || input.representation_observation_bindings != receipt.representation_observation_bindings
    {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_input_replay_mismatch".into(),
        ));
    }

    let expected_operation_key = learning_finalization_operation_key(
        &receipt.learning_finalization_input_sha256,
        &receipt.prior_corpus_digest,
        &receipt.new_corpus_digest,
    )?;
    if receipt.operation_key != expected_operation_key {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_operation_key_invalid".into(),
        ));
    }

    let input_observations = canonical_observations(&input.observations);
    if observation_set_digest(&input_observations)? != receipt.new_corpus_digest
        || input_observations.len() != receipt.new_observation_count
    {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_input_corpus_mismatch".into(),
        ));
    }
    let engine = BrainEngine::open(root, BrainConfig::default())?;
    engine.require_canonical_runtime_config()?;
    match permitted_inflight_operation {
        None => engine.require_no_incomplete_corpus_transition()?,
        Some(operation_key) if *operation_key == receipt.operation_key => {
            let inflight = root
                .join("state/corpus_transitions/inflight")
                .join(operation_key.as_str());
            let _ = existing_directory_under_root(root, &inflight)?;
        }
        Some(_) => {
            return Err(BrainError::Integrity(
                "learning_finalization_recovery_operation_mismatch".into(),
            ))
        }
    }
    let current = canonical_observations(&engine.load_persisted_observations()?);
    if observation_set_digest(&current)? != receipt.new_corpus_digest
        || current.len() != receipt.new_observation_count
    {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_current_corpus_mismatch".into(),
        ));
    }
    let current_by_id = current
        .iter()
        .map(|observation| {
            Ok((
                observation.observation_id.clone(),
                Sha256Digest::parse(digest_json(observation)?)?,
            ))
        })
        .collect::<BrainResult<BTreeMap<_, _>>>()?;
    let mut source_ids = BTreeSet::new();
    let mut source_hashes = BTreeSet::new();
    let mut promoted_hashes = BTreeSet::new();
    for binding in &receipt.representation_observation_bindings {
        binding.adaptive_source_observation.verify(root)?;
        binding
            .representation_destination_observation
            .verify(root)?;
        if !source_ids.insert(binding.observation_id.as_str())
            || !source_hashes.insert(binding.adaptive_source_observation.sha256.as_str())
            || !promoted_hashes.insert(binding.promoted_observation_semantic_sha256.as_str())
            || current_by_id.get(&binding.observation_id)
                != Some(&binding.promoted_observation_semantic_sha256)
        {
            return Err(BrainError::Integrity(
                "learning_finalization_receipt_observation_binding_invalid".into(),
            ));
        }
    }
    if source_ids.len() != current_by_id.len() || current_by_id.len() != input_observations.len() {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_observation_binding_coverage_invalid".into(),
        ));
    }

    let report_path = root
        .join("state/reports")
        .join(format!("{}.json", receipt.report_sha256));
    let report_reference = PrivateFileReference::new(report_path, receipt.report_sha256.clone());
    let report: ReconstructionReport =
        serde_json::from_slice(&report_reference.read_verified(root)?)?;
    if report.observation_set_digest != receipt.new_corpus_digest || !report.promotion.allowed {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_report_corpus_invalid".into(),
        ));
    }
    let mut rederived_report = engine.analyze_canonical(&input_observations)?;
    if !rederived_report.promotion.allowed {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_rederived_report_not_promotable".into(),
        ));
    }
    let mut recorded_report_without_dense = report.clone();
    for field in &mut recorded_report_without_dense.fields {
        field.dense_materialization = None;
    }
    for field in &mut rederived_report.fields {
        field.dense_materialization = None;
    }
    if recorded_report_without_dense != rederived_report {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_report_rederivation_mismatch".into(),
        ));
    }
    engine.verify_dense_field_materializations(
        &report.fields,
        &report.skill_source_mixtures,
        &input_observations,
    )?;
    let commit_path = root
        .join("state/commits")
        .join(format!("{}.json", receipt.commit_operation_key));
    let commit_reference =
        PrivateFileReference::new(commit_path, receipt.commit_receipt_sha256.clone());
    let commit: CommitReceipt = serde_json::from_slice(&commit_reference.read_verified(root)?)?;
    verify_learning_finalization_commit_binding(
        root,
        &receipt,
        &commit,
        &report,
        &input_observations,
    )?;
    let transition_intent_dir = match permitted_inflight_operation {
        None => learning_finalization_archive_dir(root, &receipt.operation_key)
            .join("transition_intent"),
        Some(operation_key) => root
            .join("state/corpus_transitions/inflight")
            .join(operation_key.as_str()),
    };
    verify_learning_finalization_archive(root, &receipt, &transition_intent_dir)?;
    verify_learning_finalization_ledger_binding(root, &receipt)?;
    Ok(receipt)
}

pub(super) fn validate_brain_config(config: &BrainConfig) -> BrainResult<()> {
    let minimum_observations_required = config
        .min_independent_apertures
        .checked_mul(2)
        .ok_or_else(|| BrainError::Invalid("brain_config_observation_count_overflow".into()))?;
    if config.min_independent_apertures < 2
        || config.min_independent_apertures > MAX_ENGINE_INDEPENDENCE_GROUPS
        || config.min_observations < minimum_observations_required
        || config.min_observations > MAX_ENGINE_OBSERVATIONS
        || config.max_rank == 0
        || config.max_rank > MAX_ENGINE_SKILL_FIELDS
        || !config.target_explained_variance.is_finite()
        || !(0.0..=1.0).contains(&config.target_explained_variance)
        || !config.ridge.is_finite()
        || config.ridge <= 0.0
        || !config.huber_delta.is_finite()
        || config.huber_delta <= 0.0
        || config.irls_rounds == 0
        || config.irls_rounds > MAX_ENGINE_IRLS_ROUNDS
        || !config.skill_match_cosine.is_finite()
        || !(0.0..=1.0).contains(&config.skill_match_cosine)
        || !config.min_functional_cv_r2.is_finite()
        || config.min_functional_cv_r2 > 1.0
        || !config.max_cycle_rms.is_finite()
        || config.max_cycle_rms < 0.0
        || !config.min_skill_coherence.is_finite()
        || !(0.0..=1.0).contains(&config.min_skill_coherence)
        || !config.min_skill_persistence.is_finite()
        || !(0.0..=1.0).contains(&config.min_skill_persistence)
        || !config.min_field_explained_variance.is_finite()
        || !(0.0..=1.0).contains(&config.min_field_explained_variance)
        || !config.max_condition_estimate.is_finite()
        || config.max_condition_estimate < 1.0
        || !config
            .max_spectral_normalized_reconstruction_rms
            .is_finite()
        || config.max_spectral_normalized_reconstruction_rms < 0.0
        || !config.min_identifiability_signal_to_noise.is_finite()
        || config.min_identifiability_signal_to_noise <= 0.0
        || !config.min_representation_match_accuracy.is_finite()
        || !(0.0..=1.0).contains(&config.min_representation_match_accuracy)
        || !config.min_representation_match_margin.is_finite()
        || config.min_representation_match_margin < 0.0
        || !config.min_representation_cv_r2.is_finite()
        || config.min_representation_cv_r2 > 1.0
    {
        return Err(BrainError::Invalid("brain_config_invalid".into()));
    }
    Ok(())
}
