use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum UnsealedCorpusRecoveryDecision {
    RequiresOriginalFinalizationReplay,
    RejectUnrecognizedLiveCorpus,
    RejectLiveAndArchiveBothPresent,
    RestorePriorCorpus,
    RollBackIntent,
}

pub(super) fn classify_unsealed_corpus_recovery(
    live_digest: Option<&CorpusDigest>,
    prior_corpus_digest: &Sha256Digest,
    new_corpus_digest: &Sha256Digest,
    archive_has_observations: bool,
) -> UnsealedCorpusRecoveryDecision {
    if live_digest.map(CorpusDigest::as_digest) == Some(new_corpus_digest) {
        return UnsealedCorpusRecoveryDecision::RequiresOriginalFinalizationReplay;
    }
    if live_digest.is_some_and(|digest| digest.as_digest() != prior_corpus_digest) {
        return UnsealedCorpusRecoveryDecision::RejectUnrecognizedLiveCorpus;
    }
    if archive_has_observations {
        if live_digest.is_some() {
            UnsealedCorpusRecoveryDecision::RejectLiveAndArchiveBothPresent
        } else {
            UnsealedCorpusRecoveryDecision::RestorePriorCorpus
        }
    } else {
        UnsealedCorpusRecoveryDecision::RollBackIntent
    }
}

impl BrainEngine {
    pub(super) fn incomplete_transition_operation_key(&self) -> BrainResult<Option<Sha256Digest>> {
        let inflight = self.root.join("state/corpus_transitions/inflight");
        match fs::symlink_metadata(&inflight) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
            Ok(_) => {}
        }
        let inflight = existing_directory_under_root(&self.root, &inflight)?;
        let mut key = None;
        for path in list_existing_private_directory(&self.root, &inflight)? {
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| {
                    BrainError::Integrity("learning_finalization_inflight_name_invalid".into())
                })?;
            let parsed = Sha256Digest::parse(name)?;
            if key.replace(parsed).is_some() {
                return Err(BrainError::Integrity(
                    "learning_finalization_inflight_transition_ambiguous".into(),
                ));
            }
        }
        Ok(key)
    }

    /// Recover an interrupted corpus transition without inventing a receipt.
    ///
    /// Forward completion after the new corpus is published still requires the
    /// original finalization input. This API only closes a verified receipt,
    /// rolls back an intent that never retired the prior corpus, or restores
    /// the archived prior corpus when the new generation was never published.
    pub fn recover_incomplete_corpus_transition(&self) -> BrainResult<CorpusTransitionRecovery> {
        self.with_engine_authority(|| self.recover_incomplete_corpus_transition_under_authority())
    }

    pub(super) fn recover_incomplete_corpus_transition_under_authority(
        &self,
    ) -> BrainResult<CorpusTransitionRecovery> {
        let Some(operation_key) = self.incomplete_transition_operation_key()? else {
            return Ok(CorpusTransitionRecovery::new(
                CorpusTransitionRecoveryOutcome::NoIncompleteTransition,
                None,
                None,
            ));
        };
        let journal = self.load_transition_journal(&operation_key)?;
        let phase = journal.as_ref().map(|value| value.phase);
        let receipt_path = learning_finalization_receipt_path(&self.root, &operation_key);
        let receipt_exists = match fs::symlink_metadata(&receipt_path) {
            Ok(_) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(error.into()),
        };
        if receipt_exists {
            let bytes = read_untrusted_private_file_bounded(
                &self.root,
                &receipt_path,
                MAX_ENGINE_JSON_BYTES,
            )?;
            let receipt: LearningFinalizationReceipt = serde_json::from_slice(&bytes)?;
            let reference =
                PrivateFileReference::new(receipt_path, Sha256Digest::digest_bytes(&bytes));
            close_verified_learning_finalization_inflight(&self.root, &receipt, &reference)?;
            self.advance_canonical_engine_head(
                HeadIncomplete::Clear,
                None,
                Some(ReportDigest::from(receipt.report_sha256.clone())),
            )?;
            return Ok(CorpusTransitionRecovery::new(
                CorpusTransitionRecoveryOutcome::ClosedVerifiedReceipt,
                Some(operation_key),
                phase.or(Some(CorpusTransitionPhase::ReceiptSealed)),
            ));
        }

        let intent_path = self
            .root
            .join("state/corpus_transitions/inflight")
            .join(operation_key.as_str())
            .join("intent.json");
        let intent: LearningCorpusTransitionIntent =
            read_private_json(&self.root, &intent_path, MAX_ENGINE_JSON_BYTES)?;
        let live_digest = {
            let observations = self.load_persisted_observations()?;
            if observations.is_empty() {
                None
            } else {
                Some(observation_set_digest(&observations)?)
            }
        };
        let archive = learning_finalization_archive_dir(&self.root, &operation_key);
        let archive_observations = archive.join("observations");
        let archive_has_observations = match fs::symlink_metadata(&archive_observations) {
            Ok(_) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(error.into()),
        };
        match classify_unsealed_corpus_recovery(
            live_digest.as_ref(),
            &intent.prior_corpus_digest,
            &intent.new_corpus_digest,
            archive_has_observations,
        ) {
            UnsealedCorpusRecoveryDecision::RequiresOriginalFinalizationReplay => {
                Ok(CorpusTransitionRecovery::new(
                    CorpusTransitionRecoveryOutcome::RequiresOriginalFinalizationReplay,
                    Some(operation_key),
                    phase.or(Some(CorpusTransitionPhase::NewCorpusPublished)),
                ))
            }
            UnsealedCorpusRecoveryDecision::RejectUnrecognizedLiveCorpus => Err(
                BrainError::Integrity("recovery_live_corpus_unrecognized".into()),
            ),
            UnsealedCorpusRecoveryDecision::RejectLiveAndArchiveBothPresent => Err(
                BrainError::Integrity("recovery_live_and_archive_both_present".into()),
            ),
            UnsealedCorpusRecoveryDecision::RestorePriorCorpus => {
                self.restore_archived_prior_corpus(&operation_key)?;
                self.abort_inflight_transition(&operation_key)?;
                self.advance_canonical_engine_head(HeadIncomplete::Clear, None, None)?;
                Ok(CorpusTransitionRecovery::new(
                    CorpusTransitionRecoveryOutcome::RestoredPriorCorpus,
                    Some(operation_key),
                    phase.or(Some(CorpusTransitionPhase::PriorArchived)),
                ))
            }
            UnsealedCorpusRecoveryDecision::RollBackIntent => {
                self.abort_inflight_transition(&operation_key)?;
                self.advance_canonical_engine_head(HeadIncomplete::Clear, None, None)?;
                Ok(CorpusTransitionRecovery::new(
                    CorpusTransitionRecoveryOutcome::RolledBackIntent,
                    Some(operation_key),
                    phase.or(Some(CorpusTransitionPhase::IntentRecorded)),
                ))
            }
        }
    }

    pub(super) fn restore_archived_prior_corpus(
        &self,
        operation_key: &Sha256Digest,
    ) -> BrainResult<()> {
        let archive = learning_finalization_archive_dir(&self.root, operation_key);
        let archive = existing_directory_under_root(&self.root, &archive)?;
        let archived_observations = archive.join("observations");
        let live_observations = self.root.join("state/observations");
        match fs::symlink_metadata(&live_observations) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let identity = inspect_private_directory(&self.root, &archived_observations)?;
                move_private_directory_transactional(
                    &self.root,
                    &archived_observations,
                    &live_observations,
                    &identity,
                )?;
            }
            Ok(_) => {
                return Err(BrainError::Integrity(
                    "recovery_live_observations_already_present".into(),
                ))
            }
            Err(error) => return Err(error.into()),
        }
        for (label, destination) in [
            ("skill_bank.json", self.bank_path()),
            (
                "shadow_skill_bank.json",
                self.root.join("state/shadow_skill_bank.json"),
            ),
            (
                "memory_current.json",
                self.root.join("state/memory/current.json"),
            ),
            ("sleep_state.json", self.root.join("state/sleep_state.json")),
            (
                "sleep_evidence_current.json",
                self.root.join("state/sleep_evidence/current.json"),
            ),
        ] {
            let source = archive.join(label);
            match fs::symlink_metadata(&source) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error.into()),
                Ok(_) => {
                    let source_bytes = read_existing_private_file_bounded(
                        &self.root,
                        &source,
                        MAX_SKILL_BANK_BYTES,
                    )?;
                    let digest = Sha256Digest::digest_bytes(&source_bytes);
                    match fs::symlink_metadata(&destination) {
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Ok(_) => {
                            return Err(BrainError::Integrity(format!(
                                "recovery_live_pointer_already_present:{label}"
                            )))
                        }
                        Err(error) => return Err(error.into()),
                    }
                    move_private_file_transactional(&self.root, &source, &destination, &digest)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn abort_inflight_transition(
        &self,
        operation_key: &Sha256Digest,
    ) -> BrainResult<()> {
        let inflight = self
            .root
            .join("state/corpus_transitions/inflight")
            .join(operation_key.as_str());
        let aborted = self
            .root
            .join("state/corpus_transitions/aborted")
            .join(operation_key.as_str());
        ensure_private_directory(
            &self.root,
            &self.root.join("state/corpus_transitions/aborted"),
        )?;
        let identity = inspect_private_directory(&self.root, &inflight)?;
        move_private_directory_transactional(&self.root, &inflight, &aborted, &identity)
    }

    /// A corpus transition deliberately fails closed after its intent is
    /// written. Until the finalizer has sealed the immutable receipt and moved
    /// that intent into the archive, no ordinary commit/sleep/runtime action
    /// may operate on a potentially half-replaced active corpus.
    pub(super) fn require_no_incomplete_corpus_transition(&self) -> BrainResult<()> {
        let inflight = self.root.join("state/corpus_transitions/inflight");
        match fs::symlink_metadata(&inflight) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
            Ok(_) => {}
        }
        let inflight = existing_directory_under_root(&self.root, &inflight).map_err(|_| {
            BrainError::Integrity("learning_corpus_transition_inflight_directory_invalid".into())
        })?;
        let entries = list_existing_private_directory(&self.root, &inflight)?;
        let Some(path) = entries.first() else {
            return Ok(());
        };
        if existing_directory_under_root(&self.root, path).is_err() {
            return Err(BrainError::Integrity(
                "learning_corpus_transition_inflight_entry_invalid".into(),
            ));
        }
        Err(BrainError::Integrity(
            "learning_corpus_transition_incomplete".into(),
        ))
    }

    pub(super) fn commit_after_verified_corpus_transition(
        &self,
        observations: &[DeltaObservation],
    ) -> BrainResult<ReconstructionReport> {
        self.require_canonical_runtime_config()?;
        let obs = canonical_observations(observations);
        let mut report = self.analyze_canonical(&obs)?;
        if report.promotion.allowed {
            let mixtures = report.skill_source_mixtures.clone();
            self.materialize_dense_fields(&mut report.fields, &mixtures, &obs)?;
        } else {
            return Err(BrainError::Integrity(
                "corpus_transition_commit_requires_promotable_report".into(),
            ));
        }
        let observation_digests = self.persist_observations(&obs)?;
        let batch_digest = digest_json(&observation_digests)?;
        if batch_digest != report.observation_set_digest {
            return Err(BrainError::Integrity(
                "commit_observation_set_digest_mismatch".into(),
            ));
        }
        // The persisted report is the authority for every artifact derived from
        // this commit. Normalize once through the shared exact wire boundary.
        let normalized = normalize_reconstruction_report_wire(&report)?;
        report = normalized.0;
        let report_bytes = normalized.1;
        let report_sha256 = ReportDigest::from(Sha256Digest::digest_bytes(&report_bytes));
        let operation_key = commit_operation_key(&batch_digest, &report, &report_sha256);
        let generation = obs
            .iter()
            .map(|observation| observation.generation)
            .max()
            .unwrap_or(0);
        let memory = build_memory_snapshot(
            &obs,
            generation,
            true,
            &report.fields,
            &report.skill_source_mixtures,
        )?;
        let memory_bytes = serialize_pretty_line(&memory)?;
        let memory_sha256 = MemoryDigest::from(Sha256Digest::digest_bytes(&memory_bytes));
        let shadow_bank_path = self.root.join("state/shadow_skill_bank.json");
        match fs::symlink_metadata(&shadow_bank_path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) => {
                return Err(BrainError::Integrity(
                    "corpus_transition_prior_shadow_bank_not_archived".into(),
                ))
            }
            Err(error) => return Err(error.into()),
        }
        let mut shadow_bank = SkillBank::default();
        assimilate_bank(
            &mut shadow_bank,
            &report.fields,
            self.config.skill_match_cosine,
        )?;
        let staged_bank_bytes = serialize_pretty_line(&shadow_bank)?;
        let shadow_bank_sha256 =
            SkillBankDigest::from(Sha256Digest::digest_bytes(&staged_bank_bytes));
        let expected_intent = CommitTransactionIntent {
            schema: "cerebro.tidex.commit_transaction_intent/v2".into(),
            operation_key: operation_key.clone(),
            batch_digest: batch_digest.clone(),
            observation_digests: observation_digests.clone(),
            report_sha256: report_sha256.clone(),
            report_promotable: true,
            generation,
            memory_sha256: memory_sha256.clone(),
            shadow_bank_sha256: Some(shadow_bank_sha256.clone()),
            prior_shadow_bank_sha256: None,
        };

        let commits_dir = self.root.join("state/commits");
        let transactions_dir = self.root.join("state/transactions");
        ensure_private_directory(&self.root, &commits_dir)?;
        ensure_private_directory(&self.root, &transactions_dir)?;
        let receipt_path = commits_dir.join(format!("{operation_key}.json"));

        // Completed batches are immutable and idempotent. Re-running the same
        // observation batch verifies every authority artifact and returns the
        // fresh analysis without touching bank, memory, or ledger.
        if fs::symlink_metadata(&receipt_path).is_ok() {
            let receipt: CommitReceipt =
                read_private_json(&self.root, &receipt_path, MAX_ENGINE_JSON_BYTES)?;
            if receipt.schema != "cerebro.tidex.commit_receipt/v2"
                || receipt.legacy_recovery
                || receipt.operation_key != expected_intent.operation_key
                || receipt.batch_digest != expected_intent.batch_digest
                || receipt.report_sha256 != expected_intent.report_sha256
                || receipt.memory_sha256 != expected_intent.memory_sha256
                || receipt.shadow_bank_sha256 != expected_intent.shadow_bank_sha256
            {
                return Err(BrainError::Integrity(
                    "commit_receipt_report_mismatch".into(),
                ));
            }
            let event = ledger::find_v2_event_by_payload_string(
                &self.root,
                "commit_transaction",
                "operation_key",
                &operation_key,
            )?
            .ok_or_else(|| BrainError::Integrity("commit_receipt_ledger_event_missing".into()))?;
            if event.event_hash != receipt.ledger_event_hash {
                return Err(BrainError::Integrity(
                    "commit_receipt_ledger_event_mismatch".into(),
                ));
            }
            verify_commit_transaction_ledger_binding(&event, &expected_intent, obs.len(), &report)?;
            let report_path = self
                .root
                .join("state/reports")
                .join(format!("{}.json", report_sha256));
            if read_existing_private_file_bounded(&self.root, &report_path, MAX_ENGINE_JSON_BYTES)?
                != report_bytes
            {
                return Err(BrainError::Integrity(
                    "commit_receipt_report_content_mismatch".into(),
                ));
            }
            let memory_path = memory_artifact_path(&self.root, &receipt.memory_sha256);
            let persisted_memory = read_existing_private_file_bounded(
                &self.root,
                &memory_path,
                MAX_ENGINE_JSON_BYTES,
            )?;
            if sha256_bytes(&persisted_memory) != receipt.memory_sha256.as_str() {
                return Err(BrainError::Integrity(
                    "commit_receipt_memory_mismatch".into(),
                ));
            }
            if persisted_memory != memory_bytes {
                return Err(BrainError::Integrity(
                    "commit_receipt_memory_content_mismatch".into(),
                ));
            }
            let bank_path = self
                .root
                .join("state/skill_banks/by-sha")
                .join(format!("{}.json", shadow_bank_sha256));
            let persisted_bank =
                read_existing_private_file_bounded(&self.root, &bank_path, MAX_SKILL_BANK_BYTES)?;
            if sha256_bytes(&persisted_bank) != shadow_bank_sha256.as_str()
                || persisted_bank != staged_bank_bytes
            {
                return Err(BrainError::Integrity(
                    "commit_receipt_shadow_bank_mismatch".into(),
                ));
            }
            let current_memory = read_existing_private_file_bounded(
                &self.root,
                &self.root.join("state/memory/current.json"),
                MAX_ENGINE_JSON_BYTES,
            )?;
            let current_shadow = read_existing_private_file_bounded(
                &self.root,
                &self.root.join("state/shadow_skill_bank.json"),
                MAX_SKILL_BANK_BYTES,
            )?;
            if current_memory != memory_bytes || current_shadow != staged_bank_bytes {
                return Err(BrainError::Integrity(
                    "commit_receipt_current_pointer_mismatch".into(),
                ));
            }
            self.advance_canonical_engine_head(
                HeadIncomplete::Preserve,
                None,
                Some(report_sha256.clone()),
            )?;
            return Ok(report);
        }

        let transaction_dir = transactions_dir.join(&operation_key);
        let intent_path = transaction_dir.join("intent.json");
        let staged_report_path = transaction_dir.join("report.json");
        let staged_memory_path = transaction_dir.join("memory.json");
        let staged_bank_path = transaction_dir.join("shadow_skill_bank.json");

        let transaction_exists = match fs::symlink_metadata(&transaction_dir) {
            Ok(_) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(error.into()),
        };
        if transaction_exists {
            let transaction_dir = existing_directory_under_root(&self.root, &transaction_dir)?;
            let intent_path = existing_regular_file_under_root(&self.root, &intent_path)?;
            let staged_report_path =
                existing_regular_file_under_root(&self.root, &staged_report_path)?;
            let staged_memory_path =
                existing_regular_file_under_root(&self.root, &staged_memory_path)?;
            let staged_bank_path = existing_regular_file_under_root(&self.root, &staged_bank_path)?;
            let stored: CommitTransactionIntent =
                read_private_json(&self.root, &intent_path, MAX_ENGINE_JSON_BYTES)?;
            if stored != expected_intent
                || read_existing_private_file_bounded(
                    &self.root,
                    &staged_report_path,
                    MAX_ENGINE_JSON_BYTES,
                )? != report_bytes
                || read_existing_private_file_bounded(
                    &self.root,
                    &staged_memory_path,
                    MAX_ENGINE_JSON_BYTES,
                )? != memory_bytes
                || read_existing_private_file_bounded(
                    &self.root,
                    &staged_bank_path,
                    MAX_SKILL_BANK_BYTES,
                )? != staged_bank_bytes
            {
                return Err(BrainError::Integrity(
                    "commit_transaction_intent_mismatch".into(),
                ));
            }
            let mut names = BTreeSet::new();
            for path in list_existing_private_directory(&self.root, &transaction_dir)? {
                let name = path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| {
                        BrainError::Integrity("commit_transaction_stage_filename_invalid".into())
                    })?
                    .to_string();
                names.insert(name);
            }
            if names
                != BTreeSet::from([
                    "intent.json".to_string(),
                    "report.json".to_string(),
                    "memory.json".to_string(),
                    "shadow_skill_bank.json".to_string(),
                ])
            {
                return Err(BrainError::Integrity(
                    "commit_transaction_stage_directory_invalid".into(),
                ));
            }
        } else {
            ensure_private_directory(&self.root, &transaction_dir)?;
            write_new_private(&self.root, &staged_report_path, &report_bytes)?;
            write_new_private(&self.root, &staged_memory_path, &memory_bytes)?;
            write_new_private(&self.root, &staged_bank_path, &staged_bank_bytes)?;
            write_new_private(
                &self.root,
                &intent_path,
                &serialize_pretty_line(&expected_intent)?,
            )?;
        }
        let intent = expected_intent;

        // There is exactly one authoritative ledger event per batch. If a
        // crash happened after append but before canonical installation, retry
        // finds the event and resumes rather than appending or assimilating again.
        let event = if let Some(existing) = ledger::find_v2_event_by_payload_string(
            &self.root,
            "commit_transaction",
            "operation_key",
            &operation_key,
        )? {
            verify_commit_transaction_ledger_binding(&existing, &intent, obs.len(), &report)?;
            existing
        } else {
            let event = ledger::append(
                &self.root,
                "commit_transaction",
                commit_transaction_payload(&intent, obs.len(), &report),
            )?;
            verify_commit_transaction_ledger_binding(&event, &intent, obs.len(), &report)?;
            event
        };

        let canonical_report = self
            .root
            .join("state/reports")
            .join(format!("{}.json", intent.report_sha256));
        write_immutable_exact(
            &self.root,
            &canonical_report,
            &report_bytes,
            &intent.report_sha256,
        )?;
        let historical_memory = memory_artifact_path(&self.root, &intent.memory_sha256);
        write_immutable_exact(
            &self.root,
            &historical_memory,
            &memory_bytes,
            &intent.memory_sha256,
        )?;
        let memory_path = self.root.join("state/memory/current.json");
        replace_private_pointer_exact(
            &self.root,
            &memory_path,
            &memory_bytes,
            &intent.memory_sha256,
        )?;
        let expected_bank = intent
            .shadow_bank_sha256
            .as_ref()
            .ok_or_else(|| BrainError::Integrity("corpus_transition_shadow_bank_missing".into()))?;
        let historical_bank = self
            .root
            .join("state/skill_banks/by-sha")
            .join(format!("{expected_bank}.json"));
        write_immutable_exact(
            &self.root,
            &historical_bank,
            &staged_bank_bytes,
            expected_bank,
        )?;
        replace_private_pointer_exact(
            &self.root,
            &self.root.join("state/shadow_skill_bank.json"),
            &staged_bank_bytes,
            expected_bank,
        )?;

        let receipt = CommitReceipt {
            schema: "cerebro.tidex.commit_receipt/v2".into(),
            operation_key: operation_key.clone(),
            batch_digest: batch_digest.clone(),
            report_sha256: intent.report_sha256.clone(),
            memory_sha256: intent.memory_sha256.clone(),
            shadow_bank_sha256: intent.shadow_bank_sha256.clone(),
            ledger_event_hash: event.event_hash,
            legacy_recovery: false,
        };
        write_new_private(&self.root, &receipt_path, &serialize_pretty_line(&receipt)?)?;
        let persisted_receipt: CommitReceipt =
            read_private_json(&self.root, &receipt_path, MAX_ENGINE_JSON_BYTES)?;
        if persisted_receipt != receipt {
            return Err(BrainError::Integrity(
                "commit_receipt_post_write_mismatch".into(),
            ));
        }
        let verified_event = ledger::find_v2_event_by_payload_string(
            &self.root,
            "commit_transaction",
            "operation_key",
            &intent.operation_key,
        )?
        .ok_or_else(|| BrainError::Integrity("commit_receipt_ledger_event_missing".into()))?;
        if verified_event.event_hash != receipt.ledger_event_hash {
            return Err(BrainError::Integrity(
                "commit_receipt_post_write_ledger_changed".into(),
            ));
        }
        verify_commit_transaction_ledger_binding(&verified_event, &intent, obs.len(), &report)?;
        let current_memory = read_existing_private_file_bounded(
            &self.root,
            &self.root.join("state/memory/current.json"),
            MAX_ENGINE_JSON_BYTES,
        )?;
        let current_shadow = read_existing_private_file_bounded(
            &self.root,
            &self.root.join("state/shadow_skill_bank.json"),
            MAX_SKILL_BANK_BYTES,
        )?;
        if current_memory != memory_bytes || current_shadow != staged_bank_bytes {
            return Err(BrainError::Integrity(
                "commit_receipt_post_write_pointer_mismatch".into(),
            ));
        }
        self.advance_canonical_engine_head(
            HeadIncomplete::Preserve,
            None,
            Some(report_sha256.clone()),
        )?;
        Ok(report)
    }

    /// Finalize a receipt-backed adaptive-learning session into a new canonical
    /// TIDE-X corpus.  This is intentionally narrower than `commit`: callers
    /// cannot supply observations, cannot merge an unrelated corpus, and
    /// cannot carry a prior shadow/active bank into the new parameter/function
    /// domain.  Any interrupted transition remains visibly incomplete and the
    /// runtime fails closed rather than attempting a heuristic recovery.
    pub fn commit_finalized_learning_session(
        &self,
        supplied: &LearningFinalizationInput,
    ) -> BrainResult<LearningFinalizationReceipt> {
        self.with_engine_authority(|| {
            self.commit_finalized_learning_session_under_authority(supplied)
        })
    }

    pub(super) fn commit_finalized_learning_session_under_authority(
        &self,
        supplied: &LearningFinalizationInput,
    ) -> BrainResult<LearningFinalizationReceipt> {
        self.require_canonical_runtime_config()?;
        let input = verify_learning_finalization_input(&self.root, supplied)?;
        let learning_finalization_input_sha256 = learning_finalization_input_sha256(&input)?;
        let representation_observation_bindings_sha256 = Sha256Digest::digest_bytes(
            &serde_json::to_vec(&input.representation_observation_bindings)?,
        );
        let observations = canonical_observations(&input.observations);
        if observations.is_empty() {
            return Err(BrainError::Integrity(
                "learning_finalization_observation_set_empty".into(),
            ));
        }

        // Validate the complete prospective corpus before it can replace any
        // active state. A failed learning reconstruction is diagnostic evidence,
        // not authority to evict the existing corpus.
        let mut prospective = self.analyze_canonical(&observations)?;
        if !prospective.promotion.allowed {
            return Err(BrainError::Integrity(format!(
                "learning_finalization_reconstruction_not_promotable:{}",
                prospective
                    .promotion
                    .reasons
                    .iter()
                    .map(|reason| reason.as_str())
                    .collect::<Vec<_>>()
                    .join("|")
            )));
        }
        let new_corpus_digest = Sha256Digest::parse(&prospective.observation_set_digest)?;
        let finalizations_dir = self.root.join("state/learning_finalizations");

        // Idempotence has to be resolved before treating the currently active
        // corpus as a *prior* corpus: after a completed transition the active
        // observations are necessarily the new corpus. Search immutable
        // receipts by the re-derived input binding and reject ambiguity.
        match fs::symlink_metadata(&finalizations_dir) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
            Ok(_) => {
                let finalizations_dir =
                    existing_directory_under_root(&self.root, &finalizations_dir)?;
                let mut matching = Vec::new();
                for path in list_existing_private_directory(&self.root, &finalizations_dir)? {
                    if path.extension().and_then(|value| value.to_str()) != Some("json") {
                        return Err(BrainError::Integrity(
                            "learning_finalization_receipt_directory_entry_invalid".into(),
                        ));
                    }
                    let path = existing_regular_file_under_root(&self.root, &path)?;
                    let bytes = read_existing_private_file_bounded(
                        &self.root,
                        &path,
                        MAX_ENGINE_JSON_BYTES,
                    )?;
                    let receipt: LearningFinalizationReceipt = serde_json::from_slice(&bytes)?;
                    if receipt.learning_finalization_input_sha256
                        == learning_finalization_input_sha256
                    {
                        matching.push((path, Sha256Digest::digest_bytes(&bytes), receipt));
                    }
                }
                if matching.len() > 1 {
                    return Err(BrainError::Integrity(
                        "learning_finalization_receipt_input_ambiguous".into(),
                    ));
                }
                if let Some((path, digest, receipt)) = matching.pop() {
                    if receipt.schema != "cerebro.tidex.learning_finalization_receipt/v1"
                        || path.file_name().and_then(|value| value.to_str())
                            != Some(format!("{}.json", receipt.operation_key).as_str())
                    {
                        return Err(BrainError::Integrity(
                            "learning_finalization_receipt_contract_invalid".into(),
                        ));
                    }
                    let reference = PrivateFileReference::new(path, digest);
                    close_verified_learning_finalization_inflight(
                        &self.root, &receipt, &reference,
                    )?;
                    let verified =
                        load_verified_learning_finalization_receipt(&self.root, &reference)?;
                    if verified.learning_finalization_input_sha256
                        != learning_finalization_input_sha256
                    {
                        return Err(BrainError::Integrity(
                            "learning_finalization_receipt_input_replay_mismatch".into(),
                        ));
                    }
                    return Ok(verified);
                }
            }
        }

        self.require_no_incomplete_corpus_transition()?;

        let prior_observations = canonical_observations(&self.load_persisted_observations()?);
        if prior_observations.is_empty() {
            return Err(BrainError::Integrity(
                "learning_finalization_prior_corpus_missing".into(),
            ));
        }
        let prior_corpus_digest =
            Sha256Digest::parse(observation_set_digest(&prior_observations)?)?;
        if prior_corpus_digest == new_corpus_digest {
            return Err(BrainError::Integrity(
                "learning_finalization_corpus_transition_not_distinct".into(),
            ));
        }

        // A corpus may be replaced only after the *authenticated* prior sleep
        // chain proves that it is revoked. A missing, malformed, or manually
        // edited state file is never permission to evict a previously live
        // certified bank.
        let prior_health = self.runtime_integrity_health()?;
        if !prior_health.integrity_healthy
            || !prior_health.current_corpus_bound
            || prior_health.certification_status != Some(CertificationStatus::Revoked)
        {
            return Err(BrainError::Integrity(
                "learning_finalization_requires_authenticated_revoked_prior_corpus".into(),
            ));
        }
        let prior_runtime_artifacts = self.snapshot_revoked_prior_runtime_artifacts()?;

        // Materialization is a pre-transition, content-addressed preparation
        // step.  It intentionally happens only after idempotent receipt replay
        // and prior-corpus authority checks; the loader below rederives it
        // without writing when a completed receipt already exists.
        let prospective_mixtures = prospective.skill_source_mixtures.clone();
        self.materialize_dense_fields(
            &mut prospective.fields,
            &prospective_mixtures,
            &observations,
        )?;
        let (_, prospective_report_bytes) = normalize_reconstruction_report_wire(&prospective)?;
        let prospective_report_sha256 = Sha256Digest::digest_bytes(&prospective_report_bytes);

        let operation_key = learning_finalization_operation_key(
            &learning_finalization_input_sha256,
            &prior_corpus_digest,
            &new_corpus_digest,
        )?;
        let receipt_path = finalizations_dir.join(format!("{operation_key}.json"));
        let transitions_root = self.root.join("state/corpus_transitions");
        let transaction_dir = transitions_root
            .join("inflight")
            .join(operation_key.as_str());
        let archive_dir = transitions_root
            .join("by-operation")
            .join(operation_key.as_str());

        if fs::symlink_metadata(&receipt_path).is_ok() {
            return Err(BrainError::Integrity(
                "learning_finalization_operation_key_collision".into(),
            ));
        }
        if fs::symlink_metadata(&transaction_dir).is_ok()
            || fs::symlink_metadata(&archive_dir).is_ok()
        {
            return Err(BrainError::Integrity(
                "learning_finalization_incomplete_transition_requires_explicit_recovery".into(),
            ));
        }

        ensure_private_directory(&self.root, &finalizations_dir)?;
        ensure_private_directory(&self.root, &transitions_root.join("inflight"))?;
        ensure_private_directory(&self.root, &transitions_root.join("by-operation"))?;
        ensure_private_directory(&self.root, &transaction_dir)?;
        let intent = LearningCorpusTransitionIntent {
            schema: "cerebro.tidex.learning_corpus_transition_intent/v1".into(),
            operation_key: operation_key.clone(),
            session_id: input.session_id.clone(),
            adaptive_receipt_sha256: input.adaptive_receipt_sha256.clone(),
            learning_finalization_input_sha256: learning_finalization_input_sha256.clone(),
            representation_evidence_receipt: input.representation_evidence_receipt.clone(),
            representation_protocol_sha256: input.representation_protocol_sha256.clone(),
            representation_observation_bindings_sha256: representation_observation_bindings_sha256
                .clone(),
            prior_corpus_digest: prior_corpus_digest.clone(),
            prior_observation_count: prior_observations.len(),
            new_corpus_digest: new_corpus_digest.clone(),
            new_observation_count: observations.len(),
            archive_dir: archive_dir.clone(),
        };
        write_new_private(
            &self.root,
            &transaction_dir.join("intent.json"),
            &serialize_pretty_line(&intent)?,
        )?;
        self.write_transition_journal(&operation_key, CorpusTransitionPhase::IntentRecorded)?;
        self.advance_canonical_engine_head(HeadIncomplete::Set(operation_key.clone()), None, None)?;

        ensure_private_directory(&self.root, &archive_dir)?;
        let observations_dir = self.root.join("state/observations");
        let observations_dir = existing_directory_under_root(&self.root, &observations_dir)
            .map_err(|_| {
                BrainError::Integrity(
                    "learning_finalization_active_observation_directory_invalid".into(),
                )
            })?;
        let archived_observations_dir = archive_dir.join("observations");
        let observations_identity = inspect_private_directory(&self.root, &observations_dir)?;
        move_private_directory_transactional(
            &self.root,
            &observations_dir,
            &archived_observations_dir,
            &observations_identity,
        )?;
        let mut archived_artifact_sha256 = BTreeMap::<String, Sha256Digest>::new();
        let prior_observation_manifest = serialize_pretty_line(&prior_observations)?;
        let prior_observation_manifest_sha =
            Sha256Digest::digest_bytes(&prior_observation_manifest);
        write_new_private(
            &self.root,
            &archive_dir.join("observations_manifest.json"),
            &prior_observation_manifest,
        )?;
        archived_artifact_sha256.insert(
            "observations_manifest.json".into(),
            prior_observation_manifest_sha,
        );
        for (source, label) in [
            (self.bank_path(), "skill_bank.json"),
            (
                self.root.join("state/shadow_skill_bank.json"),
                "shadow_skill_bank.json",
            ),
            (
                self.root.join("state/memory/current.json"),
                "memory_current.json",
            ),
            (self.root.join("state/sleep_state.json"), "sleep_state.json"),
            (
                self.root.join("state/sleep_evidence/current.json"),
                "sleep_evidence_current.json",
            ),
        ] {
            archive_private_state_file(
                &self.root,
                &source,
                &archive_dir.join(label),
                label,
                prior_runtime_artifacts.get(label),
                &mut archived_artifact_sha256,
            )?;
        }
        self.write_transition_journal(&operation_key, CorpusTransitionPhase::PriorArchived)?;

        // The new corpus is installed only after all old live pointers have
        // been archived. `commit` will independently materialize its shadow
        // state and write a transaction/ledger receipt for this exact corpus.
        self.write_transition_journal(&operation_key, CorpusTransitionPhase::NewCorpusStaged)?;
        let persisted = self.persist_observations(&observations)?;
        if Sha256Digest::parse(digest_json(&persisted)?)? != new_corpus_digest {
            return Err(BrainError::Integrity(
                "learning_finalization_new_observation_digest_mismatch".into(),
            ));
        }
        self.write_transition_journal(&operation_key, CorpusTransitionPhase::NewCorpusPublished)?;
        let report = self.commit_after_verified_corpus_transition(&observations)?;
        let report_bytes = serialize_pretty_line(&report)?;
        let report_sha256 = Sha256Digest::digest_bytes(&report_bytes);
        if report_sha256 != prospective_report_sha256
            || report.observation_set_digest != new_corpus_digest.as_str()
            || !report.promotion.allowed
        {
            return Err(BrainError::Integrity(
                "learning_finalization_committed_report_mismatch".into(),
            ));
        }
        let commit_operation_key = Sha256Digest::parse(commit_operation_key(
            new_corpus_digest.as_str(),
            &report,
            report_sha256.as_str(),
        ))?;
        let commit_path = self
            .root
            .join("state/commits")
            .join(format!("{commit_operation_key}.json"));
        let commit_bytes =
            read_existing_private_file_bounded(&self.root, &commit_path, MAX_ENGINE_JSON_BYTES)?;
        let commit_receipt_sha256 = Sha256Digest::digest_bytes(&commit_bytes);
        self.write_transition_journal(&operation_key, CorpusTransitionPhase::CommitSealed)?;

        let event = ledger::append(
            &self.root,
            "learning_corpus_transition",
            json!({
                "schema":"cerebro.tidex.learning_corpus_transition/v1",
                "operation_key":operation_key,
                "session_id":input.session_id,
                "adaptive_receipt_sha256":input.adaptive_receipt_sha256,
                "learning_finalization_input_sha256":learning_finalization_input_sha256,
                "representation_evidence_receipt_path":input.representation_evidence_receipt.path,
                "representation_evidence_receipt_sha256":input.representation_evidence_receipt.sha256,
                "representation_protocol_sha256":input.representation_protocol_sha256,
                "representation_observation_bindings_sha256":representation_observation_bindings_sha256,
                "prior_corpus_digest":prior_corpus_digest,
                "prior_observation_count":prior_observations.len(),
                "new_corpus_digest":new_corpus_digest,
                "new_observation_count":observations.len(),
                "report_sha256":report_sha256,
                "commit_operation_key":commit_operation_key,
                "commit_receipt_sha256":commit_receipt_sha256,
                "archived_artifact_sha256":archived_artifact_sha256,
            }),
        )?;
        let receipt = LearningFinalizationReceipt {
            schema: "cerebro.tidex.learning_finalization_receipt/v1".into(),
            operation_key,
            session_id: input.session_id,
            adaptive_receipt_sha256: input.adaptive_receipt_sha256,
            learning_finalization_input_sha256,
            representation_evidence_receipt: input.representation_evidence_receipt,
            representation_protocol_sha256: input.representation_protocol_sha256,
            representation_observation_bindings_sha256,
            representation_observation_bindings: input.representation_observation_bindings,
            prior_corpus_digest,
            prior_observation_count: prior_observations.len(),
            new_corpus_digest,
            new_observation_count: observations.len(),
            archived_artifact_sha256,
            report_sha256,
            commit_operation_key,
            commit_receipt_sha256,
            ledger_event_hash: Sha256Digest::parse(event.event_hash)?,
        };
        let receipt_bytes = serialize_pretty_line(&receipt)?;
        let receipt_sha256 = Sha256Digest::digest_bytes(&receipt_bytes);
        write_new_private(&self.root, &receipt_path, &receipt_bytes)?;
        let closed_intent = archive_dir.join("transition_intent");
        let transaction_identity = inspect_private_directory(&self.root, &transaction_dir)?;
        move_private_directory_transactional(
            &self.root,
            &transaction_dir,
            &closed_intent,
            &transaction_identity,
        )?;
        self.write_transition_journal(
            &receipt.operation_key,
            CorpusTransitionPhase::ReceiptSealed,
        )?;
        self.advance_canonical_engine_head(
            HeadIncomplete::Clear,
            None,
            Some(ReportDigest::from(receipt.report_sha256.clone())),
        )?;
        load_verified_learning_finalization_receipt(
            &self.root,
            &PrivateFileReference::new(receipt_path, receipt_sha256),
        )
    }

    /// Read the mutable sleep-evidence pointer only when it has the exact
    /// identity selected for the current transaction. An absent bundle is a
    /// valid revocation input; a newly appearing or replaced bundle is not.
    pub(super) fn verify_current_sleep_evidence_pointer(
        &self,
        expected_sha256: Option<&str>,
    ) -> BrainResult<Option<Vec<u8>>> {
        let path = self.root.join("state/sleep_evidence/current.json");
        match (expected_sha256, fs::symlink_metadata(&path)) {
            (Some(expected), Ok(_)) => {
                let expected = Sha256Digest::parse(expected)?;
                let bytes = PrivateFileReference::new(path, expected)
                    .read_verified_bounded(&self.root, MAX_ENGINE_JSON_BYTES)
                    .map_err(|_| {
                        BrainError::Integrity("sleep_evidence_changed_during_transaction".into())
                    })?;
                Ok(Some(bytes))
            }
            (Some(_), Err(error)) if error.kind() == std::io::ErrorKind::NotFound => Err(
                BrainError::Integrity("sleep_evidence_current_pointer_missing".into()),
            ),
            (Some(_), Err(error)) => Err(error.into()),
            (None, Err(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            (None, Ok(_)) => Err(BrainError::Integrity(
                "sleep_installation_unexpected_current_evidence".into(),
            )),
            (None, Err(error)) => Err(error.into()),
        }
    }

    /// Recheck every live pointer and immutable history a sleep receipt claims
    /// immediately before reporting success. This keeps a concurrent pointer
    /// replacement from becoming an observable successful sleep transaction.
    pub(super) fn verify_current_sleep_installation(
        &self,
        receipt: &SleepReceipt,
        state: &Value,
    ) -> BrainResult<()> {
        let sleep_path = self.root.join("state/sleep_state.json");
        let current_sleep_state =
            read_existing_private_file_bounded(&self.root, &sleep_path, MAX_ENGINE_JSON_BYTES)?;
        verify_sleep_receipt_ledger_binding(
            &self.root,
            receipt,
            state,
            &sha256_bytes(&current_sleep_state),
        )?;
        let report_history = self
            .root
            .join("state/reports")
            .join(format!("{}.json", receipt.report_sha256));
        let memory_history = memory_artifact_path(&self.root, &receipt.memory_sha256);
        let state_history = self
            .root
            .join("state/sleep/by-sha")
            .join(format!("{}.json", receipt.sleep_state_sha256));
        for (path, digest, label) in [
            (&report_history, receipt.report_sha256.as_str(), "report"),
            (&memory_history, receipt.memory_sha256.as_str(), "memory"),
            (&state_history, receipt.sleep_state_sha256.as_str(), "state"),
        ] {
            if !private_file_digest_matches(&self.root, path, digest) {
                return Err(BrainError::Integrity(format!(
                    "sleep_installation_{label}_history_mismatch"
                )));
            }
        }
        if !private_file_digest_matches(
            &self.root,
            &self.root.join("state/memory/current.json"),
            receipt.memory_sha256.as_str(),
        ) {
            return Err(BrainError::Integrity(
                "sleep_installation_current_memory_mismatch".into(),
            ));
        }
        match &receipt.active_bank_sha256 {
            Some(bank_sha) => {
                let bank_history = self
                    .root
                    .join("state/skill_banks/by-sha")
                    .join(format!("{bank_sha}.json"));
                if !private_file_digest_matches(&self.root, &bank_history, bank_sha.as_str())
                    || !private_file_digest_matches(
                        &self.root,
                        &self.bank_path(),
                        bank_sha.as_str(),
                    )
                {
                    return Err(BrainError::Integrity(
                        "sleep_installation_current_bank_mismatch".into(),
                    ));
                }
            }
            None => match fs::symlink_metadata(self.bank_path()) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Ok(_) => {
                    return Err(BrainError::Integrity(
                        "sleep_installation_unexpected_active_bank".into(),
                    ))
                }
                Err(error) => return Err(error.into()),
            },
        }
        match &receipt.evidence_bundle_sha256 {
            Some(evidence_sha) => {
                let evidence_history = self
                    .root
                    .join("state/sleep_evidence/by-sha")
                    .join(format!("{evidence_sha}.json"));
                let _current_evidence = self
                    .verify_current_sleep_evidence_pointer(Some(evidence_sha))?
                    .ok_or_else(|| {
                        BrainError::Integrity("sleep_evidence_current_pointer_missing".into())
                    })?;
                if !private_file_digest_matches(
                    &self.root,
                    &evidence_history,
                    evidence_sha.as_str(),
                ) {
                    return Err(BrainError::Integrity(
                        "sleep_installation_current_evidence_mismatch".into(),
                    ));
                }
            }
            None => {
                self.verify_current_sleep_evidence_pointer(None)?;
            }
        }
        Ok(())
    }

    /// Snapshot the exact live authority artifacts of the authenticated,
    /// revoked prior runtime. Each later archive rename compares its source
    /// bytes to this snapshot, so a pointer replacement after preflight turns
    /// the corpus transition into a visible fail-closed incomplete operation.
    pub(super) fn snapshot_revoked_prior_runtime_artifacts(
        &self,
    ) -> BrainResult<BTreeMap<String, Sha256Digest>> {
        let sleep_path = self.root.join("state/sleep_state.json");
        let state_bytes =
            read_existing_private_file_bounded(&self.root, &sleep_path, MAX_ENGINE_JSON_BYTES)?;
        let state: Value = serde_json::from_slice(&state_bytes)?;
        let operation_key =
            Sha256Digest::parse(required_sleep_state_string(&state, "operation_key")?)?;
        let receipt_path = self
            .root
            .join("state/sleep_receipts")
            .join(format!("{operation_key}.json"));
        let receipt: SleepReceipt =
            read_private_json(&self.root, &receipt_path, MAX_ENGINE_JSON_BYTES)?;
        if receipt.operation_key != operation_key.as_str() {
            return Err(BrainError::Integrity(
                "learning_finalization_prior_sleep_receipt_operation_invalid".into(),
            ));
        }
        self.verify_current_sleep_installation(&receipt, &state)?;

        let active_bank_sha = receipt.active_bank_sha256.as_deref().ok_or_else(|| {
            BrainError::Integrity("learning_finalization_prior_active_bank_missing".into())
        })?;
        let evidence_sha = receipt.evidence_bundle_sha256.as_deref().ok_or_else(|| {
            BrainError::Integrity("learning_finalization_prior_sleep_evidence_missing".into())
        })?;
        let mut artifacts = BTreeMap::new();
        artifacts.insert(
            "skill_bank.json".into(),
            Sha256Digest::parse(active_bank_sha)?,
        );
        artifacts.insert(
            "memory_current.json".into(),
            Sha256Digest::parse(&receipt.memory_sha256)?,
        );
        artifacts.insert(
            "sleep_state.json".into(),
            Sha256Digest::parse(&receipt.sleep_state_sha256)?,
        );
        artifacts.insert(
            "sleep_evidence_current.json".into(),
            Sha256Digest::parse(evidence_sha)?,
        );
        Ok(artifacts)
    }

    /// The absence of an active bank is an initialization condition, never a
    /// recovery fallback. Once a sleep state exists, it must authenticate an
    /// already-revoked, explicitly bankless runtime before sleep is permitted
    /// to construct an empty bank again.
    pub(super) fn authorize_empty_bank_bootstrap(
        &self,
        has_prior_sleep_state: bool,
        previous: &Value,
    ) -> BrainResult<()> {
        if !has_prior_sleep_state {
            return Ok(());
        }
        if required_sleep_state_string(previous, "schema")? != "cerebro.tidex.sleep_state/v5"
            || CertificationStatus::parse(required_sleep_state_string(
                previous,
                "certification_status",
            )?) != Some(CertificationStatus::Revoked)
            || previous.get("active_skill_count").and_then(Value::as_u64) != Some(0)
            || previous.get("promoted").and_then(Value::as_bool) != Some(false)
            || previous.get("active_bank_sha256") != Some(&Value::Null)
        {
            return Err(BrainError::Integrity(
                "empty_bank_bootstrap_prior_state_not_explicitly_revoked".into(),
            ));
        }
        let operation_key =
            Sha256Digest::parse(required_sleep_state_string(previous, "operation_key")?)?;
        let receipt_path = self
            .root
            .join("state/sleep_receipts")
            .join(format!("{operation_key}.json"));
        let receipt: SleepReceipt =
            read_private_json(&self.root, &receipt_path, MAX_ENGINE_JSON_BYTES)?;
        if receipt.operation_key != operation_key.as_str() || receipt.active_bank_sha256.is_some() {
            return Err(BrainError::Integrity(
                "empty_bank_bootstrap_prior_receipt_invalid".into(),
            ));
        }
        self.verify_current_sleep_installation(&receipt, previous)
    }

    pub fn sleep_cycle(&self) -> BrainResult<SleepReport> {
        self.with_engine_authority(|| self.sleep_cycle_under_authority())
    }

    pub(super) fn sleep_cycle_under_authority(&self) -> BrainResult<SleepReport> {
        self.require_canonical_runtime_config()?;
        self.require_no_incomplete_corpus_transition()?;
        let observations = canonical_observations(&self.load_persisted_observations()?);
        if observations.len() < self.config.min_observations {
            return Err(BrainError::Invalid(
                "sleep_requires_persisted_observations".into(),
            ));
        }
        let mut reconstruction = self.analyze_canonical(&observations)?;
        // Dense materialization is deterministic from authenticated observations
        // and is not an activation.  It must happen before evidence verification
        // so replay/trust artifacts bind to the exact report that sleep may later
        // promote; otherwise the evidence can only self-attest a sibling report.
        if reconstruction.promotion.allowed {
            let mixtures = reconstruction.skill_source_mixtures.clone();
            self.materialize_dense_fields(&mut reconstruction.fields, &mixtures, &observations)?;
        }
        // Sleep and learning-finalization must assign one exact identity to the
        // same canonical reconstruction. Use the shared exact wire authority.
        let normalized = normalize_reconstruction_report_wire(&reconstruction)?;
        reconstruction = normalized.0;
        let report_bytes = normalized.1;
        let report_sha256 = ReportDigest::from(Sha256Digest::digest_bytes(&report_bytes));
        let corpus_digest = reconstruction.observation_set_digest.clone();
        let analysis_key = sleep_analysis_key(&corpus_digest, &reconstruction);
        let sleep_path = self.root.join("state/sleep_state.json");
        let (previous, has_prior_sleep_state) = match fs::symlink_metadata(&sleep_path) {
            Ok(_) => {
                let bytes = read_existing_private_file_bounded(
                    &self.root,
                    &sleep_path,
                    MAX_ENGINE_JSON_BYTES,
                )?;
                (serde_json::from_slice::<serde_json::Value>(&bytes)?, true)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (json!({}), false),
            Err(error) => return Err(error.into()),
        };

        // This is the one explicit initialization transition. No persisted
        // bank is substituted or accepted as a fallback.
        let prior_bank = match self.load_bank_for_initial_sleep_bootstrap()? {
            Some(bank) => bank,
            None => {
                self.authorize_empty_bank_bootstrap(has_prior_sleep_state, &previous)?;
                SkillBank::default()
            }
        };
        let diagnostics = diagnose_consolidation(
            &prior_bank,
            &reconstruction.fields,
            self.config.skill_match_cosine,
        )?;
        let evidence_path = self.root.join("state/sleep_evidence/current.json");
        let evidence_bundle_sha256 = match fs::symlink_metadata(&evidence_path) {
            Ok(_) => {
                let bytes = read_existing_private_file_bounded(
                    &self.root,
                    &evidence_path,
                    MAX_ENGINE_JSON_BYTES,
                )?;
                Some(EvidenceBundleDigest::from(Sha256Digest::digest_bytes(
                    &bytes,
                )))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let evidence_verification = match load_sleep_evidence(&self.root) {
            Ok(bundle) => verify_sleep_evidence(
                &self.root,
                &bundle,
                &SleepEvidenceExpectation {
                    corpus_digest: &corpus_digest,
                    report_sha256: &report_sha256,
                    source_tree_digest: &reconstruction.source_tree_digest,
                    config_digest: &reconstruction.config_digest,
                    analysis_version_digest: &reconstruction.analysis_version_digest,
                    fields: &reconstruction.fields,
                    observations: &observations,
                    source_mixtures: &reconstruction.skill_source_mixtures,
                },
            )?,
            Err(error) => SleepEvidenceVerification::failed(error.to_string()),
        };
        let certified = reconstruction.promotion.allowed && evidence_verification.verified;
        let certification_status = if certified {
            CertificationStatus::Certified
        } else {
            CertificationStatus::Revoked
        };
        let previous_same_certified = previous
            .get("analysis_key")
            .and_then(serde_json::Value::as_str)
            == Some(analysis_key.as_str())
            && previous
                .get("certification_status")
                .and_then(serde_json::Value::as_str)
                == Some(CertificationStatus::Certified.as_str());
        let should_promote = certified && !previous_same_certified;

        let mut bank = prior_bank.clone();
        let promoted = if should_promote {
            reconcile_full_corpus(
                &mut bank,
                &reconstruction.fields,
                self.config.skill_match_cosine,
            )?;
            true
        } else {
            false
        };
        let memory_fields = align_incoming_identities(
            &bank,
            &reconstruction.fields,
            self.config.skill_match_cosine,
        )?;
        let memory = build_memory_snapshot(
            &observations,
            bank.generation,
            certified,
            &memory_fields,
            &reconstruction.skill_source_mixtures,
        )?;
        let memory_bytes = serialize_pretty_line(&memory)?;
        let memory_sha256 = MemoryDigest::from(Sha256Digest::digest_bytes(&memory_bytes));

        let bank_bytes = if bank.fields.is_empty() {
            None
        } else if promoted {
            Some(serialize_pretty_line(&bank)?)
        } else {
            // A revoked/non-promoting sleep must bind to the exact active-bank
            // bytes already installed. Re-serializing an older schema under the
            // current struct can change bytes without changing semantics and
            // would make the receipt point at a bank that was never activated.
            match fs::symlink_metadata(self.bank_path()) {
                Ok(_) => Some(read_existing_private_file_bounded(
                    &self.root,
                    &self.bank_path(),
                    MAX_SKILL_BANK_BYTES,
                )?),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => return Err(error.into()),
            }
        };
        let active_bank_sha256 = bank_bytes
            .as_ref()
            .map(|bytes| SkillBankDigest::from(Sha256Digest::digest_bytes(bytes)));
        let operation_key = sleep_operation_key(
            &analysis_key,
            evidence_bundle_sha256.as_deref(),
            certification_status,
            active_bank_sha256.as_deref(),
        );
        // `promoted` describes what this invocation did. The persisted sleep
        // state, however, is part of the immutable transaction identified by
        // `operation_key`. Replaying that exact operation must preserve the
        // original state bytes (including promotion and pre-promotion
        // diagnostics) so receipt recovery and idempotency cannot manufacture a
        // sibling state under one operation identity.
        let replaying_same_operation =
            previous.get("operation_key").and_then(Value::as_str) == Some(operation_key.as_str());
        let transaction_promoted = if replaying_same_operation {
            previous
                .get("promoted")
                .and_then(Value::as_bool)
                .ok_or_else(|| BrainError::Integrity("sleep_state_promoted_invalid".into()))?
        } else {
            promoted
        };
        let state = if replaying_same_operation {
            previous.clone()
        } else {
            json!({
                "schema":"cerebro.tidex.sleep_state/v5",
                "operation_key":operation_key,
                "analysis_key":analysis_key,
                "corpus_digest":corpus_digest,
                "source_tree_digest":reconstruction.source_tree_digest,
                "config_digest":reconstruction.config_digest,
                "analysis_version_digest":reconstruction.analysis_version_digest,
                "report_sha256":report_sha256,
                "promoted":transaction_promoted,
                "certification_status":certification_status,
                "active_generation":bank.generation,
                "active_skill_count":bank.fields.len(),
                "active_bank_sha256":active_bank_sha256,
                "memory_digest":memory_sha256,
                "evidence_bundle_sha256":evidence_bundle_sha256,
                "evidence_verification":evidence_verification,
                "diagnostics":diagnostics,
            })
        };
        let state_bytes = serialize_pretty_line(&state)?;
        let state_sha256 = sha256_bytes(&state_bytes);

        let receipts_dir = self.root.join("state/sleep_receipts");
        let transactions_dir = self.root.join("state/sleep_transactions");
        ensure_private_directory(&self.root, &receipts_dir)?;
        ensure_private_directory(&self.root, &transactions_dir)?;
        let receipt_path = receipts_dir.join(format!("{operation_key}.json"));
        let receipt_exists = match fs::symlink_metadata(&receipt_path) {
            Ok(_) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(error.into()),
        };
        if receipt_exists {
            let receipt: SleepReceipt =
                read_private_json(&self.root, &receipt_path, MAX_ENGINE_JSON_BYTES)?;
            if receipt.operation_key != operation_key
                || receipt.analysis_key != analysis_key
                || receipt.report_sha256 != report_sha256
                || receipt.memory_sha256 != memory_sha256
                || receipt.active_bank_sha256 != active_bank_sha256
                || receipt.evidence_bundle_sha256 != evidence_bundle_sha256
                || receipt.sleep_state_sha256 != state_sha256
            {
                return Err(BrainError::Integrity("sleep_receipt_mismatch".into()));
            }
            let current_sleep_state =
                read_existing_private_file_bounded(&self.root, &sleep_path, MAX_ENGINE_JSON_BYTES)?;
            verify_sleep_receipt_ledger_binding(
                &self.root,
                &receipt,
                &state,
                &sha256_bytes(&current_sleep_state),
            )?;
            let report_history = self
                .root
                .join("state/reports")
                .join(format!("{report_sha256}.json"));
            let memory_history = memory_artifact_path(&self.root, &memory_sha256);
            let state_history = self
                .root
                .join("state/sleep/by-sha")
                .join(format!("{state_sha256}.json"));
            for (path, digest, label) in [
                (&report_history, report_sha256.as_str(), "report"),
                (&memory_history, memory_sha256.as_str(), "memory"),
                (&state_history, state_sha256.as_str(), "state"),
            ] {
                if !private_file_digest_matches(&self.root, path, digest) {
                    return Err(BrainError::Integrity(format!(
                        "sleep_receipt_{label}_artifact_mismatch"
                    )));
                }
            }
            if let Some(bank_sha) = &active_bank_sha256 {
                let bank_history = self
                    .root
                    .join("state/skill_banks/by-sha")
                    .join(format!("{bank_sha}.json"));
                if !private_file_digest_matches(&self.root, &bank_history, bank_sha.as_str()) {
                    return Err(BrainError::Integrity(
                        "sleep_receipt_bank_artifact_mismatch".into(),
                    ));
                }
            }
            if let Some(evidence_sha) = &receipt.evidence_bundle_sha256 {
                let evidence_history = self
                    .root
                    .join("state/sleep_evidence/by-sha")
                    .join(format!("{evidence_sha}.json"));
                if !private_file_digest_matches(
                    &self.root,
                    &evidence_history,
                    evidence_sha.as_str(),
                ) {
                    return Err(BrainError::Integrity(
                        "sleep_receipt_evidence_artifact_mismatch".into(),
                    ));
                }
            }
            self.verify_current_sleep_installation(&receipt, &state)?;
            self.advance_canonical_engine_head(
                HeadIncomplete::Preserve,
                Some(certification_status.as_str()),
                Some(report_sha256.clone()),
            )?;
            return Ok(SleepReport {
                schema: "cerebro.tidex.sleep/v4".into(),
                corpus_digest,
                observation_count: observations.len(),
                promoted: false,
                idempotent: true,
                active_skill_count: bank.fields.len(),
                memory_digest: memory_sha256,
                evidence_bundle_sha256,
                evidence_verification,
                diagnostics,
                reconstruction,
            });
        }

        let transaction_dir = transactions_dir.join(&operation_key);
        let intent_path = transaction_dir.join("intent.json");
        let staged_report = transaction_dir.join("report.json");
        let staged_memory = transaction_dir.join("memory.json");
        let staged_state = transaction_dir.join("sleep_state.json");
        let staged_bank = transaction_dir.join("skill_bank.json");
        let intent = SleepTransactionIntent {
            schema: "cerebro.tidex.sleep_transaction_intent/v1".into(),
            operation_key: operation_key.clone(),
            analysis_key: analysis_key.clone(),
            corpus_digest: corpus_digest.clone(),
            analysis_version_digest: reconstruction.analysis_version_digest.clone(),
            config_digest: reconstruction.config_digest.clone(),
            report_sha256: report_sha256.clone(),
            memory_sha256: memory_sha256.clone(),
            active_bank_sha256: active_bank_sha256.clone(),
            sleep_state_sha256: state_sha256.clone(),
            evidence_bundle_sha256: evidence_bundle_sha256.clone(),
            evidence_verified: evidence_verification.verified,
            certification_status,
            promoted: transaction_promoted,
        };
        let transaction_exists = match fs::symlink_metadata(&transaction_dir) {
            Ok(_) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(error.into()),
        };
        if transaction_exists {
            let transaction_dir = existing_directory_under_root(&self.root, &transaction_dir)?;
            let stored: SleepTransactionIntent =
                read_private_json(&self.root, &intent_path, MAX_ENGINE_JSON_BYTES)?;
            if stored != intent
                || !private_file_digest_matches(&self.root, &staged_report, report_sha256.as_str())
                || !private_file_digest_matches(&self.root, &staged_memory, memory_sha256.as_str())
                || !private_file_digest_matches(&self.root, &staged_state, state_sha256.as_str())
                || active_bank_sha256.as_ref().is_some_and(|sha| {
                    !private_file_digest_matches(&self.root, &staged_bank, sha.as_str())
                })
            {
                return Err(BrainError::Integrity(
                    "sleep_transaction_stage_mismatch".into(),
                ));
            }
            let _ = transaction_dir;
        } else {
            ensure_private_directory(&self.root, &transaction_dir)?;
            write_new_private(&self.root, &staged_report, &report_bytes)?;
            write_new_private(&self.root, &staged_memory, &memory_bytes)?;
            write_new_private(&self.root, &staged_state, &state_bytes)?;
            if let Some(bytes) = &bank_bytes {
                write_new_private(&self.root, &staged_bank, bytes)?;
            }
            write_new_private(&self.root, &intent_path, &serialize_pretty_line(&intent)?)?;
        }

        // The ledger event authorizes pointer changes. Recheck the mutable
        // evidence pointer immediately before that authorization, so an
        // external evidence replacement cannot leave a half-applied sleep
        // transaction merely by racing the earlier measurement.
        self.verify_current_sleep_evidence_pointer(evidence_bundle_sha256.as_deref())?;

        let event = if let Some(existing) = ledger::find_v2_event_by_payload_string(
            &self.root,
            "sleep_transaction",
            "operation_key",
            &operation_key,
        )? {
            verify_sleep_transaction_ledger_binding(&existing, &intent)?;
            existing
        } else {
            let event = ledger::append(
                &self.root,
                "sleep_transaction",
                json!({
                    "schema":"cerebro.tidex.sleep_transaction/v1",
                    "operation_key":operation_key,
                    "analysis_key":analysis_key,
                    "corpus_digest":corpus_digest,
                    "analysis_version_digest":reconstruction.analysis_version_digest,
                    "config_digest":reconstruction.config_digest,
                    "report_sha256":report_sha256,
                    "memory_sha256":memory_sha256,
                    "active_bank_sha256":active_bank_sha256,
                    "sleep_state_sha256":state_sha256,
                    "evidence_bundle_sha256":evidence_bundle_sha256,
                    "evidence_verified":evidence_verification.verified,
                    "certification_status":certification_status,
                    "promoted":transaction_promoted,
                }),
            )?;
            verify_sleep_transaction_ledger_binding(&event, &intent)?;
            event
        };

        let report_history = self
            .root
            .join("state/reports")
            .join(format!("{report_sha256}.json"));
        write_immutable_exact(&self.root, &report_history, &report_bytes, &report_sha256)?;
        let memory_history = memory_artifact_path(&self.root, &memory_sha256);
        write_immutable_exact(&self.root, &memory_history, &memory_bytes, &memory_sha256)?;
        replace_private_pointer_exact(
            &self.root,
            &self.root.join("state/memory/current.json"),
            &memory_bytes,
            &memory_sha256,
        )?;
        if let (Some(bytes), Some(bank_sha)) = (&bank_bytes, &active_bank_sha256) {
            let bank_history = self
                .root
                .join("state/skill_banks/by-sha")
                .join(format!("{bank_sha}.json"));
            write_immutable_exact(&self.root, &bank_history, bytes, bank_sha)?;
            if promoted {
                replace_private_pointer_exact(&self.root, &self.bank_path(), bytes, bank_sha)?;
            }
        }
        if let Some(evidence_sha) = &evidence_bundle_sha256 {
            let evidence_bytes = self
                .verify_current_sleep_evidence_pointer(Some(evidence_sha))?
                .ok_or_else(|| {
                    BrainError::Integrity("sleep_evidence_current_pointer_missing".into())
                })?;
            let evidence_history = self
                .root
                .join("state/sleep_evidence/by-sha")
                .join(format!("{evidence_sha}.json"));
            write_immutable_exact(&self.root, &evidence_history, &evidence_bytes, evidence_sha)?;
        }
        let state_history = self
            .root
            .join("state/sleep/by-sha")
            .join(format!("{state_sha256}.json"));
        write_immutable_exact(&self.root, &state_history, &state_bytes, &state_sha256)?;
        replace_private_pointer_exact(&self.root, &sleep_path, &state_bytes, &state_sha256)?;

        let receipt = SleepReceipt {
            schema: "cerebro.tidex.sleep_receipt/v1".into(),
            operation_key,
            analysis_key,
            report_sha256: report_sha256.clone(),
            memory_sha256: memory_sha256.clone(),
            active_bank_sha256,
            evidence_bundle_sha256: evidence_bundle_sha256.clone(),
            sleep_state_sha256: state_sha256,
            ledger_event_hash: event.event_hash,
        };
        write_new_private(&self.root, &receipt_path, &serialize_pretty_line(&receipt)?)?;
        // Reopen the current pointer through the same receipt/ledger verifier
        // before this sleep result is handed back.  A concurrent state change
        // therefore fails closed instead of returning a transient success.
        self.verify_current_sleep_installation(&receipt, &state)?;
        self.advance_canonical_engine_head(
            HeadIncomplete::Preserve,
            Some(certification_status.as_str()),
            Some(report_sha256.clone()),
        )?;
        Ok(SleepReport {
            schema: "cerebro.tidex.sleep/v4".into(),
            corpus_digest,
            observation_count: observations.len(),
            promoted,
            idempotent: replaying_same_operation,
            active_skill_count: bank.fields.len(),
            memory_digest: memory_sha256,
            evidence_bundle_sha256,
            evidence_verification,
            diagnostics,
            reconstruction,
        })
    }
}
