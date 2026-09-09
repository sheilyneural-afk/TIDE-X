use super::*;

pub(super) fn canonical_head_compare_and_swap_matches(
    expected: Option<&CanonicalEngineHead>,
    current: Option<&CanonicalEngineHead>,
) -> bool {
    match (expected, current) {
        (None, None) => true,
        (Some(expected), Some(actual)) => expected.manifest_digest == actual.manifest_digest,
        _ => false,
    }
}

impl BrainEngine {
    pub(super) fn canonical_engine_head_path(&self) -> PathBuf {
        self.root.join("state/canonical_engine_head.json")
    }

    pub(super) fn canonical_engine_head_history_path(
        &self,
        digest: &CanonicalEngineHeadDigest,
    ) -> PathBuf {
        self.root
            .join("state/canonical_engine_heads/by-sha")
            .join(format!("{}.json", digest.as_str()))
    }

    pub(super) fn persist_canonical_engine_head_history(
        &self,
        head: &CanonicalEngineHead,
    ) -> BrainResult<()> {
        head.authenticate()?;
        let bytes = serialize_pretty_line(head)?;
        let path = self.canonical_engine_head_history_path(&head.manifest_digest);
        write_new_private(&self.root, &path, &bytes)?;
        let persisted: CanonicalEngineHead =
            read_private_json(&self.root, &path, CANONICAL_ENGINE_HEAD_MAX_BYTES)?;
        persisted.authenticate()?;
        if persisted != *head {
            return Err(BrainError::Integrity(
                "canonical_engine_head_history_content_mismatch".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn verify_current_canonical_engine_head(&self) -> BrainResult<CanonicalEngineHead> {
        let head = self
            .load_canonical_head_if_present()?
            .ok_or_else(|| BrainError::Integrity("canonical_engine_head_missing".into()))?;

        let historical: CanonicalEngineHead = read_private_json(
            &self.root,
            &self.canonical_engine_head_history_path(&head.manifest_digest),
            CANONICAL_ENGINE_HEAD_MAX_BYTES,
        )?;
        historical.authenticate()?;
        if historical != head {
            return Err(BrainError::Integrity(
                "canonical_engine_head_history_current_mismatch".into(),
            ));
        }
        if let Some(parent_digest) = &head.parent_digest {
            let parent: CanonicalEngineHead = read_private_json(
                &self.root,
                &self.canonical_engine_head_history_path(parent_digest),
                CANONICAL_ENGINE_HEAD_MAX_BYTES,
            )?;
            parent.authenticate()?;
            if &parent.manifest_digest != parent_digest
                || Some(parent.revision) != head.parent_revision
                || parent.revision.checked_add(1) != Some(head.revision)
            {
                return Err(BrainError::Integrity(
                    "canonical_engine_head_parent_history_mismatch".into(),
                ));
            }
        }

        let observations = self.load_persisted_observations()?;
        let (corpus_digest, observation_count) = if observations.is_empty() {
            (None, 0)
        } else {
            (
                Some(observation_set_digest(&observations)?),
                observations.len(),
            )
        };
        let active_bank_sha256 = self
            .optional_pointer_digest(&self.bank_path())?
            .map(SkillBankDigest::from);
        let memory_sha256 = self
            .optional_pointer_digest(&self.root.join("state/memory/current.json"))?
            .map(MemoryDigest::from);
        let sleep_state_sha256 =
            self.optional_pointer_digest(&self.root.join("state/sleep_state.json"))?;
        let evidence_bundle_sha256 = self
            .optional_pointer_digest(&self.root.join("state/sleep_evidence/current.json"))?
            .map(EvidenceBundleDigest::from);
        let incomplete_transition = self.incomplete_transition_operation_key()?;

        if head.corpus_digest != corpus_digest
            || head.observation_count != observation_count
            || head.active_bank_sha256 != active_bank_sha256
            || head.memory_sha256 != memory_sha256
            || head.sleep_state_sha256 != sleep_state_sha256
            || head.evidence_bundle_sha256 != evidence_bundle_sha256
            || head.incomplete_transition != incomplete_transition
        {
            return Err(BrainError::Integrity(
                "canonical_engine_head_live_authority_mismatch".into(),
            ));
        }

        let sleep_path = self.root.join("state/sleep_state.json");
        match read_untrusted_private_file_bounded(&self.root, &sleep_path, MAX_ENGINE_JSON_BYTES) {
            Ok(bytes) => {
                let state: Value = serde_json::from_slice(&bytes)?;
                if state.get("schema").and_then(Value::as_str)
                    != Some("cerebro.tidex.sleep_state/v5")
                {
                    return Err(BrainError::Integrity(
                        "canonical_engine_head_sleep_state_schema_invalid".into(),
                    ));
                }
                let certification_status = state
                    .get("certification_status")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let report_sha256 = state
                    .get("report_sha256")
                    .and_then(Value::as_str)
                    .map(|value| Sha256Digest::parse(value).map(ReportDigest::from))
                    .transpose()?;
                if head.certification_status != certification_status
                    || head.reconstruction_report_sha256 != report_sha256
                {
                    return Err(BrainError::Integrity(
                        "canonical_engine_head_sleep_authority_mismatch".into(),
                    ));
                }
            }
            Err(BrainError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                if head.certification_status.is_some() {
                    return Err(BrainError::Integrity(
                        "canonical_engine_head_certification_without_sleep_state".into(),
                    ));
                }
            }
            Err(error) => return Err(error),
        }
        Ok(head)
    }

    pub(super) fn transition_journal_path(&self, operation_key: &Sha256Digest) -> PathBuf {
        self.root
            .join("state/corpus_transitions/journals")
            .join(format!("{operation_key}.json"))
    }

    /// Read the current canonical head if one has been published. Absence is
    /// valid for stores that have not yet completed a governed mutation.
    pub fn load_canonical_head(&self) -> BrainResult<Option<CanonicalEngineHead>> {
        self.load_canonical_head_if_present()
    }

    pub(super) fn load_canonical_head_if_present(
        &self,
    ) -> BrainResult<Option<CanonicalEngineHead>> {
        let bytes = match read_untrusted_private_file_bounded(
            &self.root,
            &self.canonical_engine_head_path(),
            CANONICAL_ENGINE_HEAD_MAX_BYTES,
        ) {
            Ok(bytes) => bytes,
            Err(BrainError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(None)
            }
            Err(error) => return Err(error),
        };
        let head: CanonicalEngineHead = serde_json::from_slice(&bytes)?;
        head.authenticate()?;
        Ok(Some(head))
    }

    pub(super) fn optional_pointer_digest(&self, path: &Path) -> BrainResult<Option<Sha256Digest>> {
        match read_untrusted_private_file_bounded(&self.root, path, MAX_ENGINE_JSON_BYTES) {
            Ok(bytes) => Ok(Some(Sha256Digest::digest_bytes(&bytes))),
            Err(BrainError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    pub(super) fn publish_canonical_head(
        &self,
        expected: Option<&CanonicalEngineHead>,
        next: &CanonicalEngineHead,
    ) -> BrainResult<()> {
        next.authenticate()?;
        let current = self.load_canonical_head_if_present()?;
        if !canonical_head_compare_and_swap_matches(expected, current.as_ref()) {
            return Err(BrainError::Integrity(
                "canonical_engine_head_compare_and_swap_conflict".into(),
            ));
        }
        if let Some(current) = current.as_ref() {
            self.persist_canonical_engine_head_history(current)?;
        }
        self.persist_canonical_engine_head_history(next)?;
        let bytes = serialize_pretty_line(next)?;
        let digest = Sha256Digest::digest_bytes(&bytes);
        replace_private_file_atomic(
            &self.root,
            &self.canonical_engine_head_path(),
            &bytes,
            Some(&digest),
        )?;
        self.verify_current_canonical_engine_head()?;
        Ok(())
    }

    pub(super) fn snapshot_canonical_engine_head(
        &self,
        revision: u64,
        parent: Option<&CanonicalEngineHead>,
        incomplete_transition: Option<Sha256Digest>,
        certification_status: Option<&str>,
        reconstruction_report_sha256: Option<ReportDigest>,
    ) -> BrainResult<CanonicalEngineHead> {
        let observations = self.load_persisted_observations()?;
        let (corpus_digest, observation_count) = if observations.is_empty() {
            (None, 0)
        } else {
            (
                Some(observation_set_digest(&observations)?),
                observations.len(),
            )
        };
        let sleep_path = self.root.join("state/sleep_state.json");
        let (live_certification_status, live_report_sha256) =
            match read_untrusted_private_file_bounded(
                &self.root,
                &sleep_path,
                MAX_ENGINE_JSON_BYTES,
            ) {
                Ok(bytes) => {
                    let state: Value = serde_json::from_slice(&bytes)?;
                    if state.get("schema").and_then(Value::as_str)
                        != Some("cerebro.tidex.sleep_state/v5")
                    {
                        return Err(BrainError::Integrity(
                            "canonical_engine_head_sleep_state_schema_invalid".into(),
                        ));
                    }
                    let certification = state
                        .get("certification_status")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    let report = state
                        .get("report_sha256")
                        .and_then(Value::as_str)
                        .map(|value| Sha256Digest::parse(value).map(ReportDigest::from))
                        .transpose()?;
                    (certification, report)
                }
                Err(BrainError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                    (None, None)
                }
                Err(error) => return Err(error),
            };
        CanonicalEngineHead {
            schema: CANONICAL_ENGINE_HEAD_SCHEMA.into(),
            revision,
            parent_revision: parent.map(|head| head.revision),
            parent_digest: parent.map(|head| head.manifest_digest.clone()),
            corpus_digest,
            observation_count,
            active_bank_sha256: self
                .optional_pointer_digest(&self.bank_path())?
                .map(SkillBankDigest::from),
            memory_sha256: self
                .optional_pointer_digest(&self.root.join("state/memory/current.json"))?
                .map(MemoryDigest::from),
            sleep_state_sha256: self
                .optional_pointer_digest(&self.root.join("state/sleep_state.json"))?,
            evidence_bundle_sha256: self
                .optional_pointer_digest(&self.root.join("state/sleep_evidence/current.json"))?
                .map(EvidenceBundleDigest::from),
            reconstruction_report_sha256: live_report_sha256.or(reconstruction_report_sha256),
            certification_status: live_certification_status
                .or_else(|| certification_status.map(str::to_string)),
            incomplete_transition,
            manifest_digest: CanonicalEngineHeadDigest::from(Sha256Digest::zero()),
        }
        .seal()
    }

    pub(super) fn advance_canonical_engine_head(
        &self,
        incomplete: HeadIncomplete,
        certification_status: Option<&str>,
        reconstruction_report_sha256: Option<ReportDigest>,
    ) -> BrainResult<CanonicalEngineHead> {
        let expected = self.load_canonical_head_if_present()?;
        let revision = expected.as_ref().map(|head| head.revision + 1).unwrap_or(0);
        if revision > HARD_MAX_ENGINE_REVISION {
            return Err(BrainError::Integrity(
                "canonical_engine_head_revision_limit_exceeded".into(),
            ));
        }
        let incomplete_transition = match incomplete {
            HeadIncomplete::Preserve => expected
                .as_ref()
                .and_then(|head| head.incomplete_transition.clone()),
            HeadIncomplete::Set(key) => Some(key),
            HeadIncomplete::Clear => None,
        };
        let next = self.snapshot_canonical_engine_head(
            revision,
            expected.as_ref(),
            incomplete_transition,
            certification_status,
            reconstruction_report_sha256.or_else(|| {
                expected
                    .as_ref()
                    .and_then(|head| head.reconstruction_report_sha256.clone())
            }),
        )?;
        self.publish_canonical_head(expected.as_ref(), &next)?;
        Ok(next)
    }

    pub(super) fn write_transition_journal(
        &self,
        operation_key: &Sha256Digest,
        phase: CorpusTransitionPhase,
    ) -> BrainResult<()> {
        let parent = self.load_canonical_head_if_present()?;
        let journal = CorpusTransitionJournal::new(operation_key.clone(), phase, parent.as_ref())?;
        journal.authenticate(Some(operation_key))?;
        let bytes = serialize_pretty_line(&journal)?;
        let digest = Sha256Digest::digest_bytes(&bytes);
        replace_private_file_atomic(
            &self.root,
            &self.transition_journal_path(operation_key),
            &bytes,
            Some(&digest),
        )?;
        Ok(())
    }

    pub(super) fn load_transition_journal(
        &self,
        operation_key: &Sha256Digest,
    ) -> BrainResult<Option<CorpusTransitionJournal>> {
        match read_private_json::<CorpusTransitionJournal>(
            &self.root,
            &self.transition_journal_path(operation_key),
            CORPUS_TRANSITION_JOURNAL_MAX_BYTES,
        ) {
            Ok(journal) => {
                journal.authenticate(Some(operation_key))?;
                Ok(Some(journal))
            }
            Err(BrainError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    pub(super) fn load_parameter_layout(&self, digest: &str) -> BrainResult<ParameterBlockLayout> {
        if !valid_digest(digest) {
            return Err(BrainError::Invalid(
                "parameter_layout_digest_invalid".into(),
            ));
        }
        let path = self
            .root
            .join("state/parameter_layouts/by-sha")
            .join(format!("{}.json", digest.to_ascii_lowercase()));
        let reference =
            PrivateFileReference::new(path, Sha256Digest::parse(digest.to_ascii_lowercase())?);
        let bytes = reference
            .read_verified_bounded(&self.root, MAX_ENGINE_JSON_BYTES)
            .map_err(|_| {
                BrainError::Integrity("parameter_layout_artifact_missing_or_invalid".into())
            })?;
        let layout: ParameterBlockLayout = serde_json::from_slice(&bytes)?;
        layout
            .validate()
            .map_err(|_| BrainError::Integrity("parameter_layout_contract_invalid".into()))?;
        Ok(layout)
    }

    /// Resolve only an engine-owned, content-addressed dense delta. Runtime
    /// SkillFields must never point to an arbitrary readable file, even if its
    /// bytes happen to form a valid dvec artifact.
    pub(super) fn verified_private_dvec(
        &self,
        reference: &DeltaArtifactRef,
    ) -> BrainResult<PathBuf> {
        if !valid_digest(&reference.sha256) || reference.parameter_count == 0 {
            return Err(BrainError::Integrity(
                "runtime_dense_reference_contract_invalid".into(),
            ));
        }
        let supplied = Path::new(&reference.path);
        let expected = self
            .root
            .join("artifacts/deltas/by-sha")
            .join(format!("{}.dvec", reference.sha256.to_ascii_lowercase()));
        if supplied != expected {
            return Err(BrainError::Integrity(
                "runtime_dense_reference_path_invalid".into(),
            ));
        }
        let expected = existing_regular_file_under_root(&self.root, &expected)?;
        let inspected = inspect_dvec(&self.root, &expected)?;
        if inspected.sha256 != reference.sha256.to_ascii_lowercase()
            || inspected.parameter_count != reference.parameter_count
        {
            return Err(BrainError::Integrity(
                "runtime_dense_reference_identity_mismatch".into(),
            ));
        }
        Ok(expected)
    }

    pub(super) fn bank_path(&self) -> PathBuf {
        self.root.join("state/skill_bank.json")
    }
    pub(super) fn validate_bank_scalar(label: &str, value: f64) -> BrainResult<()> {
        if !value.is_finite() {
            return Err(BrainError::Integrity(format!(
                "skill_bank_{label}_nonfinite"
            )));
        }
        if value.abs() > f64::MAX.sqrt() {
            return Err(BrainError::Integrity(format!(
                "skill_bank_{label}_magnitude_too_large"
            )));
        }
        Ok(())
    }

    pub(super) fn validate_skill_bank_semantics(bank: &SkillBank) -> BrainResult<()> {
        if bank.fields.len() > MAX_ENGINE_SKILL_FIELDS {
            return Err(BrainError::Integrity(
                "skill_bank_field_limit_exceeded".into(),
            ));
        }
        let expected_direction_dimension = bank.fields.first().map(|field| field.direction.len());
        let mut ids = BTreeSet::new();
        for field in &bank.fields {
            if !ids.insert(field.skill_id.clone())
                || field.direction.is_empty()
                || field.direction.len() > MAX_ENGINE_PARAMETER_DIMENSION
                || expected_direction_dimension != Some(field.direction.len())
                || field.reconstruction_id.is_unassigned()
                || field.lineage_id.is_unassigned()
                || field.generation_created > bank.generation
                || field.functional_signature.len() > MAX_ENGINE_FUNCTIONAL_RESPONSE_DIMENSION
                || field.representation_signature.len() > MAX_ENGINE_FUNCTIONAL_RESPONSE_DIMENSION
            {
                return Err(BrainError::Integrity(format!(
                    "skill_bank_field_contract_invalid:{}",
                    field.skill_id
                )));
            }
            if field.support == 0 || field.support != field.evidence_support_digests.len() {
                return Err(BrainError::Integrity(format!(
                    "skill_bank_field_support_mismatch:{}",
                    field.skill_id
                )));
            }
            if field
                .evidence_support_digests
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(BrainError::Integrity(format!(
                    "skill_bank_field_support_not_sorted:{}",
                    field.skill_id
                )));
            }
            let parent_ids = field.parent_skill_ids.iter().collect::<BTreeSet<_>>();
            if parent_ids.len() != field.parent_skill_ids.len()
                || parent_ids.contains(&field.skill_id)
            {
                return Err(BrainError::Integrity(format!(
                    "skill_bank_field_parent_contract_invalid:{}",
                    field.skill_id
                )));
            }
            for (index, value) in field.direction.iter().enumerate() {
                Self::validate_bank_scalar(
                    &format!("field_{}_direction_{}", field.skill_id, index),
                    *value,
                )?;
            }
            for (index, value) in field.functional_signature.iter().enumerate() {
                Self::validate_bank_scalar(
                    &format!("field_{}_functional_{}", field.skill_id, index),
                    *value,
                )?;
            }
            for (index, value) in field.representation_signature.iter().enumerate() {
                Self::validate_bank_scalar(
                    &format!("field_{}_representation_{}", field.skill_id, index),
                    *value,
                )?;
            }
            for value in [
                field.singular_value,
                field.explained_variance,
                field.persistence,
                field.coherence,
                field.uncertainty,
            ] {
                Self::validate_bank_scalar(&format!("field_{}_scalars", field.skill_id), value)?;
            }
            if field.singular_value < 0.0
                || !(0.0..=1.0).contains(&field.explained_variance)
                || !(0.0..=1.0).contains(&field.persistence)
                || !(0.0..=1.0).contains(&field.coherence)
                || field.uncertainty < 0.0
            {
                return Err(BrainError::Integrity(format!(
                    "skill_bank_field_scalar_range_invalid:{}",
                    field.skill_id
                )));
            }
            if let Some(geometry) = &field.structured_geometry {
                if geometry.skill_id != field.skill_id
                    || geometry.source_support_indices.is_empty()
                    || geometry
                        .source_support_indices
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                    || geometry.blocks.is_empty()
                    || geometry.max_local_rank == 0
                    || !geometry.mean_effective_rank.is_finite()
                    || geometry.mean_effective_rank <= 0.0
                {
                    return Err(BrainError::Integrity(format!(
                        "skill_bank_structured_geometry_invalid:{}",
                        field.skill_id
                    )));
                }
                for block in &geometry.blocks {
                    if block.block_name.trim().is_empty()
                        || block.block_name.len() > MAX_ENGINE_TEXT_BYTES
                        || block.count == 0
                        || block.shape.is_empty()
                        || block.shape.contains(&0)
                        || block.selected_rank != block.axes.len()
                        || block.selected_rank > block.count
                        || !block.effective_rank.is_finite()
                        || block.effective_rank < 0.0
                        || !block.retained_energy.is_finite()
                        || !(0.0..=1.0).contains(&block.retained_energy)
                        || !block.block_energy.is_finite()
                        || block.block_energy < 0.0
                        || !block.normalized_block_energy.is_finite()
                        || block.normalized_block_energy < 0.0
                        || !block.reconstruction_rms.is_finite()
                        || block.reconstruction_rms < 0.0
                    {
                        return Err(BrainError::Integrity(format!(
                            "skill_bank_structured_block_invalid:{}",
                            field.skill_id
                        )));
                    }
                    let shape_count = block.shape.iter().try_fold(1usize, |acc, value| {
                        acc.checked_mul(*value).ok_or_else(|| {
                            BrainError::Integrity(
                                "skill_bank_structured_block_shape_overflow".into(),
                            )
                        })
                    })?;
                    if shape_count != block.count
                        || block.axes.iter().any(|axis| {
                            !axis.singular_value.is_finite()
                                || axis.singular_value <= 0.0
                                || axis.source_coefficients.is_empty()
                                || axis
                                    .source_coefficients
                                    .iter()
                                    .any(|value| !value.is_finite())
                        })
                    {
                        return Err(BrainError::Integrity(format!(
                            "skill_bank_structured_axis_invalid:{}",
                            field.skill_id
                        )));
                    }
                }
            }
            if let Some(reference) = &field.dense_materialization {
                if !valid_digest(&reference.sha256) || reference.parameter_count == 0 {
                    return Err(BrainError::Integrity(format!(
                        "skill_bank_dense_reference_invalid:{}",
                        field.skill_id
                    )));
                }
            }
            if field.structured_geometry.is_some() != field.parameter_layout_sha256.is_some()
                || field.dense_materialization.is_some() != field.parameter_layout_sha256.is_some()
            {
                return Err(BrainError::Integrity(format!(
                    "skill_bank_runtime_materialization_binding_invalid:{}",
                    field.skill_id
                )));
            }
        }
        Ok(())
    }

    /// Load the installed active bank.  Absence is never silently translated
    /// into an empty bank: callers that intentionally establish the very
    /// first bank must use the explicit bootstrap path below.
    pub fn load_bank(&self) -> BrainResult<SkillBank> {
        let p = self.bank_path();
        let raw = match read_untrusted_private_file_bounded(&self.root, &p, MAX_SKILL_BANK_BYTES) {
            Ok(raw) => raw,
            Err(BrainError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(BrainError::Integrity("active_skill_bank_missing".into()));
            }
            Err(error) => {
                return Err(BrainError::Integrity(format!(
                    "active_skill_bank_invalid:{error}"
                )));
            }
        };
        let bank: SkillBank = serde_json::from_slice(&raw)?;
        Self::validate_skill_bank_semantics(&bank)?;
        Ok(bank)
    }

    /// A missing active bank is meaningful only while establishing the first
    /// sleep transaction.  This preserves that explicit bootstrap state
    /// without exposing a generic empty-bank fallback to runtime authority.
    pub(super) fn load_bank_for_initial_sleep_bootstrap(&self) -> BrainResult<Option<SkillBank>> {
        let p = self.bank_path();
        match fs::symlink_metadata(&p) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
            Ok(_) => self.load_bank().map(Some),
        }
    }

    pub(super) fn persist_observations(
        &self,
        obs: &[DeltaObservation],
    ) -> BrainResult<Vec<String>> {
        let mut prepared = Vec::new();
        let mut digests = Vec::new();
        for observation in obs {
            let digest = digest_json(observation)?;
            let bytes = serialize_pretty_line(observation)?;
            let name = format!("{}-{}.json", observation.observation_id, &digest[..16]);
            prepared.push((name, digest.clone(), bytes));
            digests.push(digest);
        }
        digests.sort();
        digests.dedup();
        if digests.len() != obs.len() {
            return Err(BrainError::Integrity(
                "persisted_observation_digest_duplicate".into(),
            ));
        }
        let batch_digest = digest_json(&digests)?;
        let staging = self
            .root
            .join("state/observation_staging")
            .join(&batch_digest);
        ensure_private_directory(&self.root, &staging)?;
        for (name, _, bytes) in &prepared {
            write_new_private(&self.root, &staging.join(name), bytes)?;
        }
        let mut staged_names = BTreeSet::new();
        for path in list_existing_private_directory(&self.root, &staging)? {
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| {
                    BrainError::Integrity("observation_staging_filename_invalid".into())
                })?
                .to_string();
            staged_names.insert(name);
        }
        let expected_names = prepared
            .iter()
            .map(|(name, _, _)| name.clone())
            .collect::<BTreeSet<_>>();
        if staged_names != expected_names {
            return Err(BrainError::Integrity(
                "observation_staging_directory_invalid".into(),
            ));
        }
        let live = self.root.join("state/observations");
        match fs::symlink_metadata(&live) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let identity = inspect_private_directory(&self.root, &staging)?;
                move_private_directory_transactional(&self.root, &staging, &live, &identity)?;
            }
            Ok(_) => {
                let live = existing_directory_under_root(&self.root, &live)?;
                for (name, _, bytes) in &prepared {
                    write_new_private(&self.root, &live.join(name), bytes)?;
                }
            }
            Err(error) => return Err(error.into()),
        }
        Ok(digests)
    }

    pub(super) fn load_persisted_observations(&self) -> BrainResult<Vec<DeltaObservation>> {
        load_observations_from_private_directory(
            &self.root,
            &self.root.join("state/observations"),
            true,
        )
    }
}
