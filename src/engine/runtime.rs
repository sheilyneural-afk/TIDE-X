use super::*;

impl BrainEngine {
    pub(super) fn verify_runtime_composition_bank(&self, bank: &SkillBank) -> BrainResult<()> {
        if bank.fields.is_empty() {
            return Err(BrainError::Integrity(
                "runtime_composition_bank_empty".into(),
            ));
        }
        let layout_ids = bank
            .fields
            .iter()
            .map(|field| {
                field.parameter_layout_sha256.clone().ok_or_else(|| {
                    BrainError::Integrity(format!(
                        "runtime_composition_layout_missing:{}",
                        field.skill_id
                    ))
                })
            })
            .collect::<BrainResult<BTreeSet<_>>>()?;
        if layout_ids.len() != 1 {
            return Err(BrainError::Integrity(
                "runtime_composition_layout_identity_invalid".into(),
            ));
        }
        let layout_id = layout_ids
            .into_iter()
            .next()
            .ok_or_else(|| BrainError::Integrity("runtime_composition_layout_missing".into()))?;
        let layout = self.load_parameter_layout(layout_id.as_str())?;
        for field in &bank.fields {
            if field.structured_geometry.is_none() {
                return Err(BrainError::Integrity(format!(
                    "runtime_composition_geometry_missing:{}",
                    field.skill_id
                )));
            }
            let reference = field.dense_materialization.as_ref().ok_or_else(|| {
                BrainError::Integrity(format!(
                    "runtime_composition_dense_missing:{}",
                    field.skill_id
                ))
            })?;
            if reference.parameter_count != layout.total_parameter_count {
                return Err(BrainError::Integrity(format!(
                    "runtime_composition_dense_count_mismatch:{}",
                    field.skill_id
                )));
            }
            self.verified_private_dvec(reference).map_err(|error| {
                BrainError::Integrity(format!(
                    "runtime_composition_dense_invalid:{}:{error}",
                    field.skill_id
                ))
            })?;
        }
        Ok(())
    }

    pub(super) fn runtime_integrity_health(&self) -> BrainResult<RuntimeIntegrityHealth> {
        let mut integrity_reasons = Vec::<String>::new();
        let canonical_runtime_config = self.config == BrainConfig::default();
        if !canonical_runtime_config {
            integrity_reasons.push("noncanonical_runtime_config_forbidden".into());
        }
        let corpus_transition_clear = match self.require_no_incomplete_corpus_transition() {
            Ok(()) => true,
            Err(error) => {
                integrity_reasons.push(format!("corpus_transition_incomplete:{error}"));
                false
            }
        };
        let ledger_status = match ledger::verify(&self.root) {
            Ok(status) => Some(status),
            Err(error) => {
                integrity_reasons.push(format!("ledger_invalid:{error}"));
                None
            }
        };
        let ledger_verified = ledger_status.is_some();

        let bank = match self.load_bank() {
            Ok(bank) => Some(bank),
            Err(error) => {
                integrity_reasons.push(format!("active_bank_unavailable:{error}"));
                None
            }
        };
        let mut bank_verified = bank.as_ref().is_some_and(|bank| !bank.fields.is_empty());
        if !bank_verified {
            integrity_reasons.push("active_bank_empty_or_missing".into());
        } else if let Some(bank) = bank.as_ref() {
            let dimension = bank.fields[0].direction.len();
            let mut ids = BTreeSet::new();
            for field in &bank.fields {
                if field.skill_id.trim().is_empty()
                    || field.reconstruction_id.is_unassigned()
                    || field.lineage_id.is_unassigned()
                    || !ids.insert(field.skill_id.as_str())
                    || dimension == 0
                    || field.direction.len() != dimension
                    || field.direction.iter().any(|value| !value.is_finite())
                    || !field.persistence.is_finite()
                    || !field.coherence.is_finite()
                    || !field.uncertainty.is_finite()
                {
                    bank_verified = false;
                    break;
                }
            }
            if !bank_verified {
                integrity_reasons.push("active_bank_contract_invalid".into());
            }
        }

        let composition_ready = if let Some(bank) = bank.as_ref().filter(|_| bank_verified) {
            match self.verify_runtime_composition_bank(bank) {
                Ok(()) => true,
                Err(error) => {
                    integrity_reasons.push(format!("runtime_composition_unready:{error}"));
                    false
                }
            }
        } else {
            false
        };

        let observations = self.load_persisted_observations()?;
        let mut current_corpus_digest = None;
        let observations_verified = match self.validate_observations(&observations).and_then(|_| {
            estimate_aperture_independence(&observations, self.config.min_independent_apertures)
                .map(|_| ())
        }) {
            Ok(()) => match observation_set_digest(&canonical_observations(&observations)) {
                Ok(digest) => {
                    current_corpus_digest = Some(digest);
                    true
                }
                Err(error) => {
                    integrity_reasons.push(format!("current_corpus_digest_invalid:{error}"));
                    false
                }
            },
            Err(error) => {
                integrity_reasons.push(format!("observations_invalid:{error}"));
                false
            }
        };

        let sleep_path = self.root.join("state/sleep_state.json");
        let state = match fs::symlink_metadata(&sleep_path) {
            Ok(_) => match existing_regular_file_under_root(&self.root, &sleep_path) {
                Ok(verified_path) => {
                    match serde_json::from_slice::<Value>(&fs::read(&verified_path)?) {
                        Ok(value)
                            if value.get("schema").and_then(Value::as_str)
                                == Some("cerebro.tidex.sleep_state/v5") =>
                        {
                            Some(value)
                        }
                        Ok(_) => {
                            integrity_reasons.push("sleep_state_schema_invalid".into());
                            None
                        }
                        Err(error) => {
                            integrity_reasons.push(format!("sleep_state_parse_failed:{error}"));
                            None
                        }
                    }
                }
                Err(error) => {
                    integrity_reasons.push(format!("sleep_state_path_invalid:{error}"));
                    None
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                integrity_reasons.push("sleep_state_missing".into());
                None
            }
            Err(error) => {
                integrity_reasons.push(format!("sleep_state_metadata_failed:{error}"));
                None
            }
        };
        let sleep_state_verified = state.is_some();
        let operation_key = state
            .as_ref()
            .and_then(|value| value.get("operation_key"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let raw_certification_status = state
            .as_ref()
            .and_then(|value| value.get("certification_status"))
            .and_then(serde_json::Value::as_str);
        let certification_status = raw_certification_status.and_then(CertificationStatus::parse);
        if raw_certification_status.is_some_and(|value| CertificationStatus::parse(value).is_none())
        {
            integrity_reasons.push("certification_status_invalid".into());
        }
        let certified = certification_status == Some(CertificationStatus::Certified);
        let evidence_verified = state
            .as_ref()
            .and_then(|value| value.get("evidence_verification"))
            .and_then(|value| value.get("verified"))
            .and_then(serde_json::Value::as_bool)
            == Some(true);

        let mut receipt_verified = false;
        let mut historical_artifacts_verified = false;
        let mut current_pointers_verified = false;
        let mut report_state_consistent = false;
        let mut current_corpus_bound = false;
        let mut analysis_current = false;
        if let (Some(state), Some(operation_key)) = (state.as_ref(), operation_key.as_deref()) {
            let receipt_path = self
                .root
                .join("state/sleep_receipts")
                .join(format!("{operation_key}.json"));
            let receipt_present = match fs::symlink_metadata(&receipt_path) {
                Ok(_) => true,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
                Err(error) => {
                    integrity_reasons.push(format!("sleep_receipt_metadata_failed:{error}"));
                    false
                }
            };
            if receipt_present {
                match existing_regular_file_under_root(&self.root, &receipt_path)
                    .and_then(|path| Ok(serde_json::from_slice::<SleepReceipt>(&fs::read(path)?)?))
                {
                    Ok(receipt) => {
                        match existing_regular_file_under_root(&self.root, &sleep_path).and_then(
                            |verified_state| {
                                verify_sleep_receipt_ledger_binding(
                                    &self.root,
                                    &receipt,
                                    state,
                                    &file_sha256(&verified_state)?,
                                )
                            },
                        ) {
                            Ok(()) => receipt_verified = true,
                            Err(error) => {
                                integrity_reasons
                                    .push(format!("sleep_receipt_chain_mismatch:{error}"));
                            }
                        }

                        let report_history = self
                            .root
                            .join("state/reports")
                            .join(format!("{}.json", receipt.report_sha256));
                        let memory_history =
                            memory_artifact_path(&self.root, &receipt.memory_sha256);
                        let state_history = self
                            .root
                            .join("state/sleep/by-sha")
                            .join(format!("{}.json", receipt.sleep_state_sha256));
                        let bank_history = receipt.active_bank_sha256.as_ref().map(|sha| {
                            self.root
                                .join("state/skill_banks/by-sha")
                                .join(format!("{sha}.json"))
                        });
                        let evidence_history = receipt.evidence_bundle_sha256.as_ref().map(|sha| {
                            self.root
                                .join("state/sleep_evidence/by-sha")
                                .join(format!("{sha}.json"))
                        });
                        historical_artifacts_verified =
                            private_file_digest_matches(
                                &self.root,
                                &report_history,
                                &receipt.report_sha256,
                            ) && private_file_digest_matches(
                                &self.root,
                                &memory_history,
                                &receipt.memory_sha256,
                            ) && private_file_digest_matches(
                                &self.root,
                                &state_history,
                                &receipt.sleep_state_sha256,
                            ) && bank_history.as_ref().is_none_or(|path| {
                                receipt.active_bank_sha256.as_ref().is_some_and(|sha| {
                                    private_file_digest_matches(&self.root, path, sha)
                                })
                            }) && evidence_history.as_ref().is_none_or(|path| {
                                receipt.evidence_bundle_sha256.as_ref().is_some_and(|sha| {
                                    private_file_digest_matches(&self.root, path, sha)
                                })
                            });
                        if !historical_artifacts_verified {
                            integrity_reasons.push("sleep_historical_artifact_mismatch".into());
                        }

                        let current_memory = self.root.join("state/memory/current.json");
                        let current_evidence = self.root.join("state/sleep_evidence/current.json");
                        current_pointers_verified =
                            receipt.active_bank_sha256.as_ref().is_some_and(|sha| {
                                private_file_digest_matches(&self.root, &self.bank_path(), sha)
                            }) && private_file_digest_matches(
                                &self.root,
                                &current_memory,
                                &receipt.memory_sha256,
                            ) && receipt.evidence_bundle_sha256.as_ref().is_some_and(|sha| {
                                private_file_digest_matches(&self.root, &current_evidence, sha)
                            });
                        if !current_pointers_verified {
                            integrity_reasons.push("current_pointer_digest_mismatch".into());
                        }

                        if historical_artifacts_verified {
                            let report_history =
                                existing_regular_file_under_root(&self.root, &report_history)?;
                            match serde_json::from_slice::<Value>(&fs::read(&report_history)?) {
                                Ok(report) => {
                                    let report_source = report
                                        .get("source_tree_digest")
                                        .and_then(serde_json::Value::as_str);
                                    let report_config = report
                                        .get("config_digest")
                                        .and_then(serde_json::Value::as_str);
                                    let report_analysis = report
                                        .get("analysis_version_digest")
                                        .and_then(serde_json::Value::as_str);
                                    let state_source = state
                                        .get("source_tree_digest")
                                        .and_then(serde_json::Value::as_str);
                                    let state_config = state
                                        .get("config_digest")
                                        .and_then(serde_json::Value::as_str);
                                    let state_analysis = state
                                        .get("analysis_version_digest")
                                        .and_then(serde_json::Value::as_str);
                                    report_state_consistent = report_source.is_some()
                                        && report_config.is_some()
                                        && report_analysis.is_some()
                                        && state_source == report_source
                                        && state_config == report_config
                                        && state_analysis == report_analysis;
                                    if !report_state_consistent {
                                        integrity_reasons
                                            .push("report_sleep_state_identity_mismatch".into());
                                    }
                                    let state_corpus = state
                                        .get("corpus_digest")
                                        .and_then(serde_json::Value::as_str);
                                    let report_corpus = report
                                        .get("observation_set_digest")
                                        .and_then(serde_json::Value::as_str);
                                    let recomputed_analysis_key =
                                        serde_json::from_value::<ReconstructionReport>(
                                            report.clone(),
                                        )
                                        .ok()
                                        .map(|typed| {
                                            sleep_analysis_key(
                                                typed.observation_set_digest.as_str(),
                                                &typed,
                                            )
                                        });
                                    current_corpus_bound = current_corpus_digest
                                        .as_deref()
                                        .is_some_and(|digest| state_corpus == Some(digest))
                                        && state_corpus == report_corpus
                                        && recomputed_analysis_key.as_deref().is_some_and(|key| {
                                            state.get("analysis_key").and_then(Value::as_str)
                                                == Some(key)
                                        });
                                    if !current_corpus_bound {
                                        integrity_reasons
                                            .push("current_corpus_sleep_binding_mismatch".into());
                                    }
                                    let (source, config, analysis) =
                                        analysis_identity(&self.config)?;
                                    analysis_current = report_source == Some(source.as_str())
                                        && report_config == Some(config.as_str())
                                        && report_analysis == Some(analysis.as_str())
                                        && current_corpus_bound;
                                }
                                Err(error) => {
                                    integrity_reasons
                                        .push(format!("historical_report_json_invalid:{error}"));
                                }
                            }
                        }
                    }
                    Err(error) => {
                        integrity_reasons.push(format!("sleep_receipt_parse_failed:{error}"));
                    }
                }
            } else {
                integrity_reasons.push("sleep_receipt_missing".into());
            }
        }

        let integrity_healthy = canonical_runtime_config
            && corpus_transition_clear
            && ledger_verified
            && bank_verified
            && observations_verified
            && sleep_state_verified
            && receipt_verified
            && historical_artifacts_verified
            && current_pointers_verified
            && report_state_consistent
            && current_corpus_bound;
        let mut execution_blockers = Vec::<String>::new();
        if !canonical_runtime_config {
            execution_blockers.push("noncanonical_runtime_config_forbidden".into());
        }
        if !corpus_transition_clear {
            execution_blockers.push("corpus_transition_incomplete".into());
        }
        if !integrity_healthy {
            execution_blockers.push("integrity_unhealthy".into());
        }
        if !analysis_current {
            execution_blockers.push("analysis_version_stale".into());
        }
        if !certified {
            execution_blockers.push(format!(
                "certification_status:{}",
                certification_status
                    .map(CertificationStatus::as_str)
                    .unwrap_or("missing")
            ));
        }
        if !evidence_verified {
            execution_blockers.push("sleep_evidence_unverified".into());
        }
        if !composition_ready {
            execution_blockers.push("runtime_composition_unready".into());
        }
        let execution_authorized = execution_blockers.is_empty();
        Ok(RuntimeIntegrityHealth {
            schema: "cerebro.tidex.runtime_integrity_health/v1".into(),
            canonical_runtime_config,
            corpus_transition_clear,
            ledger_verified,
            bank_verified,
            composition_ready,
            observations_verified,
            sleep_state_verified,
            receipt_verified,
            historical_artifacts_verified,
            current_pointers_verified,
            report_state_consistent,
            current_corpus_bound,
            analysis_current,
            certified,
            evidence_verified,
            integrity_healthy,
            execution_authorized,
            operation_key,
            certification_status,
            integrity_reasons,
            execution_blockers,
        })
    }

    pub(super) fn require_current_certification(&self) -> BrainResult<()> {
        self.require_canonical_runtime_config()?;
        let health = self.runtime_integrity_health()?;
        if !health.execution_authorized {
            return Err(BrainError::Integrity(format!(
                "runtime_execution_not_authorized:{}",
                health.execution_blockers.join("|")
            )));
        }
        Ok(())
    }

    pub(super) fn canonical_evidence_path(
        &self,
        raw: &str,
        expected_sha256: &str,
    ) -> BrainResult<PathBuf> {
        if expected_sha256.len() != 64
            || !expected_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(BrainError::Invalid(
                "runtime_evidence_digest_invalid".into(),
            ));
        }
        let path = PathBuf::from(raw);
        let path = existing_regular_file_under_root(&self.root, &path)
            .map_err(|_| BrainError::Integrity("runtime_evidence_path_invalid".into()))?;
        if file_sha256(&path)? != expected_sha256 {
            return Err(BrainError::Integrity(
                "runtime_evidence_identity_mismatch".into(),
            ));
        }
        Ok(path)
    }

    pub(super) fn load_verified_runtime_evidence(
        &self,
        bank: &SkillBank,
    ) -> BrainResult<(ProtectedCortex, Matrix, f64, CausalCreditReport)> {
        let observations = canonical_observations(&self.load_persisted_observations()?);
        let mut reconstruction = self.analyze_canonical(&observations)?;
        if reconstruction.promotion.allowed {
            let mixtures = reconstruction.skill_source_mixtures.clone();
            self.materialize_dense_fields(&mut reconstruction.fields, &mixtures, &observations)?;
        }
        let reconstruction_bytes = serialize_pretty_line(&reconstruction)?;
        let reconstruction_sha256 =
            ReportDigest::from(Sha256Digest::digest_bytes(&reconstruction_bytes));
        let bundle = load_sleep_evidence(&self.root)?;
        let verification = verify_sleep_evidence(
            &self.root,
            &bundle,
            &SleepEvidenceExpectation {
                corpus_digest: &reconstruction.observation_set_digest,
                report_sha256: &reconstruction_sha256,
                source_tree_digest: &reconstruction.source_tree_digest,
                config_digest: &reconstruction.config_digest,
                analysis_version_digest: &reconstruction.analysis_version_digest,
                fields: &reconstruction.fields,
                observations: &observations,
                source_mixtures: &reconstruction.skill_source_mixtures,
            },
        )?;
        if !verification.verified {
            return Err(BrainError::Integrity(format!(
                "runtime_evidence_reverification_failed:{}",
                verification.reasons.join("|")
            )));
        }
        let bank_ids = bank
            .fields
            .iter()
            .map(|field| field.skill_id.clone())
            .collect::<Vec<_>>();
        if bank_ids != bundle.field_ids {
            return Err(BrainError::Integrity(
                "runtime_evidence_bank_identity_mismatch".into(),
            ));
        }

        let protected_path = self.canonical_evidence_path(
            &bundle.protection.protected_map_path,
            &bundle.protection.protected_map_sha256,
        )?;
        let protected_wrapper: serde_json::Value =
            serde_json::from_slice(&fs::read(protected_path)?)?;
        if protected_wrapper
            .get("schema")
            .and_then(serde_json::Value::as_str)
            != Some("cerebro.tidex.protected_map_benchmark/v2")
            || protected_wrapper
                .get("task_labels_used")
                .and_then(serde_json::Value::as_bool)
                != Some(false)
        {
            return Err(BrainError::Integrity(
                "runtime_protected_map_contract_invalid".into(),
            ));
        }
        let protected_map: ProtectedMapArtifactReport = serde_json::from_value(
            protected_wrapper
                .get("map")
                .cloned()
                .ok_or_else(|| BrainError::Integrity("runtime_protected_map_missing".into()))?,
        )?;
        let protected = load_protected_cortex(&self.root, &protected_map)?;

        let interaction_path = self.canonical_evidence_path(
            &bundle.interaction.source_path,
            &bundle.interaction.source_sha256,
        )?;
        let interaction_payload: serde_json::Value =
            serde_json::from_slice(&fs::read(interaction_path)?)?;
        if interaction_payload
            .get("schema")
            .and_then(serde_json::Value::as_str)
            != Some("cerebro.tidex.trust_region_benchmark/v3")
        {
            return Err(BrainError::Integrity(
                "runtime_interaction_contract_invalid".into(),
            ));
        }
        let interaction_ids = interaction_payload
            .get("field_ids")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| BrainError::Integrity("runtime_interaction_ids_missing".into()))?
            .iter()
            .map(|value| {
                let value = value.as_str().ok_or_else(|| {
                    BrainError::Integrity("runtime_interaction_id_invalid".into())
                })?;
                SkillId::parse(value)
                    .map_err(|_| BrainError::Integrity("runtime_interaction_id_invalid".into()))
            })
            .collect::<BrainResult<Vec<_>>>()?;
        if interaction_ids != bank_ids {
            return Err(BrainError::Integrity(
                "runtime_interaction_bank_identity_mismatch".into(),
            ));
        }
        let interaction_rows = interaction_payload
            .get("interaction_matrix")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| BrainError::Integrity("runtime_interaction_matrix_missing".into()))?
            .iter()
            .map(|row| {
                row.as_array()
                    .ok_or_else(|| BrainError::Integrity("runtime_interaction_row_invalid".into()))?
                    .iter()
                    .map(|value| {
                        value.as_f64().ok_or_else(|| {
                            BrainError::Integrity("runtime_interaction_value_invalid".into())
                        })
                    })
                    .collect::<BrainResult<Vec<_>>>()
            })
            .collect::<BrainResult<Vec<_>>>()?;
        let interaction = Matrix::from_rows(&interaction_rows)?;
        if interaction.rows != bank.fields.len() || interaction.cols != bank.fields.len() {
            return Err(BrainError::Integrity(
                "runtime_interaction_shape_mismatch".into(),
            ));
        }
        let max_quadratic_cost = interaction_payload
            .get("diagonal_budget")
            .and_then(serde_json::Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .ok_or_else(|| BrainError::Integrity("runtime_trust_budget_invalid".into()))?;
        let trust_causal_credit_sha256 = interaction_payload
            .get("causal_credit_sha256")
            .and_then(serde_json::Value::as_str)
            .filter(|value| valid_digest(value))
            .ok_or_else(|| BrainError::Integrity("runtime_trust_causal_digest_missing".into()))?;
        if trust_causal_credit_sha256 != bundle.causal_credit.credit_source_sha256.as_str() {
            return Err(BrainError::Integrity(
                "runtime_trust_causal_digest_mismatch".into(),
            ));
        }
        let trust_payload = interaction_payload
            .get("trust_region")
            .ok_or_else(|| BrainError::Integrity("runtime_trust_payload_missing".into()))?;
        if trust_payload
            .get("allocation_policy")
            .and_then(serde_json::Value::as_str)
            != Some("causal_priority_contraction/v1")
        {
            return Err(BrainError::Integrity(
                "runtime_trust_policy_not_causal".into(),
            ));
        }
        let stored_causal_priority_weights = trust_payload
            .get("causal_priority_weights")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| BrainError::Integrity("runtime_trust_causal_weights_missing".into()))?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .ok_or_else(|| {
                        BrainError::Integrity("runtime_trust_causal_weight_invalid".into())
                    })
            })
            .collect::<BrainResult<Vec<_>>>()?;
        let component_retention = trust_payload
            .get("component_retention")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                BrainError::Integrity("runtime_trust_component_retention_missing".into())
            })?;
        if stored_causal_priority_weights.len() != bank.fields.len()
            || component_retention.len() != bank.fields.len()
            || component_retention.iter().any(|value| {
                value
                    .as_f64()
                    .is_none_or(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
            })
        {
            return Err(BrainError::Integrity(
                "runtime_trust_causal_contract_invalid".into(),
            ));
        }

        let causal_path = self.canonical_evidence_path(
            &bundle.causal_credit.credit_source_path,
            &bundle.causal_credit.credit_source_sha256,
        )?;
        let causal_wrapper: serde_json::Value = serde_json::from_slice(&fs::read(causal_path)?)?;
        if causal_wrapper
            .get("schema")
            .and_then(serde_json::Value::as_str)
            != Some("cerebro.tidex.causal_credit_benchmark/v3")
            || causal_wrapper
                .get("blind_data_accessed")
                .and_then(serde_json::Value::as_bool)
                != Some(false)
        {
            return Err(BrainError::Integrity(
                "runtime_causal_credit_contract_invalid".into(),
            ));
        }
        let causal: CausalCreditReport = serde_json::from_value(
            causal_wrapper
                .get("causal_credit")
                .cloned()
                .ok_or_else(|| BrainError::Integrity("runtime_causal_credit_missing".into()))?,
        )?;
        let causal_ids = causal
            .fields
            .iter()
            .map(|field| field.skill_id.clone())
            .collect::<BTreeSet<_>>();
        let bank_id_set = bank_ids.iter().cloned().collect::<BTreeSet<_>>();
        if causal_ids != bank_id_set || !causal.unresolved_fields.is_empty() {
            return Err(BrainError::Integrity(
                "runtime_causal_credit_identity_unresolved".into(),
            ));
        }
        let expected_causal_priority_weights =
            certified_causal_priority_weights(&causal, &bank_ids)?;
        let causal_tolerance = f64::EPSILON.sqrt() * bank_ids.len().max(1) as f64 * 32.0;
        if stored_causal_priority_weights
            .iter()
            .zip(&expected_causal_priority_weights)
            .any(|(stored, expected)| {
                (stored - expected).abs() > causal_tolerance * (1.0 + expected.abs())
            })
        {
            return Err(BrainError::Integrity(
                "runtime_trust_causal_weight_mismatch".into(),
            ));
        }
        Ok((protected, interaction, max_quadratic_cost, causal))
    }

    pub(super) fn compose_verified_inputs(
        &self,
        bank: &SkillBank,
        activation: &BTreeMap<SkillId, f64>,
        protected: &ProtectedCortex,
        interaction_metric: &Matrix,
        max_quadratic_cost: f64,
        causal_credit: &CausalCreditReport,
    ) -> BrainResult<GovernedComposition> {
        if bank.fields.is_empty() || activation.is_empty() {
            return Err(BrainError::Invalid("skill_activation_empty".into()));
        }
        let mut proposed = vec![0.0; bank.fields.len()];
        for (id, coefficient) in activation {
            if !coefficient.is_finite() {
                return Err(BrainError::Invalid("activation_non_finite".into()));
            }
            let field_index = bank
                .fields
                .iter()
                .position(|field| &field.skill_id == id)
                .ok_or_else(|| BrainError::Invalid(format!("skill_not_found:{id}")))?;
            proposed[field_index] = *coefficient;
        }
        let field_ids = bank
            .fields
            .iter()
            .map(|field| field.skill_id.clone())
            .collect::<Vec<_>>();
        let causal_priority_weights = certified_causal_priority_weights(causal_credit, &field_ids)?;
        let trust = apply_causal_priority_trust_region(
            interaction_metric,
            &proposed,
            max_quadratic_cost,
            &causal_priority_weights,
        )?;

        let layout_ids = bank
            .fields
            .iter()
            .map(|field| {
                field.parameter_layout_sha256.clone().ok_or_else(|| {
                    BrainError::Integrity(format!(
                        "runtime_field_layout_missing:{}",
                        field.skill_id
                    ))
                })
            })
            .collect::<BrainResult<BTreeSet<_>>>()?;
        if layout_ids.len() != 1 {
            return Err(BrainError::Integrity(
                "runtime_skill_bank_multiple_parameter_layouts".into(),
            ));
        }
        let layout_sha = layout_ids.into_iter().next().ok_or_else(|| {
            BrainError::Integrity("runtime_skill_bank_parameter_layout_missing".into())
        })?;
        let layout = self.load_parameter_layout(layout_sha.as_str())?;
        if protected.parameter_importance.len() != layout.total_parameter_count as usize {
            return Err(BrainError::Invalid(
                "protected_cortex_parameter_space_mismatch".into(),
            ));
        }
        let mut delta = vec![0.0f64; layout.total_parameter_count as usize];
        for (field, coefficient) in bank.fields.iter().zip(&trust.accepted_coefficients) {
            if coefficient.abs() <= f64::EPSILON {
                continue;
            }
            if field.structured_geometry.is_none() {
                return Err(BrainError::Integrity(format!(
                    "runtime_field_structured_geometry_missing:{}",
                    field.skill_id
                )));
            }
            let reference = field.dense_materialization.as_ref().ok_or_else(|| {
                BrainError::Integrity(format!(
                    "runtime_dense_materialization_missing:{}",
                    field.skill_id
                ))
            })?;
            if reference.parameter_count != layout.total_parameter_count {
                return Err(BrainError::Integrity(format!(
                    "runtime_dense_materialization_count_mismatch:{}",
                    field.skill_id
                )));
            }
            self.verified_private_dvec(reference)?;
            let values = read_dvec_f32(&self.root, reference)?;
            if values.len() != delta.len() {
                return Err(BrainError::Integrity(
                    "runtime_dense_materialization_length_mismatch".into(),
                ));
            }
            for (output, value) in delta.iter_mut().zip(values) {
                *output += coefficient * f64::from(value);
            }
        }
        if delta.iter().all(|value| value.abs() <= f64::EPSILON) {
            return Err(BrainError::Invalid("runtime_composed_delta_zero".into()));
        }
        let protection = project_to_safe_subspace(&delta, protected)?;
        if !protection.allowed {
            return Err(BrainError::Integrity(format!(
                "protected_cortex_damage_budget_exceeded:{:.6e}",
                protection.damage_ratio
            )));
        }
        Ok(GovernedComposition {
            delta: protection.projected.clone(),
            trust_region: trust,
            protection,
        })
    }

    pub(super) fn load_current_observation_by_semantic_sha256(
        &self,
        digest: &str,
    ) -> BrainResult<DeltaObservation> {
        if !valid_digest(digest) {
            return Err(BrainError::Invalid(
                "governed_composition_source_observation_digest_invalid".into(),
            ));
        }
        let mut matches = Vec::new();
        for observation in self.load_persisted_observations()? {
            if digest_json(&observation)? == digest {
                matches.push(observation);
            }
        }
        if matches.len() != 1 {
            return Err(BrainError::Integrity(
                "governed_composition_source_observation_not_current".into(),
            ));
        }
        matches.into_iter().next().ok_or_else(|| {
            BrainError::Integrity("governed_composition_source_observation_missing".into())
        })
    }

    pub(super) fn require_current_observation_digest(&self, digest: &str) -> BrainResult<()> {
        let _ = self.load_current_observation_by_semantic_sha256(digest)?;
        Ok(())
    }

    /// Compose only from the currently certified TIDE-X evidence. This stays
    /// crate-private: executable deltas leave the engine only through an
    /// immutable governed-composition receipt.
    pub(super) fn compose(
        &self,
        activation: &BTreeMap<SkillId, f64>,
    ) -> BrainResult<GovernedComposition> {
        self.require_current_certification()?;
        let bank = self.load_bank()?;
        let (protected, interaction, budget, causal) =
            self.load_verified_runtime_evidence(&bank)?;
        self.compose_verified_inputs(&bank, activation, &protected, &interaction, budget, &causal)
    }

    /// Compute a composition through current certified causal trust and record
    /// its executable projected delta in an immutable, ledger-bound receipt.
    /// This is the canonical operational hand-off for external actuators and
    /// for LearnedController supervision; no caller can attach free target
    /// coefficients to an observation without this receipt.
    pub(super) fn compose_and_record(
        &self,
        activation: &BTreeMap<SkillId, f64>,
        source_observation_sha256: &str,
    ) -> BrainResult<RecordedGovernedComposition> {
        self.require_current_observation_digest(source_observation_sha256)?;
        let composition = self.compose(activation)?;
        let bank = self.load_bank()?;
        let field_ids = bank
            .fields
            .iter()
            .map(|field| field.skill_id.clone())
            .collect::<Vec<_>>();
        if field_ids.is_empty()
            || activation.is_empty()
            || activation.values().any(|value| !value.is_finite())
        {
            return Err(BrainError::Integrity(
                "governed_composition_activation_contract_invalid".into(),
            ));
        }
        let bank_path = self.bank_path();
        let bank_path = existing_regular_file_under_root(&self.root, &bank_path).map_err(|_| {
            BrainError::Integrity("governed_composition_active_bank_missing".into())
        })?;
        let active_bank_sha256 =
            SkillBankDigest::from(Sha256Digest::parse(file_sha256(&bank_path)?)?);
        let sleep_path = existing_regular_file_under_root(
            &self.root,
            &self.root.join("state/sleep_state.json"),
        )?;
        let sleep_state: serde_json::Value = serde_json::from_slice(&fs::read(sleep_path)?)?;
        let report_sha256 = ReportDigest::from(Sha256Digest::parse(
            sleep_state
                .get("report_sha256")
                .and_then(serde_json::Value::as_str)
                .filter(|digest| valid_digest(digest))
                .ok_or_else(|| {
                    BrainError::Integrity("governed_composition_report_digest_missing".into())
                })?,
        )?);
        let evidence_bundle_sha256 = EvidenceBundleDigest::from(Sha256Digest::parse(
            sleep_state
                .get("evidence_bundle_sha256")
                .and_then(serde_json::Value::as_str)
                .filter(|digest| valid_digest(digest))
                .ok_or_else(|| {
                    BrainError::Integrity("governed_composition_evidence_digest_missing".into())
                })?,
        )?);
        if sleep_state
            .get("active_bank_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(active_bank_sha256.as_str())
        {
            return Err(BrainError::Integrity(
                "governed_composition_sleep_bank_mismatch".into(),
            ));
        }
        let evidence = load_sleep_evidence(&self.root)?;
        let causal_credit_sha256 = evidence.causal_credit.credit_source_sha256.clone();
        if !valid_digest(&causal_credit_sha256) {
            return Err(BrainError::Integrity(
                "governed_composition_causal_digest_invalid".into(),
            ));
        }
        let layout_ids = bank
            .fields
            .iter()
            .map(|field| {
                field.parameter_layout_sha256.clone().ok_or_else(|| {
                    BrainError::Integrity(format!(
                        "governed_composition_layout_missing:{}",
                        field.skill_id
                    ))
                })
            })
            .collect::<BrainResult<BTreeSet<_>>>()?;
        if layout_ids.len() != 1 {
            return Err(BrainError::Integrity(
                "governed_composition_layout_identity_invalid".into(),
            ));
        }
        let legacy_layout_sha256 = layout_ids
            .into_iter()
            .next()
            .ok_or_else(|| BrainError::Integrity("governed_composition_layout_missing".into()))?;
        let parameter_layout = self.load_parameter_layout(legacy_layout_sha256.as_str())?;
        let parameter_layout_authority = ParameterLayoutAuthority::open(&self.root)?;
        let parameter_layout_artifact =
            parameter_layout_authority.persist(parameter_layout.clone())?;
        let authenticated_layout =
            parameter_layout_authority.authenticate(parameter_layout_artifact.clone())?;
        let parameter_layout_sha256 = authenticated_layout.artifact.parameter_layout_sha256;
        let projected_values = composition
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
        let projected_delta = ArtifactWriteAuthority::for_internal_root(&self.root)?
            .create_content_addressed_dvec(&projected_values)?;
        if projected_delta.parameter_count != parameter_layout.total_parameter_count {
            return Err(BrainError::Integrity(
                "governed_composition_projected_delta_layout_mismatch".into(),
            ));
        }
        let operation_key = governed_composition_operation_key(&GovernedCompositionOperation {
            report_sha256: &report_sha256,
            active_bank_sha256: &active_bank_sha256,
            evidence_bundle_sha256: &evidence_bundle_sha256,
            causal_credit_sha256: &causal_credit_sha256,
            parameter_layout_artifact_sha256: &parameter_layout_artifact.sha256,
            parameter_layout_sha256: &parameter_layout_sha256,
            activation,
            projected_delta_sha256: &projected_delta.sha256,
            source_observation_sha256,
        })?;
        let root = self.root.join("state/governed_compositions");
        let by_sha = root.join("by-sha");
        let by_operation = root.join("by-operation");
        ensure_private_directory(&self.root, &by_sha)?;
        ensure_private_directory(&self.root, &by_operation)?;
        let pointer_path = by_operation.join(format!("{operation_key}.json"));
        if fs::symlink_metadata(&pointer_path).is_ok() {
            let pointer_path = existing_regular_file_under_root(&self.root, &pointer_path)?;
            let pointer: GovernedCompositionPointer =
                serde_json::from_slice(&fs::read(&pointer_path)?)?;
            if pointer.schema != "cerebro.tidex.governed_composition_pointer/v2"
                || pointer.operation_key != operation_key
                || !valid_digest(&pointer.receipt_sha256)
            {
                return Err(BrainError::Integrity(
                    "governed_composition_pointer_contract_invalid".into(),
                ));
            }
            let receipt_path = by_sha.join(format!("{}.json", pointer.receipt_sha256));
            let receipt = load_verified_governed_composition_receipt(
                &self.root,
                &receipt_path,
                &pointer.receipt_sha256,
            )?;
            if receipt.operation_key != operation_key
                || receipt.requested_activation != *activation
                || receipt.source_observation_sha256 != source_observation_sha256
            {
                return Err(BrainError::Integrity(
                    "governed_composition_pointer_receipt_mismatch".into(),
                ));
            }
            let event = ledger::find_v2_event_by_payload_string(
                &self.root,
                "governed_composition_receipt",
                "receipt_sha256",
                &pointer.receipt_sha256,
            )?
            .ok_or_else(|| {
                BrainError::Integrity("governed_composition_pointer_ledger_missing".into())
            })?;
            return Ok(RecordedGovernedComposition {
                receipt_path: receipt_path.to_string_lossy().into_owned(),
                receipt_sha256: pointer.receipt_sha256,
                ledger_event_hash: event.event_hash,
                receipt,
            });
        }

        let receipt = GovernedCompositionReceipt {
            schema: "cerebro.tidex.governed_composition_receipt/v2".into(),
            operation_key: operation_key.clone(),
            report_sha256: report_sha256.clone(),
            active_bank_sha256: active_bank_sha256.clone(),
            evidence_bundle_sha256: evidence_bundle_sha256.clone(),
            causal_credit_sha256: causal_credit_sha256.clone(),
            parameter_layout_artifact: parameter_layout_artifact.clone(),
            parameter_layout_sha256: parameter_layout_sha256.clone(),
            field_ids,
            requested_activation: activation.clone(),
            accepted_coefficients: composition.trust_region.accepted_coefficients.clone(),
            trust_region: composition.trust_region.clone(),
            projected_delta,
            protection: GovernedCompositionProtection {
                damage_ratio: composition.protection.damage_ratio,
                allowed: composition.protection.allowed,
                removed_energy: composition.protection.removed_energy,
                protected_rank: composition.protection.protected_rank,
                max_weighted_residual: composition.protection.max_weighted_residual,
            },
            source_observation_sha256: source_observation_sha256.to_string(),
        };
        let receipt_bytes = serialize_pretty_line(&receipt)?;
        let receipt_sha256 = sha256_bytes(&receipt_bytes);
        let receipt_path = by_sha.join(format!("{receipt_sha256}.json"));
        write_new_private(&self.root, &receipt_path, &receipt_bytes)?;
        let event = if let Some(existing) = ledger::find_v2_event_by_payload_string(
            &self.root,
            "governed_composition_receipt",
            "receipt_sha256",
            &receipt_sha256,
        )? {
            let payload = existing.payload()?;
            if payload.get("schema").and_then(serde_json::Value::as_str)
                != Some("cerebro.tidex.governed_composition_ledger_binding/v2")
                || payload
                    .get("receipt_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(receipt_sha256.as_str())
                || payload
                    .get("operation_key")
                    .and_then(serde_json::Value::as_str)
                    != Some(operation_key.as_str())
                || payload
                    .get("report_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(report_sha256.as_str())
                || payload
                    .get("active_bank_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(active_bank_sha256.as_str())
                || payload
                    .get("causal_credit_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(causal_credit_sha256.as_str())
                || payload
                    .get("evidence_bundle_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(evidence_bundle_sha256.as_str())
                || payload
                    .get("parameter_layout_artifact_path")
                    .and_then(serde_json::Value::as_str)
                    != Some(parameter_layout_artifact.path.to_string_lossy().as_ref())
                || payload
                    .get("parameter_layout_artifact_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(parameter_layout_artifact.sha256.as_str())
                || payload
                    .get("parameter_layout_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(parameter_layout_sha256.as_digest().as_str())
                || payload
                    .get("source_observation_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(source_observation_sha256)
            {
                return Err(BrainError::Integrity(
                    "governed_composition_ledger_payload_mismatch".into(),
                ));
            }
            existing
        } else {
            ledger::append(
                &self.root,
                "governed_composition_receipt",
                json!({
                    "schema":"cerebro.tidex.governed_composition_ledger_binding/v2",
                    "receipt_sha256":receipt_sha256,
                    "operation_key":operation_key,
                    "report_sha256":report_sha256,
                    "active_bank_sha256":active_bank_sha256,
                    "evidence_bundle_sha256":evidence_bundle_sha256,
                    "causal_credit_sha256":causal_credit_sha256,
                    "parameter_layout_artifact_path":parameter_layout_artifact.path,
                    "parameter_layout_artifact_sha256":parameter_layout_artifact.sha256,
                    "parameter_layout_sha256":parameter_layout_sha256,
                    "source_observation_sha256":source_observation_sha256,
                }),
            )?
        };
        // Reopen through the full causal/trust/protection rederivation before
        // publishing an operation pointer.  If sleep, bank, evidence, or the
        // source observation changed after `compose`, this fails closed rather
        // than handing a stale in-memory delta to an actuator.
        let verified_before_pointer =
            load_verified_governed_composition_receipt(&self.root, &receipt_path, &receipt_sha256)?;
        if verified_before_pointer != receipt {
            return Err(BrainError::Integrity(
                "governed_composition_pre_pointer_revalidation_mismatch".into(),
            ));
        }
        let pointer = GovernedCompositionPointer {
            schema: "cerebro.tidex.governed_composition_pointer/v2".into(),
            operation_key,
            receipt_sha256: receipt_sha256.clone(),
        };
        let pointer_bytes = serialize_pretty_line(&pointer)?;
        match fs::symlink_metadata(&pointer_path) {
            Ok(_) => {
                let pointer_path = existing_regular_file_under_root(&self.root, &pointer_path)?;
                if fs::read(&pointer_path)? != pointer_bytes {
                    return Err(BrainError::Integrity(
                        "governed_composition_pointer_artifact_collision".into(),
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                write_new_private(&self.root, &pointer_path, &pointer_bytes)?;
            }
            Err(error) => return Err(error.into()),
        }
        let receipt =
            load_verified_governed_composition_receipt(&self.root, &receipt_path, &receipt_sha256)?;
        let verified_event = ledger::find_v2_event_by_payload_string(
            &self.root,
            "governed_composition_receipt",
            "receipt_sha256",
            &receipt_sha256,
        )?
        .ok_or_else(|| {
            BrainError::Integrity("governed_composition_receipt_ledger_missing".into())
        })?;
        if verified_event.event_hash != event.event_hash {
            return Err(BrainError::Integrity(
                "governed_composition_post_pointer_ledger_changed".into(),
            ));
        }
        Ok(RecordedGovernedComposition {
            receipt_path: receipt_path.to_string_lossy().into_owned(),
            receipt_sha256,
            ledger_event_hash: event.event_hash,
            receipt,
        })
    }

    pub(super) fn cognitive_route_activation(
        &self,
        route: &FieldRoutingDecision,
    ) -> BrainResult<BTreeMap<SkillId, f64>> {
        if route.schema != "cerebro.tidex.cognitive_field_routing/v1"
            || route.field_ids.is_empty()
            || route.field_ids.len() != route.coefficients.len()
            || route.selected_field_ids.is_empty()
            || route
                .coefficients
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(BrainError::Invalid(
                "cognitive_route_contract_invalid".into(),
            ));
        }
        let unique_ids = route.field_ids.iter().collect::<BTreeSet<_>>();
        let selected = route.selected_field_ids.iter().collect::<BTreeSet<_>>();
        if unique_ids.len() != route.field_ids.len()
            || selected.len() != route.selected_field_ids.len()
            || selected.iter().any(|id| !unique_ids.contains(id))
        {
            return Err(BrainError::Invalid(
                "cognitive_route_identity_invalid".into(),
            ));
        }
        let activation = route
            .field_ids
            .iter()
            .zip(&route.coefficients)
            .filter(|(_, coefficient)| **coefficient > f64::EPSILON)
            .map(|(id, coefficient)| (id.clone(), *coefficient))
            .collect::<BTreeMap<_, _>>();
        let activated_ids = activation.keys().collect::<BTreeSet<_>>();
        if activation.is_empty()
            || selected.len() != activated_ids.len()
            || selected.iter().any(|id| !activated_ids.contains(id))
        {
            return Err(BrainError::Invalid(
                "cognitive_route_selection_mismatch".into(),
            ));
        }
        Ok(activation)
    }

    /// Record an executable Dynamic Cognitive Field route through the shared
    /// causal-trust/Protected-Cortex authority. A route never exposes a raw
    /// parameter delta to callers.
    pub(super) fn compose_cognitive_route_and_record(
        &self,
        route: &FieldRoutingDecision,
        source_observation_sha256: &str,
    ) -> BrainResult<RecordedGovernedComposition> {
        self.require_current_observation_digest(source_observation_sha256)?;
        let activation = self.cognitive_route_activation(route)?;
        self.compose_and_record(&activation, source_observation_sha256)
    }

    pub(super) fn learned_controller_activation(
        &self,
        runtime: &RuntimeLearnedController,
        state: &[f64],
        observation: &[f64],
    ) -> BrainResult<BTreeMap<SkillId, f64>> {
        self.require_current_certification()?;
        let bank = self.load_bank()?;
        let bank_ids = bank
            .fields
            .iter()
            .map(|field| field.skill_id.clone())
            .collect::<Vec<_>>();
        if runtime.schema != "cerebro.tidex.runtime_learned_controller/v1"
            || runtime.field_ids != bank_ids
            || runtime.controller.coefficient_dim != bank.fields.len()
        {
            return Err(BrainError::Integrity(
                "runtime_learned_controller_identity_mismatch".into(),
            ));
        }
        let decision = runtime.controller.decide(state, observation)?;
        let activation = runtime
            .field_ids
            .iter()
            .zip(&decision.coefficients)
            .filter(|(_, coefficient)| coefficient.abs() > f64::EPSILON)
            .map(|(id, coefficient)| (id.clone(), *coefficient))
            .collect::<BTreeMap<_, _>>();
        if activation.is_empty() {
            return Err(BrainError::Invalid(
                "runtime_learned_controller_zero_activation".into(),
            ));
        }
        Ok(activation)
    }

    /// Execute the current receipt-verified LearnedController through its
    /// canonical, durable execution cycle.  The caller can supply only a
    /// finite state vector and a semantic reference to an active observation;
    /// the controller, functional response, activation, causal trust,
    /// protection result, composition delta, and execution receipt are all
    /// resolved and sealed by TIDE-X.
    pub fn compose_current_learned_controller_and_record(
        &self,
        invocation: &ControllerInvocation,
    ) -> BrainResult<RecordedControllerExecution> {
        self.with_engine_authority(|| {
            self.compose_current_learned_controller_and_record_under_authority(invocation)
        })
    }

    pub(super) fn compose_current_learned_controller_and_record_under_authority(
        &self,
        invocation: &ControllerInvocation,
    ) -> BrainResult<RecordedControllerExecution> {
        self.require_canonical_runtime_config()?;
        invocation.validate()?;
        let before =
            load_persisted_runtime_learned_controller(&self.root, invocation.session_id.as_str())?;
        let source = self.load_current_observation_by_semantic_sha256(
            &invocation.promoted_observation_semantic_sha256,
        )?;
        let activation = self.learned_controller_activation(
            &before.receipt.runtime_controller,
            &invocation.state_before,
            &source.functional_response,
        )?;
        let composition = self.compose_and_record(
            &activation,
            &invocation.promoted_observation_semantic_sha256,
        )?;
        let after =
            load_persisted_runtime_learned_controller(&self.root, invocation.session_id.as_str())?;
        if before.receipt_sha256 != after.receipt_sha256 {
            return Err(BrainError::Integrity(
                "controller_execution_controller_receipt_changed_during_composition".into(),
            ));
        }
        let governed = load_verified_governed_composition_receipt(
            &self.root,
            Path::new(&composition.receipt_path),
            &composition.receipt_sha256,
        )?;
        if governed != composition.receipt
            || governed.source_observation_sha256 != invocation.promoted_observation_semantic_sha256
        {
            return Err(BrainError::Integrity(
                "controller_execution_governed_composition_binding_invalid".into(),
            ));
        }
        let recorded = persist_controller_execution(
            &self.root,
            ControllerExecutionReceipt {
                schema: "cerebro.tidex.controller_execution_receipt/v1".into(),
                session_id: invocation.session_id.clone(),
                invocation: invocation.clone(),
                // This is a semantic digest of the strict invocation contract,
                // not a hash self-asserted by an arbitrary CLI file encoding.
                invocation_sha256: Sha256Digest::parse(digest_json(invocation)?)?,
                state_before_sha256: Sha256Digest::parse(digest_json(&invocation.state_before)?)?,
                controller_receipt_sha256: Sha256Digest::parse(after.receipt_sha256)?,
                promoted_observation_semantic_sha256: invocation
                    .promoted_observation_semantic_sha256
                    .clone(),
                governed_composition_receipt_sha256: Sha256Digest::parse(
                    composition.receipt_sha256,
                )?,
            },
        )?;
        verify_controller_execution_receipt(&self.root, &recorded)?;
        Ok(recorded)
    }

    pub(super) fn cognitive_route_from_drive(
        &self,
        initial: &[f64],
        drive: &CognitiveFieldDrive,
        top_k: usize,
        minimum_activation: f64,
    ) -> BrainResult<(CognitiveFieldState, FieldRoutingDecision)> {
        self.require_current_certification()?;
        let bank = self.load_bank()?;
        let (_, interaction, _, causal) = self.load_verified_runtime_evidence(&bank)?;
        let model = DynamicCognitiveField::build(
            &bank.fields,
            &interaction,
            &causal,
            CognitiveFieldConfig::default(),
        )?;
        let state = model.evolve(initial, drive)?;
        if !state.converged {
            return Err(BrainError::Numerical(
                "runtime_cognitive_field_not_converged".into(),
            ));
        }
        let route = model.route_top_k(&state, top_k, minimum_activation)?;
        Ok((state, route))
    }

    /// Evolve the Dynamic Cognitive Field and make its chosen route executable
    /// only by creating a governed composition receipt. This is the canonical
    /// runtime integration for the Cognitive Field, not a bench-only side
    /// channel.
    pub fn compose_cognitive_drive_and_record(
        &self,
        initial: &[f64],
        drive: &CognitiveFieldDrive,
        top_k: usize,
        minimum_activation: f64,
        source_observation_sha256: &str,
    ) -> BrainResult<RecordedGovernedCognitiveComposition> {
        self.with_engine_authority(|| {
            self.compose_cognitive_drive_and_record_under_authority(
                initial,
                drive,
                top_k,
                minimum_activation,
                source_observation_sha256,
            )
        })
    }

    pub(super) fn compose_cognitive_drive_and_record_under_authority(
        &self,
        initial: &[f64],
        drive: &CognitiveFieldDrive,
        top_k: usize,
        minimum_activation: f64,
        source_observation_sha256: &str,
    ) -> BrainResult<RecordedGovernedCognitiveComposition> {
        self.require_current_observation_digest(source_observation_sha256)?;
        let (state, route) =
            self.cognitive_route_from_drive(initial, drive, top_k, minimum_activation)?;
        let composition =
            self.compose_cognitive_route_and_record(&route, source_observation_sha256)?;
        Ok(RecordedGovernedCognitiveComposition {
            state,
            route,
            composition,
        })
    }

    pub fn search_by_function(
        &self,
        query: &[f64],
        limit: usize,
    ) -> BrainResult<Vec<(SkillId, f64)>> {
        if query.is_empty()
            || query.len() > MAX_ENGINE_FUNCTIONAL_RESPONSE_DIMENSION
            || query.iter().any(|value| !value.is_finite())
            || limit > MAX_ENGINE_SKILL_FIELDS
        {
            return Err(BrainError::Invalid(
                "functional_search_request_invalid".into(),
            ));
        }
        let bank = self.load_bank()?;
        self.require_current_certification()?;
        let mut scored = bank
            .fields
            .iter()
            .filter(|f| f.functional_signature.len() == query.len() && !query.is_empty())
            .map(|f| Ok((f.skill_id.clone(), cosine(&f.functional_signature, query)?)))
            .collect::<BrainResult<Vec<_>>>()?;
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(limit);
        Ok(scored)
    }

    pub fn status(&self) -> BrainResult<serde_json::Value> {
        let ledger = ledger::verify(&self.root)?;
        let bank = self.load_bank()?;
        let health = self.runtime_integrity_health()?;
        let head = self.load_canonical_head_if_present()?;
        Ok(json!({
            "schema":"cerebro.tidex.status/v3",
            "revision":head.as_ref().map(|value| value.revision),
            "head_digest":head.as_ref().map(|value| value.manifest_digest.clone()),
            "corpus_digest":head.as_ref().and_then(|value| value.corpus_digest.clone()),
            "incomplete_transition":head.as_ref().and_then(|value| value.incomplete_transition.clone()),
            "ledger_events":ledger.events,
            "ledger_head":ledger.head,
            "skill_generation":bank.generation,
            "skill_count":bank.fields.len(),
            "external_integration":false,
            "network_runtime":false,
            "integrity":health,
        }))
    }

    pub fn skill_subspace_overlap(fields: &[SkillField], truth: &[Vec<f64>]) -> BrainResult<f64> {
        if fields.is_empty() || truth.is_empty() {
            return Ok(0.0);
        }
        // Latent subspaces are identifiable before their internal coordinate
        // system is. Measure how much of each known direction lies in the
        // recovered orthonormal span, rather than demanding an arbitrary PCA
        // axis to equal an arbitrary ground-truth axis.
        let mut contributions = Vec::with_capacity(truth.len());
        for truth_direction in truth {
            let truth_norm = norm(truth_direction)?.max(1e-15);
            let mut projected_terms = Vec::with_capacity(fields.len());
            for field in fields {
                let value = crate::linalg::dot(truth_direction, &field.direction)?;
                let squared = value.abs() * value.abs();
                if !squared.is_finite() {
                    return Err(BrainError::Numerical(
                        "skill_subspace_overlap_projected_overflow".into(),
                    ));
                }
                projected_terms.push(squared);
            }
            let projected = compensated_sum(projected_terms)?;
            if !projected.is_finite() {
                return Err(BrainError::Numerical(
                    "skill_subspace_overlap_projected_overflow".into(),
                ));
            }
            let ratio = projected.sqrt() / truth_norm;
            if !ratio.is_finite() {
                return Err(BrainError::Numerical(
                    "skill_subspace_overlap_ratio_overflow".into(),
                ));
            }
            contributions.push(ratio);
        }
        let total = compensated_sum(contributions)?;
        let overlap = total / truth.len() as f64;
        if !overlap.is_finite() {
            return Err(BrainError::Numerical(
                "skill_subspace_overlap_nonfinite".into(),
            ));
        }
        Ok(overlap)
    }
    pub fn bank_energy(&self) -> BrainResult<f64> {
        let bank = self.load_bank()?;
        let energy = compensated_sum(bank.fields.iter().flat_map(|field| {
            field
                .direction
                .iter()
                .map(|value| value.abs() * value.abs())
        }))
        .map_err(|error| BrainError::Integrity(format!("skill_bank_energy_invalid:{error}")))?;
        self.require_current_certification()?;
        Ok(energy)
    }
}
