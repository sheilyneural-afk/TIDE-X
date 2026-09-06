#![allow(clippy::needless_range_loop)]

use super::*;

impl BrainEngine {
    pub(super) fn structured_sources(
        &self,
        observations: &[DeltaObservation],
    ) -> BrainResult<Option<(String, ParameterBlockLayout, Vec<StructuredSource>)>> {
        let populated = observations
            .iter()
            .filter(|observation| {
                observation.dense_artifact.is_some()
                    || observation.parameter_layout_sha256.is_some()
            })
            .count();
        if populated == 0 {
            if self.config.require_structured_geometry_for_promotion {
                return Err(BrainError::Invalid(
                    "structured_geometry_dense_evidence_required".into(),
                ));
            }
            return Ok(None);
        }
        if populated != observations.len() {
            return Err(BrainError::Invalid(
                "structured_geometry_partial_dense_evidence".into(),
            ));
        }
        let layout_ids = observations
            .iter()
            .map(|observation| {
                observation
                    .parameter_layout_sha256
                    .clone()
                    .ok_or_else(|| BrainError::Invalid("parameter_layout_reference_missing".into()))
            })
            .collect::<BrainResult<BTreeSet<_>>>()?;
        if layout_ids.len() != 1 {
            return Err(BrainError::Invalid(
                "structured_geometry_multiple_parameter_layouts".into(),
            ));
        }
        let layout_sha = layout_ids.into_iter().next().ok_or_else(|| {
            BrainError::Integrity("structured_geometry_parameter_layout_missing".into())
        })?;
        let layout = self.load_parameter_layout(layout_sha.as_str())?;
        let mut sources = Vec::with_capacity(observations.len());
        for observation in observations {
            let artifact = observation
                .dense_artifact
                .clone()
                .ok_or_else(|| BrainError::Invalid("dense_artifact_reference_missing".into()))?;
            let path = self.verified_private_dvec(&artifact)?;
            let inspected = inspect_dvec(&self.root, &path)?;
            if inspected.sha256 != artifact.sha256.to_ascii_lowercase()
                || inspected.parameter_count != artifact.parameter_count
                || inspected.parameter_count != layout.total_parameter_count
            {
                return Err(BrainError::Integrity(format!(
                    "dense_artifact_reference_mismatch:{}",
                    observation.observation_id
                )));
            }
            sources.push(StructuredSource {
                observation_id: observation.observation_id.clone(),
                artifact,
                reliability: observation.reliability,
            });
        }
        Ok(Some((layout_sha.to_string(), layout, sources)))
    }

    pub(super) fn materialize_dense_fields(
        &self,
        fields: &mut [SkillField],
        source_mixtures: &[Vec<f64>],
        observations: &[DeltaObservation],
    ) -> BrainResult<()> {
        if fields.is_empty()
            || fields.len() != source_mixtures.len()
            || source_mixtures
                .iter()
                .any(|mixture| mixture.len() != observations.len())
        {
            return Err(BrainError::Invalid(
                "dense_field_materialization_shape".into(),
            ));
        }
        let (layout_sha, layout, _) = self
            .structured_sources(observations)?
            .ok_or_else(|| BrainError::Invalid("dense_field_sources_required".into()))?;
        for (field_index, field) in fields.iter_mut().enumerate() {
            if self.config.require_structured_geometry_for_promotion
                && field.structured_geometry.is_none()
            {
                return Err(BrainError::Integrity(format!(
                    "dense_field_missing_structured_geometry:{}",
                    field.skill_id
                )));
            }
            if field.parameter_layout_sha256.as_deref() != Some(layout_sha.as_str()) {
                return Err(BrainError::Integrity(format!(
                    "dense_field_layout_identity_mismatch:{}",
                    field.skill_id
                )));
            }
            let mixture = &source_mixtures[field_index];
            let support = source_support_indices(mixture)?;
            let mut sources = Vec::with_capacity(support.len());
            for index in support {
                let reference = observations[index].dense_artifact.clone().ok_or_else(|| {
                    BrainError::Integrity("dense_field_source_artifact_missing".into())
                })?;
                sources.push((reference, mixture[index]));
            }
            if sources.is_empty() {
                return Err(BrainError::Numerical(format!(
                    "dense_field_materialization_support_empty:{}",
                    field.skill_id
                )));
            }
            let materialized = ArtifactWriteAuthority::for_internal_root(&self.root)?
                .combine_content_addressed_dvec(&sources)?;
            if materialized.parameter_count != layout.total_parameter_count {
                return Err(BrainError::Integrity(format!(
                    "dense_field_materialized_count_mismatch:{}",
                    field.skill_id
                )));
            }
            field.dense_materialization = Some(materialized);
        }
        Ok(())
    }

    /// Re-derive dense field references without creating artifacts.  A receipt
    /// verifier must never materialize a missing candidate as a side effect:
    /// the exact content-addressed dvec must already exist and match the same
    /// f64-to-f32 arithmetic used at promotion time.
    pub(super) fn verify_dense_field_materializations(
        &self,
        fields: &[SkillField],
        source_mixtures: &[Vec<f64>],
        observations: &[DeltaObservation],
    ) -> BrainResult<()> {
        if fields.is_empty()
            || fields.len() != source_mixtures.len()
            || source_mixtures
                .iter()
                .any(|mixture| mixture.len() != observations.len())
        {
            return Err(BrainError::Integrity(
                "dense_field_materialization_verification_shape".into(),
            ));
        }
        let (layout_sha, layout, _) = self
            .structured_sources(observations)?
            .ok_or_else(|| BrainError::Integrity("dense_field_sources_required".into()))?;
        for (field, mixture) in fields.iter().zip(source_mixtures) {
            if self.config.require_structured_geometry_for_promotion
                && field.structured_geometry.is_none()
            {
                return Err(BrainError::Integrity(format!(
                    "dense_field_missing_structured_geometry:{}",
                    field.skill_id
                )));
            }
            if field.parameter_layout_sha256.as_deref() != Some(layout_sha.as_str()) {
                return Err(BrainError::Integrity(format!(
                    "dense_field_layout_identity_mismatch:{}",
                    field.skill_id
                )));
            }
            let reference = field.dense_materialization.as_ref().ok_or_else(|| {
                BrainError::Integrity(format!(
                    "dense_field_materialization_missing:{}",
                    field.skill_id
                ))
            })?;
            let support = source_support_indices(mixture)?;
            if support.is_empty() {
                return Err(BrainError::Integrity(format!(
                    "dense_field_materialization_support_empty:{}",
                    field.skill_id
                )));
            }
            let sources = support
                .into_iter()
                .map(|index| {
                    let source = observations[index].dense_artifact.clone().ok_or_else(|| {
                        BrainError::Integrity("dense_field_source_artifact_missing".into())
                    })?;
                    Ok((source, mixture[index]))
                })
                .collect::<BrainResult<Vec<_>>>()?;
            let derived = derive_content_addressed_dvec_combination(&self.root, &sources)?;
            if &derived != reference || reference.parameter_count != layout.total_parameter_count {
                return Err(BrainError::Integrity(format!(
                    "dense_field_materialization_rederivation_mismatch:{}",
                    field.skill_id
                )));
            }
            self.verified_private_dvec(reference)?;
        }
        Ok(())
    }

    pub(super) fn representation_observations(
        &self,
        observations: &[DeltaObservation],
    ) -> BrainResult<Option<(String, Vec<RepresentationObservation>)>> {
        let populated = observations
            .iter()
            .filter(|observation| {
                observation.representation_artifact.is_some()
                    || observation.representation_protocol_sha256.is_some()
            })
            .count();
        if populated == 0 {
            if self.config.require_dual_space_for_promotion {
                return Err(BrainError::Invalid(
                    "dual_space_representation_evidence_required".into(),
                ));
            }
            return Ok(None);
        }
        if populated != observations.len() {
            return Err(BrainError::Invalid(
                "dual_space_partial_representation_evidence".into(),
            ));
        }
        let protocol_ids = observations
            .iter()
            .map(|observation| {
                observation
                    .representation_protocol_sha256
                    .clone()
                    .ok_or_else(|| {
                        BrainError::Invalid("representation_protocol_reference_missing".into())
                    })
            })
            .collect::<BrainResult<BTreeSet<_>>>()?;
        if protocol_ids.len() != 1 {
            return Err(BrainError::Invalid(
                "dual_space_multiple_representation_protocols".into(),
            ));
        }
        let protocol_sha = protocol_ids.into_iter().next().ok_or_else(|| {
            BrainError::Integrity("dual_space_representation_protocol_missing".into())
        })?;
        let protocol_path = self
            .root
            .join("state/representation_protocols/by-sha")
            .join(format!("{}.json", protocol_sha.to_ascii_lowercase()));
        let protocol_path =
            existing_regular_file_under_root(&self.root, &protocol_path).map_err(|_| {
                BrainError::Integrity("representation_protocol_artifact_invalid".into())
            })?;
        if file_sha256(&protocol_path)? != protocol_sha.to_ascii_lowercase() {
            return Err(BrainError::Integrity(
                "representation_protocol_artifact_invalid".into(),
            ));
        }
        let protocol: serde_json::Value = serde_json::from_slice(&fs::read(&protocol_path)?)?;
        let string_digest = |key: &str| -> BrainResult<String> {
            let value = protocol
                .get(key)
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    BrainError::Integrity(format!("representation_protocol_missing_digest:{key}"))
                })?;
            if !valid_digest(value) {
                return Err(BrainError::Integrity(format!(
                    "representation_protocol_invalid_digest:{key}"
                )));
            }
            Ok(value.to_string())
        };
        if protocol.get("schema").and_then(serde_json::Value::as_str)
            != Some("cerebro.tidex.representation_protocol/v1")
            || protocol
                .get("task_labels_used")
                .and_then(serde_json::Value::as_bool)
                != Some(false)
            || protocol
                .get("probe_vocabulary_overlap")
                .and_then(serde_json::Value::as_array)
                .is_none_or(|values| !values.is_empty())
        {
            return Err(BrainError::Integrity(
                "representation_protocol_semantic_contract_invalid".into(),
            ));
        }
        let probe_sha = string_digest("probe_sha256")?;
        let probe_text_sha = string_digest("probe_text_sha256")?;
        let _forbidden_vocab_sha = string_digest("forbidden_vocabulary_sha256")?;
        let _source_representation_sha = string_digest("source_representation_sha256")?;
        if probe_sha != probe_text_sha {
            return Err(BrainError::Integrity(
                "representation_probe_text_digest_mismatch".into(),
            ));
        }
        let sketch_dim = protocol
            .get("sketch_dim")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                BrainError::Integrity("representation_protocol_sketch_dim_invalid".into())
            })?;
        let probe_count = protocol
            .get("probe_count")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                BrainError::Integrity("representation_protocol_probe_count_invalid".into())
            })?;
        let layer_count = protocol
            .get("layer_count")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                BrainError::Integrity("representation_protocol_layer_count_invalid".into())
            })?;
        let hidden_dim = protocol
            .get("hidden_dim")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                BrainError::Integrity("representation_protocol_hidden_dim_invalid".into())
            })?;
        let raw_dimension = protocol
            .get("raw_dimension_per_observation")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                BrainError::Integrity("representation_protocol_raw_dim_invalid".into())
            })?;
        let expected_raw = probe_count
            .checked_mul(layer_count)
            .and_then(|value| value.checked_mul(hidden_dim))
            .ok_or_else(|| {
                BrainError::Invalid("representation_protocol_dimension_overflow".into())
            })?;
        if expected_raw != raw_dimension {
            return Err(BrainError::Integrity(
                "representation_protocol_raw_dimension_mismatch".into(),
            ));
        }
        let mut result = Vec::with_capacity(observations.len());
        for observation in observations {
            let reference = observation
                .representation_artifact
                .as_ref()
                .ok_or_else(|| {
                    BrainError::Invalid("representation_artifact_reference_missing".into())
                })?;
            let path = existing_regular_file_under_root(&self.root, Path::new(&reference.path))
                .map_err(|_| {
                    BrainError::Integrity("representation_artifact_path_invalid".into())
                })?;
            if path != Path::new(&reference.path) {
                return Err(BrainError::Integrity(
                    "representation_artifact_path_normalization_mismatch".into(),
                ));
            }
            if reference.element_count != sketch_dim {
                return Err(BrainError::Integrity(format!(
                    "representation_artifact_count_mismatch:{}",
                    observation.observation_id
                )));
            }
            let shift = read_f64_artifact(&self.root, reference)?;
            if shift.len() != sketch_dim as usize {
                return Err(BrainError::Integrity(
                    "representation_artifact_loaded_count_mismatch".into(),
                ));
            }
            result.push(RepresentationObservation {
                observation_id: observation.observation_id.clone(),
                shift,
            });
        }
        Ok(Some((protocol_sha.to_string(), result)))
    }

    pub(super) fn validate_observations(&self, obs: &[DeltaObservation]) -> BrainResult<usize> {
        if obs.len() < self.config.min_observations {
            return Err(BrainError::Invalid("minimum_six_observations".into()));
        }
        if obs.len() > MAX_ENGINE_OBSERVATIONS {
            return Err(BrainError::Invalid(
                "observation_count_limit_exceeded".into(),
            ));
        }
        let dim = obs[0].delta.len();
        if dim == 0 {
            return Err(BrainError::Invalid("empty_delta".into()));
        }
        if dim > MAX_ENGINE_PARAMETER_DIMENSION {
            return Err(BrainError::Invalid(
                "parameter_dimension_limit_exceeded".into(),
            ));
        }
        let total_delta_elements = obs
            .len()
            .checked_mul(dim)
            .ok_or_else(|| BrainError::Invalid("delta_element_count_overflow".into()))?;
        if total_delta_elements > MAX_ENGINE_TOTAL_DELTA_ELEMENTS {
            return Err(BrainError::Invalid(
                "delta_element_count_limit_exceeded".into(),
            ));
        }

        let mut ids = BTreeSet::new();
        let mut groups = BTreeSet::new();
        let mut total_functional_elements = 0usize;
        for o in obs {
            if !ids.insert(o.observation_id.as_str()) {
                return Err(BrainError::Invalid("duplicate_observation_id".into()));
            }
            if o.from_checkpoint.trim().is_empty()
                || o.to_checkpoint.trim().is_empty()
                || o.from_checkpoint == o.to_checkpoint
                || o.from_checkpoint.len() > MAX_ENGINE_TEXT_BYTES
                || o.to_checkpoint.len() > MAX_ENGINE_TEXT_BYTES
            {
                return Err(BrainError::Invalid("checkpoint_edge_invalid".into()));
            }
            if o.delta.len() != dim || o.delta.iter().any(|v| !v.is_finite()) {
                return Err(BrainError::Invalid("delta_invalid".into()));
            }
            if !o.reliability.is_finite() || o.reliability <= 0.0 || o.reliability > 1.0 {
                return Err(BrainError::Invalid("reliability_invalid".into()));
            }
            if o.independence_group.trim().is_empty()
                || o.independence_group.len() > MAX_ENGINE_TEXT_BYTES
            {
                return Err(BrainError::Invalid("independence_group_required".into()));
            }
            groups.insert(o.independence_group.as_str());
            if groups.len() > MAX_ENGINE_INDEPENDENCE_GROUPS {
                return Err(BrainError::Invalid(
                    "independence_group_limit_exceeded".into(),
                ));
            }
            if o.confounders.len() > MAX_ENGINE_CONFOUNDERS_PER_OBSERVATION
                || o.confounders.iter().any(|confounder| {
                    confounder.name.trim().is_empty()
                        || confounder.name.len() > MAX_ENGINE_TEXT_BYTES
                        || !confounder.value.is_finite()
                })
            {
                return Err(BrainError::Invalid("confounder_contract_invalid".into()));
            }
            let mut confounder_names = BTreeSet::new();
            if o.confounders
                .iter()
                .any(|confounder| !confounder_names.insert(confounder.name.as_str()))
            {
                return Err(BrainError::Invalid("confounder_duplicate_name".into()));
            }
            let lineage = &o.experiment_lineage;
            if lineage.run_id.trim().is_empty()
                || lineage.replicate_id.trim().is_empty()
                || lineage.randomization_id.trim().is_empty()
                || lineage.run_id.len() > MAX_ENGINE_TEXT_BYTES
                || lineage.replicate_id.len() > MAX_ENGINE_TEXT_BYTES
                || lineage.randomization_id.len() > MAX_ENGINE_TEXT_BYTES
                || !valid_digest(&lineage.dataset_split_digest)
                || !valid_digest(&lineage.initial_checkpoint_digest)
                || !valid_digest(&lineage.optimizer_config_digest)
                || !valid_digest(&lineage.template_config_digest)
            {
                return Err(BrainError::Invalid("experiment_lineage_required".into()));
            }
            if o.functional_response.len() > MAX_ENGINE_FUNCTIONAL_RESPONSE_DIMENSION
                || o.functional_response.iter().any(|v| !v.is_finite())
            {
                return Err(BrainError::Invalid(
                    "functional_response_contract_invalid".into(),
                ));
            }
            total_functional_elements = total_functional_elements
                .checked_add(o.functional_response.len())
                .ok_or_else(|| {
                    BrainError::Invalid("functional_response_element_count_overflow".into())
                })?;
            if total_functional_elements > MAX_ENGINE_TOTAL_DELTA_ELEMENTS {
                return Err(BrainError::Invalid(
                    "functional_response_element_count_limit_exceeded".into(),
                ));
            }
        }
        Ok(dim)
    }

    pub fn analyze(&self, observations: &[DeltaObservation]) -> BrainResult<ReconstructionReport> {
        let canonical = canonical_observations(observations);
        self.analyze_canonical(&canonical)
    }

    pub(super) fn analyze_canonical(
        &self,
        obs: &[DeltaObservation],
    ) -> BrainResult<ReconstructionReport> {
        let dim = self.validate_observations(obs)?;
        let (source_tree_digest, config_digest, analysis_version_digest) =
            analysis_identity(&self.config)?;
        let observation_set_digest = observation_set_digest(obs)?;
        let sbas = reconstruct_trajectory(obs, self.config.ridge)?;
        let conf = remove_confounders(obs, self.config.ridge)?;
        let aperture_independence =
            estimate_aperture_independence(obs, self.config.min_independent_apertures)?;
        let group_independence_weights = aperture_independence
            .group_ids
            .iter()
            .cloned()
            .zip(
                aperture_independence
                    .group_independence_weights
                    .iter()
                    .copied(),
            )
            .collect::<BTreeMap<_, _>>();
        let raw = Matrix::from_rows(
            &obs.iter()
                .map(|observation| observation.delta.clone())
                .collect::<Vec<_>>(),
        )?;
        let mut group_counts = BTreeMap::<&str, usize>::new();
        for o in obs {
            *group_counts
                .entry(o.independence_group.as_str())
                .or_default() += 1;
        }
        let weights = obs
            .iter()
            .map(|o| {
                let independence_weight = group_independence_weights
                    .get(o.independence_group.as_str())
                    .copied()
                    .ok_or_else(|| {
                        BrainError::Integrity("analysis_group_independence_weight_missing".into())
                    })?;
                Ok::<f64, BrainError>(
                    o.reliability * independence_weight
                        / (group_counts[o.independence_group.as_str()] as f64).sqrt(),
                )
            })
            .collect::<BrainResult<Vec<_>>>()?;
        let groups = obs
            .iter()
            .map(|o| o.independence_group.clone())
            .collect::<Vec<_>>();
        let generation = obs.iter().map(|o| o.generation).max().unwrap_or(0);

        // Hypothesis A: energy/spectral tomography over confounder-residualized
        // deltas. This remains useful when capabilities are continuously mixed.
        let mut spectral =
            reconstruct_skill_fields(&conf.residuals, &weights, &groups, generation, &self.config)?;
        let spectral_functional = fit_functional_map(
            &spectral.coefficients,
            obs,
            self.config.ridge,
            self.config.min_independent_apertures,
        )?;
        attach_signatures(&mut spectral.fields, &spectral_functional)?;
        let input_rms = stable_rms(obs.iter().flat_map(|o| o.delta.iter().copied()))?.max(1e-15);
        let spectral_normalized_rms = spectral.reconstruction_rms / input_rms;
        let transform_t = conf.source_transform.transpose();
        let mut spectral_source_mixtures = Vec::with_capacity(spectral.source_mixtures.len());
        for mix in &spectral.source_mixtures {
            spectral_source_mixtures.push(transform_t.matvec(mix)?);
        }

        // Hypothesis B: persistent-skill tomography. Discovery uses parameter
        // geometry only and receives no task names or functional outcomes.
        // Functional identity is attached and validated afterwards by holding
        // out one independent aperture at a time.
        let (persistent, persistent_error) = match reconstruct_persistent_skill_fields(
            &raw,
            obs,
            generation,
            self.config.min_observations,
            self.config.min_independent_apertures,
        ) {
            Ok(result) => (Some(result), None),
            Err(error) => match persistent_not_applicable_reason(&error) {
                Some(reason) => (None, Some(reason)),
                None => return Err(error),
            },
        };
        let use_persistent = persistent
            .as_ref()
            .is_some_and(|persistent| persistent.functional_cv_r2 > spectral_functional.cv_r2);

        let (
            inverse_mode,
            mut fields,
            skill_source_mixtures,
            selected_rank,
            effective_rank,
            condition_estimate,
            reconstruction_rms,
            normalized_reconstruction_rms,
            functional_cv_r2,
            selected_coefficients,
        ) = if let Some(persistent) = persistent.as_ref().filter(|_| use_persistent) {
            (
                ReconstructionInverseMode::Persistent,
                persistent.fields.clone(),
                persistent.source_mixtures.clone(),
                persistent.selected_rank,
                persistent.effective_rank,
                persistent.condition_estimate,
                persistent.reconstruction_rms,
                persistent.normalized_reconstruction_rms,
                persistent.functional_cv_r2,
                persistent.coefficients.clone(),
            )
        } else {
            (
                ReconstructionInverseMode::Spectral,
                spectral.fields.clone(),
                spectral_source_mixtures,
                spectral.selected_rank,
                spectral.effective_rank,
                spectral.condition_estimate,
                spectral.reconstruction_rms,
                spectral_normalized_rms,
                spectral_functional.cv_r2,
                spectral.coefficients.clone(),
            )
        };

        attach_evidence_support(&mut fields, &skill_source_mixtures, obs)?;

        let persistent = persistent.as_ref();
        let persistent_functional_cv_r2 = persistent.map(|value| value.functional_cv_r2);
        let persistent_coherence_threshold = persistent.map(|value| value.coherence_threshold);
        let persistent_coherence_gap = persistent.map(|value| value.coherence_gap);
        let persistent_coverage_ratio = persistent.map(|value| value.coverage_ratio);
        let persistent_cluster_stability = persistent.map(|value| value.cluster_stability);
        let persistent_parametric_cluster_stability =
            persistent.map(|value| value.parametric_cluster_stability);
        let persistent_functional_cluster_stability =
            persistent.map(|value| value.functional_cluster_stability);
        let persistent_cluster_identity_min_margin =
            persistent.map(|value| value.cluster_identity_min_margin);
        let persistent_cluster_assignment_consistent =
            persistent.map(|value| value.cluster_assignment_consistent);
        let persistent_min_holdout_similarity =
            persistent.map(|value| value.min_holdout_similarity);
        let persistent_cluster_sizes = persistent
            .map(|value| value.cluster_sizes.clone())
            .unwrap_or_default();
        let resolution_map = resolution_map(
            &fields,
            &selected_coefficients,
            &weights,
            reconstruction_rms,
            self.config.ridge,
            self.config.min_identifiability_signal_to_noise,
        )?;

        let mut structured_block_count = 0usize;
        let mut structured_max_local_rank = 0usize;
        let mut structured_mean_effective_rank = 0.0f64;
        if let Some((layout_sha, layout, sources)) = self.structured_sources(obs)? {
            let geometry = reconstruct_structured_geometry(
                &self.root,
                &fields,
                &skill_source_mixtures,
                &sources,
                &layout,
                &self.config,
            )?;
            if geometry.skills.len() != fields.len()
                || geometry.block_count != layout.blocks.len()
                || geometry.total_parameter_count != layout.total_parameter_count
            {
                return Err(BrainError::Integrity(
                    "structured_geometry_result_contract_mismatch".into(),
                ));
            }
            let numerical_tolerance =
                f64::EPSILON.sqrt() * (layout.blocks.len().max(1) as f64).sqrt() * 16.0;
            let mut effective_rank_sum = 0.0;
            for (field, skill_geometry) in fields.iter_mut().zip(geometry.skills) {
                if skill_geometry.skill_id != field.skill_id
                    || skill_geometry.blocks.len() != layout.blocks.len()
                {
                    return Err(BrainError::Integrity(
                        "structured_geometry_skill_identity_mismatch".into(),
                    ));
                }
                let mut active_blocks = 0usize;
                for (block_geometry, block_layout) in
                    skill_geometry.blocks.iter().zip(&layout.blocks)
                {
                    if block_geometry.block_name != block_layout.name
                        || block_geometry.offset != block_layout.offset
                        || block_geometry.count != block_layout.count
                        || block_geometry.shape != block_layout.shape
                        || block_geometry.selected_rank != block_geometry.axes.len()
                    {
                        return Err(BrainError::Integrity(
                            "structured_geometry_block_contract_mismatch".into(),
                        ));
                    }
                    if block_geometry.block_energy > f64::EPSILON {
                        active_blocks += 1;
                        if block_geometry.selected_rank == 0
                            || block_geometry.retained_energy + numerical_tolerance
                                < self.config.target_explained_variance
                            || !block_geometry.effective_rank.is_finite()
                            || block_geometry.effective_rank <= 0.0
                        {
                            return Err(BrainError::Numerical(
                                "structured_geometry_active_block_unresolved".into(),
                            ));
                        }
                    }
                    for axis in &block_geometry.axes {
                        if axis.source_coefficients.len() != obs.len()
                            || !axis.singular_value.is_finite()
                            || axis.singular_value <= 0.0
                            || axis
                                .source_coefficients
                                .iter()
                                .any(|value| !value.is_finite())
                        {
                            return Err(BrainError::Integrity(
                                "structured_geometry_axis_contract_mismatch".into(),
                            ));
                        }
                    }
                }
                if active_blocks == 0 || skill_geometry.max_local_rank == 0 {
                    return Err(BrainError::Numerical(
                        "structured_geometry_skill_has_no_active_blocks".into(),
                    ));
                }
                structured_block_count = structured_block_count.max(active_blocks);
                structured_max_local_rank =
                    structured_max_local_rank.max(skill_geometry.max_local_rank);
                effective_rank_sum += skill_geometry.mean_effective_rank;
                field.parameter_layout_sha256 = Some(Sha256Digest::parse(layout_sha.clone())?);
                field.structured_geometry = Some(skill_geometry);
            }
            structured_mean_effective_rank = effective_rank_sum / fields.len().max(1) as f64;
        }

        let mut reasons = Vec::<PromotionBlocker>::new();
        if sbas.cycle_rms > self.config.max_cycle_rms {
            reasons.push(PromotionBlocker::CycleConsistencyFailed);
        }
        if functional_cv_r2 < self.config.min_functional_cv_r2 {
            reasons.push(PromotionBlocker::FunctionalCrossValidationFailed);
        }
        if fields.is_empty() {
            reasons.push(PromotionBlocker::NoSkillFields);
        }
        let eligible = fields
            .iter()
            .filter(|field| field.explained_variance >= self.config.min_field_explained_variance)
            .collect::<Vec<_>>();
        if eligible
            .iter()
            .any(|field| field.coherence < self.config.min_skill_coherence)
        {
            reasons.push(PromotionBlocker::SkillCoherenceFailed);
        }
        if eligible
            .iter()
            .any(|field| field.persistence < self.config.min_skill_persistence)
        {
            reasons.push(PromotionBlocker::SkillPersistenceFailed);
        }
        if group_counts.len() < self.config.min_independent_apertures {
            reasons.push(PromotionBlocker::InsufficientDeclaredApertures);
        }
        if !aperture_independence.independent_enough {
            reasons.push(PromotionBlocker::ApertureIndependenceUnresolved);
        }
        if !resolution_map.all_fields_resolved {
            reasons.push(PromotionBlocker::SkillIdentifiabilityUnresolved);
        }
        if !condition_estimate.is_finite()
            || condition_estimate > self.config.max_condition_estimate
        {
            reasons.push(PromotionBlocker::TomographyIllConditioned);
        }
        if inverse_mode == ReconstructionInverseMode::Spectral
            && normalized_reconstruction_rms
                > self.config.max_spectral_normalized_reconstruction_rms
        {
            reasons.push(PromotionBlocker::ReconstructionErrorHigh);
        }
        if inverse_mode == ReconstructionInverseMode::Persistent {
            let persistent = persistent.ok_or_else(|| {
                BrainError::Integrity("persistent_inverse_selected_without_result".into())
            })?;
            if persistent.coverage_ratio.to_bits() != 1.0_f64.to_bits() {
                reasons.push(PromotionBlocker::PersistentObservationCoverageIncomplete);
            }
            let identity_tolerance =
                f64::EPSILON.sqrt() * (persistent.selected_rank.max(1) as f64).sqrt() * 16.0;
            if !persistent.cluster_assignment_consistent {
                reasons.push(PromotionBlocker::PersistentClusterAssignmentInconsistentAcrossSpaces);
            }
            if persistent.cluster_identity_min_margin <= identity_tolerance {
                reasons.push(PromotionBlocker::PersistentClusterIdentityMarginUnresolved);
            }
            if persistent.coherence_gap <= 1e-12 {
                reasons.push(PromotionBlocker::PersistentCoherenceGapNotIdentifiable);
            }
            if persistent.min_holdout_similarity <= 0.0 {
                reasons.push(PromotionBlocker::PersistentHoldoutAlignmentNonpositive);
            }
        }

        if self.config.require_structured_geometry_for_promotion
            && fields.iter().any(|field| {
                field.structured_geometry.is_none() || field.parameter_layout_sha256.is_none()
            })
        {
            reasons.push(PromotionBlocker::StructuredGeometryRequiredForPromotion);
        }

        let mut metrics = BTreeMap::new();
        metrics.insert("cycle_rms".into(), sbas.cycle_rms);
        metrics.insert("functional_cv_r2".into(), functional_cv_r2);
        metrics.insert(
            "spectral_functional_cv_r2".into(),
            spectral_functional.cv_r2,
        );
        if let Some(value) = persistent_functional_cv_r2 {
            metrics.insert("persistent_functional_cv_r2".into(), value);
        }
        if let Some(value) = persistent_cluster_stability {
            metrics.insert("persistent_cluster_stability".into(), value);
        }
        if let Some(value) = persistent_parametric_cluster_stability {
            metrics.insert("persistent_parametric_cluster_stability".into(), value);
        }
        if let Some(value) = persistent_functional_cluster_stability {
            metrics.insert("persistent_functional_cluster_stability".into(), value);
        }
        if let Some(value) = persistent_cluster_identity_min_margin {
            metrics.insert("persistent_cluster_identity_min_margin".into(), value);
        }
        if let Some(value) = persistent_cluster_assignment_consistent {
            metrics.insert(
                "persistent_cluster_assignment_consistent".into(),
                if value { 1.0 } else { 0.0 },
            );
        }
        if let Some(value) = persistent_min_holdout_similarity {
            metrics.insert("persistent_min_holdout_similarity".into(), value);
        }
        metrics.insert(
            "aperture_effective_group_rank".into(),
            aperture_independence.effective_group_rank,
        );
        metrics.insert(
            "aperture_effective_independent_groups".into(),
            aperture_independence.effective_independent_groups,
        );
        metrics.insert(
            "aperture_numerical_group_rank".into(),
            aperture_independence.numerical_design_rank as f64,
        );
        metrics.insert(
            "identifiability_resolved_rank".into(),
            resolution_map.resolved_rank as f64,
        );
        metrics.insert(
            "identifiability_min_principal_angle_degrees".into(),
            resolution_map.min_principal_angle_degrees,
        );
        metrics.insert(
            "normalized_reconstruction_rms".into(),
            normalized_reconstruction_rms,
        );
        metrics.insert("effective_rank".into(), effective_rank);
        metrics.insert("condition_estimate".into(), condition_estimate);
        metrics.insert(
            "structured_active_block_count".into(),
            structured_block_count as f64,
        );
        metrics.insert(
            "structured_max_local_rank".into(),
            structured_max_local_rank as f64,
        );
        metrics.insert(
            "structured_mean_effective_rank".into(),
            structured_mean_effective_rank,
        );
        let field_coefficients = (0..selected_coefficients.rows)
            .map(|row| selected_coefficients.row_vec(row))
            .collect::<Vec<_>>();
        let parameter_promotable = reasons.is_empty();
        let mut representation_protocol_sha256 = None;
        let mut representation_cv_r2 = None;
        let mut representation_match_accuracy = None;
        let mut representation_mean_matched_cosine = None;
        let mut representation_min_match_margin = None;
        let mut dual_space_verified = None;
        if let Some((protocol_sha, representations)) = self.representation_observations(obs)? {
            let dual_model = DualSpaceModel {
                fields: &fields,
                field_coefficients: &field_coefficients,
                skill_source_mixtures: &skill_source_mixtures,
                parameter_inverse_mode: inverse_mode,
                parameter_promotable,
                functional_cv_r2,
            };
            let dual = analyze_dual_space(
                &dual_model,
                obs,
                &representations,
                DualSpaceAnalysisConfig {
                    ridge: self.config.ridge,
                    minimum_independence_groups: self.config.min_independent_apertures,
                    minimum_representation_cv_r2: self.config.min_representation_cv_r2,
                    minimum_match_accuracy: self.config.min_representation_match_accuracy,
                    minimum_match_margin: self.config.min_representation_match_margin,
                },
            )?;
            if dual.fields.len() != fields.len() {
                return Err(BrainError::Integrity(
                    "dual_space_field_count_mismatch".into(),
                ));
            }
            for (field, dual_field) in fields.iter_mut().zip(&dual.fields) {
                if field.skill_id != dual_field.skill_id
                    || field.functional_signature != dual_field.functional_signature
                    || dual_field.representation_signature.is_empty()
                    || dual_field
                        .representation_signature
                        .iter()
                        .any(|value| !value.is_finite())
                {
                    return Err(BrainError::Integrity(
                        "dual_space_field_identity_mismatch".into(),
                    ));
                }
                field.representation_signature = dual_field.representation_signature.clone();
            }
            metrics.insert("representation_cv_r2".into(), dual.representation_cv_r2);
            metrics.insert(
                "representation_match_accuracy".into(),
                dual.representation_match_accuracy,
            );
            metrics.insert(
                "representation_mean_matched_cosine".into(),
                dual.representation_mean_matched_cosine,
            );
            metrics.insert(
                "representation_min_match_margin".into(),
                dual.representation_min_match_margin,
            );
            metrics.insert(
                "dual_space_verified".into(),
                if dual.dual_space_verified { 1.0 } else { 0.0 },
            );
            if self.config.require_dual_space_for_promotion && !dual.representation_supported {
                reasons.push(PromotionBlocker::DualSpaceRepresentationGeneralizationUnverified);
            }
            representation_protocol_sha256 = Some(protocol_sha);
            representation_cv_r2 = Some(dual.representation_cv_r2);
            representation_match_accuracy = Some(dual.representation_match_accuracy);
            representation_mean_matched_cosine = Some(dual.representation_mean_matched_cosine);
            representation_min_match_margin = Some(dual.representation_min_match_margin);
            dual_space_verified = Some(dual.dual_space_verified);
        }
        if self.config.require_dual_space_for_promotion
            && fields
                .iter()
                .any(|field| field.representation_signature.is_empty())
        {
            reasons.push(PromotionBlocker::DualSpaceRepresentationSignatureMissing);
        }
        let promotion = PromotionDecision {
            allowed: reasons.is_empty(),
            reasons,
            metrics,
        };
        promotion.validate()?;

        Ok(ReconstructionReport {
            schema: "cerebro.tidex.reconstruction/v7".into(),
            source_tree_digest,
            config_digest,
            analysis_version_digest,
            observation_count: obs.len(),
            observation_set_digest,
            parameter_dimension: dim,
            independence_groups: group_counts.len(),
            aperture_independence,
            resolution_map,
            confounder_names: conf.design_names,
            confounder_explained_fraction: conf.explained_fraction,
            cycle_rms: sbas.cycle_rms,
            max_edge_residual: sbas.max_edge_residual,
            selected_rank,
            effective_rank,
            condition_estimate,
            reconstruction_rms,
            normalized_reconstruction_rms,
            functional_cv_r2,
            inverse_mode,
            spectral_functional_cv_r2: spectral_functional.cv_r2,
            persistent_functional_cv_r2,
            persistent_coherence_threshold,
            persistent_coherence_gap,
            persistent_coverage_ratio,
            persistent_cluster_stability,
            persistent_parametric_cluster_stability,
            persistent_functional_cluster_stability,
            persistent_cluster_identity_min_margin,
            persistent_cluster_assignment_consistent,
            persistent_min_holdout_similarity,
            persistent_cluster_sizes,
            persistent_error,
            representation_protocol_sha256,
            representation_cv_r2,
            representation_match_accuracy,
            representation_mean_matched_cosine,
            representation_min_match_margin,
            dual_space_verified,
            fields,
            field_coefficients,
            skill_source_mixtures,
            promotion,
        })
    }
}
