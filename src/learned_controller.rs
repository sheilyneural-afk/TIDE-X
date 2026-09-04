#![allow(clippy::needless_range_loop)]

use crate::authority::existing_regular_file_under_root;
use crate::contracts::{DeltaObservation, SkillBank, SkillField};
use crate::digest::Sha256Digest;
use crate::engine::{
    load_verified_governed_composition_receipt, load_verified_learning_finalization_receipt,
    LearningFinalizationReceipt,
};
use crate::error::{BrainError, BrainResult};
use crate::identity::{ObservationId, SessionId};
use crate::learning_orchestrator::{
    confined_existing_file, ensure_private_dir, load_persistent_adaptive_learning_receipt,
    private_directory_if_present, private_regular_file_if_present, sha256_bytes, write_new_private,
    write_private_atomic, EvidenceReference, LoadedAdaptiveLearningReceipt,
};
use crate::ledger;
use crate::linalg::{weighted_normal_solve, Matrix};
use crate::parametric_program::{apply_parametric_transition, compose_skill_fields};
use crate::security::{secure_file, verify_private_root};
use crate::validation::regression_r2;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControllerExample {
    pub state_before: Vec<f64>,
    pub observation: Vec<f64>,
    pub target_coefficients: Vec<f64>,
    pub reliability: f64,
    pub independence_group: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearnedController {
    pub schema: String,
    pub state_dim: usize,
    pub observation_dim: usize,
    pub coefficient_dim: usize,
    pub feature_dim: usize,
    /// coefficient output -> learned weights over state/observation features.
    pub weights: Vec<Vec<f64>>,
    pub observation_min: Vec<f64>,
    pub observation_max: Vec<f64>,
    pub ood_margin_fraction: f64,
    pub training_rms: f64,
    pub grouped_cv_r2: f64,
    pub training_groups: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeLearnedController {
    pub(crate) schema: String,
    pub(crate) field_ids: Vec<String>,
    pub(crate) controller: LearnedController,
}

/// A training record is intentionally not an in-memory observation/label pair.
/// Its learned observation comes from an authenticated aperture observation and
/// its state/label pair must exactly match an immutable supervision artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControllerTrainingRecord {
    pub schema: String,
    pub state_before: Vec<f64>,
    pub observation_id: String,
    pub observation: EvidenceReference,
    pub target_coefficients: Vec<f64>,
    pub reliability: f64,
    pub independence_group: String,
    pub adaptive_evidence_sha256: String,
    pub supervision: EvidenceReference,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControllerTrainingDataset {
    pub schema: String,
    pub session_id: String,
    pub target_digest: String,
    pub records: Vec<ControllerTrainingRecord>,
}

/// The immutable supervision record makes state and coefficient labels
/// auditable rather than allowing a caller to inject a free-form target vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControllerSupervisionEvidence {
    pub schema: String,
    pub session_id: String,
    pub target_digest: String,
    pub adaptive_evidence_sha256: String,
    pub observation_id: String,
    pub active_bank_sha256: String,
    pub field_ids: Vec<String>,
    /// Engine-issued receipt for the governed composition that produced the
    /// coefficient label. This makes free-form labels fail closed.
    pub governed_composition_receipt: EvidenceReference,
    pub state_before: Vec<f64>,
    pub target_coefficients: Vec<f64>,
    pub reliability: f64,
    pub independence_group: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearnedControllerPolicy {
    pub schema: String,
    pub ridge: f64,
    pub ood_margin_fraction: f64,
    pub minimum_grouped_cv_r2: f64,
    pub maximum_training_rms: f64,
    pub minimum_independent_groups: usize,
}

/// Inputs that tie a controller to the promoted CEREBRO/TIDE-X state. The
/// caller supplies hashes, but they are all re-derived from private artifacts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearnedControllerBinding {
    pub schema: String,
    pub session_id: String,
    pub target_digest: String,
    pub adaptive_receipt_sha256: String,
    /// Authenticated engine finalization receipt. It is the sole bridge from
    /// immutable raw adaptive evidence to the semantic digest of the exact
    /// promoted observation accepted by runtime composition.
    pub finalization_receipt: EvidenceReference,
    pub reconstruction_report: EvidenceReference,
    pub active_bank_sha256: String,
    pub dataset_sha256: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LearnedControllerEventKind {
    #[serde(rename = "controller_trained")]
    ControllerTrained,
}

impl LearnedControllerEventKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ControllerTrained => "controller_trained",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearnedControllerReceipt {
    pub schema: String,
    pub event_kind: LearnedControllerEventKind,
    pub generation: u64,
    pub prior_receipt_sha256: Option<String>,
    pub binding: LearnedControllerBinding,
    pub field_ids: Vec<String>,
    pub field_fingerprint: String,
    /// Internal, content-addressed dataset copy. The input location is not
    /// trusted after this point.
    pub dataset: EvidenceReference,
    pub policy: LearnedControllerPolicy,
    pub policy_digest: String,
    pub runtime_controller: RuntimeLearnedController,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct LearnedControllerPointer {
    schema: String,
    session_id: String,
    receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoadedLearnedControllerReceipt {
    pub receipt_sha256: String,
    pub receipt: LearnedControllerReceipt,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControllerDecision {
    pub coefficients: Vec<f64>,
    pub ood_margin_used: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearnedProgramStep {
    pub observation: Vec<f64>,
    pub state_before: Vec<f64>,
    pub field_coefficients: Vec<f64>,
    pub composed_operator: Vec<f64>,
    pub state_after: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearnedProgramExecution {
    pub final_state_index: usize,
    pub final_state: Vec<f64>,
    pub steps: Vec<LearnedProgramStep>,
}

fn validate_examples(examples: &[ControllerExample]) -> BrainResult<(usize, usize, usize)> {
    if examples.len() < 6 {
        return Err(BrainError::Invalid(
            "learned_controller_minimum_six_examples".into(),
        ));
    }
    let state_dim = examples[0].state_before.len();
    let observation_dim = examples[0].observation.len();
    let coefficient_dim = examples[0].target_coefficients.len();
    if state_dim == 0 || observation_dim == 0 || coefficient_dim == 0 {
        return Err(BrainError::Invalid(
            "learned_controller_zero_dimension".into(),
        ));
    }
    for (index, example) in examples.iter().enumerate() {
        if example.state_before.len() != state_dim
            || example.observation.len() != observation_dim
            || example.target_coefficients.len() != coefficient_dim
            || example
                .state_before
                .iter()
                .chain(&example.observation)
                .chain(&example.target_coefficients)
                .any(|value| !value.is_finite())
            || !example.reliability.is_finite()
            || !(0.0..=1.0).contains(&example.reliability)
            || example.reliability <= 0.0
            || example.independence_group.trim().is_empty()
        {
            return Err(BrainError::Invalid(format!(
                "learned_controller_example_invalid:{index}"
            )));
        }
    }
    Ok((state_dim, observation_dim, coefficient_dim))
}

/// Generic feature map. There are no semantic route IDs or operator lookups:
/// bias + recurrent state + observation + every state×observation interaction.
fn feature_vector(state: &[f64], observation: &[f64]) -> BrainResult<Vec<f64>> {
    if state.is_empty()
        || observation.is_empty()
        || state
            .iter()
            .chain(observation)
            .any(|value| !value.is_finite())
    {
        return Err(BrainError::Invalid(
            "learned_controller_feature_input_invalid".into(),
        ));
    }
    let mut features =
        Vec::with_capacity(1 + state.len() + observation.len() + state.len() * observation.len());
    features.push(1.0);
    features.extend_from_slice(state);
    features.extend_from_slice(observation);
    for state_value in state {
        for observation_value in observation {
            features.push(state_value * observation_value);
        }
    }
    Ok(features)
}

fn design_matrix(examples: &[ControllerExample]) -> BrainResult<Matrix> {
    Matrix::from_rows(
        &examples
            .iter()
            .map(|example| feature_vector(&example.state_before, &example.observation))
            .collect::<BrainResult<Vec<_>>>()?,
    )
}

fn fit_weights(
    examples: &[ControllerExample],
    coefficient_dim: usize,
    ridge: f64,
) -> BrainResult<Vec<Vec<f64>>> {
    let design = design_matrix(examples)?;
    let reliability = examples
        .iter()
        .map(|example| example.reliability)
        .collect::<Vec<_>>();
    let mut weights = Vec::with_capacity(coefficient_dim);
    for output in 0..coefficient_dim {
        let target = examples
            .iter()
            .map(|example| example.target_coefficients[output])
            .collect::<Vec<_>>();
        weights.push(weighted_normal_solve(
            &design,
            &target,
            &reliability,
            ridge,
        )?);
    }
    Ok(weights)
}

fn predict_raw(weights: &[Vec<f64>], features: &[f64]) -> BrainResult<Vec<f64>> {
    if weights.is_empty()
        || features.is_empty()
        || weights
            .iter()
            .any(|row| row.len() != features.len() || row.iter().any(|value| !value.is_finite()))
    {
        return Err(BrainError::Invalid(
            "learned_controller_weight_shape".into(),
        ));
    }
    Ok(weights
        .iter()
        .map(|row| row.iter().zip(features).map(|(w, x)| w * x).sum())
        .collect())
}

fn grouped_cv_r2(
    examples: &[ControllerExample],
    coefficient_dim: usize,
    ridge: f64,
) -> BrainResult<f64> {
    let groups = examples
        .iter()
        .map(|example| example.independence_group.clone())
        .collect::<BTreeSet<_>>();
    if groups.len() < 3 {
        return Err(BrainError::Invalid(
            "learned_controller_cv_requires_three_groups".into(),
        ));
    }
    let mut actual = Vec::new();
    let mut predicted = Vec::new();
    for holdout in groups {
        let train = examples
            .iter()
            .filter(|example| example.independence_group != holdout)
            .cloned()
            .collect::<Vec<_>>();
        let test = examples
            .iter()
            .filter(|example| example.independence_group == holdout)
            .cloned()
            .collect::<Vec<_>>();
        if train.len() < 4 || test.is_empty() {
            return Err(BrainError::Invalid(
                "learned_controller_cv_fold_cardinality".into(),
            ));
        }
        let weights = fit_weights(&train, coefficient_dim, ridge)?;
        for example in test {
            let features = feature_vector(&example.state_before, &example.observation)?;
            actual.push(example.target_coefficients);
            predicted.push(predict_raw(&weights, &features)?);
        }
    }
    regression_r2(&actual, &predicted)
}

pub fn train_learned_controller(
    examples: &[ControllerExample],
    ridge: f64,
    ood_margin_fraction: f64,
) -> BrainResult<LearnedController> {
    let (state_dim, observation_dim, coefficient_dim) = validate_examples(examples)?;
    if !ridge.is_finite()
        || ridge <= 0.0
        || !ood_margin_fraction.is_finite()
        || !(0.0..=1.0).contains(&ood_margin_fraction)
    {
        return Err(BrainError::Invalid(
            "learned_controller_training_config".into(),
        ));
    }
    let weights = fit_weights(examples, coefficient_dim, ridge)?;
    let feature_dim = feature_vector(&examples[0].state_before, &examples[0].observation)?.len();
    let mut observation_min = vec![f64::INFINITY; observation_dim];
    let mut observation_max = vec![f64::NEG_INFINITY; observation_dim];
    for example in examples {
        for index in 0..observation_dim {
            observation_min[index] = observation_min[index].min(example.observation[index]);
            observation_max[index] = observation_max[index].max(example.observation[index]);
        }
    }
    let mut squared_error = 0.0;
    let mut values = 0usize;
    for example in examples {
        let features = feature_vector(&example.state_before, &example.observation)?;
        let prediction = predict_raw(&weights, &features)?;
        for (target, predicted) in example.target_coefficients.iter().zip(prediction) {
            squared_error += (target - predicted) * (target - predicted);
            values += 1;
        }
    }
    let training_groups = examples
        .iter()
        .map(|example| example.independence_group.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    Ok(LearnedController {
        schema: "cerebro.tidex.learned_controller/v1".into(),
        state_dim,
        observation_dim,
        coefficient_dim,
        feature_dim,
        weights,
        observation_min,
        observation_max,
        ood_margin_fraction,
        training_rms: (squared_error / values.max(1) as f64).sqrt(),
        grouped_cv_r2: grouped_cv_r2(examples, coefficient_dim, ridge)?,
        training_groups,
    })
}

pub(crate) fn train_runtime_learned_controller(
    fields: &[SkillField],
    examples: &[ControllerExample],
    ridge: f64,
    ood_margin_fraction: f64,
) -> BrainResult<RuntimeLearnedController> {
    let field_ids = canonical_field_ids(fields)?;
    let controller = train_learned_controller(examples, ridge, ood_margin_fraction)?;
    if controller.coefficient_dim != fields.len() {
        return Err(BrainError::Invalid(
            "runtime_learned_controller_coefficient_shape".into(),
        ));
    }
    Ok(RuntimeLearnedController {
        schema: "cerebro.tidex.runtime_learned_controller/v1".into(),
        field_ids,
        controller,
    })
}

fn canonical_field_ids(fields: &[SkillField]) -> BrainResult<Vec<String>> {
    if fields.is_empty() {
        return Err(BrainError::Invalid(
            "runtime_learned_controller_fields_empty".into(),
        ));
    }
    let mut unique = BTreeSet::new();
    fields
        .iter()
        .map(|field| {
            if field.skill_id.trim().is_empty() || !unique.insert(field.skill_id.clone()) {
                return Err(BrainError::Invalid(
                    "runtime_learned_controller_field_identity".into(),
                ));
            }
            Ok(field.skill_id.clone())
        })
        .collect()
}

fn controller_policy_digest(policy: &LearnedControllerPolicy) -> BrainResult<String> {
    Ok(sha256_bytes(&serde_json::to_vec(policy)?))
}

fn field_fingerprint(field_ids: &[String]) -> BrainResult<String> {
    Ok(sha256_bytes(&serde_json::to_vec(field_ids)?))
}

fn validate_persisted_controller_policy(policy: &LearnedControllerPolicy) -> BrainResult<()> {
    if policy.schema != "cerebro.tidex.learned_controller_policy/v1"
        || !policy.ridge.is_finite()
        || policy.ridge <= 0.0
        || !policy.ood_margin_fraction.is_finite()
        || !(0.0..=1.0).contains(&policy.ood_margin_fraction)
        || !policy.minimum_grouped_cv_r2.is_finite()
        || !(0.0..=1.0).contains(&policy.minimum_grouped_cv_r2)
        || !policy.maximum_training_rms.is_finite()
        || policy.maximum_training_rms < 0.0
        || policy.minimum_independent_groups < 3
    {
        return Err(BrainError::Invalid(
            "persisted_learned_controller_policy_invalid".into(),
        ));
    }
    Ok(())
}

fn controller_state_root(root: &Path) -> PathBuf {
    root.join("state/learned_controllers")
}

fn controller_receipt_path(root: &Path, digest: &str) -> PathBuf {
    controller_state_root(root)
        .join("by-sha")
        .join(format!("{digest}.json"))
}

fn controller_pointer_path(root: &Path, session_id: &str) -> PathBuf {
    controller_state_root(root)
        .join("current")
        .join(format!("{session_id}.json"))
}

fn controller_lock_path(root: &Path, session_id: &str) -> PathBuf {
    controller_state_root(root)
        .join("locks")
        .join(format!("{session_id}.lock"))
}

fn controller_dataset_path(root: &Path, digest: &str) -> PathBuf {
    root.join("state/learned_controller_datasets/by-sha")
        .join(format!("{digest}.json"))
}

/// Preflight the complete learned-controller persistence topology without
/// creating or chmodding it. The shared path authority performs every actual
/// resolution; this only enumerates the canonical locations so a hostile
/// sibling cannot be discovered only after a lock is written.
fn validate_controller_state_topology(root: &Path, session_id: &str) -> BrainResult<()> {
    if SessionId::parse(session_id).is_err() {
        return Err(BrainError::Invalid(
            "learned_controller_session_id_invalid".into(),
        ));
    }
    let state = root.join("state");
    let controllers = controller_state_root(root);
    let datasets = root.join("state/learned_controller_datasets");
    for directory in [
        state,
        controllers.clone(),
        controllers.join("by-sha"),
        controllers.join("current"),
        controllers.join("locks"),
        datasets.clone(),
        datasets.join("by-sha"),
    ] {
        let _ = private_directory_if_present(root, &directory)?;
    }
    for path in [
        controller_pointer_path(root, session_id),
        controller_lock_path(root, session_id),
    ] {
        let _ = private_regular_file_if_present(root, &path)?;
    }
    Ok(())
}

struct ControllerLock {
    root: PathBuf,
    path: PathBuf,
}

impl Drop for ControllerLock {
    fn drop(&mut self) {
        if existing_regular_file_under_root(&self.root, &self.path).is_ok() {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn acquire_controller_lock(root: &Path, session_id: &str) -> BrainResult<ControllerLock> {
    validate_controller_state_topology(root, session_id)?;
    let path = controller_lock_path(root, session_id);
    let parent = path
        .parent()
        .ok_or_else(|| BrainError::Integrity("learned_controller_lock_parent_missing".into()))?;
    if private_regular_file_if_present(root, &path)?.is_some() {
        return Err(BrainError::Integrity(
            "learned_controller_session_locked_or_unavailable".into(),
        ));
    }
    ensure_private_dir(root, parent)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)
        .map_err(|error| {
            BrainError::Integrity(format!(
                "learned_controller_session_locked_or_unavailable:{error}"
            ))
        })?;
    file.write_all(session_id.as_bytes())?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    secure_file(&path)?;
    existing_regular_file_under_root(root, &path)?;
    Ok(ControllerLock {
        root: root.to_path_buf(),
        path,
    })
}

fn parse_referenced_json<T: serde::de::DeserializeOwned>(
    root: &Path,
    reference: &EvidenceReference,
) -> BrainResult<T> {
    if !Sha256Digest::is_valid_str(reference.sha256.as_str()) {
        return Err(BrainError::Integrity(
            "learned_controller_evidence_reference_digest_invalid".into(),
        ));
    }
    let path = confined_existing_file(root, &reference.path, Some(&reference.sha256))?;
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn active_bank_under_root(root: &Path, expected_digest: &str) -> BrainResult<SkillBank> {
    if !Sha256Digest::is_valid_str(expected_digest) {
        return Err(BrainError::Integrity(
            "learned_controller_active_bank_digest_invalid".into(),
        ));
    }
    let active_path = confined_existing_file(
        root,
        root.join("state/skill_bank.json"),
        Some(expected_digest),
    )?;
    let bank: SkillBank = serde_json::from_slice(&fs::read(&active_path)?)?;
    let _ = canonical_field_ids(&bank.fields)?;
    let historical_path = confined_existing_file(
        root,
        root.join("state/skill_banks/by-sha")
            .join(format!("{expected_digest}.json")),
        Some(expected_digest),
    )?;
    let historical: SkillBank = serde_json::from_slice(&fs::read(historical_path)?)?;
    if historical != bank {
        return Err(BrainError::Integrity(
            "learned_controller_active_bank_history_content_mismatch".into(),
        ));
    }
    Ok(bank)
}

fn report_field_ids(report: &Value) -> BrainResult<Vec<String>> {
    if report.get("schema").and_then(Value::as_str) != Some("cerebro.tidex.reconstruction/v6")
        || report
            .get("promotion")
            .and_then(|promotion| promotion.get("allowed"))
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(BrainError::Integrity(
            "learned_controller_reconstruction_report_not_promoted".into(),
        ));
    }
    let fields = report
        .get("fields")
        .and_then(Value::as_array)
        .ok_or_else(|| BrainError::Integrity("learned_controller_report_fields_missing".into()))?;
    fields
        .iter()
        .map(|field| {
            field
                .get("skill_id")
                .and_then(Value::as_str)
                .filter(|id| !id.trim().is_empty())
                .map(str::to_string)
                .ok_or_else(|| {
                    BrainError::Integrity("learned_controller_report_field_identity_invalid".into())
                })
        })
        .collect()
}

fn verify_reconstruction_report_binding(
    root: &Path,
    binding: &LearnedControllerBinding,
    field_ids: &[String],
) -> BrainResult<()> {
    let report: Value = parse_referenced_json(root, &binding.reconstruction_report)?;
    if report_field_ids(&report)? != field_ids {
        return Err(BrainError::Integrity(
            "learned_controller_report_active_bank_field_order_mismatch".into(),
        ));
    }
    Ok(())
}

/// The engine-owned finalization receipt is the sole authority that can bridge
/// the immutable raw experiment artifact to a semantic observation digest in
/// the promoted runtime corpus. A controller never accepts that digest from a
/// data set or supervision label.
fn verify_finalization_binding_under_root(
    root: &Path,
    binding: &LearnedControllerBinding,
) -> BrainResult<LearningFinalizationReceipt> {
    let finalization =
        load_verified_learning_finalization_receipt(root, &binding.finalization_receipt)?;
    if finalization.session_id.as_str() != binding.session_id
        || finalization.adaptive_receipt_sha256.as_str() != binding.adaptive_receipt_sha256
        || finalization.report_sha256 != binding.reconstruction_report.sha256
    {
        return Err(BrainError::Integrity(
            "learned_controller_finalization_binding_invalid".into(),
        ));
    }
    Ok(finalization)
}

fn validate_controller_binding_under_root(
    root: &Path,
    binding: &LearnedControllerBinding,
) -> BrainResult<(
    SkillBank,
    LoadedAdaptiveLearningReceipt,
    LearningFinalizationReceipt,
)> {
    if binding.schema != "cerebro.tidex.learned_controller_binding/v1"
        || SessionId::parse(&binding.session_id).is_err()
        || !Sha256Digest::is_valid_str(&binding.target_digest)
        || !Sha256Digest::is_valid_str(&binding.adaptive_receipt_sha256)
        || !Sha256Digest::is_valid_str(binding.finalization_receipt.sha256.as_str())
        || !Sha256Digest::is_valid_str(&binding.active_bank_sha256)
        || !Sha256Digest::is_valid_str(&binding.dataset_sha256)
    {
        return Err(BrainError::Integrity(
            "learned_controller_binding_contract_invalid".into(),
        ));
    }
    let adaptive = load_persistent_adaptive_learning_receipt(root, &binding.session_id)?;
    if adaptive.receipt_sha256 != binding.adaptive_receipt_sha256
        || adaptive.receipt.target_digest != binding.target_digest
        || adaptive.receipt.cycle.target_digest != binding.target_digest
        || adaptive.receipt.cycle.pending_step.is_some()
        || adaptive.receipt.cycle.completed_evidence.len()
            != adaptive.receipt.cycle.target.plan_steps
    {
        return Err(BrainError::Integrity(
            "learned_controller_adaptive_cycle_not_complete_or_bound".into(),
        ));
    }
    let bank = active_bank_under_root(root, &binding.active_bank_sha256)?;
    let field_ids = canonical_field_ids(&bank.fields)?;
    verify_reconstruction_report_binding(root, binding, &field_ids)?;
    let finalization = verify_finalization_binding_under_root(root, binding)?;
    Ok((bank, adaptive, finalization))
}

fn promoted_observation_semantic_sha256(
    root: &Path,
    finalization: &LearningFinalizationReceipt,
    observation_id: &str,
    raw_observation: &EvidenceReference,
) -> BrainResult<Sha256Digest> {
    let observation_id = ObservationId::parse(observation_id).map_err(|_| {
        BrainError::Integrity("learned_controller_training_record_observation_id_invalid".into())
    })?;
    let raw_path = raw_observation.verify(root)?;
    let mut matches = finalization
        .representation_observation_bindings
        .iter()
        .filter(|binding| {
            binding.observation_id == observation_id
                && binding.adaptive_source_observation == *raw_observation
        });
    let mapping = matches.next().ok_or_else(|| {
        BrainError::Integrity("learned_controller_promoted_observation_mapping_missing".into())
    })?;
    if matches.next().is_some() {
        return Err(BrainError::Integrity(
            "learned_controller_promoted_observation_mapping_ambiguous_or_invalid".into(),
        ));
    }
    let mapped_source = mapping.adaptive_source_observation.verify(root)?;
    if raw_path != mapped_source {
        return Err(BrainError::Integrity(
            "learned_controller_promoted_observation_source_path_mismatch".into(),
        ));
    }
    // The engine loader replayed this destination into the current promoted
    // corpus. Re-read its immutable identity here as well so a replacement
    // after that replay cannot leave a controller label bound to stale data.
    mapping
        .representation_destination_observation
        .verify(root)?;
    Ok(mapping.promoted_observation_semantic_sha256.clone())
}

fn validate_supervision(
    root: &Path,
    record: &ControllerTrainingRecord,
    expected_evidence: &crate::learning_orchestrator::LearningExperimentEvidence,
    field_ids: &[String],
    binding: &LearnedControllerBinding,
    finalization: &LearningFinalizationReceipt,
) -> BrainResult<ControllerExample> {
    if record.schema != "cerebro.tidex.learned_controller_training_record/v1"
        || ObservationId::parse(&record.observation_id).is_err()
        || !Sha256Digest::is_valid_str(&record.adaptive_evidence_sha256)
        || record.observation != expected_evidence.observation
        || record.observation_id != expected_evidence.observation_id
    {
        return Err(BrainError::Integrity(
            "learned_controller_training_record_observation_binding_invalid".into(),
        ));
    }
    let observation: DeltaObservation = parse_referenced_json(root, &record.observation)?;
    if observation.observation_id != record.observation_id
        || observation.independence_group != record.independence_group
        || observation.functional_response.is_empty()
        || observation
            .functional_response
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(BrainError::Integrity(
            "learned_controller_training_record_observation_contract_invalid".into(),
        ));
    }
    let supervision: ControllerSupervisionEvidence =
        parse_referenced_json(root, &record.supervision)?;
    if supervision.schema != "cerebro.tidex.learned_controller_supervision/v1"
        || supervision.session_id != binding.session_id
        || supervision.target_digest != binding.target_digest
        || supervision.adaptive_evidence_sha256 != record.adaptive_evidence_sha256
        || supervision.observation_id != record.observation_id
        || supervision.active_bank_sha256 != binding.active_bank_sha256
        || supervision.field_ids != field_ids
        || supervision.state_before != record.state_before
        || supervision.target_coefficients != record.target_coefficients
        || supervision.reliability != record.reliability
        || supervision.independence_group != record.independence_group
    {
        return Err(BrainError::Integrity(
            "learned_controller_supervision_record_mismatch".into(),
        ));
    }
    let promoted_semantic_sha256 = promoted_observation_semantic_sha256(
        root,
        finalization,
        &record.observation_id,
        &record.observation,
    )?;
    validate_governed_composition_supervision(
        root,
        &supervision.governed_composition_receipt,
        &promoted_semantic_sha256,
        &supervision.target_coefficients,
        field_ids,
        binding,
    )?;
    Ok(ControllerExample {
        state_before: record.state_before.clone(),
        observation: observation.functional_response,
        target_coefficients: record.target_coefficients.clone(),
        reliability: record.reliability,
        independence_group: record.independence_group.clone(),
    })
}

fn same_supervision_coefficients(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            (left - right).abs() <= f64::EPSILON.sqrt() * 64.0 * (1.0 + left.abs().max(right.abs()))
        })
}

fn same_controller_float(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= f64::EPSILON.sqrt() * 64.0 * (1.0 + left.abs().max(right.abs()))
}

fn same_controller_vector(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| same_controller_float(*left, *right))
}

fn same_runtime_controller(
    left: &RuntimeLearnedController,
    right: &RuntimeLearnedController,
) -> bool {
    left.schema == right.schema
        && left.field_ids == right.field_ids
        && left.controller.schema == right.controller.schema
        && left.controller.state_dim == right.controller.state_dim
        && left.controller.observation_dim == right.controller.observation_dim
        && left.controller.coefficient_dim == right.controller.coefficient_dim
        && left.controller.feature_dim == right.controller.feature_dim
        && left.controller.weights.len() == right.controller.weights.len()
        && left
            .controller
            .weights
            .iter()
            .zip(&right.controller.weights)
            .all(|(left, right)| same_controller_vector(left, right))
        && same_controller_vector(
            &left.controller.observation_min,
            &right.controller.observation_min,
        )
        && same_controller_vector(
            &left.controller.observation_max,
            &right.controller.observation_max,
        )
        && same_controller_float(
            left.controller.ood_margin_fraction,
            right.controller.ood_margin_fraction,
        )
        && same_controller_float(left.controller.training_rms, right.controller.training_rms)
        && same_controller_float(
            left.controller.grouped_cv_r2,
            right.controller.grouped_cv_r2,
        )
        && left.controller.training_groups == right.controller.training_groups
}

fn validate_governed_composition_supervision(
    root: &Path,
    reference: &EvidenceReference,
    promoted_observation_semantic_sha256: &Sha256Digest,
    target_coefficients: &[f64],
    field_ids: &[String],
    binding: &LearnedControllerBinding,
) -> BrainResult<()> {
    let receipt =
        load_verified_governed_composition_receipt(root, &reference.path, &reference.sha256)?;
    if receipt.report_sha256 != binding.reconstruction_report.sha256
        || receipt.active_bank_sha256 != binding.active_bank_sha256
        || receipt.source_observation_sha256 != promoted_observation_semantic_sha256.as_str()
        || receipt.field_ids != field_ids
        || !same_supervision_coefficients(&receipt.accepted_coefficients, target_coefficients)
    {
        return Err(BrainError::Integrity(
            "learned_controller_governed_composition_supervision_invalid".into(),
        ));
    }
    Ok(())
}

fn validate_training_dataset_under_root(
    root: &Path,
    dataset_reference: &EvidenceReference,
    binding: &LearnedControllerBinding,
    field_ids: &[String],
    adaptive: &LoadedAdaptiveLearningReceipt,
    finalization: &LearningFinalizationReceipt,
) -> BrainResult<(ControllerTrainingDataset, Vec<ControllerExample>)> {
    if dataset_reference.sha256 != binding.dataset_sha256 {
        return Err(BrainError::Integrity(
            "learned_controller_dataset_binding_digest_mismatch".into(),
        ));
    }
    let dataset: ControllerTrainingDataset = parse_referenced_json(root, dataset_reference)?;
    if dataset.schema != "cerebro.tidex.learned_controller_training_dataset/v1"
        || dataset.session_id != binding.session_id
        || dataset.target_digest != binding.target_digest
        || dataset.records.len() < 6
    {
        return Err(BrainError::Integrity(
            "learned_controller_dataset_contract_invalid".into(),
        ));
    }
    let mut completed = BTreeMap::new();
    for (digest, evidence) in adaptive
        .receipt
        .cycle
        .completed_evidence_sha256
        .iter()
        .zip(&adaptive.receipt.cycle.completed_evidence)
    {
        completed.insert(digest.as_str(), evidence);
    }
    let mut adaptive_evidence_digests = BTreeSet::new();
    let mut supervision_digests = BTreeSet::new();
    let mut examples = Vec::with_capacity(dataset.records.len());
    for record in &dataset.records {
        let expected = completed
            .get(record.adaptive_evidence_sha256.as_str())
            .ok_or_else(|| {
                BrainError::Integrity(
                    "learned_controller_training_record_unknown_adaptive_evidence".into(),
                )
            })?;
        if !adaptive_evidence_digests.insert(record.adaptive_evidence_sha256.clone())
            || !supervision_digests.insert(record.supervision.sha256.clone())
        {
            return Err(BrainError::Integrity(
                "learned_controller_training_evidence_or_supervision_reused".into(),
            ));
        }
        examples.push(validate_supervision(
            root,
            record,
            expected,
            field_ids,
            binding,
            finalization,
        )?);
    }
    let (_, _, coefficient_dim) = validate_examples(&examples)?;
    if coefficient_dim != field_ids.len() {
        return Err(BrainError::Integrity(
            "learned_controller_dataset_coefficient_field_mismatch".into(),
        ));
    }
    Ok((dataset, examples))
}

fn validate_controller_quality(
    runtime: &RuntimeLearnedController,
    policy: &LearnedControllerPolicy,
    field_ids: &[String],
) -> BrainResult<()> {
    if runtime.schema != "cerebro.tidex.runtime_learned_controller/v1"
        || runtime.field_ids != field_ids
        || runtime.controller.schema != "cerebro.tidex.learned_controller/v1"
        || runtime.controller.coefficient_dim != field_ids.len()
        || runtime.controller.training_groups < policy.minimum_independent_groups
        || !runtime.controller.grouped_cv_r2.is_finite()
        || runtime.controller.grouped_cv_r2 < policy.minimum_grouped_cv_r2
        || runtime.controller.grouped_cv_r2 > 1.0 + 1e-9
        || !runtime.controller.training_rms.is_finite()
        || runtime.controller.training_rms > policy.maximum_training_rms
    {
        return Err(BrainError::Integrity(
            "learned_controller_quality_gate_failed".into(),
        ));
    }
    Ok(())
}

fn persist_controller_dataset(
    root: &Path,
    digest: &str,
    raw: &[u8],
) -> BrainResult<EvidenceReference> {
    if !Sha256Digest::is_valid_str(digest) || sha256_bytes(raw) != digest {
        return Err(BrainError::Integrity(
            "learned_controller_dataset_content_digest_invalid".into(),
        ));
    }
    let path = controller_dataset_path(root, digest);
    if private_regular_file_if_present(root, &path)?.is_some() {
        let _ = confined_existing_file(root, &path, Some(digest)).map_err(|_| {
            BrainError::Integrity("learned_controller_dataset_artifact_collision".into())
        })?;
    } else {
        write_new_private(root, &path, raw)?;
    }
    Ok(EvidenceReference {
        path,
        sha256: Sha256Digest::parse(digest)?,
    })
}

fn load_controller_receipt_by_sha(
    root: &Path,
    digest: &str,
) -> BrainResult<LearnedControllerReceipt> {
    if !Sha256Digest::is_valid_str(digest) {
        return Err(BrainError::Integrity(
            "learned_controller_receipt_digest_invalid".into(),
        ));
    }
    let path = controller_receipt_path(root, digest);
    let path = confined_existing_file(root, &path, Some(digest))?;
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn verify_controller_ledger_binding(
    root: &Path,
    digest: &str,
    receipt: &LearnedControllerReceipt,
) -> BrainResult<()> {
    let event = ledger::find_v2_event_by_payload_string(
        root,
        "learned_controller_receipt",
        "receipt_sha256",
        digest,
    )?
    .ok_or_else(|| BrainError::Integrity("learned_controller_receipt_ledger_missing".into()))?;
    let payload = event.payload()?;
    if payload.get("session_id").and_then(Value::as_str)
        != Some(receipt.binding.session_id.as_str())
        || payload.get("target_digest").and_then(Value::as_str)
            != Some(receipt.binding.target_digest.as_str())
        || payload
            .get("finalization_receipt_sha256")
            .and_then(Value::as_str)
            != Some(receipt.binding.finalization_receipt.sha256.as_str())
        || payload.get("active_bank_sha256").and_then(Value::as_str)
            != Some(receipt.binding.active_bank_sha256.as_str())
        || payload.get("dataset_sha256").and_then(Value::as_str)
            != Some(receipt.binding.dataset_sha256.as_str())
    {
        return Err(BrainError::Integrity(
            "learned_controller_receipt_ledger_payload_mismatch".into(),
        ));
    }
    Ok(())
}

fn validate_controller_receipt_under_root(
    root: &Path,
    digest: &str,
) -> BrainResult<LearnedControllerReceipt> {
    let receipt = load_controller_receipt_by_sha(root, digest)?;
    if receipt.schema != "cerebro.tidex.learned_controller_receipt/v1"
        || receipt.event_kind != LearnedControllerEventKind::ControllerTrained
        || receipt.policy_digest != controller_policy_digest(&receipt.policy)?
        || receipt.field_fingerprint != field_fingerprint(&receipt.field_ids)?
        || receipt.runtime_controller.field_ids != receipt.field_ids
        || receipt.dataset.sha256 != receipt.binding.dataset_sha256
        || SessionId::parse(&receipt.binding.session_id).is_err()
        || receipt
            .prior_receipt_sha256
            .as_deref()
            .is_some_and(|prior| !Sha256Digest::is_valid_str(prior))
    {
        return Err(BrainError::Integrity(
            "learned_controller_receipt_contract_invalid".into(),
        ));
    }
    verify_controller_ledger_binding(root, digest, &receipt)?;
    let (bank, adaptive, finalization) =
        validate_controller_binding_under_root(root, &receipt.binding)?;
    let field_ids = canonical_field_ids(&bank.fields)?;
    if field_ids != receipt.field_ids {
        return Err(BrainError::Integrity(
            "learned_controller_receipt_active_field_order_mismatch".into(),
        ));
    }
    let (_, examples) = validate_training_dataset_under_root(
        root,
        &receipt.dataset,
        &receipt.binding,
        &field_ids,
        &adaptive,
        &finalization,
    )?;
    let expected = train_runtime_learned_controller(
        &bank.fields,
        &examples,
        receipt.policy.ridge,
        receipt.policy.ood_margin_fraction,
    )?;
    validate_controller_quality(&expected, &receipt.policy, &field_ids)?;
    if !same_runtime_controller(&expected, &receipt.runtime_controller) {
        return Err(BrainError::Integrity(
            "learned_controller_receipt_model_replay_mismatch".into(),
        ));
    }
    if let Some(prior_digest) = &receipt.prior_receipt_sha256 {
        let prior = load_controller_receipt_by_sha(root, prior_digest)?;
        verify_controller_ledger_binding(root, prior_digest, &prior)?;
        if prior.binding.session_id != receipt.binding.session_id
            || prior.binding.target_digest != receipt.binding.target_digest
            || prior.binding.adaptive_receipt_sha256 != receipt.binding.adaptive_receipt_sha256
            || prior.binding.finalization_receipt != receipt.binding.finalization_receipt
            || prior.binding.reconstruction_report != receipt.binding.reconstruction_report
            || prior.binding.active_bank_sha256 != receipt.binding.active_bank_sha256
            || receipt.generation != prior.generation.saturating_add(1)
        {
            return Err(BrainError::Integrity(
                "learned_controller_receipt_lineage_invalid".into(),
            ));
        }
    } else if receipt.generation != 0 {
        return Err(BrainError::Integrity(
            "learned_controller_receipt_initial_generation_invalid".into(),
        ));
    }
    Ok(receipt)
}

fn load_current_controller_receipt_under_root(
    root: &Path,
    session_id: &str,
) -> BrainResult<LoadedLearnedControllerReceipt> {
    if SessionId::parse(session_id).is_err() {
        return Err(BrainError::Invalid(
            "learned_controller_session_id_invalid".into(),
        ));
    }
    validate_controller_state_topology(root, session_id)?;
    let path = confined_existing_file(root, controller_pointer_path(root, session_id), None)
        .map_err(|_| {
            BrainError::Integrity("learned_controller_current_pointer_missing_or_invalid".into())
        })?;
    let pointer: LearnedControllerPointer = serde_json::from_slice(&fs::read(path)?)?;
    if pointer.schema != "cerebro.tidex.learned_controller_pointer/v1"
        || pointer.session_id != session_id
        || !Sha256Digest::is_valid_str(&pointer.receipt_sha256)
    {
        return Err(BrainError::Integrity(
            "learned_controller_current_pointer_contract_invalid".into(),
        ));
    }
    let receipt = validate_controller_receipt_under_root(root, &pointer.receipt_sha256)?;
    Ok(LoadedLearnedControllerReceipt {
        receipt_sha256: pointer.receipt_sha256,
        receipt,
    })
}

fn persist_controller_receipt_under_root(
    root: &Path,
    receipt: LearnedControllerReceipt,
) -> BrainResult<LoadedLearnedControllerReceipt> {
    let raw = serde_json::to_vec_pretty(&receipt)?;
    let digest = sha256_bytes(&raw);
    let receipt_path = controller_receipt_path(root, &digest);
    if private_regular_file_if_present(root, &receipt_path)?.is_some() {
        return Err(BrainError::Integrity(
            "learned_controller_receipt_digest_already_exists".into(),
        ));
    }
    write_new_private(root, &receipt_path, &raw)?;
    let event = ledger::append(
        root,
        "learned_controller_receipt",
        json!({
            "schema":"cerebro.tidex.learned_controller_ledger_binding/v1",
            "receipt_sha256":digest,
            "session_id":receipt.binding.session_id,
            "target_digest":receipt.binding.target_digest,
            "adaptive_receipt_sha256":receipt.binding.adaptive_receipt_sha256,
            "finalization_receipt_sha256":receipt.binding.finalization_receipt.sha256,
            "reconstruction_report_sha256":receipt.binding.reconstruction_report.sha256,
            "active_bank_sha256":receipt.binding.active_bank_sha256,
            "dataset_sha256":receipt.binding.dataset_sha256,
            "generation":receipt.generation,
        }),
    )?;
    let payload = event.payload()?;
    if payload.get("receipt_sha256").and_then(Value::as_str) != Some(digest.as_str()) {
        return Err(BrainError::Integrity(
            "learned_controller_ledger_receipt_digest_mismatch".into(),
        ));
    }
    let pointer = LearnedControllerPointer {
        schema: "cerebro.tidex.learned_controller_pointer/v1".into(),
        session_id: receipt.binding.session_id.clone(),
        receipt_sha256: digest.clone(),
    };
    let mut pointer_raw = serde_json::to_vec_pretty(&pointer)?;
    pointer_raw.push(b'\n');
    write_private_atomic(
        root,
        &controller_pointer_path(root, &receipt.binding.session_id),
        &pointer_raw,
    )?;
    let loaded = load_current_controller_receipt_under_root(root, &receipt.binding.session_id)?;
    // The reloader authenticates the exact receipt bytes by digest, verifies
    // the ledger binding and deterministically replays the controller. The
    // digest is the canonical post-write identity; a second `PartialEq` over
    // deserialized float fields is neither stronger nor stable.
    if loaded.receipt_sha256 != digest {
        return Err(BrainError::Integrity(
            "learned_controller_receipt_postwrite_revalidation_failed".into(),
        ));
    }
    Ok(loaded)
}

fn train_persisted_runtime_learned_controller_under_root(
    root: &Path,
    dataset_path: &Path,
    policy: &LearnedControllerPolicy,
    binding: &LearnedControllerBinding,
) -> BrainResult<LoadedLearnedControllerReceipt> {
    validate_persisted_controller_policy(policy)?;
    validate_controller_state_topology(root, &binding.session_id)?;
    // Authenticate the caller-provided immutable data set before creating a
    // lock or any controller-state directory.
    let _ = confined_existing_file(root, dataset_path, Some(&binding.dataset_sha256))?;
    let _lock = acquire_controller_lock(root, &binding.session_id)?;
    let (bank, adaptive, finalization) = validate_controller_binding_under_root(root, binding)?;
    let field_ids = canonical_field_ids(&bank.fields)?;
    let source_path = confined_existing_file(root, dataset_path, Some(&binding.dataset_sha256))?;
    let raw = fs::read(&source_path)?;
    let source_reference = EvidenceReference {
        path: source_path,
        sha256: Sha256Digest::parse(&binding.dataset_sha256)?,
    };
    let (_, examples) = validate_training_dataset_under_root(
        root,
        &source_reference,
        binding,
        &field_ids,
        &adaptive,
        &finalization,
    )?;
    let dataset = persist_controller_dataset(root, &binding.dataset_sha256, &raw)?;
    let runtime = train_runtime_learned_controller(
        &bank.fields,
        &examples,
        policy.ridge,
        policy.ood_margin_fraction,
    )?;
    validate_controller_quality(&runtime, policy, &field_ids)?;
    let pointer = controller_pointer_path(root, &binding.session_id);
    let previous = if private_regular_file_if_present(root, &pointer)?.is_some() {
        Some(load_current_controller_receipt_under_root(
            root,
            &binding.session_id,
        )?)
    } else {
        None
    };
    let receipt = LearnedControllerReceipt {
        schema: "cerebro.tidex.learned_controller_receipt/v1".into(),
        event_kind: LearnedControllerEventKind::ControllerTrained,
        generation: previous
            .as_ref()
            .map(|previous| previous.receipt.generation.saturating_add(1))
            .unwrap_or(0),
        prior_receipt_sha256: previous.map(|previous| previous.receipt_sha256),
        binding: binding.clone(),
        field_ids: field_ids.clone(),
        field_fingerprint: field_fingerprint(&field_ids)?,
        dataset,
        policy: policy.clone(),
        policy_digest: controller_policy_digest(policy)?,
        runtime_controller: runtime,
    };
    persist_controller_receipt_under_root(root, receipt)
}

/// Train the canonical runtime controller only from a hash-bound data set,
/// completed adaptive cycle, promoted report and exact active SkillBank.
pub fn train_persisted_runtime_learned_controller(
    root: impl AsRef<Path>,
    dataset_path: impl AsRef<Path>,
    policy: &LearnedControllerPolicy,
    binding: &LearnedControllerBinding,
) -> BrainResult<LoadedLearnedControllerReceipt> {
    let root = verify_private_root(root.as_ref())?;
    train_persisted_runtime_learned_controller_under_root(
        &root,
        dataset_path.as_ref(),
        policy,
        binding,
    )
}

/// Load and deterministically replay the current persisted controller receipt.
/// Any changed bank, report, adaptive receipt, data set or supervision evidence
/// rejects before the controller reaches runtime composition.
pub fn load_persisted_runtime_learned_controller(
    root: impl AsRef<Path>,
    session_id: &str,
) -> BrainResult<LoadedLearnedControllerReceipt> {
    let root = verify_private_root(root.as_ref())?;
    load_current_controller_receipt_under_root(&root, session_id)
}

pub fn load_verified_persisted_runtime_learned_controller(
    root: impl AsRef<Path>,
    session_id: &str,
) -> BrainResult<RuntimeLearnedController> {
    Ok(load_persisted_runtime_learned_controller(root, session_id)?
        .receipt
        .runtime_controller)
}

impl LearnedController {
    pub fn decide(&self, state: &[f64], observation: &[f64]) -> BrainResult<ControllerDecision> {
        if state.len() != self.state_dim
            || observation.len() != self.observation_dim
            || state
                .iter()
                .chain(observation)
                .any(|value| !value.is_finite())
        {
            return Err(BrainError::Invalid(
                "learned_controller_runtime_shape".into(),
            ));
        }
        let mut margin_used = Vec::with_capacity(self.observation_dim);
        for index in 0..self.observation_dim {
            let span = (self.observation_max[index] - self.observation_min[index])
                .abs()
                .max(1e-9);
            let margin = span * self.ood_margin_fraction;
            let low = self.observation_min[index] - margin;
            let high = self.observation_max[index] + margin;
            if observation[index] < low || observation[index] > high {
                return Err(BrainError::Invalid(format!(
                    "learned_controller_observation_ood:{index}"
                )));
            }
            margin_used.push(margin);
        }
        let features = feature_vector(state, observation)?;
        if features.len() != self.feature_dim {
            return Err(BrainError::Integrity(
                "learned_controller_feature_contract_changed".into(),
            ));
        }
        Ok(ControllerDecision {
            coefficients: predict_raw(&self.weights, &features)?,
            ood_margin_used: margin_used,
        })
    }
}

pub fn execute_learned_program(
    fields: &[SkillField],
    controller: &LearnedController,
    initial_state: &[f64],
    observations: &[Vec<f64>],
) -> BrainResult<LearnedProgramExecution> {
    if fields.len() != controller.coefficient_dim || initial_state.len() != controller.state_dim {
        return Err(BrainError::Invalid(
            "learned_controller_execution_contract".into(),
        ));
    }
    let mut state = initial_state.to_vec();
    let mut steps = Vec::with_capacity(observations.len());
    for observation in observations {
        let decision = controller.decide(&state, observation)?;
        let operator = compose_skill_fields(fields, &decision.coefficients)?;
        let before = state.clone();
        state = apply_parametric_transition(&state, &operator, controller.state_dim)?;
        steps.push(LearnedProgramStep {
            observation: observation.clone(),
            state_before: before,
            field_coefficients: decision.coefficients,
            composed_operator: operator,
            state_after: state.clone(),
        });
    }
    let final_state_index = state
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .ok_or_else(|| BrainError::Invalid("learned_controller_empty_final_state".into()))?;
    Ok(LearnedProgramExecution {
        final_state_index,
        final_state: state,
        steps,
    })
}

/// Execute a learned recurrent controller when SkillFields synthesize direct
/// next-state logits rather than a full transition matrix. This stricter
/// actuator prevents an observation-only controller from hiding an entire
/// state-conditional policy inside the columns of one matrix.
pub fn execute_learned_winner_take_all(
    fields: &[SkillField],
    controller: &LearnedController,
    initial_state: &[f64],
    observations: &[Vec<f64>],
) -> BrainResult<LearnedProgramExecution> {
    if fields.len() != controller.coefficient_dim
        || initial_state.len() != controller.state_dim
        || fields
            .iter()
            .any(|field| field.direction.len() != controller.state_dim)
    {
        return Err(BrainError::Invalid(
            "learned_controller_wta_execution_contract".into(),
        ));
    }
    let mut state = initial_state.to_vec();
    let mut steps = Vec::with_capacity(observations.len());
    for observation in observations {
        let decision = controller.decide(&state, observation)?;
        let logits = compose_skill_fields(fields, &decision.coefficients)?;
        let winner = logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .ok_or_else(|| BrainError::Invalid("learned_controller_wta_empty_logits".into()))?;
        let before = state.clone();
        state = vec![0.0; controller.state_dim];
        state[winner] = 1.0;
        steps.push(LearnedProgramStep {
            observation: observation.clone(),
            state_before: before,
            field_coefficients: decision.coefficients,
            composed_operator: logits,
            state_after: state.clone(),
        });
    }
    let final_state_index = state
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .ok_or_else(|| BrainError::Invalid("learned_controller_wta_empty_final_state".into()))?;
    Ok(LearnedProgramExecution {
        final_state_index,
        final_state: state,
        steps,
    })
}

pub fn controller_group_summary(examples: &[ControllerExample]) -> BTreeMap<String, usize> {
    let mut groups = BTreeMap::new();
    for example in examples {
        *groups
            .entry(example.independence_group.clone())
            .or_default() += 1;
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authority::PrivateFileReference;
    use crate::identity::SessionId;
    use crate::learning_finalization::RepresentationObservationBinding;
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cerebro-controller-finalization-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        root
    }

    fn finalization_with_mapping(
        source_path: &Path,
        source_sha256: &str,
        promoted_sha256: &str,
    ) -> LearningFinalizationReceipt {
        let source_digest = Sha256Digest::parse(source_sha256).unwrap();
        LearningFinalizationReceipt {
            schema: "cerebro.tidex.learning_finalization_receipt/v1".into(),
            operation_key: Sha256Digest::parse("a".repeat(64)).unwrap(),
            session_id: SessionId::parse("session").unwrap(),
            adaptive_receipt_sha256: Sha256Digest::parse("b".repeat(64)).unwrap(),
            learning_finalization_input_sha256: Sha256Digest::parse("c".repeat(64)).unwrap(),
            representation_evidence_receipt: PrivateFileReference::new(
                source_path,
                source_digest.clone(),
            ),
            representation_protocol_sha256: Sha256Digest::parse("d".repeat(64)).unwrap(),
            representation_observation_bindings_sha256: Sha256Digest::parse("e".repeat(64))
                .unwrap(),
            representation_observation_bindings: vec![RepresentationObservationBinding {
                observation_id: ObservationId::parse("observation-a").unwrap(),
                adaptive_source_observation: PrivateFileReference::new(
                    source_path,
                    source_digest.clone(),
                ),
                representation_destination_observation: PrivateFileReference::new(
                    source_path,
                    source_digest,
                ),
                promoted_observation_semantic_sha256: Sha256Digest::parse(promoted_sha256).unwrap(),
            }],
            prior_corpus_digest: Sha256Digest::parse("0".repeat(64)).unwrap(),
            prior_observation_count: 1,
            new_corpus_digest: Sha256Digest::parse("1".repeat(64)).unwrap(),
            new_observation_count: 1,
            archived_artifact_sha256: BTreeMap::new(),
            report_sha256: Sha256Digest::parse("2".repeat(64)).unwrap(),
            commit_operation_key: Sha256Digest::parse("3".repeat(64)).unwrap(),
            commit_receipt_sha256: Sha256Digest::parse("4".repeat(64)).unwrap(),
            ledger_event_hash: Sha256Digest::parse("5".repeat(64)).unwrap(),
        }
    }

    #[test]
    fn controller_learns_state_observation_interaction_and_rejects_ood() {
        let examples = (0..4)
            .flat_map(|group| {
                (0..2).flat_map(move |state_index| {
                    [-1.0, 1.0]
                        .into_iter()
                        .map(move |signal| ControllerExample {
                            state_before: if state_index == 0 {
                                vec![1.0, 0.0]
                            } else {
                                vec![0.0, 1.0]
                            },
                            observation: vec![signal, group as f64 * 0.01],
                            target_coefficients: vec![
                                if state_index == 0 { signal } else { -signal },
                                if state_index == 0 { 1.0 } else { -1.0 },
                            ],
                            reliability: 1.0,
                            independence_group: format!("g{group}"),
                        })
                })
            })
            .collect::<Vec<_>>();
        let controller = train_learned_controller(&examples, 1e-8, 0.25).unwrap();
        assert!(
            controller.grouped_cv_r2 > 0.99,
            "{}",
            controller.grouped_cv_r2
        );
        let a = controller.decide(&[1.0, 0.0], &[1.0, 0.015]).unwrap();
        let b = controller.decide(&[0.0, 1.0], &[1.0, 0.015]).unwrap();
        assert!(a.coefficients[0] > 0.9);
        assert!(b.coefficients[0] < -0.9);
        assert!(controller.decide(&[1.0, 0.0], &[3.0, 0.0]).is_err());
    }
    #[test]
    fn runtime_controller_binds_coefficients_to_skillfield_identity() {
        let fields = vec![
            SkillField {
                skill_id: "a".into(),
                reconstruction_id: "ra".into(),
                lineage_id: "la".into(),
                generation_created: 1,
                direction: vec![1.0, 0.0],
                structured_geometry: None,
                dense_materialization: None,
                parameter_layout_sha256: None,
                representation_signature: vec![],
                singular_value: 1.0,
                explained_variance: 1.0,
                persistence: 1.0,
                coherence: 1.0,
                uncertainty: 0.0,
                evidence_support_digests: vec![],
                support: 1,
                functional_signature: vec![1.0],
                parent_skill_ids: vec![],
            },
            SkillField {
                skill_id: "b".into(),
                reconstruction_id: "rb".into(),
                lineage_id: "lb".into(),
                generation_created: 1,
                direction: vec![0.0, 1.0],
                structured_geometry: None,
                dense_materialization: None,
                parameter_layout_sha256: None,
                representation_signature: vec![],
                singular_value: 1.0,
                explained_variance: 1.0,
                persistence: 1.0,
                coherence: 1.0,
                uncertainty: 0.0,
                evidence_support_digests: vec![],
                support: 1,
                functional_signature: vec![0.0],
                parent_skill_ids: vec![],
            },
        ];
        let examples = (0..4)
            .flat_map(|group| {
                [0.0, 1.0].into_iter().map(move |signal| ControllerExample {
                    state_before: vec![1.0, 0.0],
                    observation: vec![signal],
                    target_coefficients: vec![1.0 - signal, signal],
                    reliability: 1.0,
                    independence_group: format!("g{group}"),
                })
            })
            .collect::<Vec<_>>();
        let runtime = train_runtime_learned_controller(&fields, &examples, 1e-8, 0.25).unwrap();
        assert_eq!(runtime.field_ids, vec!["a", "b"]);
        assert_eq!(runtime.controller.coefficient_dim, 2);
        assert!(runtime.controller.grouped_cv_r2 > 0.99);
    }

    #[test]
    fn persisted_controller_policy_and_quality_gates_fail_closed() {
        let invalid_policy = LearnedControllerPolicy {
            schema: "cerebro.tidex.learned_controller_policy/v1".into(),
            ridge: 1e-8,
            ood_margin_fraction: 0.25,
            minimum_grouped_cv_r2: 0.95,
            maximum_training_rms: 0.01,
            minimum_independent_groups: 2,
        };
        assert!(validate_persisted_controller_policy(&invalid_policy).is_err());

        let policy = LearnedControllerPolicy {
            minimum_independent_groups: 3,
            ..invalid_policy
        };
        validate_persisted_controller_policy(&policy).unwrap();
        let runtime = RuntimeLearnedController {
            schema: "cerebro.tidex.runtime_learned_controller/v1".into(),
            field_ids: vec!["field-a".into()],
            controller: LearnedController {
                schema: "cerebro.tidex.learned_controller/v1".into(),
                state_dim: 1,
                observation_dim: 1,
                coefficient_dim: 1,
                feature_dim: 4,
                weights: vec![vec![0.0; 4]],
                observation_min: vec![0.0],
                observation_max: vec![1.0],
                ood_margin_fraction: 0.25,
                training_rms: 0.02,
                grouped_cv_r2: 0.94,
                training_groups: 2,
            },
        };
        assert!(validate_controller_quality(&runtime, &policy, &runtime.field_ids).is_err());
    }

    #[test]
    fn supervision_contract_has_no_free_label_fallback() {
        let raw = serde_json::json!({
            "schema":"cerebro.tidex.learned_controller_supervision/v1",
            "session_id":"session",
            "target_digest":"a".repeat(64),
            "adaptive_evidence_sha256":"b".repeat(64),
            "observation_id":"observation",
            "active_bank_sha256":"c".repeat(64),
            "field_ids":["field-a"],
            "state_before":[1.0],
            "target_coefficients":[0.5],
            "reliability":1.0,
            "independence_group":"group-a"
        });
        assert!(serde_json::from_value::<ControllerSupervisionEvidence>(raw).is_err());
    }

    #[test]
    fn controller_binding_requires_authenticated_finalization_reference() {
        let raw = serde_json::json!({
            "schema":"cerebro.tidex.learned_controller_binding/v1",
            "session_id":"session",
            "target_digest":"a".repeat(64),
            "adaptive_receipt_sha256":"b".repeat(64),
            "reconstruction_report":{"path":"/home/yo/cerebro/state/reports/report.json","sha256":"c".repeat(64)},
            "active_bank_sha256":"d".repeat(64),
            "dataset_sha256":"e".repeat(64)
        });
        assert!(serde_json::from_value::<LearnedControllerBinding>(raw).is_err());
    }

    #[test]
    fn raw_adaptive_evidence_must_map_to_the_promoted_semantic_observation() {
        let root = temporary_root("mapping");
        let state = root.join("state");
        fs::create_dir(&state).unwrap();
        let source_path = state.join("adaptive-observation.json");
        let source_bytes = b"sealed-adaptive-observation";
        fs::write(&source_path, source_bytes).unwrap();
        let source_sha256 = sha256_bytes(source_bytes);
        let raw = EvidenceReference {
            path: source_path.clone(),
            sha256: Sha256Digest::parse(&source_sha256).unwrap(),
        };
        let mut finalization =
            finalization_with_mapping(&source_path, &source_sha256, &"f".repeat(64));
        assert_eq!(
            promoted_observation_semantic_sha256(&root, &finalization, "observation-a", &raw,)
                .unwrap(),
            Sha256Digest::parse("f".repeat(64)).unwrap()
        );
        finalization.representation_observation_bindings[0]
            .adaptive_source_observation
            .sha256 = Sha256Digest::zero();
        assert!(
            promoted_observation_semantic_sha256(&root, &finalization, "observation-a", &raw,)
                .is_err()
        );
        let mut destination_tampered =
            finalization_with_mapping(&source_path, &source_sha256, &"f".repeat(64));
        destination_tampered.representation_observation_bindings[0]
            .representation_destination_observation
            .sha256 = Sha256Digest::zero();
        assert!(promoted_observation_semantic_sha256(
            &root,
            &destination_tampered,
            "observation-a",
            &raw,
        )
        .is_err());
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn controller_lock_rejects_symlinked_controller_root_without_mutation() {
        let root = temporary_root("symlink-controller-root");
        let outside = temporary_root("symlink-controller-root-outside");
        fs::create_dir(root.join("state")).unwrap();
        symlink(&outside, root.join("state/learned_controllers")).unwrap();

        assert!(acquire_controller_lock(&root, "session-symlink-controller").is_err());
        assert!(fs::read_dir(&outside).unwrap().next().is_none());

        fs::remove_dir_all(&root).unwrap();
        fs::remove_dir_all(&outside).unwrap();
    }
}
