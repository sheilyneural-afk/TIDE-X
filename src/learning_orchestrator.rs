use crate::active::{
    assimilate_aperture_result, choose_active_aperture, plan_active_apertures, ActiveAperturePlan,
    GaussianAperturePosterior,
};
pub use crate::authority::PrivateFileReference as EvidenceReference;
use crate::authority::{
    create_private_immutable, existing_directory_if_present, existing_regular_file_if_present,
    existing_regular_file_under_root, read_existing_private_file_bounded,
    replace_private_file_atomic, with_private_authority_lock,
};
use crate::contracts::{ApertureCandidate, DeltaObservation};
use crate::digest::{
    AdaptiveLearningPolicyDigest, AdaptiveLearningReceiptDigest, LearningEvidenceDigest,
    LearningTargetDigest, Sha256Digest,
};
use crate::error::{BrainError, BrainResult};
use crate::identity::{ApertureId, CapabilityId, LearningTargetId, ObservationId, SessionId};
use crate::ledger;
use crate::linalg::{dot, norm, Matrix};
use crate::security::verify_private_root;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use std::fs;
use std::path::{Path, PathBuf};

const MAX_LEARNING_AUTHORITY_JSON_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearningTarget {
    pub target_id: LearningTargetId,
    pub capability_ids: Vec<CapabilityId>,
    #[serde(default = "default_candidate_budget")]
    pub candidate_budget: usize,
    #[serde(default = "default_plan_steps")]
    pub plan_steps: usize,
    #[serde(default = "default_noise_variance")]
    pub noise_variance: f64,
    #[serde(default = "default_cost_weight")]
    pub cost_weight: f64,
    #[serde(default = "default_risk_weight")]
    pub risk_weight: f64,
}

fn default_candidate_budget() -> usize {
    256
}
fn default_plan_steps() -> usize {
    12
}
fn default_noise_variance() -> f64 {
    0.08
}
fn default_cost_weight() -> f64 {
    0.02
}
fn default_risk_weight() -> f64 {
    0.02
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LearningAperture {
    pub aperture_id: ApertureId,
    pub capability_weights: Vec<f64>,
    pub active_capability_ids: Vec<CapabilityId>,
    pub information_gain: f64,
    pub objective: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AutonomousLearningPlan {
    pub schema: String,
    pub target_id: LearningTargetId,
    pub target_digest: LearningTargetDigest,
    pub capability_ids: Vec<CapabilityId>,
    pub candidate_count: usize,
    pub design_rank: usize,
    pub minimum_capability_coverage: usize,
    pub maximum_capability_coverage: usize,
    pub plan: ActiveAperturePlan,
    pub apertures: Vec<LearningAperture>,
}

/// Explicit policy for a real sequential learning cycle.  The policy is kept
/// separate from `LearningTarget` so an offline prospective design remains a
/// pure information-design artifact, while a live session must declare how a
/// realized outcome affects its next experiment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveLearningPolicy {
    pub schema: String,
    /// Positive multiplier for the posterior-predicted outcome in the live
    /// aperture objective. It is mandatory: a zero-weight policy is not a
    /// live outcome-conditioned learning policy.
    pub outcome_utility_weight: f64,
    /// `true` learns toward larger observed values; `false` learns toward
    /// smaller observed values. This must be explicit because outcome signs
    /// are domain-specific.
    pub maximize_observed_value: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveLearningSession {
    pub schema: String,
    pub target_digest: LearningTargetDigest,
    pub policy: AdaptiveLearningPolicy,
    pub policy_digest: AdaptiveLearningPolicyDigest,
    pub capability_ids: Vec<CapabilityId>,
    pub posterior: GaussianAperturePosterior,
    pub completed_aperture_ids: Vec<ApertureId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveLearningStep {
    pub schema: String,
    pub aperture_id: ApertureId,
    pub capability_weights: Vec<f64>,
    pub active_capability_ids: Vec<CapabilityId>,
    pub noise_variance: f64,
    pub cost: f64,
    pub risk: f64,
    pub information_gain: f64,
    /// The posterior expectation of the result produced by this aperture.
    pub predicted_outcome: f64,
    /// Signed utility contribution from the explicit live policy.
    pub outcome_utility: f64,
    pub objective: f64,
}

/// Immutable evidence produced by an actual aperture experiment. The runner
/// must emit this envelope only after the experiment, its measurement and all
/// referenced artifacts exist. The Rust cycle never fabricates one.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LearningExperimentEvidence {
    pub schema: String,
    pub session_id: SessionId,
    pub target_digest: LearningTargetDigest,
    pub aperture_id: ApertureId,
    pub observed_value: f64,
    pub observation_id: ObservationId,
    /// The exact experimental observation used to derive `observed_value`.
    pub observation: EvidenceReference,
    pub evidence_files: Vec<EvidenceReference>,
}

/// The mutable logical state is represented only inside immutable receipts.
/// A small atomic pointer selects the current receipt; the receipt history is
/// append-only and hash-addressed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveLearningCycle {
    pub schema: String,
    pub session_id: SessionId,
    pub target: LearningTarget,
    pub target_digest: LearningTargetDigest,
    pub policy_digest: AdaptiveLearningPolicyDigest,
    pub session: AdaptiveLearningSession,
    pub pending_step: Option<AdaptiveLearningStep>,
    pub completed_evidence: Vec<LearningExperimentEvidence>,
    pub completed_evidence_sha256: Vec<LearningEvidenceDigest>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AdaptiveLearningEventKind {
    #[serde(rename = "session_started")]
    SessionStarted,
    #[serde(rename = "aperture_issued")]
    ApertureIssued,
    #[serde(rename = "result_assimilated")]
    ResultAssimilated,
}

impl AdaptiveLearningEventKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SessionStarted => "session_started",
            Self::ApertureIssued => "aperture_issued",
            Self::ResultAssimilated => "result_assimilated",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveLearningReceipt {
    pub schema: String,
    pub event_kind: AdaptiveLearningEventKind,
    pub generation: u64,
    pub session_id: SessionId,
    pub target_digest: LearningTargetDigest,
    pub policy_digest: AdaptiveLearningPolicyDigest,
    pub prior_receipt_sha256: Option<AdaptiveLearningReceiptDigest>,
    pub evidence_sha256: Option<LearningEvidenceDigest>,
    pub cycle: AdaptiveLearningCycle,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct AdaptiveLearningPointer {
    schema: String,
    session_id: SessionId,
    receipt_sha256: AdaptiveLearningReceiptDigest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoadedAdaptiveLearningReceipt {
    pub receipt_sha256: AdaptiveLearningReceiptDigest,
    pub receipt: AdaptiveLearningReceipt,
}

fn validate_target(target: &LearningTarget) -> BrainResult<()> {
    if target.capability_ids.len() < 2
        || target.capability_ids.len() > 256
        || target.candidate_budget < target.capability_ids.len()
        || target.plan_steps < target.capability_ids.len()
        || target.plan_steps > target.candidate_budget
        || !target.noise_variance.is_finite()
        || target.noise_variance <= 0.0
        || !target.cost_weight.is_finite()
        || target.cost_weight < 0.0
        || !target.risk_weight.is_finite()
        || target.risk_weight < 0.0
    {
        return Err(BrainError::Invalid("learning_target_invalid".into()));
    }
    let mut seen = BTreeSet::new();
    for id in &target.capability_ids {
        if !seen.insert(id) {
            return Err(BrainError::Invalid(
                "learning_target_capability_identity_invalid".into(),
            ));
        }
    }
    Ok(())
}

fn validate_policy(policy: &AdaptiveLearningPolicy) -> BrainResult<()> {
    if policy.schema != "cerebro.tidex.adaptive_learning_policy/v1"
        || !policy.outcome_utility_weight.is_finite()
        || policy.outcome_utility_weight <= 0.0
    {
        return Err(BrainError::Invalid(
            "adaptive_learning_policy_contract_invalid".into(),
        ));
    }
    Ok(())
}

fn normalized(mut values: Vec<f64>) -> BrainResult<Vec<f64>> {
    let magnitude = norm(&values)?;
    if !magnitude.is_finite() || magnitude <= 1e-15 {
        return Err(BrainError::Invalid(
            "learning_aperture_zero_direction".into(),
        ));
    }
    for value in &mut values {
        *value /= magnitude;
    }
    Ok(values)
}

fn stable_u64(target_id: &str, counter: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:LEARNING-APERTURE-SEED:v1\0");
    hasher.update(target_id.as_bytes());
    hasher.update(counter.to_be_bytes());
    let digest = hasher.finalize();
    let mut prefix = [0_u8; 8];
    prefix.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(prefix)
}

fn next_random(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn candidate_key(weights: &[f64]) -> Vec<u8> {
    weights
        .iter()
        .map(|value| if value.abs() > 1e-12 { 1 } else { 0 })
        .collect()
}

fn generate_candidates(target: &LearningTarget) -> BrainResult<Vec<ApertureCandidate>> {
    validate_target(target)?;
    let dimension = target.capability_ids.len();
    let mut rows = Vec::<(ApertureId, Vec<f64>, f64, f64)>::new();
    let mut seen = BTreeSet::<Vec<u8>>::new();

    // Identity apertures guarantee a spanning design even if every mixed
    // candidate is later rejected by information/cost trade-offs.
    for index in 0..dimension {
        let mut direction = vec![0.0; dimension];
        direction[index] = 1.0;
        seen.insert(candidate_key(&direction));
        rows.push((
            ApertureId::parse(format!("isolate-{index:03}"))?,
            direction,
            4.0,
            0.02,
        ));
    }

    // Pair apertures expose pairwise interaction directions without requiring
    // one full training run per capability.
    'pairs: for left in 0..dimension {
        for right in (left + 1)..dimension {
            if rows.len() >= target.candidate_budget {
                break 'pairs;
            }
            let mut direction = vec![0.0; dimension];
            direction[left] = 1.0;
            direction[right] = 1.0;
            let key = candidate_key(&direction);
            if seen.insert(key) {
                rows.push((
                    ApertureId::parse(format!("pair-{left:03}-{right:03}"))?,
                    direction,
                    0.8,
                    0.03,
                ));
            }
        }
    }

    // Deterministic balanced mixed apertures. They are intentionally generated
    // from the target identity rather than task names, so the planner remains
    // generic across independent learning producers and targets.
    let mut counter = 0u64;
    while rows.len() < target.candidate_budget {
        let mut state = stable_u64(target.target_id.as_str(), counter);
        counter = counter.wrapping_add(1);
        let desired = (2 + (next_random(&mut state) as usize % dimension.saturating_sub(1).max(1)))
            .min(dimension);
        let mut indices = (0..dimension).collect::<Vec<_>>();
        for index in (1..indices.len()).rev() {
            let swap = next_random(&mut state) as usize % (index + 1);
            indices.swap(index, swap);
        }
        let mut direction = vec![0.0; dimension];
        for index in indices.into_iter().take(desired) {
            direction[index] = 1.0;
        }
        let key = candidate_key(&direction);
        if !seen.insert(key) {
            if seen.len()
                >= (1usize.checked_shl(dimension.min(20) as u32).unwrap_or(0)).saturating_sub(1)
            {
                break;
            }
            continue;
        }
        rows.push((
            ApertureId::parse(format!("mixed-{:04}", rows.len()))?,
            direction,
            0.65 + 0.04 * desired as f64,
            (0.02 + 0.005 * desired as f64).min(0.25),
        ));
    }

    rows.into_iter()
        .map(|(aperture_id, direction, cost, risk)| {
            Ok(ApertureCandidate {
                aperture_id,
                sensing_vector: normalized(direction)?,
                noise_variance: target.noise_variance,
                cost,
                risk,
            })
        })
        .collect()
}

fn rank(rows: &[Vec<f64>]) -> BrainResult<usize> {
    if rows.is_empty() {
        return Ok(0);
    }
    let matrix = Matrix::from_rows(rows)?;
    let gram = matrix.transpose().matmul(&matrix)?;
    let eigen = crate::linalg::symmetric_eigen_jacobi(
        &gram,
        1e-12,
        gram.rows
            .saturating_mul(gram.rows)
            .saturating_mul(200)
            .max(200),
    )?;
    let max = eigen
        .iter()
        .map(|(value, _)| *value)
        .fold(0.0_f64, f64::max)
        .max(1e-15);
    let tolerance = max * f64::EPSILON.sqrt() * (gram.rows.max(1) as f64);
    Ok(eigen.iter().filter(|(value, _)| *value > tolerance).count())
}

pub fn plan_autonomous_learning(target: &LearningTarget) -> BrainResult<AutonomousLearningPlan> {
    validate_target(target)?;
    let candidates = generate_candidates(target)?;
    let covariance = Matrix::identity(target.capability_ids.len());
    let plan = plan_active_apertures(
        &candidates,
        &covariance,
        target.cost_weight,
        target.risk_weight,
        target.plan_steps,
        false,
    )?;
    let by_id = candidates
        .iter()
        .map(|candidate| (candidate.aperture_id.as_str(), candidate))
        .collect::<BTreeMap<_, _>>();
    let mut coverage = vec![0usize; target.capability_ids.len()];
    let mut apertures = Vec::with_capacity(plan.steps.len());
    let mut selected_rows = Vec::with_capacity(plan.steps.len());
    for step in &plan.steps {
        let candidate = by_id
            .get(step.aperture_id.as_str())
            .ok_or_else(|| BrainError::Integrity("learning_selected_aperture_missing".into()))?;
        selected_rows.push(candidate.sensing_vector.clone());
        let mut active = Vec::new();
        for (index, weight) in candidate.sensing_vector.iter().enumerate() {
            if weight.abs() > 1e-12 {
                coverage[index] += 1;
                active.push(target.capability_ids[index].clone());
            }
        }
        apertures.push(LearningAperture {
            aperture_id: step.aperture_id.clone(),
            capability_weights: candidate.sensing_vector.clone(),
            active_capability_ids: active,
            information_gain: step.information_gain,
            objective: step.objective,
        });
    }
    let design_rank = rank(&selected_rows)?;
    let minimum_capability_coverage = coverage.iter().copied().min().unwrap_or(0);
    let maximum_capability_coverage = coverage.iter().copied().max().unwrap_or(0);
    if design_rank < target.capability_ids.len() || minimum_capability_coverage < 3 {
        return Err(BrainError::Numerical(format!(
            "learning_plan_not_identifiable:rank={design_rank}:dimension={}:min_coverage={minimum_capability_coverage}",
            target.capability_ids.len()
        )));
    }
    let target_digest = target_digest(target)?;
    Ok(AutonomousLearningPlan {
        schema: "cerebro.tidex.autonomous_learning_plan/v1".into(),
        target_id: target.target_id.clone(),
        target_digest,
        capability_ids: target.capability_ids.clone(),
        candidate_count: candidates.len(),
        design_rank,
        minimum_capability_coverage,
        maximum_capability_coverage,
        plan,
        apertures,
    })
}

fn target_digest(target: &LearningTarget) -> BrainResult<LearningTargetDigest> {
    Ok(LearningTargetDigest::from(Sha256Digest::digest_bytes(
        &serde_json::to_vec(target)?,
    )))
}

fn policy_digest(policy: &AdaptiveLearningPolicy) -> BrainResult<AdaptiveLearningPolicyDigest> {
    Ok(AdaptiveLearningPolicyDigest::from(
        Sha256Digest::digest_bytes(&serde_json::to_vec(policy)?),
    ))
}

pub fn start_adaptive_learning(
    target: &LearningTarget,
    policy: &AdaptiveLearningPolicy,
) -> BrainResult<AdaptiveLearningSession> {
    validate_target(target)?;
    validate_policy(policy)?;
    let dimension = target.capability_ids.len();
    Ok(AdaptiveLearningSession {
        schema: "cerebro.tidex.adaptive_learning_session/v1".into(),
        target_digest: target_digest(target)?,
        policy: policy.clone(),
        policy_digest: policy_digest(policy)?,
        capability_ids: target.capability_ids.clone(),
        posterior: GaussianAperturePosterior {
            mean: vec![0.0; dimension],
            covariance: (0..dimension)
                .map(|row| {
                    (0..dimension)
                        .map(|col| if row == col { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect(),
        },
        completed_aperture_ids: Vec::new(),
    })
}

fn validate_adaptive_session(
    target: &LearningTarget,
    session: &AdaptiveLearningSession,
) -> BrainResult<()> {
    validate_target(target)?;
    if session.schema != "cerebro.tidex.adaptive_learning_session/v1"
        || session.target_digest != target_digest(target)?
        || session.policy_digest != policy_digest(&session.policy)?
        || session.capability_ids != target.capability_ids
        || session.posterior.mean.len() != target.capability_ids.len()
        || session.posterior.covariance.len() != target.capability_ids.len()
        || session
            .posterior
            .covariance
            .iter()
            .any(|row| row.len() != target.capability_ids.len())
    {
        return Err(BrainError::Integrity(
            "adaptive_learning_session_contract_invalid".into(),
        ));
    }
    validate_policy(&session.policy)?;
    let unique = session
        .completed_aperture_ids
        .iter()
        .collect::<BTreeSet<_>>();
    if unique.len() != session.completed_aperture_ids.len() {
        return Err(BrainError::Integrity(
            "adaptive_learning_completed_aperture_duplicate".into(),
        ));
    }
    Ok(())
}

fn same_f64(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= f64::EPSILON.sqrt() * 64.0 * (1.0 + left.abs().max(right.abs()))
}

fn same_f64_vector(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| same_f64(*left, *right))
}

fn same_policy(left: &AdaptiveLearningPolicy, right: &AdaptiveLearningPolicy) -> bool {
    left.schema == right.schema
        && left.maximize_observed_value == right.maximize_observed_value
        && same_f64(left.outcome_utility_weight, right.outcome_utility_weight)
}

fn same_target(left: &LearningTarget, right: &LearningTarget) -> bool {
    left.target_id == right.target_id
        && left.capability_ids == right.capability_ids
        && left.candidate_budget == right.candidate_budget
        && left.plan_steps == right.plan_steps
        && same_f64(left.noise_variance, right.noise_variance)
        && same_f64(left.cost_weight, right.cost_weight)
        && same_f64(left.risk_weight, right.risk_weight)
}

fn same_session(left: &AdaptiveLearningSession, right: &AdaptiveLearningSession) -> bool {
    left.schema == right.schema
        && left.target_digest == right.target_digest
        && same_policy(&left.policy, &right.policy)
        && left.policy_digest == right.policy_digest
        && left.capability_ids == right.capability_ids
        && same_f64_vector(&left.posterior.mean, &right.posterior.mean)
        && left.posterior.covariance.len() == right.posterior.covariance.len()
        && left
            .posterior
            .covariance
            .iter()
            .zip(&right.posterior.covariance)
            .all(|(left, right)| same_f64_vector(left, right))
        && left.completed_aperture_ids == right.completed_aperture_ids
}

fn same_step(left: &AdaptiveLearningStep, right: &AdaptiveLearningStep) -> bool {
    left.schema == right.schema
        && left.aperture_id == right.aperture_id
        && same_f64_vector(&left.capability_weights, &right.capability_weights)
        && left.active_capability_ids == right.active_capability_ids
        && same_f64(left.noise_variance, right.noise_variance)
        && same_f64(left.cost, right.cost)
        && same_f64(left.risk, right.risk)
        && same_f64(left.information_gain, right.information_gain)
        && same_f64(left.predicted_outcome, right.predicted_outcome)
        && same_f64(left.outcome_utility, right.outcome_utility)
        && same_f64(left.objective, right.objective)
}

fn same_evidence_reference(left: &EvidenceReference, right: &EvidenceReference) -> bool {
    left.path == right.path && left.sha256 == right.sha256
}

fn same_experiment_evidence(
    left: &LearningExperimentEvidence,
    right: &LearningExperimentEvidence,
) -> bool {
    left.schema == right.schema
        && left.session_id == right.session_id
        && left.target_digest == right.target_digest
        && left.aperture_id == right.aperture_id
        && same_f64(left.observed_value, right.observed_value)
        && left.observation_id == right.observation_id
        && same_evidence_reference(&left.observation, &right.observation)
        && left.evidence_files.len() == right.evidence_files.len()
        && left
            .evidence_files
            .iter()
            .zip(&right.evidence_files)
            .all(|(left, right)| same_evidence_reference(left, right))
}

fn same_evidence_sequence(
    left: &[LearningExperimentEvidence],
    right: &[LearningExperimentEvidence],
) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| same_experiment_evidence(left, right))
}

pub fn next_learning_aperture(
    target: &LearningTarget,
    session: &AdaptiveLearningSession,
) -> BrainResult<AdaptiveLearningStep> {
    validate_adaptive_session(target, session)?;
    if session.completed_aperture_ids.len() >= target.plan_steps {
        return Err(BrainError::Invalid(
            "adaptive_learning_plan_step_budget_exhausted".into(),
        ));
    }
    let completed = session
        .completed_aperture_ids
        .iter()
        .map(ApertureId::as_str)
        .collect::<BTreeSet<_>>();
    let remaining = generate_candidates(target)?
        .into_iter()
        .filter(|candidate| !completed.contains(candidate.aperture_id.as_str()))
        .collect::<Vec<_>>();
    if remaining.is_empty() {
        return Err(BrainError::Invalid(
            "adaptive_learning_no_remaining_apertures".into(),
        ));
    }
    let covariance = Matrix::from_rows(&session.posterior.covariance)?;
    let direction = if session.policy.maximize_observed_value {
        1.0
    } else {
        -1.0
    };
    // Reuse the authoritative active-aperture scorer for information, cost and
    // risk. The explicit posterior-mean term turns the live cycle into a
    // genuine outcome-conditioned design after assimilation.
    let mut selected: Option<(ApertureCandidate, f64, f64, f64, f64)> = None;
    for candidate in remaining {
        let score = choose_active_aperture(
            std::slice::from_ref(&candidate),
            &covariance,
            target.cost_weight,
            target.risk_weight,
        )?;
        let predicted_outcome = dot(&candidate.sensing_vector, &session.posterior.mean)?;
        let outcome_utility = direction * session.policy.outcome_utility_weight * predicted_outcome;
        let objective = score.objective + outcome_utility;
        if !objective.is_finite() || !predicted_outcome.is_finite() || !outcome_utility.is_finite()
        {
            return Err(BrainError::Numerical(
                "adaptive_learning_outcome_objective_non_finite".into(),
            ));
        }
        let replace = selected
            .as_ref()
            .is_none_or(|(current, _, _, _, current_objective)| {
                objective > *current_objective
                    || (objective == *current_objective
                        && candidate.aperture_id < current.aperture_id)
            });
        if replace {
            selected = Some((
                candidate,
                score.information_gain,
                predicted_outcome,
                outcome_utility,
                objective,
            ));
        }
    }
    let (selected, information_gain, predicted_outcome, outcome_utility, objective) = selected
        .ok_or_else(|| BrainError::Integrity("adaptive_learning_selected_missing".into()))?;
    let active_capability_ids = selected
        .sensing_vector
        .iter()
        .enumerate()
        .filter(|(_, value)| value.abs() > 1e-12)
        .map(|(index, _)| target.capability_ids[index].clone())
        .collect::<Vec<_>>();
    Ok(AdaptiveLearningStep {
        schema: "cerebro.tidex.adaptive_learning_step/v1".into(),
        aperture_id: selected.aperture_id,
        capability_weights: selected.sensing_vector,
        active_capability_ids,
        noise_variance: selected.noise_variance,
        cost: selected.cost,
        risk: selected.risk,
        information_gain,
        predicted_outcome,
        outcome_utility,
        objective,
    })
}

pub fn assimilate_learning_result(
    target: &LearningTarget,
    session: &AdaptiveLearningSession,
    step: &AdaptiveLearningStep,
    observed_value: f64,
) -> BrainResult<AdaptiveLearningSession> {
    validate_adaptive_session(target, session)?;
    if step.schema != "cerebro.tidex.adaptive_learning_step/v1"
        || session
            .completed_aperture_ids
            .iter()
            .any(|id| id == &step.aperture_id)
    {
        return Err(BrainError::Invalid(
            "adaptive_learning_step_contract_invalid".into(),
        ));
    }
    let expected = next_learning_aperture(target, session)?;
    if !same_step(&expected, step) {
        return Err(BrainError::Integrity(
            "adaptive_learning_step_not_current_canonical_choice".into(),
        ));
    }
    let canonical = generate_candidates(target)?
        .into_iter()
        .find(|candidate| candidate.aperture_id == step.aperture_id)
        .ok_or_else(|| BrainError::Integrity("adaptive_learning_aperture_unknown".into()))?;
    let posterior = assimilate_aperture_result(&session.posterior, &canonical, observed_value)?;
    let mut completed = session.completed_aperture_ids.clone();
    completed.push(step.aperture_id.clone());
    Ok(AdaptiveLearningSession {
        schema: session.schema.clone(),
        target_digest: session.target_digest.clone(),
        policy: session.policy.clone(),
        policy_digest: session.policy_digest.clone(),
        capability_ids: session.capability_ids.clone(),
        posterior,
        completed_aperture_ids: completed,
    })
}

#[cfg(test)]
pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Return an existing regular private file, rejecting a symlink at the leaf
/// or any parent.  A missing leaf is distinct from an invalid existing entry
/// so immutable writers can safely reserve a new content-addressed path.
pub(crate) fn private_regular_file_if_present(
    root: &Path,
    path: &Path,
) -> BrainResult<Option<PathBuf>> {
    existing_regular_file_if_present(root, path)
}

/// Resolve an existing private directory, if present, through the shared
/// authority.  Presence is deliberately separated from validity so callers
/// can preflight an entire state topology without creating or chmodding it.
pub(crate) fn private_directory_if_present(
    root: &Path,
    path: &Path,
) -> BrainResult<Option<PathBuf>> {
    existing_directory_if_present(root, path)
}

/// Create a new immutable authority file. Existing content is never silently
/// accepted here: callers that deliberately support idempotent evidence must
/// authenticate it first through `private_regular_file_if_present`.
pub(crate) fn write_new_private(root: &Path, path: &Path, bytes: &[u8]) -> BrainResult<()> {
    create_private_immutable(root, path, bytes)?;
    Ok(())
}

/// Atomically replace one authenticated `current` pointer. Receipts and
/// evidence use the immutable writer above; this is intentionally restricted
/// to a normal regular-file pointer under the verified root.
pub(crate) fn write_private_atomic(root: &Path, path: &Path, bytes: &[u8]) -> BrainResult<()> {
    replace_private_file_atomic(root, path, bytes, None)?;
    Ok(())
}

fn learning_sessions_root(root: &Path) -> PathBuf {
    root.join("state/learning_sessions")
}

fn receipt_path(root: &Path, digest: &str) -> PathBuf {
    learning_sessions_root(root)
        .join("by-sha")
        .join(format!("{digest}.json"))
}

fn pointer_path(root: &Path, session_id: &str) -> PathBuf {
    learning_sessions_root(root)
        .join("current")
        .join(format!("{session_id}.json"))
}

fn marker_path(root: &Path, session_id: &str) -> PathBuf {
    learning_sessions_root(root)
        .join("session-ids")
        .join(format!("{session_id}.json"))
}

fn lock_path(root: &Path, session_id: &str) -> PathBuf {
    learning_sessions_root(root)
        .join("locks")
        .join(format!("{session_id}.lock"))
}

fn experiment_evidence_path(root: &Path, digest: &str) -> PathBuf {
    root.join("state/learning_evidence/by-sha")
        .join(format!("{digest}.json"))
}

/// Inspect every persistent adaptive-learning location before an operation is
/// allowed to create, chmod, read, or replace any one of them.  Each actual
/// resolution is delegated to `authority`; this only enumerates the canonical
/// topology so one hostile sibling (for example `current`) cannot cause a
/// write to `locks` before the operation fails closed.
fn validate_learning_state_topology(root: &Path, session_id: &str) -> BrainResult<()> {
    if SessionId::parse(session_id).is_err() {
        return Err(BrainError::Invalid(
            "adaptive_learning_session_id_invalid".into(),
        ));
    }
    let state = root.join("state");
    let sessions = learning_sessions_root(root);
    let evidence = root.join("state/learning_evidence");
    for directory in [
        state,
        sessions.clone(),
        sessions.join("by-sha"),
        sessions.join("current"),
        sessions.join("session-ids"),
        sessions.join("locks"),
        evidence.clone(),
        evidence.join("by-sha"),
    ] {
        let _ = private_directory_if_present(root, &directory)?;
    }
    for path in [
        pointer_path(root, session_id),
        marker_path(root, session_id),
        lock_path(root, session_id),
    ] {
        let _ = private_regular_file_if_present(root, &path)?;
    }
    Ok(())
}

fn with_session_lock<T, F>(root: &Path, session_id: &str, operation: F) -> BrainResult<T>
where
    F: FnOnce() -> BrainResult<T>,
{
    with_private_authority_lock(root, &lock_path(root, session_id), operation)
}

pub(crate) fn confined_existing_file(
    root: &Path,
    raw: impl AsRef<Path>,
    expected_sha256: Option<&str>,
) -> BrainResult<PathBuf> {
    let path = existing_regular_file_under_root(root, raw.as_ref())?;
    if let Some(expected) = expected_sha256 {
        let expected = Sha256Digest::parse(expected).map_err(|_| {
            BrainError::Integrity("adaptive_learning_evidence_digest_mismatch".into())
        })?;
        EvidenceReference::new(path.clone(), expected).verify(root)?;
    }
    Ok(path)
}

fn validate_experiment_evidence(
    root: &Path,
    evidence: &LearningExperimentEvidence,
    session_id: &SessionId,
    target_digest: &LearningTargetDigest,
    aperture_id: &ApertureId,
    capability_weights: &[f64],
) -> BrainResult<()> {
    if evidence.schema != "cerebro.tidex.learning_experiment_evidence/v1"
        || evidence.session_id != *session_id
        || evidence.target_digest != *target_digest
        || evidence.aperture_id != *aperture_id
        || !evidence.observed_value.is_finite()
        || evidence.evidence_files.is_empty()
        || !Sha256Digest::is_valid_str(evidence.observation.sha256.as_str())
    {
        return Err(BrainError::Integrity(
            "adaptive_learning_experiment_evidence_contract_invalid".into(),
        ));
    }
    let observation_raw = evidence
        .observation
        .read_verified_bounded(root, MAX_LEARNING_AUTHORITY_JSON_BYTES)?;
    let observation: DeltaObservation = serde_json::from_slice(&observation_raw)?;
    if observation.observation_id != evidence.observation_id
        || observation.independence_group != aperture_id.as_str()
        || observation.delta.is_empty()
        || observation.delta.iter().any(|value| !value.is_finite())
        || observation.functional_response.len() != capability_weights.len()
        || observation
            .functional_response
            .iter()
            .any(|value| !value.is_finite())
        || !observation.reliability.is_finite()
        || !(0.0..=1.0).contains(&observation.reliability)
        || observation.reliability <= 0.0
        || observation.dense_artifact.is_none()
        || observation
            .parameter_layout_sha256
            .as_deref()
            .is_none_or(|digest| !Sha256Digest::is_valid_str(digest))
    {
        return Err(BrainError::Integrity(
            "adaptive_learning_observation_contract_invalid".into(),
        ));
    }
    let dense = observation.dense_artifact.as_ref().ok_or_else(|| {
        BrainError::Integrity("adaptive_learning_observation_dense_artifact_missing".into())
    })?;
    let dense_path = confined_existing_file(root, &dense.path, Some(&dense.sha256))?;
    if dense_path.as_path() != Path::new(&dense.path) {
        return Err(BrainError::Integrity(
            "adaptive_learning_observation_dense_artifact_noncanonical".into(),
        ));
    }
    let dense_values = crate::artifact::read_dvec_f32(root, dense)?;
    if dense_values.len() as u64 != dense.parameter_count || dense_values.is_empty() {
        return Err(BrainError::Integrity(
            "adaptive_learning_observation_dense_artifact_contract_invalid".into(),
        ));
    }
    let layout_digest = observation
        .parameter_layout_sha256
        .as_ref()
        .ok_or_else(|| {
            BrainError::Integrity("adaptive_learning_observation_layout_digest_missing".into())
        })?;
    let layout_path = root
        .join("state/parameter_layouts/by-sha")
        .join(format!("{layout_digest}.json"));
    let _ = confined_existing_file(root, &layout_path, Some(layout_digest)).map_err(|_| {
        BrainError::Integrity("adaptive_learning_observation_layout_artifact_invalid".into())
    })?;
    let derived = dot(capability_weights, &observation.functional_response)?;
    let tolerance = f64::EPSILON.sqrt()
        * (1.0 + derived.abs().max(evidence.observed_value.abs()))
        * capability_weights.len().max(1) as f64
        * 32.0;
    if (derived - evidence.observed_value).abs() > tolerance {
        return Err(BrainError::Integrity(
            "adaptive_learning_observed_value_not_derived_from_observation".into(),
        ));
    }
    let mut unique = BTreeSet::new();
    for reference in &evidence.evidence_files {
        if !Sha256Digest::is_valid_str(reference.sha256.as_str())
            || !unique.insert((reference.path.clone(), reference.sha256.clone()))
        {
            return Err(BrainError::Integrity(
                "adaptive_learning_experiment_evidence_reference_invalid".into(),
            ));
        }
        let _ = confined_existing_file(root, &reference.path, Some(&reference.sha256))?;
    }
    Ok(())
}

fn validate_cycle(root: &Path, cycle: &AdaptiveLearningCycle) -> BrainResult<()> {
    if cycle.schema != "cerebro.tidex.adaptive_learning_cycle/v1"
        || cycle.target_digest != target_digest(&cycle.target)?
        || cycle.policy_digest != policy_digest(&cycle.session.policy)?
        || cycle.session.target_digest != cycle.target_digest
        || cycle.session.policy_digest != cycle.policy_digest
        || cycle.session.capability_ids != cycle.target.capability_ids
        || cycle.completed_evidence.len() != cycle.session.completed_aperture_ids.len()
        || cycle.completed_evidence.len() != cycle.completed_evidence_sha256.len()
    {
        return Err(BrainError::Integrity(
            "adaptive_learning_cycle_contract_invalid".into(),
        ));
    }
    validate_adaptive_session(&cycle.target, &cycle.session)?;
    let mut completed_ids = BTreeSet::new();
    for ((evidence, digest), completed_id) in cycle
        .completed_evidence
        .iter()
        .zip(&cycle.completed_evidence_sha256)
        .zip(&cycle.session.completed_aperture_ids)
    {
        if evidence.aperture_id != *completed_id || !completed_ids.insert(completed_id.clone()) {
            return Err(BrainError::Integrity(
                "adaptive_learning_completed_evidence_identity_invalid".into(),
            ));
        }
        let reference = EvidenceReference::new(
            experiment_evidence_path(root, digest.as_str()),
            digest.as_digest().clone(),
        );
        let raw = reference
            .read_verified_bounded(root, MAX_LEARNING_AUTHORITY_JSON_BYTES)
            .map_err(|_| {
                BrainError::Integrity(
                    "adaptive_learning_completed_evidence_artifact_invalid".into(),
                )
            })?;
        let persisted: LearningExperimentEvidence = serde_json::from_slice(&raw)?;
        if !same_experiment_evidence(&persisted, evidence) {
            return Err(BrainError::Integrity(
                "adaptive_learning_completed_evidence_content_mismatch".into(),
            ));
        }
        let candidate = generate_candidates(&cycle.target)?
            .into_iter()
            .find(|candidate| candidate.aperture_id == evidence.aperture_id)
            .ok_or_else(|| {
                BrainError::Integrity(
                    "adaptive_learning_completed_evidence_aperture_unknown".into(),
                )
            })?;
        // The stored evidence is tied to the aperture identity and concrete
        // artifacts. The exact live score is validated by receipt transition
        // replay below, where the predecessor posterior is available.
        validate_experiment_evidence(
            root,
            evidence,
            &cycle.session_id,
            &cycle.target_digest,
            &evidence.aperture_id,
            &candidate.sensing_vector,
        )?;
    }
    if let Some(pending) = &cycle.pending_step {
        if cycle
            .session
            .completed_aperture_ids
            .iter()
            .any(|id| id == &pending.aperture_id)
            || !same_step(
                &next_learning_aperture(&cycle.target, &cycle.session)?,
                pending,
            )
        {
            return Err(BrainError::Integrity(
                "adaptive_learning_pending_step_not_canonical".into(),
            ));
        }
    }
    Ok(())
}

fn validate_receipt_shape(root: &Path, receipt: &AdaptiveLearningReceipt) -> BrainResult<()> {
    if receipt.schema != "cerebro.tidex.adaptive_learning_receipt/v1"
        || receipt.cycle.session_id != receipt.session_id
        || receipt.cycle.target_digest != receipt.target_digest
        || receipt.cycle.policy_digest != receipt.policy_digest
    {
        return Err(BrainError::Integrity(
            "adaptive_learning_receipt_contract_invalid".into(),
        ));
    }
    validate_cycle(root, &receipt.cycle)
}

fn load_receipt_by_sha(
    root: &Path,
    digest: &AdaptiveLearningReceiptDigest,
) -> BrainResult<AdaptiveLearningReceipt> {
    let reference = EvidenceReference::new(
        receipt_path(root, digest.as_str()),
        digest.as_digest().clone(),
    );
    let raw = reference
        .read_verified_bounded(root, MAX_LEARNING_AUTHORITY_JSON_BYTES)
        .map_err(|_| BrainError::Integrity("adaptive_learning_receipt_artifact_invalid".into()))?;
    Ok(serde_json::from_slice(&raw)?)
}

fn verify_receipt_ledger_binding(
    root: &Path,
    digest: &AdaptiveLearningReceiptDigest,
    receipt: &AdaptiveLearningReceipt,
) -> BrainResult<()> {
    let event = ledger::find_v2_event_by_payload_string(
        root,
        "adaptive_learning_receipt",
        "receipt_sha256",
        digest.as_str(),
    )?
    .ok_or_else(|| {
        BrainError::Integrity("adaptive_learning_receipt_ledger_event_missing".into())
    })?;
    let payload = event.payload()?;
    if payload
        .get("session_id")
        .and_then(serde_json::Value::as_str)
        != Some(receipt.session_id.as_str())
        || payload
            .get("target_digest")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.target_digest.as_str())
        || payload
            .get("event_kind")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.event_kind.as_str())
    {
        return Err(BrainError::Integrity(
            "adaptive_learning_receipt_ledger_payload_mismatch".into(),
        ));
    }
    Ok(())
}

fn same_cycle_base(left: &AdaptiveLearningCycle, right: &AdaptiveLearningCycle) -> bool {
    left.schema == right.schema
        && left.session_id == right.session_id
        && same_target(&left.target, &right.target)
        && left.target_digest == right.target_digest
        && left.policy_digest == right.policy_digest
        && same_session(&left.session, &right.session)
        && same_evidence_sequence(&left.completed_evidence, &right.completed_evidence)
        && left.completed_evidence_sha256 == right.completed_evidence_sha256
}

fn validate_receipt_transition(
    root: &Path,
    digest: &AdaptiveLearningReceiptDigest,
    seen: &mut BTreeSet<AdaptiveLearningReceiptDigest>,
) -> BrainResult<AdaptiveLearningReceipt> {
    if !seen.insert(digest.clone()) {
        return Err(BrainError::Integrity(
            "adaptive_learning_receipt_chain_cycle".into(),
        ));
    }
    let receipt = load_receipt_by_sha(root, digest)?;
    validate_receipt_shape(root, &receipt)?;
    verify_receipt_ledger_binding(root, digest, &receipt)?;
    match (&receipt.prior_receipt_sha256, receipt.event_kind) {
        (None, AdaptiveLearningEventKind::SessionStarted) => {
            if receipt.generation != 0
                || receipt.evidence_sha256.is_some()
                || receipt.cycle.pending_step.is_some()
                || !receipt.cycle.completed_evidence.is_empty()
                || !receipt.cycle.completed_evidence_sha256.is_empty()
                || !same_session(
                    &receipt.cycle.session,
                    &start_adaptive_learning(&receipt.cycle.target, &receipt.cycle.session.policy)?,
                )
            {
                return Err(BrainError::Integrity(
                    "adaptive_learning_start_receipt_invalid".into(),
                ));
            }
        }
        (Some(prior_digest), AdaptiveLearningEventKind::ApertureIssued) => {
            if receipt.evidence_sha256.is_some() {
                return Err(BrainError::Integrity(
                    "adaptive_learning_issue_receipt_evidence_unexpected".into(),
                ));
            }
            let prior = validate_receipt_transition(root, prior_digest, seen)?;
            if receipt.generation != prior.generation.saturating_add(1)
                || receipt.session_id != prior.session_id
                || receipt.target_digest != prior.target_digest
                || receipt.policy_digest != prior.policy_digest
                || prior.cycle.pending_step.is_some()
            {
                return Err(BrainError::Integrity(
                    "adaptive_learning_issue_receipt_lineage_invalid".into(),
                ));
            }
            let pending = receipt.cycle.pending_step.as_ref().ok_or_else(|| {
                BrainError::Integrity("adaptive_learning_issue_pending_step_missing".into())
            })?;
            if !same_cycle_base(&prior.cycle, &receipt.cycle)
                || !same_step(
                    pending,
                    &next_learning_aperture(&prior.cycle.target, &prior.cycle.session)?,
                )
            {
                return Err(BrainError::Integrity(
                    "adaptive_learning_issue_receipt_transition_invalid".into(),
                ));
            }
        }
        (Some(prior_digest), AdaptiveLearningEventKind::ResultAssimilated) => {
            let evidence_digest = receipt.evidence_sha256.as_ref().ok_or_else(|| {
                BrainError::Integrity(
                    "adaptive_learning_assimilation_evidence_digest_missing".into(),
                )
            })?;
            let prior = validate_receipt_transition(root, prior_digest, seen)?;
            let pending = prior.cycle.pending_step.as_ref().ok_or_else(|| {
                BrainError::Integrity("adaptive_learning_assimilation_pending_step_missing".into())
            })?;
            let evidence = receipt.cycle.completed_evidence.last().ok_or_else(|| {
                BrainError::Integrity("adaptive_learning_assimilation_evidence_missing".into())
            })?;
            if receipt.generation != prior.generation.saturating_add(1)
                || receipt.session_id != prior.session_id
                || receipt.target_digest != prior.target_digest
                || receipt.policy_digest != prior.policy_digest
                || receipt.cycle.pending_step.is_some()
                || receipt.cycle.completed_evidence.len()
                    != prior.cycle.completed_evidence.len().saturating_add(1)
                || receipt.cycle.completed_evidence_sha256.len()
                    != prior
                        .cycle
                        .completed_evidence_sha256
                        .len()
                        .saturating_add(1)
                || !same_evidence_sequence(
                    &receipt.cycle.completed_evidence[..prior.cycle.completed_evidence.len()],
                    &prior.cycle.completed_evidence,
                )
                || receipt.cycle.completed_evidence_sha256
                    [..prior.cycle.completed_evidence_sha256.len()]
                    != prior.cycle.completed_evidence_sha256[..]
                || receipt
                    .cycle
                    .completed_evidence_sha256
                    .last()
                    .map(LearningEvidenceDigest::as_str)
                    != Some(evidence_digest.as_str())
            {
                return Err(BrainError::Integrity(
                    "adaptive_learning_assimilation_receipt_lineage_invalid".into(),
                ));
            }
            validate_experiment_evidence(
                root,
                evidence,
                &receipt.session_id,
                &receipt.target_digest,
                &pending.aperture_id,
                &pending.capability_weights,
            )?;
            let expected_session = assimilate_learning_result(
                &prior.cycle.target,
                &prior.cycle.session,
                pending,
                evidence.observed_value,
            )?;
            if !same_target(&receipt.cycle.target, &prior.cycle.target)
                || receipt.cycle.target_digest != prior.cycle.target_digest
                || receipt.cycle.policy_digest != prior.cycle.policy_digest
                || !same_session(&receipt.cycle.session, &expected_session)
            {
                return Err(BrainError::Integrity(
                    "adaptive_learning_assimilation_receipt_transition_invalid".into(),
                ));
            }
        }
        _ => {
            return Err(BrainError::Integrity(
                "adaptive_learning_receipt_event_lineage_invalid".into(),
            ));
        }
    }
    Ok(receipt)
}

fn load_current_receipt_under_root(
    root: &Path,
    session_id: &str,
) -> BrainResult<LoadedAdaptiveLearningReceipt> {
    if SessionId::parse(session_id).is_err() {
        return Err(BrainError::Invalid(
            "adaptive_learning_session_id_invalid".into(),
        ));
    }
    validate_learning_state_topology(root, session_id)?;
    let pointer_raw = read_existing_private_file_bounded(
        root,
        &pointer_path(root, session_id),
        MAX_LEARNING_AUTHORITY_JSON_BYTES,
    )
    .map_err(|_| {
        BrainError::Integrity("adaptive_learning_current_pointer_missing_or_invalid".into())
    })?;
    let pointer: AdaptiveLearningPointer = serde_json::from_slice(&pointer_raw)?;
    if pointer.schema != "cerebro.tidex.adaptive_learning_pointer/v1"
        || pointer.session_id.as_str() != session_id
    {
        return Err(BrainError::Integrity(
            "adaptive_learning_current_pointer_contract_invalid".into(),
        ));
    }
    let mut seen = BTreeSet::new();
    let receipt = validate_receipt_transition(root, &pointer.receipt_sha256, &mut seen)?;
    if receipt.session_id.as_str() != session_id {
        return Err(BrainError::Integrity(
            "adaptive_learning_current_pointer_session_mismatch".into(),
        ));
    }
    Ok(LoadedAdaptiveLearningReceipt {
        receipt_sha256: pointer.receipt_sha256,
        receipt,
    })
}

fn persist_receipt_under_root(
    root: &Path,
    receipt: AdaptiveLearningReceipt,
) -> BrainResult<LoadedAdaptiveLearningReceipt> {
    validate_receipt_shape(root, &receipt)?;
    let raw = serde_json::to_vec_pretty(&receipt)?;
    let digest = AdaptiveLearningReceiptDigest::from(Sha256Digest::digest_bytes(&raw));
    let path = receipt_path(root, digest.as_str());
    if private_regular_file_if_present(root, &path)?.is_some() {
        return Err(BrainError::Integrity(
            "adaptive_learning_receipt_digest_already_exists".into(),
        ));
    }
    write_new_private(root, &path, &raw)?;
    let event = ledger::append(
        root,
        "adaptive_learning_receipt",
        json!({
            "schema":"cerebro.tidex.adaptive_learning_ledger_binding/v1",
            "receipt_sha256":&digest,
            "session_id":&receipt.session_id,
            "target_digest":&receipt.target_digest,
            "policy_digest":&receipt.policy_digest,
            "event_kind":receipt.event_kind,
            "generation":receipt.generation,
        }),
    )?;
    let payload = event.payload()?;
    if payload
        .get("receipt_sha256")
        .and_then(serde_json::Value::as_str)
        != Some(digest.as_str())
    {
        return Err(BrainError::Integrity(
            "adaptive_learning_ledger_receipt_digest_mismatch".into(),
        ));
    }
    let pointer = AdaptiveLearningPointer {
        schema: "cerebro.tidex.adaptive_learning_pointer/v1".into(),
        session_id: receipt.session_id.clone(),
        receipt_sha256: digest.clone(),
    };
    let mut pointer_raw = serde_json::to_vec_pretty(&pointer)?;
    pointer_raw.push(b'\n');
    write_private_atomic(
        root,
        &pointer_path(root, receipt.session_id.as_str()),
        &pointer_raw,
    )?;
    let loaded = load_current_receipt_under_root(root, receipt.session_id.as_str())?;
    // `load_current_receipt_under_root` has re-read the pointer, verified the
    // content-addressed receipt and replayed its ledger-bound transition. The
    // digest is the exact serialized receipt identity; comparing re-parsed
    // floating-point structs again is weaker and can reject a byte-identical
    // receipt solely on non-semantic representation details.
    if loaded.receipt_sha256 != digest {
        return Err(BrainError::Integrity(
            "adaptive_learning_receipt_postwrite_revalidation_failed".into(),
        ));
    }
    Ok(loaded)
}

fn start_persistent_adaptive_learning_under_root(
    root: &Path,
    session_id: &str,
    target: &LearningTarget,
    policy: &AdaptiveLearningPolicy,
) -> BrainResult<LoadedAdaptiveLearningReceipt> {
    if SessionId::parse(session_id).is_err() {
        return Err(BrainError::Invalid(
            "adaptive_learning_session_id_invalid".into(),
        ));
    }
    validate_target(target)?;
    validate_policy(policy)?;
    validate_learning_state_topology(root, session_id)?;
    with_session_lock(root, session_id, || {
        if private_regular_file_if_present(root, &pointer_path(root, session_id))?.is_some()
            || private_regular_file_if_present(root, &marker_path(root, session_id))?.is_some()
        {
            return Err(BrainError::Integrity(
                "adaptive_learning_session_id_already_reserved".into(),
            ));
        }
        let target_digest = target_digest(target)?;
        let policy_digest = policy_digest(policy)?;
        let session_id = SessionId::parse(session_id)?;
        let marker = json!({
            "schema":"cerebro.tidex.adaptive_learning_session_marker/v1",
            "session_id":&session_id,
            "target_digest":&target_digest,
            "policy_digest":&policy_digest,
        });
        write_new_private(
            root,
            &marker_path(root, session_id.as_str()),
            &serde_json::to_vec_pretty(&marker)?,
        )?;
        let session = start_adaptive_learning(target, policy)?;
        let cycle = AdaptiveLearningCycle {
            schema: "cerebro.tidex.adaptive_learning_cycle/v1".into(),
            session_id: session_id.clone(),
            target: target.clone(),
            target_digest: target_digest.clone(),
            policy_digest: policy_digest.clone(),
            session,
            pending_step: None,
            completed_evidence: Vec::new(),
            completed_evidence_sha256: Vec::new(),
        };
        persist_receipt_under_root(
            root,
            AdaptiveLearningReceipt {
                schema: "cerebro.tidex.adaptive_learning_receipt/v1".into(),
                event_kind: AdaptiveLearningEventKind::SessionStarted,
                generation: 0,
                session_id,
                target_digest,
                policy_digest,
                prior_receipt_sha256: None,
                evidence_sha256: None,
                cycle,
            },
        )
    })
}

/// Start the only persistent, receipt-backed adaptive-learning lifecycle.
pub fn start_persistent_adaptive_learning(
    root: impl AsRef<Path>,
    session_id: &str,
    target: &LearningTarget,
    policy: &AdaptiveLearningPolicy,
) -> BrainResult<LoadedAdaptiveLearningReceipt> {
    let root = verify_private_root(root.as_ref())?;
    start_persistent_adaptive_learning_under_root(&root, session_id, target, policy)
}

fn issue_next_persistent_learning_aperture_under_root(
    root: &Path,
    session_id: &str,
) -> BrainResult<LoadedAdaptiveLearningReceipt> {
    validate_learning_state_topology(root, session_id)?;
    with_session_lock(root, session_id, || {
        let current = load_current_receipt_under_root(root, session_id)?;
        if current.receipt.cycle.pending_step.is_some() {
            return Err(BrainError::Integrity(
                "adaptive_learning_pending_aperture_must_be_assimilated".into(),
            ));
        }
        let step = next_learning_aperture(
            &current.receipt.cycle.target,
            &current.receipt.cycle.session,
        )?;
        let mut cycle = current.receipt.cycle.clone();
        cycle.pending_step = Some(step);
        persist_receipt_under_root(
            root,
            AdaptiveLearningReceipt {
                schema: "cerebro.tidex.adaptive_learning_receipt/v1".into(),
                event_kind: AdaptiveLearningEventKind::ApertureIssued,
                generation: current.receipt.generation.saturating_add(1),
                session_id: current.receipt.session_id.clone(),
                target_digest: current.receipt.target_digest.clone(),
                policy_digest: current.receipt.policy_digest.clone(),
                prior_receipt_sha256: Some(current.receipt_sha256),
                evidence_sha256: None,
                cycle,
            },
        )
    })
}

/// Atomically issue exactly one canonical next aperture. A second issue is
/// rejected until an actual evidence envelope has been assimilated.
pub fn issue_next_persistent_learning_aperture(
    root: impl AsRef<Path>,
    session_id: &str,
) -> BrainResult<LoadedAdaptiveLearningReceipt> {
    let root = verify_private_root(root.as_ref())?;
    issue_next_persistent_learning_aperture_under_root(&root, session_id)
}

fn read_experiment_evidence_from_path(
    root: &Path,
    path: &Path,
) -> BrainResult<(LearningExperimentEvidence, Vec<u8>, LearningEvidenceDigest)> {
    let raw = read_existing_private_file_bounded(root, path, MAX_LEARNING_AUTHORITY_JSON_BYTES)?;
    let digest = LearningEvidenceDigest::from(Sha256Digest::digest_bytes(&raw));
    let evidence: LearningExperimentEvidence = serde_json::from_slice(&raw)?;
    Ok((evidence, raw, digest))
}

fn persist_experiment_evidence(
    root: &Path,
    digest: &LearningEvidenceDigest,
    raw: &[u8],
) -> BrainResult<()> {
    if Sha256Digest::digest_bytes(raw) != *digest.as_digest() {
        return Err(BrainError::Integrity(
            "adaptive_learning_evidence_content_digest_invalid".into(),
        ));
    }
    let path = experiment_evidence_path(root, digest.as_str());
    if private_regular_file_if_present(root, &path)?.is_some() {
        let _ = confined_existing_file(root, &path, Some(digest.as_str())).map_err(|_| {
            BrainError::Integrity("adaptive_learning_evidence_artifact_collision".into())
        })?;
        return Ok(());
    }
    write_new_private(root, &path, raw)
}

fn assimilate_persistent_learning_evidence_under_root(
    root: &Path,
    session_id: &str,
    evidence_path: &Path,
) -> BrainResult<LoadedAdaptiveLearningReceipt> {
    validate_learning_state_topology(root, session_id)?;
    // Authenticate the caller-provided experiment envelope before obtaining a
    // lock, so a hostile evidence path cannot create any learning state.
    let _ =
        read_existing_private_file_bounded(root, evidence_path, MAX_LEARNING_AUTHORITY_JSON_BYTES)?;
    with_session_lock(root, session_id, || {
        let current = load_current_receipt_under_root(root, session_id)?;
        let pending =
            current.receipt.cycle.pending_step.as_ref().ok_or_else(|| {
                BrainError::Integrity("adaptive_learning_no_pending_aperture".into())
            })?;
        let (evidence, raw, evidence_digest) =
            read_experiment_evidence_from_path(root, evidence_path)?;
        validate_experiment_evidence(
            root,
            &evidence,
            &current.receipt.session_id,
            &current.receipt.target_digest,
            &pending.aperture_id,
            &pending.capability_weights,
        )?;
        if current
            .receipt
            .cycle
            .completed_evidence_sha256
            .iter()
            .any(|digest| digest == &evidence_digest)
        {
            return Err(BrainError::Integrity(
                "adaptive_learning_evidence_already_assimilated".into(),
            ));
        }
        persist_experiment_evidence(root, &evidence_digest, &raw)?;
        let updated_session = assimilate_learning_result(
            &current.receipt.cycle.target,
            &current.receipt.cycle.session,
            pending,
            evidence.observed_value,
        )?;
        let mut cycle = current.receipt.cycle.clone();
        cycle.session = updated_session;
        cycle.pending_step = None;
        cycle.completed_evidence.push(evidence);
        cycle
            .completed_evidence_sha256
            .push(evidence_digest.clone());
        persist_receipt_under_root(
            root,
            AdaptiveLearningReceipt {
                schema: "cerebro.tidex.adaptive_learning_receipt/v1".into(),
                event_kind: AdaptiveLearningEventKind::ResultAssimilated,
                generation: current.receipt.generation.saturating_add(1),
                session_id: current.receipt.session_id.clone(),
                target_digest: current.receipt.target_digest.clone(),
                policy_digest: current.receipt.policy_digest.clone(),
                prior_receipt_sha256: Some(current.receipt_sha256),
                evidence_sha256: Some(evidence_digest),
                cycle,
            },
        )
    })
}

/// Assimilate a real, content-addressed experiment evidence envelope. Missing,
/// changed or out-of-root evidence is a hard error; no outcome is synthesized.
pub fn assimilate_persistent_learning_evidence(
    root: impl AsRef<Path>,
    session_id: &str,
    evidence_path: impl AsRef<Path>,
) -> BrainResult<LoadedAdaptiveLearningReceipt> {
    let root = verify_private_root(root.as_ref())?;
    assimilate_persistent_learning_evidence_under_root(&root, session_id, evidence_path.as_ref())
}

pub fn load_persistent_adaptive_learning_receipt(
    root: impl AsRef<Path>,
    session_id: &str,
) -> BrainResult<LoadedAdaptiveLearningReceipt> {
    let root = verify_private_root(root.as_ref())?;
    load_current_receipt_under_root(&root, session_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::create_content_addressed_dvec;
    use crate::contracts::ExperimentLineage;
    use crate::digest::Sha256Digest;
    use crate::security::secure_dir;
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn policy(weight: f64) -> AdaptiveLearningPolicy {
        AdaptiveLearningPolicy {
            schema: "cerebro.tidex.adaptive_learning_policy/v1".into(),
            outcome_utility_weight: weight,
            maximize_observed_value: true,
        }
    }

    #[test]
    fn versioned_learning_policy_rejects_unknown_wire_fields() {
        let wire = serde_json::json!({
            "schema": "cerebro.tidex.adaptive_learning_policy/v1",
            "outcome_utility_weight": 1.0,
            "maximize_observed_value": true,
            "unreviewed_override": true
        });
        assert!(serde_json::from_value::<AdaptiveLearningPolicy>(wire).is_err());
    }

    fn adaptive_target() -> LearningTarget {
        LearningTarget {
            target_id: LearningTargetId::parse("adaptive").unwrap(),
            capability_ids: ["a", "b", "c"]
                .into_iter()
                .map(|id| CapabilityId::parse(id).unwrap())
                .collect(),
            candidate_budget: 32,
            plan_steps: 6,
            noise_variance: 0.1,
            cost_weight: 0.0,
            risk_weight: 0.0,
        }
    }

    fn temporary_private_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cerebro-tidex-learning-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        secure_dir(&root).unwrap();
        root
    }

    fn write_test_evidence(
        root: &Path,
        session_id: &str,
        target_digest: &LearningTargetDigest,
        step: &AdaptiveLearningStep,
    ) -> PathBuf {
        let layout_raw = br#"{\"schema\":\"cerebro.tidex.parameter_layout/test\"}"#.to_vec();
        let layout_digest = sha256_bytes(&layout_raw);
        let layout_path = root
            .join("state/parameter_layouts/by-sha")
            .join(format!("{layout_digest}.json"));
        write_new_private(root, &layout_path, &layout_raw).unwrap();
        let dense = create_content_addressed_dvec(root, &[0.25, -0.5]).unwrap();
        let functional_response = (0..step.capability_weights.len())
            .map(|index| 0.2 + index as f64 * 0.15)
            .collect::<Vec<_>>();
        let observation = DeltaObservation {
            observation_id: crate::identity::ObservationId::parse(format!(
                "observation-{}",
                step.aperture_id
            ))
            .unwrap(),
            from_checkpoint: "checkpoint-before".into(),
            to_checkpoint: "checkpoint-after".into(),
            generation: 1,
            delta: vec![0.25, -0.5],
            functional_response,
            confounders: Vec::new(),
            reliability: 1.0,
            independence_group: step.aperture_id.to_string(),
            experiment_lineage: ExperimentLineage {
                run_id: format!("run-{}", step.aperture_id),
                replicate_id: "replicate-0".into(),
                randomization_id: "randomization-0".into(),
                dataset_split_digest: "a".repeat(64),
                initial_checkpoint_digest: "b".repeat(64),
                optimizer_config_digest: "c".repeat(64),
                template_config_digest: "d".repeat(64),
            },
            dense_artifact: Some(dense),
            parameter_layout_sha256: Some(Sha256Digest::parse(layout_digest).unwrap()),
            representation_artifact: None,
            representation_protocol_sha256: None,
            provenance_digest: crate::digest::ProvenanceDigest::from(
                Sha256Digest::parse("e".repeat(64)).unwrap(),
            ),
        };
        let observation_raw = serde_json::to_vec_pretty(&observation).unwrap();
        let observation_path = root
            .join("state/observations")
            .join(format!("{}.json", observation.observation_id));
        write_new_private(root, &observation_path, &observation_raw).unwrap();
        let support_path = root
            .join("state/experiment_support")
            .join(format!("{}.json", step.aperture_id));
        let support_raw = br#"{\"measurement\":\"sealed\"}"#.to_vec();
        write_new_private(root, &support_path, &support_raw).unwrap();
        let evidence = LearningExperimentEvidence {
            schema: "cerebro.tidex.learning_experiment_evidence/v1".into(),
            session_id: SessionId::parse(session_id).unwrap(),
            target_digest: target_digest.clone(),
            aperture_id: step.aperture_id.clone(),
            observed_value: dot(&step.capability_weights, &observation.functional_response)
                .unwrap(),
            observation_id: ObservationId::parse(&observation.observation_id).unwrap(),
            observation: EvidenceReference {
                path: observation_path.clone(),
                sha256: Sha256Digest::parse(sha256_bytes(&observation_raw)).unwrap(),
            },
            evidence_files: vec![EvidenceReference {
                path: support_path.clone(),
                sha256: Sha256Digest::parse(sha256_bytes(&support_raw)).unwrap(),
            }],
        };
        let evidence_path = root
            .join("state/experiment_envelopes")
            .join(format!("{}.json", step.aperture_id));
        write_new_private(
            root,
            &evidence_path,
            &serde_json::to_vec_pretty(&evidence).unwrap(),
        )
        .unwrap();
        evidence_path
    }

    #[test]
    fn learning_plan_is_full_rank_and_covers_every_target() {
        let target = LearningTarget {
            target_id: LearningTargetId::parse("producer-test").unwrap(),
            capability_ids: (0..10)
                .map(|index| CapabilityId::parse(format!("f{index}")).unwrap())
                .collect(),
            candidate_budget: 192,
            plan_steps: 30,
            noise_variance: 0.08,
            cost_weight: 0.02,
            risk_weight: 0.02,
        };
        let plan = plan_autonomous_learning(&target).unwrap();
        assert_eq!(plan.design_rank, 10);
        assert_eq!(plan.apertures.len(), 30);
        assert!(plan.minimum_capability_coverage >= 3);
        assert!(plan.plan.final_posterior_trace < plan.plan.initial_posterior_trace);
    }

    #[test]
    fn learning_plan_is_deterministic_for_same_target() {
        let target = LearningTarget {
            target_id: LearningTargetId::parse("stable").unwrap(),
            capability_ids: ["a", "b", "c", "d"]
                .into_iter()
                .map(|id| CapabilityId::parse(id).unwrap())
                .collect(),
            candidate_budget: 64,
            plan_steps: 12,
            noise_variance: 0.1,
            cost_weight: 0.0,
            risk_weight: 0.0,
        };
        assert_eq!(
            plan_autonomous_learning(&target).unwrap(),
            plan_autonomous_learning(&target).unwrap()
        );
    }
    #[test]
    fn adaptive_learning_assimilates_realized_result_before_next_choice() {
        let target = adaptive_target();
        let session = start_adaptive_learning(&target, &policy(10.0)).unwrap();
        let first = next_learning_aperture(&target, &session).unwrap();
        let prior_trace = session
            .posterior
            .covariance
            .iter()
            .enumerate()
            .map(|(index, row)| row[index])
            .sum::<f64>();
        let updated = assimilate_learning_result(&target, &session, &first, 0.75).unwrap();
        let next = next_learning_aperture(&target, &updated).unwrap();
        let posterior_trace = updated
            .posterior
            .covariance
            .iter()
            .enumerate()
            .map(|(index, row)| row[index])
            .sum::<f64>();
        assert!(posterior_trace < prior_trace);
        assert_eq!(
            updated.completed_aperture_ids,
            vec![first.aperture_id.clone()]
        );
        assert_ne!(first.aperture_id, next.aperture_id);
        assert!(updated
            .posterior
            .mean
            .iter()
            .any(|value| value.abs() > 1e-12));
    }

    #[test]
    fn realized_outcome_changes_the_next_canonical_aperture() {
        let target = adaptive_target();
        let session = start_adaptive_learning(&target, &policy(100.0)).unwrap();
        let first = next_learning_aperture(&target, &session).unwrap();
        let favorable = assimilate_learning_result(&target, &session, &first, 1.0).unwrap();
        let unfavorable = assimilate_learning_result(&target, &session, &first, -1.0).unwrap();
        let favorable_next = next_learning_aperture(&target, &favorable).unwrap();
        let unfavorable_next = next_learning_aperture(&target, &unfavorable).unwrap();
        assert_ne!(favorable_next.aperture_id, unfavorable_next.aperture_id);
        assert!(favorable_next.outcome_utility > 0.0);
        assert!(unfavorable_next.outcome_utility >= 0.0);
    }

    #[test]
    fn persistent_cycle_rejects_unbound_outcomes_and_tampered_receipts() {
        let root = temporary_private_root("persistent");
        let target = adaptive_target();
        let started = start_persistent_adaptive_learning_under_root(
            &root,
            "session-persistent",
            &target,
            &policy(10.0),
        )
        .unwrap();
        let issued =
            issue_next_persistent_learning_aperture_under_root(&root, "session-persistent")
                .unwrap();
        assert!(
            issue_next_persistent_learning_aperture_under_root(&root, "session-persistent")
                .is_err()
        );
        let step = issued.receipt.cycle.pending_step.as_ref().unwrap();
        let evidence_path = write_test_evidence(
            &root,
            "session-persistent",
            &started.receipt.target_digest,
            step,
        );
        let evidence_raw = fs::read(&evidence_path).unwrap();
        let mut malformed: LearningExperimentEvidence =
            serde_json::from_slice(&evidence_raw).unwrap();
        let mut invalid_wire: serde_json::Value = serde_json::from_slice(&evidence_raw).unwrap();
        invalid_wire["observation_id"] = serde_json::Value::String("../outside-observation".into());
        fs::write(
            &evidence_path,
            serde_json::to_vec_pretty(&invalid_wire).unwrap(),
        )
        .unwrap();
        assert!(assimilate_persistent_learning_evidence_under_root(
            &root,
            "session-persistent",
            &evidence_path,
        )
        .is_err());
        malformed.observed_value += 1.0;
        fs::write(
            &evidence_path,
            serde_json::to_vec_pretty(&malformed).unwrap(),
        )
        .unwrap();
        assert!(assimilate_persistent_learning_evidence_under_root(
            &root,
            "session-persistent",
            &evidence_path,
        )
        .is_err());
        malformed.observed_value -= 1.0;
        fs::write(
            &evidence_path,
            serde_json::to_vec_pretty(&malformed).unwrap(),
        )
        .unwrap();
        let assimilated = assimilate_persistent_learning_evidence_under_root(
            &root,
            "session-persistent",
            &evidence_path,
        )
        .unwrap();
        assert_eq!(assimilated.receipt.cycle.completed_evidence.len(), 1);
        assert!(load_current_receipt_under_root(&root, "session-persistent").is_ok());
        fs::write(pointer_path(&root, "session-persistent"), b"{}\n").unwrap();
        assert!(load_current_receipt_under_root(&root, "session-persistent").is_err());
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn persistent_start_rejects_symlinked_learning_sessions_without_mutation() {
        let root = temporary_private_root("symlink-learning-sessions");
        let outside = temporary_private_root("symlink-learning-sessions-outside");
        fs::create_dir(root.join("state")).unwrap();
        symlink(&outside, root.join("state/learning_sessions")).unwrap();

        assert!(start_persistent_adaptive_learning_under_root(
            &root,
            "session-symlink-sessions",
            &adaptive_target(),
            &policy(10.0),
        )
        .is_err());
        assert!(fs::read_dir(&outside).unwrap().next().is_none());

        fs::remove_dir_all(&root).unwrap();
        fs::remove_dir_all(&outside).unwrap();
    }

    #[test]
    fn session_lock_rejects_symlinked_locks_without_mutation() {
        let root = temporary_private_root("symlink-learning-locks");
        let outside = temporary_private_root("symlink-learning-locks-outside");
        fs::create_dir_all(root.join("state/learning_sessions")).unwrap();
        symlink(&outside, root.join("state/learning_sessions/locks")).unwrap();

        assert!(with_session_lock(&root, "session-symlink-locks", || Ok(())).is_err());
        assert!(fs::read_dir(&outside).unwrap().next().is_none());

        fs::remove_dir_all(&root).unwrap();
        fs::remove_dir_all(&outside).unwrap();
    }

    #[test]
    fn current_pointer_symlink_is_rejected_without_external_mutation() {
        let root = temporary_private_root("symlink-learning-current");
        let outside = temporary_private_root("symlink-learning-current-outside");
        let session_id = "session-symlink-current";
        start_persistent_adaptive_learning_under_root(
            &root,
            session_id,
            &adaptive_target(),
            &policy(10.0),
        )
        .unwrap();
        let pointer = pointer_path(&root, session_id);
        let preserved = outside.join("current-pointer.json");
        fs::rename(&pointer, &preserved).unwrap();
        let original = fs::read(&preserved).unwrap();
        symlink(&preserved, &pointer).unwrap();

        assert!(load_current_receipt_under_root(&root, session_id).is_err());
        assert_eq!(fs::read(&preserved).unwrap(), original);

        fs::remove_dir_all(&root).unwrap();
        fs::remove_dir_all(&outside).unwrap();
    }
}
