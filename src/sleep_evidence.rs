use crate::artifact::{read_dvec_f32, DeltaArtifactRef};
use crate::authority::{
    existing_regular_file_under_root, read_existing_private_file_bounded, PrivateFileReference,
};
use crate::causal_credit::{
    certified_causal_priority_weights, estimate_causal_credit, CausalCreditReport,
    CounterfactualEvaluation,
};
use crate::contracts::{DeltaObservation, SkillField};
use crate::digest::{
    AnalysisVersionDigest, CausalCreditDigest, ConfigDigest, CorpusDigest, ProtectedMapDigest,
    ReportDigest, Sha256Digest, SourceTreeDigest,
};
use crate::error::{BrainError, BrainResult};
use crate::identity::SkillId;
use crate::interaction::second_order_interactions;
use crate::linalg::Matrix;
use crate::protected_map::{
    build_protected_cortex_map, load_protected_cortex, ProtectedMapArtifactReport,
    SensitivityEvidence,
};
use crate::trust_region::{apply_causal_priority_trust_region, TrustRegionAllocationPolicy};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const NONINFERIORITY_95_Z: f64 = 1.959_963_984_540_054;
const MAX_SLEEP_EVIDENCE_JSON_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProtectionEvidenceSummary {
    pub source_path: String,
    /// Physical byte identity of a heterogeneous external evidence file.
    pub source_sha256: Sha256Digest,
    pub protected_map_path: String,
    pub protected_map_sha256: ProtectedMapDigest,
    pub probe_count: usize,
    pub parameter_dimension: usize,
    pub selected_rank: usize,
    pub causal_damage_supported_probes: usize,
    pub sensitivity_damage_correlation: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InteractionEvidenceSummary {
    pub source_path: String,
    /// Physical byte identity; the referenced payload owns its semantic type.
    pub source_sha256: Sha256Digest,
    /// Digest of the exact causal-credit wrapper whose lower confidence bounds
    /// determined the trust-region contraction order.
    pub causal_credit_sha256: CausalCreditDigest,
    pub field_ids: Vec<SkillId>,
    pub dense_parameter_dimension: usize,
    pub proposed_quadratic_cost: f64,
    pub accepted_quadratic_cost: f64,
    pub diagonal_budget: f64,
    pub trust_scale: f64,
    pub allocation_policy: TrustRegionAllocationPolicy,
    pub causal_priority_weights: Vec<f64>,
    pub component_retention: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReplayEvidenceSummary {
    pub baseline_source_path: String,
    /// Physical replay files are heterogeneous across evaluators.
    pub baseline_source_sha256: Sha256Digest,
    pub candidate_source_path: String,
    pub candidate_source_sha256: Sha256Digest,
    pub paired_independent_groups: usize,
    pub mean_utility_delta: f64,
    pub standard_error: f64,
    #[serde(default)]
    pub max_acceptable_utility_loss: Option<f64>,
    pub blind_data_accessed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CausalCreditEvidenceSummary {
    pub replay_source_path: String,
    pub replay_source_sha256: Sha256Digest,
    pub credit_source_path: String,
    pub credit_source_sha256: CausalCreditDigest,
    pub report_sha256: ReportDigest,
    /// Legacy producer plan identity is not yet governed by one canonical
    /// schema, so assigning a stronger semantic domain would be premature.
    pub plan_sha256: Sha256Digest,
    pub independent_group_count: usize,
    pub field_count: usize,
    pub resolved_field_count: usize,
    pub interaction_count: usize,
    pub resolved_interaction_count: usize,
    pub blind_data_accessed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SleepEvidenceBundle {
    pub schema: String,
    pub corpus_digest: CorpusDigest,
    pub source_tree_digest: SourceTreeDigest,
    pub config_digest: ConfigDigest,
    pub analysis_version_digest: AnalysisVersionDigest,
    pub field_ids: Vec<SkillId>,
    pub protection: ProtectionEvidenceSummary,
    pub interaction: InteractionEvidenceSummary,
    pub replay: ReplayEvidenceSummary,
    pub causal_credit: CausalCreditEvidenceSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SleepEvidenceVerification {
    pub schema: String,
    pub verified: bool,
    pub protection_verified: bool,
    pub interaction_verified: bool,
    pub replay_verified: bool,
    pub causal_credit_verified: bool,
    pub replay_zero_effect_z: f64,
    pub replay_lower_confidence_bound: f64,
    pub noninferiority_margin: Option<f64>,
    pub reasons: Vec<String>,
}

pub struct SleepEvidenceExpectation<'a> {
    pub corpus_digest: &'a CorpusDigest,
    /// SHA256 of the exact canonical reconstruction report being certified.
    /// Replay, trust and causal-credit artifacts must all bind to this report;
    /// the evidence bundle is not allowed to self-attest a different digest.
    pub report_sha256: &'a ReportDigest,
    pub source_tree_digest: &'a SourceTreeDigest,
    pub config_digest: &'a ConfigDigest,
    pub analysis_version_digest: &'a AnalysisVersionDigest,
    pub fields: &'a [SkillField],
    pub observations: &'a [DeltaObservation],
    pub source_mixtures: &'a [Vec<f64>],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CounterfactualReplayArtifact {
    schema: String,
    report_sha256: ReportDigest,
    plan_sha256: Sha256Digest,
    field_ids: Vec<SkillId>,
    field_coefficients: Vec<f64>,
    validation_seeds: Vec<u64>,
    validation_tasks: Vec<String>,
    count_per_task: u64,
    blind_data_accessed: bool,
    evaluations: Vec<CounterfactualEvaluation>,
    raw_results: Vec<CounterfactualReplayRow>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CounterfactualReplayRow {
    mask: u64,
    active_fields: Vec<SkillId>,
    seed: u64,
    metrics: BTreeMap<String, f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustRegionFunctionalReplayArtifact {
    schema: String,
    report_sha256: ReportDigest,
    trust_region_sha256: Sha256Digest,
    plan_sha256: Sha256Digest,
    field_ids: Vec<SkillId>,
    accepted_coefficients: Vec<f64>,
    validation_seeds: Vec<u64>,
    validation_tasks: Vec<String>,
    count_per_task: u64,
    blind_data_accessed: bool,
    rows: Vec<TrustRegionFunctionalReplayRow>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustRegionFunctionalReplayRow {
    seed: u64,
    metrics: BTreeMap<String, f64>,
    mean: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
enum CausalPriorityWeightKind {
    #[serde(rename = "causal_lower_confidence_bound_95")]
    LowerConfidenceBound95,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustRegionArtifactIdentity {
    schema: String,
    report_sha256: ReportDigest,
    protected_map_sha256: ProtectedMapDigest,
    causal_plan_sha256: Sha256Digest,
    causal_credit_sha256: CausalCreditDigest,
    causal_priority_weight_kind: CausalPriorityWeightKind,
    field_ids: Vec<SkillId>,
    dense_parameter_dimension: usize,
    interaction_matrix: Vec<Vec<f64>>,
    diagonal_budget: f64,
    trust_region: TrustRegionDecisionIdentity,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustRegionDecisionIdentity {
    proposed_coefficients: Vec<f64>,
    accepted_coefficients: Vec<f64>,
    proposed_quadratic_cost: f64,
    accepted_quadratic_cost: f64,
    max_quadratic_cost: f64,
    scale: f64,
    constrained: bool,
    allocation_policy: TrustRegionAllocationPolicy,
    component_retention: Vec<f64>,
    causal_priority_weights: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct RecomputedReplayStatistics {
    paired_independent_groups: usize,
    mean_utility_delta: f64,
    standard_error: f64,
    max_acceptable_utility_loss: f64,
    zero_effect_z: f64,
    lower_confidence_bound: f64,
}

impl SleepEvidenceVerification {
    pub fn failed(reason: impl Into<String>) -> Self {
        Self {
            schema: "cerebro.tidex.sleep_evidence_verification/v3".into(),
            verified: false,
            protection_verified: false,
            interaction_verified: false,
            replay_verified: false,
            causal_credit_verified: false,
            replay_zero_effect_z: f64::INFINITY,
            replay_lower_confidence_bound: f64::NEG_INFINITY,
            noninferiority_margin: None,
            reasons: vec![reason.into()],
        }
    }
}

/// Resolve evidence only from an existing regular file below the private root.
///
/// Evidence bundles are untrusted input until every path they name has passed
/// this check.  In particular, checking only a canonicalized leaf would allow
/// a symlinked intermediate directory to redirect a supposedly private
/// artifact.  The shared authority helper rejects symlinks in every component
/// and rejects non-regular leaf objects before any evidence bytes are read.
fn confined_path(root: &Path, raw: impl AsRef<Path>) -> BrainResult<PathBuf> {
    let root_metadata = fs::symlink_metadata(root)?;
    if root_metadata.file_type().is_symlink() || !root_metadata.file_type().is_dir() {
        return Err(BrainError::Integrity(
            "sleep_evidence_private_root_invalid".into(),
        ));
    }
    let path = raw.as_ref();
    if !path.is_absolute() {
        return Err(BrainError::Invalid(
            "sleep_evidence_path_not_absolute".into(),
        ));
    }
    existing_regular_file_under_root(root, path)
}

fn verify_file(root: &Path, path: &str, expected_sha: &impl AsRef<str>) -> BrainResult<bool> {
    let path = confined_path(root, path)?;
    let expected = Sha256Digest::parse(expected_sha.as_ref())?;
    match PrivateFileReference::new(path, expected)
        .read_verified_bounded(root, MAX_SLEEP_EVIDENCE_JSON_BYTES)
    {
        Ok(_) => Ok(true),
        Err(BrainError::Integrity(message))
            if message == "private_file_reference_digest_mismatch" =>
        {
            Ok(false)
        }
        Err(error) => Err(error),
    }
}

fn valid_sha256(value: &impl AsRef<str>) -> bool {
    let value = value.as_ref();
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn finite_scalar_match(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= 64.0 * f64::EPSILON * (1.0 + left.abs().max(right.abs()))
}

fn finite_vector_match(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| finite_scalar_match(*left, *right))
}

fn exact_unique_nonempty(values: &[String]) -> bool {
    !values.is_empty()
        && values.iter().all(|value| !value.trim().is_empty())
        && values.iter().collect::<BTreeSet<_>>().len() == values.len()
}

fn exact_unique_seeds(values: &[u64]) -> bool {
    values.len() >= 3 && values.iter().collect::<BTreeSet<_>>().len() == values.len()
}

fn exact_metric_schema(metrics: &BTreeMap<String, f64>, tasks: &[String]) -> bool {
    metrics.len() == tasks.len()
        && tasks.iter().all(|task| metrics.contains_key(task))
        && metrics.values().all(|value| value.is_finite())
}

fn mean_metrics(metrics: &BTreeMap<String, f64>, tasks: &[String]) -> BrainResult<f64> {
    if !exact_metric_schema(metrics, tasks) {
        return Err(BrainError::Integrity(
            "functional_replay_metric_schema_invalid".into(),
        ));
    }
    let sum = tasks.iter().try_fold(0.0f64, |total, task| {
        let value = metrics
            .get(task)
            .copied()
            .ok_or_else(|| BrainError::Integrity("functional_replay_metric_missing".into()))?;
        let next = total + value;
        if !next.is_finite() {
            return Err(BrainError::Numerical(
                "functional_replay_metric_sum_nonfinite".into(),
            ));
        }
        Ok(next)
    })?;
    Ok(sum / tasks.len() as f64)
}

fn expected_full_mask(field_count: usize) -> BrainResult<u64> {
    if field_count == 0 || field_count >= u64::BITS as usize {
        return Err(BrainError::Integrity(
            "functional_replay_field_count_invalid".into(),
        ));
    }
    Ok((1u64 << field_count) - 1)
}

fn checked_replay_identity(
    root: &Path,
    bundle: &SleepEvidenceBundle,
    expected: &SleepEvidenceExpectation<'_>,
    expected_ids: &[SkillId],
    baseline: &CounterfactualReplayArtifact,
    candidate: &TrustRegionFunctionalReplayArtifact,
) -> BrainResult<()> {
    if !valid_sha256(expected.report_sha256)
        || !valid_sha256(&bundle.causal_credit.report_sha256)
        || !valid_sha256(&bundle.causal_credit.plan_sha256)
        || baseline.schema != "cerebro.tidex.counterfactual_replay/v3"
        || candidate.schema != "cerebro.tidex.trust_region_functional_validation/v2"
        || baseline.blind_data_accessed
        || candidate.blind_data_accessed
        || bundle.replay.blind_data_accessed
        || baseline.report_sha256.as_str() != expected.report_sha256.as_str()
        || candidate.report_sha256.as_str() != expected.report_sha256.as_str()
        || bundle.causal_credit.report_sha256.as_str() != expected.report_sha256.as_str()
        || baseline.plan_sha256 != candidate.plan_sha256
        || baseline.plan_sha256 != bundle.causal_credit.plan_sha256
        || baseline.field_ids.as_slice() != expected_ids
        || candidate.field_ids.as_slice() != expected_ids
        || expected_ids.iter().collect::<BTreeSet<_>>().len() != expected_ids.len()
        || !exact_unique_seeds(&baseline.validation_seeds)
        || baseline.validation_seeds != candidate.validation_seeds
        || !exact_unique_nonempty(&baseline.validation_tasks)
        || baseline.validation_tasks != candidate.validation_tasks
        || baseline.count_per_task == 0
        || baseline.count_per_task != candidate.count_per_task
        || baseline.field_coefficients.len() != expected_ids.len()
        || candidate.accepted_coefficients.len() != expected_ids.len()
        || baseline
            .field_coefficients
            .iter()
            .chain(&candidate.accepted_coefficients)
            .any(|value| !value.is_finite())
        || candidate.trust_region_sha256 != bundle.interaction.source_sha256
    {
        return Err(BrainError::Integrity(
            "functional_replay_identity_contract_invalid".into(),
        ));
    }

    let trust_raw = PrivateFileReference::new(
        PathBuf::from(&bundle.interaction.source_path),
        bundle.interaction.source_sha256.clone(),
    )
    .read_verified_bounded(root, MAX_SLEEP_EVIDENCE_JSON_BYTES)?;
    let trust: TrustRegionArtifactIdentity = serde_json::from_slice(&trust_raw)?;
    if trust.schema != "cerebro.tidex.trust_region_benchmark/v3"
        || trust.report_sha256.as_str() != expected.report_sha256.as_str()
        || trust.protected_map_sha256 != bundle.protection.protected_map_sha256
        || trust.causal_plan_sha256 != baseline.plan_sha256
        || trust.causal_credit_sha256 != bundle.causal_credit.credit_source_sha256
        || trust.causal_credit_sha256 != bundle.interaction.causal_credit_sha256
        || !valid_sha256(&trust.causal_credit_sha256)
        || !valid_sha256(&bundle.interaction.causal_credit_sha256)
        || trust.causal_priority_weight_kind != CausalPriorityWeightKind::LowerConfidenceBound95
        || trust.field_ids.as_slice() != expected_ids
        || trust.dense_parameter_dimension != bundle.interaction.dense_parameter_dimension
        || !finite_scalar_match(trust.diagonal_budget, bundle.interaction.diagonal_budget)
        || trust.interaction_matrix.len() != expected_ids.len()
        || trust.interaction_matrix.iter().any(|row| {
            row.len() != expected_ids.len() || row.iter().any(|value| !value.is_finite())
        })
        || trust.trust_region.proposed_coefficients.len() != expected_ids.len()
        || trust.trust_region.accepted_coefficients.len() != expected_ids.len()
        || trust.trust_region.component_retention.len() != expected_ids.len()
        || trust.trust_region.causal_priority_weights.len() != expected_ids.len()
        || trust.trust_region.allocation_policy
            != TrustRegionAllocationPolicy::CausalPriorityContractionV1
        || trust.trust_region.allocation_policy != bundle.interaction.allocation_policy
        || !finite_vector_match(
            &trust.trust_region.causal_priority_weights,
            &bundle.interaction.causal_priority_weights,
        )
        || !finite_vector_match(
            &trust.trust_region.component_retention,
            &bundle.interaction.component_retention,
        )
        || !trust.trust_region.max_quadratic_cost.is_finite()
        || trust
            .trust_region
            .component_retention
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || trust
            .trust_region
            .causal_priority_weights
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || trust
            .trust_region
            .proposed_coefficients
            .iter()
            .zip(&baseline.field_coefficients)
            .any(|(left, right)| !finite_scalar_match(*left, *right))
        || trust
            .trust_region
            .accepted_coefficients
            .iter()
            .zip(&candidate.accepted_coefficients)
            .any(|(left, right)| !finite_scalar_match(*left, *right))
    {
        return Err(BrainError::Integrity(
            "functional_replay_trust_identity_mismatch".into(),
        ));
    }
    Ok(())
}

fn baseline_full_coalition_rows(
    baseline: &CounterfactualReplayArtifact,
    expected_ids: &[SkillId],
) -> BrainResult<BTreeMap<u64, BTreeMap<String, f64>>> {
    let mask = expected_full_mask(expected_ids.len())?;
    let expected_seeds = baseline
        .validation_seeds
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut rows = BTreeMap::new();
    for row in &baseline.raw_results {
        if row.mask != mask {
            continue;
        }
        if row.active_fields.as_slice() != expected_ids
            || !expected_seeds.contains(&row.seed)
            || !exact_metric_schema(&row.metrics, &baseline.validation_tasks)
            || rows.insert(row.seed, row.metrics.clone()).is_some()
        {
            return Err(BrainError::Integrity(
                "functional_replay_full_coalition_row_invalid".into(),
            ));
        }
    }
    if rows.len() != expected_seeds.len()
        || rows.keys().copied().collect::<BTreeSet<_>>() != expected_seeds
    {
        return Err(BrainError::Integrity(
            "functional_replay_full_coalition_coverage_incomplete".into(),
        ));
    }
    Ok(rows)
}

fn verify_full_coalition_evaluations(
    baseline: &CounterfactualReplayArtifact,
    expected_ids: &[SkillId],
    full_rows: &BTreeMap<u64, BTreeMap<String, f64>>,
) -> BrainResult<()> {
    let mut expected = BTreeMap::<(u64, String), f64>::new();
    for (seed, metrics) in full_rows {
        for task in &baseline.validation_tasks {
            let utility = metrics.get(task).copied().ok_or_else(|| {
                BrainError::Integrity("functional_replay_full_metric_missing".into())
            })?;
            expected.insert((*seed, task.clone()), utility);
        }
    }
    let mut observed = BTreeMap::<(u64, String), f64>::new();
    for evaluation in &baseline.evaluations {
        if evaluation.active_fields.as_slice() != expected_ids {
            continue;
        }
        if !evaluation.utility.is_finite() {
            return Err(BrainError::Integrity(
                "functional_replay_evaluation_nonfinite".into(),
            ));
        }
        let Some((seed_text, task)) = evaluation.context_id.split_once(':') else {
            return Err(BrainError::Integrity(
                "functional_replay_evaluation_context_invalid".into(),
            ));
        };
        let seed = seed_text
            .strip_prefix("validation-seed-")
            .ok_or_else(|| {
                BrainError::Integrity("functional_replay_evaluation_context_invalid".into())
            })?
            .parse::<u64>()
            .map_err(|_| {
                BrainError::Integrity("functional_replay_evaluation_context_invalid".into())
            })?;
        if evaluation.independence_group != format!("validation-seed-{seed}")
            || !expected.contains_key(&(seed, task.to_string()))
            || observed
                .insert((seed, task.to_string()), evaluation.utility)
                .is_some()
        {
            return Err(BrainError::Integrity(
                "functional_replay_evaluation_identity_invalid".into(),
            ));
        }
    }
    if observed.len() != expected.len()
        || expected.iter().any(|(key, value)| {
            observed
                .get(key)
                .is_none_or(|observed_value| !finite_scalar_match(*value, *observed_value))
        })
    {
        return Err(BrainError::Integrity(
            "functional_replay_raw_evaluation_mismatch".into(),
        ));
    }
    Ok(())
}

fn recompute_functional_replay(
    root: &Path,
    bundle: &SleepEvidenceBundle,
    expected: &SleepEvidenceExpectation<'_>,
    expected_ids: &[SkillId],
) -> BrainResult<RecomputedReplayStatistics> {
    if !verify_file(
        root,
        &bundle.replay.baseline_source_path,
        &bundle.replay.baseline_source_sha256,
    )? || !verify_file(
        root,
        &bundle.replay.candidate_source_path,
        &bundle.replay.candidate_source_sha256,
    )? || !verify_file(
        root,
        &bundle.interaction.source_path,
        &bundle.interaction.source_sha256,
    )? {
        return Err(BrainError::Integrity(
            "functional_replay_source_digest_mismatch".into(),
        ));
    }
    let baseline_raw = PrivateFileReference::new(
        PathBuf::from(&bundle.replay.baseline_source_path),
        bundle.replay.baseline_source_sha256.clone(),
    )
    .read_verified_bounded(root, MAX_SLEEP_EVIDENCE_JSON_BYTES)?;
    let candidate_raw = PrivateFileReference::new(
        PathBuf::from(&bundle.replay.candidate_source_path),
        bundle.replay.candidate_source_sha256.clone(),
    )
    .read_verified_bounded(root, MAX_SLEEP_EVIDENCE_JSON_BYTES)?;
    let baseline: CounterfactualReplayArtifact = serde_json::from_slice(&baseline_raw)?;
    let candidate: TrustRegionFunctionalReplayArtifact = serde_json::from_slice(&candidate_raw)?;
    checked_replay_identity(root, bundle, expected, expected_ids, &baseline, &candidate)?;

    let baseline_rows = baseline_full_coalition_rows(&baseline, expected_ids)?;
    verify_full_coalition_evaluations(&baseline, expected_ids, &baseline_rows)?;
    let expected_seeds = baseline
        .validation_seeds
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut candidate_rows = BTreeMap::new();
    for row in &candidate.rows {
        if !expected_seeds.contains(&row.seed)
            || !exact_metric_schema(&row.metrics, &baseline.validation_tasks)
            || !finite_scalar_match(
                row.mean,
                mean_metrics(&row.metrics, &baseline.validation_tasks)?,
            )
            || candidate_rows
                .insert(row.seed, row.metrics.clone())
                .is_some()
        {
            return Err(BrainError::Integrity(
                "functional_replay_candidate_row_invalid".into(),
            ));
        }
    }
    if candidate_rows.len() != expected_seeds.len()
        || candidate_rows.keys().copied().collect::<BTreeSet<_>>() != expected_seeds
    {
        return Err(BrainError::Integrity(
            "functional_replay_candidate_coverage_incomplete".into(),
        ));
    }

    let mut differences = Vec::with_capacity(expected_seeds.len());
    for seed in &baseline.validation_seeds {
        let baseline_mean = mean_metrics(
            baseline_rows.get(seed).ok_or_else(|| {
                BrainError::Integrity("functional_replay_baseline_seed_missing".into())
            })?,
            &baseline.validation_tasks,
        )?;
        let candidate_mean = mean_metrics(
            candidate_rows.get(seed).ok_or_else(|| {
                BrainError::Integrity("functional_replay_candidate_seed_missing".into())
            })?,
            &baseline.validation_tasks,
        )?;
        let difference = candidate_mean - baseline_mean;
        if !difference.is_finite() {
            return Err(BrainError::Numerical(
                "functional_replay_difference_nonfinite".into(),
            ));
        }
        differences.push(difference);
    }
    let count = differences.len();
    let mean_utility_delta = differences.iter().sum::<f64>() / count as f64;
    if !mean_utility_delta.is_finite() {
        return Err(BrainError::Numerical(
            "functional_replay_mean_nonfinite".into(),
        ));
    }
    let sample_variance = differences
        .iter()
        .map(|difference| (difference - mean_utility_delta).powi(2))
        .sum::<f64>()
        / (count - 1) as f64;
    let standard_error = (sample_variance / count as f64).sqrt();
    let max_acceptable_utility_loss = 1.0 / baseline.count_per_task as f64;
    if !standard_error.is_finite()
        || !max_acceptable_utility_loss.is_finite()
        || max_acceptable_utility_loss <= 0.0
    {
        return Err(BrainError::Numerical(
            "functional_replay_statistics_invalid".into(),
        ));
    }
    let zero_effect_z = if standard_error > 0.0 {
        mean_utility_delta.abs() / standard_error
    } else if mean_utility_delta == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let lower_confidence_bound = mean_utility_delta - NONINFERIORITY_95_Z * standard_error;
    if !lower_confidence_bound.is_finite() {
        return Err(BrainError::Numerical(
            "functional_replay_confidence_bound_invalid".into(),
        ));
    }
    Ok(RecomputedReplayStatistics {
        paired_independent_groups: count,
        mean_utility_delta,
        standard_error,
        max_acceptable_utility_loss,
        zero_effect_z,
        lower_confidence_bound,
    })
}

#[derive(Debug, Deserialize)]
struct ProtectionSourceRow {
    probe_id: String,
    artifact: DeltaArtifactRef,
    causal_damage_per_parameter_norm: f64,
    reliability: f64,
}

#[derive(Debug, Deserialize)]
struct ProtectionSourcePayload {
    schema: String,
    task_labels_used: bool,
    evidence: Vec<ProtectionSourceRow>,
}

fn load_sensitivity_evidence(
    root: &Path,
    source_path: &str,
    source_sha256: &Sha256Digest,
) -> BrainResult<Vec<SensitivityEvidence>> {
    let source_raw = PrivateFileReference::new(PathBuf::from(source_path), source_sha256.clone())
        .read_verified_bounded(root, MAX_SLEEP_EVIDENCE_JSON_BYTES)?;
    let payload: ProtectionSourcePayload = serde_json::from_slice(&source_raw)?;
    if payload.schema != "cerebro.tidex.protected_sensitivity_evidence/v1"
        || payload.task_labels_used
        || payload.evidence.len() < 2
    {
        return Err(BrainError::Integrity(
            "protected_sensitivity_source_contract_invalid".into(),
        ));
    }
    payload
        .evidence
        .into_iter()
        .enumerate()
        .map(|(index, row)| {
            let artifact_path = confined_path(root, &row.artifact.path).map_err(|error| {
                BrainError::Integrity(format!(
                    "protected_sensitivity_artifact_path_invalid:{index}:{error}"
                ))
            })?;
            let mut artifact = row.artifact;
            artifact.path = artifact_path;
            let values = read_dvec_f32(root, &artifact)?
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>();
            Ok(SensitivityEvidence {
                probe_id: crate::identity::ProbeId::parse(row.probe_id).map_err(|_| {
                    BrainError::Integrity("protected_sensitivity_probe_id_invalid".into())
                })?,
                sensitivity: values,
                reliability: row.reliability,
                causal_damage: Some(row.causal_damage_per_parameter_norm),
            })
        })
        .collect()
}

fn verify_protection_artifacts(
    root: &Path,
    source_path: &str,
    source_sha256: &Sha256Digest,
) -> BrainResult<bool> {
    Ok(load_sensitivity_evidence(root, source_path, source_sha256).is_ok())
}

fn verify_causal_credit_artifacts(
    root: &Path,
    summary: &CausalCreditEvidenceSummary,
    expected_ids: &[SkillId],
    expected_report_sha256: &ReportDigest,
) -> BrainResult<bool> {
    if !verify_file(
        root,
        &summary.replay_source_path,
        &summary.replay_source_sha256,
    )? || !verify_file(
        root,
        &summary.credit_source_path,
        &summary.credit_source_sha256,
    )? || summary.blind_data_accessed
        || !valid_sha256(expected_report_sha256)
        || summary.report_sha256.as_str() != expected_report_sha256.as_str()
        || summary.independent_group_count < 3
        || summary.field_count != expected_ids.len()
        || summary.resolved_field_count != expected_ids.len()
        || summary.interaction_count
            != expected_ids.len() * expected_ids.len().saturating_sub(1) / 2
        || summary.resolved_interaction_count != summary.interaction_count
        || !valid_sha256(&summary.plan_sha256)
    {
        return Ok(false);
    }
    let replay_raw = PrivateFileReference::new(
        PathBuf::from(&summary.replay_source_path),
        summary.replay_source_sha256.clone(),
    )
    .read_verified_bounded(root, MAX_SLEEP_EVIDENCE_JSON_BYTES)?;
    let replay: serde_json::Value = serde_json::from_slice(&replay_raw)?;
    if replay.get("schema").and_then(serde_json::Value::as_str)
        != Some("cerebro.tidex.counterfactual_replay/v3")
        || replay
            .get("blind_data_accessed")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        || replay
            .get("report_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(summary.report_sha256.as_str())
        || replay
            .get("plan_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(summary.plan_sha256.as_str())
    {
        return Ok(false);
    }
    let replay_ids = replay
        .get("field_ids")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| BrainError::Integrity("causal_replay_field_ids_missing".into()))?
        .iter()
        .map(|value| {
            let value = value
                .as_str()
                .ok_or_else(|| BrainError::Integrity("causal_replay_field_id_invalid".into()))?;
            SkillId::parse(value)
                .map_err(|_| BrainError::Integrity("causal_replay_field_id_invalid".into()))
        })
        .collect::<BrainResult<Vec<_>>>()?;
    if replay_ids != expected_ids {
        return Ok(false);
    }
    let evaluations_value = replay
        .get("evaluations")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| BrainError::Integrity("causal_replay_evaluations_missing".into()))?;
    if evaluations_value.is_empty()
        || evaluations_value.iter().any(|row| {
            row.get("context_id")
                .and_then(serde_json::Value::as_str)
                .is_none()
                || row
                    .get("independence_group")
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(str::is_empty)
                || row
                    .get("active_fields")
                    .and_then(serde_json::Value::as_array)
                    .is_none()
                || row
                    .get("utility")
                    .and_then(serde_json::Value::as_f64)
                    .is_none()
        })
    {
        return Ok(false);
    }
    let evaluations: Vec<CounterfactualEvaluation> =
        serde_json::from_value(serde_json::Value::Array(evaluations_value.clone()))?;
    let recomputed_credit = estimate_causal_credit(&evaluations)?;

    let credit_raw = PrivateFileReference::new(
        PathBuf::from(&summary.credit_source_path),
        summary.credit_source_sha256.as_digest().clone(),
    )
    .read_verified_bounded(root, MAX_SLEEP_EVIDENCE_JSON_BYTES)?;
    let wrapper: serde_json::Value = serde_json::from_slice(&credit_raw)?;
    if wrapper.get("schema").and_then(serde_json::Value::as_str)
        != Some("cerebro.tidex.causal_credit_benchmark/v3")
        || wrapper
            .get("blind_data_accessed")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        || wrapper
            .get("replay_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(summary.replay_source_sha256.as_str())
        || wrapper
            .get("report_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(expected_report_sha256)
        || wrapper
            .get("plan_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(summary.plan_sha256.as_str())
    {
        return Ok(false);
    }
    let wrapper_field_ids = wrapper
        .get("field_ids")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| BrainError::Integrity("causal_credit_wrapper_field_ids_missing".into()))?
        .iter()
        .map(|value| {
            let value = value.as_str().ok_or_else(|| {
                BrainError::Integrity("causal_credit_wrapper_field_id_invalid".into())
            })?;
            SkillId::parse(value)
                .map_err(|_| BrainError::Integrity("causal_credit_wrapper_field_id_invalid".into()))
        })
        .collect::<BrainResult<Vec<_>>>()?;
    if wrapper_field_ids != expected_ids {
        return Ok(false);
    }
    let credit = wrapper
        .get("causal_credit")
        .ok_or_else(|| BrainError::Integrity("causal_credit_payload_missing".into()))?;
    let stored_credit: CausalCreditReport = serde_json::from_value(credit.clone())?;
    let stored_matches_recomputed = stored_credit.schema == recomputed_credit.schema
        && stored_credit.context_count == recomputed_credit.context_count
        && stored_credit.independent_group_count == recomputed_credit.independent_group_count
        && stored_credit.field_count == recomputed_credit.field_count
        && stored_credit.unresolved_fields == recomputed_credit.unresolved_fields
        && stored_credit.fields.len() == recomputed_credit.fields.len()
        && stored_credit
            .fields
            .iter()
            .zip(&recomputed_credit.fields)
            .all(|(left, right)| {
                left.skill_id == right.skill_id
                    && left.matched_pairs == right.matched_pairs
                    && left.independent_contexts == right.independent_contexts
                    && finite_scalar_match(left.mean_marginal_effect, right.mean_marginal_effect)
                    && finite_scalar_match(left.standard_error, right.standard_error)
                    && finite_scalar_match(
                        left.lower_confidence_bound,
                        right.lower_confidence_bound,
                    )
                    && finite_scalar_match(left.positive_fraction, right.positive_fraction)
                    && left.resolved == right.resolved
                    && left.beneficial == right.beneficial
            })
        && stored_credit.pair_interactions.len() == recomputed_credit.pair_interactions.len()
        && stored_credit
            .pair_interactions
            .iter()
            .zip(&recomputed_credit.pair_interactions)
            .all(|(left, right)| {
                left.left_skill_id == right.left_skill_id
                    && left.right_skill_id == right.right_skill_id
                    && left.matched_quads == right.matched_quads
                    && left.independent_contexts == right.independent_contexts
                    && finite_scalar_match(
                        left.mean_interaction_effect,
                        right.mean_interaction_effect,
                    )
                    && finite_scalar_match(left.standard_error, right.standard_error)
                    && left.resolved == right.resolved
            });
    if !stored_matches_recomputed {
        return Ok(false);
    }
    if credit.get("schema").and_then(serde_json::Value::as_str)
        != Some("cerebro.tidex.causal_credit/v3")
        || credit
            .get("independent_group_count")
            .and_then(serde_json::Value::as_u64)
            != Some(summary.independent_group_count as u64)
        || credit
            .get("field_count")
            .and_then(serde_json::Value::as_u64)
            != Some(summary.field_count as u64)
    {
        return Ok(false);
    }
    let fields = credit
        .get("fields")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| BrainError::Integrity("causal_credit_fields_missing".into()))?;
    let field_ids = fields
        .iter()
        .map(|field| {
            let skill_id = field
                .get("skill_id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| BrainError::Integrity("causal_credit_skill_id_missing".into()))?;
            let resolved = field
                .get("resolved")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let contexts = field
                .get("independent_contexts")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let pairs = field
                .get("matched_pairs")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let effect = field
                .get("mean_marginal_effect")
                .and_then(serde_json::Value::as_f64);
            let error = field
                .get("standard_error")
                .and_then(serde_json::Value::as_f64);
            let lower_bound = field
                .get("lower_confidence_bound")
                .and_then(serde_json::Value::as_f64);
            let beneficial = field.get("beneficial").and_then(serde_json::Value::as_bool);
            if !resolved
                || contexts < 3
                || pairs == 0
                || effect.is_none()
                || error.is_none()
                || lower_bound.is_none()
                || beneficial.is_none()
            {
                return Err(BrainError::Integrity(
                    "causal_credit_field_support_unresolved".into(),
                ));
            }
            SkillId::parse(skill_id)
        })
        .collect::<BrainResult<Vec<_>>>()?;
    let mut sorted_field_ids = field_ids;
    let mut sorted_expected = expected_ids.to_vec();
    sorted_field_ids.sort();
    sorted_expected.sort();
    if sorted_field_ids != sorted_expected || fields.len() != summary.resolved_field_count {
        return Ok(false);
    }
    let interactions = credit
        .get("pair_interactions")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| BrainError::Integrity("causal_credit_interactions_missing".into()))?;
    if interactions.len() != summary.interaction_count {
        return Ok(false);
    }
    let expected_set = sorted_expected
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    for interaction in interactions {
        let left = interaction
            .get("left_skill_id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let right = interaction
            .get("right_skill_id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if left == right
            || !expected_set.contains(left)
            || !expected_set.contains(right)
            || interaction
                .get("resolved")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
            || interaction
                .get("independent_contexts")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                < 3
            || interaction
                .get("matched_quads")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
            || interaction
                .get("mean_interaction_effect")
                .and_then(serde_json::Value::as_f64)
                .is_none()
            || interaction
                .get("standard_error")
                .and_then(serde_json::Value::as_f64)
                .is_none()
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Load causal credit only after its source replay and wrapper have both been
/// recomputed and matched. The returned priority vector is the authoritative
/// lower-confidence-bound ordering used by the trust region; absence of that
/// vector is an evidence failure, never a uniform-scaling fallback.
fn verified_causal_credit_with_weights(
    root: &Path,
    summary: &CausalCreditEvidenceSummary,
    expected_ids: &[SkillId],
    expected_report_sha256: &ReportDigest,
) -> BrainResult<Option<(CausalCreditReport, Vec<f64>)>> {
    if !verify_causal_credit_artifacts(root, summary, expected_ids, expected_report_sha256)? {
        return Ok(None);
    }
    let credit_raw = PrivateFileReference::new(
        PathBuf::from(&summary.credit_source_path),
        summary.credit_source_sha256.as_digest().clone(),
    )
    .read_verified_bounded(root, MAX_SLEEP_EVIDENCE_JSON_BYTES)?;
    let wrapper: serde_json::Value = serde_json::from_slice(&credit_raw)?;
    let report: CausalCreditReport = serde_json::from_value(
        wrapper
            .get("causal_credit")
            .cloned()
            .ok_or_else(|| BrainError::Integrity("causal_credit_payload_missing".into()))?,
    )?;
    match certified_causal_priority_weights(&report, expected_ids) {
        Ok(weights) => Ok(Some((report, weights))),
        Err(_) => Ok(None),
    }
}

fn dense_fields_from_observations(
    root: &Path,
    expectation: &SleepEvidenceExpectation<'_>,
    parameter_dimension: usize,
) -> BrainResult<Vec<Vec<f64>>> {
    if expectation.fields.is_empty()
        || expectation.source_mixtures.len() != expectation.fields.len()
        || expectation
            .source_mixtures
            .iter()
            .any(|row| row.len() != expectation.observations.len())
        || parameter_dimension == 0
    {
        return Err(BrainError::Invalid(
            "sleep_dense_field_source_shape_invalid".into(),
        ));
    }
    let mut fields = Vec::with_capacity(expectation.fields.len());
    for (field_index, mixture) in expectation.source_mixtures.iter().enumerate() {
        let mut dense = vec![0.0f64; parameter_dimension];
        let mut support = 0usize;
        for (coefficient, observation) in mixture.iter().zip(expectation.observations) {
            if !coefficient.is_finite() {
                return Err(BrainError::Invalid(format!(
                    "sleep_dense_field_coefficient_invalid:{field_index}"
                )));
            }
            if coefficient.abs() <= f64::EPSILON {
                continue;
            }
            let reference = observation.dense_artifact.as_ref().ok_or_else(|| {
                BrainError::Integrity(format!(
                    "sleep_dense_field_source_missing:{}",
                    observation.observation_id
                ))
            })?;
            if reference.parameter_count != parameter_dimension as u64 {
                return Err(BrainError::Integrity(format!(
                    "sleep_dense_field_source_dimension:{}",
                    observation.observation_id
                )));
            }
            let values = read_dvec_f32(root, reference)?;
            if values.len() != parameter_dimension {
                return Err(BrainError::Integrity(
                    "sleep_dense_field_loaded_dimension".into(),
                ));
            }
            for (output, value) in dense.iter_mut().zip(values) {
                let sum = *output + coefficient * f64::from(value);
                if !sum.is_finite() || sum.abs() > f32::MAX as f64 {
                    return Err(BrainError::Numerical(
                        "sleep_dense_field_nonfinite_or_overflow".into(),
                    ));
                }
                *output = f64::from(sum as f32);
            }
            support += 1;
        }
        if support == 0 {
            return Err(BrainError::Integrity(format!(
                "sleep_dense_field_support_empty:{field_index}"
            )));
        }
        fields.push(dense);
    }
    Ok(fields)
}

pub fn verify_sleep_evidence(
    root: &Path,
    bundle: &SleepEvidenceBundle,
    expected: &SleepEvidenceExpectation<'_>,
) -> BrainResult<SleepEvidenceVerification> {
    if bundle.schema != "cerebro.tidex.sleep_evidence/v4" {
        return Err(BrainError::Invalid("sleep_evidence_schema_invalid".into()));
    }
    let mut reasons = Vec::new();
    if bundle.corpus_digest.as_str() != expected.corpus_digest.as_str() {
        reasons.push("sleep_evidence_corpus_mismatch".into());
    }
    if bundle.source_tree_digest.as_str() != expected.source_tree_digest.as_str()
        || bundle.config_digest.as_str() != expected.config_digest.as_str()
        || bundle.analysis_version_digest.as_str() != expected.analysis_version_digest.as_str()
    {
        reasons.push("sleep_evidence_analysis_identity_mismatch".into());
    }
    let expected_ids = expected
        .fields
        .iter()
        .map(|field| field.skill_id.clone())
        .collect::<Vec<_>>();
    if bundle.field_ids != expected_ids {
        reasons.push("sleep_evidence_field_identity_mismatch".into());
    }

    let protection_file = verify_file(
        root,
        &bundle.protection.source_path,
        &bundle.protection.source_sha256,
    )?;
    let protection_artifacts = verify_protection_artifacts(
        root,
        &bundle.protection.source_path,
        &bundle.protection.source_sha256,
    )?;
    let protected_map_file = verify_file(
        root,
        &bundle.protection.protected_map_path,
        &bundle.protection.protected_map_sha256,
    )?;
    let mut protected_map_verified = false;
    let mut verified_cortex = None;
    if protected_map_file {
        let map_raw = PrivateFileReference::new(
            PathBuf::from(&bundle.protection.protected_map_path),
            bundle.protection.protected_map_sha256.as_digest().clone(),
        )
        .read_verified_bounded(root, MAX_SLEEP_EVIDENCE_JSON_BYTES)?;
        let wrapper: serde_json::Value = serde_json::from_slice(&map_raw)?;
        if wrapper.get("schema").and_then(serde_json::Value::as_str)
            == Some("cerebro.tidex.protected_map_benchmark/v2")
            && wrapper
                .get("task_labels_used")
                .and_then(serde_json::Value::as_bool)
                == Some(false)
        {
            if let Some(map_value) = wrapper.get("map") {
                let map: ProtectedMapArtifactReport = serde_json::from_value(map_value.clone())?;
                let cortex = load_protected_cortex(root, &map)?;
                let sensitivity = load_sensitivity_evidence(
                    root,
                    &bundle.protection.source_path,
                    &bundle.protection.source_sha256,
                )?;
                let recomputed = build_protected_cortex_map(
                    &sensitivity,
                    map.retained_sensitivity_energy,
                    map.cortex.max_damage_ratio,
                )?;
                let tolerance = f64::EPSILON.sqrt() * map.parameter_dimension.max(1) as f64 * 8.0;
                let scalar_match = recomputed.probe_count == map.probe_count
                    && recomputed.parameter_dimension == map.parameter_dimension
                    && recomputed.selected_rank == map.selected_rank
                    && (recomputed.effective_rank - map.effective_rank).abs() <= tolerance
                    && (recomputed.retained_sensitivity_energy - map.retained_sensitivity_energy)
                        .abs()
                        <= tolerance
                    && (recomputed.fisher_trace - map.fisher_trace).abs() <= tolerance
                    && recomputed.causal_damage_supported_probes
                        == map.causal_damage_supported_probes
                    && match (
                        recomputed.sensitivity_damage_correlation,
                        map.sensitivity_damage_correlation,
                    ) {
                        (Some(left), Some(right)) => (left - right).abs() <= tolerance,
                        (None, None) => true,
                        _ => false,
                    };
                let parameter_match = cortex.parameter_importance.len()
                    == recomputed.cortex.parameter_importance.len()
                    && cortex
                        .parameter_importance
                        .iter()
                        .zip(&recomputed.cortex.parameter_importance)
                        .all(|(left, right)| (left - right).abs() <= tolerance);
                let direction_match = cortex.directions.len() == recomputed.cortex.directions.len()
                    && cortex
                        .directions
                        .iter()
                        .zip(&recomputed.cortex.directions)
                        .all(|(stored, current)| {
                            stored.probe_id == current.probe_id
                                && (stored.importance - current.importance).abs() <= tolerance
                                && stored.direction.len() == current.direction.len()
                                && stored
                                    .direction
                                    .iter()
                                    .zip(&current.direction)
                                    .all(|(left, right)| (left - right).abs() <= tolerance)
                        });
                protected_map_verified = scalar_match
                    && parameter_match
                    && direction_match
                    && cortex.max_damage_ratio == recomputed.cortex.max_damage_ratio
                    && map.parameter_dimension == bundle.protection.parameter_dimension
                    && map.selected_rank == bundle.protection.selected_rank;
                if protected_map_verified {
                    verified_cortex = Some(cortex);
                }
            }
        }
    }
    let protection_verified = protection_file
        && protection_artifacts
        && protected_map_verified
        && bundle.protection.probe_count >= 2
        && bundle.protection.parameter_dimension > 0
        && bundle.protection.selected_rank > 0
        && bundle.protection.causal_damage_supported_probes >= 2
        && bundle
            .protection
            .sensitivity_damage_correlation
            .is_some_and(|value| value.is_finite() && value > 0.0);
    if !protection_verified {
        reasons.push("sleep_protection_evidence_unverified".into());
    }

    // Causal credit is an input to the trust rule, not a post-hoc diagnostic.
    // Its source replay is recomputed before any trust artifact can be
    // accepted, and it must yield a complete positive LCB priority vector.
    let causal_credit = verified_causal_credit_with_weights(
        root,
        &bundle.causal_credit,
        &expected_ids,
        expected.report_sha256,
    )?;
    let causal_credit_verified = causal_credit.is_some();
    if !causal_credit_verified {
        reasons.push("sleep_causal_credit_evidence_unverified".into());
    }

    let interaction_file = verify_file(
        root,
        &bundle.interaction.source_path,
        &bundle.interaction.source_sha256,
    )?;
    let mut interaction_artifact_verified = false;
    if interaction_file && bundle.interaction.field_ids == expected_ids && causal_credit.is_some() {
        let interaction_raw = PrivateFileReference::new(
            PathBuf::from(&bundle.interaction.source_path),
            bundle.interaction.source_sha256.clone(),
        )
        .read_verified_bounded(root, MAX_SLEEP_EVIDENCE_JSON_BYTES)?;
        let payload: serde_json::Value = serde_json::from_slice(&interaction_raw)?;
        if payload.get("schema").and_then(serde_json::Value::as_str)
            == Some("cerebro.tidex.trust_region_benchmark/v3")
            && payload
                .get("report_sha256")
                .and_then(serde_json::Value::as_str)
                == Some(expected.report_sha256)
            && payload
                .get("protected_map_sha256")
                .and_then(serde_json::Value::as_str)
                == Some(bundle.protection.protected_map_sha256.as_str())
            && payload
                .get("causal_plan_sha256")
                .and_then(serde_json::Value::as_str)
                == Some(bundle.causal_credit.plan_sha256.as_str())
            && payload
                .get("causal_credit_sha256")
                .and_then(serde_json::Value::as_str)
                == Some(bundle.causal_credit.credit_source_sha256.as_str())
            && payload
                .get("causal_priority_weight_kind")
                .and_then(serde_json::Value::as_str)
                == Some("causal_lower_confidence_bound_95")
            && payload
                .get("field_ids")
                .and_then(serde_json::Value::as_array)
                .is_some()
        {
            let field_ids = payload
                .get("field_ids")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| BrainError::Integrity("sleep_field_ids_missing".into()))?
                .iter()
                .map(|value| {
                    let value = value
                        .as_str()
                        .ok_or_else(|| BrainError::Integrity("sleep_field_id_invalid".into()))?;
                    SkillId::parse(value)
                        .map_err(|_| BrainError::Integrity("sleep_field_id_invalid".into()))
                })
                .collect::<BrainResult<Vec<_>>>()?;
            let rows = payload
                .get("interaction_matrix")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| BrainError::Integrity("sleep_interaction_matrix_missing".into()))?
                .iter()
                .map(|row| {
                    row.as_array()
                        .ok_or_else(|| {
                            BrainError::Integrity("sleep_interaction_row_invalid".into())
                        })?
                        .iter()
                        .map(|value| {
                            value
                                .as_f64()
                                .filter(|value| value.is_finite())
                                .ok_or_else(|| {
                                    BrainError::Integrity("sleep_interaction_value_invalid".into())
                                })
                        })
                        .collect::<BrainResult<Vec<_>>>()
                })
                .collect::<BrainResult<Vec<_>>>()?;
            let matrix = Matrix::from_rows(&rows)?;
            let recomputed_matrix = if let Some(cortex) = verified_cortex.as_ref() {
                let dense = dense_fields_from_observations(
                    root,
                    expected,
                    bundle.interaction.dense_parameter_dimension,
                )?;
                Some(second_order_interactions(
                    &dense,
                    &cortex.parameter_importance,
                )?)
            } else {
                None
            };
            let trust: TrustRegionDecisionIdentity = serde_json::from_value(
                payload
                    .get("trust_region")
                    .cloned()
                    .ok_or_else(|| BrainError::Integrity("sleep_trust_region_missing".into()))?,
            )?;
            let budget = payload
                .get("diagonal_budget")
                .and_then(serde_json::Value::as_f64)
                .filter(|value| value.is_finite() && *value >= 0.0)
                .ok_or_else(|| BrainError::Integrity("sleep_trust_budget_missing".into()))?;
            let causal_priority_weights = causal_credit
                .as_ref()
                .map(|(_, weights)| weights)
                .ok_or_else(|| BrainError::Integrity("sleep_causal_credit_missing".into()))?;
            let recomputed = apply_causal_priority_trust_region(
                &matrix,
                &trust.proposed_coefficients,
                budget,
                causal_priority_weights,
            )?;
            let tolerance = f64::EPSILON.sqrt() * matrix.rows.max(1) as f64 * 32.0;
            let scalar_matches = |stored: f64, current: f64| {
                stored.is_finite()
                    && current.is_finite()
                    && (stored - current).abs() <= tolerance * (1.0 + current.abs())
            };
            let interaction_matrix_matches = recomputed_matrix.as_ref().is_some_and(|current| {
                current.rows == matrix.rows
                    && current.cols == matrix.cols
                    && (0..matrix.rows).all(|row| {
                        (0..matrix.cols).all(|col| {
                            (current.get(row, col) - matrix.get(row, col)).abs() <= tolerance
                        })
                    })
            });
            interaction_artifact_verified = field_ids == expected_ids
                && matrix.rows == expected_ids.len()
                && matrix.cols == expected_ids.len()
                && interaction_matrix_matches
                && bundle.interaction.dense_parameter_dimension
                    == bundle.protection.parameter_dimension
                && bundle.interaction.causal_credit_sha256
                    == bundle.causal_credit.credit_source_sha256
                && valid_sha256(&bundle.interaction.causal_credit_sha256)
                && bundle.interaction.allocation_policy
                    == TrustRegionAllocationPolicy::CausalPriorityContractionV1
                && trust.proposed_coefficients.len() == expected_ids.len()
                && trust.accepted_coefficients.len() == expected_ids.len()
                && trust.allocation_policy
                    == TrustRegionAllocationPolicy::CausalPriorityContractionV1
                && trust.component_retention.len() == expected_ids.len()
                && trust.causal_priority_weights.len() == expected_ids.len()
                && trust
                    .accepted_coefficients
                    .iter()
                    .zip(&recomputed.accepted_coefficients)
                    .all(|(stored, current)| scalar_matches(*stored, *current))
                && trust
                    .component_retention
                    .iter()
                    .zip(&recomputed.component_retention)
                    .all(|(stored, current)| scalar_matches(*stored, *current))
                && trust
                    .causal_priority_weights
                    .iter()
                    .zip(causal_priority_weights)
                    .all(|(stored, current)| scalar_matches(*stored, *current))
                && finite_vector_match(
                    &bundle.interaction.causal_priority_weights,
                    causal_priority_weights,
                )
                && finite_vector_match(
                    &bundle.interaction.component_retention,
                    &recomputed.component_retention,
                )
                && trust.constrained == recomputed.constrained
                && scalar_matches(
                    trust.proposed_quadratic_cost,
                    recomputed.proposed_quadratic_cost,
                )
                && scalar_matches(
                    trust.accepted_quadratic_cost,
                    recomputed.accepted_quadratic_cost,
                )
                && scalar_matches(trust.max_quadratic_cost, budget)
                && scalar_matches(trust.scale, recomputed.scale)
                && scalar_matches(
                    recomputed.proposed_quadratic_cost,
                    bundle.interaction.proposed_quadratic_cost,
                )
                && scalar_matches(
                    recomputed.accepted_quadratic_cost,
                    bundle.interaction.accepted_quadratic_cost,
                )
                && scalar_matches(budget, bundle.interaction.diagonal_budget)
                && scalar_matches(recomputed.scale, bundle.interaction.trust_scale);
        }
    }
    let interaction_verified = interaction_artifact_verified
        && bundle.interaction.proposed_quadratic_cost.is_finite()
        && bundle.interaction.accepted_quadratic_cost.is_finite()
        && bundle.interaction.diagonal_budget.is_finite()
        && bundle.interaction.trust_scale.is_finite()
        && (0.0..=1.0).contains(&bundle.interaction.trust_scale)
        && bundle.interaction.allocation_policy
            == TrustRegionAllocationPolicy::CausalPriorityContractionV1
        && bundle.interaction.causal_priority_weights.len() == expected_ids.len()
        && bundle.interaction.component_retention.len() == expected_ids.len()
        && bundle
            .interaction
            .causal_priority_weights
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        && bundle
            .interaction
            .component_retention
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        && bundle.interaction.accepted_quadratic_cost
            <= bundle.interaction.diagonal_budget * (1.0 + 1e-10) + 1e-12;
    if !interaction_verified {
        reasons.push("sleep_interaction_trust_evidence_unverified".into());
    }

    // The replay summary is a projection, never evidence. Reconstruct the
    // paired seed-level utility deltas from the immutable baseline and
    // candidate artifacts, then require the summary to match that derivation.
    // This prevents an author from certifying sleep by merely editing aggregate
    // mean/SE fields in the evidence bundle.
    let (replay_verified, replay_zero_effect_z, replay_lower_confidence_bound, margin) =
        match recompute_functional_replay(root, bundle, expected, &expected_ids) {
            Ok(statistics) => {
                let summary_matches = !bundle.replay.blind_data_accessed
                    && bundle.replay.paired_independent_groups
                        == statistics.paired_independent_groups
                    && finite_scalar_match(
                        bundle.replay.mean_utility_delta,
                        statistics.mean_utility_delta,
                    )
                    && finite_scalar_match(bundle.replay.standard_error, statistics.standard_error)
                    && bundle
                        .replay
                        .max_acceptable_utility_loss
                        .is_some_and(|value| {
                            finite_scalar_match(value, statistics.max_acceptable_utility_loss)
                        });
                if !summary_matches {
                    reasons.push("sleep_functional_replay_summary_mismatch".into());
                }
                let verified = summary_matches
                    && statistics.lower_confidence_bound >= -statistics.max_acceptable_utility_loss;
                if !verified && summary_matches {
                    reasons.push("sleep_functional_replay_noninferior_unproven".into());
                }
                (
                    verified,
                    statistics.zero_effect_z,
                    statistics.lower_confidence_bound,
                    Some(statistics.max_acceptable_utility_loss),
                )
            }
            Err(error) => {
                reasons.push(format!("sleep_functional_replay_artifact_invalid:{error}"));
                (false, f64::INFINITY, f64::NEG_INFINITY, None)
            }
        };

    Ok(SleepEvidenceVerification {
        schema: "cerebro.tidex.sleep_evidence_verification/v3".into(),
        verified: reasons.is_empty(),
        protection_verified,
        interaction_verified,
        replay_verified,
        causal_credit_verified,
        replay_zero_effect_z,
        replay_lower_confidence_bound,
        noninferiority_margin: margin,
        reasons,
    })
}

pub fn load_sleep_evidence(root: &Path) -> BrainResult<SleepEvidenceBundle> {
    let path = root.join("state/sleep_evidence/current.json");
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_file() => {}
        Ok(_) => {
            return Err(BrainError::Integrity(
                "sleep_evidence_current_not_regular".into(),
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(BrainError::Integrity("sleep_evidence_missing".into()));
        }
        Err(error) => return Err(error.into()),
    }
    let raw = read_existing_private_file_bounded(root, &path, MAX_SLEEP_EVIDENCE_JSON_BYTES)?;
    Ok(serde_json::from_slice(&raw)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::create_content_addressed_dvec;
    use crate::contracts::{DeltaObservation, ExperimentLineage, SkillField};
    use crate::protected_map::{build_protected_cortex_map, persist_protected_map};
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn isolated_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tidex-sleep-evidence-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn confined_path_rejects_symlinked_and_nonregular_evidence_paths() {
        let root = isolated_root("path-confinement");
        let evidence_dir = root.join("evidence");
        fs::create_dir_all(&evidence_dir).unwrap();
        let regular = evidence_dir.join("regular.json");
        fs::write(&regular, b"{}").unwrap();
        assert_eq!(confined_path(&root, &regular).unwrap(), regular);

        let leaf_link = evidence_dir.join("leaf-link.json");
        symlink(&regular, &leaf_link).unwrap();
        assert!(confined_path(&root, &leaf_link).is_err());

        let parent_link = root.join("redirect");
        symlink(&evidence_dir, &parent_link).unwrap();
        assert!(confined_path(&root, parent_link.join("regular.json")).is_err());

        let directory_leaf = evidence_dir.join("not-a-file");
        fs::create_dir(&directory_leaf).unwrap();
        assert!(confined_path(&root, &directory_leaf).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn load_sleep_evidence_rejects_symlink_or_directory_current_pointer() {
        let root = isolated_root("current-pointer");
        let pointer = root.join("state/sleep_evidence/current.json");
        fs::create_dir_all(pointer.parent().unwrap()).unwrap();
        let target = root.join("evidence.json");
        fs::write(&target, b"{}").unwrap();

        symlink(&target, &pointer).unwrap();
        assert!(load_sleep_evidence(&root).is_err());
        fs::remove_file(&pointer).unwrap();

        fs::create_dir(&pointer).unwrap();
        assert!(load_sleep_evidence(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sensitivity_loader_rejects_artifact_under_symlinked_parent() {
        let root = isolated_root("sensitivity-artifact");
        let artifact = create_content_addressed_dvec(&root, &[1.0]).unwrap();
        let redirected_parent = root.join("redirected-artifacts");
        symlink(root.join("artifacts/deltas/by-sha"), &redirected_parent).unwrap();
        let redirected_artifact = redirected_parent.join(artifact.path.file_name().unwrap());
        let source = root.join("protection.json");
        let payload = serde_json::json!({
            "schema":"cerebro.tidex.protected_sensitivity_evidence/v1",
            "task_labels_used":false,
            "evidence":[
                {
                    "probe_id":"p1",
                    "artifact":{
                        "path":redirected_artifact,
                        "sha256":artifact.sha256.clone(),
                        "parameter_count":artifact.parameter_count
                    },
                    "causal_damage_per_parameter_norm":1.0,
                    "reliability":1.0
                },
                {
                    "probe_id":"p2",
                    "artifact":{
                        "path":artifact.path,
                        "sha256":artifact.sha256,
                        "parameter_count":artifact.parameter_count
                    },
                    "causal_damage_per_parameter_norm":1.0,
                    "reliability":1.0
                }
            ]
        });
        fs::write(&source, serde_json::to_vec(&payload).unwrap()).unwrap();
        let source_sha256 = crate::artifact::sha256_file(&source).unwrap();
        assert!(
            load_sensitivity_evidence(&root, source.to_str().unwrap(), &source_sha256).is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }

    fn field() -> SkillField {
        SkillField {
            skill_id: crate::identity::SkillId::parse("s").unwrap(),
            reconstruction_id: Default::default(),
            lineage_id: Default::default(),
            generation_created: 1,
            direction: vec![1.0],
            structured_geometry: None,
            dense_materialization: None,
            parameter_layout_sha256: None,
            representation_signature: Vec::new(),
            singular_value: 1.0,
            explained_variance: 1.0,
            persistence: 1.0,
            coherence: 1.0,
            uncertainty: 0.0,
            evidence_support_digests: Vec::new(),
            support: 1,
            functional_signature: vec![],
            parent_skill_ids: vec![],
        }
    }

    #[test]
    fn sleep_evidence_recomputes_replay_and_rejects_summary_forgery() {
        let root =
            std::env::temp_dir().join(format!("tidex-sleep-evidence-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let gradient_a = create_content_addressed_dvec(&root, &[1.0]).unwrap();
        let gradient_b = create_content_addressed_dvec(&root, &[2.0]).unwrap();
        let interaction = root.join("interaction.json");
        let baseline = root.join("causal_replay.json");
        let candidate = root.join("candidate_replay.json");
        let causal_credit = root.join("causal_credit.json");
        let report_sha = ReportDigest::from(Sha256Digest::parse("1".repeat(64)).unwrap());
        let plan_sha = Sha256Digest::parse("2".repeat(64)).unwrap();
        let validation_seeds = vec![11_u64, 12, 13];
        let validation_tasks = vec!["task".to_string()];

        let mut causal_evaluations = Vec::new();
        let mut raw_results = Vec::new();
        for seed in &validation_seeds {
            let context_id = format!("validation-seed-{seed}:task");
            let independence_group = format!("validation-seed-{seed}");
            causal_evaluations.push(CounterfactualEvaluation {
                context_id: context_id.clone(),
                independence_group: independence_group.clone(),
                active_fields: Vec::new(),
                utility: 0.4,
            });
            causal_evaluations.push(CounterfactualEvaluation {
                context_id,
                independence_group,
                active_fields: vec![crate::identity::SkillId::parse("s").unwrap()],
                utility: 0.5,
            });
            raw_results.push(serde_json::json!({
                "mask": 0,
                "active_fields": [],
                "seed": seed,
                "metrics": {"task": 0.4},
            }));
            raw_results.push(serde_json::json!({
                "mask": 1,
                "active_fields": ["s"],
                "seed": seed,
                "metrics": {"task": 0.5},
            }));
        }
        fs::write(
            &baseline,
            serde_json::to_vec(&serde_json::json!({
                "schema":"cerebro.tidex.counterfactual_replay/v3",
                "report_sha256":report_sha.clone(),
                "plan_sha256":plan_sha.clone(),
                "field_ids":["s"],
                "field_coefficients":[1.0],
                "validation_seeds":validation_seeds.clone(),
                "validation_tasks":validation_tasks.clone(),
                "count_per_task":100,
                "blind_data_accessed":false,
                "evaluations":causal_evaluations.clone(),
                "raw_results":raw_results,
            }))
            .unwrap(),
        )
        .unwrap();
        let causal_report = estimate_causal_credit(&causal_evaluations).unwrap();
        let causal_priority_weights = certified_causal_priority_weights(
            &causal_report,
            &[crate::identity::SkillId::parse("s").unwrap()],
        )
        .unwrap();
        fs::write(
            &causal_credit,
            serde_json::to_vec(&serde_json::json!({
                "schema":"cerebro.tidex.causal_credit_benchmark/v3",
                "blind_data_accessed":false,
                "replay_sha256":crate::artifact::sha256_file(&baseline).unwrap(),
                "report_sha256":report_sha.clone(),
                "plan_sha256":plan_sha.clone(),
                "field_ids":["s"],
                "causal_credit":causal_report.clone(),
            }))
            .unwrap(),
        )
        .unwrap();
        let sha = |p: &Path| crate::artifact::sha256_file(p).unwrap();
        let a = root.join("protection.json");
        let protection_payload = serde_json::json!({
            "schema":"cerebro.tidex.protected_sensitivity_evidence/v1",
            "task_labels_used":false,
            "evidence":[
                {"probe_id":"p1","artifact":gradient_a,"causal_damage_per_parameter_norm":1.0,"reliability":1.0},
                {"probe_id":"p2","artifact":gradient_b,"causal_damage_per_parameter_norm":2.0,"reliability":1.0}
            ]
        });
        fs::write(&a, serde_json::to_vec(&protection_payload).unwrap()).unwrap();
        let dense_map = build_protected_cortex_map(
            &[
                SensitivityEvidence {
                    probe_id: "p1".into(),
                    sensitivity: vec![1.0],
                    reliability: 1.0,
                    causal_damage: Some(1.0),
                },
                SensitivityEvidence {
                    probe_id: "p2".into(),
                    sensitivity: vec![2.0],
                    reliability: 1.0,
                    causal_damage: Some(2.0),
                },
            ],
            1.0,
            1.0,
        )
        .unwrap();
        let protected_map = persist_protected_map(&root, &dense_map).unwrap();
        let protected_map_path = root.join("protected_map_v2.json");
        fs::write(
            &protected_map_path,
            serde_json::to_vec(&serde_json::json!({
                "schema":"cerebro.tidex.protected_map_benchmark/v2",
                "task_labels_used":false,
                "map":protected_map
            }))
            .unwrap(),
        )
        .unwrap();
        let expected_trust = apply_causal_priority_trust_region(
            &Matrix::from_rows(&[vec![2.0]]).unwrap(),
            &[1.0],
            1.0,
            &causal_priority_weights,
        )
        .unwrap();
        let accepted_coefficients = expected_trust.accepted_coefficients.clone();
        let trust_payload = serde_json::json!({
            "schema":"cerebro.tidex.trust_region_benchmark/v3",
            "report_sha256":report_sha.clone(),
            "protected_map_sha256":sha(&protected_map_path),
            "causal_plan_sha256":plan_sha.clone(),
            "causal_credit_sha256":sha(&causal_credit),
            "causal_priority_weight_kind":"causal_lower_confidence_bound_95",
            "field_ids":["s"],
            "dense_parameter_dimension":1,
            "interaction_matrix":[[2.0]],
            "diagonal_budget":1.0,
            "trust_region":expected_trust.clone(),
        });
        fs::write(&interaction, serde_json::to_vec(&trust_payload).unwrap()).unwrap();
        let interaction_sha = sha(&interaction);
        let write_candidate = |scores: &[f64]| {
            assert_eq!(scores.len(), validation_seeds.len());
            let rows = validation_seeds
                .iter()
                .copied()
                .zip(scores.iter().copied())
                .map(|(seed, score)| {
                    serde_json::json!({
                        "seed":seed,
                        "metrics":{"task":score},
                        "mean":score,
                    })
                })
                .collect::<Vec<_>>();
            fs::write(
                &candidate,
                serde_json::to_vec(&serde_json::json!({
                    "schema":"cerebro.tidex.trust_region_functional_validation/v2",
                    "report_sha256":report_sha.clone(),
                    "trust_region_sha256":interaction_sha.clone(),
                    "plan_sha256":plan_sha.clone(),
                    "field_ids":["s"],
                    "accepted_coefficients":accepted_coefficients.clone(),
                    "validation_seeds":validation_seeds.clone(),
                    "validation_tasks":validation_tasks.clone(),
                    "count_per_task":100,
                    "rows":rows,
                    "blind_data_accessed":false,
                }))
                .unwrap(),
            )
            .unwrap();
        };
        write_candidate(&[0.45, 0.45, 0.45]);
        let mut bundle = SleepEvidenceBundle {
            schema: "cerebro.tidex.sleep_evidence/v4".into(),
            corpus_digest: CorpusDigest::from(Sha256Digest::digest_bytes(b"x")),
            source_tree_digest: SourceTreeDigest::from(Sha256Digest::digest_bytes(b"source")),
            config_digest: ConfigDigest::from(Sha256Digest::digest_bytes(b"config")),
            analysis_version_digest: AnalysisVersionDigest::from(Sha256Digest::digest_bytes(
                b"analysis",
            )),
            field_ids: vec![SkillId::parse("s").unwrap()],
            protection: ProtectionEvidenceSummary {
                source_path: a.to_string_lossy().into(),
                source_sha256: sha(&a),
                protected_map_path: protected_map_path.to_string_lossy().into(),
                protected_map_sha256: ProtectedMapDigest::from(sha(&protected_map_path)),
                probe_count: 2,
                parameter_dimension: 1,
                selected_rank: 1,
                causal_damage_supported_probes: 2,
                sensitivity_damage_correlation: Some(1.0),
            },
            interaction: InteractionEvidenceSummary {
                source_path: interaction.to_string_lossy().into(),
                source_sha256: interaction_sha.clone(),
                causal_credit_sha256: CausalCreditDigest::from(sha(&causal_credit)),
                field_ids: vec![SkillId::parse("s").unwrap()],
                dense_parameter_dimension: 1,
                proposed_quadratic_cost: 2.0,
                accepted_quadratic_cost: 1.0,
                diagonal_budget: 1.0,
                trust_scale: 2.0_f64.sqrt().recip(),
                allocation_policy: TrustRegionAllocationPolicy::CausalPriorityContractionV1,
                causal_priority_weights,
                component_retention: expected_trust.component_retention.clone(),
            },
            replay: ReplayEvidenceSummary {
                baseline_source_path: baseline.to_string_lossy().into(),
                baseline_source_sha256: sha(&baseline),
                candidate_source_path: candidate.to_string_lossy().into(),
                candidate_source_sha256: sha(&candidate),
                paired_independent_groups: 3,
                mean_utility_delta: -0.05,
                standard_error: 0.0,
                max_acceptable_utility_loss: Some(0.01),
                blind_data_accessed: false,
            },
            causal_credit: CausalCreditEvidenceSummary {
                replay_source_path: baseline.to_string_lossy().into(),
                replay_source_sha256: sha(&baseline),
                credit_source_path: causal_credit.to_string_lossy().into(),
                credit_source_sha256: CausalCreditDigest::from(sha(&causal_credit)),
                report_sha256: report_sha.clone(),
                plan_sha256: plan_sha.clone(),
                independent_group_count: 3,
                field_count: 1,
                resolved_field_count: 1,
                interaction_count: 0,
                resolved_interaction_count: 0,
                blind_data_accessed: false,
            },
        };
        let expected_fields = [field()];
        let dense_update = create_content_addressed_dvec(&root, &[(2.0f32).sqrt()]).unwrap();
        let observations = [DeltaObservation {
            observation_id: crate::identity::ObservationId::parse("o1").unwrap(),
            from_checkpoint: "base".into(),
            to_checkpoint: "next".into(),
            generation: 1,
            delta: vec![1.0],
            functional_response: vec![],
            confounders: vec![],
            reliability: 1.0,
            independence_group: "g1".into(),
            experiment_lineage: ExperimentLineage::default(),
            dense_artifact: Some(dense_update),
            parameter_layout_sha256: None,
            representation_artifact: None,
            representation_protocol_sha256: None,
            provenance_digest: crate::digest::ProvenanceDigest::from(
                crate::digest::Sha256Digest::digest_bytes(b"test"),
            ),
        }];
        let source_mixtures = [vec![1.0]];
        let expectation = SleepEvidenceExpectation {
            corpus_digest: &bundle.corpus_digest,
            report_sha256: &report_sha,
            source_tree_digest: &bundle.source_tree_digest,
            config_digest: &bundle.config_digest,
            analysis_version_digest: &bundle.analysis_version_digest,
            fields: &expected_fields,
            observations: &observations,
            source_mixtures: &source_mixtures,
        };
        let bad = verify_sleep_evidence(&root, &bundle, &expectation).unwrap();
        assert!(!bad.verified);
        assert!(!bad.replay_verified);
        assert!(
            bad.reasons
                .iter()
                .any(|reason| reason == "sleep_functional_replay_noninferior_unproven"),
            "unexpected verification reasons: {:?}",
            bad.reasons
        );

        write_candidate(&[0.495, 0.495, 0.495]);
        bundle.replay.candidate_source_sha256 = sha(&candidate);
        bundle.replay.mean_utility_delta = -0.005;
        bundle.replay.standard_error = 0.0;
        let good = verify_sleep_evidence(&root, &bundle, &expectation).unwrap();
        assert!(good.verified);
        assert!(good.replay_verified);
        assert!(good.protection_verified);
        assert!(good.interaction_verified);
        assert!(good.causal_credit_verified);
        assert!(good.replay_lower_confidence_bound >= -0.01);

        let mut unknown_field = serde_json::to_value(&bundle).unwrap();
        unknown_field
            .as_object_mut()
            .unwrap()
            .insert("uncommitted_digest".into(), true.into());
        assert!(serde_json::from_value::<SleepEvidenceBundle>(unknown_field).is_err());

        let mut cross_domain = bundle.clone();
        cross_domain.protection.protected_map_sha256 = ProtectedMapDigest::from(
            cross_domain
                .causal_credit
                .credit_source_sha256
                .as_digest()
                .clone(),
        );
        let crossed = verify_sleep_evidence(&root, &cross_domain, &expectation).unwrap();
        assert!(!crossed.verified);
        assert!(!crossed.protection_verified);

        let mut causal_priority_forgery = bundle.clone();
        causal_priority_forgery.interaction.causal_priority_weights = vec![999.0];
        let causal_priority_verification =
            verify_sleep_evidence(&root, &causal_priority_forgery, &expectation).unwrap();
        assert!(!causal_priority_verification.verified);
        assert!(!causal_priority_verification.interaction_verified);
        assert!(causal_priority_verification
            .reasons
            .iter()
            .any(|reason| reason == "sleep_interaction_trust_evidence_unverified"));

        let mut forged = bundle.clone();
        forged.replay.mean_utility_delta = 0.0;
        let forged_verification = verify_sleep_evidence(&root, &forged, &expectation).unwrap();
        assert!(!forged_verification.verified);
        assert!(!forged_verification.replay_verified);
        assert!(forged_verification
            .reasons
            .iter()
            .any(|reason| reason == "sleep_functional_replay_summary_mismatch"));
        let _ = fs::remove_dir_all(root);
    }
}
