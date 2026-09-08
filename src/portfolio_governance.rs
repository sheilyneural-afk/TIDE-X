//! Universal governance for candidate evolution in TIDE-X.
//!
//! This module derives conservative candidate decisions from paired raw
//! observations. It never executes, persists, promotes, or activates a
//! candidate. Hard invariants precede ranking; independent groups, not repeat
//! rows, are the unit of evidence; and every resource loop is bounded.

use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::finite::FiniteF64;
use crate::linalg::stable_rms;
use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;

const REPORT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:PAIRED-EVALUATION:v1\0";
const GATE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:CANDIDATE-GATE:v1\0";
const METRIC_CATALOG_DOMAIN: &[u8] = b"CEREBRO:TIDEX:METRIC-CATALOG:v1\0";
const ROBUST_POLICY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:ROBUST-EVALUATION-POLICY:v1\0";
const OBSERVATION_SET_DOMAIN: &[u8] = b"CEREBRO:TIDEX:PAIRED-OBSERVATIONS:v1\0";
const INDEPENDENCE_DESIGN_DOMAIN: &[u8] = b"CEREBRO:TIDEX:INDEPENDENCE-DESIGN:v1\0";
const GATE_POLICY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:CANDIDATE-GATE-POLICY:v1\0";
const PETFC_TRAJECTORY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:PETFC-TRAJECTORY:v1\0";
const PETFC_POLICY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:PETFC-POLICY:v1\0";
const PETFC_ASSESSMENT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:PETFC-ASSESSMENT:v1\0";
const PARETO_SELECTION_DOMAIN: &[u8] = b"CEREBRO:TIDEX:PARETO-SELECTION:v1\0";
const BUDGET_POLICY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:ADAPTIVE-BUDGET-POLICY:v1\0";
const BUDGET_PLAN_DOMAIN: &[u8] = b"CEREBRO:TIDEX:ADAPTIVE-BUDGET-PLAN:v1\0";
const CANARY_POLICY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:CANARY-POLICY:v1\0";
const CANARY_STATE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:CANARY-STATE:v1\0";
const CANARY_BUCKET_DOMAIN: &[u8] = b"CEREBRO:TIDEX:CANARY-BUCKET:v1\0";
const MAX_ID_BYTES: usize = 128;
const MAX_METRICS: usize = 128;
const MAX_OBSERVATIONS: usize = 1_000_000;
const MAX_GROUPS: usize = 65_536;
const MAX_PAIRS_PER_GROUP: usize = 65_536;
const MAX_CANDIDATES: usize = 4_096;
const MAX_TRAJECTORY_POINTS: usize = 4_096;
const MAX_CANARY_STAGES: usize = 32;
const MAX_GATE_HISTORY_PER_CANDIDATE: usize = 4_096;
const MAX_TOTAL_GATE_HISTORY: usize = 65_536;
const MAX_PARETO_WORK_UNITS: usize = 4_000_000;
const MAX_ADAPTIVE_WORK_UNITS: usize = 4_000_000;
const PER_MILLION: u32 = 1_000_000;
const NUMERICAL_EPSILON: f64 = 1.0e-12;

fn invalid(code: &str) -> BrainError {
    BrainError::Invalid(code.into())
}

fn integrity(code: &str) -> BrainError {
    BrainError::Integrity(code.into())
}

fn validate_id(value: &str, code: &str) -> BrainResult<()> {
    if value.is_empty()
        || value.len() > MAX_ID_BYTES
        || value.starts_with('.')
        || value.ends_with('.')
        || value.contains("..")
        || value.bytes().any(|byte| {
            !(byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
        })
    {
        return Err(invalid(code));
    }
    Ok(())
}

macro_rules! typed_id {
    ($name:ident, $code:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn parse(value: impl Into<String>) -> BrainResult<Self> {
                let value = value.into();
                validate_id(&value, $code)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::parse(value).map_err(D::Error::custom)
            }
        }
    };
}

macro_rules! sealed_digest {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
        #[serde(transparent)]
        pub struct $name(Sha256Digest);

        impl $name {
            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }

            fn from_projection(domain: &[u8], projection: &[u8]) -> Self {
                Self(Sha256Digest::digest_domain(domain, projection))
            }
        }
    };
}

typed_id!(MetricId, "governance_metric_id_invalid");
typed_id!(VariantId, "governance_variant_id_invalid");
typed_id!(EvidenceId, "governance_evidence_id_invalid");
typed_id!(TrajectoryId, "governance_trajectory_id_invalid");
typed_id!(IndependenceGroupId, "governance_group_id_invalid");
typed_id!(PairId, "governance_pair_id_invalid");
typed_id!(CanarySubjectId, "governance_canary_subject_id_invalid");

sealed_digest!(MetricCatalogDigest);
sealed_digest!(RobustEvaluationPolicyDigest);
sealed_digest!(ObservationSetDigest);
sealed_digest!(IndependenceDesignDigest);
sealed_digest!(PairedEvaluationDigest);
sealed_digest!(CandidateGatePolicyDigest);
sealed_digest!(CandidateGateDigest);
sealed_digest!(PetfcTrajectoryDigest);
sealed_digest!(PetfcPolicyDigest);
sealed_digest!(PetfcAssessmentDigest);
sealed_digest!(ParetoSelectionDigest);
sealed_digest!(AdaptiveBudgetPolicyDigest);
sealed_digest!(AdaptiveBudgetPlanDigest);
sealed_digest!(CanaryPolicyDigest);
sealed_digest!(CanaryStateDigest);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    Maximize,
    Minimize,
}

impl MetricDirection {
    fn orient(self, value: f64) -> f64 {
        match self {
            Self::Maximize => value,
            Self::Minimize => -value,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HardInvariant {
    AtLeast {
        threshold: FiniteF64,
    },
    AtMost {
        threshold: FiniteF64,
    },
    Within {
        minimum: FiniteF64,
        maximum: FiniteF64,
    },
}

impl HardInvariant {
    pub fn at_least(threshold: f64) -> BrainResult<Self> {
        Ok(Self::AtLeast {
            threshold: FiniteF64::new(threshold)?,
        })
    }

    pub fn at_most(threshold: f64) -> BrainResult<Self> {
        Ok(Self::AtMost {
            threshold: FiniteF64::new(threshold)?,
        })
    }

    pub fn within(minimum: f64, maximum: f64) -> BrainResult<Self> {
        let minimum = FiniteF64::new(minimum)?;
        let maximum = FiniteF64::new(maximum)?;
        if minimum.get() > maximum.get() {
            return Err(invalid("governance_invariant_interval_invalid"));
        }
        Ok(Self::Within { minimum, maximum })
    }

    fn validate(&self) -> BrainResult<()> {
        if let Self::Within { minimum, maximum } = self {
            if minimum.get() > maximum.get() {
                return Err(invalid("governance_invariant_interval_invalid"));
            }
        }
        Ok(())
    }

    fn classify_interval(&self, lower: f64, upper: f64) -> InvariantIntervalStatus {
        match self {
            Self::AtLeast { threshold } if upper < threshold.get() => {
                InvariantIntervalStatus::ProvenViolation
            }
            Self::AtLeast { threshold } if lower >= threshold.get() => {
                InvariantIntervalStatus::Established
            }
            Self::AtMost { threshold } if lower > threshold.get() => {
                InvariantIntervalStatus::ProvenViolation
            }
            Self::AtMost { threshold } if upper <= threshold.get() => {
                InvariantIntervalStatus::Established
            }
            Self::Within { minimum, .. } if upper < minimum.get() => {
                InvariantIntervalStatus::ProvenViolation
            }
            Self::Within { maximum, .. } if lower > maximum.get() => {
                InvariantIntervalStatus::ProvenViolation
            }
            Self::Within { minimum, maximum }
                if lower >= minimum.get() && upper <= maximum.get() =>
            {
                InvariantIntervalStatus::Established
            }
            _ => InvariantIntervalStatus::NotEstablished,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InvariantIntervalStatus {
    Established,
    ProvenViolation,
    NotEstablished,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MetricSpec {
    metric_id: MetricId,
    direction: MetricDirection,
    hard_invariant: Option<HardInvariant>,
}

impl MetricSpec {
    pub fn new(
        metric_id: MetricId,
        direction: MetricDirection,
        hard_invariant: Option<HardInvariant>,
    ) -> BrainResult<Self> {
        if let Some(invariant) = &hard_invariant {
            invariant.validate()?;
        }
        Ok(Self {
            metric_id,
            direction,
            hard_invariant,
        })
    }

    pub fn metric_id(&self) -> &MetricId {
        &self.metric_id
    }

    pub fn direction(&self) -> MetricDirection {
        self.direction
    }

    pub fn hard_invariant(&self) -> Option<&HardInvariant> {
        self.hard_invariant.as_ref()
    }
}

fn canonical_specs(specs: &[MetricSpec]) -> BrainResult<BTreeMap<MetricId, MetricSpec>> {
    if specs.is_empty() || specs.len() > MAX_METRICS {
        return Err(invalid("governance_metric_catalog_size_invalid"));
    }
    let mut result = BTreeMap::new();
    for spec in specs {
        if let Some(invariant) = &spec.hard_invariant {
            invariant.validate()?;
        }
        if result
            .insert(spec.metric_id.clone(), spec.clone())
            .is_some()
        {
            return Err(invalid("governance_metric_id_duplicate"));
        }
    }
    Ok(result)
}

fn metric_catalog_digest(
    catalog: &BTreeMap<MetricId, MetricSpec>,
) -> BrainResult<MetricCatalogDigest> {
    Ok(MetricCatalogDigest::from_projection(
        METRIC_CATALOG_DOMAIN,
        &serde_json::to_vec(catalog)?,
    ))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationWindow {
    start_tick: u64,
    end_tick: u64,
}

impl ObservationWindow {
    pub fn new(start_tick: u64, end_tick: u64) -> BrainResult<Self> {
        if end_tick < start_tick {
            return Err(invalid("governance_observation_window_invalid"));
        }
        Ok(Self {
            start_tick,
            end_tick,
        })
    }

    pub fn start_tick(&self) -> u64 {
        self.start_tick
    }

    pub fn end_tick(&self) -> u64 {
        self.end_tick
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairedExperimentalUnit {
    independence_group: IndependenceGroupId,
    pair_id: PairId,
    evidence_id: EvidenceId,
    observation_window: ObservationWindow,
}

impl PairedExperimentalUnit {
    pub fn new(
        independence_group: IndependenceGroupId,
        pair_id: PairId,
        evidence_id: EvidenceId,
        observation_window: ObservationWindow,
    ) -> Self {
        Self {
            independence_group,
            pair_id,
            evidence_id,
            observation_window,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairedObservation {
    metric_id: MetricId,
    unit: PairedExperimentalUnit,
    baseline_value: FiniteF64,
    candidate_value: FiniteF64,
}

impl PairedObservation {
    pub fn new(
        metric_id: MetricId,
        unit: PairedExperimentalUnit,
        baseline_value: f64,
        candidate_value: f64,
    ) -> BrainResult<Self> {
        Ok(Self {
            metric_id,
            unit,
            baseline_value: FiniteF64::new(baseline_value)?,
            candidate_value: FiniteF64::new(candidate_value)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobustEvaluationPolicy {
    minimum_independent_groups: usize,
    median_of_means_blocks: usize,
    maximum_observations: usize,
    policy_digest: RobustEvaluationPolicyDigest,
}

#[derive(Serialize)]
struct RobustEvaluationPolicyProjection {
    minimum_independent_groups: usize,
    median_of_means_blocks: usize,
    maximum_observations: usize,
}

impl RobustEvaluationPolicy {
    pub fn new(
        minimum_independent_groups: usize,
        median_of_means_blocks: usize,
        maximum_observations: usize,
    ) -> BrainResult<Self> {
        let projection = RobustEvaluationPolicyProjection {
            minimum_independent_groups,
            median_of_means_blocks,
            maximum_observations,
        };
        if projection.minimum_independent_groups < 2
            || projection.minimum_independent_groups > MAX_GROUPS
            || projection.median_of_means_blocks < 2
            || projection.median_of_means_blocks > projection.minimum_independent_groups
            || projection.maximum_observations == 0
            || projection.maximum_observations > MAX_OBSERVATIONS
        {
            return Err(invalid("governance_robust_policy_invalid"));
        }
        let policy_digest = RobustEvaluationPolicyDigest::from_projection(
            ROBUST_POLICY_DOMAIN,
            &serde_json::to_vec(&projection)?,
        );
        Ok(Self {
            minimum_independent_groups,
            median_of_means_blocks,
            maximum_observations,
            policy_digest,
        })
    }

    pub fn conservative_default() -> BrainResult<Self> {
        Self::new(3, 3, 100_000)
    }

    pub fn digest(&self) -> &RobustEvaluationPolicyDigest {
        &self.policy_digest
    }

    pub fn minimum_independent_groups(&self) -> usize {
        self.minimum_independent_groups
    }

    pub fn maximum_observations(&self) -> usize {
        self.maximum_observations
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairedMetricEstimate {
    metric_id: MetricId,
    independent_groups: usize,
    paired_count: usize,
    baseline_center: FiniteF64,
    candidate_center: FiniteF64,
    paired_effect: FiniteF64,
    baseline_observed_radius: Option<FiniteF64>,
    candidate_observed_radius: Option<FiniteF64>,
    paired_effect_observed_radius: Option<FiniteF64>,
}

impl PairedMetricEstimate {
    pub fn independent_groups(&self) -> usize {
        self.independent_groups
    }

    pub fn baseline_center(&self) -> f64 {
        self.baseline_center.get()
    }

    pub fn candidate_center(&self) -> f64 {
        self.candidate_center.get()
    }

    pub fn paired_effect(&self) -> f64 {
        self.paired_effect.get()
    }

    pub fn baseline_observed_radius(&self) -> Option<f64> {
        self.baseline_observed_radius.map(FiniteF64::get)
    }

    pub fn candidate_observed_radius(&self) -> Option<f64> {
        self.candidate_observed_radius.map(FiniteF64::get)
    }

    pub fn paired_effect_observed_radius(&self) -> Option<f64> {
        self.paired_effect_observed_radius.map(FiniteF64::get)
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairedEvaluationReport {
    baseline_id: VariantId,
    candidate_id: VariantId,
    metric_catalog_digest: MetricCatalogDigest,
    evaluation_policy_digest: RobustEvaluationPolicyDigest,
    observation_set_digest: ObservationSetDigest,
    independence_design_digest: IndependenceDesignDigest,
    estimates: BTreeMap<MetricId, PairedMetricEstimate>,
    pair_ids: BTreeSet<PairId>,
    evidence_ids: BTreeSet<EvidenceId>,
    observation_window: ObservationWindow,
    observation_count: usize,
    independent_group_count: usize,
    minimum_required_groups: usize,
    evidence_sufficient: bool,
    report_digest: PairedEvaluationDigest,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct ReportProjection<'a> {
    baseline_id: &'a VariantId,
    candidate_id: &'a VariantId,
    metric_catalog_digest: &'a MetricCatalogDigest,
    evaluation_policy_digest: &'a RobustEvaluationPolicyDigest,
    observation_set_digest: &'a ObservationSetDigest,
    independence_design_digest: &'a IndependenceDesignDigest,
    estimates: &'a BTreeMap<MetricId, PairedMetricEstimate>,
    pair_ids: &'a BTreeSet<PairId>,
    evidence_ids: &'a BTreeSet<EvidenceId>,
    observation_window: &'a ObservationWindow,
    observation_count: usize,
    independent_group_count: usize,
    minimum_required_groups: usize,
    evidence_sufficient: bool,
}

impl PairedEvaluationReport {
    pub fn baseline_id(&self) -> &VariantId {
        &self.baseline_id
    }

    pub fn candidate_id(&self) -> &VariantId {
        &self.candidate_id
    }

    pub fn estimates(&self) -> &BTreeMap<MetricId, PairedMetricEstimate> {
        &self.estimates
    }

    pub fn digest(&self) -> &PairedEvaluationDigest {
        &self.report_digest
    }

    pub fn metric_catalog_digest(&self) -> &MetricCatalogDigest {
        &self.metric_catalog_digest
    }

    pub fn evaluation_policy_digest(&self) -> &RobustEvaluationPolicyDigest {
        &self.evaluation_policy_digest
    }

    pub fn observation_window(&self) -> &ObservationWindow {
        &self.observation_window
    }

    pub fn evidence_sufficient(&self) -> bool {
        self.evidence_sufficient
    }

    pub fn independent_group_count(&self) -> usize {
        self.independent_group_count
    }

    /// Identity of the declared independent-group set, without metric values,
    /// candidate identity, repetitions, or observation time. This proves
    /// stable binding and detects reuse; it does not by itself prove that a
    /// caller-chosen label denotes a statistically independent source.
    pub fn independence_design_digest(&self) -> &IndependenceDesignDigest {
        &self.independence_design_digest
    }

    fn projection(&self) -> ReportProjection<'_> {
        ReportProjection {
            baseline_id: &self.baseline_id,
            candidate_id: &self.candidate_id,
            metric_catalog_digest: &self.metric_catalog_digest,
            evaluation_policy_digest: &self.evaluation_policy_digest,
            observation_set_digest: &self.observation_set_digest,
            independence_design_digest: &self.independence_design_digest,
            estimates: &self.estimates,
            pair_ids: &self.pair_ids,
            evidence_ids: &self.evidence_ids,
            observation_window: &self.observation_window,
            observation_count: self.observation_count,
            independent_group_count: self.independent_group_count,
            minimum_required_groups: self.minimum_required_groups,
            evidence_sufficient: self.evidence_sufficient,
        }
    }

    fn authenticate(&self) -> BrainResult<()> {
        let calculated = PairedEvaluationDigest::from_projection(
            REPORT_DOMAIN,
            &serde_json::to_vec(&self.projection())?,
        );
        if calculated != self.report_digest {
            return Err(integrity("governance_report_digest_mismatch"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CanonicalObservation {
    metric_id: MetricId,
    unit: PairedExperimentalUnit,
    baseline_value: FiniteF64,
    candidate_value: FiniteF64,
}

#[derive(Debug, Clone)]
struct PairedValues {
    unit: PairedExperimentalUnit,
    baseline: f64,
    candidate: f64,
}

/// Reduce already-authorized paired observations to conservative group-level
/// envelopes. This stays crate-private because arbitrary caller-supplied group
/// labels are not proof of independence; the owning evaluation adapter must
/// establish that provenance before invoking this reducer.
pub(crate) fn evaluate_paired_groups(
    baseline_id: VariantId,
    candidate_id: VariantId,
    specs: &[MetricSpec],
    observations: &[PairedObservation],
    policy: &RobustEvaluationPolicy,
) -> BrainResult<PairedEvaluationReport> {
    if baseline_id == candidate_id {
        return Err(invalid("governance_candidate_equals_baseline"));
    }
    let specs = canonical_specs(specs)?;
    let metric_catalog_digest = metric_catalog_digest(&specs)?;
    if observations.is_empty()
        || observations.len() > policy.maximum_observations
        || observations.len() > MAX_OBSERVATIONS
    {
        return Err(invalid("governance_observation_count_invalid"));
    }
    let mut canonical_observations = observations
        .iter()
        .map(|observation| CanonicalObservation {
            metric_id: observation.metric_id.clone(),
            unit: observation.unit.clone(),
            baseline_value: observation.baseline_value,
            candidate_value: observation.candidate_value,
        })
        .collect::<Vec<_>>();
    canonical_observations.sort_by(|left, right| {
        (
            &left.metric_id,
            &left.unit.pair_id,
            &left.unit.independence_group,
        )
            .cmp(&(
                &right.metric_id,
                &right.unit.pair_id,
                &right.unit.independence_group,
            ))
    });
    let observation_set_digest = ObservationSetDigest::from_projection(
        OBSERVATION_SET_DOMAIN,
        &serde_json::to_vec(&canonical_observations)?,
    );

    let mut pair_registry =
        BTreeMap::<PairId, (IndependenceGroupId, EvidenceId, ObservationWindow)>::new();
    let mut evidence_registry = BTreeMap::<EvidenceId, PairId>::new();
    let mut by_metric = BTreeMap::<MetricId, BTreeMap<PairId, PairedValues>>::new();
    for observation in observations {
        if !specs.contains_key(&observation.metric_id) {
            return Err(invalid("governance_observation_metric_unknown"));
        }
        let binding = (
            observation.unit.independence_group.clone(),
            observation.unit.evidence_id.clone(),
            observation.unit.observation_window.clone(),
        );
        if let Some(existing) = pair_registry.get(&observation.unit.pair_id) {
            if existing != &binding {
                return Err(integrity("governance_pair_identity_relabelled"));
            }
        } else {
            pair_registry.insert(observation.unit.pair_id.clone(), binding);
        }
        if let Some(existing_pair) = evidence_registry.get(&observation.unit.evidence_id) {
            if existing_pair != &observation.unit.pair_id {
                return Err(integrity("governance_evidence_reused_across_pairs"));
            }
        } else {
            evidence_registry.insert(
                observation.unit.evidence_id.clone(),
                observation.unit.pair_id.clone(),
            );
        }
        if by_metric
            .entry(observation.metric_id.clone())
            .or_default()
            .insert(
                observation.unit.pair_id.clone(),
                PairedValues {
                    unit: observation.unit.clone(),
                    baseline: observation.baseline_value.get(),
                    candidate: observation.candidate_value.get(),
                },
            )
            .is_some()
        {
            return Err(integrity("governance_paired_observation_duplicate"));
        }
    }
    if by_metric.len() != specs.len() {
        return Err(integrity("governance_metric_observations_missing"));
    }
    let expected_pairs = by_metric
        .values()
        .next()
        .map(|pairs| pairs.keys().cloned().collect::<BTreeSet<_>>())
        .ok_or_else(|| integrity("governance_metric_observations_missing"))?;
    if expected_pairs.is_empty()
        || by_metric
            .values()
            .any(|pairs| pairs.keys().cloned().collect::<BTreeSet<_>>() != expected_pairs)
    {
        return Err(integrity("governance_cross_metric_pairing_mismatch"));
    }
    let groups = pair_registry
        .values()
        .map(|(group, _, _)| group)
        .collect::<BTreeSet<_>>();
    if groups.len() > MAX_GROUPS {
        return Err(invalid("governance_group_count_exceeded"));
    }
    let mut pairs_per_group = BTreeMap::<&IndependenceGroupId, usize>::new();
    for pair_id in &expected_pairs {
        let group = &pair_registry
            .get(pair_id)
            .ok_or_else(|| integrity("governance_pair_registry_missing"))?
            .0;
        *pairs_per_group.entry(group).or_default() += 1;
    }
    if pairs_per_group
        .values()
        .any(|count| *count > MAX_PAIRS_PER_GROUP)
    {
        return Err(invalid("governance_pairs_per_group_exceeded"));
    }
    let independence_design_digest = IndependenceDesignDigest::from_projection(
        INDEPENDENCE_DESIGN_DOMAIN,
        &serde_json::to_vec(&groups)?,
    );

    let mut estimates = BTreeMap::new();
    for metric_id in specs.keys() {
        let pairs = by_metric
            .get(metric_id)
            .ok_or_else(|| integrity("governance_metric_observations_missing"))?;
        let mut grouped = BTreeMap::<IndependenceGroupId, Vec<(f64, f64)>>::new();
        for values in pairs.values() {
            grouped
                .entry(values.unit.independence_group.clone())
                .or_default()
                .push((values.baseline, values.candidate));
        }
        let mut baselines = Vec::with_capacity(grouped.len());
        let mut candidates = Vec::with_capacity(grouped.len());
        let mut effects = Vec::with_capacity(grouped.len());
        for values in grouped.values() {
            let baseline = stable_mean(values.iter().map(|(value, _)| *value))?;
            let candidate = stable_mean(values.iter().map(|(_, value)| *value))?;
            // Preserve the paired design numerically: average within-pair
            // differences directly instead of subtracting two large group
            // means and losing a small but representable effect.
            let effect = stable_mean(
                values
                    .iter()
                    .map(|(baseline, candidate)| *candidate - *baseline),
            )?;
            baselines.push(baseline);
            candidates.push(candidate);
            effects.push(effect);
        }
        let blocks = policy.median_of_means_blocks.min(grouped.len()).max(1);
        let baseline_center = median_of_means(&baselines, blocks)?;
        let candidate_center = median_of_means(&candidates, blocks)?;
        let paired_effect = median_of_means(&effects, blocks)?;
        let baseline_radius = observed_radius(&baselines, baseline_center)?;
        let candidate_radius = observed_radius(&candidates, candidate_center)?;
        let effect_radius = observed_radius(&effects, paired_effect)?;
        estimates.insert(
            metric_id.clone(),
            PairedMetricEstimate {
                metric_id: metric_id.clone(),
                independent_groups: grouped.len(),
                paired_count: pairs.len(),
                baseline_center: FiniteF64::new(baseline_center)?,
                candidate_center: FiniteF64::new(candidate_center)?,
                paired_effect: FiniteF64::new(paired_effect)?,
                baseline_observed_radius: baseline_radius.map(FiniteF64::new).transpose()?,
                candidate_observed_radius: candidate_radius.map(FiniteF64::new).transpose()?,
                paired_effect_observed_radius: effect_radius.map(FiniteF64::new).transpose()?,
            },
        );
    }
    let pair_ids = pair_registry.keys().cloned().collect::<BTreeSet<_>>();
    let evidence_ids = evidence_registry.keys().cloned().collect::<BTreeSet<_>>();
    let start_tick = pair_registry
        .values()
        .map(|(_, _, window)| window.start_tick)
        .min()
        .ok_or_else(|| integrity("governance_observation_window_missing"))?;
    let end_tick = pair_registry
        .values()
        .map(|(_, _, window)| window.end_tick)
        .max()
        .ok_or_else(|| integrity("governance_observation_window_missing"))?;
    let observation_window = ObservationWindow::new(start_tick, end_tick)?;
    let independent_group_count = groups.len();
    let evidence_sufficient = independent_group_count >= policy.minimum_independent_groups;
    let projection = ReportProjection {
        baseline_id: &baseline_id,
        candidate_id: &candidate_id,
        metric_catalog_digest: &metric_catalog_digest,
        evaluation_policy_digest: &policy.policy_digest,
        observation_set_digest: &observation_set_digest,
        independence_design_digest: &independence_design_digest,
        estimates: &estimates,
        pair_ids: &pair_ids,
        evidence_ids: &evidence_ids,
        observation_window: &observation_window,
        observation_count: observations.len(),
        independent_group_count,
        minimum_required_groups: policy.minimum_independent_groups,
        evidence_sufficient,
    };
    let report_digest =
        PairedEvaluationDigest::from_projection(REPORT_DOMAIN, &serde_json::to_vec(&projection)?);
    let report = PairedEvaluationReport {
        baseline_id,
        candidate_id,
        metric_catalog_digest,
        evaluation_policy_digest: policy.policy_digest.clone(),
        observation_set_digest,
        independence_design_digest,
        estimates,
        pair_ids,
        evidence_ids,
        observation_window,
        observation_count: observations.len(),
        independent_group_count,
        minimum_required_groups: policy.minimum_independent_groups,
        evidence_sufficient,
        report_digest,
    };
    report.authenticate()?;
    Ok(report)
}

fn stable_mean(values: impl IntoIterator<Item = f64>) -> BrainResult<f64> {
    let values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("governance_mean_input_invalid"));
    }
    let count =
        u64::try_from(values.len()).map_err(|_| invalid("governance_mean_count_overflow"))?;
    let scale = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Ok(0.0);
    }
    let mut sum = 0.0_f64;
    let mut correction = 0.0_f64;
    for value in values {
        let value = value / scale;
        let updated = sum + value;
        if sum.abs() >= value.abs() {
            correction += (sum - updated) + value;
        } else {
            correction += (value - updated) + sum;
        }
        sum = updated;
    }
    // Divide before rescaling so repeated large finite values do not overflow
    // merely while computing their equally large finite mean.
    let mean = ((sum + correction) / count as f64) * scale;
    if !mean.is_finite() {
        return Err(BrainError::Numerical("governance_mean_nonfinite".into()));
    }
    Ok(mean)
}

fn median_of_means(values: &[f64], blocks: usize) -> BrainResult<f64> {
    if values.is_empty() || blocks == 0 || blocks > values.len() {
        return Err(invalid("governance_median_of_means_shape"));
    }
    let mut partitions = vec![Vec::new(); blocks];
    for (index, value) in values.iter().enumerate() {
        partitions[index % blocks].push(*value);
    }
    let mut means = partitions
        .into_iter()
        .map(stable_mean)
        .collect::<BrainResult<Vec<_>>>()?;
    median(&mut means)
}

fn median(values: &mut [f64]) -> BrainResult<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("governance_median_input_invalid"));
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        Ok(values[middle])
    } else {
        stable_mean([values[middle - 1], values[middle]])
    }
}

/// Maximum observed deviation of independent-group centers around the robust
/// center. This is deliberately an empirical envelope, not a claimed
/// population confidence interval.
fn observed_radius(values: &[f64], center: f64) -> BrainResult<Option<f64>> {
    if values.is_empty() || !center.is_finite() {
        return Err(invalid("governance_observed_radius_input_invalid"));
    }
    if values.len() < 2 {
        return Ok(None);
    }
    let radius = values
        .iter()
        .map(|value| (value - center).abs())
        .max_by(f64::total_cmp)
        .ok_or_else(|| invalid("governance_observed_radius_empty"))?;
    if !radius.is_finite() {
        return Err(BrainError::Numerical(
            "governance_observed_radius_nonfinite".into(),
        ));
    }
    Ok(Some(radius))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateGatePolicy {
    metric_catalog_digest: MetricCatalogDigest,
    minimum_independent_groups: usize,
    uncertainty_multiplier: FiniteF64,
    minimum_improvement: BTreeMap<MetricId, FiniteF64>,
    policy_digest: CandidateGatePolicyDigest,
}

#[derive(Serialize)]
struct CandidateGatePolicyProjection<'a> {
    metric_catalog_digest: &'a MetricCatalogDigest,
    minimum_independent_groups: usize,
    uncertainty_multiplier: FiniteF64,
    minimum_improvement: &'a BTreeMap<MetricId, FiniteF64>,
}

impl CandidateGatePolicy {
    pub fn new(
        specs: &[MetricSpec],
        minimum_independent_groups: usize,
        uncertainty_multiplier: f64,
        minimum_improvement: BTreeMap<MetricId, FiniteF64>,
    ) -> BrainResult<Self> {
        let catalog = canonical_specs(specs)?;
        let metric_catalog_digest = metric_catalog_digest(&catalog)?;
        let uncertainty_multiplier = FiniteF64::new(uncertainty_multiplier)?;
        if !(2..=MAX_GROUPS).contains(&minimum_independent_groups)
            || !(0.0..=10.0).contains(&uncertainty_multiplier.get())
            || uncertainty_multiplier.get() == 0.0
            || minimum_improvement.len() != catalog.len()
            || minimum_improvement
                .keys()
                .any(|metric| !catalog.contains_key(metric))
        {
            return Err(invalid("governance_gate_policy_invalid"));
        }
        let projection = CandidateGatePolicyProjection {
            metric_catalog_digest: &metric_catalog_digest,
            minimum_independent_groups,
            uncertainty_multiplier,
            minimum_improvement: &minimum_improvement,
        };
        let policy_digest = CandidateGatePolicyDigest::from_projection(
            GATE_POLICY_DOMAIN,
            &serde_json::to_vec(&projection)?,
        );
        Ok(Self {
            metric_catalog_digest,
            minimum_independent_groups,
            uncertainty_multiplier,
            minimum_improvement,
            policy_digest,
        })
    }

    pub fn digest(&self) -> &CandidateGatePolicyDigest {
        &self.policy_digest
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "reason", rename_all = "snake_case", deny_unknown_fields)]
pub enum GateReason {
    ProvenHardViolation { metric_id: MetricId },
    HardInvariantNotEstablished { metric_id: MetricId },
    InsufficientGroups { metric_id: MetricId },
    ImprovementBelowThreshold { metric_id: MetricId },
    ImprovementUncertain { metric_id: MetricId },
    PersistentRegression { metric_id: MetricId },
    Oscillation { metric_id: MetricId },
    FunctionalDrift { metric_id: MetricId },
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CandidateGateDisposition {
    AdvanceCandidate,
    Reject,
    BoundedUnknown,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CandidateGateDecision {
    baseline_id: VariantId,
    candidate_id: VariantId,
    report_digest: PairedEvaluationDigest,
    metric_catalog_digest: MetricCatalogDigest,
    evaluation_policy_digest: RobustEvaluationPolicyDigest,
    independence_design_digest: IndependenceDesignDigest,
    source_evidence_ids: BTreeSet<EvidenceId>,
    source_observation_window: ObservationWindow,
    policy_digest: CandidateGatePolicyDigest,
    disposition: CandidateGateDisposition,
    reasons: BTreeSet<GateReason>,
    conservative_improvements: BTreeMap<MetricId, FiniteF64>,
    minimum_observed_groups: usize,
    decision_digest: CandidateGateDigest,
}

#[derive(Serialize)]
struct GateProjection<'a> {
    baseline_id: &'a VariantId,
    candidate_id: &'a VariantId,
    report_digest: &'a PairedEvaluationDigest,
    metric_catalog_digest: &'a MetricCatalogDigest,
    evaluation_policy_digest: &'a RobustEvaluationPolicyDigest,
    independence_design_digest: &'a IndependenceDesignDigest,
    source_evidence_ids: &'a BTreeSet<EvidenceId>,
    source_observation_window: &'a ObservationWindow,
    policy_digest: &'a CandidateGatePolicyDigest,
    disposition: CandidateGateDisposition,
    reasons: &'a BTreeSet<GateReason>,
    conservative_improvements: &'a BTreeMap<MetricId, FiniteF64>,
    minimum_observed_groups: usize,
}

impl CandidateGateDecision {
    pub fn candidate_id(&self) -> &VariantId {
        &self.candidate_id
    }

    pub fn disposition(&self) -> CandidateGateDisposition {
        self.disposition
    }

    pub fn reasons(&self) -> &BTreeSet<GateReason> {
        &self.reasons
    }

    pub fn digest(&self) -> &CandidateGateDigest {
        &self.decision_digest
    }

    pub fn report_digest(&self) -> &PairedEvaluationDigest {
        &self.report_digest
    }

    pub fn metric_catalog_digest(&self) -> &MetricCatalogDigest {
        &self.metric_catalog_digest
    }

    pub fn evaluation_policy_digest(&self) -> &RobustEvaluationPolicyDigest {
        &self.evaluation_policy_digest
    }

    pub fn independence_design_digest(&self) -> &IndependenceDesignDigest {
        &self.independence_design_digest
    }

    pub fn source_observation_window(&self) -> &ObservationWindow {
        &self.source_observation_window
    }

    pub fn conservative_improvements(&self) -> &BTreeMap<MetricId, FiniteF64> {
        &self.conservative_improvements
    }

    pub const fn authorizes_promotion(&self) -> bool {
        false
    }

    fn projection(&self) -> GateProjection<'_> {
        GateProjection {
            baseline_id: &self.baseline_id,
            candidate_id: &self.candidate_id,
            report_digest: &self.report_digest,
            metric_catalog_digest: &self.metric_catalog_digest,
            evaluation_policy_digest: &self.evaluation_policy_digest,
            independence_design_digest: &self.independence_design_digest,
            source_evidence_ids: &self.source_evidence_ids,
            source_observation_window: &self.source_observation_window,
            policy_digest: &self.policy_digest,
            disposition: self.disposition,
            reasons: &self.reasons,
            conservative_improvements: &self.conservative_improvements,
            minimum_observed_groups: self.minimum_observed_groups,
        }
    }

    fn authenticate(&self) -> BrainResult<()> {
        let calculated = CandidateGateDigest::from_projection(
            GATE_DOMAIN,
            &serde_json::to_vec(&self.projection())?,
        );
        if calculated != self.decision_digest {
            return Err(integrity("governance_gate_digest_mismatch"));
        }
        Ok(())
    }

    fn seal(draft: CandidateGateDecisionDraft) -> BrainResult<Self> {
        let CandidateGateDecisionDraft {
            baseline_id,
            candidate_id,
            report_digest,
            metric_catalog_digest,
            evaluation_policy_digest,
            independence_design_digest,
            source_evidence_ids,
            source_observation_window,
            policy_digest,
            disposition,
            reasons,
            conservative_improvements,
            minimum_observed_groups,
        } = draft;
        let projection = GateProjection {
            baseline_id: &baseline_id,
            candidate_id: &candidate_id,
            report_digest: &report_digest,
            metric_catalog_digest: &metric_catalog_digest,
            evaluation_policy_digest: &evaluation_policy_digest,
            independence_design_digest: &independence_design_digest,
            source_evidence_ids: &source_evidence_ids,
            source_observation_window: &source_observation_window,
            policy_digest: &policy_digest,
            disposition,
            reasons: &reasons,
            conservative_improvements: &conservative_improvements,
            minimum_observed_groups,
        };
        let decision_digest =
            CandidateGateDigest::from_projection(GATE_DOMAIN, &serde_json::to_vec(&projection)?);
        let decision = Self {
            baseline_id,
            candidate_id,
            report_digest,
            metric_catalog_digest,
            evaluation_policy_digest,
            independence_design_digest,
            source_evidence_ids,
            source_observation_window,
            policy_digest,
            disposition,
            reasons,
            conservative_improvements,
            minimum_observed_groups,
            decision_digest,
        };
        decision.authenticate()?;
        Ok(decision)
    }
}

struct CandidateGateDecisionDraft {
    baseline_id: VariantId,
    candidate_id: VariantId,
    report_digest: PairedEvaluationDigest,
    metric_catalog_digest: MetricCatalogDigest,
    evaluation_policy_digest: RobustEvaluationPolicyDigest,
    independence_design_digest: IndependenceDesignDigest,
    source_evidence_ids: BTreeSet<EvidenceId>,
    source_observation_window: ObservationWindow,
    policy_digest: CandidateGatePolicyDigest,
    disposition: CandidateGateDisposition,
    reasons: BTreeSet<GateReason>,
    conservative_improvements: BTreeMap<MetricId, FiniteF64>,
    minimum_observed_groups: usize,
}

pub fn decide_candidate(
    specs: &[MetricSpec],
    report: &PairedEvaluationReport,
    policy: &CandidateGatePolicy,
) -> BrainResult<CandidateGateDecision> {
    report.authenticate()?;
    let catalog = canonical_specs(specs)?;
    let catalog_digest = metric_catalog_digest(&catalog)?;
    if report.metric_catalog_digest != catalog_digest
        || policy.metric_catalog_digest != catalog_digest
        || policy.minimum_independent_groups < report.minimum_required_groups
        || policy.minimum_improvement.len() != catalog.len()
        || policy
            .minimum_improvement
            .keys()
            .any(|metric| !catalog.contains_key(metric))
        || report.estimates.len() != catalog.len()
        || report
            .estimates
            .keys()
            .any(|metric| !catalog.contains_key(metric))
    {
        return Err(integrity("governance_gate_metric_binding_mismatch"));
    }
    let multiplier = policy.uncertainty_multiplier.get();
    let mut hard = BTreeSet::new();
    let mut rejected = BTreeSet::new();
    let mut unknown = BTreeSet::new();
    let mut conservative_improvements = BTreeMap::new();
    let mut minimum_groups = usize::MAX;
    for (metric_id, spec) in &catalog {
        let estimate = report
            .estimates
            .get(metric_id)
            .ok_or_else(|| integrity("governance_gate_metric_missing"))?;
        minimum_groups = minimum_groups.min(estimate.independent_groups);
        let candidate_radius = estimate
            .candidate_observed_radius()
            .map(|value| multiplier * value);
        if let Some(invariant) = spec.hard_invariant.as_ref() {
            match candidate_radius {
                Some(radius) => match invariant.classify_interval(
                    estimate.candidate_center() - radius,
                    estimate.candidate_center() + radius,
                ) {
                    InvariantIntervalStatus::ProvenViolation => {
                        hard.insert(GateReason::ProvenHardViolation {
                            metric_id: metric_id.clone(),
                        });
                    }
                    InvariantIntervalStatus::NotEstablished => {
                        unknown.insert(GateReason::HardInvariantNotEstablished {
                            metric_id: metric_id.clone(),
                        });
                    }
                    InvariantIntervalStatus::Established => {}
                },
                None => {
                    unknown.insert(GateReason::InsufficientGroups {
                        metric_id: metric_id.clone(),
                    });
                }
            }
        }
        if estimate.independent_groups < policy.minimum_independent_groups {
            unknown.insert(GateReason::InsufficientGroups {
                metric_id: metric_id.clone(),
            });
            continue;
        }
        let Some(raw_effect_radius) = estimate.paired_effect_observed_radius() else {
            unknown.insert(GateReason::InsufficientGroups {
                metric_id: metric_id.clone(),
            });
            continue;
        };
        let effect_radius = multiplier * raw_effect_radius;
        let oriented = spec.direction.orient(estimate.paired_effect());
        let conservative = oriented - effect_radius;
        conservative_improvements.insert(metric_id.clone(), FiniteF64::new(conservative)?);
        let required = policy
            .minimum_improvement
            .get(metric_id)
            .ok_or_else(|| integrity("governance_gate_threshold_missing"))?
            .get();
        if conservative >= required {
            continue;
        }
        if oriented + effect_radius < required {
            rejected.insert(GateReason::ImprovementBelowThreshold {
                metric_id: metric_id.clone(),
            });
        } else {
            unknown.insert(GateReason::ImprovementUncertain {
                metric_id: metric_id.clone(),
            });
        }
    }
    let disposition = if !hard.is_empty() || !rejected.is_empty() {
        CandidateGateDisposition::Reject
    } else if !unknown.is_empty() {
        CandidateGateDisposition::BoundedUnknown
    } else {
        CandidateGateDisposition::AdvanceCandidate
    };
    // Preserve every detected fact for audit even when a higher-severity fact
    // determines the disposition.
    let reasons = hard.into_iter().chain(rejected).chain(unknown).collect();
    CandidateGateDecision::seal(CandidateGateDecisionDraft {
        baseline_id: report.baseline_id.clone(),
        candidate_id: report.candidate_id.clone(),
        report_digest: report.report_digest.clone(),
        metric_catalog_digest: catalog_digest,
        evaluation_policy_digest: report.evaluation_policy_digest.clone(),
        independence_design_digest: report.independence_design_digest.clone(),
        source_evidence_ids: report.evidence_ids.clone(),
        source_observation_window: report.observation_window.clone(),
        policy_digest: policy.policy_digest.clone(),
        disposition,
        reasons,
        conservative_improvements,
        minimum_observed_groups: minimum_groups,
    })
}

// ---------------------------------------------------------------------------
// PETFC: functional trajectory geometry and conservation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PetfcMetricPolicy {
    metric_id: MetricId,
    normalization_scale: FiniteF64,
    maximum_normalized_endpoint_degradation: FiniteF64,
}

impl PetfcMetricPolicy {
    pub fn new(
        metric_id: MetricId,
        normalization_scale: f64,
        maximum_normalized_endpoint_degradation: f64,
    ) -> BrainResult<Self> {
        let normalization_scale = FiniteF64::new(normalization_scale)?;
        let maximum_normalized_endpoint_degradation =
            FiniteF64::new(maximum_normalized_endpoint_degradation)?;
        if normalization_scale.get() <= 0.0 || maximum_normalized_endpoint_degradation.get() < 0.0 {
            return Err(invalid("petfc_metric_policy_invalid"));
        }
        Ok(Self {
            metric_id,
            normalization_scale,
            maximum_normalized_endpoint_degradation,
        })
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PetfcPathLimits {
    minimum_reports: usize,
    maximum_reports: usize,
    maximum_step_distance: FiniteF64,
    maximum_tortuosity: FiniteF64,
    maximum_waste: FiniteF64,
    minimum_path_efficiency: FiniteF64,
}

impl PetfcPathLimits {
    pub fn new(
        minimum_reports: usize,
        maximum_reports: usize,
        maximum_step_distance: f64,
        maximum_tortuosity: f64,
        maximum_waste: f64,
        minimum_path_efficiency: f64,
    ) -> BrainResult<Self> {
        let result = Self {
            minimum_reports,
            maximum_reports,
            maximum_step_distance: FiniteF64::new(maximum_step_distance)?,
            maximum_tortuosity: FiniteF64::new(maximum_tortuosity)?,
            maximum_waste: FiniteF64::new(maximum_waste)?,
            minimum_path_efficiency: FiniteF64::new(minimum_path_efficiency)?,
        };
        if result.minimum_reports < 2
            || result.maximum_reports < result.minimum_reports
            || result.maximum_reports >= MAX_TRAJECTORY_POINTS
            || result.maximum_step_distance.get() <= 0.0
            || result.maximum_tortuosity.get() < 1.0
            || !(0.0..=1.0).contains(&result.maximum_waste.get())
            || result.minimum_path_efficiency.get() < 0.0
        {
            return Err(invalid("petfc_path_limits_invalid"));
        }
        Ok(result)
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PetfcConservationLimits {
    maximum_soft_degradation_sum: FiniteF64,
    maximum_degraded_metric_count: usize,
    maximum_distributed_degradation: FiniteF64,
}

impl PetfcConservationLimits {
    pub fn new(
        maximum_soft_degradation_sum: f64,
        maximum_degraded_metric_count: usize,
        maximum_distributed_degradation: f64,
    ) -> BrainResult<Self> {
        let result = Self {
            maximum_soft_degradation_sum: FiniteF64::new(maximum_soft_degradation_sum)?,
            maximum_degraded_metric_count,
            maximum_distributed_degradation: FiniteF64::new(maximum_distributed_degradation)?,
        };
        if result.maximum_soft_degradation_sum.get() < 0.0
            || result.maximum_degraded_metric_count > MAX_METRICS
            || result.maximum_distributed_degradation.get() < 1.0
        {
            return Err(invalid("petfc_conservation_limits_invalid"));
        }
        Ok(result)
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PetfcUtilityPolicy {
    path_penalty: FiniteF64,
    conservation_penalty: FiniteF64,
    tortuosity_penalty: FiniteF64,
    minimum_utility: FiniteF64,
}

impl PetfcUtilityPolicy {
    pub fn new(
        path_penalty: f64,
        conservation_penalty: f64,
        tortuosity_penalty: f64,
        minimum_utility: f64,
    ) -> BrainResult<Self> {
        let result = Self {
            path_penalty: FiniteF64::new(path_penalty)?,
            conservation_penalty: FiniteF64::new(conservation_penalty)?,
            tortuosity_penalty: FiniteF64::new(tortuosity_penalty)?,
            minimum_utility: FiniteF64::new(minimum_utility)?,
        };
        if result.path_penalty.get() < 0.0
            || result.conservation_penalty.get() < 0.0
            || result.tortuosity_penalty.get() < 0.0
        {
            return Err(invalid("petfc_utility_policy_invalid"));
        }
        Ok(result)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PetfcPolicy {
    metric_catalog_digest: MetricCatalogDigest,
    quality_metric_id: MetricId,
    metrics: BTreeMap<MetricId, PetfcMetricPolicy>,
    path: PetfcPathLimits,
    conservation: PetfcConservationLimits,
    utility: PetfcUtilityPolicy,
    policy_digest: PetfcPolicyDigest,
}

#[derive(Serialize)]
struct PetfcPolicyProjection<'a> {
    metric_catalog_digest: &'a MetricCatalogDigest,
    quality_metric_id: &'a MetricId,
    metrics: &'a BTreeMap<MetricId, PetfcMetricPolicy>,
    path: &'a PetfcPathLimits,
    conservation: &'a PetfcConservationLimits,
    utility: &'a PetfcUtilityPolicy,
}

impl PetfcPolicy {
    pub fn new(
        specs: &[MetricSpec],
        quality_metric_id: MetricId,
        metric_policies: Vec<PetfcMetricPolicy>,
        path: PetfcPathLimits,
        conservation: PetfcConservationLimits,
        utility: PetfcUtilityPolicy,
    ) -> BrainResult<Self> {
        let catalog = canonical_specs(specs)?;
        if !catalog.contains_key(&quality_metric_id)
            || metric_policies.len() != catalog.len()
            || metric_policies.len() > MAX_METRICS
        {
            return Err(invalid("petfc_policy_metric_binding_invalid"));
        }
        let mut metrics = BTreeMap::new();
        for metric in metric_policies {
            if !catalog.contains_key(&metric.metric_id)
                || metrics.insert(metric.metric_id.clone(), metric).is_some()
            {
                return Err(invalid("petfc_policy_metric_binding_invalid"));
            }
        }
        let metric_catalog_digest = metric_catalog_digest(&catalog)?;
        let projection = PetfcPolicyProjection {
            metric_catalog_digest: &metric_catalog_digest,
            quality_metric_id: &quality_metric_id,
            metrics: &metrics,
            path: &path,
            conservation: &conservation,
            utility: &utility,
        };
        let policy_digest = PetfcPolicyDigest::from_projection(
            PETFC_POLICY_DOMAIN,
            &serde_json::to_vec(&projection)?,
        );
        Ok(Self {
            metric_catalog_digest,
            quality_metric_id,
            metrics,
            path,
            conservation,
            utility,
            policy_digest,
        })
    }

    pub fn digest(&self) -> &PetfcPolicyDigest {
        &self.policy_digest
    }

    pub fn quality_metric_id(&self) -> &MetricId {
        &self.quality_metric_id
    }

    pub fn quality_normalization_scale(&self) -> BrainResult<f64> {
        self.metrics
            .get(&self.quality_metric_id)
            .map(|metric| metric.normalization_scale.get())
            .ok_or_else(|| integrity("petfc_quality_metric_policy_missing"))
    }

    pub fn maximum_conservation_sum(&self) -> f64 {
        self.conservation.maximum_soft_degradation_sum.get()
    }

    pub fn maximum_distributed_degradation(&self) -> f64 {
        self.conservation.maximum_distributed_degradation.get()
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct FunctionalCoordinate {
    center: FiniteF64,
    observed_radius: Option<FiniteF64>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct FunctionalPoint {
    checkpoint_id: VariantId,
    source_report_digest: PairedEvaluationDigest,
    observation_window: ObservationWindow,
    source_evidence_sufficient: bool,
    /// Geometry is expressed as paired effects relative to the immutable
    /// incumbent. It is never reconstructed by subtracting absolute centers
    /// measured in different reports.
    coordinates: BTreeMap<MetricId, FunctionalCoordinate>,
    /// Absolute candidate observations are a separate channel used only for
    /// absolute contracts such as hard invariants.
    absolute_coordinates: BTreeMap<MetricId, FunctionalCoordinate>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PetfcTrajectory {
    baseline_id: VariantId,
    current_candidate_id: VariantId,
    metric_catalog_digest: MetricCatalogDigest,
    evaluation_policy_digest: RobustEvaluationPolicyDigest,
    independence_design_digest: IndependenceDesignDigest,
    petfc_policy_digest: PetfcPolicyDigest,
    points: Vec<FunctionalPoint>,
    report_digests: BTreeSet<PairedEvaluationDigest>,
    evidence_ids: BTreeSet<EvidenceId>,
    trajectory_digest: PetfcTrajectoryDigest,
}

#[derive(Serialize)]
struct PetfcTrajectoryProjection<'a> {
    baseline_id: &'a VariantId,
    current_candidate_id: &'a VariantId,
    metric_catalog_digest: &'a MetricCatalogDigest,
    evaluation_policy_digest: &'a RobustEvaluationPolicyDigest,
    independence_design_digest: &'a IndependenceDesignDigest,
    petfc_policy_digest: &'a PetfcPolicyDigest,
    points: &'a [FunctionalPoint],
    report_digests: &'a BTreeSet<PairedEvaluationDigest>,
    evidence_ids: &'a BTreeSet<EvidenceId>,
}

struct PetfcTrajectoryDraft {
    baseline_id: VariantId,
    current_candidate_id: VariantId,
    metric_catalog_digest: MetricCatalogDigest,
    evaluation_policy_digest: RobustEvaluationPolicyDigest,
    independence_design_digest: IndependenceDesignDigest,
    petfc_policy_digest: PetfcPolicyDigest,
    points: Vec<FunctionalPoint>,
    report_digests: BTreeSet<PairedEvaluationDigest>,
    evidence_ids: BTreeSet<EvidenceId>,
}

impl PetfcTrajectory {
    pub fn start(report: &PairedEvaluationReport, policy: &PetfcPolicy) -> BrainResult<Self> {
        report.authenticate()?;
        if report.metric_catalog_digest != policy.metric_catalog_digest {
            return Err(integrity("petfc_trajectory_policy_binding_mismatch"));
        }
        let baseline_absolute_coordinates = report
            .estimates
            .iter()
            .map(|(metric_id, estimate)| {
                (
                    metric_id.clone(),
                    FunctionalCoordinate {
                        center: estimate.baseline_center,
                        observed_radius: estimate.baseline_observed_radius,
                    },
                )
            })
            .collect();
        let zero = FiniteF64::new(0.0)?;
        let baseline_effect_coordinates = report
            .estimates
            .keys()
            .map(|metric_id| {
                (
                    metric_id.clone(),
                    FunctionalCoordinate {
                        center: zero,
                        observed_radius: Some(zero),
                    },
                )
            })
            .collect();
        let candidate_effect_coordinates = report
            .estimates
            .iter()
            .map(|(metric_id, estimate)| {
                (
                    metric_id.clone(),
                    FunctionalCoordinate {
                        center: estimate.paired_effect,
                        observed_radius: estimate.paired_effect_observed_radius,
                    },
                )
            })
            .collect();
        let candidate_absolute_coordinates = report
            .estimates
            .iter()
            .map(|(metric_id, estimate)| {
                (
                    metric_id.clone(),
                    FunctionalCoordinate {
                        center: estimate.candidate_center,
                        observed_radius: estimate.candidate_observed_radius,
                    },
                )
            })
            .collect();
        let points = vec![
            FunctionalPoint {
                checkpoint_id: report.baseline_id.clone(),
                source_report_digest: report.report_digest.clone(),
                observation_window: report.observation_window.clone(),
                source_evidence_sufficient: report.evidence_sufficient,
                coordinates: baseline_effect_coordinates,
                absolute_coordinates: baseline_absolute_coordinates,
            },
            FunctionalPoint {
                checkpoint_id: report.candidate_id.clone(),
                source_report_digest: report.report_digest.clone(),
                observation_window: report.observation_window.clone(),
                source_evidence_sufficient: report.evidence_sufficient,
                coordinates: candidate_effect_coordinates,
                absolute_coordinates: candidate_absolute_coordinates,
            },
        ];
        Self::seal(PetfcTrajectoryDraft {
            baseline_id: report.baseline_id.clone(),
            current_candidate_id: report.candidate_id.clone(),
            metric_catalog_digest: report.metric_catalog_digest.clone(),
            evaluation_policy_digest: report.evaluation_policy_digest.clone(),
            independence_design_digest: report.independence_design_digest.clone(),
            petfc_policy_digest: policy.policy_digest.clone(),
            points,
            report_digests: BTreeSet::from([report.report_digest.clone()]),
            evidence_ids: report.evidence_ids.clone(),
        })
    }

    pub fn append_report(mut self, report: &PairedEvaluationReport) -> BrainResult<Self> {
        self.authenticate()?;
        report.authenticate()?;
        if self.points.len() >= MAX_TRAJECTORY_POINTS
            || report.baseline_id != self.baseline_id
            || report.metric_catalog_digest != self.metric_catalog_digest
            || report.evaluation_policy_digest != self.evaluation_policy_digest
            || report.independence_design_digest != self.independence_design_digest
            || self.report_digests.contains(&report.report_digest)
            || !self.evidence_ids.is_disjoint(&report.evidence_ids)
            || self
                .points
                .iter()
                .any(|point| point.checkpoint_id == report.candidate_id)
        {
            return Err(integrity("petfc_trajectory_report_binding_invalid"));
        }
        let previous_end = self
            .points
            .last()
            .ok_or_else(|| integrity("petfc_trajectory_empty"))?
            .observation_window
            .end_tick;
        if report.observation_window.start_tick <= previous_end {
            return Err(integrity("petfc_trajectory_time_not_monotonic"));
        }
        let coordinates = report
            .estimates
            .iter()
            .map(|(metric_id, estimate)| {
                (
                    metric_id.clone(),
                    FunctionalCoordinate {
                        center: estimate.paired_effect,
                        observed_radius: estimate.paired_effect_observed_radius,
                    },
                )
            })
            .collect();
        let absolute_coordinates = report
            .estimates
            .iter()
            .map(|(metric_id, estimate)| {
                (
                    metric_id.clone(),
                    FunctionalCoordinate {
                        center: estimate.candidate_center,
                        observed_radius: estimate.candidate_observed_radius,
                    },
                )
            })
            .collect();
        self.points.push(FunctionalPoint {
            checkpoint_id: report.candidate_id.clone(),
            source_report_digest: report.report_digest.clone(),
            observation_window: report.observation_window.clone(),
            source_evidence_sufficient: report.evidence_sufficient,
            coordinates,
            absolute_coordinates,
        });
        self.current_candidate_id = report.candidate_id.clone();
        self.report_digests.insert(report.report_digest.clone());
        self.evidence_ids
            .extend(report.evidence_ids.iter().cloned());
        Self::seal(PetfcTrajectoryDraft {
            baseline_id: self.baseline_id,
            current_candidate_id: self.current_candidate_id,
            metric_catalog_digest: self.metric_catalog_digest,
            evaluation_policy_digest: self.evaluation_policy_digest,
            independence_design_digest: self.independence_design_digest,
            petfc_policy_digest: self.petfc_policy_digest,
            points: self.points,
            report_digests: self.report_digests,
            evidence_ids: self.evidence_ids,
        })
    }

    pub fn current_candidate_id(&self) -> &VariantId {
        &self.current_candidate_id
    }

    pub fn digest(&self) -> &PetfcTrajectoryDigest {
        &self.trajectory_digest
    }

    fn seal(draft: PetfcTrajectoryDraft) -> BrainResult<Self> {
        let PetfcTrajectoryDraft {
            baseline_id,
            current_candidate_id,
            metric_catalog_digest,
            evaluation_policy_digest,
            independence_design_digest,
            petfc_policy_digest,
            points,
            report_digests,
            evidence_ids,
        } = draft;
        if points.len() < 2
            || points.len() > MAX_TRAJECTORY_POINTS
            || report_digests.is_empty()
            || evidence_ids.is_empty()
            || points.first().map(|point| &point.checkpoint_id) != Some(&baseline_id)
            || points.last().map(|point| &point.checkpoint_id) != Some(&current_candidate_id)
            || points
                .iter()
                .any(|point| point.coordinates.len() != point.absolute_coordinates.len())
        {
            return Err(invalid("petfc_trajectory_shape_invalid"));
        }
        let unique_checkpoints = points
            .iter()
            .map(|point| &point.checkpoint_id)
            .collect::<BTreeSet<_>>();
        if unique_checkpoints.len() != points.len() {
            return Err(invalid("petfc_trajectory_checkpoint_duplicate"));
        }
        let projection = PetfcTrajectoryProjection {
            baseline_id: &baseline_id,
            current_candidate_id: &current_candidate_id,
            metric_catalog_digest: &metric_catalog_digest,
            evaluation_policy_digest: &evaluation_policy_digest,
            independence_design_digest: &independence_design_digest,
            petfc_policy_digest: &petfc_policy_digest,
            points: &points,
            report_digests: &report_digests,
            evidence_ids: &evidence_ids,
        };
        let trajectory_digest = PetfcTrajectoryDigest::from_projection(
            PETFC_TRAJECTORY_DOMAIN,
            &serde_json::to_vec(&projection)?,
        );
        Ok(Self {
            baseline_id,
            current_candidate_id,
            metric_catalog_digest,
            evaluation_policy_digest,
            independence_design_digest,
            petfc_policy_digest,
            points,
            report_digests,
            evidence_ids,
            trajectory_digest,
        })
    }

    fn authenticate(&self) -> BrainResult<()> {
        let projection = PetfcTrajectoryProjection {
            baseline_id: &self.baseline_id,
            current_candidate_id: &self.current_candidate_id,
            metric_catalog_digest: &self.metric_catalog_digest,
            evaluation_policy_digest: &self.evaluation_policy_digest,
            independence_design_digest: &self.independence_design_digest,
            petfc_policy_digest: &self.petfc_policy_digest,
            points: &self.points,
            report_digests: &self.report_digests,
            evidence_ids: &self.evidence_ids,
        };
        let calculated = PetfcTrajectoryDigest::from_projection(
            PETFC_TRAJECTORY_DOMAIN,
            &serde_json::to_vec(&projection)?,
        );
        if calculated != self.trajectory_digest {
            return Err(integrity("petfc_trajectory_digest_mismatch"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PetfcDisposition {
    CompatibleForNextGate,
    Reject,
    RollbackRequired,
    BoundedUnknown,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "reason", rename_all = "snake_case", deny_unknown_fields)]
pub enum PetfcReason {
    ProvenHardViolation {
        metric_id: MetricId,
        checkpoint_id: VariantId,
    },
    HardInvariantNotEstablished {
        metric_id: MetricId,
        checkpoint_id: VariantId,
    },
    InsufficientReports,
    InsufficientIndependentGroups,
    MissingObservedRadius {
        metric_id: MetricId,
    },
    PathNotIdentifiable,
    ClosedReturn,
    StepDistanceExceeded,
    TortuosityExceeded,
    WasteExceeded,
    EfficiencyBelowThreshold,
    ConservationBudgetExceeded,
    MetricDegradationExceeded {
        metric_id: MetricId,
    },
    DistributedDegradationExceeded,
    UtilityBelowThreshold,
    QualityNotImproved,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PetfcAssessment {
    baseline_id: VariantId,
    candidate_id: VariantId,
    metric_catalog_digest: MetricCatalogDigest,
    evaluation_policy_digest: RobustEvaluationPolicyDigest,
    independence_design_digest: IndependenceDesignDigest,
    terminal_report_digest: PairedEvaluationDigest,
    trajectory_evidence_ids: BTreeSet<EvidenceId>,
    trajectory_observation_window: ObservationWindow,
    trajectory_digest: PetfcTrajectoryDigest,
    policy_digest: PetfcPolicyDigest,
    disposition: PetfcDisposition,
    reasons: BTreeSet<PetfcReason>,
    path_length_lower: Option<FiniteF64>,
    path_length_upper: Option<FiniteF64>,
    endpoint_distance_lower: Option<FiniteF64>,
    endpoint_distance_upper: Option<FiniteF64>,
    maximum_step_upper: Option<FiniteF64>,
    tortuosity_upper: Option<FiniteF64>,
    waste_upper: Option<FiniteF64>,
    quality_gain_lower: Option<FiniteF64>,
    path_efficiency_lower: Option<FiniteF64>,
    conservation_sum_upper: Option<FiniteF64>,
    conservation_max_upper: Option<FiniteF64>,
    degraded_metric_count: usize,
    distributed_degradation_upper: Option<FiniteF64>,
    utility_lower: Option<FiniteF64>,
    assessment_digest: PetfcAssessmentDigest,
}

#[derive(Serialize)]
struct PetfcAssessmentProjection<'a> {
    baseline_id: &'a VariantId,
    candidate_id: &'a VariantId,
    metric_catalog_digest: &'a MetricCatalogDigest,
    evaluation_policy_digest: &'a RobustEvaluationPolicyDigest,
    independence_design_digest: &'a IndependenceDesignDigest,
    terminal_report_digest: &'a PairedEvaluationDigest,
    trajectory_evidence_ids: &'a BTreeSet<EvidenceId>,
    trajectory_observation_window: &'a ObservationWindow,
    trajectory_digest: &'a PetfcTrajectoryDigest,
    policy_digest: &'a PetfcPolicyDigest,
    disposition: PetfcDisposition,
    reasons: &'a BTreeSet<PetfcReason>,
    path_length_lower: Option<FiniteF64>,
    path_length_upper: Option<FiniteF64>,
    endpoint_distance_lower: Option<FiniteF64>,
    endpoint_distance_upper: Option<FiniteF64>,
    maximum_step_upper: Option<FiniteF64>,
    tortuosity_upper: Option<FiniteF64>,
    waste_upper: Option<FiniteF64>,
    quality_gain_lower: Option<FiniteF64>,
    path_efficiency_lower: Option<FiniteF64>,
    conservation_sum_upper: Option<FiniteF64>,
    conservation_max_upper: Option<FiniteF64>,
    degraded_metric_count: usize,
    distributed_degradation_upper: Option<FiniteF64>,
    utility_lower: Option<FiniteF64>,
}

impl PetfcAssessment {
    pub fn disposition(&self) -> PetfcDisposition {
        self.disposition
    }

    pub fn reasons(&self) -> &BTreeSet<PetfcReason> {
        &self.reasons
    }

    pub fn path_length_upper(&self) -> Option<f64> {
        self.path_length_upper.map(FiniteF64::get)
    }

    pub fn endpoint_distance_lower(&self) -> Option<f64> {
        self.endpoint_distance_lower.map(FiniteF64::get)
    }

    pub fn tortuosity_upper(&self) -> Option<f64> {
        self.tortuosity_upper.map(FiniteF64::get)
    }

    pub fn waste_upper(&self) -> Option<f64> {
        self.waste_upper.map(FiniteF64::get)
    }

    pub fn quality_gain_lower(&self) -> Option<f64> {
        self.quality_gain_lower.map(FiniteF64::get)
    }

    pub fn path_efficiency_lower(&self) -> Option<f64> {
        self.path_efficiency_lower.map(FiniteF64::get)
    }

    pub fn conservation_sum_upper(&self) -> Option<f64> {
        self.conservation_sum_upper.map(FiniteF64::get)
    }

    pub fn distributed_degradation_upper(&self) -> Option<f64> {
        self.distributed_degradation_upper.map(FiniteF64::get)
    }

    pub fn digest(&self) -> &PetfcAssessmentDigest {
        &self.assessment_digest
    }

    pub const fn authorizes_promotion(&self) -> bool {
        false
    }

    fn authenticate(&self) -> BrainResult<()> {
        let projection = PetfcAssessmentProjection {
            baseline_id: &self.baseline_id,
            candidate_id: &self.candidate_id,
            metric_catalog_digest: &self.metric_catalog_digest,
            evaluation_policy_digest: &self.evaluation_policy_digest,
            independence_design_digest: &self.independence_design_digest,
            terminal_report_digest: &self.terminal_report_digest,
            trajectory_evidence_ids: &self.trajectory_evidence_ids,
            trajectory_observation_window: &self.trajectory_observation_window,
            trajectory_digest: &self.trajectory_digest,
            policy_digest: &self.policy_digest,
            disposition: self.disposition,
            reasons: &self.reasons,
            path_length_lower: self.path_length_lower,
            path_length_upper: self.path_length_upper,
            endpoint_distance_lower: self.endpoint_distance_lower,
            endpoint_distance_upper: self.endpoint_distance_upper,
            maximum_step_upper: self.maximum_step_upper,
            tortuosity_upper: self.tortuosity_upper,
            waste_upper: self.waste_upper,
            quality_gain_lower: self.quality_gain_lower,
            path_efficiency_lower: self.path_efficiency_lower,
            conservation_sum_upper: self.conservation_sum_upper,
            conservation_max_upper: self.conservation_max_upper,
            degraded_metric_count: self.degraded_metric_count,
            distributed_degradation_upper: self.distributed_degradation_upper,
            utility_lower: self.utility_lower,
        };
        let calculated = PetfcAssessmentDigest::from_projection(
            PETFC_ASSESSMENT_DOMAIN,
            &serde_json::to_vec(&projection)?,
        );
        if calculated != self.assessment_digest {
            return Err(integrity("petfc_assessment_digest_mismatch"));
        }
        Ok(())
    }
}

fn stable_nonnegative_sum(values: impl IntoIterator<Item = f64>) -> BrainResult<f64> {
    let mut sum = 0.0_f64;
    let mut correction = 0.0_f64;
    for value in values {
        if !value.is_finite() || value < 0.0 {
            return Err(BrainError::Numerical("petfc_sum_input_invalid".into()));
        }
        let updated = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - updated) + value
        } else {
            (value - updated) + sum
        };
        sum = updated;
    }
    let result = sum + correction;
    if !result.is_finite() {
        return Err(BrainError::Numerical("petfc_sum_nonfinite".into()));
    }
    Ok(result)
}

fn interval_distance_bounds(
    left: &FunctionalPoint,
    right: &FunctionalPoint,
    metric_policy: &BTreeMap<MetricId, PetfcMetricPolicy>,
) -> BrainResult<Option<(f64, f64)>> {
    if left.coordinates.len() != metric_policy.len()
        || right.coordinates.len() != metric_policy.len()
    {
        return Err(integrity("petfc_coordinate_schema_mismatch"));
    }
    let mut lower_components = Vec::with_capacity(metric_policy.len());
    let mut upper_components = Vec::with_capacity(metric_policy.len());
    for (metric_id, policy) in metric_policy {
        let left = left
            .coordinates
            .get(metric_id)
            .ok_or_else(|| integrity("petfc_coordinate_missing"))?;
        let right = right
            .coordinates
            .get(metric_id)
            .ok_or_else(|| integrity("petfc_coordinate_missing"))?;
        let (Some(left_radius), Some(right_radius)) = (left.observed_radius, right.observed_radius)
        else {
            return Ok(None);
        };
        let difference = left.center.get() - right.center.get();
        let radius = left_radius.get() + right_radius.get();
        if !difference.is_finite() || !radius.is_finite() {
            return Err(BrainError::Numerical(
                "petfc_coordinate_interval_nonfinite".into(),
            ));
        }
        let scale = policy.normalization_scale.get();
        lower_components.push((difference.abs() - radius).max(0.0) / scale);
        upper_components.push((difference.abs() + radius) / scale);
    }
    Ok(Some((
        stable_rms(lower_components)?,
        stable_rms(upper_components)?,
    )))
}

fn finite_option(value: Option<f64>) -> BrainResult<Option<FiniteF64>> {
    value.map(FiniteF64::new).transpose()
}

pub fn evaluate_petfc(
    specs: &[MetricSpec],
    trajectory: &PetfcTrajectory,
    policy: &PetfcPolicy,
) -> BrainResult<PetfcAssessment> {
    trajectory.authenticate()?;
    let catalog = canonical_specs(specs)?;
    let catalog_digest = metric_catalog_digest(&catalog)?;
    if trajectory.metric_catalog_digest != catalog_digest
        || policy.metric_catalog_digest != catalog_digest
        || trajectory.petfc_policy_digest != policy.policy_digest
        || policy.metrics.len() != catalog.len()
        || policy
            .metrics
            .keys()
            .any(|metric| !catalog.contains_key(metric))
    {
        return Err(integrity("petfc_metric_catalog_binding_mismatch"));
    }

    let report_count = trajectory.report_digests.len();
    if report_count > policy.path.maximum_reports {
        return Err(invalid("petfc_report_limit_exceeded"));
    }
    let mut hard_reasons = BTreeSet::new();
    let mut unknown_reasons = BTreeSet::new();
    let mut soft_reasons = BTreeSet::new();
    if report_count < policy.path.minimum_reports {
        unknown_reasons.insert(PetfcReason::InsufficientReports);
    }
    if trajectory
        .points
        .iter()
        .any(|point| !point.source_evidence_sufficient)
    {
        unknown_reasons.insert(PetfcReason::InsufficientIndependentGroups);
    }

    for point in &trajectory.points {
        for (metric_id, spec) in &catalog {
            let coordinate = point
                .absolute_coordinates
                .get(metric_id)
                .ok_or_else(|| integrity("petfc_absolute_coordinate_missing"))?;
            let Some(radius) = coordinate.observed_radius else {
                unknown_reasons.insert(PetfcReason::MissingObservedRadius {
                    metric_id: metric_id.clone(),
                });
                continue;
            };
            let lower = coordinate.center.get() - radius.get();
            let upper = coordinate.center.get() + radius.get();
            if !lower.is_finite() || !upper.is_finite() {
                return Err(BrainError::Numerical(
                    "petfc_absolute_interval_nonfinite".into(),
                ));
            }
            if let Some(invariant) = spec.hard_invariant.as_ref() {
                match invariant.classify_interval(lower, upper) {
                    InvariantIntervalStatus::ProvenViolation => {
                        hard_reasons.insert(PetfcReason::ProvenHardViolation {
                            metric_id: metric_id.clone(),
                            checkpoint_id: point.checkpoint_id.clone(),
                        });
                    }
                    InvariantIntervalStatus::NotEstablished => {
                        unknown_reasons.insert(PetfcReason::HardInvariantNotEstablished {
                            metric_id: metric_id.clone(),
                            checkpoint_id: point.checkpoint_id.clone(),
                        });
                    }
                    InvariantIntervalStatus::Established => {}
                }
            }
        }
    }

    let mut step_bounds = Vec::with_capacity(trajectory.points.len().saturating_sub(1));
    for pair in trajectory.points.windows(2) {
        match interval_distance_bounds(&pair[0], &pair[1], &policy.metrics)? {
            Some(bounds) => step_bounds.push(bounds),
            None => {
                unknown_reasons.insert(PetfcReason::PathNotIdentifiable);
            }
        }
    }
    let endpoint_bounds = interval_distance_bounds(
        trajectory
            .points
            .first()
            .ok_or_else(|| integrity("petfc_trajectory_empty"))?,
        trajectory
            .points
            .last()
            .ok_or_else(|| integrity("petfc_trajectory_empty"))?,
        &policy.metrics,
    )?;

    let (path_lower, path_upper, maximum_step_upper) = if step_bounds.len()
        == trajectory.points.len().saturating_sub(1)
        && !step_bounds.is_empty()
    {
        let lower = stable_nonnegative_sum(step_bounds.iter().map(|(value, _)| *value))?;
        let upper = stable_nonnegative_sum(step_bounds.iter().map(|(_, value)| *value))?;
        let maximum = step_bounds
            .iter()
            .map(|(_, value)| *value)
            .max_by(f64::total_cmp)
            .ok_or_else(|| integrity("petfc_step_bounds_empty"))?;
        (Some(lower), Some(upper), Some(maximum))
    } else {
        (None, None, None)
    };
    let (endpoint_lower, endpoint_upper) = endpoint_bounds.unzip();

    if maximum_step_upper.is_some_and(|value| value > policy.path.maximum_step_distance.get()) {
        soft_reasons.insert(PetfcReason::StepDistanceExceeded);
    }

    let tortuosity_upper = match (path_upper, endpoint_lower) {
        (Some(path), Some(_endpoint)) if path <= NUMERICAL_EPSILON => Some(1.0),
        (Some(path), Some(endpoint)) if endpoint > NUMERICAL_EPSILON => Some(path / endpoint),
        (Some(path), Some(_)) if path > NUMERICAL_EPSILON => {
            if endpoint_upper.is_some_and(|upper| upper <= NUMERICAL_EPSILON) {
                soft_reasons.insert(PetfcReason::ClosedReturn);
            } else {
                unknown_reasons.insert(PetfcReason::PathNotIdentifiable);
            }
            None
        }
        _ => None,
    };
    if tortuosity_upper.is_some_and(|value| value > policy.path.maximum_tortuosity.get()) {
        soft_reasons.insert(PetfcReason::TortuosityExceeded);
    }
    let waste_upper = match (path_upper, endpoint_lower) {
        (Some(path), Some(_)) if path <= NUMERICAL_EPSILON => Some(0.0),
        (Some(path), Some(endpoint)) => Some((1.0 - endpoint / path).clamp(0.0, 1.0)),
        _ => None,
    };
    if waste_upper.is_some_and(|value| value > policy.path.maximum_waste.get()) {
        soft_reasons.insert(PetfcReason::WasteExceeded);
    }

    let first = trajectory
        .points
        .first()
        .ok_or_else(|| integrity("petfc_trajectory_empty"))?;
    let last = trajectory
        .points
        .last()
        .ok_or_else(|| integrity("petfc_trajectory_empty"))?;
    let quality_spec = catalog
        .get(&policy.quality_metric_id)
        .ok_or_else(|| integrity("petfc_quality_metric_missing"))?;
    let quality_policy = policy
        .metrics
        .get(&policy.quality_metric_id)
        .ok_or_else(|| integrity("petfc_quality_policy_missing"))?;
    let quality_start = first
        .coordinates
        .get(&policy.quality_metric_id)
        .ok_or_else(|| integrity("petfc_quality_coordinate_missing"))?;
    let quality_end = last
        .coordinates
        .get(&policy.quality_metric_id)
        .ok_or_else(|| integrity("petfc_quality_coordinate_missing"))?;
    let quality_gain_lower = match (quality_start.observed_radius, quality_end.observed_radius) {
        (Some(start_radius), Some(end_radius)) => {
            let oriented_gain = quality_spec.direction.orient(quality_end.center.get())
                - quality_spec.direction.orient(quality_start.center.get());
            let uncertainty = start_radius.get() + end_radius.get();
            let value = (oriented_gain - uncertainty) / quality_policy.normalization_scale.get();
            if !value.is_finite() {
                return Err(BrainError::Numerical("petfc_quality_gain_nonfinite".into()));
            }
            Some(value)
        }
        _ => {
            unknown_reasons.insert(PetfcReason::MissingObservedRadius {
                metric_id: policy.quality_metric_id.clone(),
            });
            None
        }
    };
    if quality_gain_lower.is_some_and(|value| value <= 0.0) {
        soft_reasons.insert(PetfcReason::QualityNotImproved);
    }
    let path_efficiency_lower = match (quality_gain_lower, path_upper) {
        (Some(_), Some(path)) if path <= NUMERICAL_EPSILON => None,
        (Some(gain), Some(path)) => Some(gain / path),
        _ => None,
    };
    if path_efficiency_lower.is_none() {
        unknown_reasons.insert(PetfcReason::PathNotIdentifiable);
    } else if path_efficiency_lower
        .is_some_and(|value| value < policy.path.minimum_path_efficiency.get())
    {
        soft_reasons.insert(PetfcReason::EfficiencyBelowThreshold);
    }

    let mut degradations = BTreeMap::new();
    for (metric_id, metric_policy) in &policy.metrics {
        let spec = catalog
            .get(metric_id)
            .ok_or_else(|| integrity("petfc_metric_spec_missing"))?;
        let start = first
            .coordinates
            .get(metric_id)
            .ok_or_else(|| integrity("petfc_coordinate_missing"))?;
        let end = last
            .coordinates
            .get(metric_id)
            .ok_or_else(|| integrity("petfc_coordinate_missing"))?;
        let (Some(start_radius), Some(end_radius)) = (start.observed_radius, end.observed_radius)
        else {
            unknown_reasons.insert(PetfcReason::MissingObservedRadius {
                metric_id: metric_id.clone(),
            });
            continue;
        };
        let oriented_start = spec.direction.orient(start.center.get());
        let oriented_end = spec.direction.orient(end.center.get());
        let worst_degradation =
            (oriented_start - oriented_end + start_radius.get() + end_radius.get()).max(0.0)
                / metric_policy.normalization_scale.get();
        if !worst_degradation.is_finite() {
            return Err(BrainError::Numerical("petfc_conservation_nonfinite".into()));
        }
        if worst_degradation > metric_policy.maximum_normalized_endpoint_degradation.get() {
            soft_reasons.insert(PetfcReason::MetricDegradationExceeded {
                metric_id: metric_id.clone(),
            });
        }
        degradations.insert(metric_id.clone(), worst_degradation);
    }
    let conservation_sum = if degradations.len() == policy.metrics.len() {
        Some(stable_nonnegative_sum(degradations.values().copied())?)
    } else {
        None
    };
    let conservation_max = if degradations.len() == policy.metrics.len() {
        degradations.values().copied().max_by(f64::total_cmp)
    } else {
        None
    };
    let degraded_metric_count = degradations
        .iter()
        .filter(|(metric_id, value)| {
            policy.metrics.get(*metric_id).is_some_and(|metric| {
                **value > metric.maximum_normalized_endpoint_degradation.get()
            })
        })
        .count();
    let distributed = match (conservation_sum, conservation_max) {
        (Some(_), Some(maximum)) if maximum <= NUMERICAL_EPSILON => Some(0.0),
        (Some(sum), Some(maximum)) => Some(sum / maximum),
        _ => None,
    };
    if conservation_sum
        .is_some_and(|value| value > policy.conservation.maximum_soft_degradation_sum.get())
    {
        soft_reasons.insert(PetfcReason::ConservationBudgetExceeded);
    }
    if degraded_metric_count > policy.conservation.maximum_degraded_metric_count {
        soft_reasons.insert(PetfcReason::ConservationBudgetExceeded);
    }
    if distributed
        .is_some_and(|value| value > policy.conservation.maximum_distributed_degradation.get())
    {
        soft_reasons.insert(PetfcReason::DistributedDegradationExceeded);
    }

    let utility_lower = match (
        quality_gain_lower,
        path_upper,
        conservation_sum,
        tortuosity_upper,
    ) {
        (Some(gain), Some(path), Some(conservation), Some(tortuosity)) => {
            let excess = (tortuosity - policy.path.maximum_tortuosity.get()).max(0.0);
            let value = gain
                - policy.utility.path_penalty.get() * path
                - policy.utility.conservation_penalty.get() * conservation
                - policy.utility.tortuosity_penalty.get() * excess * excess;
            if !value.is_finite() {
                return Err(BrainError::Numerical("petfc_utility_nonfinite".into()));
            }
            Some(value)
        }
        _ => None,
    };
    if utility_lower.is_some_and(|value| value < policy.utility.minimum_utility.get()) {
        soft_reasons.insert(PetfcReason::UtilityBelowThreshold);
    }

    let baseline_has_hard_violation = hard_reasons.iter().any(|reason| {
        matches!(
            reason,
            PetfcReason::ProvenHardViolation { checkpoint_id, .. }
                if checkpoint_id == &trajectory.baseline_id
        )
    });
    let disposition = if baseline_has_hard_violation {
        // Rolling back to an already-invalid baseline is not a truthful
        // remedy. Reject the trajectory and leave remediation to a separate
        // safety authority.
        PetfcDisposition::Reject
    } else if !hard_reasons.is_empty() {
        PetfcDisposition::RollbackRequired
    } else if !soft_reasons.is_empty() {
        PetfcDisposition::Reject
    } else if !unknown_reasons.is_empty() {
        PetfcDisposition::BoundedUnknown
    } else {
        PetfcDisposition::CompatibleForNextGate
    };
    let reasons = hard_reasons
        .into_iter()
        .chain(soft_reasons)
        .chain(unknown_reasons)
        .collect();
    let path_length_lower = finite_option(path_lower)?;
    let path_length_upper = finite_option(path_upper)?;
    let endpoint_distance_lower = finite_option(endpoint_lower)?;
    let endpoint_distance_upper = finite_option(endpoint_upper)?;
    let maximum_step_upper = finite_option(maximum_step_upper)?;
    let tortuosity_upper = finite_option(tortuosity_upper)?;
    let waste_upper = finite_option(waste_upper)?;
    let quality_gain_lower = finite_option(quality_gain_lower)?;
    let path_efficiency_lower = finite_option(path_efficiency_lower)?;
    let conservation_sum_upper = finite_option(conservation_sum)?;
    let conservation_max_upper = finite_option(conservation_max)?;
    let distributed_degradation_upper = finite_option(distributed)?;
    let utility_lower = finite_option(utility_lower)?;
    let terminal_report_digest = trajectory
        .points
        .last()
        .ok_or_else(|| integrity("petfc_trajectory_empty"))?
        .source_report_digest
        .clone();
    let trajectory_observation_window = ObservationWindow::new(
        trajectory
            .points
            .first()
            .ok_or_else(|| integrity("petfc_trajectory_empty"))?
            .observation_window
            .start_tick,
        trajectory
            .points
            .last()
            .ok_or_else(|| integrity("petfc_trajectory_empty"))?
            .observation_window
            .end_tick,
    )?;
    let projection = PetfcAssessmentProjection {
        baseline_id: &trajectory.baseline_id,
        candidate_id: &trajectory.current_candidate_id,
        metric_catalog_digest: &trajectory.metric_catalog_digest,
        evaluation_policy_digest: &trajectory.evaluation_policy_digest,
        independence_design_digest: &trajectory.independence_design_digest,
        terminal_report_digest: &terminal_report_digest,
        trajectory_evidence_ids: &trajectory.evidence_ids,
        trajectory_observation_window: &trajectory_observation_window,
        trajectory_digest: &trajectory.trajectory_digest,
        policy_digest: &policy.policy_digest,
        disposition,
        reasons: &reasons,
        path_length_lower,
        path_length_upper,
        endpoint_distance_lower,
        endpoint_distance_upper,
        maximum_step_upper,
        tortuosity_upper,
        waste_upper,
        quality_gain_lower,
        path_efficiency_lower,
        conservation_sum_upper,
        conservation_max_upper,
        degraded_metric_count,
        distributed_degradation_upper,
        utility_lower,
    };
    let assessment_digest = PetfcAssessmentDigest::from_projection(
        PETFC_ASSESSMENT_DOMAIN,
        &serde_json::to_vec(&projection)?,
    );
    let assessment = PetfcAssessment {
        baseline_id: trajectory.baseline_id.clone(),
        candidate_id: trajectory.current_candidate_id.clone(),
        metric_catalog_digest: trajectory.metric_catalog_digest.clone(),
        evaluation_policy_digest: trajectory.evaluation_policy_digest.clone(),
        independence_design_digest: trajectory.independence_design_digest.clone(),
        terminal_report_digest,
        trajectory_evidence_ids: trajectory.evidence_ids.clone(),
        trajectory_observation_window,
        trajectory_digest: trajectory.trajectory_digest.clone(),
        policy_digest: policy.policy_digest.clone(),
        disposition,
        reasons,
        path_length_lower,
        path_length_upper,
        endpoint_distance_lower,
        endpoint_distance_upper,
        maximum_step_upper,
        tortuosity_upper,
        waste_upper,
        quality_gain_lower,
        path_efficiency_lower,
        conservation_sum_upper,
        conservation_max_upper,
        degraded_metric_count,
        distributed_degradation_upper,
        utility_lower,
        assessment_digest,
    };
    assessment.authenticate()?;
    Ok(assessment)
}

// ---------------------------------------------------------------------------
// Pareto selection over candidates that already passed both evidence gates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ParetoCandidate {
    baseline_id: VariantId,
    candidate_id: VariantId,
    metric_catalog_digest: MetricCatalogDigest,
    evaluation_policy_digest: RobustEvaluationPolicyDigest,
    independence_design_digest: IndependenceDesignDigest,
    gate_policy_digest: CandidateGatePolicyDigest,
    petfc_policy_digest: PetfcPolicyDigest,
    gate_digest: CandidateGateDigest,
    petfc_assessment_digest: PetfcAssessmentDigest,
    conservative_objectives: BTreeMap<MetricId, FiniteF64>,
}

impl ParetoCandidate {
    pub fn from_verified_gates(
        gate: &CandidateGateDecision,
        petfc: &PetfcAssessment,
    ) -> BrainResult<Self> {
        gate.authenticate()?;
        petfc.authenticate()?;
        if gate.disposition != CandidateGateDisposition::AdvanceCandidate
            || petfc.disposition != PetfcDisposition::CompatibleForNextGate
            || gate.candidate_id != petfc.candidate_id
            || gate.baseline_id != petfc.baseline_id
            || gate.metric_catalog_digest != petfc.metric_catalog_digest
            || gate.evaluation_policy_digest != petfc.evaluation_policy_digest
            || !gate
                .source_evidence_ids
                .is_disjoint(&petfc.trajectory_evidence_ids)
            || !(gate.source_observation_window.end_tick
                < petfc.trajectory_observation_window.start_tick
                || gate.source_observation_window.start_tick
                    > petfc.trajectory_observation_window.end_tick)
            || gate.conservative_improvements.is_empty()
        {
            return Err(integrity("pareto_candidate_not_eligible"));
        }
        Ok(Self {
            baseline_id: gate.baseline_id.clone(),
            candidate_id: gate.candidate_id.clone(),
            metric_catalog_digest: gate.metric_catalog_digest.clone(),
            evaluation_policy_digest: gate.evaluation_policy_digest.clone(),
            independence_design_digest: gate.independence_design_digest.clone(),
            gate_policy_digest: gate.policy_digest.clone(),
            petfc_policy_digest: petfc.policy_digest.clone(),
            gate_digest: gate.decision_digest.clone(),
            petfc_assessment_digest: petfc.assessment_digest.clone(),
            conservative_objectives: gate.conservative_improvements.clone(),
        })
    }

    pub fn candidate_id(&self) -> &VariantId {
        &self.candidate_id
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ParetoSelection {
    selected: Vec<VariantId>,
    witnesses: BTreeMap<VariantId, (CandidateGateDigest, PetfcAssessmentDigest)>,
    selection_digest: ParetoSelectionDigest,
}

#[derive(Serialize)]
struct ParetoSelectionProjection<'a> {
    selected: &'a [VariantId],
    witnesses: &'a BTreeMap<VariantId, (CandidateGateDigest, PetfcAssessmentDigest)>,
}

impl ParetoSelection {
    pub fn selected(&self) -> &[VariantId] {
        &self.selected
    }

    pub fn digest(&self) -> &ParetoSelectionDigest {
        &self.selection_digest
    }

    pub const fn authorizes_promotion(&self) -> bool {
        false
    }
}

fn dominates(left: &ParetoCandidate, right: &ParetoCandidate) -> BrainResult<bool> {
    if left.conservative_objectives.len() != right.conservative_objectives.len()
        || left
            .conservative_objectives
            .keys()
            .any(|metric| !right.conservative_objectives.contains_key(metric))
    {
        return Err(integrity("pareto_objective_schema_mismatch"));
    }
    let mut strictly_better = false;
    for (metric_id, left_value) in &left.conservative_objectives {
        let right_value = right
            .conservative_objectives
            .get(metric_id)
            .ok_or_else(|| integrity("pareto_objective_missing"))?;
        if left_value.get() < right_value.get() {
            return Ok(false);
        }
        strictly_better |= left_value.get() > right_value.get();
    }
    Ok(strictly_better)
}

pub fn deterministic_pareto_front(candidates: &[ParetoCandidate]) -> BrainResult<ParetoSelection> {
    if candidates.is_empty() || candidates.len() > MAX_CANDIDATES {
        return Err(invalid("pareto_candidate_count_invalid"));
    }
    let metric_count = candidates[0].conservative_objectives.len();
    let comparisons = candidates
        .len()
        .checked_mul(candidates.len().saturating_sub(1))
        .and_then(|pairs| pairs.checked_mul(metric_count))
        .ok_or_else(|| invalid("pareto_work_estimate_overflow"))?;
    if comparisons > MAX_PARETO_WORK_UNITS {
        return Err(invalid("pareto_work_limit_exceeded"));
    }
    let expected_metrics = candidates[0]
        .conservative_objectives
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    let expected_baseline = &candidates[0].baseline_id;
    let expected_catalog = &candidates[0].metric_catalog_digest;
    let expected_evaluation_policy = &candidates[0].evaluation_policy_digest;
    let expected_independence_design = &candidates[0].independence_design_digest;
    let expected_gate_policy = &candidates[0].gate_policy_digest;
    let expected_petfc_policy = &candidates[0].petfc_policy_digest;
    if expected_metrics.is_empty() {
        return Err(invalid("pareto_objectives_empty"));
    }
    let mut by_id = BTreeMap::new();
    for candidate in candidates {
        if candidate
            .conservative_objectives
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
            != expected_metrics
            || &candidate.baseline_id != expected_baseline
            || &candidate.metric_catalog_digest != expected_catalog
            || &candidate.evaluation_policy_digest != expected_evaluation_policy
            || &candidate.independence_design_digest != expected_independence_design
            || &candidate.gate_policy_digest != expected_gate_policy
            || &candidate.petfc_policy_digest != expected_petfc_policy
            || by_id
                .insert(candidate.candidate_id.clone(), candidate)
                .is_some()
        {
            return Err(integrity("pareto_candidate_catalog_invalid"));
        }
    }
    let mut selected = Vec::new();
    for candidate in by_id.values() {
        let mut dominated = false;
        for alternative in by_id.values() {
            if alternative.candidate_id != candidate.candidate_id
                && dominates(alternative, candidate)?
            {
                dominated = true;
                break;
            }
        }
        if !dominated {
            selected.push(candidate.candidate_id.clone());
        }
    }
    let witnesses = selected
        .iter()
        .map(|candidate_id| {
            let candidate = by_id
                .get(candidate_id)
                .ok_or_else(|| integrity("pareto_selected_candidate_missing"))?;
            Ok((
                candidate_id.clone(),
                (
                    candidate.gate_digest.clone(),
                    candidate.petfc_assessment_digest.clone(),
                ),
            ))
        })
        .collect::<BrainResult<BTreeMap<_, _>>>()?;
    let projection = ParetoSelectionProjection {
        selected: &selected,
        witnesses: &witnesses,
    };
    let selection_digest = ParetoSelectionDigest::from_projection(
        PARETO_SELECTION_DOMAIN,
        &serde_json::to_vec(&projection)?,
    );
    Ok(ParetoSelection {
        selected,
        witnesses,
        selection_digest,
    })
}

// ---------------------------------------------------------------------------
// Adaptive experimental budget. Utilities are derived from sealed gate
// decisions; callers never provide expected information gain.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(transparent)]
pub struct ComputeUnits(NonZeroU64);

impl ComputeUnits {
    pub fn new(value: u64) -> BrainResult<Self> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or_else(|| invalid("adaptive_budget_compute_units_zero"))
    }

    pub fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveTrialCandidate {
    candidate_id: VariantId,
    gate_history: Vec<CandidateGateDecision>,
    trial_cost: ComputeUnits,
    maximum_additional_trials: usize,
}

impl AdaptiveTrialCandidate {
    pub fn from_gate_history(
        candidate_id: VariantId,
        gate_history: Vec<CandidateGateDecision>,
        trial_cost: ComputeUnits,
        maximum_additional_trials: usize,
    ) -> BrainResult<Self> {
        if maximum_additional_trials == 0
            || maximum_additional_trials > MAX_OBSERVATIONS
            || gate_history.len() > MAX_GATE_HISTORY_PER_CANDIDATE
        {
            return Err(invalid("adaptive_budget_candidate_trial_limit_invalid"));
        }
        let mut digests = BTreeSet::new();
        for gate in &gate_history {
            gate.authenticate()?;
            if gate.candidate_id != candidate_id
                || gate.disposition != CandidateGateDisposition::AdvanceCandidate
                || !digests.insert(gate.decision_digest.clone())
            {
                return Err(integrity("adaptive_budget_gate_history_invalid"));
            }
        }
        Ok(Self {
            candidate_id,
            gate_history,
            trial_cost,
            maximum_additional_trials,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveBudgetPolicy {
    priority_metric_id: MetricId,
    total_budget: ComputeUnits,
    holdout_reserve: ComputeUnits,
    minimum_trials_per_candidate: usize,
    maximum_total_trials: usize,
    exploration_strength: FiniteF64,
    cost_penalty: FiniteF64,
    policy_digest: AdaptiveBudgetPolicyDigest,
}

#[derive(Serialize)]
struct AdaptiveBudgetPolicyProjection<'a> {
    priority_metric_id: &'a MetricId,
    total_budget: ComputeUnits,
    holdout_reserve: ComputeUnits,
    minimum_trials_per_candidate: usize,
    maximum_total_trials: usize,
    exploration_strength: FiniteF64,
    cost_penalty: FiniteF64,
}

#[derive(Debug, Clone)]
pub struct AdaptiveBudgetPolicyDraft {
    pub priority_metric_id: MetricId,
    pub total_budget: ComputeUnits,
    pub holdout_reserve: ComputeUnits,
    pub minimum_trials_per_candidate: usize,
    pub maximum_total_trials: usize,
    pub exploration_strength: f64,
    pub cost_penalty: f64,
}

impl AdaptiveBudgetPolicy {
    pub fn new(draft: AdaptiveBudgetPolicyDraft) -> BrainResult<Self> {
        let exploration_strength = FiniteF64::new(draft.exploration_strength)?;
        let cost_penalty = FiniteF64::new(draft.cost_penalty)?;
        if draft.holdout_reserve.get() >= draft.total_budget.get()
            || draft.minimum_trials_per_candidate == 0
            || draft.maximum_total_trials == 0
            || draft.maximum_total_trials > MAX_OBSERVATIONS
            || exploration_strength.get() <= 0.0
            || cost_penalty.get() < 0.0
        {
            return Err(invalid("adaptive_budget_policy_invalid"));
        }
        let projection = AdaptiveBudgetPolicyProjection {
            priority_metric_id: &draft.priority_metric_id,
            total_budget: draft.total_budget,
            holdout_reserve: draft.holdout_reserve,
            minimum_trials_per_candidate: draft.minimum_trials_per_candidate,
            maximum_total_trials: draft.maximum_total_trials,
            exploration_strength,
            cost_penalty,
        };
        let policy_digest = AdaptiveBudgetPolicyDigest::from_projection(
            BUDGET_POLICY_DOMAIN,
            &serde_json::to_vec(&projection)?,
        );
        Ok(Self {
            priority_metric_id: draft.priority_metric_id,
            total_budget: draft.total_budget,
            holdout_reserve: draft.holdout_reserve,
            minimum_trials_per_candidate: draft.minimum_trials_per_candidate,
            maximum_total_trials: draft.maximum_total_trials,
            exploration_strength,
            cost_penalty,
            policy_digest,
        })
    }

    pub fn digest(&self) -> &AdaptiveBudgetPolicyDigest {
        &self.policy_digest
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrialAllocationReason {
    MandatoryCoverage,
    DerivedUpperConfidenceBound,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TrialAllocation {
    candidate_id: VariantId,
    candidate_trial_ordinal: usize,
    cost: ComputeUnits,
    reason: TrialAllocationReason,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveBudgetDisposition {
    Planned,
    BoundedUnknownInsufficientMandatoryCoverage,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveBudgetPlan {
    policy_digest: AdaptiveBudgetPolicyDigest,
    candidate_set_digest: Sha256Digest,
    disposition: AdaptiveBudgetDisposition,
    allocations: Vec<TrialAllocation>,
    consumed_budget: u64,
    holdout_reserve: ComputeUnits,
    unspent_budget: u64,
    plan_digest: AdaptiveBudgetPlanDigest,
}

#[derive(Serialize)]
struct AdaptiveBudgetPlanProjection<'a> {
    policy_digest: &'a AdaptiveBudgetPolicyDigest,
    candidate_set_digest: &'a Sha256Digest,
    disposition: AdaptiveBudgetDisposition,
    allocations: &'a [TrialAllocation],
    consumed_budget: u64,
    holdout_reserve: ComputeUnits,
    unspent_budget: u64,
}

impl AdaptiveBudgetPlan {
    pub fn disposition(&self) -> AdaptiveBudgetDisposition {
        self.disposition
    }

    pub fn allocations(&self) -> &[TrialAllocation] {
        &self.allocations
    }

    pub fn digest(&self) -> &AdaptiveBudgetPlanDigest {
        &self.plan_digest
    }

    pub const fn authorizes_promotion(&self) -> bool {
        false
    }
}

fn derived_priority_history(
    candidate: &AdaptiveTrialCandidate,
    metric_id: &MetricId,
) -> BrainResult<Vec<f64>> {
    candidate
        .gate_history
        .iter()
        .map(|gate| {
            gate.conservative_improvements
                .get(metric_id)
                .map(|value| value.get())
                .ok_or_else(|| integrity("adaptive_budget_priority_metric_missing"))
        })
        .collect()
}

fn adaptive_score(
    history: &[f64],
    planned_count: usize,
    total_effective_trials: usize,
    trial_cost: ComputeUnits,
    policy: &AdaptiveBudgetPolicy,
) -> BrainResult<f64> {
    let count = history
        .len()
        .checked_add(planned_count)
        .ok_or_else(|| invalid("adaptive_budget_count_overflow"))?;
    let empirical = if history.is_empty() {
        0.0
    } else {
        stable_mean(history.iter().copied())?
    };
    let denominator = (count + 1) as f64;
    let numerator = (total_effective_trials + 2) as f64;
    let exploration = policy.exploration_strength.get() * (numerator.ln() / denominator).sqrt();
    let cost = trial_cost.get() as f64;
    let score = empirical + exploration - policy.cost_penalty.get() * cost.ln_1p();
    if !score.is_finite() {
        return Err(BrainError::Numerical(
            "adaptive_budget_score_nonfinite".into(),
        ));
    }
    Ok(score)
}

pub fn allocate_adaptive_budget(
    candidates: &[AdaptiveTrialCandidate],
    policy: &AdaptiveBudgetPolicy,
) -> BrainResult<AdaptiveBudgetPlan> {
    if candidates.is_empty() || candidates.len() > MAX_CANDIDATES {
        return Err(invalid("adaptive_budget_candidate_count_invalid"));
    }
    let mut by_id = BTreeMap::new();
    let mut total_gate_history = 0_usize;
    for candidate in candidates {
        if candidate.maximum_additional_trials < policy.minimum_trials_per_candidate
            || by_id
                .insert(candidate.candidate_id.clone(), candidate)
                .is_some()
        {
            return Err(invalid("adaptive_budget_candidate_invalid"));
        }
        total_gate_history = total_gate_history
            .checked_add(candidate.gate_history.len())
            .ok_or_else(|| invalid("adaptive_budget_gate_history_overflow"))?;
        if total_gate_history > MAX_TOTAL_GATE_HISTORY {
            return Err(invalid("adaptive_budget_gate_history_limit_exceeded"));
        }
        for gate in &candidate.gate_history {
            gate.authenticate()?;
        }
    }
    // A planning iteration scans every candidate and, for every viable one,
    // derives its score from its bounded history. Charge the conservative
    // upper bound before allocating or hashing potentially large vectors.
    let per_iteration_work = by_id
        .len()
        .checked_add(total_gate_history)
        .ok_or_else(|| invalid("adaptive_budget_work_estimate_overflow"))?;
    let adaptive_work = policy
        .maximum_total_trials
        .checked_mul(per_iteration_work)
        .and_then(|work| work.checked_add(total_gate_history))
        .ok_or_else(|| invalid("adaptive_budget_work_estimate_overflow"))?;
    if adaptive_work > MAX_ADAPTIVE_WORK_UNITS {
        return Err(invalid("adaptive_budget_work_limit_exceeded"));
    }
    let candidate_projection = by_id
        .values()
        .map(|candidate| {
            (
                &candidate.candidate_id,
                candidate
                    .gate_history
                    .iter()
                    .map(|gate| &gate.decision_digest)
                    .collect::<Vec<_>>(),
                candidate.trial_cost,
                candidate.maximum_additional_trials,
            )
        })
        .collect::<Vec<_>>();
    let candidate_set_digest = Sha256Digest::digest_domain(
        BUDGET_PLAN_DOMAIN,
        &serde_json::to_vec(&candidate_projection)?,
    );
    let allocatable = policy
        .total_budget
        .get()
        .checked_sub(policy.holdout_reserve.get())
        .ok_or_else(|| invalid("adaptive_budget_reserve_exceeds_total"))?;
    let mandatory_cost = by_id.values().try_fold(0_u64, |total, candidate| {
        let candidate_cost = candidate
            .trial_cost
            .get()
            .checked_mul(policy.minimum_trials_per_candidate as u64)
            .ok_or_else(|| invalid("adaptive_budget_cost_overflow"))?;
        total
            .checked_add(candidate_cost)
            .ok_or_else(|| invalid("adaptive_budget_cost_overflow"))
    })?;
    if mandatory_cost > allocatable
        || by_id
            .len()
            .checked_mul(policy.minimum_trials_per_candidate)
            .is_none_or(|count| count > policy.maximum_total_trials)
    {
        let disposition = AdaptiveBudgetDisposition::BoundedUnknownInsufficientMandatoryCoverage;
        let allocations = Vec::new();
        let projection = AdaptiveBudgetPlanProjection {
            policy_digest: &policy.policy_digest,
            candidate_set_digest: &candidate_set_digest,
            disposition,
            allocations: &allocations,
            consumed_budget: 0,
            holdout_reserve: policy.holdout_reserve,
            unspent_budget: allocatable,
        };
        let plan_digest = AdaptiveBudgetPlanDigest::from_projection(
            BUDGET_PLAN_DOMAIN,
            &serde_json::to_vec(&projection)?,
        );
        return Ok(AdaptiveBudgetPlan {
            policy_digest: policy.policy_digest.clone(),
            candidate_set_digest,
            disposition,
            allocations,
            consumed_budget: 0,
            holdout_reserve: policy.holdout_reserve,
            unspent_budget: allocatable,
            plan_digest,
        });
    }

    let mut allocations = Vec::new();
    let mut planned = BTreeMap::<VariantId, usize>::new();
    let mut consumed = 0_u64;
    for ordinal in 0..policy.minimum_trials_per_candidate {
        for candidate in by_id.values() {
            consumed = consumed
                .checked_add(candidate.trial_cost.get())
                .ok_or_else(|| invalid("adaptive_budget_cost_overflow"))?;
            let count = planned.entry(candidate.candidate_id.clone()).or_default();
            *count += 1;
            allocations.push(TrialAllocation {
                candidate_id: candidate.candidate_id.clone(),
                candidate_trial_ordinal: ordinal,
                cost: candidate.trial_cost,
                reason: TrialAllocationReason::MandatoryCoverage,
            });
        }
    }

    while allocations.len() < policy.maximum_total_trials {
        let remaining = allocatable
            .checked_sub(consumed)
            .ok_or_else(|| integrity("adaptive_budget_consumption_invalid"))?;
        let total_history = by_id
            .values()
            .try_fold(allocations.len(), |total, candidate| {
                total
                    .checked_add(candidate.gate_history.len())
                    .ok_or_else(|| invalid("adaptive_budget_count_overflow"))
            })?;
        let mut best: Option<(&AdaptiveTrialCandidate, f64)> = None;
        for candidate in by_id.values() {
            let count = *planned.get(&candidate.candidate_id).unwrap_or(&0);
            if count >= candidate.maximum_additional_trials
                || candidate.trial_cost.get() > remaining
            {
                continue;
            }
            let history = derived_priority_history(candidate, &policy.priority_metric_id)?;
            let score =
                adaptive_score(&history, count, total_history, candidate.trial_cost, policy)?;
            let replace = best.as_ref().is_none_or(|(current, current_score)| {
                score.total_cmp(current_score) == Ordering::Greater
                    || (score.total_cmp(current_score) == Ordering::Equal
                        && candidate.candidate_id < current.candidate_id)
            });
            if replace {
                best = Some((candidate, score));
            }
        }
        let Some((candidate, _)) = best else {
            break;
        };
        consumed = consumed
            .checked_add(candidate.trial_cost.get())
            .ok_or_else(|| invalid("adaptive_budget_cost_overflow"))?;
        let count = planned.entry(candidate.candidate_id.clone()).or_default();
        let ordinal = *count;
        *count += 1;
        allocations.push(TrialAllocation {
            candidate_id: candidate.candidate_id.clone(),
            candidate_trial_ordinal: ordinal,
            cost: candidate.trial_cost,
            reason: TrialAllocationReason::DerivedUpperConfidenceBound,
        });
    }
    let unspent_budget = allocatable
        .checked_sub(consumed)
        .ok_or_else(|| integrity("adaptive_budget_consumption_invalid"))?;
    let disposition = AdaptiveBudgetDisposition::Planned;
    let projection = AdaptiveBudgetPlanProjection {
        policy_digest: &policy.policy_digest,
        candidate_set_digest: &candidate_set_digest,
        disposition,
        allocations: &allocations,
        consumed_budget: consumed,
        holdout_reserve: policy.holdout_reserve,
        unspent_budget,
    };
    let plan_digest = AdaptiveBudgetPlanDigest::from_projection(
        BUDGET_PLAN_DOMAIN,
        &serde_json::to_vec(&projection)?,
    );
    Ok(AdaptiveBudgetPlan {
        policy_digest: policy.policy_digest.clone(),
        candidate_set_digest,
        disposition,
        allocations,
        consumed_budget: consumed,
        holdout_reserve: policy.holdout_reserve,
        unspent_budget,
        plan_digest,
    })
}

// ---------------------------------------------------------------------------
// Sticky, staged canary evaluation. This module selects subjects and derives
// rollback requirements; an external runtime authority must execute rollback.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CanaryStage {
    exposure_per_million: u32,
    minimum_independent_groups: usize,
}

impl CanaryStage {
    pub fn new(exposure_per_million: u32, minimum_independent_groups: usize) -> BrainResult<Self> {
        if exposure_per_million == 0
            || exposure_per_million > PER_MILLION
            || !(2..=MAX_GROUPS).contains(&minimum_independent_groups)
        {
            return Err(invalid("canary_stage_invalid"));
        }
        Ok(Self {
            exposure_per_million,
            minimum_independent_groups,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanaryPolicy {
    metric_catalog_digest: MetricCatalogDigest,
    assignment_salt: Sha256Digest,
    stages: Vec<CanaryStage>,
    uncertainty_multiplier: FiniteF64,
    maximum_regression: BTreeMap<MetricId, FiniteF64>,
    policy_digest: CanaryPolicyDigest,
}

#[derive(Serialize)]
struct CanaryPolicyProjection<'a> {
    metric_catalog_digest: &'a MetricCatalogDigest,
    assignment_salt: &'a Sha256Digest,
    stages: &'a [CanaryStage],
    uncertainty_multiplier: FiniteF64,
    maximum_regression: &'a BTreeMap<MetricId, FiniteF64>,
}

impl CanaryPolicy {
    pub fn new(
        specs: &[MetricSpec],
        assignment_salt: Sha256Digest,
        stages: Vec<CanaryStage>,
        uncertainty_multiplier: f64,
        maximum_regression: BTreeMap<MetricId, FiniteF64>,
    ) -> BrainResult<Self> {
        let catalog = canonical_specs(specs)?;
        let metric_catalog_digest = metric_catalog_digest(&catalog)?;
        let uncertainty_multiplier = FiniteF64::new(uncertainty_multiplier)?;
        if assignment_salt == Sha256Digest::zero()
            || stages.is_empty()
            || stages.len() > MAX_CANARY_STAGES
            || uncertainty_multiplier.get() <= 0.0
            || uncertainty_multiplier.get() > 10.0
            || maximum_regression.len() != catalog.len()
            || maximum_regression
                .iter()
                .any(|(metric, value)| !catalog.contains_key(metric) || value.get() < 0.0)
        {
            return Err(invalid("canary_policy_invalid"));
        }
        for pair in stages.windows(2) {
            if pair[0].exposure_per_million >= pair[1].exposure_per_million
                || pair[0].minimum_independent_groups > pair[1].minimum_independent_groups
            {
                return Err(invalid("canary_stage_order_invalid"));
            }
        }
        let projection = CanaryPolicyProjection {
            metric_catalog_digest: &metric_catalog_digest,
            assignment_salt: &assignment_salt,
            stages: &stages,
            uncertainty_multiplier,
            maximum_regression: &maximum_regression,
        };
        let policy_digest = CanaryPolicyDigest::from_projection(
            CANARY_POLICY_DOMAIN,
            &serde_json::to_vec(&projection)?,
        );
        Ok(Self {
            metric_catalog_digest,
            assignment_salt,
            stages,
            uncertainty_multiplier,
            maximum_regression,
            policy_digest,
        })
    }

    pub fn digest(&self) -> &CanaryPolicyDigest {
        &self.policy_digest
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(tag = "phase", rename_all = "snake_case", deny_unknown_fields)]
pub enum CanaryPhase {
    AwaitingStage { stage_index: usize },
    CandidateValidated,
    RollbackRequired { stage_index: usize },
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CanaryStageDisposition {
    AdvanceStage,
    CandidateValidated,
    BoundedUnknown,
    RollbackRequired,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "reason", rename_all = "snake_case", deny_unknown_fields)]
pub enum CanaryReason {
    ProvenHardViolation { metric_id: MetricId },
    HardInvariantNotEstablished { metric_id: MetricId },
    DefiniteRegression { metric_id: MetricId },
    PossibleRegression { metric_id: MetricId },
    InsufficientGroups { metric_id: MetricId },
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CanaryState {
    baseline_id: VariantId,
    candidate_id: VariantId,
    gate_digest: CandidateGateDigest,
    evaluation_policy_digest: RobustEvaluationPolicyDigest,
    policy_digest: CanaryPolicyDigest,
    phase: CanaryPhase,
    last_disposition: Option<CanaryStageDisposition>,
    last_reasons: BTreeSet<CanaryReason>,
    evaluated_reports: BTreeSet<PairedEvaluationDigest>,
    evaluated_evidence_ids: BTreeSet<EvidenceId>,
    last_observation_end_tick: Option<u64>,
    state_digest: CanaryStateDigest,
}

#[derive(Serialize)]
struct CanaryStateProjection<'a> {
    baseline_id: &'a VariantId,
    candidate_id: &'a VariantId,
    gate_digest: &'a CandidateGateDigest,
    evaluation_policy_digest: &'a RobustEvaluationPolicyDigest,
    policy_digest: &'a CanaryPolicyDigest,
    phase: CanaryPhase,
    last_disposition: Option<CanaryStageDisposition>,
    last_reasons: &'a BTreeSet<CanaryReason>,
    evaluated_reports: &'a BTreeSet<PairedEvaluationDigest>,
    evaluated_evidence_ids: &'a BTreeSet<EvidenceId>,
    last_observation_end_tick: Option<u64>,
}

impl CanaryState {
    pub fn start(gate: &CandidateGateDecision, policy: &CanaryPolicy) -> BrainResult<Self> {
        gate.authenticate()?;
        if gate.disposition != CandidateGateDisposition::AdvanceCandidate
            || gate.metric_catalog_digest != policy.metric_catalog_digest
        {
            return Err(integrity("canary_candidate_not_eligible"));
        }
        Self::seal(CanaryStateDraft {
            baseline_id: gate.baseline_id.clone(),
            candidate_id: gate.candidate_id.clone(),
            gate_digest: gate.decision_digest.clone(),
            evaluation_policy_digest: gate.evaluation_policy_digest.clone(),
            policy_digest: policy.policy_digest.clone(),
            phase: CanaryPhase::AwaitingStage { stage_index: 0 },
            last_disposition: None,
            last_reasons: BTreeSet::new(),
            evaluated_reports: BTreeSet::new(),
            // Canary evidence must be fresh relative to the offline gate as
            // well as fresh between canary stages.
            evaluated_evidence_ids: gate.source_evidence_ids.clone(),
            last_observation_end_tick: Some(gate.source_observation_window.end_tick),
        })
    }

    pub fn phase(&self) -> CanaryPhase {
        self.phase
    }

    pub fn last_disposition(&self) -> Option<CanaryStageDisposition> {
        self.last_disposition
    }

    pub fn last_reasons(&self) -> &BTreeSet<CanaryReason> {
        &self.last_reasons
    }

    pub fn digest(&self) -> &CanaryStateDigest {
        &self.state_digest
    }

    pub const fn authorizes_promotion(&self) -> bool {
        false
    }

    pub fn subject_is_selected(
        &self,
        subject: &CanarySubjectId,
        policy: &CanaryPolicy,
    ) -> BrainResult<bool> {
        self.authenticate()?;
        if self.policy_digest != policy.policy_digest {
            return Err(integrity("canary_policy_binding_mismatch"));
        }
        let stage_index = match self.phase {
            CanaryPhase::AwaitingStage { stage_index } => stage_index,
            CanaryPhase::CandidateValidated => policy.stages.len().saturating_sub(1),
            CanaryPhase::RollbackRequired { .. } => return Ok(false),
        };
        let stage = policy
            .stages
            .get(stage_index)
            .ok_or_else(|| integrity("canary_stage_missing"))?;
        let payload = serde_json::to_vec(&(
            &policy.assignment_salt,
            &self.baseline_id,
            &self.candidate_id,
            subject,
        ))?;
        let digest = Sha256Digest::digest_domain(CANARY_BUCKET_DOMAIN, &payload);
        let bucket = u64::from_str_radix(&digest.as_str()[..16], 16)
            .map_err(|_| integrity("canary_bucket_digest_invalid"))?
            % u64::from(PER_MILLION);
        Ok(bucket < u64::from(stage.exposure_per_million))
    }

    pub fn evaluate_stage(
        self,
        specs: &[MetricSpec],
        report: &PairedEvaluationReport,
        policy: &CanaryPolicy,
    ) -> BrainResult<Self> {
        self.authenticate()?;
        report.authenticate()?;
        if matches!(self.phase, CanaryPhase::RollbackRequired { .. }) {
            return Err(integrity("canary_rollback_requirement_is_sticky"));
        }
        if self.phase == CanaryPhase::CandidateValidated {
            return Err(integrity("canary_already_validated"));
        }
        let CanaryPhase::AwaitingStage { stage_index } = self.phase else {
            return Err(integrity("canary_phase_invalid"));
        };
        let stage = policy
            .stages
            .get(stage_index)
            .ok_or_else(|| integrity("canary_stage_missing"))?;
        let catalog = canonical_specs(specs)?;
        let catalog_digest = metric_catalog_digest(&catalog)?;
        if self.policy_digest != policy.policy_digest
            || report.baseline_id != self.baseline_id
            || report.candidate_id != self.candidate_id
            || report.metric_catalog_digest != catalog_digest
            || policy.metric_catalog_digest != catalog_digest
            || report.evaluation_policy_digest != self.evaluation_policy_digest
            || self.evaluated_reports.contains(&report.report_digest)
            || !self
                .evaluated_evidence_ids
                .is_disjoint(&report.evidence_ids)
            || self
                .last_observation_end_tick
                .is_some_and(|end| report.observation_window.start_tick <= end)
        {
            return Err(integrity("canary_evaluation_binding_invalid"));
        }

        let multiplier = policy.uncertainty_multiplier.get();
        let mut hard = BTreeSet::new();
        let mut definite_regressions = BTreeSet::new();
        let mut unknown = BTreeSet::new();
        for (metric_id, spec) in &catalog {
            let estimate = report
                .estimates
                .get(metric_id)
                .ok_or_else(|| integrity("canary_metric_missing"))?;
            if let Some(invariant) = spec.hard_invariant.as_ref() {
                match estimate.candidate_observed_radius() {
                    Some(raw_radius) => {
                        let radius = multiplier * raw_radius;
                        match invariant.classify_interval(
                            estimate.candidate_center() - radius,
                            estimate.candidate_center() + radius,
                        ) {
                            InvariantIntervalStatus::ProvenViolation => {
                                hard.insert(CanaryReason::ProvenHardViolation {
                                    metric_id: metric_id.clone(),
                                });
                            }
                            InvariantIntervalStatus::NotEstablished => {
                                unknown.insert(CanaryReason::HardInvariantNotEstablished {
                                    metric_id: metric_id.clone(),
                                });
                            }
                            InvariantIntervalStatus::Established => {}
                        }
                    }
                    None => {
                        unknown.insert(CanaryReason::InsufficientGroups {
                            metric_id: metric_id.clone(),
                        });
                    }
                }
            }
            if estimate.independent_groups < stage.minimum_independent_groups {
                unknown.insert(CanaryReason::InsufficientGroups {
                    metric_id: metric_id.clone(),
                });
                continue;
            }
            let Some(raw_radius) = estimate.paired_effect_observed_radius() else {
                unknown.insert(CanaryReason::InsufficientGroups {
                    metric_id: metric_id.clone(),
                });
                continue;
            };
            let oriented = spec.direction.orient(estimate.paired_effect());
            let radius = multiplier * raw_radius;
            let allowed_regression = policy
                .maximum_regression
                .get(metric_id)
                .ok_or_else(|| integrity("canary_regression_limit_missing"))?
                .get();
            if oriented + radius < -allowed_regression {
                definite_regressions.insert(CanaryReason::DefiniteRegression {
                    metric_id: metric_id.clone(),
                });
            } else if oriented - radius < -allowed_regression {
                unknown.insert(CanaryReason::PossibleRegression {
                    metric_id: metric_id.clone(),
                });
            }
        }

        let (phase, disposition) = if !hard.is_empty() || !definite_regressions.is_empty() {
            (
                CanaryPhase::RollbackRequired { stage_index },
                CanaryStageDisposition::RollbackRequired,
            )
        } else if !unknown.is_empty() {
            (
                CanaryPhase::AwaitingStage { stage_index },
                CanaryStageDisposition::BoundedUnknown,
            )
        } else if stage_index + 1 < policy.stages.len() {
            (
                CanaryPhase::AwaitingStage {
                    stage_index: stage_index + 1,
                },
                CanaryStageDisposition::AdvanceStage,
            )
        } else {
            (
                CanaryPhase::CandidateValidated,
                CanaryStageDisposition::CandidateValidated,
            )
        };
        let reasons = hard
            .into_iter()
            .chain(definite_regressions)
            .chain(unknown)
            .collect();
        let mut evaluated_reports = self.evaluated_reports;
        evaluated_reports.insert(report.report_digest.clone());
        let mut evaluated_evidence_ids = self.evaluated_evidence_ids;
        evaluated_evidence_ids.extend(report.evidence_ids.iter().cloned());
        Self::seal(CanaryStateDraft {
            baseline_id: self.baseline_id,
            candidate_id: self.candidate_id,
            gate_digest: self.gate_digest,
            evaluation_policy_digest: self.evaluation_policy_digest,
            policy_digest: self.policy_digest,
            phase,
            last_disposition: Some(disposition),
            last_reasons: reasons,
            evaluated_reports,
            evaluated_evidence_ids,
            last_observation_end_tick: Some(report.observation_window.end_tick),
        })
    }

    fn seal(draft: CanaryStateDraft) -> BrainResult<Self> {
        let CanaryStateDraft {
            baseline_id,
            candidate_id,
            gate_digest,
            evaluation_policy_digest,
            policy_digest,
            phase,
            last_disposition,
            last_reasons,
            evaluated_reports,
            evaluated_evidence_ids,
            last_observation_end_tick,
        } = draft;
        let projection = CanaryStateProjection {
            baseline_id: &baseline_id,
            candidate_id: &candidate_id,
            gate_digest: &gate_digest,
            evaluation_policy_digest: &evaluation_policy_digest,
            policy_digest: &policy_digest,
            phase,
            last_disposition,
            last_reasons: &last_reasons,
            evaluated_reports: &evaluated_reports,
            evaluated_evidence_ids: &evaluated_evidence_ids,
            last_observation_end_tick,
        };
        let state_digest = CanaryStateDigest::from_projection(
            CANARY_STATE_DOMAIN,
            &serde_json::to_vec(&projection)?,
        );
        Ok(Self {
            baseline_id,
            candidate_id,
            gate_digest,
            evaluation_policy_digest,
            policy_digest,
            phase,
            last_disposition,
            last_reasons,
            evaluated_reports,
            evaluated_evidence_ids,
            last_observation_end_tick,
            state_digest,
        })
    }

    fn authenticate(&self) -> BrainResult<()> {
        let projection = CanaryStateProjection {
            baseline_id: &self.baseline_id,
            candidate_id: &self.candidate_id,
            gate_digest: &self.gate_digest,
            evaluation_policy_digest: &self.evaluation_policy_digest,
            policy_digest: &self.policy_digest,
            phase: self.phase,
            last_disposition: self.last_disposition,
            last_reasons: &self.last_reasons,
            evaluated_reports: &self.evaluated_reports,
            evaluated_evidence_ids: &self.evaluated_evidence_ids,
            last_observation_end_tick: self.last_observation_end_tick,
        };
        let calculated = CanaryStateDigest::from_projection(
            CANARY_STATE_DOMAIN,
            &serde_json::to_vec(&projection)?,
        );
        if calculated != self.state_digest {
            return Err(integrity("canary_state_digest_mismatch"));
        }
        Ok(())
    }
}

struct CanaryStateDraft {
    baseline_id: VariantId,
    candidate_id: VariantId,
    gate_digest: CandidateGateDigest,
    evaluation_policy_digest: RobustEvaluationPolicyDigest,
    policy_digest: CanaryPolicyDigest,
    phase: CanaryPhase,
    last_disposition: Option<CanaryStageDisposition>,
    last_reasons: BTreeSet<CanaryReason>,
    evaluated_reports: BTreeSet<PairedEvaluationDigest>,
    evaluated_evidence_ids: BTreeSet<EvidenceId>,
    last_observation_end_tick: Option<u64>,
}

/// Authenticates the complete governance chain required before an adapter can
/// be considered for promotion. The caller still owns persistence,
/// materialization, authorization and activation; this function only proves
/// that the supplied sealed witnesses form one eligible, independent chain.
pub(crate) fn authenticate_adapter_promotion_witnesses(
    expected_candidate: &VariantId,
    gate: &CandidateGateDecision,
    petfc: &PetfcAssessment,
    canary: &CanaryState,
) -> BrainResult<()> {
    gate.authenticate()?;
    petfc.authenticate()?;
    canary.authenticate()?;

    if &gate.candidate_id != expected_candidate
        || &petfc.candidate_id != expected_candidate
        || &canary.candidate_id != expected_candidate
        || gate.baseline_id != petfc.baseline_id
        || gate.baseline_id != canary.baseline_id
        || gate.metric_catalog_digest != petfc.metric_catalog_digest
        || gate.evaluation_policy_digest != petfc.evaluation_policy_digest
        || gate.evaluation_policy_digest != canary.evaluation_policy_digest
        || gate.independence_design_digest != petfc.independence_design_digest
        || gate.decision_digest != canary.gate_digest
    {
        return Err(integrity("adapter_promotion_governance_binding_mismatch"));
    }

    if gate.disposition != CandidateGateDisposition::AdvanceCandidate
        || petfc.disposition != PetfcDisposition::CompatibleForNextGate
        || canary.phase != CanaryPhase::CandidateValidated
        || canary.last_disposition != Some(CanaryStageDisposition::CandidateValidated)
        || !gate.reasons.is_empty()
        || !petfc.reasons.is_empty()
        || !canary.last_reasons.is_empty()
    {
        return Err(integrity("adapter_promotion_governance_not_approved"));
    }

    // Enforce a strict evidence chronology: PETFC trajectory, offline gate,
    // then fresh canary stages. This prevents one observation from satisfying
    // multiple authorities under different labels.
    if gate.source_evidence_ids.is_empty()
        || petfc.trajectory_evidence_ids.is_empty()
        || canary.evaluated_reports.is_empty()
        || !gate
            .source_evidence_ids
            .is_disjoint(&petfc.trajectory_evidence_ids)
        || !petfc
            .trajectory_evidence_ids
            .is_disjoint(&canary.evaluated_evidence_ids)
        || !canary
            .evaluated_evidence_ids
            .is_superset(&gate.source_evidence_ids)
        || canary.evaluated_evidence_ids.len() <= gate.source_evidence_ids.len()
        || petfc.trajectory_observation_window.end_tick >= gate.source_observation_window.start_tick
        || canary
            .last_observation_end_tick
            .is_none_or(|end| end <= gate.source_observation_window.end_tick)
    {
        return Err(integrity("adapter_promotion_governance_evidence_invalid"));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(value: &str) -> MetricId {
        MetricId::parse(value).unwrap()
    }

    fn variant(value: &str) -> VariantId {
        VariantId::parse(value).unwrap()
    }

    fn default_specs() -> Vec<MetricSpec> {
        vec![
            MetricSpec::new(
                metric("quality"),
                MetricDirection::Maximize,
                Some(HardInvariant::at_least(0.0).unwrap()),
            )
            .unwrap(),
            MetricSpec::new(
                metric("loss"),
                MetricDirection::Minimize,
                Some(HardInvariant::at_most(1.0).unwrap()),
            )
            .unwrap(),
        ]
    }

    fn report_with_rows(
        specs: &[MetricSpec],
        baseline: &str,
        candidate: &str,
        start_tick: u64,
        rows: &[(f64, f64, f64, f64)],
        minimum_groups: usize,
    ) -> PairedEvaluationReport {
        let mut observations = Vec::new();
        for (index, (baseline_quality, candidate_quality, baseline_loss, candidate_loss)) in
            rows.iter().copied().enumerate()
        {
            let unit = PairedExperimentalUnit::new(
                // A group denotes the independent experimental unit, not the
                // observation time. Keeping that identity stable is required
                // for longitudinal PETFC and staged-canary comparisons.
                IndependenceGroupId::parse(format!("group-{index}")).unwrap(),
                PairId::parse(format!("pair-{start_tick}-{index}")).unwrap(),
                EvidenceId::parse(format!("evidence-{start_tick}-{index}")).unwrap(),
                ObservationWindow::new(start_tick, start_tick + 1).unwrap(),
            );
            observations.push(
                PairedObservation::new(
                    metric("quality"),
                    unit.clone(),
                    baseline_quality,
                    candidate_quality,
                )
                .unwrap(),
            );
            observations.push(
                PairedObservation::new(metric("loss"), unit, baseline_loss, candidate_loss)
                    .unwrap(),
            );
        }
        evaluate_paired_groups(
            variant(baseline),
            variant(candidate),
            specs,
            &observations,
            &RobustEvaluationPolicy::new(minimum_groups, minimum_groups.min(3), 1_000).unwrap(),
        )
        .unwrap()
    }

    fn constant_rows(
        groups: usize,
        baseline_quality: f64,
        candidate_quality: f64,
        baseline_loss: f64,
        candidate_loss: f64,
    ) -> Vec<(f64, f64, f64, f64)> {
        vec![
            (
                baseline_quality,
                candidate_quality,
                baseline_loss,
                candidate_loss,
            );
            groups
        ]
    }

    fn gate_policy(specs: &[MetricSpec], minimum_groups: usize) -> CandidateGatePolicy {
        CandidateGatePolicy::new(
            specs,
            minimum_groups,
            1.0,
            BTreeMap::from([
                (metric("loss"), FiniteF64::new(0.0).unwrap()),
                (metric("quality"), FiniteF64::new(0.0).unwrap()),
            ]),
        )
        .unwrap()
    }

    fn petfc_policy(specs: &[MetricSpec]) -> PetfcPolicy {
        PetfcPolicy::new(
            specs,
            metric("quality"),
            vec![
                PetfcMetricPolicy::new(metric("quality"), 1.0, 0.0).unwrap(),
                PetfcMetricPolicy::new(metric("loss"), 1.0, 0.0).unwrap(),
            ],
            PetfcPathLimits::new(2, 8, 2.0, 2.0, 0.6, 0.0).unwrap(),
            PetfcConservationLimits::new(0.0, 0, 2.0).unwrap(),
            PetfcUtilityPolicy::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        )
        .unwrap()
    }

    fn eligible_candidate(
        candidate: &str,
        final_quality: f64,
        final_loss: f64,
        start_tick: u64,
    ) -> (CandidateGateDecision, PetfcAssessment) {
        let specs = default_specs();
        let middle = format!("{candidate}-middle");
        let first = report_with_rows(
            &specs,
            "baseline",
            &middle,
            start_tick,
            &constant_rows(3, 0.2, 0.4, 0.8, 0.6),
            3,
        );
        let second = report_with_rows(
            &specs,
            "baseline",
            candidate,
            start_tick + 10,
            &constant_rows(3, 0.2, final_quality, 0.8, final_loss),
            3,
        );
        // Pareto objectives must be measured against the same global
        // baseline as the PETFC trajectory, not against a candidate-specific
        // predecessor on that trajectory.
        let global_gate_report = report_with_rows(
            &specs,
            "baseline",
            candidate,
            start_tick + 20,
            &constant_rows(3, 0.2, final_quality, 0.8, final_loss),
            3,
        );
        let gate = decide_candidate(&specs, &global_gate_report, &gate_policy(&specs, 3)).unwrap();
        let petfc = petfc_policy(&specs);
        let trajectory = PetfcTrajectory::start(&first, &petfc)
            .unwrap()
            .append_report(&second)
            .unwrap();
        let assessment = evaluate_petfc(&specs, &trajectory, &petfc).unwrap();
        assert_eq!(
            assessment.disposition(),
            PetfcDisposition::CompatibleForNextGate
        );
        (gate, assessment)
    }

    #[test]
    fn paired_evidence_rejects_group_relabel_and_evidence_reuse() {
        let specs = default_specs();
        let shared_pair = PairId::parse("pair-shared").unwrap();
        let quality_unit = PairedExperimentalUnit::new(
            IndependenceGroupId::parse("g-1").unwrap(),
            shared_pair.clone(),
            EvidenceId::parse("e-1").unwrap(),
            ObservationWindow::new(1, 2).unwrap(),
        );
        let loss_unit = PairedExperimentalUnit::new(
            IndependenceGroupId::parse("g-2").unwrap(),
            shared_pair,
            EvidenceId::parse("e-1").unwrap(),
            ObservationWindow::new(1, 2).unwrap(),
        );
        let observations = vec![
            PairedObservation::new(metric("quality"), quality_unit, 0.2, 0.3).unwrap(),
            PairedObservation::new(metric("loss"), loss_unit, 0.8, 0.7).unwrap(),
        ];
        let error = evaluate_paired_groups(
            variant("base"),
            variant("candidate"),
            &specs,
            &observations,
            &RobustEvaluationPolicy::new(2, 2, 10).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("pair_identity_relabelled"));
    }

    #[test]
    fn absolute_level_and_paired_effect_have_separate_empirical_envelopes() {
        let specs = default_specs();
        let report = report_with_rows(
            &specs,
            "base",
            "candidate",
            10,
            &[
                (0.0, 0.1, 0.1, 0.2),
                (10.0, 10.1, 0.4, 0.5),
                (20.0, 20.1, 0.7, 0.8),
            ],
            3,
        );
        let estimate = report.estimates().get(&metric("quality")).unwrap();
        assert!(estimate.candidate_observed_radius().unwrap() > 9.0);
        assert!(estimate.paired_effect_observed_radius().unwrap() < 1.0e-12);

        let mut tampered = report.clone();
        tampered
            .estimates
            .get_mut(&metric("quality"))
            .unwrap()
            .candidate_center = FiniteF64::new(99.0).unwrap();
        assert!(tampered.authenticate().is_err());
    }

    #[test]
    fn evaluation_minimum_cannot_be_weakened_by_gate_policy() {
        let specs = default_specs();
        let report = report_with_rows(
            &specs,
            "base",
            "candidate",
            20,
            &constant_rows(3, 0.2, 0.3, 0.8, 0.7),
            4,
        );
        assert!(!report.evidence_sufficient());
        assert!(decide_candidate(&specs, &report, &gate_policy(&specs, 3)).is_err());
        let decision = decide_candidate(&specs, &report, &gate_policy(&specs, 4)).unwrap();
        assert_eq!(
            decision.disposition(),
            CandidateGateDisposition::BoundedUnknown
        );
    }

    #[test]
    fn hard_invariant_overlap_is_unknown_not_a_proven_violation() {
        let specs = vec![
            MetricSpec::new(
                metric("quality"),
                MetricDirection::Maximize,
                Some(HardInvariant::at_most(15.0).unwrap()),
            )
            .unwrap(),
            MetricSpec::new(metric("loss"), MetricDirection::Minimize, None).unwrap(),
        ];
        let report = report_with_rows(
            &specs,
            "base",
            "candidate",
            30,
            &[
                (0.0, 1.0, 0.8, 0.7),
                (10.0, 11.0, 0.8, 0.7),
                (20.0, 21.0, 0.8, 0.7),
            ],
            3,
        );
        let decision = decide_candidate(&specs, &report, &gate_policy(&specs, 3)).unwrap();
        assert_eq!(
            decision.disposition(),
            CandidateGateDisposition::BoundedUnknown
        );
        assert!(decision
            .reasons()
            .contains(&GateReason::HardInvariantNotEstablished {
                metric_id: metric("quality")
            }));
        assert!(!decision
            .reasons()
            .contains(&GateReason::ProvenHardViolation {
                metric_id: metric("quality")
            }));
    }

    #[test]
    fn hard_invariant_rejects_only_when_violation_is_proven() {
        let specs = vec![
            MetricSpec::new(
                metric("quality"),
                MetricDirection::Maximize,
                Some(HardInvariant::at_most(15.0).unwrap()),
            )
            .unwrap(),
            MetricSpec::new(metric("loss"), MetricDirection::Minimize, None).unwrap(),
        ];
        let report = report_with_rows(
            &specs,
            "base",
            "candidate",
            35,
            &constant_rows(3, 0.0, 16.0, 0.8, 0.7),
            3,
        );
        let decision = decide_candidate(&specs, &report, &gate_policy(&specs, 3)).unwrap();
        assert_eq!(decision.disposition(), CandidateGateDisposition::Reject);
        assert!(decision
            .reasons()
            .contains(&GateReason::ProvenHardViolation {
                metric_id: metric("quality")
            }));
    }

    #[test]
    fn petfc_direct_path_has_unit_tortuosity_and_zero_waste() {
        let specs = default_specs();
        let first = report_with_rows(
            &specs,
            "base",
            "middle",
            40,
            &constant_rows(3, 0.2, 0.4, 0.8, 0.6),
            3,
        );
        let second = report_with_rows(
            &specs,
            "base",
            "candidate",
            50,
            &constant_rows(3, 0.2, 0.6, 0.8, 0.4),
            3,
        );
        let policy = petfc_policy(&specs);
        let trajectory = PetfcTrajectory::start(&first, &policy)
            .unwrap()
            .append_report(&second)
            .unwrap();
        let assessment = evaluate_petfc(&specs, &trajectory, &policy).unwrap();
        assert_eq!(
            assessment.disposition(),
            PetfcDisposition::CompatibleForNextGate
        );
        assert!((assessment.path_length_upper().unwrap() - 0.4).abs() < 1.0e-12);
        assert!((assessment.endpoint_distance_lower().unwrap() - 0.4).abs() < 1.0e-12);
        assert!((assessment.tortuosity_upper().unwrap() - 1.0).abs() < 1.0e-12);
        assert!(assessment.waste_upper().unwrap().abs() < 1.0e-12);
        assert!(!assessment.authorizes_promotion());
    }

    #[test]
    fn petfc_reproduces_pythagorean_staircase_and_enforces_time_chain() {
        let specs = vec![
            MetricSpec::new(metric("quality"), MetricDirection::Maximize, None).unwrap(),
            MetricSpec::new(metric("loss"), MetricDirection::Maximize, None).unwrap(),
        ];
        let first = report_with_rows(
            &specs,
            "base",
            "middle",
            60,
            &constant_rows(3, 0.0, 1.0, 0.0, 0.0),
            3,
        );
        let overlapping = report_with_rows(
            &specs,
            "base",
            "candidate",
            61,
            &constant_rows(3, 0.0, 1.0, 0.0, 1.0),
            3,
        );
        let second = report_with_rows(
            &specs,
            "base",
            "candidate",
            70,
            &constant_rows(3, 0.0, 1.0, 0.0, 1.0),
            3,
        );
        let policy = PetfcPolicy::new(
            &specs,
            metric("quality"),
            vec![
                PetfcMetricPolicy::new(metric("quality"), 1.0, 0.0).unwrap(),
                PetfcMetricPolicy::new(metric("loss"), 1.0, 0.0).unwrap(),
            ],
            PetfcPathLimits::new(2, 8, 2.0, 2.0, 0.5, 0.0).unwrap(),
            PetfcConservationLimits::new(0.0, 0, 2.0).unwrap(),
            PetfcUtilityPolicy::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        )
        .unwrap();
        assert!(PetfcTrajectory::start(&first, &policy)
            .unwrap()
            .append_report(&overlapping)
            .is_err());
        let trajectory = PetfcTrajectory::start(&first, &policy)
            .unwrap()
            .append_report(&second)
            .unwrap();
        let assessment = evaluate_petfc(&specs, &trajectory, &policy).unwrap();
        let expected = 2.0_f64.sqrt();
        assert!((assessment.tortuosity_upper().unwrap() - expected).abs() < 1.0e-12);
        assert!((assessment.waste_upper().unwrap() - (1.0 - 1.0 / expected)).abs() < 1.0e-12);
    }

    #[test]
    fn petfc_geometry_uses_incumbent_anchored_effects_and_rejects_duplicate_checkpoint() {
        let specs = vec![
            MetricSpec::new(metric("quality"), MetricDirection::Maximize, None).unwrap(),
            MetricSpec::new(metric("loss"), MetricDirection::Maximize, None).unwrap(),
        ];
        let first = report_with_rows(
            &specs,
            "base",
            "candidate-a",
            74,
            &constant_rows(3, 100.0, 101.0, 50.0, 50.0),
            3,
        );
        let second = report_with_rows(
            &specs,
            "base",
            "candidate-b",
            84,
            &constant_rows(3, 1_000.0, 1_002.0, -500.0, -500.0),
            3,
        );
        let duplicate = report_with_rows(
            &specs,
            "base",
            "candidate-b",
            94,
            &constant_rows(3, 2_000.0, 2_002.0, 0.0, 0.0),
            3,
        );
        let policy = PetfcPolicy::new(
            &specs,
            metric("quality"),
            vec![
                PetfcMetricPolicy::new(metric("quality"), 1.0, 0.0).unwrap(),
                PetfcMetricPolicy::new(metric("loss"), 1.0, 0.0).unwrap(),
            ],
            PetfcPathLimits::new(2, 8, 10.0, 2.0, 0.6, 0.0).unwrap(),
            PetfcConservationLimits::new(0.0, 0, 2.0).unwrap(),
            PetfcUtilityPolicy::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        )
        .unwrap();
        let trajectory = PetfcTrajectory::start(&first, &policy)
            .unwrap()
            .append_report(&second)
            .unwrap();
        let assessment = evaluate_petfc(&specs, &trajectory, &policy).unwrap();
        let expected = 2.0_f64.sqrt();
        assert!((assessment.path_length_upper().unwrap() - expected).abs() < 1.0e-12);
        assert!((assessment.endpoint_distance_lower().unwrap() - expected).abs() < 1.0e-12);
        assert!(trajectory.append_report(&duplicate).is_err());
    }

    #[test]
    fn petfc_invariant_overlap_is_bounded_unknown_not_rollback() {
        let specs = vec![
            MetricSpec::new(
                metric("quality"),
                MetricDirection::Maximize,
                Some(HardInvariant::at_most(15.0).unwrap()),
            )
            .unwrap(),
            MetricSpec::new(metric("loss"), MetricDirection::Minimize, None).unwrap(),
        ];
        let rows = [
            (0.0, 14.0, 0.8, 0.7),
            (0.0, 15.0, 0.8, 0.7),
            (0.0, 16.0, 0.8, 0.7),
        ];
        let first = report_with_rows(&specs, "base", "candidate-a", 104, &rows, 3);
        let second = report_with_rows(&specs, "base", "candidate-b", 114, &rows, 3);
        let policy = PetfcPolicy::new(
            &specs,
            metric("quality"),
            vec![
                PetfcMetricPolicy::new(metric("quality"), 100.0, 0.0).unwrap(),
                PetfcMetricPolicy::new(metric("loss"), 1.0, 0.0).unwrap(),
            ],
            PetfcPathLimits::new(2, 8, 2.0, 2.0, 0.6, 0.0).unwrap(),
            PetfcConservationLimits::new(0.0, 0, 2.0).unwrap(),
            PetfcUtilityPolicy::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        )
        .unwrap();
        let assessment = evaluate_petfc(
            &specs,
            &PetfcTrajectory::start(&first, &policy)
                .unwrap()
                .append_report(&second)
                .unwrap(),
            &policy,
        )
        .unwrap();
        assert_eq!(assessment.disposition(), PetfcDisposition::BoundedUnknown);
        assert!(assessment
            .reasons()
            .contains(&PetfcReason::HardInvariantNotEstablished {
                metric_id: metric("quality"),
                checkpoint_id: variant("candidate-b"),
            }));
        assert!(!assessment
            .reasons()
            .iter()
            .any(|reason| matches!(reason, PetfcReason::ProvenHardViolation { .. })));
    }

    #[test]
    fn petfc_policy_is_precommitted_and_cannot_be_changed_after_observation() {
        let specs = default_specs();
        let report = report_with_rows(
            &specs,
            "base",
            "candidate",
            80,
            &constant_rows(3, 0.2, 0.4, 0.8, 0.6),
            3,
        );
        let committed = petfc_policy(&specs);
        let trajectory = PetfcTrajectory::start(&report, &committed).unwrap();
        let changed = PetfcPolicy::new(
            &specs,
            metric("quality"),
            vec![
                PetfcMetricPolicy::new(metric("quality"), 1.0, 0.0).unwrap(),
                PetfcMetricPolicy::new(metric("loss"), 1.0, 0.0).unwrap(),
            ],
            PetfcPathLimits::new(2, 8, 2.0, 2.0, 0.6, 0.0).unwrap(),
            PetfcConservationLimits::new(0.0, 0, 2.0).unwrap(),
            PetfcUtilityPolicy::new(0.1, 0.0, 0.0, 0.0).unwrap(),
        )
        .unwrap();
        assert!(evaluate_petfc(&specs, &trajectory, &changed).is_err());
    }

    #[test]
    fn pareto_requires_both_sealed_gates_and_is_order_invariant() {
        let (gate_a, petfc_a) = eligible_candidate("candidate-a", 0.7, 0.3, 100);
        let (gate_b, petfc_b) = eligible_candidate("candidate-b", 0.5, 0.5, 200);
        let candidate_a = ParetoCandidate::from_verified_gates(&gate_a, &petfc_a).unwrap();
        let candidate_b = ParetoCandidate::from_verified_gates(&gate_b, &petfc_b).unwrap();
        let forward =
            deterministic_pareto_front(&[candidate_a.clone(), candidate_b.clone()]).unwrap();
        let reverse = deterministic_pareto_front(&[candidate_b, candidate_a]).unwrap();
        assert_eq!(forward.selected(), reverse.selected());
        assert_eq!(forward.selected(), &[variant("candidate-a")]);
        assert!(!forward.authorizes_promotion());

        assert!(ParetoCandidate::from_verified_gates(&gate_a, &petfc_b).is_err());

        let specs = default_specs();
        let wrong_baseline_report = report_with_rows(
            &specs,
            "different-baseline",
            "candidate-a",
            500,
            &constant_rows(3, 0.2, 0.7, 0.8, 0.3),
            3,
        );
        let wrong_baseline_gate =
            decide_candidate(&specs, &wrong_baseline_report, &gate_policy(&specs, 3)).unwrap();
        assert!(ParetoCandidate::from_verified_gates(&wrong_baseline_gate, &petfc_a).is_err());
    }

    #[test]
    fn pareto_rejects_quadratic_work_before_pairwise_scan() {
        let (gate, petfc) = eligible_candidate("candidate-template", 0.7, 0.3, 600);
        let template = ParetoCandidate::from_verified_gates(&gate, &petfc).unwrap();
        let candidates = (0..2_002)
            .map(|index| {
                let mut candidate = template.clone();
                candidate.candidate_id = variant(&format!("candidate-{index}"));
                candidate
            })
            .collect::<Vec<_>>();
        let error = deterministic_pareto_front(&candidates).unwrap_err();
        assert!(
            matches!(error, BrainError::Invalid(message) if message == "pareto_work_limit_exceeded")
        );
    }

    #[test]
    fn adaptive_budget_covers_every_candidate_before_exploitation() {
        let candidates = vec![
            AdaptiveTrialCandidate::from_gate_history(
                variant("candidate-a"),
                Vec::new(),
                ComputeUnits::new(2).unwrap(),
                8,
            )
            .unwrap(),
            AdaptiveTrialCandidate::from_gate_history(
                variant("candidate-b"),
                Vec::new(),
                ComputeUnits::new(2).unwrap(),
                8,
            )
            .unwrap(),
        ];
        let policy = AdaptiveBudgetPolicy::new(AdaptiveBudgetPolicyDraft {
            priority_metric_id: metric("quality"),
            total_budget: ComputeUnits::new(14).unwrap(),
            holdout_reserve: ComputeUnits::new(2).unwrap(),
            minimum_trials_per_candidate: 1,
            maximum_total_trials: 6,
            exploration_strength: 1.0,
            cost_penalty: 0.0,
        })
        .unwrap();
        let plan = allocate_adaptive_budget(&candidates, &policy).unwrap();
        assert_eq!(plan.disposition(), AdaptiveBudgetDisposition::Planned);
        assert_eq!(plan.allocations().len(), 6);
        assert_eq!(
            plan.allocations()[0].reason,
            TrialAllocationReason::MandatoryCoverage
        );
        assert_eq!(
            plan.allocations()[1].reason,
            TrialAllocationReason::MandatoryCoverage
        );
        assert_ne!(
            plan.allocations()[0].candidate_id,
            plan.allocations()[1].candidate_id
        );
        assert!(!plan.authorizes_promotion());
    }

    #[test]
    fn adaptive_budget_fails_closed_when_mandatory_coverage_does_not_fit() {
        let candidates = vec![
            AdaptiveTrialCandidate::from_gate_history(
                variant("candidate-a"),
                Vec::new(),
                ComputeUnits::new(5).unwrap(),
                2,
            )
            .unwrap(),
            AdaptiveTrialCandidate::from_gate_history(
                variant("candidate-b"),
                Vec::new(),
                ComputeUnits::new(5).unwrap(),
                2,
            )
            .unwrap(),
        ];
        let policy = AdaptiveBudgetPolicy::new(AdaptiveBudgetPolicyDraft {
            priority_metric_id: metric("quality"),
            total_budget: ComputeUnits::new(10).unwrap(),
            holdout_reserve: ComputeUnits::new(2).unwrap(),
            minimum_trials_per_candidate: 1,
            maximum_total_trials: 2,
            exploration_strength: 1.0,
            cost_penalty: 0.0,
        })
        .unwrap();
        let plan = allocate_adaptive_budget(&candidates, &policy).unwrap();
        assert_eq!(
            plan.disposition(),
            AdaptiveBudgetDisposition::BoundedUnknownInsufficientMandatoryCoverage
        );
        assert!(plan.allocations().is_empty());
    }

    #[test]
    fn adaptive_budget_rejects_unbounded_history_and_work_before_planning() {
        let specs = default_specs();
        let report = report_with_rows(
            &specs,
            "base",
            "candidate-a",
            700,
            &constant_rows(3, 0.2, 0.4, 0.8, 0.6),
            3,
        );
        let gate = decide_candidate(&specs, &report, &gate_policy(&specs, 3)).unwrap();
        let oversized_history = vec![gate; MAX_GATE_HISTORY_PER_CANDIDATE + 1];
        assert!(AdaptiveTrialCandidate::from_gate_history(
            variant("candidate-a"),
            oversized_history,
            ComputeUnits::new(1).unwrap(),
            2,
        )
        .is_err());

        let candidates = (0..5)
            .map(|index| {
                AdaptiveTrialCandidate::from_gate_history(
                    variant(&format!("work-candidate-{index}")),
                    Vec::new(),
                    ComputeUnits::new(1).unwrap(),
                    MAX_OBSERVATIONS,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let policy = AdaptiveBudgetPolicy::new(AdaptiveBudgetPolicyDraft {
            priority_metric_id: metric("quality"),
            total_budget: ComputeUnits::new(2_000_000).unwrap(),
            holdout_reserve: ComputeUnits::new(1).unwrap(),
            minimum_trials_per_candidate: 1,
            maximum_total_trials: MAX_OBSERVATIONS,
            exploration_strength: 1.0,
            cost_penalty: 0.0,
        })
        .unwrap();
        let error = allocate_adaptive_budget(&candidates, &policy).unwrap_err();
        assert!(
            matches!(error, BrainError::Invalid(message) if message == "adaptive_budget_work_limit_exceeded")
        );
    }

    fn canary_policy(specs: &[MetricSpec]) -> CanaryPolicy {
        CanaryPolicy::new(
            specs,
            Sha256Digest::digest_bytes(b"canary-test-salt"),
            vec![
                CanaryStage::new(100_000, 3).unwrap(),
                CanaryStage::new(500_000, 4).unwrap(),
            ],
            1.0,
            BTreeMap::from([
                (metric("loss"), FiniteF64::new(0.0).unwrap()),
                (metric("quality"), FiniteF64::new(0.0).unwrap()),
            ]),
        )
        .unwrap()
    }

    fn validated_canary(specs: &[MetricSpec], gate: &CandidateGateDecision) -> CanaryState {
        let policy = canary_policy(specs);
        let first_tick = gate.source_observation_window.end_tick + 10;
        let first = report_with_rows(
            specs,
            gate.baseline_id.as_str(),
            gate.candidate_id.as_str(),
            first_tick,
            &constant_rows(3, 0.2, 0.7, 0.8, 0.3),
            3,
        );
        let state = CanaryState::start(gate, &policy)
            .unwrap()
            .evaluate_stage(specs, &first, &policy)
            .unwrap();
        let second = report_with_rows(
            specs,
            gate.baseline_id.as_str(),
            gate.candidate_id.as_str(),
            first_tick + 10,
            &constant_rows(4, 0.2, 0.7, 0.8, 0.3),
            3,
        );
        state.evaluate_stage(specs, &second, &policy).unwrap()
    }

    #[test]
    fn adapter_promotion_witnesses_accept_one_fully_bound_chain() {
        let specs = default_specs();
        let (gate, petfc) = eligible_candidate("candidate", 0.7, 0.3, 100);
        let canary = validated_canary(&specs, &gate);

        authenticate_adapter_promotion_witnesses(&variant("candidate"), &gate, &petfc, &canary)
            .unwrap();
    }

    #[test]
    fn adapter_promotion_witnesses_reject_wrong_candidate_or_pending_canary() {
        let specs = default_specs();
        let (gate, petfc) = eligible_candidate("candidate", 0.7, 0.3, 100);
        let policy = canary_policy(&specs);
        let pending = CanaryState::start(&gate, &policy).unwrap();

        assert!(authenticate_adapter_promotion_witnesses(
            &variant("different-candidate"),
            &gate,
            &petfc,
            &pending,
        )
        .is_err());
        assert!(authenticate_adapter_promotion_witnesses(
            &variant("candidate"),
            &gate,
            &petfc,
            &pending,
        )
        .is_err());
    }

    #[test]
    fn adapter_promotion_witnesses_reject_incoherent_evaluation_design() {
        let specs = default_specs();
        let (gate, _) = eligible_candidate("candidate", 0.7, 0.3, 100);
        let petfc_policy = petfc_policy(&specs);
        let first = report_with_rows(
            &specs,
            "baseline",
            "candidate-middle",
            100,
            &constant_rows(4, 0.2, 0.4, 0.8, 0.6),
            4,
        );
        let second = report_with_rows(
            &specs,
            "baseline",
            "candidate",
            110,
            &constant_rows(4, 0.2, 0.7, 0.8, 0.3),
            4,
        );
        let trajectory = PetfcTrajectory::start(&first, &petfc_policy)
            .unwrap()
            .append_report(&second)
            .unwrap();
        let petfc = evaluate_petfc(&specs, &trajectory, &petfc_policy).unwrap();
        let canary = validated_canary(&specs, &gate);

        assert_eq!(petfc.disposition(), PetfcDisposition::CompatibleForNextGate);
        assert!(authenticate_adapter_promotion_witnesses(
            &variant("candidate"),
            &gate,
            &petfc,
            &canary,
        )
        .is_err());
    }

    #[test]
    fn canary_hard_violation_overrides_insufficient_groups_and_is_sticky() {
        let specs = default_specs();
        let initial = report_with_rows(
            &specs,
            "base",
            "candidate",
            300,
            &constant_rows(3, 0.2, 0.4, 0.8, 0.6),
            3,
        );
        let gate = decide_candidate(&specs, &initial, &gate_policy(&specs, 3)).unwrap();
        let policy = canary_policy(&specs);
        let state = CanaryState::start(&gate, &policy).unwrap();
        let bad = report_with_rows(
            &specs,
            "base",
            "candidate",
            310,
            &constant_rows(2, 0.2, 0.4, 0.8, 1.5),
            3,
        );
        let state = state.evaluate_stage(&specs, &bad, &policy).unwrap();
        assert_eq!(
            state.last_disposition(),
            Some(CanaryStageDisposition::RollbackRequired)
        );
        assert!(matches!(
            state.phase(),
            CanaryPhase::RollbackRequired { stage_index: 0 }
        ));
        assert!(state
            .last_reasons()
            .contains(&CanaryReason::ProvenHardViolation {
                metric_id: metric("loss")
            }));
        let later_good = report_with_rows(
            &specs,
            "base",
            "candidate",
            320,
            &constant_rows(4, 0.2, 0.5, 0.8, 0.5),
            3,
        );
        assert!(state.evaluate_stage(&specs, &later_good, &policy).is_err());
    }

    #[test]
    fn canary_assignment_is_sticky_and_monotonic_across_stages() {
        let specs = default_specs();
        let initial = report_with_rows(
            &specs,
            "base",
            "candidate",
            400,
            &constant_rows(3, 0.2, 0.4, 0.8, 0.6),
            3,
        );
        let gate = decide_candidate(&specs, &initial, &gate_policy(&specs, 3)).unwrap();
        let policy = canary_policy(&specs);
        let state = CanaryState::start(&gate, &policy).unwrap();
        let subjects = (0..100)
            .map(|index| CanarySubjectId::parse(format!("subject-{index}")).unwrap())
            .collect::<Vec<_>>();
        let stage_zero = subjects
            .iter()
            .map(|subject| state.subject_is_selected(subject, &policy).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            stage_zero,
            subjects
                .iter()
                .map(|subject| state.subject_is_selected(subject, &policy).unwrap())
                .collect::<Vec<_>>()
        );
        let stage_report = report_with_rows(
            &specs,
            "base",
            "candidate",
            410,
            &constant_rows(3, 0.2, 0.5, 0.8, 0.5),
            3,
        );
        let state = state
            .evaluate_stage(&specs, &stage_report, &policy)
            .unwrap();
        assert_eq!(
            state.last_disposition(),
            Some(CanaryStageDisposition::AdvanceStage)
        );
        for (subject, was_selected) in subjects.iter().zip(stage_zero) {
            if was_selected {
                assert!(state.subject_is_selected(subject, &policy).unwrap());
            }
        }
        assert!(!state.authorizes_promotion());
    }

    #[test]
    fn canary_requires_fresh_nonoverlapping_evidence_after_the_gate_and_each_stage() {
        let specs = default_specs();
        let gate_report = report_with_rows(
            &specs,
            "base",
            "candidate",
            500,
            &constant_rows(3, 0.2, 0.4, 0.8, 0.6),
            3,
        );
        let gate = decide_candidate(&specs, &gate_report, &gate_policy(&specs, 3)).unwrap();
        let policy = canary_policy(&specs);
        let state = CanaryState::start(&gate, &policy).unwrap();
        assert!(state
            .clone()
            .evaluate_stage(&specs, &gate_report, &policy)
            .is_err());

        let first_stage = report_with_rows(
            &specs,
            "base",
            "candidate",
            510,
            &constant_rows(3, 0.2, 0.5, 0.8, 0.5),
            3,
        );
        let advanced = state.evaluate_stage(&specs, &first_stage, &policy).unwrap();
        let reused_stage_evidence = report_with_rows(
            &specs,
            "base",
            "candidate",
            510,
            &constant_rows(4, 0.2, 0.6, 0.8, 0.4),
            3,
        );
        assert!(advanced
            .evaluate_stage(&specs, &reused_stage_evidence, &policy)
            .is_err());
    }
}
