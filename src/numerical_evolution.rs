//! End-to-end numerical candidate evolution for TIDE-X.
//!
//! This module connects the bounded least-squares portfolio, paired holdout
//! evaluation, PETFC trajectory governance, and advisory procedural memory.
//! It is deliberately narrower than capability acquisition: a successful
//! cycle proves only that one numerical candidate passed this sealed numerical
//! contract. It never claims semantic equivalence, residency, or promotion.

use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::finite::FiniteF64;
use crate::portfolio_governance::{
    decide_candidate, evaluate_paired_groups, evaluate_petfc, CandidateGateDecision,
    CandidateGateDisposition, CandidateGatePolicy, EvidenceId, GateReason, IndependenceGroupId,
    MetricDirection, MetricId, MetricSpec, ObservationWindow, PairId, PairedEvaluationDigest,
    PairedEvaluationReport, PairedExperimentalUnit, PairedObservation, PetfcAssessment,
    PetfcDisposition, PetfcPolicy, PetfcTrajectory, RobustEvaluationPolicy, VariantId,
};
use crate::procedural_memory::{
    Applicability, AttemptBindings, AttemptLineage, AttemptOutcome, AttemptStatus,
    CapabilityContext, CorrectionRecord, DeclaredBackendImplementationDigest, DriftRecord,
    EvaluatorPolicyDigest, ExecutionSemantics, FailureKind, FailureRecord, FailureSeverity,
    FailureStage, FunctionalDriftObservation, MatrixDimensions, MatrixStructure, NumericPrecision,
    NumericProblemProfile, OutcomeMetrics, ProceduralAttemptDigest, ProceduralLineageId,
    ProceduralMemory, RecordDisposition, SolverArtifactContext, SolverCandidateDigest,
    SolverConfiguration, SolverFamily, SolverParameters, SolverPolicyDigest,
    SolverRepresentationKind, SolverRunFailureRecord, SolverRunSubject, StatisticalProfile,
    VerifiedAttemptDraft, VerifiedCandidateVariantBinding, VerifiedSolverRunFailureDraft,
};
use crate::solver_portfolio::{
    measure_candidate_with_policy, solve_with_portfolio, CandidateEvaluation,
    CandidateRepresentation, ExactSolverProblemDigest, ExternalCandidateProposal,
    LeastSquaresProblem, NumericalBackendRole, NumericalDiagnostics, PortfolioPolicy,
    SolverPortfolioReport, SolverStatus,
};
use std::collections::{BTreeMap, BTreeSet};

const FIT_METRIC: &str = "numerical.normalized_fit";
const WORST_ERROR_METRIC: &str = "numerical.normalized_worst_error";
const GROUP_ID_DOMAIN: &[u8] = b"CEREBRO:TIDEX:NUMERICAL-EVALUATION-GROUP:v1\0";
const EVIDENCE_ID_DOMAIN: &[u8] = b"CEREBRO:TIDEX:NUMERICAL-EVIDENCE:v1\0";
const PAIR_ID_DOMAIN: &[u8] = b"CEREBRO:TIDEX:NUMERICAL-PAIR:v1\0";
const EVALUATOR_POLICY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:NUMERICAL-EVALUATOR-POLICY:v2\0";
const LINEAGE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:NUMERICAL-EVOLUTION-LINEAGE:v1\0";
const EXPERIMENTAL_INPUT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:NUMERICAL-EXPERIMENTAL-INPUT:v1\0";
const RESEARCH_TRIAL_DOMAIN: &[u8] = b"CEREBRO:TIDEX:NUMERICAL-RESEARCH-TRIAL:v1\0";
const MAX_EVALUATION_GROUPS: usize = 65_536;

fn invalid(code: &str) -> BrainError {
    BrainError::Invalid(code.into())
}

fn integrity(code: &str) -> BrainError {
    BrainError::Integrity(code.into())
}

fn fit_metric_id() -> BrainResult<MetricId> {
    MetricId::parse(FIT_METRIC)
}

fn worst_error_metric_id() -> BrainResult<MetricId> {
    MetricId::parse(WORST_ERROR_METRIC)
}

/// Exact metric catalog implemented by this numerical evaluator.
pub fn numerical_metric_specs(
    minimum_normalized_fit: f64,
    maximum_normalized_worst_error: f64,
) -> BrainResult<Vec<MetricSpec>> {
    if !(0.0..=1.0).contains(&minimum_normalized_fit)
        || !maximum_normalized_worst_error.is_finite()
        || maximum_normalized_worst_error < 0.0
    {
        return Err(invalid("numerical_metric_threshold_invalid"));
    }
    Ok(vec![
        MetricSpec::new(
            fit_metric_id()?,
            MetricDirection::Maximize,
            Some(crate::portfolio_governance::HardInvariant::at_least(
                minimum_normalized_fit,
            )?),
        )?,
        MetricSpec::new(
            worst_error_metric_id()?,
            MetricDirection::Minimize,
            Some(crate::portfolio_governance::HardInvariant::at_most(
                maximum_normalized_worst_error,
            )?),
        )?,
    ])
}

/// Explicit aggregate ceilings for one end-to-end evaluation. Per-problem
/// solver limits are necessary but insufficient because a request can contain
/// many individually valid holdout groups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericalEvaluationLimits {
    maximum_groups: usize,
    maximum_total_cases: usize,
    maximum_total_scalar_elements: usize,
}

impl NumericalEvaluationLimits {
    pub fn new(
        maximum_groups: usize,
        maximum_total_cases: usize,
        maximum_total_scalar_elements: usize,
    ) -> BrainResult<Self> {
        if maximum_groups == 0
            || maximum_groups > MAX_EVALUATION_GROUPS
            || maximum_total_cases == 0
            || maximum_total_scalar_elements == 0
        {
            return Err(invalid("numerical_evaluation_limits_invalid"));
        }
        Ok(Self {
            maximum_groups,
            maximum_total_cases,
            maximum_total_scalar_elements,
        })
    }

    pub fn maximum_groups(&self) -> usize {
        self.maximum_groups
    }

    pub fn maximum_total_cases(&self) -> usize {
        self.maximum_total_cases
    }

    pub fn maximum_total_scalar_elements(&self) -> usize {
        self.maximum_total_scalar_elements
    }

    fn validate_input(&self, input: &NumericalEvolutionInput) -> BrainResult<()> {
        if input.evaluation_groups.len() > self.maximum_groups {
            return Err(invalid("numerical_evaluation_group_budget_exceeded"));
        }
        let mut problems = std::iter::once(&input.training_problem)
            .chain(input.evaluation_groups.iter().map(|group| &group.problem));
        let (total_cases, total_elements) =
            problems.try_fold((0_usize, 0_usize), |(cases, elements), problem| {
                let next_cases = cases
                    .checked_add(problem.case_count())
                    .ok_or_else(|| invalid("numerical_evaluation_case_budget_overflow"))?;
                let row_width = problem
                    .input_dimension()
                    .checked_add(problem.output_dimension())
                    .ok_or_else(|| invalid("numerical_evaluation_element_budget_overflow"))?;
                let problem_elements = problem
                    .case_count()
                    .checked_mul(row_width)
                    .ok_or_else(|| invalid("numerical_evaluation_element_budget_overflow"))?;
                let next_elements = elements
                    .checked_add(problem_elements)
                    .ok_or_else(|| invalid("numerical_evaluation_element_budget_overflow"))?;
                Ok::<(usize, usize), BrainError>((next_cases, next_elements))
            })?;
        if total_cases > self.maximum_total_cases
            || total_elements > self.maximum_total_scalar_elements
        {
            return Err(invalid("numerical_evaluation_aggregate_budget_exceeded"));
        }
        Ok(())
    }
}

/// A fully precommitted policy set. All components are constructed before any
/// candidate or holdout result is observed.
#[derive(Debug, Clone)]
pub struct NumericalEvolutionPolicy {
    solver: PortfolioPolicy,
    metric_specs: Vec<MetricSpec>,
    robust_evaluation: RobustEvaluationPolicy,
    candidate_gate: CandidateGatePolicy,
    petfc: PetfcPolicy,
    evaluation_limits: NumericalEvaluationLimits,
    target_scale_floor: FiniteF64,
}

impl NumericalEvolutionPolicy {
    pub fn new(
        solver: PortfolioPolicy,
        metric_specs: Vec<MetricSpec>,
        robust_evaluation: RobustEvaluationPolicy,
        candidate_gate: CandidateGatePolicy,
        petfc: PetfcPolicy,
        evaluation_limits: NumericalEvaluationLimits,
        target_scale_floor: f64,
    ) -> BrainResult<Self> {
        let target_scale_floor = FiniteF64::new(target_scale_floor)?;
        let fit = fit_metric_id()?;
        let worst = worst_error_metric_id()?;
        let maximum_policy_observations = evaluation_limits
            .maximum_groups
            .checked_mul(metric_specs.len())
            .ok_or_else(|| invalid("numerical_evolution_policy_observation_overflow"))?;
        if target_scale_floor.get() <= 0.0
            || metric_specs.len() != 2
            || metric_specs
                .iter()
                .map(MetricSpec::metric_id)
                .collect::<BTreeSet<_>>()
                != BTreeSet::from([&fit, &worst])
            || metric_specs.iter().any(|spec| {
                (spec.metric_id() == &fit && spec.direction() != MetricDirection::Maximize)
                    || (spec.metric_id() == &worst && spec.direction() != MetricDirection::Minimize)
                    || spec.hard_invariant().is_none()
            })
            || petfc.quality_metric_id() != &fit
            || evaluation_limits.maximum_groups < robust_evaluation.minimum_independent_groups()
            || maximum_policy_observations > robust_evaluation.maximum_observations()
        {
            return Err(invalid("numerical_evolution_policy_invalid"));
        }
        solver.digest()?;
        Ok(Self {
            solver,
            metric_specs,
            robust_evaluation,
            candidate_gate,
            petfc,
            evaluation_limits,
            target_scale_floor,
        })
    }

    pub fn solver(&self) -> &PortfolioPolicy {
        &self.solver
    }

    pub fn evaluation_limits(&self) -> &NumericalEvaluationLimits {
        &self.evaluation_limits
    }
}

/// One exact holdout problem. Its group identity is derived from all matrix
/// bits. Construction establishes byte identity, not causal or statistical
/// independence; [`NumericalEvolutionInput`] additionally enforces exact
/// input-unit disjointness between training and holdout groups. In the absence
/// of an authenticated sample-identity authority, repeated/noisy observations
/// with the same input cannot honestly be claimed as independent units and are
/// rejected even if their targets differ.
#[derive(Debug, Clone)]
pub struct NumericalEvaluationGroup {
    problem: LeastSquaresProblem,
    exact_digest: ExactSolverProblemDigest,
}

impl NumericalEvaluationGroup {
    pub fn new(problem: LeastSquaresProblem) -> BrainResult<Self> {
        let exact_digest = problem.digest()?;
        Ok(Self {
            problem,
            exact_digest,
        })
    }

    pub fn exact_digest(&self) -> &ExactSolverProblemDigest {
        &self.exact_digest
    }
}

#[derive(Debug, Clone)]
pub struct NumericalEvolutionInput {
    training_problem: LeastSquaresProblem,
    baseline: CandidateRepresentation,
    evaluation_groups: Vec<NumericalEvaluationGroup>,
    observation_window: ObservationWindow,
    revision: u64,
}

impl NumericalEvolutionInput {
    pub fn new(
        training_problem: LeastSquaresProblem,
        baseline: CandidateRepresentation,
        evaluation_groups: Vec<NumericalEvaluationGroup>,
        revision: u64,
    ) -> BrainResult<Self> {
        if revision == 0
            || evaluation_groups.is_empty()
            || evaluation_groups.len() > MAX_EVALUATION_GROUPS
        {
            return Err(invalid("numerical_evolution_input_invalid"));
        }
        let training_digest = training_problem.digest()?;
        baseline.exact_digest()?;
        if baseline.rows() != training_problem.output_dimension()
            || baseline.columns() != training_problem.input_dimension()
        {
            return Err(invalid("numerical_baseline_shape_mismatch"));
        }
        let mut group_digests = BTreeSet::new();
        let training_inputs = experimental_input_digests(&training_problem)?;
        let mut holdout_inputs = BTreeSet::new();
        for group in &evaluation_groups {
            if group.exact_digest == training_digest
                || group.problem.input_dimension() != training_problem.input_dimension()
                || group.problem.output_dimension() != training_problem.output_dimension()
                || !group_digests.insert(group.exact_digest.clone())
            {
                return Err(invalid("numerical_evaluation_group_invalid"));
            }
            let group_inputs = experimental_input_digests(&group.problem)?;
            if !training_inputs.is_disjoint(&group_inputs)
                || group_inputs
                    .iter()
                    .any(|unit| !holdout_inputs.insert(unit.clone()))
            {
                return Err(invalid("numerical_evaluation_unit_overlap"));
            }
        }
        let observation_window = logical_observation_window(revision)?;
        Ok(Self {
            training_problem,
            baseline,
            evaluation_groups,
            observation_window,
            revision,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericalEvolutionDisposition {
    SolverRejected,
    SolverBoundedUnknown,
    NoNewEvidence,
    CandidateRejected,
    CandidateBoundedUnknown,
    CandidateValidatedForFurtherGates,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumericalFunctionalDriftObservation {
    NoObservedChange,
    BoundedUnknown,
    Recorded {
        record: Box<DriftRecord>,
        disposition: RecordDisposition,
    },
}

#[derive(Debug, Clone)]
pub struct NumericalEvolutionCycle {
    disposition: NumericalEvolutionDisposition,
    solver_report: SolverPortfolioReport,
    candidate: Option<CandidateRepresentation>,
    paired_report: Option<PairedEvaluationReport>,
    candidate_gate: Option<CandidateGateDecision>,
    trajectory: Option<PetfcTrajectory>,
    petfc_assessment: Option<PetfcAssessment>,
    procedural_attempt: Option<crate::procedural_memory::SolverAttempt>,
    solver_run_failure: Option<SolverRunFailureRecord>,
}

impl NumericalEvolutionCycle {
    pub fn disposition(&self) -> NumericalEvolutionDisposition {
        self.disposition
    }

    pub fn candidate(&self) -> Option<&CandidateRepresentation> {
        self.candidate.as_ref()
    }

    pub fn paired_report(&self) -> Option<&PairedEvaluationReport> {
        self.paired_report.as_ref()
    }

    pub fn candidate_gate(&self) -> Option<&CandidateGateDecision> {
        self.candidate_gate.as_ref()
    }

    pub fn petfc_assessment(&self) -> Option<&PetfcAssessment> {
        self.petfc_assessment.as_ref()
    }

    pub fn trajectory(&self) -> Option<&PetfcTrajectory> {
        self.trajectory.as_ref()
    }

    pub fn procedural_attempt(&self) -> Option<&crate::procedural_memory::SolverAttempt> {
        self.procedural_attempt.as_ref()
    }

    pub fn solver_run_failure(&self) -> Option<&SolverRunFailureRecord> {
        self.solver_run_failure.as_ref()
    }

    pub fn solver_report(&self) -> &SolverPortfolioReport {
        &self.solver_report
    }

    /// Numerical evolution produces candidates and evidence only.
    pub const fn authorizes_promotion(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LineageKey {
    problem: ExactSolverProblemDigest,
    policy: SolverPolicyDigest,
}

#[derive(Debug, Clone)]
struct LineagePosition {
    lineage_id: ProceduralLineageId,
    ordinal: u64,
    attempt_digest: ProceduralAttemptDigest,
}

struct NumericalAttemptSources<'a> {
    input: &'a NumericalEvolutionInput,
    solver_report: &'a SolverPortfolioReport,
    selected: &'a CandidateEvaluation,
    candidate_exact_digest: &'a crate::solver_portfolio::ExactSolverCandidateDigest,
    candidate_binding: &'a VerifiedCandidateVariantBinding,
    paired_report: &'a PairedEvaluationReport,
    candidate_gate: &'a CandidateGateDecision,
}

/// Stateful reducer for one capability context. A revision is consumed only
/// after either a complete numerical/governance reduction or an authenticated
/// terminal portfolio result. Candidate-free terminal results use the
/// dedicated solver-run record and are never disguised as candidate attempts.
#[derive(Debug)]
pub struct NumericalEvolutionEngine {
    context: CapabilityContext,
    policy: NumericalEvolutionPolicy,
    procedural_memory: ProceduralMemory,
    /// Fixed checkpoint against which every research candidate is evaluated.
    /// It changes only through a future, explicit promotion API; `evolve`
    /// never promotes a candidate implicitly.
    incumbent_id: Option<VariantId>,
    research_trajectory: Option<PetfcTrajectory>,
    lineages: BTreeMap<LineageKey, LineagePosition>,
    known_reports: BTreeSet<PairedEvaluationDigest>,
    known_research_trials: BTreeSet<Sha256Digest>,
    attempt_candidates: BTreeMap<ProceduralAttemptDigest, VerifiedCandidateVariantBinding>,
    last_revision: u64,
}

impl NumericalEvolutionEngine {
    pub fn new(context: CapabilityContext, policy: NumericalEvolutionPolicy) -> BrainResult<Self> {
        Ok(Self {
            context,
            policy,
            procedural_memory: ProceduralMemory::rebuild(Vec::new(), Vec::new())?,
            incumbent_id: None,
            research_trajectory: None,
            lineages: BTreeMap::new(),
            known_reports: BTreeSet::new(),
            known_research_trials: BTreeSet::new(),
            attempt_candidates: BTreeMap::new(),
            last_revision: 0,
        })
    }

    pub fn procedural_memory(&self) -> &ProceduralMemory {
        &self.procedural_memory
    }

    pub fn incumbent_id(&self) -> Option<&VariantId> {
        self.incumbent_id.as_ref()
    }

    pub fn research_trajectory(&self) -> Option<&PetfcTrajectory> {
        self.research_trajectory.as_ref()
    }

    pub fn evolve(
        &mut self,
        input: NumericalEvolutionInput,
        external_proposals: &[ExternalCandidateProposal],
    ) -> BrainResult<NumericalEvolutionCycle> {
        if input.revision <= self.last_revision {
            return Err(invalid("numerical_evolution_revision_not_monotonic"));
        }
        self.policy.evaluation_limits.validate_input(&input)?;
        let baseline_exact_digest = input.baseline.exact_digest()?;
        let baseline_id = variant_from_digest(baseline_exact_digest.as_str())?;
        match &self.incumbent_id {
            Some(incumbent) if incumbent != &baseline_id => {
                return Err(integrity("numerical_baseline_not_incumbent"));
            }
            Some(_) => {}
            None => {
                // The caller explicitly supplies the starting checkpoint. It
                // becomes the fixed incumbent before any solver outcome, so a
                // failed or unknown run cannot later be used to swap it.
                self.incumbent_id = Some(baseline_id.clone());
            }
        }
        let observation_count = input
            .evaluation_groups
            .len()
            .checked_mul(self.policy.metric_specs.len())
            .ok_or_else(|| invalid("numerical_observation_count_overflow"))?;
        if observation_count > self.policy.robust_evaluation.maximum_observations() {
            return Err(invalid("numerical_observation_budget_exceeded"));
        }
        let solver_report = solve_with_portfolio(
            &input.training_problem,
            &self.policy.solver,
            external_proposals,
        )?;
        let Some(selected) = solver_report.selected() else {
            let disposition = match solver_report.status() {
                SolverStatus::Rejected => NumericalEvolutionDisposition::SolverRejected,
                SolverStatus::BoundedUnknown => NumericalEvolutionDisposition::SolverBoundedUnknown,
                SolverStatus::Accepted => {
                    return Err(integrity("numerical_solver_selected_candidate_missing"));
                }
            };
            if let Some(rejected) = solver_report.canonical_observed_rejection()? {
                let evaluation_index = solver_report
                    .evaluations()
                    .iter()
                    .position(|evaluation| std::ptr::eq(evaluation, rejected))
                    .ok_or_else(|| integrity("numerical_rejected_candidate_index_missing"))?;
                let candidate = rejected
                    .candidate()
                    .ok_or_else(|| integrity("numerical_rejected_candidate_missing"))?
                    .clone();
                let attempt = self.derive_rejected_candidate_attempt(
                    &input,
                    &solver_report,
                    rejected,
                    evaluation_index,
                )?;
                let (lineage_key, lineage_position) = self.validate_lineage_update(&attempt)?;
                self.procedural_memory.record_attempt(attempt.clone())?;
                self.lineages.insert(lineage_key, lineage_position);
                self.last_revision = input.revision;
                return Ok(NumericalEvolutionCycle {
                    disposition,
                    solver_report,
                    candidate: Some(candidate),
                    paired_report: None,
                    candidate_gate: None,
                    trajectory: self.research_trajectory.clone(),
                    petfc_assessment: None,
                    procedural_attempt: Some(attempt),
                    solver_run_failure: None,
                });
            }
            if let Some(candidate) = canonical_materialized_candidate(&solver_report)? {
                // A materialized bounded-unknown candidate is neither an
                // observed rejection nor a candidate-free run. Preserve the
                // authenticated portfolio report, consume its revision, and
                // decline to manufacture either kind of procedural evidence.
                self.last_revision = input.revision;
                return Ok(NumericalEvolutionCycle {
                    disposition,
                    solver_report,
                    candidate: Some(candidate),
                    paired_report: None,
                    candidate_gate: None,
                    trajectory: self.research_trajectory.clone(),
                    petfc_assessment: None,
                    procedural_attempt: None,
                    solver_run_failure: None,
                });
            }
            let failure = self.derive_solver_run_failure(&input, &solver_report)?;
            self.procedural_memory
                .record_solver_run_failure(failure.clone())?;
            // The portfolio actually executed and reached a terminal typed
            // result. Consume the revision so the caller cannot replay the
            // same failed run as if it were fresh evidence.
            self.last_revision = input.revision;
            return Ok(NumericalEvolutionCycle {
                disposition,
                solver_report,
                candidate: None,
                paired_report: None,
                candidate_gate: None,
                trajectory: self.research_trajectory.clone(),
                petfc_assessment: None,
                procedural_attempt: None,
                solver_run_failure: Some(failure),
            });
        };
        if selected.status() != SolverStatus::Accepted {
            return Err(integrity(
                "numerical_solver_selected_candidate_not_accepted",
            ));
        }
        let candidate = selected
            .candidate()
            .ok_or_else(|| integrity("numerical_solver_candidate_missing"))?
            .clone();
        let candidate_exact_digest = candidate.exact_digest()?;
        // Variant identity names the exact checkpoint, not its transient role
        // in one comparison. Otherwise yesterday's `candidate.<digest>` could
        // never equal today's `baseline.<digest>` and every valid trajectory
        // would fork at the second step.
        let candidate_id = variant_from_digest(candidate_exact_digest.as_str())?;
        if candidate_exact_digest == baseline_exact_digest {
            // A completed solve that reproduces the exact incumbent yields no
            // candidate evidence. It still consumes the logical revision so
            // it cannot be replayed as fresh work.
            self.last_revision = input.revision;
            return Ok(NumericalEvolutionCycle {
                disposition: NumericalEvolutionDisposition::NoNewEvidence,
                solver_report,
                candidate: Some(candidate),
                paired_report: None,
                candidate_gate: None,
                trajectory: self.research_trajectory.clone(),
                petfc_assessment: None,
                procedural_attempt: None,
                solver_run_failure: None,
            });
        }
        let research_trial_digest = research_trial_digest(
            &input,
            &baseline_exact_digest,
            &candidate_exact_digest,
            &evaluator_policy_digest(&self.policy)?,
        );
        if self.known_research_trials.contains(&research_trial_digest) {
            // Re-running a deterministic candidate against the identical
            // suite and policy is reproducibility, not fresh evidence. The
            // revision is consumed, but no report, trajectory point, or
            // procedural attempt is duplicated.
            self.last_revision = input.revision;
            return Ok(NumericalEvolutionCycle {
                disposition: NumericalEvolutionDisposition::NoNewEvidence,
                solver_report,
                candidate: Some(candidate),
                paired_report: None,
                candidate_gate: None,
                trajectory: self.research_trajectory.clone(),
                petfc_assessment: None,
                procedural_attempt: None,
                solver_run_failure: None,
            });
        }

        let observations =
            self.derive_observations(&input, &baseline_id, &candidate_id, &candidate)?;
        let paired_report = evaluate_paired_groups(
            baseline_id,
            candidate_id,
            &self.policy.metric_specs,
            &observations,
            &self.policy.robust_evaluation,
        )?;
        let candidate_gate = decide_candidate(
            &self.policy.metric_specs,
            &paired_report,
            &self.policy.candidate_gate,
        )?;
        let next_trajectory = match &self.research_trajectory {
            Some(trajectory) => trajectory.clone().append_report(&paired_report)?,
            None => PetfcTrajectory::start(&paired_report, &self.policy.petfc)?,
        };
        let petfc_assessment = evaluate_petfc(
            &self.policy.metric_specs,
            &next_trajectory,
            &self.policy.petfc,
        )?;
        let candidate_binding = VerifiedCandidateVariantBinding::from_candidate(
            paired_report.candidate_id().clone(),
            &candidate,
        )?;
        let attempt = self.derive_attempt(NumericalAttemptSources {
            input: &input,
            solver_report: &solver_report,
            selected,
            candidate_exact_digest: &candidate_exact_digest,
            candidate_binding: &candidate_binding,
            paired_report: &paired_report,
            candidate_gate: &candidate_gate,
        })?;
        let (lineage_key, lineage_position) = self.validate_lineage_update(&attempt)?;
        if self.known_reports.contains(paired_report.digest())
            || self.attempt_candidates.contains_key(attempt.digest())
        {
            return Err(integrity("numerical_evolution_evidence_reuse"));
        }
        self.procedural_memory.record_attempt(attempt.clone())?;
        self.lineages.insert(lineage_key, lineage_position);
        self.known_reports.insert(paired_report.digest().clone());
        self.known_research_trials.insert(research_trial_digest);
        self.attempt_candidates
            .insert(attempt.digest().clone(), candidate_binding);
        self.research_trajectory = Some(next_trajectory.clone());
        self.last_revision = input.revision;

        let disposition = match (
            candidate_gate.disposition(),
            petfc_assessment.disposition(),
            attempt.outcome().status(),
        ) {
            (
                CandidateGateDisposition::AdvanceCandidate,
                PetfcDisposition::CompatibleForNextGate,
                AttemptStatus::Validated,
            ) => NumericalEvolutionDisposition::CandidateValidatedForFurtherGates,
            (CandidateGateDisposition::Reject, _, _)
            | (_, PetfcDisposition::Reject | PetfcDisposition::RollbackRequired, _) => {
                NumericalEvolutionDisposition::CandidateRejected
            }
            _ => NumericalEvolutionDisposition::CandidateBoundedUnknown,
        };
        Ok(NumericalEvolutionCycle {
            disposition,
            solver_report,
            candidate: Some(candidate),
            paired_report: Some(paired_report),
            candidate_gate: Some(candidate_gate),
            trajectory: Some(next_trajectory),
            petfc_assessment: Some(petfc_assessment),
            procedural_attempt: Some(attempt),
            solver_run_failure: None,
        })
    }

    fn derive_solver_run_failure(
        &self,
        input: &NumericalEvolutionInput,
        report: &SolverPortfolioReport,
    ) -> BrainResult<SolverRunFailureRecord> {
        let policy_digest =
            SolverPolicyDigest::bind_exact_digest(self.policy.solver.digest()?.as_digest());
        let applicability =
            applicability_from_report(&input.training_problem, report.diagnostics())?;
        let draft = VerifiedSolverRunFailureDraft::new(
            self.context.clone(),
            &input.training_problem,
            policy_digest,
            applicability,
            SolverRunSubject::Portfolio,
        )?;
        SolverRunFailureRecord::seal(draft, input.revision, None, report)
    }

    /// Record only a non-causal functional change proven by two reports that
    /// this engine actually reduced. Detection and invalidation revisions are
    /// recovered from procedural memory; the caller cannot choose them or a
    /// causal drift label. This deterministic adapter suppresses exact replay,
    /// so temporal drift remains reserved for future instrumented frontends
    /// that can supply genuinely new authenticated observations.
    pub fn observe_functional_change(
        &mut self,
        attempt: &crate::procedural_memory::SolverAttempt,
        previous: &PairedEvaluationReport,
        current: &PairedEvaluationReport,
    ) -> BrainResult<NumericalFunctionalDriftObservation> {
        attempt.authenticate()?;
        if !self.known_reports.contains(previous.digest())
            || !self.known_reports.contains(current.digest())
        {
            return Err(integrity("numerical_functional_drift_scope_invalid"));
        }
        let candidate_binding = self
            .attempt_candidates
            .get(attempt.digest())
            .ok_or_else(|| integrity("numerical_functional_drift_attempt_unknown"))?;
        match self.procedural_memory.derive_functional_change(
            candidate_binding,
            previous,
            current,
        )? {
            FunctionalDriftObservation::NoObservedChange => {
                Ok(NumericalFunctionalDriftObservation::NoObservedChange)
            }
            FunctionalDriftObservation::BoundedUnknown => {
                Ok(NumericalFunctionalDriftObservation::BoundedUnknown)
            }
            FunctionalDriftObservation::Detected(record) => {
                let disposition = self.procedural_memory.record_drift((*record).clone())?;
                Ok(NumericalFunctionalDriftObservation::Recorded {
                    record,
                    disposition,
                })
            }
        }
    }

    fn derive_observations(
        &self,
        input: &NumericalEvolutionInput,
        baseline_id: &VariantId,
        candidate_id: &VariantId,
        candidate: &CandidateRepresentation,
    ) -> BrainResult<Vec<PairedObservation>> {
        let mut observations = Vec::with_capacity(
            input
                .evaluation_groups
                .len()
                .checked_mul(2)
                .ok_or_else(|| invalid("numerical_observation_capacity_overflow"))?,
        );
        for group in &input.evaluation_groups {
            let baseline_metrics = measure_candidate_with_policy(
                &input.baseline,
                &group.problem,
                &self.policy.solver,
            )?;
            let candidate_metrics =
                measure_candidate_with_policy(candidate, &group.problem, &self.policy.solver)?;
            let scalar_count = group
                .problem
                .case_count()
                .checked_mul(group.problem.output_dimension())
                .ok_or_else(|| invalid("numerical_target_scalar_count_overflow"))?;
            let target_rms = baseline_metrics.target_norm() / (scalar_count as f64).sqrt();
            let scale = target_rms.max(self.policy.target_scale_floor.get());
            let baseline_fit = 1.0 / (1.0 + baseline_metrics.root_mean_square_residual() / scale);
            let candidate_fit = 1.0 / (1.0 + candidate_metrics.root_mean_square_residual() / scale);
            let baseline_worst = baseline_metrics.maximum_absolute_residual() / scale;
            let candidate_worst = candidate_metrics.maximum_absolute_residual() / scale;
            if [baseline_fit, candidate_fit, baseline_worst, candidate_worst]
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(BrainError::Numerical(
                    "numerical_evaluation_metric_nonfinite".into(),
                ));
            }
            let group_token = semantic_token(GROUP_ID_DOMAIN, &[group.exact_digest.as_str()]);
            let evidence_token = semantic_token(
                EVIDENCE_ID_DOMAIN,
                &[
                    group.exact_digest.as_str(),
                    baseline_id.as_str(),
                    candidate_id.as_str(),
                    &input.observation_window.start_tick().to_string(),
                    &input.observation_window.end_tick().to_string(),
                ],
            );
            let pair_token = semantic_token(PAIR_ID_DOMAIN, &[evidence_token.as_str()]);
            let unit = PairedExperimentalUnit::new(
                IndependenceGroupId::parse(format!("group.{}", group_token.as_str()))?,
                PairId::parse(format!("pair.{}", pair_token.as_str()))?,
                EvidenceId::parse(format!("evidence.{}", evidence_token.as_str()))?,
                input.observation_window.clone(),
            );
            observations.push(PairedObservation::new(
                fit_metric_id()?,
                unit.clone(),
                baseline_fit,
                candidate_fit,
            )?);
            observations.push(PairedObservation::new(
                worst_error_metric_id()?,
                unit,
                baseline_worst,
                candidate_worst,
            )?);
        }
        Ok(observations)
    }

    fn derive_attempt(
        &self,
        sources: NumericalAttemptSources<'_>,
    ) -> BrainResult<crate::procedural_memory::SolverAttempt> {
        let NumericalAttemptSources {
            input,
            solver_report,
            selected,
            candidate_exact_digest,
            candidate_binding,
            paired_report: report,
            candidate_gate: gate,
        } = sources;
        let research_validation =
            crate::procedural_memory::ResearchValidation::from_verified_paired_report(
                report,
                gate,
                candidate_binding,
            )?;
        // The portfolio exposes convergence diagnostics, but not an
        // instrumented operation counter. Dimensions and SVD sweeps permit a
        // cost estimate, not exact metering, so procedural memory must retain
        // this measurement as unknown instead of storing a fabricated count.
        let compute_units = None;
        let metrics = OutcomeMetrics::from_verified_numerical_sources(
            selected,
            candidate_binding,
            report,
            gate,
            compute_units,
        )?;
        let metrics_complete = metrics.supports_validated_status();
        // External proposals have independently checked numerical behavior,
        // but their producing implementation is not yet part of the sealed
        // procedural configuration. Do not learn a positive algorithmic rule
        // from an unbound implementation name.
        let provenance_complete =
            selected.backend().role() != NumericalBackendRole::ExternalProposal;
        let (status, failure, correction) =
            attempt_disposition(gate, metrics_complete && provenance_complete);
        let outcome = AttemptOutcome::new(status, metrics, failure, correction)?;
        let applied_correction = None;
        let draft = self.derive_attempt_draft(
            input,
            solver_report,
            selected,
            candidate_exact_digest,
            applied_correction,
        )?;
        crate::procedural_memory::SolverAttempt::seal(draft, Some(research_validation), outcome)
    }

    fn derive_rejected_candidate_attempt(
        &self,
        input: &NumericalEvolutionInput,
        solver_report: &SolverPortfolioReport,
        rejected: &CandidateEvaluation,
        evaluation_index: usize,
    ) -> BrainResult<crate::procedural_memory::SolverAttempt> {
        if rejected.status() != SolverStatus::Rejected {
            return Err(invalid("numerical_rejected_candidate_not_attributable"));
        }
        let candidate_digest = rejected
            .candidate()
            .ok_or_else(|| integrity("numerical_rejected_candidate_missing"))?
            .exact_digest()?;
        let draft =
            self.derive_attempt_draft(input, solver_report, rejected, &candidate_digest, None)?;
        crate::procedural_memory::SolverAttempt::seal_rejected_candidate(
            draft,
            solver_report,
            evaluation_index,
        )
    }

    fn derive_attempt_draft(
        &self,
        input: &NumericalEvolutionInput,
        solver_report: &SolverPortfolioReport,
        evaluation: &CandidateEvaluation,
        candidate_exact_digest: &crate::solver_portfolio::ExactSolverCandidateDigest,
        applied_correction: Option<crate::procedural_memory::CorrectionKind>,
    ) -> BrainResult<VerifiedAttemptDraft> {
        let problem_digest = input.training_problem.digest()?;
        let solver_policy_digest =
            SolverPolicyDigest::bind_exact_digest(self.policy.solver.digest()?.as_digest());
        let candidate_digest =
            SolverCandidateDigest::bind_exact_digest(candidate_exact_digest.as_digest());
        let bindings = AttemptBindings::new(
            self.context.clone(),
            SolverArtifactContext::new(
                problem_digest.clone(),
                solver_policy_digest.clone(),
                candidate_digest,
            )?,
        )?;
        let applicability =
            applicability_from_report(&input.training_problem, solver_report.diagnostics())?;
        let configuration = configuration_from_selected(evaluation, &self.policy.solver)?;
        let key = LineageKey {
            problem: problem_digest,
            policy: solver_policy_digest,
        };
        let lineage = match self.lineages.get(&key) {
            Some(position) => AttemptLineage::child(
                position.lineage_id.clone(),
                position
                    .ordinal
                    .checked_add(1)
                    .ok_or_else(|| invalid("numerical_lineage_ordinal_overflow"))?,
                position.attempt_digest.clone(),
            )?,
            None => AttemptLineage::root(self.root_lineage_id(&key)?),
        };
        VerifiedAttemptDraft::new(
            bindings,
            applicability,
            configuration,
            lineage,
            input.revision,
            applied_correction,
        )
    }

    fn root_lineage_id(&self, key: &LineageKey) -> BrainResult<ProceduralLineageId> {
        let context = serde_json::to_vec(&self.context)?;
        let mut frame = Vec::with_capacity(
            LINEAGE_DOMAIN.len()
                + context.len()
                + key.problem.as_str().len()
                + key.policy.as_str().len()
                + 3 * std::mem::size_of::<u64>(),
        );
        for component in [
            LINEAGE_DOMAIN,
            context.as_slice(),
            key.problem.as_str().as_bytes(),
            key.policy.as_str().as_bytes(),
        ] {
            frame.extend_from_slice(&(component.len() as u64).to_be_bytes());
            frame.extend_from_slice(component);
        }
        Ok(ProceduralLineageId::of_exact_bytes(&frame))
    }

    fn validate_lineage_update(
        &self,
        attempt: &crate::procedural_memory::SolverAttempt,
    ) -> BrainResult<(LineageKey, LineagePosition)> {
        let problem = attempt.bindings().problem_digest().clone();
        let policy =
            SolverPolicyDigest::bind_exact_digest(self.policy.solver.digest()?.as_digest());
        let key = LineageKey { problem, policy };
        let expected_ordinal = match self.lineages.get(&key) {
            Some(position) => position
                .ordinal
                .checked_add(1)
                .ok_or_else(|| invalid("numerical_lineage_ordinal_overflow"))?,
            None => 0,
        };
        let expected_lineage_id = match self.lineages.get(&key) {
            Some(position) => position.lineage_id.clone(),
            None => self.root_lineage_id(&key)?,
        };
        if attempt.lineage_ordinal() != expected_ordinal
            || attempt.lineage_id() != &expected_lineage_id
        {
            return Err(integrity("numerical_lineage_state_mismatch"));
        }
        Ok((
            key,
            LineagePosition {
                lineage_id: attempt.lineage_id().clone(),
                ordinal: attempt.lineage_ordinal(),
                attempt_digest: attempt.digest().clone(),
            },
        ))
    }
}

fn variant_from_digest(digest: &str) -> BrainResult<VariantId> {
    VariantId::parse(format!("numerical.checkpoint.{digest}"))
}

fn canonical_materialized_candidate(
    report: &SolverPortfolioReport,
) -> BrainResult<Option<CandidateRepresentation>> {
    report.exact_digest()?;
    let mut candidates = report
        .evaluations()
        .iter()
        .filter_map(CandidateEvaluation::candidate)
        .map(|candidate| Ok((candidate.exact_digest()?, candidate.clone())))
        .collect::<BrainResult<Vec<_>>>()?;
    candidates.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(candidates
        .into_iter()
        .next()
        .map(|(_, candidate)| candidate))
}

/// Derive a sealed logical interval from the engine revision. Wall-clock or
/// caller-selected ticks would let a caller manufacture apparent freshness.
fn logical_observation_window(revision: u64) -> BrainResult<ObservationWindow> {
    let start_tick = revision
        .checked_mul(2)
        .ok_or_else(|| invalid("numerical_observation_window_overflow"))?;
    let end_tick = start_tick
        .checked_add(1)
        .ok_or_else(|| invalid("numerical_observation_window_overflow"))?;
    ObservationWindow::new(start_tick, end_tick)
}

fn research_trial_digest(
    input: &NumericalEvolutionInput,
    baseline: &crate::solver_portfolio::ExactSolverCandidateDigest,
    candidate: &crate::solver_portfolio::ExactSolverCandidateDigest,
    evaluator_policy: &EvaluatorPolicyDigest,
) -> Sha256Digest {
    let mut components = vec![
        baseline.as_str().to_owned(),
        candidate.as_str().to_owned(),
        evaluator_policy.as_str().to_owned(),
    ];
    let mut suite = input
        .evaluation_groups
        .iter()
        .map(|group| group.exact_digest.as_str().to_owned())
        .collect::<Vec<_>>();
    suite.sort();
    components.extend(suite);
    let component_refs = components.iter().map(String::as_str).collect::<Vec<_>>();
    semantic_token(RESEARCH_TRIAL_DOMAIN, &component_refs)
}

fn semantic_token(domain: &[u8], components: &[&str]) -> Sha256Digest {
    let mut frame = Vec::new();
    for component in components {
        frame.extend_from_slice(&(component.len() as u64).to_be_bytes());
        frame.extend_from_slice(component.as_bytes());
    }
    Sha256Digest::digest_domain(domain, &frame)
}

/// Conservative structural identity for an experimental input. Targets are
/// deliberately excluded: changing a label by one bit must not make a reused
/// input appear independent. A future adapter may admit repeated measurements
/// only by supplying an authenticated sample/provenance identity.
fn experimental_input_digests(
    problem: &LeastSquaresProblem,
) -> BrainResult<BTreeSet<Sha256Digest>> {
    let input_dimension = u64::try_from(problem.input_dimension())
        .map_err(|_| invalid("numerical_unit_input_dimension_overflow"))?;
    let mut units = BTreeSet::new();
    for case in 0..problem.case_count() {
        let mut frame = Vec::new();
        frame.extend_from_slice(&input_dimension.to_be_bytes());
        frame.push(b'X');
        for value in problem.inputs().row(case) {
            // Signed zero is numerically indistinguishable in this solver and
            // therefore may not be used to relabel a reused unit.
            let bits = if *value == 0.0 { 0 } else { value.to_bits() };
            frame.extend_from_slice(&bits.to_be_bytes());
        }
        let digest = Sha256Digest::digest_domain(EXPERIMENTAL_INPUT_DOMAIN, &frame);
        if !units.insert(digest) {
            return Err(invalid("numerical_duplicate_experimental_input"));
        }
    }
    if units.is_empty() {
        return Err(invalid("numerical_evaluation_unit_empty"));
    }
    Ok(units)
}

fn applicability_from_report(
    problem: &LeastSquaresProblem,
    diagnostics: Option<&NumericalDiagnostics>,
) -> BrainResult<Applicability> {
    let total_inputs = problem.inputs().as_slice().len();
    let zero_count = problem
        .inputs()
        .as_slice()
        .iter()
        .filter(|value| **value == 0.0)
        .count();
    let sparsity = if total_inputs == 0 {
        None
    } else {
        Some(zero_count as f64 / total_inputs as f64)
    };
    Applicability::new(
        MatrixDimensions::new(
            u64::try_from(problem.case_count())
                .map_err(|_| invalid("numerical_case_count_overflow"))?,
            u64::try_from(problem.input_dimension())
                .map_err(|_| invalid("numerical_input_dimension_overflow"))?,
            diagnostics
                .map(|diagnostics| {
                    u64::try_from(diagnostics.numerical_rank())
                        .map_err(|_| invalid("numerical_rank_overflow"))
                })
                .transpose()?,
            0,
        )?,
        StatisticalProfile::new(
            diagnostics
                .and_then(NumericalDiagnostics::finite_singular_condition_number)
                .map(|condition| condition.max(1.0).log10()),
            None,
            sparsity,
            None,
        )?,
        NumericProblemProfile::new(
            NumericPrecision::Float64,
            MatrixStructure::Dense,
            ExecutionSemantics::PureTensor,
        ),
    )
}

fn configuration_from_selected(
    selected: &CandidateEvaluation,
    policy: &PortfolioPolicy,
) -> BrainResult<SolverConfiguration> {
    // Never combine the relative and absolute thresholds with `max`: they use
    // different scales. Prefer the relative contract when configured; an
    // absolute-only policy retains its sole nonzero threshold. The exact pair
    // remains committed by `SolverPolicyDigest`.
    let tolerance = if policy.relative_residual_tolerance() > 0.0 {
        policy.relative_residual_tolerance()
    } else {
        policy.absolute_residual_tolerance()
    };
    match selected.backend().role() {
        NumericalBackendRole::BuiltInCholeskyRidge => SolverConfiguration::new(
            SolverFamily::CholeskyRidgeLowRank,
            SolverParameters::LowRank {
                rank: u32::try_from(
                    selected
                        .constructed_rank()
                        .ok_or_else(|| integrity("numerical_cholesky_rank_missing"))?,
                )
                .map_err(|_| invalid("numerical_cholesky_rank_overflow"))?,
            },
            Some(policy.cholesky_damping()),
            Some(tolerance),
            None,
        ),
        NumericalBackendRole::BuiltInDirectJacobiSvd => SolverConfiguration::new(
            SolverFamily::DirectJacobiSvd,
            SolverParameters::Direct,
            None,
            Some(tolerance),
            Some(
                u32::try_from(policy.max_svd_sweeps())
                    .map_err(|_| invalid("numerical_svd_sweep_overflow"))?,
            ),
        ),
        NumericalBackendRole::ExternalProposal => SolverConfiguration::new(
            SolverFamily::ExternalProposal,
            SolverParameters::External {
                representation: match selected
                    .candidate()
                    .ok_or_else(|| integrity("numerical_external_candidate_missing"))?
                {
                    CandidateRepresentation::LowRank { .. } => SolverRepresentationKind::LowRank,
                    CandidateRepresentation::Dense { .. } => SolverRepresentationKind::Dense,
                    CandidateRepresentation::Sparse { .. } => SolverRepresentationKind::Sparse,
                    CandidateRepresentation::Block { .. } => SolverRepresentationKind::Block,
                },
                declared_implementation_digest: DeclaredBackendImplementationDigest::of_exact_bytes(
                    selected.backend().implementation().as_bytes(),
                ),
            },
            None,
            None,
            None,
        ),
    }
}

fn evaluator_policy_digest(
    policy: &NumericalEvolutionPolicy,
) -> BrainResult<EvaluatorPolicyDigest> {
    let solver = policy.solver.digest()?;
    let maximum_groups = policy.evaluation_limits.maximum_groups.to_string();
    let maximum_total_cases = policy.evaluation_limits.maximum_total_cases.to_string();
    let maximum_total_scalar_elements = policy
        .evaluation_limits
        .maximum_total_scalar_elements
        .to_string();
    let target_scale_floor_bits = policy.target_scale_floor.get().to_bits().to_string();
    let token = semantic_token(
        EVALUATOR_POLICY_DOMAIN,
        &[
            solver.as_str(),
            policy.robust_evaluation.digest().as_str(),
            policy.candidate_gate.digest().as_str(),
            &maximum_groups,
            &maximum_total_cases,
            &maximum_total_scalar_elements,
            &target_scale_floor_bits,
        ],
    );
    Ok(EvaluatorPolicyDigest::bind_exact_digest(&token))
}

fn attempt_disposition(
    gate: &CandidateGateDecision,
    metrics_complete: bool,
) -> (
    AttemptStatus,
    Option<FailureRecord>,
    Option<CorrectionRecord>,
) {
    let hard_invariant = gate
        .reasons()
        .iter()
        .any(|reason| matches!(reason, GateReason::ProvenHardViolation { .. }));
    if hard_invariant {
        return (
            AttemptStatus::Rejected,
            Some(FailureRecord::new(
                FailureKind::InvariantRegression,
                FailureStage::ResearchValidation,
                FailureSeverity::HardInvariant,
            )),
            None,
        );
    }

    match gate.disposition() {
        CandidateGateDisposition::AdvanceCandidate if metrics_complete => {
            (AttemptStatus::Validated, None, None)
        }
        CandidateGateDisposition::AdvanceCandidate => (AttemptStatus::Inconclusive, None, None),
        CandidateGateDisposition::Reject => (
            AttemptStatus::Rejected,
            Some(FailureRecord::new(
                FailureKind::ResidualTooLarge,
                FailureStage::ResearchValidation,
                FailureSeverity::Serious,
            )),
            None,
        ),
        CandidateGateDisposition::BoundedUnknown => (AttemptStatus::Inconclusive, None, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digest::{CapabilityIrDigest, SystemEnvelopeDigest};
    use crate::identity::CapabilityId;
    use crate::portfolio_governance::{
        CandidateGatePolicy, PetfcConservationLimits, PetfcMetricPolicy, PetfcPathLimits,
        PetfcUtilityPolicy,
    };
    use crate::procedural_memory::{BaseArtifactDigest, TargetProfileDigest};
    use crate::solver_portfolio::SparseEntry;

    fn context() -> CapabilityContext {
        CapabilityContext::new(
            SystemEnvelopeDigest::from_computed(Sha256Digest::digest_bytes(b"envelope")),
            CapabilityId::parse("numerical.linear-map:v1").unwrap(),
            CapabilityIrDigest::from_computed(Sha256Digest::digest_bytes(b"ir")),
            BaseArtifactDigest::bind_exact_digest(&Sha256Digest::digest_bytes(b"base")),
            TargetProfileDigest::bind_exact_digest(&Sha256Digest::digest_bytes(b"target")),
        )
        .unwrap()
    }

    fn policy() -> NumericalEvolutionPolicy {
        policy_with_limits(NumericalEvaluationLimits::new(16, 256, 4_096).unwrap())
    }

    fn policy_with_limits(limits: NumericalEvaluationLimits) -> NumericalEvolutionPolicy {
        let specs = numerical_metric_specs(0.45, 2.0).unwrap();
        let robust = RobustEvaluationPolicy::new(3, 3, 100).unwrap();
        let gate = CandidateGatePolicy::new(
            &specs,
            3,
            1.0,
            BTreeMap::from([
                (fit_metric_id().unwrap(), FiniteF64::new(0.0).unwrap()),
                (
                    worst_error_metric_id().unwrap(),
                    FiniteF64::new(0.0).unwrap(),
                ),
            ]),
        )
        .unwrap();
        let petfc = PetfcPolicy::new(
            &specs,
            fit_metric_id().unwrap(),
            vec![
                PetfcMetricPolicy::new(fit_metric_id().unwrap(), 1.0, 0.5).unwrap(),
                PetfcMetricPolicy::new(worst_error_metric_id().unwrap(), 1.0, 1.5).unwrap(),
            ],
            PetfcPathLimits::new(2, 8, 3.0, 4.0, 0.9, 0.0).unwrap(),
            PetfcConservationLimits::new(2.0, 2, 4.0).unwrap(),
            PetfcUtilityPolicy::new(0.0, 0.0, 0.0, -2.0).unwrap(),
        )
        .unwrap();
        NumericalEvolutionPolicy::new(
            PortfolioPolicy::default()
                .with_residual_tolerances(0.6, 1.0e-10)
                .unwrap(),
            specs,
            robust,
            gate,
            petfc,
            limits,
            1.0e-12,
        )
        .unwrap()
    }

    fn groups() -> Vec<NumericalEvaluationGroup> {
        [3.0, 5.0, 7.0]
            .into_iter()
            .map(|offset| {
                NumericalEvaluationGroup::new(
                    LeastSquaresProblem::new(
                        vec![vec![offset], vec![offset + 0.5]],
                        vec![vec![2.0 * offset], vec![2.0 * (offset + 0.5)]],
                    )
                    .unwrap(),
                )
                .unwrap()
            })
            .collect()
    }

    fn two_dimensional_groups() -> Vec<NumericalEvaluationGroup> {
        [3.0, 7.0, 11.0]
            .into_iter()
            .map(|offset| {
                NumericalEvaluationGroup::new(
                    LeastSquaresProblem::new(
                        vec![vec![offset, offset + 1.0], vec![offset + 2.0, offset + 3.0]],
                        vec![
                            vec![2.0 * offset, 2.0 * (offset + 1.0)],
                            vec![2.0 * (offset + 2.0), 2.0 * (offset + 3.0)],
                        ],
                    )
                    .unwrap(),
                )
                .unwrap()
            })
            .collect()
    }

    fn exact_sparse_proposal() -> ExternalCandidateProposal {
        ExternalCandidateProposal::candidate(
            "tests.exact-sparse/v1",
            CandidateRepresentation::Sparse {
                rows: 2,
                columns: 2,
                entries: vec![
                    SparseEntry::new(0, 0, 2.0).unwrap(),
                    SparseEntry::new(1, 1, 2.0).unwrap(),
                ],
            },
        )
        .unwrap()
    }

    #[test]
    fn two_real_revisions_reach_petfc_without_authorizing_promotion() {
        let mut engine = NumericalEvolutionEngine::new(context(), policy()).unwrap();
        let zero = CandidateRepresentation::Dense {
            rows: 1,
            columns: 1,
            weights: vec![0.0],
        };
        let first = engine
            .evolve(
                NumericalEvolutionInput::new(
                    LeastSquaresProblem::new(vec![vec![1.0]], vec![vec![1.5]]).unwrap(),
                    zero.clone(),
                    groups(),
                    1,
                )
                .unwrap(),
                &[],
            )
            .unwrap();
        assert_eq!(
            first.disposition(),
            NumericalEvolutionDisposition::CandidateBoundedUnknown
        );
        let second = engine
            .evolve(
                NumericalEvolutionInput::new(
                    LeastSquaresProblem::new(
                        vec![vec![1.0], vec![2.0]],
                        vec![vec![2.0], vec![4.0]],
                    )
                    .unwrap(),
                    zero.clone(),
                    groups(),
                    2,
                )
                .unwrap(),
                &[],
            )
            .unwrap();
        assert_eq!(
            second.disposition(),
            NumericalEvolutionDisposition::CandidateValidatedForFurtherGates
        );
        assert!(!second.authorizes_promotion());
        assert_eq!(
            engine.incumbent_id(),
            Some(&variant_from_digest(zero.exact_digest().unwrap().as_str()).unwrap())
        );
        assert_eq!(engine.procedural_memory().attempt_count(), 2);
    }

    #[test]
    fn duplicate_or_training_reused_holdout_is_rejected() {
        let training = LeastSquaresProblem::new(vec![vec![1.0]], vec![vec![2.0]]).unwrap();
        let duplicate = NumericalEvaluationGroup::new(training.clone()).unwrap();
        assert!(NumericalEvolutionInput::new(
            training,
            CandidateRepresentation::Dense {
                rows: 1,
                columns: 1,
                weights: vec![0.0],
            },
            vec![duplicate],
            1,
        )
        .is_err());
    }

    #[test]
    fn partial_training_holdout_overlap_is_rejected() {
        let training =
            LeastSquaresProblem::new(vec![vec![1.0], vec![2.0]], vec![vec![2.0], vec![4.0]])
                .unwrap();
        let partially_reused = NumericalEvaluationGroup::new(
            LeastSquaresProblem::new(vec![vec![2.0], vec![3.0]], vec![vec![4.0], vec![6.0]])
                .unwrap(),
        )
        .unwrap();
        let result = NumericalEvolutionInput::new(
            training,
            CandidateRepresentation::Dense {
                rows: 1,
                columns: 1,
                weights: vec![0.0],
            },
            vec![partially_reused],
            1,
        );
        assert!(matches!(
            result,
            Err(BrainError::Invalid(code)) if code == "numerical_evaluation_unit_overlap"
        ));
    }

    #[test]
    fn changing_target_cannot_relabel_a_reused_input_as_independent() {
        let relabeled = NumericalEvaluationGroup::new(
            LeastSquaresProblem::new(vec![vec![2.0], vec![3.0]], vec![vec![-99.0], vec![6.0]])
                .unwrap(),
        )
        .unwrap();
        let result = NumericalEvolutionInput::new(
            LeastSquaresProblem::new(vec![vec![1.0], vec![2.0]], vec![vec![2.0], vec![4.0]])
                .unwrap(),
            CandidateRepresentation::Dense {
                rows: 1,
                columns: 1,
                weights: vec![0.0],
            },
            vec![relabeled],
            1,
        );
        assert!(matches!(
            result,
            Err(BrainError::Invalid(code)) if code == "numerical_evaluation_unit_overlap"
        ));
    }

    #[test]
    fn partial_overlap_between_holdout_groups_is_rejected() {
        let first = NumericalEvaluationGroup::new(
            LeastSquaresProblem::new(vec![vec![3.0], vec![4.0]], vec![vec![6.0], vec![8.0]])
                .unwrap(),
        )
        .unwrap();
        let second = NumericalEvaluationGroup::new(
            LeastSquaresProblem::new(vec![vec![4.0], vec![5.0]], vec![vec![8.0], vec![10.0]])
                .unwrap(),
        )
        .unwrap();
        let result = NumericalEvolutionInput::new(
            LeastSquaresProblem::new(vec![vec![1.0]], vec![vec![2.0]]).unwrap(),
            CandidateRepresentation::Dense {
                rows: 1,
                columns: 1,
                weights: vec![0.0],
            },
            vec![first, second],
            1,
        );
        assert!(matches!(
            result,
            Err(BrainError::Invalid(code)) if code == "numerical_evaluation_unit_overlap"
        ));
    }

    #[test]
    fn signed_zero_cannot_relabel_a_reused_unit() {
        let reused = NumericalEvaluationGroup::new(
            LeastSquaresProblem::new(vec![vec![-0.0]], vec![vec![1.0]]).unwrap(),
        )
        .unwrap();
        let result = NumericalEvolutionInput::new(
            LeastSquaresProblem::new(vec![vec![0.0]], vec![vec![1.0]]).unwrap(),
            CandidateRepresentation::Dense {
                rows: 1,
                columns: 1,
                weights: vec![0.0],
            },
            vec![reused],
            1,
        );
        assert!(matches!(
            result,
            Err(BrainError::Invalid(code)) if code == "numerical_evaluation_unit_overlap"
        ));
    }

    #[test]
    fn duplicate_unit_inside_one_group_is_rejected() {
        let duplicate = NumericalEvaluationGroup::new(
            LeastSquaresProblem::new(vec![vec![3.0], vec![3.0]], vec![vec![6.0], vec![6.5]])
                .unwrap(),
        )
        .unwrap();
        let result = NumericalEvolutionInput::new(
            LeastSquaresProblem::new(vec![vec![1.0]], vec![vec![2.0]]).unwrap(),
            CandidateRepresentation::Dense {
                rows: 1,
                columns: 1,
                weights: vec![0.0],
            },
            vec![duplicate],
            1,
        );
        assert!(matches!(
            result,
            Err(BrainError::Invalid(code)) if code == "numerical_duplicate_experimental_input"
        ));
    }

    #[test]
    fn terminal_solver_result_consumes_its_revision() {
        let mut engine = NumericalEvolutionEngine::new(context(), policy()).unwrap();
        let make_input = || {
            NumericalEvolutionInput::new(
                LeastSquaresProblem::new(vec![vec![0.0]], vec![vec![1.0]]).unwrap(),
                CandidateRepresentation::Dense {
                    rows: 1,
                    columns: 1,
                    weights: vec![0.0],
                },
                groups(),
                1,
            )
            .unwrap()
        };
        let first = engine.evolve(make_input(), &[]).unwrap();
        assert_eq!(
            first.disposition(),
            NumericalEvolutionDisposition::SolverRejected
        );
        assert!(first.candidate().is_some());
        assert!(first.procedural_attempt().is_some());
        assert!(first.solver_run_failure().is_none());
        assert_eq!(engine.procedural_memory().attempt_count(), 1);
        assert_eq!(engine.procedural_memory().solver_run_failure_count(), 0);
        assert!(matches!(
            engine.evolve(make_input(), &[]),
            Err(BrainError::Invalid(code)) if code == "numerical_evolution_revision_not_monotonic"
        ));
    }

    #[test]
    fn pre_materialization_solver_limit_uses_candidate_free_record() {
        let mut engine = NumericalEvolutionEngine::new(context(), policy()).unwrap();
        let training_inputs = (0..65).map(|index| vec![index as f64]).collect();
        let training_targets = (0..65).map(|index| vec![2.0 * index as f64]).collect();
        let cycle = engine
            .evolve(
                NumericalEvolutionInput::new(
                    LeastSquaresProblem::new(training_inputs, training_targets).unwrap(),
                    CandidateRepresentation::Dense {
                        rows: 1,
                        columns: 1,
                        weights: vec![0.0],
                    },
                    [100.0, 110.0, 120.0]
                        .into_iter()
                        .map(|offset| {
                            NumericalEvaluationGroup::new(
                                LeastSquaresProblem::new(
                                    vec![vec![offset], vec![offset + 1.0]],
                                    vec![vec![2.0 * offset], vec![2.0 * (offset + 1.0)]],
                                )
                                .unwrap(),
                            )
                            .unwrap()
                        })
                        .collect(),
                    1,
                )
                .unwrap(),
                &[],
            )
            .unwrap();
        assert_eq!(
            cycle.disposition(),
            NumericalEvolutionDisposition::SolverBoundedUnknown
        );
        assert!(cycle.candidate().is_none());
        assert!(cycle.procedural_attempt().is_none());
        assert!(cycle.solver_run_failure().is_some());
        assert_eq!(engine.procedural_memory().attempt_count(), 0);
        assert_eq!(engine.procedural_memory().solver_run_failure_count(), 1);
    }

    #[test]
    fn aggregate_evaluation_budget_is_enforced_before_solving() {
        let mut engine = NumericalEvolutionEngine::new(
            context(),
            policy_with_limits(NumericalEvaluationLimits::new(3, 256, 4_096).unwrap()),
        )
        .unwrap();
        let mut excessive_groups = groups();
        excessive_groups.push(
            NumericalEvaluationGroup::new(
                LeastSquaresProblem::new(vec![vec![9.0], vec![9.5]], vec![vec![18.0], vec![19.0]])
                    .unwrap(),
            )
            .unwrap(),
        );
        let result = engine.evolve(
            NumericalEvolutionInput::new(
                LeastSquaresProblem::new(vec![vec![1.0]], vec![vec![2.0]]).unwrap(),
                CandidateRepresentation::Dense {
                    rows: 1,
                    columns: 1,
                    weights: vec![0.0],
                },
                excessive_groups,
                1,
            )
            .unwrap(),
            &[],
        );
        assert!(matches!(
            result,
            Err(BrainError::Invalid(code)) if code == "numerical_evaluation_group_budget_exceeded"
        ));
        assert_eq!(engine.procedural_memory().attempt_count(), 0);
        assert_eq!(engine.procedural_memory().solver_run_failure_count(), 0);
    }

    #[test]
    fn rollback_candidate_never_replaces_the_incumbent() {
        let mut engine = NumericalEvolutionEngine::new(context(), policy()).unwrap();
        let incumbent = CandidateRepresentation::Dense {
            rows: 1,
            columns: 1,
            weights: vec![0.0],
        };
        let adverse_groups = [3.0, 5.0, 7.0]
            .into_iter()
            .map(|offset| {
                NumericalEvaluationGroup::new(
                    LeastSquaresProblem::new(
                        vec![vec![offset], vec![offset + 0.5]],
                        vec![vec![-2.0 * offset], vec![-2.0 * (offset + 0.5)]],
                    )
                    .unwrap(),
                )
                .unwrap()
            })
            .collect();
        let cycle = engine
            .evolve(
                NumericalEvolutionInput::new(
                    LeastSquaresProblem::new(
                        vec![vec![1.0], vec![2.0]],
                        vec![vec![2.0], vec![4.0]],
                    )
                    .unwrap(),
                    incumbent.clone(),
                    adverse_groups,
                    1,
                )
                .unwrap(),
                &[],
            )
            .unwrap();
        assert_eq!(
            cycle.disposition(),
            NumericalEvolutionDisposition::CandidateRejected
        );
        assert_eq!(
            cycle.petfc_assessment().unwrap().disposition(),
            PetfcDisposition::RollbackRequired
        );
        let incumbent_id = variant_from_digest(incumbent.exact_digest().unwrap().as_str()).unwrap();
        assert_eq!(engine.incumbent_id(), Some(&incumbent_id));
        assert_ne!(
            cycle.candidate().unwrap().exact_digest().unwrap(),
            incumbent.exact_digest().unwrap()
        );
    }

    #[test]
    fn same_problem_revisions_advance_one_consistent_lineage() {
        let mut engine = NumericalEvolutionEngine::new(context(), policy()).unwrap();
        let incumbent = CandidateRepresentation::Dense {
            rows: 2,
            columns: 2,
            weights: vec![0.0; 4],
        };
        let training = || {
            LeastSquaresProblem::new(
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            )
            .unwrap()
        };
        engine
            .evolve(
                NumericalEvolutionInput::new(
                    training(),
                    incumbent.clone(),
                    two_dimensional_groups(),
                    1,
                )
                .unwrap(),
                &[],
            )
            .unwrap();
        let sparse = exact_sparse_proposal();
        engine
            .evolve(
                NumericalEvolutionInput::new(training(), incumbent, two_dimensional_groups(), 2)
                    .unwrap(),
                &[sparse],
            )
            .unwrap();
        assert_eq!(engine.procedural_memory().attempt_count(), 2);
    }

    #[test]
    fn exact_reproduction_is_no_new_evidence_and_cannot_be_replayed() {
        let mut engine = NumericalEvolutionEngine::new(context(), policy()).unwrap();
        let incumbent = CandidateRepresentation::Dense {
            rows: 1,
            columns: 1,
            weights: vec![2.0],
        };
        let make_input = || {
            NumericalEvolutionInput::new(
                LeastSquaresProblem::new(vec![vec![1.0], vec![2.0]], vec![vec![2.0], vec![4.0]])
                    .unwrap(),
                incumbent.clone(),
                groups(),
                1,
            )
            .unwrap()
        };
        let cycle = engine.evolve(make_input(), &[]).unwrap();
        assert_eq!(
            cycle.disposition(),
            NumericalEvolutionDisposition::NoNewEvidence
        );
        assert!(cycle.procedural_attempt().is_none());
        assert_eq!(engine.procedural_memory().attempt_count(), 0);
        assert!(matches!(
            engine.evolve(make_input(), &[]),
            Err(BrainError::Invalid(code)) if code == "numerical_evolution_revision_not_monotonic"
        ));
    }

    #[test]
    fn deterministic_repeat_is_not_fresh_evidence_or_functional_drift() {
        let mut engine = NumericalEvolutionEngine::new(context(), policy()).unwrap();
        let zero = CandidateRepresentation::Dense {
            rows: 1,
            columns: 1,
            weights: vec![0.0],
        };
        let first = engine
            .evolve(
                NumericalEvolutionInput::new(
                    LeastSquaresProblem::new(vec![vec![1.0]], vec![vec![1.5]]).unwrap(),
                    zero.clone(),
                    groups(),
                    1,
                )
                .unwrap(),
                &[],
            )
            .unwrap();
        let second = engine
            .evolve(
                NumericalEvolutionInput::new(
                    LeastSquaresProblem::new(
                        vec![vec![1.0], vec![2.0]],
                        vec![vec![2.0], vec![4.0]],
                    )
                    .unwrap(),
                    zero.clone(),
                    groups(),
                    2,
                )
                .unwrap(),
                &[],
            )
            .unwrap();
        assert!(first.paired_report().is_some());
        assert!(first.procedural_attempt().is_some());
        assert!(second.paired_report().is_some());
        assert!(second.procedural_attempt().is_some());
        let third = engine
            .evolve(
                NumericalEvolutionInput::new(
                    LeastSquaresProblem::new(vec![vec![1.0]], vec![vec![1.5]]).unwrap(),
                    zero,
                    groups(),
                    3,
                )
                .unwrap(),
                &[],
            )
            .unwrap();
        assert_eq!(
            third.disposition(),
            NumericalEvolutionDisposition::NoNewEvidence
        );
        assert!(third.paired_report().is_none());
        assert!(third.procedural_attempt().is_none());
        assert_eq!(engine.procedural_memory().attempt_count(), 2);
        assert_eq!(engine.procedural_memory().drift_count(), 0);
    }

    #[test]
    fn trajectory_rejects_changed_holdout_design() {
        let mut engine = NumericalEvolutionEngine::new(context(), policy()).unwrap();
        let incumbent = CandidateRepresentation::Dense {
            rows: 1,
            columns: 1,
            weights: vec![0.0],
        };
        engine
            .evolve(
                NumericalEvolutionInput::new(
                    LeastSquaresProblem::new(vec![vec![1.0]], vec![vec![1.5]]).unwrap(),
                    incumbent.clone(),
                    groups(),
                    1,
                )
                .unwrap(),
                &[],
            )
            .unwrap();
        let mut changed = groups();
        changed.pop();
        changed.push(
            NumericalEvaluationGroup::new(
                LeastSquaresProblem::new(vec![vec![9.0]], vec![vec![18.0]]).unwrap(),
            )
            .unwrap(),
        );
        let result = engine.evolve(
            NumericalEvolutionInput::new(
                LeastSquaresProblem::new(vec![vec![1.0], vec![2.0]], vec![vec![2.0], vec![4.0]])
                    .unwrap(),
                incumbent,
                changed,
                2,
            )
            .unwrap(),
            &[],
        );
        assert!(matches!(
            result,
            Err(BrainError::Integrity(code)) if code == "petfc_trajectory_report_binding_invalid"
        ));
    }
}
