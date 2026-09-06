//! Universal, advisory-only procedural experience memory.
//!
//! This module remembers authenticated solver attempts without introducing a
//! second authority or persistence layer.  [`ProceduralMemory`] is a bounded,
//! deterministic reducer over records supplied by the caller.  Its indexes are
//! disposable projections and can always be rebuilt from the records.
//!
//! A retrieved [`ProceduralAdvice`] can prioritize or discourage another
//! experiment.  It can never authorize residency, promotion, mutation, or
//! execution.

use crate::digest::{
    CapabilityIrDigest, EvaluationReceiptDigest, Sha256Digest, SystemEnvelopeDigest,
};
use crate::error::{BrainError, BrainResult};
use crate::finite::FiniteF64;
use crate::identity::CapabilityId;
use crate::portfolio_governance::{
    CandidateGateDecision, CandidateGateDisposition, PairedEvaluationReport, VariantId,
};
use crate::solver_portfolio::{
    CandidateEvaluation, CandidateRepresentation, ExactSolverProblemDigest, LeastSquaresProblem,
    NumericalBackendRole, SolverStatus,
};
use crate::solver_portfolio::{
    CholeskyGate, EvaluationReason, SolverPortfolioReport, SolverPortfolioReportDigest,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};

const ATTEMPT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:PROCEDURAL-ATTEMPT:v4\0";
const APPLICABILITY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:SOLVER-APPLICABILITY:v2\0";
const CONFIGURATION_DOMAIN: &[u8] = b"CEREBRO:TIDEX:SOLVER-CONFIGURATION:v2\0";
const BASE_ARTIFACT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:BASE-ARTIFACT:v1\0";
const TARGET_PROFILE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:TARGET-PROFILE:v1\0";
const SOLVER_POLICY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:SOLVER-POLICY:v1\0";
const SOLVER_CANDIDATE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:SOLVER-CANDIDATE:v1\0";
const DECLARED_BACKEND_IMPLEMENTATION_DOMAIN: &[u8] =
    b"CEREBRO:TIDEX:DECLARED-BACKEND-IMPLEMENTATION:v2\0";
const EVALUATOR_POLICY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:EVALUATOR-POLICY:v1\0";
const EVALUATION_DESIGN_DOMAIN: &[u8] = b"CEREBRO:TIDEX:EVALUATION-DESIGN:v2\0";
const LINEAGE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:PROCEDURAL-LINEAGE:v1\0";
const DRIFT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:PROCEDURAL-DRIFT:v4\0";
const OBSERVED_DRIFT_EVIDENCE_DOMAIN: &[u8] =
    b"CEREBRO:TIDEX:OBSERVED-FUNCTIONAL-DRIFT-EVIDENCE:v2\0";
const SOLVER_RUN_RECEIPT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:SOLVER-RUN-RECEIPT:v2\0";
const SOLVER_CANDIDATE_REJECTION_RECEIPT_DOMAIN: &[u8] =
    b"CEREBRO:TIDEX:SOLVER-CANDIDATE-REJECTION-RECEIPT:v1\0";
const SOLVER_RUN_FAILURE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:SOLVER-RUN-FAILURE:v4\0";
const RETRIEVAL_POLICY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:PROCEDURAL-RETRIEVAL-POLICY:v4\0";
const RESEARCH_VALIDATION_RECEIPT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:RESEARCH-VALIDATION-RECEIPT:v1\0";
const RESEARCH_REPORT_RECEIPT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:RESEARCH-REPORT-RECEIPT:v1\0";

pub const MAX_ATTEMPTS: usize = 16_384;
pub const MAX_DRIFT_RECORDS: usize = 4_096;
pub const MAX_SOLVER_RUN_FAILURES: usize = 16_384;
pub const MAX_ADVICE_RESULTS: usize = 128;
pub const MAX_INDEPENDENT_EVALUATION_GROUPS: u32 = 1_000_000;
pub const MAX_MATRIX_DIMENSION: u64 = 1 << 40;
/// Maximum number of explicitly materialized scalar cells represented by one
/// problem description. Operator-only problems are not materialized and use
/// the independent per-axis ceiling instead.
pub const MAX_EXPLICIT_MATRIX_ELEMENTS: u64 = 1 << 31;
pub const MAX_BLOCKS: u32 = 1_000_000;
pub const MAX_ITERATIONS: u32 = 100_000_000;
pub const MAX_COMPUTE_UNITS: u64 = 1_000_000_000_000_000;

fn invalid(code: &str) -> BrainError {
    BrainError::Invalid(code.into())
}

fn integrity(code: &str) -> BrainError {
    BrainError::Integrity(code.into())
}

fn is_zero_digest(value: &str) -> bool {
    value.bytes().all(|byte| byte == b'0')
}

fn canonical_finite(value: f64) -> BrainResult<FiniteF64> {
    FiniteF64::new(if value == 0.0 { 0.0 } else { value })
}

fn is_canonical_finite(value: FiniteF64) -> bool {
    value.get() != 0.0 || value.bits() == 0.0_f64.to_bits()
}

macro_rules! memory_digest {
    ($(#[$metadata:meta])* $name:ident, $domain:ident) => {
        $(#[$metadata])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Sha256Digest);

        impl $name {
            fn computed(payload: &[u8]) -> Self {
                Self(Sha256Digest::digest_domain($domain, payload))
            }

            /// Bind an already authenticated byte identity into this semantic
            /// domain.  Equal raw hashes in different domains remain distinct
            /// Rust types and receive different semantic digest values.
            pub fn bind_exact_digest(value: &Sha256Digest) -> Self {
                Self::computed(value.as_str().as_bytes())
            }

            /// Commit exact bytes directly when no prior byte digest exists.
            pub fn of_exact_bytes(bytes: &[u8]) -> Self {
                Self::computed(bytes)
            }

            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(self.as_str())
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                self.0.serialize(serializer)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                Ok(Self(Sha256Digest::deserialize(deserializer)?))
            }
        }
    };
}

memory_digest!(BaseArtifactDigest, BASE_ARTIFACT_DOMAIN);
memory_digest!(TargetProfileDigest, TARGET_PROFILE_DOMAIN);
memory_digest!(SolverPolicyDigest, SOLVER_POLICY_DOMAIN);
memory_digest!(SolverCandidateDigest, SOLVER_CANDIDATE_DOMAIN);
memory_digest!(
    /// Identity of an external backend's declared implementation label. This
    /// is not proof that the backend executed that implementation.
    DeclaredBackendImplementationDigest,
    DECLARED_BACKEND_IMPLEMENTATION_DOMAIN
);
memory_digest!(EvaluatorPolicyDigest, EVALUATOR_POLICY_DOMAIN);
memory_digest!(EvaluationDesignDigest, EVALUATION_DESIGN_DOMAIN);
memory_digest!(ResearchReportReceiptDigest, RESEARCH_REPORT_RECEIPT_DOMAIN);
memory_digest!(ProceduralLineageId, LINEAGE_DOMAIN);
memory_digest!(SolverRunReceiptDigest, SOLVER_RUN_RECEIPT_DOMAIN);
memory_digest!(
    SolverCandidateRejectionReceiptDigest,
    SOLVER_CANDIDATE_REJECTION_RECEIPT_DOMAIN
);
memory_digest!(SolverRunFailureDigest, SOLVER_RUN_FAILURE_DOMAIN);
memory_digest!(ProceduralRetrievalPolicyDigest, RETRIEVAL_POLICY_DOMAIN);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ApplicabilityDigest(Sha256Digest);

impl ApplicabilityDigest {
    fn computed(payload: &[u8]) -> Self {
        Self(Sha256Digest::digest_domain(APPLICABILITY_DOMAIN, payload))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl Display for ApplicabilityDigest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for ApplicabilityDigest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ApplicabilityDigest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self(Sha256Digest::deserialize(deserializer)?))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SolverConfigurationDigest(Sha256Digest);

impl SolverConfigurationDigest {
    fn computed(payload: &[u8]) -> Self {
        Self(Sha256Digest::digest_domain(CONFIGURATION_DOMAIN, payload))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl Display for SolverConfigurationDigest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for SolverConfigurationDigest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SolverConfigurationDigest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self(Sha256Digest::deserialize(deserializer)?))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProceduralAttemptDigest(Sha256Digest);

impl ProceduralAttemptDigest {
    fn computed(payload: &[u8]) -> Self {
        Self(Sha256Digest::digest_domain(ATTEMPT_DOMAIN, payload))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl Display for ProceduralAttemptDigest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for ProceduralAttemptDigest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ProceduralAttemptDigest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self(Sha256Digest::deserialize(deserializer)?))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DriftRecordDigest(Sha256Digest);

impl DriftRecordDigest {
    fn computed(payload: &[u8]) -> Self {
        Self(Sha256Digest::digest_domain(DRIFT_DOMAIN, payload))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl Display for DriftRecordDigest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for DriftRecordDigest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for DriftRecordDigest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self(Sha256Digest::deserialize(deserializer)?))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum ProceduralSchema {
    #[serde(rename = "cerebro.tidex.procedural_memory/v4")]
    Current,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum NumericPrecision {
    Bfloat16,
    Float16,
    Float32,
    Float64,
    Mixed,
    Exact,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum MatrixStructure {
    Dense,
    Sparse,
    GroupSparse,
    BlockStructured,
    LowRankObserved,
    Mixed,
    OperatorOnly,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionSemantics {
    PureTensor,
    Stateful,
    ExternalEffects,
    HybridRuntime,
    SoftwareOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MatrixDimensions {
    rows: u64,
    columns: u64,
    /// `Some(0)` is an observed zero-rank design. `None` means that no rank
    /// diagnostic was available; the two states must never be conflated.
    estimated_effective_rank: Option<u64>,
    /// Number of declared input-feature groups/blocks. It is bounded by the
    /// input-column count, not by matrix rank or case count.
    block_count: u32,
}

impl MatrixDimensions {
    pub fn new(
        rows: u64,
        columns: u64,
        estimated_effective_rank: Option<u64>,
        block_count: u32,
    ) -> BrainResult<Self> {
        let value = Self {
            rows,
            columns,
            estimated_effective_rank,
            block_count,
        };
        value.validate_axes()?;
        Ok(value)
    }

    fn validate_axes(&self) -> BrainResult<()> {
        let min_dimension = self.rows.min(self.columns);
        if self.rows == 0
            || self.columns == 0
            || self.rows > MAX_MATRIX_DIMENSION
            || self.columns > MAX_MATRIX_DIMENSION
            || self
                .estimated_effective_rank
                .is_some_and(|rank| rank > min_dimension)
            || self.block_count > MAX_BLOCKS
            || u64::from(self.block_count) > self.columns
        {
            return Err(invalid("procedural_matrix_dimensions_invalid"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StatisticalProfile {
    log10_condition: Option<FiniteF64>,
    outlier_fraction: Option<FiniteF64>,
    observed_sparsity: Option<FiniteF64>,
    noise_fraction: Option<FiniteF64>,
}

impl StatisticalProfile {
    /// Describe only statistics that were actually measured. `None` means
    /// unknown; it is never interpreted as zero or as a perfect match.
    pub fn new(
        log10_condition: Option<f64>,
        outlier_fraction: Option<f64>,
        observed_sparsity: Option<f64>,
        noise_fraction: Option<f64>,
    ) -> BrainResult<Self> {
        let value = Self {
            log10_condition: log10_condition.map(canonical_finite).transpose()?,
            outlier_fraction: outlier_fraction.map(canonical_finite).transpose()?,
            observed_sparsity: observed_sparsity.map(canonical_finite).transpose()?,
            noise_fraction: noise_fraction.map(canonical_finite).transpose()?,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn fully_observed(
        log10_condition: Option<f64>,
        outlier_fraction: f64,
        observed_sparsity: f64,
        noise_fraction: f64,
    ) -> BrainResult<Self> {
        Self::new(
            log10_condition,
            Some(outlier_fraction),
            Some(observed_sparsity),
            Some(noise_fraction),
        )
    }

    fn validate(&self) -> BrainResult<()> {
        if self
            .log10_condition
            .is_some_and(|value| value.get() < 0.0 || !is_canonical_finite(value))
            || self
                .outlier_fraction
                .is_some_and(|value| !unit_interval(value) || !is_canonical_finite(value))
            || self
                .observed_sparsity
                .is_some_and(|value| !unit_interval(value) || !is_canonical_finite(value))
            || self
                .noise_fraction
                .is_some_and(|value| !unit_interval(value) || !is_canonical_finite(value))
        {
            return Err(invalid("procedural_statistical_profile_invalid"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NumericProblemProfile {
    precision: NumericPrecision,
    structure: MatrixStructure,
    execution_semantics: ExecutionSemantics,
}

impl NumericProblemProfile {
    pub fn new(
        precision: NumericPrecision,
        structure: MatrixStructure,
        execution_semantics: ExecutionSemantics,
    ) -> Self {
        Self {
            precision,
            structure,
            execution_semantics,
        }
    }
}

/// Retrieval-only mathematical context of one exact solver problem.  It is
/// deliberately incapable of standing in for [`ExactSolverProblemDigest`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Applicability {
    dimensions: MatrixDimensions,
    statistics: StatisticalProfile,
    profile: NumericProblemProfile,
}

impl Applicability {
    pub fn new(
        dimensions: MatrixDimensions,
        statistics: StatisticalProfile,
        profile: NumericProblemProfile,
    ) -> BrainResult<Self> {
        let value = Self {
            dimensions,
            statistics,
            profile,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> BrainResult<()> {
        self.dimensions.validate_axes()?;
        self.statistics.validate()?;
        let block_count_valid = match self.profile.structure {
            MatrixStructure::BlockStructured | MatrixStructure::GroupSparse => {
                self.dimensions.block_count > 0
            }
            MatrixStructure::Dense
            | MatrixStructure::Sparse
            | MatrixStructure::LowRankObserved
            | MatrixStructure::Mixed
            | MatrixStructure::OperatorOnly => self.dimensions.block_count == 0,
        };
        if !block_count_valid {
            return Err(invalid("procedural_block_count_semantics_invalid"));
        }
        if self.profile.structure != MatrixStructure::OperatorOnly {
            let elements = self
                .dimensions
                .rows
                .checked_mul(self.dimensions.columns)
                .ok_or_else(|| invalid("procedural_matrix_elements_overflow"))?;
            if elements > MAX_EXPLICIT_MATRIX_ELEMENTS {
                return Err(invalid("procedural_explicit_matrix_budget_exceeded"));
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> BrainResult<ApplicabilityDigest> {
        self.validate()?;
        Ok(ApplicabilityDigest::computed(&serde_json::to_vec(self)?))
    }

    pub fn rows(&self) -> u64 {
        self.dimensions.rows
    }

    pub fn columns(&self) -> u64 {
        self.dimensions.columns
    }

    pub fn estimated_effective_rank(&self) -> Option<u64> {
        self.dimensions.estimated_effective_rank
    }

    pub fn execution_semantics(&self) -> ExecutionSemantics {
        self.profile.execution_semantics
    }
}

fn unit_interval(value: FiniteF64) -> bool {
    (0.0..=1.0).contains(&value.get())
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SolverFamily {
    CholeskyRidgeLowRank,
    /// The bounded one-sided Jacobi SVD implemented by the built-in
    /// portfolio. This is deliberately not mislabeled as GELSD or a
    /// divide-and-conquer SVD.
    DirectJacobiSvd,
    PivotedQr,
    DivideConquerSvd,
    IterativeLsqr,
    IterativeLsmr,
    RandomizedSvd,
    RobustMEstimator,
    SparseProximal,
    GroupSparseProximal,
    BlockStructured,
    FullRankGradient,
    OrthogonalizedFullRank,
    /// A candidate proposed by an external backend whose implementation
    /// family is not trusted. Only its validated storage representation is
    /// remembered.
    ExternalProposal,
    HybridRuntime,
    SoftwareOnly,
}

/// Family-specific structural parameters. A sparse budget can therefore never
/// be mislabeled as a matrix rank, and an iterative ceiling cannot be attached
/// to a direct decomposition by accident.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SolverParameters {
    LowRank {
        rank: u32,
    },
    RandomizedLowRank {
        rank: u32,
        oversampling: u32,
    },
    Direct,
    Iterative,
    Robust {
        tuning_constant: FiniteF64,
    },
    Sparse {
        maximum_nonzero: u64,
    },
    GroupSparse {
        maximum_groups: u32,
    },
    BlockStructured {
        maximum_blocks: u32,
    },
    FullRank,
    External {
        representation: SolverRepresentationKind,
        declared_implementation_digest: DeclaredBackendImplementationDigest,
    },
    Hybrid,
    Software,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SolverRepresentationKind {
    LowRank,
    Dense,
    Sparse,
    Block,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SolverConfiguration {
    family: SolverFamily,
    parameters: SolverParameters,
    regularization: Option<FiniteF64>,
    tolerance: Option<FiniteF64>,
    max_iterations: Option<u32>,
}

impl SolverConfiguration {
    pub fn new(
        family: SolverFamily,
        parameters: SolverParameters,
        regularization: Option<f64>,
        tolerance: Option<f64>,
        max_iterations: Option<u32>,
    ) -> BrainResult<Self> {
        let value = Self {
            family,
            parameters,
            regularization: regularization.map(canonical_finite).transpose()?,
            tolerance: tolerance.map(canonical_finite).transpose()?,
            max_iterations,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> BrainResult<()> {
        if self
            .regularization
            .is_some_and(|value| value.get() < 0.0 || !is_canonical_finite(value))
            || self
                .tolerance
                .is_some_and(|value| value.get() <= 0.0 || !is_canonical_finite(value))
            || self
                .max_iterations
                .is_some_and(|value| value == 0 || value > MAX_ITERATIONS)
        {
            return Err(invalid("solver_configuration_invalid"));
        }
        let parameters_valid = match (self.family, &self.parameters) {
            (SolverFamily::CholeskyRidgeLowRank, SolverParameters::LowRank { rank }) => *rank > 0,
            (
                SolverFamily::RandomizedSvd,
                SolverParameters::RandomizedLowRank { rank, oversampling },
            ) => *rank > 0 && *oversampling > 0 && rank.checked_add(*oversampling).is_some(),
            (
                SolverFamily::DirectJacobiSvd
                | SolverFamily::PivotedQr
                | SolverFamily::DivideConquerSvd,
                SolverParameters::Direct,
            ) => true,
            (
                SolverFamily::IterativeLsqr | SolverFamily::IterativeLsmr,
                SolverParameters::Iterative,
            ) => true,
            (SolverFamily::RobustMEstimator, SolverParameters::Robust { tuning_constant }) => {
                tuning_constant.get() > 0.0 && is_canonical_finite(*tuning_constant)
            }
            (SolverFamily::SparseProximal, SolverParameters::Sparse { maximum_nonzero }) => {
                *maximum_nonzero > 0 && *maximum_nonzero <= MAX_EXPLICIT_MATRIX_ELEMENTS
            }
            (
                SolverFamily::GroupSparseProximal,
                SolverParameters::GroupSparse { maximum_groups },
            ) => *maximum_groups > 0 && *maximum_groups <= MAX_BLOCKS,
            (
                SolverFamily::BlockStructured,
                SolverParameters::BlockStructured { maximum_blocks },
            ) => *maximum_blocks > 0 && *maximum_blocks <= MAX_BLOCKS,
            (
                SolverFamily::FullRankGradient | SolverFamily::OrthogonalizedFullRank,
                SolverParameters::FullRank,
            ) => true,
            (
                SolverFamily::ExternalProposal,
                SolverParameters::External {
                    declared_implementation_digest,
                    ..
                },
            ) => !is_zero_digest(declared_implementation_digest.as_str()),
            (SolverFamily::HybridRuntime, SolverParameters::Hybrid) => true,
            (SolverFamily::SoftwareOnly, SolverParameters::Software) => true,
            _ => false,
        };
        if !parameters_valid {
            return Err(invalid("solver_configuration_parameter_semantics_invalid"));
        }
        let control_fields_valid = match self.family {
            SolverFamily::CholeskyRidgeLowRank => {
                self.regularization
                    .is_some_and(|regularization| regularization.get() > 0.0)
                    && self.tolerance.is_some()
                    && self.max_iterations.is_none()
            }
            SolverFamily::DirectJacobiSvd => {
                self.regularization.is_none()
                    && self.tolerance.is_some()
                    && self.max_iterations.is_some()
            }
            SolverFamily::PivotedQr | SolverFamily::DivideConquerSvd => {
                self.regularization.is_none()
                    && self.tolerance.is_some()
                    && self.max_iterations.is_none()
            }
            SolverFamily::IterativeLsqr
            | SolverFamily::IterativeLsmr
            | SolverFamily::RobustMEstimator
            | SolverFamily::SparseProximal
            | SolverFamily::GroupSparseProximal
            | SolverFamily::BlockStructured
            | SolverFamily::FullRankGradient
            | SolverFamily::OrthogonalizedFullRank => {
                self.regularization.is_none()
                    && self.tolerance.is_some()
                    && self.max_iterations.is_some()
            }
            SolverFamily::RandomizedSvd => {
                self.regularization.is_none()
                    && self.tolerance.is_some()
                    && self.max_iterations.is_some()
            }
            SolverFamily::ExternalProposal
            | SolverFamily::HybridRuntime
            | SolverFamily::SoftwareOnly => {
                self.regularization.is_none()
                    && self.tolerance.is_none()
                    && self.max_iterations.is_none()
            }
        };
        if !control_fields_valid {
            return Err(invalid("solver_configuration_control_fields_invalid"));
        }
        Ok(())
    }

    fn validate_for(&self, applicability: &Applicability) -> BrainResult<()> {
        self.validate()?;
        applicability.validate()?;
        let minimum_dimension = applicability
            .dimensions
            .rows
            .min(applicability.dimensions.columns);
        let compatible = match &self.parameters {
            SolverParameters::LowRank { rank } => u64::from(*rank) <= minimum_dimension,
            SolverParameters::RandomizedLowRank { rank, oversampling } => rank
                .checked_add(*oversampling)
                .is_some_and(|sketch_rank| u64::from(sketch_rank) <= minimum_dimension),
            // Candidate sparsity is over output-by-input parameters, whereas
            // Applicability dimensions describe the case-by-input design.
            // Without output dimensionality no tighter comparison is sound.
            SolverParameters::Sparse { .. } => true,
            SolverParameters::GroupSparse { maximum_groups } => {
                applicability.profile.structure == MatrixStructure::GroupSparse
                    && *maximum_groups <= applicability.dimensions.block_count
            }
            SolverParameters::BlockStructured { maximum_blocks } => {
                applicability.profile.structure == MatrixStructure::BlockStructured
                    && *maximum_blocks <= applicability.dimensions.block_count
            }
            SolverParameters::Direct
            | SolverParameters::Iterative
            | SolverParameters::Robust { .. }
            | SolverParameters::FullRank
            | SolverParameters::External { .. }
            | SolverParameters::Hybrid
            | SolverParameters::Software => true,
        };
        if !compatible {
            return Err(invalid("solver_configuration_applicability_mismatch"));
        }
        Ok(())
    }

    pub fn family(&self) -> SolverFamily {
        self.family
    }

    pub fn rank(&self) -> Option<u32> {
        match &self.parameters {
            SolverParameters::LowRank { rank }
            | SolverParameters::RandomizedLowRank { rank, .. } => Some(*rank),
            _ => None,
        }
    }

    pub fn digest(&self) -> BrainResult<SolverConfigurationDigest> {
        self.validate()?;
        Ok(SolverConfigurationDigest::computed(&serde_json::to_vec(
            self,
        )?))
    }
}

/// All semantic objects to which an attempt is exactly bound.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CapabilityContext {
    system_envelope_digest: SystemEnvelopeDigest,
    capability_id: CapabilityId,
    capability_ir_digest: CapabilityIrDigest,
    base_artifact_digest: BaseArtifactDigest,
    target_profile_digest: TargetProfileDigest,
}

impl CapabilityContext {
    pub fn new(
        system_envelope_digest: SystemEnvelopeDigest,
        capability_id: CapabilityId,
        capability_ir_digest: CapabilityIrDigest,
        base_artifact_digest: BaseArtifactDigest,
        target_profile_digest: TargetProfileDigest,
    ) -> BrainResult<Self> {
        let value = Self {
            system_envelope_digest,
            capability_id,
            capability_ir_digest,
            base_artifact_digest,
            target_profile_digest,
        };
        value.validate_nonzero()?;
        Ok(value)
    }

    fn validate_nonzero(&self) -> BrainResult<()> {
        if is_zero_digest(self.system_envelope_digest.as_str())
            || is_zero_digest(self.capability_ir_digest.as_str())
            || is_zero_digest(self.base_artifact_digest.as_str())
            || is_zero_digest(self.target_profile_digest.as_str())
        {
            return Err(invalid("procedural_capability_context_zero_digest"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SolverArtifactContext {
    problem_digest: ExactSolverProblemDigest,
    solver_policy_digest: SolverPolicyDigest,
    candidate_digest: SolverCandidateDigest,
}

impl SolverArtifactContext {
    pub fn new(
        problem_digest: ExactSolverProblemDigest,
        solver_policy_digest: SolverPolicyDigest,
        candidate_digest: SolverCandidateDigest,
    ) -> BrainResult<Self> {
        let value = Self {
            problem_digest,
            solver_policy_digest,
            candidate_digest,
        };
        value.validate_nonzero()?;
        Ok(value)
    }

    fn validate_nonzero(&self) -> BrainResult<()> {
        if is_zero_digest(self.problem_digest.as_str())
            || is_zero_digest(self.solver_policy_digest.as_str())
            || is_zero_digest(self.candidate_digest.as_str())
        {
            return Err(invalid("procedural_solver_context_zero_digest"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AttemptBindings {
    capability: CapabilityContext,
    solver: SolverArtifactContext,
}

impl AttemptBindings {
    pub fn new(capability: CapabilityContext, solver: SolverArtifactContext) -> BrainResult<Self> {
        let value = Self { capability, solver };
        value.validate_nonzero()?;
        Ok(value)
    }

    fn validate_nonzero(&self) -> BrainResult<()> {
        self.capability.validate_nonzero()?;
        self.solver.validate_nonzero()
    }

    pub fn capability_id(&self) -> &CapabilityId {
        &self.capability.capability_id
    }

    pub fn problem_digest(&self) -> &ExactSolverProblemDigest {
        &self.solver.problem_digest
    }

    pub fn candidate_digest(&self) -> &SolverCandidateDigest {
        &self.solver.candidate_digest
    }
}

/// Typed bridge between the governance layer's opaque variant identity and an
/// exact numerical candidate artifact. It avoids teaching procedural memory
/// any orchestrator-specific naming convention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifiedCandidateVariantBinding {
    variant_id: VariantId,
    candidate_digest: SolverCandidateDigest,
}

impl VerifiedCandidateVariantBinding {
    pub(crate) fn from_candidate(
        variant_id: VariantId,
        candidate: &CandidateRepresentation,
    ) -> BrainResult<Self> {
        let exact_digest = candidate.exact_digest()?;
        Ok(Self {
            variant_id,
            candidate_digest: SolverCandidateDigest::bind_exact_digest(exact_digest.as_digest()),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResearchValidation {
    /// Exact paired report identity, independent of any later decision made
    /// from it. This is indexed so drift cannot invent report revisions.
    report_receipt_digest: ResearchReportReceiptDigest,
    /// Step-local receipt over the exact paired report and the exact research
    /// gate decision. It deliberately excludes PETFC and promotion state.
    receipt_digest: EvaluationReceiptDigest,
    evaluator_policy_digest: EvaluatorPolicyDigest,
    /// Exact holdout-suite identity. This must not be replaced by the
    /// training-problem digest stored in [`AttemptBindings`].
    evaluation_design_digest: EvaluationDesignDigest,
    evaluated_candidate_digest: SolverCandidateDigest,
    independent_group_count: u32,
}

impl ResearchValidation {
    fn new(
        report_receipt_digest: ResearchReportReceiptDigest,
        receipt_digest: EvaluationReceiptDigest,
        evaluator_policy_digest: EvaluatorPolicyDigest,
        evaluation_design_digest: EvaluationDesignDigest,
        evaluated_candidate_digest: SolverCandidateDigest,
        independent_group_count: u32,
    ) -> BrainResult<Self> {
        let value = Self {
            report_receipt_digest,
            receipt_digest,
            evaluator_policy_digest,
            evaluation_design_digest,
            evaluated_candidate_digest,
            independent_group_count,
        };
        value.validate()?;
        Ok(value)
    }

    /// Bind suite identity and group cardinality directly to one sealed paired
    /// report plus its exact non-authorizing research gate. Callers cannot
    /// substitute a training problem, invent a receipt, or fold a later PETFC
    /// trajectory/promotion decision into this reusable validation.
    pub(crate) fn from_verified_paired_report(
        report: &PairedEvaluationReport,
        gate: &CandidateGateDecision,
        candidate_binding: &VerifiedCandidateVariantBinding,
    ) -> BrainResult<Self> {
        if report.candidate_id() != &candidate_binding.variant_id
            || gate.candidate_id() != report.candidate_id()
            || gate.report_digest() != report.digest()
            || gate.metric_catalog_digest() != report.metric_catalog_digest()
            || gate.evaluation_policy_digest() != report.evaluation_policy_digest()
            || gate.independence_design_digest() != report.independence_design_digest()
            || gate.source_observation_window() != report.observation_window()
        {
            return Err(integrity("research_validation_candidate_binding_mismatch"));
        }
        let independent_group_count = u32::try_from(report.independent_group_count())
            .map_err(|_| invalid("research_validation_group_count_overflow"))?;
        let report_receipt_digest = research_report_receipt(report);
        let mut receipt_frame = Vec::new();
        for digest in [report.digest().as_str(), gate.digest().as_str()] {
            let length = u64::try_from(digest.len())
                .map_err(|_| invalid("research_validation_receipt_length_overflow"))?;
            receipt_frame.extend_from_slice(&length.to_be_bytes());
            receipt_frame.extend_from_slice(digest.as_bytes());
        }
        let receipt_digest = EvaluationReceiptDigest::from(Sha256Digest::digest_domain(
            RESEARCH_VALIDATION_RECEIPT_DOMAIN,
            &receipt_frame,
        ));
        Self::new(
            report_receipt_digest,
            receipt_digest,
            EvaluatorPolicyDigest::of_exact_bytes(
                report.evaluation_policy_digest().as_str().as_bytes(),
            ),
            research_evaluation_design(report)?,
            candidate_binding.candidate_digest.clone(),
            independent_group_count,
        )
    }

    fn validate(&self) -> BrainResult<()> {
        if is_zero_digest(self.report_receipt_digest.as_str())
            || is_zero_digest(self.receipt_digest.as_str())
            || is_zero_digest(self.evaluator_policy_digest.as_str())
            || is_zero_digest(self.evaluation_design_digest.as_str())
            || self.independent_group_count == 0
            || self.independent_group_count > MAX_INDEPENDENT_EVALUATION_GROUPS
        {
            return Err(invalid("research_validation_invalid"));
        }
        Ok(())
    }

    pub fn receipt_digest(&self) -> &EvaluationReceiptDigest {
        &self.receipt_digest
    }
}

fn research_report_receipt(report: &PairedEvaluationReport) -> ResearchReportReceiptDigest {
    ResearchReportReceiptDigest::of_exact_bytes(report.digest().as_str().as_bytes())
}

fn research_evaluation_design(
    report: &PairedEvaluationReport,
) -> BrainResult<EvaluationDesignDigest> {
    let mut frame = Vec::new();
    for digest in [
        report.independence_design_digest().as_str(),
        report.metric_catalog_digest().as_str(),
        report.evaluation_policy_digest().as_str(),
    ] {
        let length = u64::try_from(digest.len())
            .map_err(|_| invalid("research_validation_design_length_overflow"))?;
        frame.extend_from_slice(&length.to_be_bytes());
        frame.extend_from_slice(digest.as_bytes());
    }
    Ok(EvaluationDesignDigest::of_exact_bytes(&frame))
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum AttemptStatus {
    Validated,
    Rejected,
    Failed,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    IllConditioned,
    RankInsufficient,
    ResidualTooLarge,
    OutlierSensitivity,
    Divergence,
    NonFinite,
    InvariantRegression,
    CrossCapabilityInterference,
    EvaluationInstability,
    ResourceExhaustion,
    ResidencyMismatch,
    EvidenceInvalid,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum FailureStage {
    Diagnosis,
    Solve,
    Materialization,
    ResearchValidation,
    Canary,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum FailureSeverity {
    Recoverable,
    Serious,
    HardInvariant,
}

impl FailureSeverity {
    fn penalty(self) -> f64 {
        match self {
            Self::Recoverable => 0.55,
            Self::Serious => 0.8,
            Self::HardInvariant => 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FailureRecord {
    kind: FailureKind,
    stage: FailureStage,
    severity: FailureSeverity,
}

impl FailureRecord {
    pub fn new(kind: FailureKind, stage: FailureStage, severity: FailureSeverity) -> Self {
        Self {
            kind,
            stage,
            severity,
        }
    }

    pub fn kind(&self) -> FailureKind {
        self.kind
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionKind {
    IncreaseRank,
    ReduceRank,
    IncreaseRegularization,
    SwitchToPivotedQr,
    SwitchToSvd,
    SwitchToIterative,
    SwitchToRobustEstimator,
    SwitchToSparse,
    SwitchToBlockStructured,
    EscalateFullRank,
    SelectHybridResidency,
    SelectSoftwareResidency,
    ExpandIndependentEvidence,
    RepairEvaluator,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionStatus {
    Proposed,
    AppliedUnverified,
    IndependentlyValidated,
    Refuted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CorrectionRecord {
    kind: CorrectionKind,
    status: CorrectionStatus,
}

impl CorrectionRecord {
    pub fn new(kind: CorrectionKind, status: CorrectionStatus) -> Self {
        Self { kind, status }
    }

    pub fn kind(&self) -> CorrectionKind {
        self.kind
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OutcomeMetrics {
    solver_relative_residual: Option<FiniteF64>,
    independent_gate_advanced: Option<bool>,
    /// Exactly metered work. `None` means that the evaluator could not
    /// account it; no sentinel value is accepted as a substitute.
    deterministic_work_units: Option<u64>,
}

impl OutcomeMetrics {
    /// Test/internal constructor. Production code derives these fields from a
    /// selected solver evaluation and its step-local independent gate.
    fn new(
        solver_relative_residual: Option<f64>,
        independent_gate_advanced: Option<bool>,
        deterministic_work_units: Option<u64>,
    ) -> BrainResult<Self> {
        let value = Self {
            solver_relative_residual: solver_relative_residual.map(canonical_finite).transpose()?,
            independent_gate_advanced,
            deterministic_work_units,
        };
        value.validate()?;
        Ok(value)
    }

    /// Derive configuration-local evidence only. PETFC trajectory assessments
    /// are intentionally excluded because an accumulated path is not caused
    /// solely by the current solver configuration.
    pub(crate) fn from_verified_numerical_sources(
        selected: &CandidateEvaluation,
        candidate_binding: &VerifiedCandidateVariantBinding,
        report: &PairedEvaluationReport,
        gate: &CandidateGateDecision,
        deterministic_work_units: Option<u64>,
    ) -> BrainResult<Self> {
        if selected.status() != SolverStatus::Accepted {
            return Err(invalid("procedural_metrics_solver_candidate_not_accepted"));
        }
        let candidate = selected
            .candidate()
            .ok_or_else(|| integrity("procedural_metrics_solver_candidate_missing"))?;
        let exact_candidate_digest = candidate.exact_digest()?;
        if SolverCandidateDigest::bind_exact_digest(exact_candidate_digest.as_digest())
            != candidate_binding.candidate_digest
        {
            return Err(integrity("procedural_metrics_candidate_binding_mismatch"));
        }
        let selected_metrics = selected
            .metrics()
            .ok_or_else(|| integrity("procedural_metrics_solver_observation_missing"))?;
        if report.candidate_id() != &candidate_binding.variant_id
            || gate.candidate_id() != &candidate_binding.variant_id
            || gate.report_digest() != report.digest()
            || gate.metric_catalog_digest() != report.metric_catalog_digest()
            || gate.evaluation_policy_digest() != report.evaluation_policy_digest()
            || gate.independence_design_digest() != report.independence_design_digest()
            || gate.source_observation_window() != report.observation_window()
        {
            return Err(integrity("procedural_metrics_gate_report_mismatch"));
        }
        let independent_gate_advanced = match gate.disposition() {
            CandidateGateDisposition::AdvanceCandidate => Some(true),
            CandidateGateDisposition::Reject => Some(false),
            CandidateGateDisposition::BoundedUnknown => None,
        };
        Self::new(
            selected_metrics.relative_residual(),
            independent_gate_advanced,
            deterministic_work_units,
        )
    }

    fn validate(&self) -> BrainResult<()> {
        if self
            .solver_relative_residual
            .is_some_and(|value| value.get() < 0.0 || !is_canonical_finite(value))
            || self
                .deterministic_work_units
                .is_some_and(|units| units == 0 || units > MAX_COMPUTE_UNITS)
        {
            return Err(invalid("procedural_outcome_metrics_invalid"));
        }
        Ok(())
    }

    fn is_complete(&self) -> bool {
        self.solver_relative_residual.is_some() && self.independent_gate_advanced == Some(true)
    }

    /// Whether the sealed functional observations are sufficient to support a
    /// `Validated` procedural status. Optional work accounting is deliberately
    /// excluded: unknown cost cannot erase established functional evidence.
    pub(crate) fn supports_validated_status(&self) -> bool {
        self.is_complete()
    }

    fn positive_utility(&self) -> Option<f64> {
        match (
            self.independent_gate_advanced,
            self.solver_relative_residual,
        ) {
            (Some(true), Some(residual)) => Some(1.0 / (1.0 + residual.get())),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AttemptOutcome {
    status: AttemptStatus,
    metrics: OutcomeMetrics,
    failure: Option<FailureRecord>,
    correction: Option<CorrectionRecord>,
}

impl AttemptOutcome {
    pub(crate) fn new(
        status: AttemptStatus,
        metrics: OutcomeMetrics,
        failure: Option<FailureRecord>,
        correction: Option<CorrectionRecord>,
    ) -> BrainResult<Self> {
        let value = Self {
            status,
            metrics,
            failure,
            correction,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> BrainResult<()> {
        self.metrics.validate()?;
        let needs_failure = matches!(self.status, AttemptStatus::Rejected | AttemptStatus::Failed);
        if needs_failure != self.failure.is_some()
            || (self.status == AttemptStatus::Validated && !self.metrics.is_complete())
            || self.correction.as_ref().is_some_and(|correction| {
                !matches!(
                    (self.status, correction.status),
                    (
                        AttemptStatus::Validated,
                        CorrectionStatus::IndependentlyValidated
                    ) | (
                        AttemptStatus::Rejected | AttemptStatus::Failed,
                        CorrectionStatus::Proposed | CorrectionStatus::Refuted,
                    ) | (
                        AttemptStatus::Inconclusive,
                        CorrectionStatus::AppliedUnverified
                    )
                )
            })
        {
            return Err(invalid("procedural_outcome_semantics_invalid"));
        }
        Ok(())
    }

    fn validate_applied_correction(
        &self,
        applied_correction: Option<CorrectionKind>,
    ) -> BrainResult<()> {
        let semantics_valid = match (applied_correction, &self.correction) {
            (None, None) => true,
            (None, Some(correction)) => correction.status == CorrectionStatus::Proposed,
            (Some(applied), Some(correction)) if applied == correction.kind => matches!(
                (self.status, correction.status),
                (
                    AttemptStatus::Validated,
                    CorrectionStatus::IndependentlyValidated
                ) | (
                    AttemptStatus::Rejected | AttemptStatus::Failed,
                    CorrectionStatus::Refuted
                ) | (
                    AttemptStatus::Inconclusive,
                    CorrectionStatus::AppliedUnverified
                )
            ),
            (Some(_), None) | (Some(_), Some(_)) => false,
        };
        if !semantics_valid {
            return Err(invalid("procedural_applied_correction_semantics_invalid"));
        }
        Ok(())
    }

    pub fn status(&self) -> AttemptStatus {
        self.status
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AttemptLineage {
    lineage_id: ProceduralLineageId,
    ordinal: u64,
    parent_attempt_digest: Option<ProceduralAttemptDigest>,
}

impl AttemptLineage {
    pub fn root(lineage_id: ProceduralLineageId) -> Self {
        Self {
            lineage_id,
            ordinal: 0,
            parent_attempt_digest: None,
        }
    }

    pub fn child(
        lineage_id: ProceduralLineageId,
        ordinal: u64,
        parent_attempt_digest: ProceduralAttemptDigest,
    ) -> BrainResult<Self> {
        if ordinal == 0 {
            return Err(invalid("procedural_lineage_child_ordinal_invalid"));
        }
        Ok(Self {
            lineage_id,
            ordinal,
            parent_attempt_digest: Some(parent_attempt_digest),
        })
    }

    fn validate_shape(&self) -> BrainResult<()> {
        match (self.ordinal, &self.parent_attempt_digest) {
            (0, None) | (1.., Some(_)) => Ok(()),
            (0, Some(_)) | (1.., None) => Err(invalid("procedural_lineage_shape_invalid")),
        }
    }
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct AttemptProjection<'a> {
    schema: ProceduralSchema,
    bindings: &'a AttemptBindings,
    applicability: &'a Applicability,
    configuration: &'a SolverConfiguration,
    lineage: &'a AttemptLineage,
    observed_revision: u64,
    research_validation: &'a Option<ResearchValidation>,
    solver_rejection_receipt: &'a Option<SolverCandidateRejectionReceiptDigest>,
    outcome: &'a AttemptOutcome,
    applied_correction: &'a Option<CorrectionKind>,
}

/// Typed construction input accepted only by crate-internal verification
/// authorities.  Keeping this separate prevents a public caller from turning
/// self-reported metrics into procedural advice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifiedAttemptDraft {
    bindings: AttemptBindings,
    applicability: Applicability,
    configuration: SolverConfiguration,
    lineage: AttemptLineage,
    observed_revision: u64,
    applied_correction: Option<CorrectionKind>,
}

impl VerifiedAttemptDraft {
    pub(crate) fn new(
        bindings: AttemptBindings,
        applicability: Applicability,
        configuration: SolverConfiguration,
        lineage: AttemptLineage,
        observed_revision: u64,
        applied_correction: Option<CorrectionKind>,
    ) -> BrainResult<Self> {
        if observed_revision == 0 {
            return Err(invalid("procedural_attempt_revision_invalid"));
        }
        Ok(Self {
            bindings,
            applicability,
            configuration,
            lineage,
            observed_revision,
            applied_correction,
        })
    }
}

/// One immutable attempt.  Deserialization is intentionally untrusted; every
/// reducer entrypoint recalculates `attempt_digest` before using the record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SolverAttempt {
    schema: ProceduralSchema,
    bindings: AttemptBindings,
    applicability: Applicability,
    configuration: SolverConfiguration,
    lineage: AttemptLineage,
    observed_revision: u64,
    research_validation: Option<ResearchValidation>,
    solver_rejection_receipt: Option<SolverCandidateRejectionReceiptDigest>,
    outcome: AttemptOutcome,
    applied_correction: Option<CorrectionKind>,
    attempt_digest: ProceduralAttemptDigest,
}

impl SolverAttempt {
    pub(crate) fn seal(
        draft: VerifiedAttemptDraft,
        research_validation: Option<ResearchValidation>,
        outcome: AttemptOutcome,
    ) -> BrainResult<Self> {
        let mut value = Self {
            schema: ProceduralSchema::Current,
            bindings: draft.bindings,
            applicability: draft.applicability,
            configuration: draft.configuration,
            lineage: draft.lineage,
            observed_revision: draft.observed_revision,
            research_validation,
            solver_rejection_receipt: None,
            outcome,
            applied_correction: draft.applied_correction,
            attempt_digest: ProceduralAttemptDigest::computed(b"draft"),
        };
        value.attempt_digest = value.calculate_digest()?;
        value.authenticate()?;
        Ok(value)
    }

    /// Seal a real candidate that the authenticated portfolio evaluated and
    /// rejected before any holdout research gate ran. This path is distinct
    /// from a candidate-free run failure and cannot create positive advice.
    pub(crate) fn seal_rejected_candidate(
        draft: VerifiedAttemptDraft,
        report: &SolverPortfolioReport,
        evaluation_index: usize,
    ) -> BrainResult<Self> {
        if report.problem_digest() != draft.bindings.problem_digest()
            || SolverPolicyDigest::bind_exact_digest(report.policy_digest().as_digest())
                != draft.bindings.solver.solver_policy_digest
        {
            return Err(integrity(
                "procedural_solver_rejection_report_binding_mismatch",
            ));
        }
        let evaluation = report
            .evaluations()
            .get(evaluation_index)
            .ok_or_else(|| invalid("procedural_solver_rejection_index_invalid"))?;
        if evaluation.status() != SolverStatus::Rejected
            || !configuration_matches_backend(&draft.configuration, evaluation)
        {
            return Err(integrity("procedural_solver_rejection_evaluation_invalid"));
        }
        let candidate = evaluation
            .candidate()
            .ok_or_else(|| integrity("procedural_solver_rejection_candidate_missing"))?;
        let exact_candidate = candidate.exact_digest()?;
        if SolverCandidateDigest::bind_exact_digest(exact_candidate.as_digest())
            != draft.bindings.solver.candidate_digest
        {
            return Err(integrity(
                "procedural_solver_rejection_candidate_binding_mismatch",
            ));
        }
        let metrics = evaluation
            .metrics()
            .ok_or_else(|| integrity("procedural_solver_rejection_metrics_missing"))?;
        let failure = failure_from_solver_rejection(evaluation.reason())?;
        let outcome = AttemptOutcome::new(
            AttemptStatus::Rejected,
            OutcomeMetrics::new(metrics.relative_residual(), None, None)?,
            Some(failure),
            None,
        )?;
        let report_digest = report.exact_digest()?;
        let evaluation_index = u64::try_from(evaluation_index)
            .map_err(|_| invalid("procedural_solver_rejection_index_overflow"))?;
        let mut receipt_frame = Vec::new();
        receipt_frame.extend_from_slice(report_digest.as_str().as_bytes());
        receipt_frame.extend_from_slice(&evaluation_index.to_be_bytes());
        let solver_rejection_receipt = Some(SolverCandidateRejectionReceiptDigest::of_exact_bytes(
            &receipt_frame,
        ));
        let mut value = Self {
            schema: ProceduralSchema::Current,
            bindings: draft.bindings,
            applicability: draft.applicability,
            configuration: draft.configuration,
            lineage: draft.lineage,
            observed_revision: draft.observed_revision,
            research_validation: None,
            solver_rejection_receipt,
            outcome,
            applied_correction: draft.applied_correction,
            attempt_digest: ProceduralAttemptDigest::computed(b"draft"),
        };
        value.attempt_digest = value.calculate_digest()?;
        value.authenticate()?;
        Ok(value)
    }

    fn projection(&self) -> AttemptProjection<'_> {
        AttemptProjection {
            schema: self.schema,
            bindings: &self.bindings,
            applicability: &self.applicability,
            configuration: &self.configuration,
            lineage: &self.lineage,
            observed_revision: self.observed_revision,
            research_validation: &self.research_validation,
            solver_rejection_receipt: &self.solver_rejection_receipt,
            outcome: &self.outcome,
            applied_correction: &self.applied_correction,
        }
    }

    fn calculate_digest(&self) -> BrainResult<ProceduralAttemptDigest> {
        Ok(ProceduralAttemptDigest::computed(&serde_json::to_vec(
            &self.projection(),
        )?))
    }

    pub fn authenticate(&self) -> BrainResult<()> {
        self.bindings.validate_nonzero()?;
        self.applicability.validate()?;
        self.configuration.validate_for(&self.applicability)?;
        self.lineage.validate_shape()?;
        self.outcome.validate()?;
        self.outcome
            .validate_applied_correction(self.applied_correction)?;
        let evidence_shape_valid = match (
            &self.research_validation,
            &self.solver_rejection_receipt,
            self.outcome.status,
        ) {
            (Some(_), None, _) => true,
            (None, Some(receipt), AttemptStatus::Rejected) => {
                !is_zero_digest(receipt.as_str()) && self.outcome.correction.is_none()
            }
            (None, None, AttemptStatus::Inconclusive) => true,
            _ => false,
        };
        if self.observed_revision == 0
            || (self.lineage.ordinal == 0 && self.applied_correction.is_some())
            || !evidence_shape_valid
            || self.research_validation.as_ref().is_some_and(|evaluation| {
                evaluation.validate().is_err()
                    || evaluation.evaluated_candidate_digest
                        != self.bindings.solver.candidate_digest
            })
        {
            return Err(invalid("procedural_attempt_binding_invalid"));
        }
        if self.calculate_digest()? != self.attempt_digest {
            return Err(integrity("procedural_attempt_digest_mismatch"));
        }
        Ok(())
    }

    pub fn digest(&self) -> &ProceduralAttemptDigest {
        &self.attempt_digest
    }

    pub fn bindings(&self) -> &AttemptBindings {
        &self.bindings
    }

    pub fn configuration(&self) -> &SolverConfiguration {
        &self.configuration
    }

    pub fn outcome(&self) -> &AttemptOutcome {
        &self.outcome
    }

    /// Exact sealed lineage identity. Callers use this value when advancing a
    /// chain; deriving a replacement from the attempt digest would fork it.
    pub fn lineage_id(&self) -> &ProceduralLineageId {
        &self.lineage.lineage_id
    }

    pub fn lineage_ordinal(&self) -> u64 {
        self.lineage.ordinal
    }
}

fn configuration_matches_backend(
    configuration: &SolverConfiguration,
    evaluation: &CandidateEvaluation,
) -> bool {
    match evaluation.backend().role() {
        NumericalBackendRole::BuiltInCholeskyRidge => {
            configuration.family == SolverFamily::CholeskyRidgeLowRank
        }
        NumericalBackendRole::BuiltInDirectJacobiSvd => {
            configuration.family == SolverFamily::DirectJacobiSvd
        }
        NumericalBackendRole::ExternalProposal => {
            configuration.family == SolverFamily::ExternalProposal
                && matches!(
                    &configuration.parameters,
                    SolverParameters::External {
                        declared_implementation_digest,
                        ..
                    } if declared_implementation_digest
                        == &DeclaredBackendImplementationDigest::of_exact_bytes(
                            evaluation.backend().implementation().as_bytes(),
                        )
                )
        }
    }
}

fn failure_from_solver_rejection(reason: &EvaluationReason) -> BrainResult<FailureRecord> {
    let (kind, severity) = match reason {
        EvaluationReason::ResidualExceedsTolerance => {
            (FailureKind::ResidualTooLarge, FailureSeverity::Serious)
        }
        EvaluationReason::CholeskyNotAuthorized(
            CholeskyGate::RankDeficient
            | CholeskyGate::Degenerate
            | CholeskyGate::CaseCountInsufficient,
        )
        | EvaluationReason::RankBudgetInsufficient => {
            (FailureKind::RankInsufficient, FailureSeverity::Serious)
        }
        EvaluationReason::CholeskyNotAuthorized(CholeskyGate::ConditionExceedsPolicy) => {
            (FailureKind::IllConditioned, FailureSeverity::Serious)
        }
        EvaluationReason::CholeskyNotAuthorized(
            CholeskyGate::CaseLimitExceeded | CholeskyGate::Float32ConversionUnsafe,
        ) => (FailureKind::ResidencyMismatch, FailureSeverity::Recoverable),
        EvaluationReason::InvalidCandidate(_) => (
            FailureKind::InvariantRegression,
            FailureSeverity::HardInvariant,
        ),
        EvaluationReason::DirectSvdDidNotConverge => {
            (FailureKind::Divergence, FailureSeverity::Serious)
        }
        EvaluationReason::ResidualToleranceUnrepresentable => {
            (FailureKind::EvidenceInvalid, FailureSeverity::Serious)
        }
        EvaluationReason::BackendFailure(_) => {
            (FailureKind::EvidenceInvalid, FailureSeverity::Serious)
        }
        EvaluationReason::ResidualWithinTolerance
        | EvaluationReason::CholeskyNotAuthorized(CholeskyGate::Authorized)
        | EvaluationReason::ResourceLimit(_)
        | EvaluationReason::BackendNotApplicable(_)
        | EvaluationReason::BackendBoundedUnknown(_) => {
            return Err(integrity("procedural_solver_rejection_reason_mismatch"));
        }
    };
    Ok(FailureRecord::new(kind, FailureStage::Solve, severity))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum DriftKind {
    /// Non-causal observation: the same candidate and exact evaluation design
    /// occupied disjoint empirical intervals in non-overlapping windows.
    ObservedFunctionalChange,
}

/// Minimal scope proven by paired functional observations. It deliberately
/// excludes training-problem and solver-backend identity: those reports prove
/// change for one exact candidate under one research design, not its cause.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct DriftScope {
    system_envelope_digest: SystemEnvelopeDigest,
    capability_id: CapabilityId,
    capability_ir_digest: CapabilityIrDigest,
    base_artifact_digest: BaseArtifactDigest,
    target_profile_digest: TargetProfileDigest,
    candidate_digest: SolverCandidateDigest,
    evaluation_design_digest: EvaluationDesignDigest,
    evaluator_policy_digest: EvaluatorPolicyDigest,
}

impl DriftScope {
    fn from_attempt(attempt: &SolverAttempt) -> Option<Self> {
        let validation = attempt.research_validation.as_ref()?;
        Some(Self {
            system_envelope_digest: attempt.bindings.capability.system_envelope_digest.clone(),
            capability_id: attempt.bindings.capability.capability_id.clone(),
            capability_ir_digest: attempt.bindings.capability.capability_ir_digest.clone(),
            base_artifact_digest: attempt.bindings.capability.base_artifact_digest.clone(),
            target_profile_digest: attempt.bindings.capability.target_profile_digest.clone(),
            candidate_digest: attempt.bindings.solver.candidate_digest.clone(),
            evaluation_design_digest: validation.evaluation_design_digest.clone(),
            evaluator_policy_digest: validation.evaluator_policy_digest.clone(),
        })
    }

    fn validate(&self) -> BrainResult<()> {
        if is_zero_digest(self.system_envelope_digest.as_str())
            || is_zero_digest(self.capability_ir_digest.as_str())
            || is_zero_digest(self.base_artifact_digest.as_str())
            || is_zero_digest(self.target_profile_digest.as_str())
            || is_zero_digest(self.candidate_digest.as_str())
            || is_zero_digest(self.evaluation_design_digest.as_str())
            || is_zero_digest(self.evaluator_policy_digest.as_str())
        {
            return Err(invalid("procedural_drift_scope_invalid"));
        }
        Ok(())
    }
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct DriftProjection<'a> {
    schema: ProceduralSchema,
    scope: &'a DriftScope,
    kind: &'a DriftKind,
    detected_revision: u64,
    invalidates_through_revision: u64,
    previous_report_receipt: &'a ResearchReportReceiptDigest,
    current_report_receipt: &'a ResearchReportReceiptDigest,
    evidence_receipt_digest: &'a EvaluationReceiptDigest,
}

/// Evidence-bound invalidation. Attempts after `invalidates_through_revision`
/// remain eligible and therefore provide the required post-drift revalidation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DriftRecord {
    schema: ProceduralSchema,
    scope: DriftScope,
    kind: DriftKind,
    detected_revision: u64,
    invalidates_through_revision: u64,
    previous_report_receipt: ResearchReportReceiptDigest,
    current_report_receipt: ResearchReportReceiptDigest,
    evidence_receipt_digest: EvaluationReceiptDigest,
    drift_digest: DriftRecordDigest,
}

impl DriftRecord {
    fn seal(
        scope: DriftScope,
        kind: DriftKind,
        detected_revision: u64,
        invalidates_through_revision: u64,
        previous_report_receipt: ResearchReportReceiptDigest,
        current_report_receipt: ResearchReportReceiptDigest,
    ) -> BrainResult<Self> {
        let evidence_receipt_digest =
            observed_drift_receipt(&previous_report_receipt, &current_report_receipt)?;
        let mut value = Self {
            schema: ProceduralSchema::Current,
            scope,
            kind,
            detected_revision,
            invalidates_through_revision,
            previous_report_receipt,
            current_report_receipt,
            evidence_receipt_digest,
            drift_digest: DriftRecordDigest::computed(b"draft"),
        };
        value.drift_digest = value.calculate_digest()?;
        value.authenticate()?;
        Ok(value)
    }

    fn projection(&self) -> DriftProjection<'_> {
        DriftProjection {
            schema: self.schema,
            scope: &self.scope,
            kind: &self.kind,
            detected_revision: self.detected_revision,
            invalidates_through_revision: self.invalidates_through_revision,
            previous_report_receipt: &self.previous_report_receipt,
            current_report_receipt: &self.current_report_receipt,
            evidence_receipt_digest: &self.evidence_receipt_digest,
        }
    }

    fn calculate_digest(&self) -> BrainResult<DriftRecordDigest> {
        Ok(DriftRecordDigest::computed(&serde_json::to_vec(
            &self.projection(),
        )?))
    }

    pub fn authenticate(&self) -> BrainResult<()> {
        self.scope.validate()?;
        if self.detected_revision == 0
            || self.invalidates_through_revision == 0
            || self.invalidates_through_revision != self.detected_revision
            || self.previous_report_receipt == self.current_report_receipt
            || is_zero_digest(self.previous_report_receipt.as_str())
            || is_zero_digest(self.current_report_receipt.as_str())
            || is_zero_digest(self.evidence_receipt_digest.as_str())
        {
            return Err(invalid("procedural_drift_record_invalid"));
        }
        if self.calculate_digest()? != self.drift_digest {
            return Err(integrity("procedural_drift_digest_mismatch"));
        }
        if observed_drift_receipt(&self.previous_report_receipt, &self.current_report_receipt)?
            != self.evidence_receipt_digest
        {
            return Err(integrity("procedural_drift_receipt_mismatch"));
        }
        Ok(())
    }

    pub fn digest(&self) -> &DriftRecordDigest {
        &self.drift_digest
    }
}

fn observed_drift_receipt(
    previous: &ResearchReportReceiptDigest,
    current: &ResearchReportReceiptDigest,
) -> BrainResult<EvaluationReceiptDigest> {
    let mut receipt_frame = Vec::new();
    for digest in [previous.as_str(), current.as_str()] {
        let length = u64::try_from(digest.len())
            .map_err(|_| invalid("procedural_drift_receipt_length_overflow"))?;
        receipt_frame.extend_from_slice(&length.to_be_bytes());
        receipt_frame.extend_from_slice(digest.as_bytes());
    }
    Ok(EvaluationReceiptDigest::from(Sha256Digest::digest_domain(
        OBSERVED_DRIFT_EVIDENCE_DOMAIN,
        &receipt_frame,
    )))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionalDriftObservation {
    NoObservedChange,
    BoundedUnknown,
    Detected(Box<DriftRecord>),
}

/// The exact authority scope of a candidate-free failure. Portfolio preflight
/// failures have no selected backend and therefore cannot be falsely assigned
/// a [`SolverConfiguration`] or implementation identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "scope", rename_all = "snake_case", deny_unknown_fields)]
pub enum SolverRunSubject {
    Portfolio,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SolverResourceLimitKind {
    Cases,
    Dimensions,
    MaterializedElements,
    Memory,
    Compute,
    Time,
    NumericalRange,
    Unclassified,
}

/// A conclusive rejection has established that this exact solver run cannot
/// be accepted under its sealed policy. No candidate existed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SolverRunRejectionReason {
    DegenerateDesign,
    RankDeficientForRequiredContract,
    NonFiniteInput,
    PolicyConstraintViolated,
    NumericalInvariantViolated,
    EvidenceInvalid,
}

/// A bounded-unknown run did not establish success or failure. Keeping these
/// reasons separate prevents resource exhaustion or unavailable diagnostics
/// from being mislabeled as evidence that a solver is bad.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SolverRunUnknownReason {
    ResourceLimit(SolverResourceLimitKind),
    BackendUnavailable,
    BackendNotApplicable,
    DiagnosticUnavailable,
    ConvergenceNotEstablished,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", content = "reason", rename_all = "snake_case")]
pub enum SolverRunFailureOutcome {
    Rejected(SolverRunRejectionReason),
    BoundedUnknown(SolverRunUnknownReason),
}

/// Candidate-free context derived by the solver orchestration authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifiedSolverRunFailureDraft {
    capability: CapabilityContext,
    problem_digest: ExactSolverProblemDigest,
    solver_policy_digest: SolverPolicyDigest,
    applicability: Applicability,
    subject: SolverRunSubject,
}

impl VerifiedSolverRunFailureDraft {
    pub(crate) fn new(
        capability: CapabilityContext,
        problem: &LeastSquaresProblem,
        solver_policy_digest: SolverPolicyDigest,
        applicability: Applicability,
        subject: SolverRunSubject,
    ) -> BrainResult<Self> {
        capability.validate_nonzero()?;
        let problem_digest = problem.digest()?;
        let case_count = u64::try_from(problem.case_count())
            .map_err(|_| invalid("procedural_solver_run_case_count_overflow"))?;
        let input_dimension = u64::try_from(problem.input_dimension())
            .map_err(|_| invalid("procedural_solver_run_input_dimension_overflow"))?;
        if is_zero_digest(solver_policy_digest.as_str())
            || applicability.rows() != case_count
            || applicability.columns() != input_dimension
        {
            return Err(invalid("procedural_solver_run_context_invalid"));
        }
        applicability.validate()?;
        Ok(Self {
            capability,
            problem_digest,
            solver_policy_digest,
            applicability,
            subject,
        })
    }
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct SolverRunFailureProjection<'a> {
    schema: ProceduralSchema,
    capability: &'a CapabilityContext,
    problem_digest: &'a ExactSolverProblemDigest,
    solver_policy_digest: &'a SolverPolicyDigest,
    applicability: &'a Applicability,
    subject: &'a SolverRunSubject,
    outcome: &'a SolverRunFailureOutcome,
    observed_revision: u64,
    deterministic_work_units: Option<u64>,
    run_receipt_digest: &'a SolverRunReceiptDigest,
}

/// Immutable evidence for a solver run that ended before any candidate was
/// produced. It is intentionally separate from [`SolverAttempt`], so no fake
/// candidate identity or fake candidate metrics are ever required.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SolverRunFailureRecord {
    schema: ProceduralSchema,
    capability: CapabilityContext,
    problem_digest: ExactSolverProblemDigest,
    solver_policy_digest: SolverPolicyDigest,
    applicability: Applicability,
    subject: SolverRunSubject,
    outcome: SolverRunFailureOutcome,
    observed_revision: u64,
    deterministic_work_units: Option<u64>,
    run_receipt_digest: SolverRunReceiptDigest,
    failure_digest: SolverRunFailureDigest,
}

impl SolverRunFailureRecord {
    pub(crate) fn seal(
        draft: VerifiedSolverRunFailureDraft,
        observed_revision: u64,
        deterministic_work_units: Option<u64>,
        report: &SolverPortfolioReport,
    ) -> BrainResult<Self> {
        if report.problem_digest() != &draft.problem_digest
            || SolverPolicyDigest::bind_exact_digest(report.policy_digest().as_digest())
                != draft.solver_policy_digest
            || report.selected().is_some()
            || report
                .evaluations()
                .iter()
                .any(|evaluation| evaluation.candidate().is_some())
        {
            return Err(integrity("procedural_solver_run_report_binding_mismatch"));
        }
        let outcome = solver_run_failure_outcome(report)?;
        let run_receipt_digest = solver_run_receipt(report.exact_digest()?);
        let mut value = Self {
            schema: ProceduralSchema::Current,
            capability: draft.capability,
            problem_digest: draft.problem_digest,
            solver_policy_digest: draft.solver_policy_digest,
            applicability: draft.applicability,
            subject: draft.subject,
            outcome,
            observed_revision,
            deterministic_work_units,
            run_receipt_digest,
            failure_digest: SolverRunFailureDigest::computed(b"draft"),
        };
        value.failure_digest = value.calculate_digest()?;
        value.authenticate()?;
        Ok(value)
    }

    fn projection(&self) -> SolverRunFailureProjection<'_> {
        SolverRunFailureProjection {
            schema: self.schema,
            capability: &self.capability,
            problem_digest: &self.problem_digest,
            solver_policy_digest: &self.solver_policy_digest,
            applicability: &self.applicability,
            subject: &self.subject,
            outcome: &self.outcome,
            observed_revision: self.observed_revision,
            deterministic_work_units: self.deterministic_work_units,
            run_receipt_digest: &self.run_receipt_digest,
        }
    }

    fn calculate_digest(&self) -> BrainResult<SolverRunFailureDigest> {
        Ok(SolverRunFailureDigest::computed(&serde_json::to_vec(
            &self.projection(),
        )?))
    }

    pub fn authenticate(&self) -> BrainResult<()> {
        self.capability.validate_nonzero()?;
        if is_zero_digest(self.problem_digest.as_str())
            || is_zero_digest(self.solver_policy_digest.as_str())
            || is_zero_digest(self.run_receipt_digest.as_str())
            || self.observed_revision == 0
            || self
                .deterministic_work_units
                .is_some_and(|units| units == 0 || units > MAX_COMPUTE_UNITS)
        {
            return Err(invalid("procedural_solver_run_failure_invalid"));
        }
        self.applicability.validate()?;
        if self.calculate_digest()? != self.failure_digest {
            return Err(integrity("procedural_solver_run_failure_digest_mismatch"));
        }
        Ok(())
    }

    pub fn digest(&self) -> &SolverRunFailureDigest {
        &self.failure_digest
    }

    pub fn outcome(&self) -> &SolverRunFailureOutcome {
        &self.outcome
    }
}

fn solver_run_receipt(report_digest: &SolverPortfolioReportDigest) -> SolverRunReceiptDigest {
    SolverRunReceiptDigest::bind_exact_digest(report_digest.as_digest())
}

fn solver_run_failure_outcome(
    report: &SolverPortfolioReport,
) -> BrainResult<SolverRunFailureOutcome> {
    match report.status() {
        SolverStatus::Accepted => Err(invalid("procedural_solver_run_has_candidate")),
        SolverStatus::Rejected => {
            let reason = match report.reason() {
                EvaluationReason::CholeskyNotAuthorized(
                    CholeskyGate::RankDeficient | CholeskyGate::Degenerate,
                ) => SolverRunRejectionReason::RankDeficientForRequiredContract,
                EvaluationReason::InvalidCandidate(_) | EvaluationReason::BackendFailure(_) => {
                    SolverRunRejectionReason::NumericalInvariantViolated
                }
                EvaluationReason::ResidualExceedsTolerance
                | EvaluationReason::CholeskyNotAuthorized(_)
                | EvaluationReason::RankBudgetInsufficient
                | EvaluationReason::BackendNotApplicable(_)
                | EvaluationReason::ResidualWithinTolerance
                | EvaluationReason::ResourceLimit(_)
                | EvaluationReason::BackendBoundedUnknown(_)
                | EvaluationReason::ResidualToleranceUnrepresentable
                | EvaluationReason::DirectSvdDidNotConverge => {
                    SolverRunRejectionReason::PolicyConstraintViolated
                }
            };
            Ok(SolverRunFailureOutcome::Rejected(reason))
        }
        SolverStatus::BoundedUnknown => {
            let reason = match report.reason() {
                EvaluationReason::ResourceLimit(resource) => {
                    SolverRunUnknownReason::ResourceLimit(classify_solver_resource(resource))
                }
                EvaluationReason::DirectSvdDidNotConverge => {
                    SolverRunUnknownReason::ConvergenceNotEstablished
                }
                EvaluationReason::BackendNotApplicable(_) => {
                    SolverRunUnknownReason::BackendNotApplicable
                }
                EvaluationReason::BackendBoundedUnknown(_)
                | EvaluationReason::ResidualToleranceUnrepresentable
                | EvaluationReason::RankBudgetInsufficient => {
                    SolverRunUnknownReason::DiagnosticUnavailable
                }
                EvaluationReason::BackendFailure(_) => SolverRunUnknownReason::BackendUnavailable,
                EvaluationReason::CholeskyNotAuthorized(_)
                | EvaluationReason::ResidualWithinTolerance
                | EvaluationReason::ResidualExceedsTolerance
                | EvaluationReason::InvalidCandidate(_) => {
                    return Err(integrity("procedural_solver_run_status_reason_mismatch"));
                }
            };
            Ok(SolverRunFailureOutcome::BoundedUnknown(reason))
        }
    }
}

fn classify_solver_resource(resource: &str) -> SolverResourceLimitKind {
    match resource {
        "solver_max_cases" => SolverResourceLimitKind::Cases,
        "solver_max_input_dimension" | "solver_max_output_dimension" => {
            SolverResourceLimitKind::Dimensions
        }
        "solver_max_working_elements"
        | "solver_max_candidate_parameters"
        | "solver_max_total_external_parameters" => SolverResourceLimitKind::MaterializedElements,
        "solver_backend_budget"
        | "solver_max_svd_work_units"
        | "solver_max_candidate_evaluation_work_units" => SolverResourceLimitKind::Compute,
        "solver_max_absolute_value" | "solver_safe_gram_magnitude" => {
            SolverResourceLimitKind::NumericalRange
        }
        _ => SolverResourceLimitKind::Unclassified,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct IsolationKey {
    system_envelope_digest: SystemEnvelopeDigest,
    capability_id: CapabilityId,
    capability_ir_digest: CapabilityIrDigest,
    base_artifact_digest: BaseArtifactDigest,
    target_profile_digest: TargetProfileDigest,
    solver_policy_digest: SolverPolicyDigest,
}

impl IsolationKey {
    fn from_bindings(bindings: &AttemptBindings) -> Self {
        Self {
            system_envelope_digest: bindings.capability.system_envelope_digest.clone(),
            capability_id: bindings.capability.capability_id.clone(),
            capability_ir_digest: bindings.capability.capability_ir_digest.clone(),
            base_artifact_digest: bindings.capability.base_artifact_digest.clone(),
            target_profile_digest: bindings.capability.target_profile_digest.clone(),
            solver_policy_digest: bindings.solver.solver_policy_digest.clone(),
        }
    }
}

/// Exact current retrieval scope.  The problem digest remains separate from
/// the isolation key so an authority can explicitly request bounded analogical
/// transfer while project/capability/base/target/policy remain exact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalScope {
    isolation: IsolationKey,
    problem_digest: ExactSolverProblemDigest,
}

impl RetrievalScope {
    pub fn from_bindings(bindings: &AttemptBindings) -> Self {
        Self {
            isolation: IsolationKey::from_bindings(bindings),
            problem_digest: bindings.solver.problem_digest.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProblemTransferPolicy {
    ExactOnly,
    FeatureBounded { minimum_similarity: FiniteF64 },
}

impl ProblemTransferPolicy {
    pub fn feature_bounded(minimum_similarity: f64) -> BrainResult<Self> {
        let minimum_similarity = canonical_finite(minimum_similarity)?;
        if !unit_interval(minimum_similarity) || minimum_similarity.get() == 0.0 {
            return Err(invalid("procedural_transfer_similarity_invalid"));
        }
        Ok(Self::FeatureBounded { minimum_similarity })
    }

    fn validate(&self) -> BrainResult<()> {
        if let Self::FeatureBounded { minimum_similarity } = self {
            if !unit_interval(*minimum_similarity)
                || minimum_similarity.get() == 0.0
                || !is_canonical_finite(*minimum_similarity)
            {
                return Err(invalid("procedural_transfer_similarity_invalid"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RetrievalPolicyProjection {
    schema: ProceduralSchema,
    version: u16,
    row_weight: FiniteF64,
    column_weight: FiniteF64,
    rank_weight: FiniteF64,
    condition_weight: FiniteF64,
    outlier_weight: FiniteF64,
    sparsity_weight: FiniteF64,
    noise_weight: FiniteF64,
    precision_weight: FiniteF64,
    structure_weight: FiniteF64,
    evidence_repetition_scale: FiniteF64,
    confidence_scale: FiniteF64,
    positive_threshold: FiniteF64,
    negative_threshold: FiniteF64,
    single_group_confidence_cap: FiniteF64,
    negative_advice_never_blocks: bool,
}

/// Closed, versioned interpretation policy for historical procedural records.
/// Changing any coefficient requires a new version and therefore a new digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProceduralRetrievalPolicy {
    projection: RetrievalPolicyProjection,
    digest: ProceduralRetrievalPolicyDigest,
}

impl ProceduralRetrievalPolicy {
    pub fn current() -> BrainResult<Self> {
        let projection = RetrievalPolicyProjection {
            schema: ProceduralSchema::Current,
            version: 4,
            row_weight: FiniteF64::new(0.12)?,
            column_weight: FiniteF64::new(0.12)?,
            rank_weight: FiniteF64::new(0.16)?,
            condition_weight: FiniteF64::new(0.16)?,
            outlier_weight: FiniteF64::new(0.10)?,
            sparsity_weight: FiniteF64::new(0.10)?,
            noise_weight: FiniteF64::new(0.08)?,
            precision_weight: FiniteF64::new(0.08)?,
            structure_weight: FiniteF64::new(0.08)?,
            evidence_repetition_scale: FiniteF64::new(8.0)?,
            confidence_scale: FiniteF64::new(3.0)?,
            positive_threshold: FiniteF64::new(0.15)?,
            negative_threshold: FiniteF64::new(-0.15)?,
            single_group_confidence_cap: FiniteF64::new(0.35)?,
            negative_advice_never_blocks: true,
        };
        let digest = ProceduralRetrievalPolicyDigest::computed(&serde_json::to_vec(&projection)?);
        Ok(Self { projection, digest })
    }

    pub fn digest(&self) -> &ProceduralRetrievalPolicyDigest {
        &self.digest
    }

    fn validate(&self) -> BrainResult<()> {
        let expected = Self::current()?;
        if self != &expected {
            return Err(integrity("procedural_retrieval_policy_not_current"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalQuery {
    scope: RetrievalScope,
    applicability: Applicability,
    current_revision: u64,
    transfer_policy: ProblemTransferPolicy,
    max_results: usize,
    retrieval_policy: ProceduralRetrievalPolicy,
}

impl RetrievalQuery {
    pub fn new(
        scope: RetrievalScope,
        applicability: Applicability,
        current_revision: u64,
        transfer_policy: ProblemTransferPolicy,
        max_results: usize,
    ) -> BrainResult<Self> {
        let value = Self {
            scope,
            applicability,
            current_revision,
            transfer_policy,
            max_results,
            retrieval_policy: ProceduralRetrievalPolicy::current()?,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> BrainResult<()> {
        self.applicability.validate()?;
        self.transfer_policy.validate()?;
        self.retrieval_policy.validate()?;
        if self.current_revision == 0
            || self.max_results == 0
            || self.max_results > MAX_ADVICE_RESULTS
            || is_zero_digest(self.scope.problem_digest.as_str())
        {
            return Err(invalid("procedural_retrieval_query_invalid"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum AdviceDisposition {
    PrioritizeExploration,
    ObserveAsControl,
    DeprioritizeButRetainControl,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum AdviceReason {
    ExactProblemEvidence,
    BoundedFeatureTransfer,
    ResearchValidatedOutcome,
    ResearchNegativeOutcome,
    SolverRejectedCandidate,
    CorrectionObserved,
    DistinctResearchGroups,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdviceExplanation {
    /// Conservative lower-bound proxy: the largest authenticated group count
    /// in any one design. Counts from different designs are never summed
    /// because their underlying membership may overlap.
    conservatively_counted_independent_groups: usize,
    validated_attempts: usize,
    negative_attempts: usize,
    mean_context_similarity: FiniteF64,
    positive_weight: FiniteF64,
    negative_weight: FiniteF64,
    reasons: BTreeSet<AdviceReason>,
    suggested_corrections: BTreeSet<CorrectionKind>,
}

/// Bounded procedural guidance.  This type intentionally has no conversion to
/// any promotion/residency decision and explicitly reports that fact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[must_use]
pub struct ProceduralAdvice {
    retrieval_policy_digest: ProceduralRetrievalPolicyDigest,
    configuration_digest: SolverConfigurationDigest,
    configuration: SolverConfiguration,
    disposition: AdviceDisposition,
    priority_score: FiniteF64,
    confidence: FiniteF64,
    explanation: AdviceExplanation,
}

impl ProceduralAdvice {
    pub fn configuration(&self) -> &SolverConfiguration {
        &self.configuration
    }

    pub fn disposition(&self) -> AdviceDisposition {
        self.disposition
    }

    pub fn priority_score(&self) -> f64 {
        self.priority_score.get()
    }

    /// Procedural memory is advisory-only by construction.
    pub const fn authorizes_promotion(&self) -> bool {
        false
    }

    /// Even strongly negative memory cannot remove the planner's baseline or
    /// exploration control. It may only change ordering and budget priority.
    pub const fn requires_independent_control(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RetrievalReport {
    advice: Vec<ProceduralAdvice>,
    solver_run_cautions: Vec<SolverRunCaution>,
    considered_records: usize,
    scope_filtered_records: usize,
    evidence_filtered_records: usize,
    drift_filtered_records: usize,
    applicability_filtered_records: usize,
    solver_run_failure_filtered_records: usize,
}

impl RetrievalReport {
    pub fn advice(&self) -> &[ProceduralAdvice] {
        &self.advice
    }

    pub fn evidence_filtered_records(&self) -> usize {
        self.evidence_filtered_records
    }

    pub fn drift_filtered_records(&self) -> usize {
        self.drift_filtered_records
    }

    /// Exact-policy cautions from runs that produced no candidate. These are
    /// separate from configuration advice because attributing a portfolio
    /// preflight failure to one solver configuration would be false.
    pub fn solver_run_cautions(&self) -> &[SolverRunCaution] {
        &self.solver_run_cautions
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[must_use]
pub struct SolverRunCaution {
    run_receipt_digest: SolverRunReceiptDigest,
    observed_revision: u64,
    outcome: SolverRunFailureOutcome,
    disposition: AdviceDisposition,
}

impl SolverRunCaution {
    pub fn disposition(&self) -> AdviceDisposition {
        self.disposition
    }

    pub fn outcome(&self) -> &SolverRunFailureOutcome {
        &self.outcome
    }

    pub const fn authorizes_promotion(&self) -> bool {
        false
    }

    pub const fn requires_independent_control(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordDisposition {
    Inserted,
    AlreadyPresent,
}

#[derive(Debug, Clone)]
struct Contribution {
    attempt_digest: ProceduralAttemptDigest,
    revision: u64,
    exact_problem: bool,
    similarity: f64,
    evidence_strength: f64,
    independent_group_count: u32,
    evidence_source: ContributionEvidence,
    status: AttemptStatus,
    positive_utility: f64,
    failure_penalty: f64,
    correction: Option<CorrectionKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContributionEvidence {
    ResearchValidation,
    SolverRejection,
}

#[derive(Debug, Default)]
struct AdviceAccumulator {
    configuration: Option<SolverConfiguration>,
    contributions: BTreeMap<EvaluationDesignDigest, Contribution>,
}

/// Bounded, non-persistent reducer. All maps below are derived indexes; no map
/// is serialized or accepted as authority input.
#[derive(Debug, Default)]
pub struct ProceduralMemory {
    attempts: BTreeMap<ProceduralAttemptDigest, SolverAttempt>,
    solver_run_failures: BTreeMap<SolverRunFailureDigest, SolverRunFailureRecord>,
    attempts_by_isolation: BTreeMap<IsolationKey, BTreeSet<ProceduralAttemptDigest>>,
    evidence_receipts: BTreeSet<EvaluationReceiptDigest>,
    research_report_receipts: BTreeMap<ResearchReportReceiptDigest, ProceduralAttemptDigest>,
    solver_candidate_rejection_receipts:
        BTreeMap<SolverCandidateRejectionReceiptDigest, ProceduralAttemptDigest>,
    solver_run_receipts: BTreeMap<SolverRunReceiptDigest, SolverRunFailureDigest>,
    /// A design digest commits the exact independent-group set. Its declared
    /// cardinality is immutable across every candidate and evaluator replay.
    evaluation_design_sizes: BTreeMap<EvaluationDesignDigest, u32>,
    lineage_positions: BTreeMap<(ProceduralLineageId, u64), ProceduralAttemptDigest>,
    drift_records: BTreeMap<DriftRecordDigest, DriftRecord>,
    invalidated_through: BTreeMap<DriftScope, u64>,
}

impl ProceduralMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn attempt_count(&self) -> usize {
        self.attempts.len()
    }

    pub fn drift_count(&self) -> usize {
        self.drift_records.len()
    }

    pub fn solver_run_failure_count(&self) -> usize {
        self.solver_run_failures.len()
    }

    /// Idempotently reduce one authenticated attempt into disposable indexes.
    pub(crate) fn record_attempt(
        &mut self,
        attempt: SolverAttempt,
    ) -> BrainResult<RecordDisposition> {
        attempt.authenticate()?;
        if let Some(existing) = self.attempts.get(attempt.digest()) {
            return if existing == &attempt {
                Ok(RecordDisposition::AlreadyPresent)
            } else {
                Err(integrity("procedural_attempt_digest_collision"))
            };
        }
        if let Some(evaluation) = &attempt.research_validation {
            if let Some(existing_digest) = self
                .research_report_receipts
                .get(&evaluation.report_receipt_digest)
            {
                let existing = self
                    .attempts
                    .get(existing_digest)
                    .ok_or_else(|| integrity("procedural_research_receipt_index_dangling"))?;
                return if attempt_evidence_semantics_equal(existing, &attempt) {
                    Ok(RecordDisposition::AlreadyPresent)
                } else {
                    Err(integrity("procedural_research_report_relabel"))
                };
            }
            if self.evidence_receipts.contains(&evaluation.receipt_digest) {
                return Err(integrity("procedural_evaluation_receipt_relabel"));
            }
            if self
                .evaluation_design_sizes
                .get(&evaluation.evaluation_design_digest)
                .is_some_and(|count| *count != evaluation.independent_group_count)
            {
                return Err(integrity("procedural_evaluation_design_relabel"));
            }
        }
        if let Some(receipt) = &attempt.solver_rejection_receipt {
            if let Some(existing_digest) = self.solver_candidate_rejection_receipts.get(receipt) {
                let existing = self.attempts.get(existing_digest).ok_or_else(|| {
                    integrity("procedural_solver_rejection_receipt_index_dangling")
                })?;
                return if attempt_evidence_semantics_equal(existing, &attempt) {
                    Ok(RecordDisposition::AlreadyPresent)
                } else {
                    Err(integrity("procedural_solver_rejection_receipt_relabel"))
                };
            }
        }
        if self.attempts.len() >= MAX_ATTEMPTS {
            return Err(invalid("procedural_attempt_capacity_exceeded"));
        }
        self.validate_lineage(&attempt)?;

        let attempt_digest = attempt.attempt_digest.clone();
        let isolation = IsolationKey::from_bindings(&attempt.bindings);
        if let Some(evaluation) = &attempt.research_validation {
            self.evidence_receipts
                .insert(evaluation.receipt_digest.clone());
            self.research_report_receipts.insert(
                evaluation.report_receipt_digest.clone(),
                attempt_digest.clone(),
            );
            self.evaluation_design_sizes
                .entry(evaluation.evaluation_design_digest.clone())
                .or_insert(evaluation.independent_group_count);
        }
        if let Some(receipt) = &attempt.solver_rejection_receipt {
            self.solver_candidate_rejection_receipts
                .insert(receipt.clone(), attempt_digest.clone());
        }
        self.lineage_positions.insert(
            (attempt.lineage.lineage_id.clone(), attempt.lineage.ordinal),
            attempt_digest.clone(),
        );
        self.attempts_by_isolation
            .entry(isolation)
            .or_default()
            .insert(attempt_digest.clone());
        self.attempts.insert(attempt_digest, attempt);
        Ok(RecordDisposition::Inserted)
    }

    fn validate_lineage(&self, attempt: &SolverAttempt) -> BrainResult<()> {
        if self
            .lineage_positions
            .contains_key(&(attempt.lineage.lineage_id.clone(), attempt.lineage.ordinal))
        {
            return Err(integrity("procedural_lineage_position_fork"));
        }
        match (
            attempt.lineage.ordinal,
            &attempt.lineage.parent_attempt_digest,
        ) {
            (0, None) => Ok(()),
            (0, Some(_)) | (_, None) => Err(invalid("procedural_lineage_shape_invalid")),
            (ordinal, Some(parent_digest)) => {
                let parent = self
                    .attempts
                    .get(parent_digest)
                    .ok_or_else(|| integrity("procedural_lineage_parent_missing"))?;
                if parent.lineage.lineage_id != attempt.lineage.lineage_id
                    || parent.lineage.ordinal.checked_add(1) != Some(ordinal)
                    || parent.observed_revision >= attempt.observed_revision
                    || IsolationKey::from_bindings(&parent.bindings)
                        != IsolationKey::from_bindings(&attempt.bindings)
                    || parent.bindings.solver.problem_digest
                        != attempt.bindings.solver.problem_digest
                {
                    return Err(integrity("procedural_lineage_parent_mismatch"));
                }
                if let Some(applied) = attempt.applied_correction {
                    let proposed = parent
                        .outcome
                        .correction
                        .as_ref()
                        .is_some_and(|correction| {
                            correction.kind == applied
                                && correction.status != CorrectionStatus::Refuted
                        });
                    if !proposed {
                        return Err(integrity("procedural_lineage_correction_mismatch"));
                    }
                    if !correction_is_realized(parent, attempt, applied) {
                        return Err(integrity("procedural_lineage_correction_not_realized"));
                    }
                }
                Ok(())
            }
        }
    }

    pub(crate) fn record_drift(&mut self, drift: DriftRecord) -> BrainResult<RecordDisposition> {
        drift.authenticate()?;
        if let Some(existing) = self.drift_records.get(drift.digest()) {
            return if existing == &drift {
                Ok(RecordDisposition::AlreadyPresent)
            } else {
                Err(integrity("procedural_drift_digest_collision"))
            };
        }
        if self.drift_records.len() >= MAX_DRIFT_RECORDS {
            return Err(invalid("procedural_drift_capacity_exceeded"));
        }
        let previous = self.attempt_for_report_receipt(&drift.previous_report_receipt)?;
        let current = self.attempt_for_report_receipt(&drift.current_report_receipt)?;
        let previous_scope = DriftScope::from_attempt(previous)
            .ok_or_else(|| integrity("procedural_drift_previous_validation_missing"))?;
        let current_scope = DriftScope::from_attempt(current)
            .ok_or_else(|| integrity("procedural_drift_current_validation_missing"))?;
        if previous_scope != drift.scope
            || current_scope != drift.scope
            || previous.observed_revision >= current.observed_revision
            || drift.detected_revision != current.observed_revision
            || drift.invalidates_through_revision != current.observed_revision
        {
            return Err(integrity("procedural_drift_revision_binding_mismatch"));
        }
        if self
            .evidence_receipts
            .contains(&drift.evidence_receipt_digest)
        {
            return Err(integrity("procedural_drift_receipt_relabel"));
        }
        self.invalidated_through
            .entry(drift.scope.clone())
            .and_modify(|revision| *revision = (*revision).max(drift.invalidates_through_revision))
            .or_insert(drift.invalidates_through_revision);
        self.evidence_receipts
            .insert(drift.evidence_receipt_digest.clone());
        self.drift_records.insert(drift.drift_digest.clone(), drift);
        Ok(RecordDisposition::Inserted)
    }

    fn attempt_for_report_receipt(
        &self,
        receipt: &ResearchReportReceiptDigest,
    ) -> BrainResult<&SolverAttempt> {
        let attempt_digest = self
            .research_report_receipts
            .get(receipt)
            .ok_or_else(|| integrity("procedural_drift_report_not_registered"))?;
        self.attempts
            .get(attempt_digest)
            .ok_or_else(|| integrity("procedural_drift_report_index_dangling"))
    }

    /// Derive only non-causal functional drift from reports already reduced
    /// into this memory. Revisions and invalidation boundaries come from the
    /// registered report-to-attempt index; callers cannot choose either.
    pub(crate) fn derive_functional_change(
        &self,
        candidate_binding: &VerifiedCandidateVariantBinding,
        previous: &PairedEvaluationReport,
        current: &PairedEvaluationReport,
    ) -> BrainResult<FunctionalDriftObservation> {
        if previous.candidate_id() != &candidate_binding.variant_id
            || current.candidate_id() != &candidate_binding.variant_id
            || previous.independence_design_digest() != current.independence_design_digest()
            || previous.metric_catalog_digest() != current.metric_catalog_digest()
            || previous.evaluation_policy_digest() != current.evaluation_policy_digest()
            || previous.independent_group_count() != current.independent_group_count()
            || previous.estimates().keys().ne(current.estimates().keys())
            || current.observation_window().start_tick() <= previous.observation_window().end_tick()
        {
            return Err(integrity("procedural_drift_report_binding_mismatch"));
        }
        if !previous.evidence_sufficient() || !current.evidence_sufficient() {
            return Ok(FunctionalDriftObservation::BoundedUnknown);
        }

        let mut missing_interval = false;
        let mut observed_change = false;
        for (metric_id, earlier) in previous.estimates() {
            let later = current
                .estimates()
                .get(metric_id)
                .ok_or_else(|| integrity("procedural_drift_metric_missing"))?;
            match (
                earlier.candidate_observed_radius(),
                later.candidate_observed_radius(),
            ) {
                (Some(earlier_radius), Some(later_radius)) => {
                    let earlier_lower = earlier.candidate_center() - earlier_radius;
                    let earlier_upper = earlier.candidate_center() + earlier_radius;
                    let later_lower = later.candidate_center() - later_radius;
                    let later_upper = later.candidate_center() + later_radius;
                    observed_change |= earlier_upper < later_lower || later_upper < earlier_lower;
                }
                _ => missing_interval = true,
            }
        }
        if !observed_change {
            return Ok(if missing_interval {
                FunctionalDriftObservation::BoundedUnknown
            } else {
                FunctionalDriftObservation::NoObservedChange
            });
        }

        let previous_receipt = research_report_receipt(previous);
        let current_receipt = research_report_receipt(current);
        let previous_attempt = match self.research_report_receipts.get(&previous_receipt) {
            Some(attempt) => self
                .attempts
                .get(attempt)
                .ok_or_else(|| integrity("procedural_drift_report_index_dangling"))?,
            None => return Ok(FunctionalDriftObservation::BoundedUnknown),
        };
        let current_attempt = match self.research_report_receipts.get(&current_receipt) {
            Some(attempt) => self
                .attempts
                .get(attempt)
                .ok_or_else(|| integrity("procedural_drift_report_index_dangling"))?,
            None => return Ok(FunctionalDriftObservation::BoundedUnknown),
        };
        if previous_attempt.bindings.solver.candidate_digest != candidate_binding.candidate_digest
            || current_attempt.bindings.solver.candidate_digest
                != candidate_binding.candidate_digest
            || previous_attempt.observed_revision >= current_attempt.observed_revision
        {
            return Err(integrity("procedural_drift_registered_binding_mismatch"));
        }
        let previous_scope = DriftScope::from_attempt(previous_attempt)
            .ok_or_else(|| integrity("procedural_drift_previous_validation_missing"))?;
        let current_scope = DriftScope::from_attempt(current_attempt)
            .ok_or_else(|| integrity("procedural_drift_current_validation_missing"))?;
        if previous_scope != current_scope
            || previous_scope.evaluation_design_digest != research_evaluation_design(previous)?
            || current_scope.evaluation_design_digest != research_evaluation_design(current)?
        {
            return Err(integrity("procedural_drift_scope_binding_mismatch"));
        }
        Ok(FunctionalDriftObservation::Detected(Box::new(
            DriftRecord::seal(
                previous_scope,
                DriftKind::ObservedFunctionalChange,
                current_attempt.observed_revision,
                current_attempt.observed_revision,
                previous_receipt,
                current_receipt,
            )?,
        )))
    }

    /// Preserve a candidate-free solver failure without allowing it to become
    /// candidate evidence or promotion authority.
    pub(crate) fn record_solver_run_failure(
        &mut self,
        failure: SolverRunFailureRecord,
    ) -> BrainResult<RecordDisposition> {
        failure.authenticate()?;
        if let Some(existing) = self.solver_run_failures.get(failure.digest()) {
            return if existing == &failure {
                Ok(RecordDisposition::AlreadyPresent)
            } else {
                Err(integrity("procedural_solver_run_failure_digest_collision"))
            };
        }
        if self.solver_run_failures.len() >= MAX_SOLVER_RUN_FAILURES {
            return Err(invalid("procedural_solver_run_failure_capacity_exceeded"));
        }
        if let Some(existing_digest) = self.solver_run_receipts.get(&failure.run_receipt_digest) {
            let existing = self
                .solver_run_failures
                .get(existing_digest)
                .ok_or_else(|| integrity("procedural_solver_run_receipt_index_dangling"))?;
            return if solver_run_receipt_semantics_equal(existing, &failure) {
                Ok(RecordDisposition::AlreadyPresent)
            } else {
                Err(integrity("procedural_solver_run_receipt_relabel"))
            };
        }
        self.solver_run_receipts.insert(
            failure.run_receipt_digest.clone(),
            failure.failure_digest.clone(),
        );
        self.solver_run_failures
            .insert(failure.failure_digest.clone(), failure);
        Ok(RecordDisposition::Inserted)
    }

    /// Canonical replay: input order cannot change the resulting reducer state.
    pub(crate) fn rebuild(
        attempts: Vec<SolverAttempt>,
        drifts: Vec<DriftRecord>,
    ) -> BrainResult<Self> {
        Self::rebuild_with_solver_failures(attempts, Vec::new(), drifts)
    }

    /// Canonical replay including failures that occurred before candidate
    /// materialization.
    pub(crate) fn rebuild_with_solver_failures(
        mut attempts: Vec<SolverAttempt>,
        mut solver_run_failures: Vec<SolverRunFailureRecord>,
        mut drifts: Vec<DriftRecord>,
    ) -> BrainResult<Self> {
        if attempts.len() > MAX_ATTEMPTS || drifts.len() > MAX_DRIFT_RECORDS {
            return Err(invalid("procedural_rebuild_capacity_exceeded"));
        }
        if solver_run_failures.len() > MAX_SOLVER_RUN_FAILURES {
            return Err(invalid("procedural_rebuild_capacity_exceeded"));
        }
        attempts.sort_by(|left, right| {
            left.lineage
                .lineage_id
                .cmp(&right.lineage.lineage_id)
                .then(left.lineage.ordinal.cmp(&right.lineage.ordinal))
                .then(left.attempt_digest.cmp(&right.attempt_digest))
        });
        solver_run_failures.sort_by(|left, right| left.failure_digest.cmp(&right.failure_digest));
        drifts.sort_by(|left, right| left.drift_digest.cmp(&right.drift_digest));
        let mut memory = Self::new();
        for attempt in attempts {
            memory.record_attempt(attempt)?;
        }
        for failure in solver_run_failures {
            memory.record_solver_run_failure(failure)?;
        }
        for drift in drifts {
            memory.record_drift(drift)?;
        }
        Ok(memory)
    }

    pub fn retrieve(&self, query: &RetrievalQuery) -> BrainResult<RetrievalReport> {
        query.validate()?;
        let mut report = RetrievalReport {
            advice: Vec::new(),
            solver_run_cautions: Vec::new(),
            considered_records: 0,
            scope_filtered_records: self.attempts.len(),
            evidence_filtered_records: 0,
            drift_filtered_records: 0,
            applicability_filtered_records: 0,
            solver_run_failure_filtered_records: self.solver_run_failures.len(),
        };
        for failure in self.solver_run_failures.values() {
            let isolation = IsolationKey {
                system_envelope_digest: failure.capability.system_envelope_digest.clone(),
                capability_id: failure.capability.capability_id.clone(),
                capability_ir_digest: failure.capability.capability_ir_digest.clone(),
                base_artifact_digest: failure.capability.base_artifact_digest.clone(),
                target_profile_digest: failure.capability.target_profile_digest.clone(),
                solver_policy_digest: failure.solver_policy_digest.clone(),
            };
            if isolation != query.scope.isolation
                || failure.problem_digest != query.scope.problem_digest
                || failure.observed_revision > query.current_revision
            {
                continue;
            }
            let disposition = match failure.outcome {
                SolverRunFailureOutcome::Rejected(_) => {
                    AdviceDisposition::DeprioritizeButRetainControl
                }
                SolverRunFailureOutcome::BoundedUnknown(_) => AdviceDisposition::ObserveAsControl,
            };
            report.solver_run_cautions.push(SolverRunCaution {
                run_receipt_digest: failure.run_receipt_digest.clone(),
                observed_revision: failure.observed_revision,
                outcome: failure.outcome.clone(),
                disposition,
            });
        }
        report.solver_run_failure_filtered_records = self
            .solver_run_failures
            .len()
            .saturating_sub(report.solver_run_cautions.len());
        report.solver_run_cautions.sort_by(|left, right| {
            right
                .observed_revision
                .cmp(&left.observed_revision)
                .then(left.run_receipt_digest.cmp(&right.run_receipt_digest))
        });
        report.solver_run_cautions.truncate(query.max_results);
        let Some(attempt_ids) = self.attempts_by_isolation.get(&query.scope.isolation) else {
            return Ok(report);
        };
        report.scope_filtered_records = self.attempts.len().saturating_sub(attempt_ids.len());
        let mut accumulators = BTreeMap::<SolverConfigurationDigest, AdviceAccumulator>::new();

        for attempt_id in attempt_ids {
            report.considered_records += 1;
            let attempt = self
                .attempts
                .get(attempt_id)
                .ok_or_else(|| integrity("procedural_derived_index_dangling"))?;
            if attempt.observed_revision > query.current_revision {
                report.applicability_filtered_records += 1;
                continue;
            }
            if DriftScope::from_attempt(attempt).is_some_and(|drift_scope| {
                self.invalidated_through
                    .get(&drift_scope)
                    .is_some_and(|revision| *revision >= attempt.observed_revision)
            }) {
                report.drift_filtered_records += 1;
                continue;
            }
            // A declared external implementation label is not an attestation
            // of the code that produced the candidate. Preserve the record,
            // but do not emit transferable advice until an implementation
            // receipt type exists and is independently verified.
            if attempt.configuration.family == SolverFamily::ExternalProposal {
                report.evidence_filtered_records += 1;
                continue;
            }
            if attempt.outcome.status == AttemptStatus::Inconclusive {
                report.evidence_filtered_records += 1;
                continue;
            }

            let exact_problem =
                attempt.bindings.solver.problem_digest == query.scope.problem_digest;
            let (evidence_design, evidence_strength, independent_group_count, evidence_source) =
                match (
                    &attempt.research_validation,
                    &attempt.solver_rejection_receipt,
                ) {
                    (Some(evaluation), None) => (
                        evaluation.evaluation_design_digest.clone(),
                        1.0 - (-(evaluation.independent_group_count as f64)
                            / query
                                .retrieval_policy
                                .projection
                                .evidence_repetition_scale
                                .get())
                        .exp(),
                        evaluation.independent_group_count,
                        ContributionEvidence::ResearchValidation,
                    ),
                    (None, Some(receipt)) if exact_problem => (
                        EvaluationDesignDigest::of_exact_bytes(receipt.as_str().as_bytes()),
                        0.5,
                        0,
                        ContributionEvidence::SolverRejection,
                    ),
                    _ => {
                        report.evidence_filtered_records += 1;
                        continue;
                    }
                };
            let similarity = if exact_problem {
                1.0
            } else {
                match &query.transfer_policy {
                    ProblemTransferPolicy::ExactOnly => {
                        report.applicability_filtered_records += 1;
                        continue;
                    }
                    ProblemTransferPolicy::FeatureBounded { minimum_similarity } => {
                        let similarity = applicability_similarity(
                            &attempt.applicability,
                            &query.applicability,
                            &query.retrieval_policy,
                        )?;
                        if similarity < minimum_similarity.get() {
                            report.applicability_filtered_records += 1;
                            continue;
                        }
                        similarity
                    }
                }
            };

            let configuration_digest = attempt.configuration.digest()?;
            let positive_utility = if attempt.outcome.status == AttemptStatus::Validated {
                attempt
                    .outcome
                    .metrics
                    .positive_utility()
                    .ok_or_else(|| integrity("procedural_validated_metrics_incomplete"))?
            } else {
                0.0
            };
            let contribution = Contribution {
                attempt_digest: attempt.attempt_digest.clone(),
                revision: attempt.observed_revision,
                exact_problem,
                similarity,
                evidence_strength,
                independent_group_count,
                evidence_source,
                status: attempt.outcome.status,
                positive_utility,
                failure_penalty: attempt
                    .outcome
                    .failure
                    .as_ref()
                    .map_or(0.0, |failure| failure.severity.penalty()),
                correction: attempt.outcome.correction.as_ref().and_then(|value| {
                    (value.status != CorrectionStatus::Refuted).then_some(value.kind)
                }),
            };
            let accumulator = accumulators.entry(configuration_digest).or_default();
            accumulator
                .configuration
                .get_or_insert_with(|| attempt.configuration.clone());
            let replace = accumulator
                .contributions
                .get(&evidence_design)
                .is_none_or(|existing| {
                    contribution.revision > existing.revision
                        || (contribution.revision == existing.revision
                            && contribution.attempt_digest < existing.attempt_digest)
                });
            if replace {
                accumulator
                    .contributions
                    .insert(evidence_design, contribution);
            }
        }

        for (configuration_digest, accumulator) in accumulators {
            let Some(configuration) = accumulator.configuration else {
                continue;
            };
            let mut positive_weight = 0.0;
            let mut negative_weight = 0.0;
            let mut utility_sum = 0.0;
            let mut similarity_sum = 0.0;
            let mut validated_attempts = 0;
            let mut negative_attempts = 0;
            let mut exact_count = 0;
            let mut conservative_group_count = 0_u32;
            let mut confidence_evidence = 0.0_f64;
            let mut reasons = BTreeSet::new();
            let mut corrections = BTreeSet::new();
            for contribution in accumulator.contributions.values() {
                let weight = contribution.similarity * contribution.evidence_strength;
                conservative_group_count =
                    conservative_group_count.max(contribution.independent_group_count);
                let confidence_units = match contribution.evidence_source {
                    ContributionEvidence::ResearchValidation => {
                        f64::from(contribution.independent_group_count)
                    }
                    ContributionEvidence::SolverRejection => 1.0,
                };
                confidence_evidence =
                    confidence_evidence.max(contribution.similarity * confidence_units);
                similarity_sum += contribution.similarity;
                if contribution.exact_problem {
                    exact_count += 1;
                }
                match contribution.status {
                    AttemptStatus::Validated => {
                        validated_attempts += 1;
                        positive_weight += weight;
                        utility_sum += weight * contribution.positive_utility;
                        reasons.insert(AdviceReason::ResearchValidatedOutcome);
                    }
                    AttemptStatus::Rejected | AttemptStatus::Failed => {
                        negative_attempts += 1;
                        negative_weight += weight;
                        utility_sum -= weight * contribution.failure_penalty;
                        reasons.insert(match contribution.evidence_source {
                            ContributionEvidence::ResearchValidation => {
                                AdviceReason::ResearchNegativeOutcome
                            }
                            ContributionEvidence::SolverRejection => {
                                AdviceReason::SolverRejectedCandidate
                            }
                        });
                    }
                    AttemptStatus::Inconclusive => {}
                }
                if let Some(correction) = contribution.correction {
                    corrections.insert(correction);
                    reasons.insert(AdviceReason::CorrectionObserved);
                }
            }
            let total_weight = positive_weight + negative_weight;
            if total_weight <= f64::EPSILON {
                continue;
            }
            if exact_count > 0 {
                reasons.insert(AdviceReason::ExactProblemEvidence);
            }
            if exact_count < accumulator.contributions.len() {
                reasons.insert(AdviceReason::BoundedFeatureTransfer);
            }
            if conservative_group_count >= 2 {
                reasons.insert(AdviceReason::DistinctResearchGroups);
            }
            let signed_evidence = (utility_sum / total_weight).clamp(-1.0, 1.0);
            let mut confidence = (1.0
                - (-confidence_evidence
                    / query.retrieval_policy.projection.confidence_scale.get())
                .exp())
            .clamp(0.0, 1.0);
            if conservative_group_count <= 1 {
                confidence = confidence.min(
                    query
                        .retrieval_policy
                        .projection
                        .single_group_confidence_cap
                        .get(),
                );
            }
            let priority_score = ((signed_evidence + 1.0) * 0.5 * confidence).clamp(0.0, 1.0);
            let disposition = if negative_attempts == 0
                && signed_evidence > query.retrieval_policy.projection.positive_threshold.get()
            {
                AdviceDisposition::PrioritizeExploration
            } else if signed_evidence < query.retrieval_policy.projection.negative_threshold.get() {
                AdviceDisposition::DeprioritizeButRetainControl
            } else {
                AdviceDisposition::ObserveAsControl
            };
            let group_count = accumulator.contributions.len();
            report.advice.push(ProceduralAdvice {
                retrieval_policy_digest: query.retrieval_policy.digest.clone(),
                configuration_digest,
                configuration,
                disposition,
                priority_score: FiniteF64::new(priority_score)?,
                confidence: FiniteF64::new(confidence)?,
                explanation: AdviceExplanation {
                    conservatively_counted_independent_groups: usize::try_from(
                        conservative_group_count,
                    )
                    .map_err(|_| invalid("procedural_independent_group_count_overflow"))?,
                    validated_attempts,
                    negative_attempts,
                    mean_context_similarity: FiniteF64::new(similarity_sum / group_count as f64)?,
                    positive_weight: FiniteF64::new(positive_weight)?,
                    negative_weight: FiniteF64::new(negative_weight)?,
                    reasons,
                    suggested_corrections: corrections,
                },
            });
        }

        report.advice.sort_by(|left, right| {
            right
                .priority_score
                .get()
                .total_cmp(&left.priority_score.get())
                .then(left.configuration_digest.cmp(&right.configuration_digest))
        });
        report.advice.truncate(query.max_results);
        Ok(report)
    }
}

/// Compare the meaning of an evidence-bearing attempt while deliberately
/// ignoring replay wrappers (revision, lineage position, and record digest).
/// The same exact report cannot become fresh evidence by being replayed later.
fn attempt_evidence_semantics_equal(left: &SolverAttempt, right: &SolverAttempt) -> bool {
    left.schema == right.schema
        && left.bindings == right.bindings
        && left.applicability == right.applicability
        && left.configuration == right.configuration
        && left.research_validation == right.research_validation
        && left.solver_rejection_receipt == right.solver_rejection_receipt
        && left.outcome == right.outcome
        && left.applied_correction == right.applied_correction
}

fn solver_run_receipt_semantics_equal(
    left: &SolverRunFailureRecord,
    right: &SolverRunFailureRecord,
) -> bool {
    left.schema == right.schema
        && left.capability == right.capability
        && left.problem_digest == right.problem_digest
        && left.solver_policy_digest == right.solver_policy_digest
        && left.applicability == right.applicability
        && left.subject == right.subject
        && left.outcome == right.outcome
        && left.deterministic_work_units == right.deterministic_work_units
        && left.run_receipt_digest == right.run_receipt_digest
}

fn correction_is_realized(
    parent: &SolverAttempt,
    child: &SolverAttempt,
    correction: CorrectionKind,
) -> bool {
    match correction {
        CorrectionKind::IncreaseRank => matches!(
            (parent.configuration.rank(), child.configuration.rank()),
            (Some(parent_rank), Some(child_rank)) if child_rank > parent_rank
        ),
        CorrectionKind::ReduceRank => matches!(
            (parent.configuration.rank(), child.configuration.rank()),
            (Some(parent_rank), Some(child_rank)) if child_rank < parent_rank
        ),
        CorrectionKind::IncreaseRegularization => matches!(
            (
                parent.configuration.regularization,
                child.configuration.regularization,
            ),
            (Some(parent_value), Some(child_value)) if child_value.get() > parent_value.get()
        ),
        CorrectionKind::SwitchToPivotedQr => child.configuration.family == SolverFamily::PivotedQr,
        CorrectionKind::SwitchToSvd => matches!(
            child.configuration.family,
            SolverFamily::DirectJacobiSvd | SolverFamily::DivideConquerSvd
        ),
        CorrectionKind::SwitchToIterative => matches!(
            child.configuration.family,
            SolverFamily::IterativeLsqr | SolverFamily::IterativeLsmr
        ),
        CorrectionKind::SwitchToRobustEstimator => {
            child.configuration.family == SolverFamily::RobustMEstimator
        }
        CorrectionKind::SwitchToSparse => matches!(
            child.configuration.family,
            SolverFamily::SparseProximal | SolverFamily::GroupSparseProximal
        ),
        CorrectionKind::SwitchToBlockStructured => {
            child.configuration.family == SolverFamily::BlockStructured
        }
        CorrectionKind::EscalateFullRank => matches!(
            child.configuration.family,
            SolverFamily::FullRankGradient | SolverFamily::OrthogonalizedFullRank
        ),
        CorrectionKind::SelectHybridResidency => {
            child.configuration.family == SolverFamily::HybridRuntime
        }
        CorrectionKind::SelectSoftwareResidency => {
            child.configuration.family == SolverFamily::SoftwareOnly
        }
        CorrectionKind::ExpandIndependentEvidence => matches!(
            (
                parent.research_validation.as_ref(),
                child.research_validation.as_ref(),
            ),
            (Some(parent_evaluation), Some(child_evaluation))
                if child_evaluation.independent_group_count
                    > parent_evaluation.independent_group_count
        ),
        CorrectionKind::RepairEvaluator => matches!(
            (
                parent.research_validation.as_ref(),
                child.research_validation.as_ref(),
            ),
            (Some(parent_evaluation), Some(child_evaluation))
                if child_evaluation.evaluator_policy_digest
                    != parent_evaluation.evaluator_policy_digest
        ),
    }
}

fn applicability_similarity(
    left: &Applicability,
    right: &Applicability,
    policy: &ProceduralRetrievalPolicy,
) -> BrainResult<f64> {
    left.validate()?;
    right.validate()?;
    if left.profile.execution_semantics != right.profile.execution_semantics {
        return Ok(0.0);
    }
    let row_score = logarithmic_ratio(left.dimensions.rows, right.dimensions.rows);
    let column_score = logarithmic_ratio(left.dimensions.columns, right.dimensions.columns);
    let rank_score = match (
        left.dimensions.estimated_effective_rank,
        right.dimensions.estimated_effective_rank,
    ) {
        (Some(left_rank), Some(right_rank)) => Some(ratio_similarity(
            left_rank,
            left.dimensions.rows.min(left.dimensions.columns),
            right_rank,
            right.dimensions.rows.min(right.dimensions.columns),
        )),
        _ => None,
    };
    let precision_score = if left.profile.precision == right.profile.precision {
        1.0
    } else {
        0.0
    };
    let structure_score = if left.profile.structure == right.profile.structure {
        1.0
    } else {
        0.0
    };
    let mut weighted_score = policy.projection.row_weight.get() * row_score
        + policy.projection.column_weight.get() * column_score
        + policy.projection.precision_weight.get() * precision_score
        + policy.projection.structure_weight.get() * structure_score;
    if let Some(rank_score) = rank_score {
        weighted_score += policy.projection.rank_weight.get() * rank_score;
    }
    let total_weight = policy.projection.row_weight.get()
        + policy.projection.column_weight.get()
        + policy.projection.rank_weight.get()
        + policy.projection.condition_weight.get()
        + policy.projection.outlier_weight.get()
        + policy.projection.sparsity_weight.get()
        + policy.projection.noise_weight.get()
        + policy.projection.precision_weight.get()
        + policy.projection.structure_weight.get();

    if let (Some(left), Some(right)) = (
        left.statistics.log10_condition,
        right.statistics.log10_condition,
    ) {
        weighted_score += policy.projection.condition_weight.get()
            * (-(left.get() - right.get()).abs() / 2.0).exp();
    }
    add_optional_unit_similarity(
        &mut weighted_score,
        policy.projection.outlier_weight.get(),
        left.statistics.outlier_fraction,
        right.statistics.outlier_fraction,
    );
    add_optional_unit_similarity(
        &mut weighted_score,
        policy.projection.sparsity_weight.get(),
        left.statistics.observed_sparsity,
        right.statistics.observed_sparsity,
    );
    add_optional_unit_similarity(
        &mut weighted_score,
        policy.projection.noise_weight.get(),
        left.statistics.noise_fraction,
        right.statistics.noise_fraction,
    );
    if total_weight <= 0.0 || !weighted_score.is_finite() {
        return Err(BrainError::Numerical(
            "procedural_similarity_weight_invalid".into(),
        ));
    }
    // Missing measurements contribute no similarity credit. This is not a
    // fabricated zero observation: it is an explicit confidence penalty that
    // prevents two unknown profiles from appearing to be a perfect match.
    Ok((weighted_score / total_weight).clamp(0.0, 1.0))
}

fn add_optional_unit_similarity(
    weighted_score: &mut f64,
    weight: f64,
    left: Option<FiniteF64>,
    right: Option<FiniteF64>,
) {
    if let (Some(left), Some(right)) = (left, right) {
        *weighted_score += weight * (1.0 - (left.get() - right.get()).abs());
    }
}

fn logarithmic_ratio(left: u64, right: u64) -> f64 {
    (-((left as f64).ln() - (right as f64).ln()).abs() / 2.0).exp()
}

fn ratio_similarity(
    left_numerator: u64,
    left_denominator: u64,
    right_numerator: u64,
    right_denominator: u64,
) -> f64 {
    let left = left_numerator as f64 / left_denominator as f64;
    let right = right_numerator as f64 / right_denominator as f64;
    (1.0 - (left - right).abs()).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portfolio_governance::{
        evaluate_paired_groups, EvidenceId, IndependenceGroupId, MetricDirection, MetricId,
        MetricSpec, ObservationWindow, PairId, PairedExperimentalUnit, PairedObservation,
        RobustEvaluationPolicy, VariantId,
    };
    use crate::solver_portfolio::{
        solve_with_portfolio, LeastSquaresProblem, PortfolioPolicy, SolverResourceLimits,
    };

    fn raw_digest(tag: u8) -> Sha256Digest {
        Sha256Digest::digest_bytes(&[tag])
    }

    fn sealed_digest<T>(tag: u8) -> T
    where
        T: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(&format!("\"{}\"", raw_digest(tag))).unwrap()
    }

    fn least_squares_problem(tag: u8) -> LeastSquaresProblem {
        LeastSquaresProblem::new(
            vec![vec![1.0, f64::from(tag)], vec![2.0, 1.0]],
            vec![vec![0.5], vec![1.5]],
        )
        .unwrap()
    }

    fn exact_problem(tag: u8) -> ExactSolverProblemDigest {
        least_squares_problem(tag).digest().unwrap()
    }

    fn applicability() -> Applicability {
        Applicability::new(
            MatrixDimensions::new(4, 2, Some(2), 0).unwrap(),
            StatisticalProfile::fully_observed(Some(2.0), 0.0, 0.2, 0.05).unwrap(),
            NumericProblemProfile::new(
                NumericPrecision::Float64,
                MatrixStructure::Dense,
                ExecutionSemantics::PureTensor,
            ),
        )
        .unwrap()
    }

    fn bindings(
        system_tag: u8,
        capability: &str,
        problem: ExactSolverProblemDigest,
        candidate_tag: u8,
    ) -> AttemptBindings {
        let capability = CapabilityContext::new(
            sealed_digest(system_tag),
            CapabilityId::parse(capability).unwrap(),
            sealed_digest(system_tag.wrapping_add(1)),
            BaseArtifactDigest::bind_exact_digest(&raw_digest(system_tag.wrapping_add(2))),
            TargetProfileDigest::bind_exact_digest(&raw_digest(system_tag.wrapping_add(3))),
        )
        .unwrap();
        let solver = SolverArtifactContext::new(
            problem,
            SolverPolicyDigest::of_exact_bytes(b"solver-policy-v1"),
            SolverCandidateDigest::of_exact_bytes(&[candidate_tag]),
        )
        .unwrap();
        AttemptBindings::new(capability, solver).unwrap()
    }

    fn config(family: SolverFamily) -> SolverConfiguration {
        let (parameters, regularization, tolerance, max_iterations) = match family {
            SolverFamily::CholeskyRidgeLowRank => (
                SolverParameters::LowRank { rank: 2 },
                Some(1e-6),
                Some(1e-9),
                None,
            ),
            SolverFamily::RandomizedSvd => (
                SolverParameters::RandomizedLowRank {
                    rank: 2,
                    oversampling: 2,
                },
                None,
                Some(1e-9),
                Some(1_000),
            ),
            SolverFamily::DirectJacobiSvd
            | SolverFamily::PivotedQr
            | SolverFamily::DivideConquerSvd => (
                SolverParameters::Direct,
                None,
                Some(1e-9),
                (family == SolverFamily::DirectJacobiSvd).then_some(1_000),
            ),
            SolverFamily::IterativeLsqr | SolverFamily::IterativeLsmr => {
                (SolverParameters::Iterative, None, Some(1e-9), Some(1_000))
            }
            SolverFamily::RobustMEstimator => (
                SolverParameters::Robust {
                    tuning_constant: FiniteF64::new(1.345).unwrap(),
                },
                None,
                Some(1e-9),
                Some(1_000),
            ),
            SolverFamily::SparseProximal => (
                SolverParameters::Sparse { maximum_nonzero: 2 },
                None,
                Some(1e-9),
                Some(1_000),
            ),
            SolverFamily::GroupSparseProximal => (
                SolverParameters::GroupSparse { maximum_groups: 2 },
                None,
                Some(1e-9),
                Some(1_000),
            ),
            SolverFamily::BlockStructured => (
                SolverParameters::BlockStructured { maximum_blocks: 2 },
                None,
                Some(1e-9),
                Some(1_000),
            ),
            SolverFamily::FullRankGradient | SolverFamily::OrthogonalizedFullRank => {
                (SolverParameters::FullRank, None, Some(1e-9), Some(1_000))
            }
            SolverFamily::ExternalProposal => (
                SolverParameters::External {
                    representation: SolverRepresentationKind::Dense,
                    declared_implementation_digest:
                        DeclaredBackendImplementationDigest::of_exact_bytes(
                            b"test.external.backend/v1",
                        ),
                },
                None,
                None,
                None,
            ),
            SolverFamily::HybridRuntime => (SolverParameters::Hybrid, None, None, None),
            SolverFamily::SoftwareOnly => (SolverParameters::Software, None, None, None),
        };
        SolverConfiguration::new(
            family,
            parameters,
            regularization,
            tolerance,
            max_iterations,
        )
        .unwrap()
    }

    fn evaluated_attempt(
        bindings: AttemptBindings,
        configuration: SolverConfiguration,
        status: AttemptStatus,
        evidence_tag: u8,
        lineage_tag: u8,
        revision: u64,
    ) -> SolverAttempt {
        let failure =
            matches!(status, AttemptStatus::Rejected | AttemptStatus::Failed).then(|| {
                FailureRecord::new(
                    FailureKind::IllConditioned,
                    FailureStage::ResearchValidation,
                    FailureSeverity::Serious,
                )
            });
        let correction = failure.as_ref().map(|_| {
            CorrectionRecord::new(CorrectionKind::SwitchToSvd, CorrectionStatus::Proposed)
        });
        let outcome = AttemptOutcome::new(
            status,
            OutcomeMetrics::new(
                Some(0.05),
                match status {
                    AttemptStatus::Validated => Some(true),
                    AttemptStatus::Rejected | AttemptStatus::Failed => Some(false),
                    AttemptStatus::Inconclusive => None,
                },
                Some(100),
            )
            .unwrap(),
            failure,
            correction,
        )
        .unwrap();
        let evaluation = ResearchValidation::new(
            ResearchReportReceiptDigest::of_exact_bytes(&[evidence_tag]),
            EvaluationReceiptDigest::from(raw_digest(evidence_tag)),
            EvaluatorPolicyDigest::of_exact_bytes(b"independent-evaluator-v1"),
            EvaluationDesignDigest::of_exact_bytes(&[evidence_tag]),
            bindings.solver.candidate_digest.clone(),
            8,
        )
        .unwrap();
        let draft = VerifiedAttemptDraft::new(
            bindings,
            applicability(),
            configuration,
            AttemptLineage::root(ProceduralLineageId::of_exact_bytes(&[lineage_tag])),
            revision,
            None,
        )
        .unwrap();
        SolverAttempt::seal(draft, Some(evaluation), outcome).unwrap()
    }

    fn query(bindings: &AttemptBindings) -> RetrievalQuery {
        RetrievalQuery::new(
            RetrievalScope::from_bindings(bindings),
            applicability(),
            100,
            ProblemTransferPolicy::ExactOnly,
            16,
        )
        .unwrap()
    }

    fn drift_report(
        start_tick: u64,
        candidate_value: f64,
        candidate_id: &VariantId,
    ) -> PairedEvaluationReport {
        let metric_id = MetricId::parse("quality.functional").unwrap();
        let specs =
            vec![MetricSpec::new(metric_id.clone(), MetricDirection::Maximize, None).unwrap()];
        let window = ObservationWindow::new(start_tick, start_tick + 1).unwrap();
        let observations = (0..3)
            .map(|group| {
                PairedObservation::new(
                    metric_id.clone(),
                    PairedExperimentalUnit::new(
                        IndependenceGroupId::parse(format!("drift.group.{group}")).unwrap(),
                        PairId::parse(format!("drift.pair.{start_tick}.{group}")).unwrap(),
                        EvidenceId::parse(format!("drift.evidence.{start_tick}.{group}")).unwrap(),
                        window.clone(),
                    ),
                    0.0,
                    candidate_value,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        evaluate_paired_groups(
            VariantId::parse("baseline.fixed").unwrap(),
            candidate_id.clone(),
            &specs,
            &observations,
            &RobustEvaluationPolicy::new(3, 3, 100).unwrap(),
        )
        .unwrap()
    }

    fn attempt_for_drift_report(
        bindings: AttemptBindings,
        report: &PairedEvaluationReport,
        revision: u64,
        lineage_tag: u8,
    ) -> SolverAttempt {
        let validation = ResearchValidation::new(
            research_report_receipt(report),
            EvaluationReceiptDigest::from(Sha256Digest::digest_domain(
                RESEARCH_VALIDATION_RECEIPT_DOMAIN,
                report.digest().as_str().as_bytes(),
            )),
            EvaluatorPolicyDigest::of_exact_bytes(
                report.evaluation_policy_digest().as_str().as_bytes(),
            ),
            research_evaluation_design(report).unwrap(),
            bindings.solver.candidate_digest.clone(),
            u32::try_from(report.independent_group_count()).unwrap(),
        )
        .unwrap();
        let draft = VerifiedAttemptDraft::new(
            bindings,
            applicability(),
            config(SolverFamily::PivotedQr),
            AttemptLineage::root(ProceduralLineageId::of_exact_bytes(&[lineage_tag])),
            revision,
            None,
        )
        .unwrap();
        let outcome = AttemptOutcome::new(
            AttemptStatus::Validated,
            OutcomeMetrics::new(Some(0.1), Some(true), None).unwrap(),
            None,
            None,
        )
        .unwrap();
        SolverAttempt::seal(draft, Some(validation), outcome).unwrap()
    }

    #[test]
    fn tamper_and_semantic_relabel_are_rejected() {
        let problem = exact_problem(1);
        let original_bindings = bindings(10, "memory.short:v1", problem.clone(), 20);
        let attempt = evaluated_attempt(
            original_bindings,
            config(SolverFamily::DivideConquerSvd),
            AttemptStatus::Validated,
            31,
            41,
            1,
        );
        let mut wire = serde_json::to_value(&attempt).unwrap();
        wire["bindings"]["capability"]["capability_id"] =
            serde_json::Value::String("memory.relabelled:v1".into());
        let tampered: SolverAttempt = serde_json::from_value(wire).unwrap();
        assert!(matches!(
            ProceduralMemory::new().record_attempt(tampered),
            Err(BrainError::Integrity(code)) if code == "procedural_attempt_digest_mismatch"
        ));

        let first_bindings = bindings(10, "memory.short:v1", problem.clone(), 21);
        let second_bindings = bindings(10, "memory.short:v1", problem, 22);
        let first = evaluated_attempt(
            first_bindings,
            config(SolverFamily::PivotedQr),
            AttemptStatus::Validated,
            55,
            56,
            1,
        );
        let second = evaluated_attempt(
            second_bindings,
            config(SolverFamily::DivideConquerSvd),
            AttemptStatus::Validated,
            55,
            57,
            1,
        );
        let mut memory = ProceduralMemory::new();
        memory.record_attempt(first).unwrap();
        assert!(matches!(
            memory.record_attempt(second),
            Err(BrainError::Integrity(code)) if code == "procedural_research_report_relabel"
        ));
    }

    #[test]
    fn evaluation_design_cardinality_cannot_be_relabelled() {
        let problem = exact_problem(32);
        let attempt_bindings = bindings(34, "memory.design:v1", problem, 35);
        let first = evaluated_attempt(
            attempt_bindings,
            config(SolverFamily::PivotedQr),
            AttemptStatus::Validated,
            36,
            37,
            1,
        );
        let mut relabelled = first.clone();
        let evaluation = relabelled.research_validation.as_mut().unwrap();
        evaluation.report_receipt_digest =
            ResearchReportReceiptDigest::of_exact_bytes(b"different-report");
        evaluation.receipt_digest = EvaluationReceiptDigest::from(raw_digest(38));
        evaluation.independent_group_count = 9;
        relabelled.lineage.lineage_id = ProceduralLineageId::of_exact_bytes(b"other-lineage");
        relabelled.observed_revision = 2;
        relabelled.attempt_digest = relabelled.calculate_digest().unwrap();
        relabelled.authenticate().unwrap();

        let mut memory = ProceduralMemory::new();
        memory.record_attempt(first).unwrap();
        assert!(matches!(
            memory.record_attempt(relabelled),
            Err(BrainError::Integrity(code)) if code == "procedural_evaluation_design_relabel"
        ));
    }

    #[test]
    fn exact_scope_prevents_cross_project_and_cross_capability_advice() {
        let problem = exact_problem(2);
        let source_bindings = bindings(60, "capability.a:v1", problem.clone(), 61);
        let attempt = evaluated_attempt(
            source_bindings,
            config(SolverFamily::PivotedQr),
            AttemptStatus::Validated,
            62,
            63,
            1,
        );
        let memory = ProceduralMemory::rebuild(vec![attempt], vec![]).unwrap();

        let other_project = bindings(70, "capability.a:v1", problem.clone(), 71);
        assert!(memory
            .retrieve(&query(&other_project))
            .unwrap()
            .advice()
            .is_empty());
        let other_capability = bindings(60, "capability.b:v1", problem, 72);
        assert!(memory
            .retrieve(&query(&other_capability))
            .unwrap()
            .advice()
            .is_empty());
    }

    #[test]
    fn functional_drift_uses_registered_scope_and_revision_and_makes_advice_stale() {
        let problem = exact_problem(33);
        let mut attempt_bindings = bindings(85, "capability.observed-drift:v1", problem, 86);
        let candidate = CandidateRepresentation::Dense {
            rows: 1,
            columns: 1,
            weights: vec![1.0],
        };
        let exact_candidate = candidate.exact_digest().unwrap();
        attempt_bindings.solver.candidate_digest =
            SolverCandidateDigest::bind_exact_digest(exact_candidate.as_digest());
        let candidate_id = VariantId::parse("candidate.fixed").unwrap();
        let candidate_binding =
            VerifiedCandidateVariantBinding::from_candidate(candidate_id.clone(), &candidate)
                .unwrap();
        let previous = drift_report(1, 1.0, &candidate_id);
        let current = drift_report(3, 10.0, &candidate_id);
        let previous_attempt = attempt_for_drift_report(attempt_bindings.clone(), &previous, 1, 87);
        let current_attempt = attempt_for_drift_report(attempt_bindings.clone(), &current, 2, 88);
        let mut memory = ProceduralMemory::new();
        memory.record_attempt(previous_attempt).unwrap();
        memory.record_attempt(current_attempt).unwrap();
        let derived = memory
            .derive_functional_change(&candidate_binding, &previous, &current)
            .unwrap();
        let FunctionalDriftObservation::Detected(record) = derived else {
            panic!("expected a sealed observed functional change");
        };
        assert_eq!(record.kind, DriftKind::ObservedFunctionalChange);
        assert_eq!(record.detected_revision, 2);
        assert_eq!(record.invalidates_through_revision, 2);
        record.authenticate().unwrap();
        assert_eq!(
            memory.record_drift((*record).clone()).unwrap(),
            RecordDisposition::Inserted
        );
        let retrieval = memory.retrieve(&query(&attempt_bindings)).unwrap();
        assert!(retrieval.advice().is_empty());
        assert_eq!(retrieval.drift_filtered_records(), 2);

        let no_change_report = drift_report(5, 1.0, &candidate_id);
        let no_change = memory
            .derive_functional_change(&candidate_binding, &previous, &no_change_report)
            .unwrap();
        assert_eq!(no_change, FunctionalDriftObservation::NoObservedChange);
    }

    #[test]
    fn unregistered_drift_reports_are_bounded_unknown_not_free_invalidation() {
        let candidate = CandidateRepresentation::Dense {
            rows: 1,
            columns: 1,
            weights: vec![1.0],
        };
        let candidate_id = VariantId::parse("candidate.unregistered").unwrap();
        let candidate_binding =
            VerifiedCandidateVariantBinding::from_candidate(candidate_id.clone(), &candidate)
                .unwrap();
        let observation = ProceduralMemory::new()
            .derive_functional_change(
                &candidate_binding,
                &drift_report(1, 1.0, &candidate_id),
                &drift_report(3, 10.0, &candidate_id),
            )
            .unwrap();
        assert_eq!(observation, FunctionalDriftObservation::BoundedUnknown);
    }

    #[test]
    fn failed_attempt_is_negative_memory_but_never_a_blocking_authority() {
        let problem = exact_problem(4);
        let good_bindings = bindings(90, "capability.solve:v1", problem.clone(), 91);
        let bad_bindings = bindings(90, "capability.solve:v1", problem, 92);
        let good = evaluated_attempt(
            good_bindings.clone(),
            config(SolverFamily::DivideConquerSvd),
            AttemptStatus::Validated,
            93,
            94,
            1,
        );
        let bad = evaluated_attempt(
            bad_bindings,
            config(SolverFamily::CholeskyRidgeLowRank),
            AttemptStatus::Failed,
            95,
            96,
            1,
        );
        let memory = ProceduralMemory::rebuild(vec![bad, good], vec![]).unwrap();
        let report = memory.retrieve(&query(&good_bindings)).unwrap();
        assert_eq!(report.advice().len(), 2);
        let good_advice = report
            .advice()
            .iter()
            .find(|value| value.configuration().family() == SolverFamily::DivideConquerSvd)
            .unwrap();
        let bad_advice = report
            .advice()
            .iter()
            .find(|value| value.configuration().family() == SolverFamily::CholeskyRidgeLowRank)
            .unwrap();
        assert!(good_advice.priority_score() > bad_advice.priority_score());
        assert_eq!(
            bad_advice.disposition(),
            AdviceDisposition::DeprioritizeButRetainControl
        );
        assert!(!bad_advice.authorizes_promotion());
        assert!(bad_advice.requires_independent_control());
    }

    #[test]
    fn declared_external_backend_identity_never_becomes_transferable_advice() {
        let problem = exact_problem(39);
        let attempt_bindings = bindings(40, "capability.external:v1", problem, 41);
        let attempt = evaluated_attempt(
            attempt_bindings.clone(),
            config(SolverFamily::ExternalProposal),
            AttemptStatus::Validated,
            42,
            43,
            1,
        );
        let memory = ProceduralMemory::rebuild(vec![attempt], vec![]).unwrap();
        let report = memory.retrieve(&query(&attempt_bindings)).unwrap();
        assert!(report.advice().is_empty());
        assert_eq!(report.evidence_filtered_records(), 1);
    }

    #[test]
    fn ranking_and_rebuild_are_invariant_to_record_order() {
        let problem = exact_problem(5);
        let first_bindings = bindings(100, "capability.order:v1", problem.clone(), 101);
        let second_bindings = bindings(100, "capability.order:v1", problem, 102);
        let first = evaluated_attempt(
            first_bindings.clone(),
            config(SolverFamily::PivotedQr),
            AttemptStatus::Validated,
            103,
            104,
            1,
        );
        let second = evaluated_attempt(
            second_bindings,
            config(SolverFamily::DivideConquerSvd),
            AttemptStatus::Validated,
            105,
            106,
            1,
        );
        let forward = ProceduralMemory::rebuild(vec![first.clone(), second.clone()], vec![])
            .unwrap()
            .retrieve(&query(&first_bindings))
            .unwrap();
        let reverse = ProceduralMemory::rebuild(vec![second, first], vec![])
            .unwrap()
            .retrieve(&query(&first_bindings))
            .unwrap();
        assert_eq!(forward, reverse);
    }

    #[test]
    fn no_independent_evidence_means_no_advice() {
        let problem = exact_problem(6);
        let attempt_bindings = bindings(110, "capability.unknown:v1", problem, 111);
        let outcome = AttemptOutcome::new(
            AttemptStatus::Inconclusive,
            OutcomeMetrics::new(None, None, None).unwrap(),
            None,
            None,
        )
        .unwrap();
        let draft = VerifiedAttemptDraft::new(
            attempt_bindings.clone(),
            applicability(),
            config(SolverFamily::PivotedQr),
            AttemptLineage::root(ProceduralLineageId::of_exact_bytes(b"unknown")),
            1,
            None,
        )
        .unwrap();
        let attempt = SolverAttempt::seal(draft, None, outcome).unwrap();
        let memory = ProceduralMemory::rebuild(vec![attempt], vec![]).unwrap();
        let report = memory.retrieve(&query(&attempt_bindings)).unwrap();
        assert!(report.advice().is_empty());
        assert_eq!(report.evidence_filtered_records(), 1);
    }

    #[test]
    fn unknown_measurements_are_not_zero_and_cannot_validate_an_attempt() {
        let unknown = StatisticalProfile::new(None, None, None, None).unwrap();
        let measured_zero = StatisticalProfile::fully_observed(None, 0.0, 0.0, 0.0).unwrap();
        assert_ne!(
            Applicability::new(
                MatrixDimensions::new(4, 2, None, 0).unwrap(),
                unknown,
                NumericProblemProfile::new(
                    NumericPrecision::Float64,
                    MatrixStructure::Dense,
                    ExecutionSemantics::PureTensor,
                ),
            )
            .unwrap()
            .digest()
            .unwrap(),
            Applicability::new(
                MatrixDimensions::new(4, 2, Some(2), 0).unwrap(),
                measured_zero,
                NumericProblemProfile::new(
                    NumericPrecision::Float64,
                    MatrixStructure::Dense,
                    ExecutionSemantics::PureTensor,
                ),
            )
            .unwrap()
            .digest()
            .unwrap()
        );

        let partial = OutcomeMetrics::new(Some(0.1), None, Some(10)).unwrap();
        assert!(matches!(
            AttemptOutcome::new(AttemptStatus::Validated, partial, None, None),
            Err(BrainError::Invalid(code)) if code == "procedural_outcome_semantics_invalid"
        ));
    }

    #[test]
    fn exact_problem_identity_is_not_the_applicability_summary() {
        let first_problem = exact_problem(7);
        let second_problem = exact_problem(8);
        assert_ne!(first_problem, second_problem);
        assert_eq!(
            applicability().digest().unwrap(),
            applicability().digest().unwrap()
        );

        let source_bindings = bindings(120, "capability.transfer:v1", first_problem, 121);
        let target_bindings = bindings(120, "capability.transfer:v1", second_problem, 122);
        let attempt = evaluated_attempt(
            source_bindings,
            config(SolverFamily::PivotedQr),
            AttemptStatus::Validated,
            123,
            124,
            1,
        );
        let memory = ProceduralMemory::rebuild(vec![attempt], vec![]).unwrap();
        assert!(memory
            .retrieve(&query(&target_bindings))
            .unwrap()
            .advice()
            .is_empty());
        let analogical = RetrievalQuery::new(
            RetrievalScope::from_bindings(&target_bindings),
            applicability(),
            100,
            ProblemTransferPolicy::feature_bounded(0.95).unwrap(),
            16,
        )
        .unwrap();
        assert_eq!(memory.retrieve(&analogical).unwrap().advice().len(), 1);
    }

    #[test]
    fn lineage_cannot_cross_scope_or_apply_an_unproposed_correction() {
        let problem = exact_problem(9);
        let parent_bindings = bindings(130, "capability.lineage:v1", problem.clone(), 131);
        let parent = evaluated_attempt(
            parent_bindings,
            config(SolverFamily::PivotedQr),
            AttemptStatus::Validated,
            132,
            133,
            1,
        );
        let lineage_id = parent.lineage.lineage_id.clone();
        let child_lineage = AttemptLineage::child(lineage_id, 1, parent.digest().clone()).unwrap();
        let cross_scope = bindings(140, "capability.lineage:v1", problem, 141);
        let child_draft = VerifiedAttemptDraft::new(
            cross_scope.clone(),
            applicability(),
            config(SolverFamily::DivideConquerSvd),
            child_lineage,
            2,
            Some(CorrectionKind::SwitchToSvd),
        )
        .unwrap();
        let evaluation = ResearchValidation::new(
            ResearchReportReceiptDigest::of_exact_bytes(b"lineage-child-report"),
            EvaluationReceiptDigest::from(raw_digest(142)),
            EvaluatorPolicyDigest::of_exact_bytes(b"independent-evaluator-v1"),
            EvaluationDesignDigest::of_exact_bytes(b"lineage-child"),
            cross_scope.solver.candidate_digest.clone(),
            8,
        )
        .unwrap();
        let outcome = AttemptOutcome::new(
            AttemptStatus::Validated,
            OutcomeMetrics::new(Some(0.01), Some(true), Some(100)).unwrap(),
            None,
            Some(CorrectionRecord::new(
                CorrectionKind::SwitchToSvd,
                CorrectionStatus::IndependentlyValidated,
            )),
        )
        .unwrap();
        let child = SolverAttempt::seal(child_draft, Some(evaluation), outcome).unwrap();
        let mut memory = ProceduralMemory::new();
        memory.record_attempt(parent).unwrap();
        assert!(matches!(
            memory.record_attempt(child),
            Err(BrainError::Integrity(code)) if code == "procedural_lineage_parent_mismatch"
        ));
    }

    #[test]
    fn lineage_cannot_claim_a_correction_that_configuration_did_not_realize() {
        let problem = exact_problem(44);
        let parent_bindings = bindings(45, "capability.correction:v1", problem.clone(), 46);
        let child_bindings = bindings(45, "capability.correction:v1", problem, 47);
        let parent = evaluated_attempt(
            parent_bindings,
            config(SolverFamily::CholeskyRidgeLowRank),
            AttemptStatus::Rejected,
            48,
            49,
            1,
        );
        let child_lineage =
            AttemptLineage::child(parent.lineage_id().clone(), 1, parent.digest().clone()).unwrap();
        let evaluation = ResearchValidation::new(
            ResearchReportReceiptDigest::of_exact_bytes(b"correction-child-report"),
            EvaluationReceiptDigest::from(raw_digest(50)),
            EvaluatorPolicyDigest::of_exact_bytes(b"independent-evaluator-v1"),
            EvaluationDesignDigest::of_exact_bytes(b"correction-child"),
            child_bindings.solver.candidate_digest.clone(),
            8,
        )
        .unwrap();
        let outcome = AttemptOutcome::new(
            AttemptStatus::Rejected,
            OutcomeMetrics::new(Some(0.5), Some(false), None).unwrap(),
            Some(FailureRecord::new(
                FailureKind::ResidualTooLarge,
                FailureStage::ResearchValidation,
                FailureSeverity::Serious,
            )),
            Some(CorrectionRecord::new(
                CorrectionKind::SwitchToSvd,
                CorrectionStatus::Refuted,
            )),
        )
        .unwrap();
        let child = SolverAttempt::seal(
            VerifiedAttemptDraft::new(
                child_bindings,
                applicability(),
                config(SolverFamily::CholeskyRidgeLowRank),
                child_lineage,
                2,
                Some(CorrectionKind::SwitchToSvd),
            )
            .unwrap(),
            Some(evaluation),
            outcome,
        )
        .unwrap();
        let mut memory = ProceduralMemory::new();
        memory.record_attempt(parent).unwrap();
        assert!(matches!(
            memory.record_attempt(child),
            Err(BrainError::Integrity(code))
                if code == "procedural_lineage_correction_not_realized"
        ));
    }

    #[test]
    fn reducer_deduplicates_exact_replay_and_enforces_resource_bounds() {
        let problem = exact_problem(10);
        let attempt_bindings = bindings(150, "capability.bounds:v1", problem, 151);
        let attempt = evaluated_attempt(
            attempt_bindings,
            config(SolverFamily::PivotedQr),
            AttemptStatus::Validated,
            152,
            153,
            1,
        );
        let mut memory = ProceduralMemory::new();
        assert_eq!(
            memory.record_attempt(attempt.clone()).unwrap(),
            RecordDisposition::Inserted
        );
        assert_eq!(
            memory.record_attempt(attempt).unwrap(),
            RecordDisposition::AlreadyPresent
        );
        assert_eq!(memory.attempt_count(), 1);

        let too_large = Applicability::new(
            MatrixDimensions::new(100_000, 100_000, Some(100), 0).unwrap(),
            StatisticalProfile::fully_observed(Some(1.0), 0.0, 0.0, 0.0).unwrap(),
            NumericProblemProfile::new(
                NumericPrecision::Float64,
                MatrixStructure::Dense,
                ExecutionSemantics::PureTensor,
            ),
        );
        assert!(matches!(
            too_large,
            Err(BrainError::Invalid(code)) if code == "procedural_explicit_matrix_budget_exceeded"
        ));
        assert!(RetrievalQuery::new(
            RetrievalScope::from_bindings(&bindings(
                160,
                "capability.query:v1",
                exact_problem(11),
                161,
            )),
            applicability(),
            1,
            ProblemTransferPolicy::ExactOnly,
            MAX_ADVICE_RESULTS + 1,
        )
        .is_err());
    }

    #[test]
    fn rank_unknown_is_distinct_from_observed_zero_and_missing_work_is_allowed() {
        let unknown_rank = Applicability::new(
            MatrixDimensions::new(4, 2, None, 0).unwrap(),
            StatisticalProfile::new(None, None, None, None).unwrap(),
            NumericProblemProfile::new(
                NumericPrecision::Float64,
                MatrixStructure::Dense,
                ExecutionSemantics::PureTensor,
            ),
        )
        .unwrap();
        let observed_zero_rank = Applicability::new(
            MatrixDimensions::new(4, 2, Some(0), 0).unwrap(),
            StatisticalProfile::new(None, None, None, None).unwrap(),
            NumericProblemProfile::new(
                NumericPrecision::Float64,
                MatrixStructure::Dense,
                ExecutionSemantics::PureTensor,
            ),
        )
        .unwrap();
        assert_ne!(
            unknown_rank.digest().unwrap(),
            observed_zero_rank.digest().unwrap()
        );

        let metrics = OutcomeMetrics::new(Some(0.1), Some(true), None).unwrap();
        assert!(metrics.supports_validated_status());
        assert!(AttemptOutcome::new(AttemptStatus::Validated, metrics, None, None).is_ok());

        let positive_zero = StatisticalProfile::fully_observed(None, 0.0, 0.0, 0.0).unwrap();
        let negative_zero = StatisticalProfile::fully_observed(None, -0.0, -0.0, -0.0).unwrap();
        assert_eq!(positive_zero, negative_zero);
    }

    #[test]
    fn missing_statistics_cannot_look_like_a_perfect_transfer_match() {
        let policy = ProceduralRetrievalPolicy::current().unwrap();
        let unknown = Applicability::new(
            MatrixDimensions::new(4, 2, None, 0).unwrap(),
            StatisticalProfile::new(None, None, None, None).unwrap(),
            NumericProblemProfile::new(
                NumericPrecision::Float64,
                MatrixStructure::Dense,
                ExecutionSemantics::PureTensor,
            ),
        )
        .unwrap();
        assert!(applicability_similarity(&unknown, &unknown, &policy).unwrap() < 0.95);
    }

    #[test]
    fn candidate_free_solver_failure_is_sealed_and_replay_safe() {
        let problem = least_squares_problem(12);
        let policy = PortfolioPolicy::default()
            .with_limits(SolverResourceLimits::new(1, 8, 8, 1_024, 1_024, 1).unwrap())
            .unwrap();
        let report = solve_with_portfolio(&problem, &policy, &[]).unwrap();
        let candidate_bindings = bindings(
            170,
            "capability.run-failure:v1",
            problem.digest().unwrap(),
            171,
        );
        let draft = VerifiedSolverRunFailureDraft::new(
            candidate_bindings.capability.clone(),
            &problem,
            SolverPolicyDigest::bind_exact_digest(policy.digest().unwrap().as_digest()),
            Applicability::new(
                MatrixDimensions::new(2, 2, None, 0).unwrap(),
                StatisticalProfile::new(None, None, None, None).unwrap(),
                NumericProblemProfile::new(
                    NumericPrecision::Float64,
                    MatrixStructure::Dense,
                    ExecutionSemantics::PureTensor,
                ),
            )
            .unwrap(),
            SolverRunSubject::Portfolio,
        )
        .unwrap();
        let failure = SolverRunFailureRecord::seal(draft, 1, None, &report).unwrap();
        assert_eq!(
            failure.outcome(),
            &SolverRunFailureOutcome::BoundedUnknown(SolverRunUnknownReason::ResourceLimit(
                SolverResourceLimitKind::Cases
            ))
        );
        let wire = serde_json::to_value(&failure).unwrap();
        assert!(wire.get("candidate_digest").is_none());
        assert!(wire.get("deterministic_work_units").unwrap().is_null());

        let mut memory = ProceduralMemory::new();
        assert_eq!(
            memory.record_solver_run_failure(failure.clone()).unwrap(),
            RecordDisposition::Inserted
        );
        assert_eq!(
            memory.record_solver_run_failure(failure).unwrap(),
            RecordDisposition::AlreadyPresent
        );
        assert_eq!(memory.solver_run_failure_count(), 1);
        let scoped_bindings = AttemptBindings::new(
            candidate_bindings.capability,
            SolverArtifactContext::new(
                problem.digest().unwrap(),
                SolverPolicyDigest::bind_exact_digest(policy.digest().unwrap().as_digest()),
                SolverCandidateDigest::of_exact_bytes(b"query-only-candidate"),
            )
            .unwrap(),
        )
        .unwrap();
        let retrieval = memory
            .retrieve(
                &RetrievalQuery::new(
                    RetrievalScope::from_bindings(&scoped_bindings),
                    applicability(),
                    1,
                    ProblemTransferPolicy::ExactOnly,
                    16,
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(retrieval.solver_run_cautions().len(), 1);
        let caution = &retrieval.solver_run_cautions()[0];
        assert_eq!(caution.disposition(), AdviceDisposition::ObserveAsControl);
        assert!(!caution.authorizes_promotion());
        assert!(caution.requires_independent_control());
    }

    #[test]
    fn materialized_solver_rejection_is_a_typed_negative_attempt_not_a_run_failure() {
        let problem = LeastSquaresProblem::new(
            vec![vec![1.0], vec![1.0], vec![1.0]],
            vec![vec![0.0], vec![1.0], vec![2.0]],
        )
        .unwrap();
        let policy = PortfolioPolicy::default()
            .with_residual_tolerances(1.0e-12, 1.0e-12)
            .unwrap();
        let report = solve_with_portfolio(&problem, &policy, &[]).unwrap();
        let rejected = report
            .canonical_observed_rejection()
            .unwrap()
            .expect("an inconsistent system must yield an observed rejection");
        let evaluation_index = report
            .evaluations()
            .iter()
            .position(|evaluation| std::ptr::eq(evaluation, rejected))
            .unwrap();
        let candidate = rejected.candidate().unwrap();
        let mut attempt_bindings = bindings(
            200,
            "capability.solver-rejection:v1",
            problem.digest().unwrap(),
            201,
        );
        attempt_bindings.solver.solver_policy_digest =
            SolverPolicyDigest::bind_exact_digest(policy.digest().unwrap().as_digest());
        attempt_bindings.solver.candidate_digest =
            SolverCandidateDigest::bind_exact_digest(candidate.exact_digest().unwrap().as_digest());
        let rejection_applicability = Applicability::new(
            MatrixDimensions::new(3, 1, Some(1), 0).unwrap(),
            StatisticalProfile::new(None, None, None, None).unwrap(),
            NumericProblemProfile::new(
                NumericPrecision::Float64,
                MatrixStructure::Dense,
                ExecutionSemantics::PureTensor,
            ),
        )
        .unwrap();
        let draft = VerifiedAttemptDraft::new(
            attempt_bindings.clone(),
            rejection_applicability.clone(),
            config(SolverFamily::DirectJacobiSvd),
            AttemptLineage::root(ProceduralLineageId::of_exact_bytes(b"solver-rejection")),
            1,
            None,
        )
        .unwrap();
        let attempt =
            SolverAttempt::seal_rejected_candidate(draft, &report, evaluation_index).unwrap();
        assert_eq!(attempt.outcome().status(), AttemptStatus::Rejected);

        let candidate_free_draft = VerifiedSolverRunFailureDraft::new(
            attempt_bindings.capability.clone(),
            &problem,
            attempt_bindings.solver.solver_policy_digest.clone(),
            rejection_applicability.clone(),
            SolverRunSubject::Portfolio,
        )
        .unwrap();
        assert!(matches!(
            SolverRunFailureRecord::seal(candidate_free_draft, 1, None, &report),
            Err(BrainError::Integrity(code))
                if code == "procedural_solver_run_report_binding_mismatch"
        ));

        let mut memory = ProceduralMemory::new();
        memory.record_attempt(attempt).unwrap();
        let retrieval = memory
            .retrieve(
                &RetrievalQuery::new(
                    RetrievalScope::from_bindings(&attempt_bindings),
                    rejection_applicability,
                    1,
                    ProblemTransferPolicy::ExactOnly,
                    16,
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(retrieval.advice().len(), 1);
        assert_eq!(
            retrieval.advice()[0].disposition(),
            AdviceDisposition::DeprioritizeButRetainControl
        );
        assert!(!retrieval.advice()[0].authorizes_promotion());
        assert!(retrieval.advice()[0].requires_independent_control());
    }

    #[test]
    fn solver_run_receipt_replay_is_not_new_evidence_and_relabel_is_rejected() {
        let problem = least_squares_problem(13);
        let policy = PortfolioPolicy::default()
            .with_limits(SolverResourceLimits::new(1, 8, 8, 1_024, 1_024, 1).unwrap())
            .unwrap();
        let report = solve_with_portfolio(&problem, &policy, &[]).unwrap();
        let candidate_bindings = bindings(
            180,
            "capability.run-relabel:v1",
            problem.digest().unwrap(),
            181,
        );
        let relabelled_bindings = bindings(
            190,
            "capability.run-relabel:v1",
            problem.digest().unwrap(),
            191,
        );
        let make_failure =
            |capability: CapabilityContext, revision: u64| -> SolverRunFailureRecord {
                SolverRunFailureRecord::seal(
                    VerifiedSolverRunFailureDraft::new(
                        capability,
                        &problem,
                        SolverPolicyDigest::bind_exact_digest(policy.digest().unwrap().as_digest()),
                        Applicability::new(
                            MatrixDimensions::new(2, 2, None, 0).unwrap(),
                            StatisticalProfile::new(None, None, None, None).unwrap(),
                            NumericProblemProfile::new(
                                NumericPrecision::Float64,
                                MatrixStructure::Dense,
                                ExecutionSemantics::PureTensor,
                            ),
                        )
                        .unwrap(),
                        SolverRunSubject::Portfolio,
                    )
                    .unwrap(),
                    revision,
                    None,
                    &report,
                )
                .unwrap()
            };
        let first = make_failure(candidate_bindings.capability.clone(), 1);
        let exact_replay = make_failure(candidate_bindings.capability.clone(), 2);
        let receipt_relabel = make_failure(relabelled_bindings.capability, 3);
        let mut memory = ProceduralMemory::new();
        memory.record_solver_run_failure(first).unwrap();
        assert_eq!(
            memory.record_solver_run_failure(exact_replay).unwrap(),
            RecordDisposition::AlreadyPresent
        );
        assert_eq!(memory.solver_run_failure_count(), 1);
        assert!(matches!(
            memory.record_solver_run_failure(receipt_relabel),
            Err(BrainError::Integrity(code)) if code == "procedural_solver_run_receipt_relabel"
        ));
    }
}
