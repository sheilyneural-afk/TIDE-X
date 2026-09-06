//! Bounded, architecture-independent least-squares solver portfolio.
//!
//! Backends in this module only propose numerical candidates. They cannot
//! attest evidence, sign a result, promote a capability, or mutate a resident
//! model. The portfolio authority validates every proposal against the typed
//! problem and independently recomputes its residual before assigning a
//! disposition.

use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::linalg::{norm, Matrix};
use crate::low_rank_math::{solve_regularized_multi_case_low_rank, MAX_LOW_RANK};
use crate::validation::{
    choose_energy_rank, effective_rank_from_spectrum, symmetric_psd_condition,
    validate_symmetric_psd,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const BUILTIN_CHOLESKY_NAME: &str = "tidex.bounded_cholesky_ridge.v1";
const BUILTIN_DIRECT_SVD_NAME: &str = "tidex.bounded_direct_jacobi_svd.v1";
const MAX_BACKEND_NAME_BYTES: usize = 128;
const MAX_BACKEND_REASON_BYTES: usize = 4_096;
const EXACT_PROBLEM_DOMAIN: &[u8] = b"CEREBRO:TIDEX:EXACT-LS-PROBLEM:v1\0";
const EXACT_CANDIDATE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:EXACT-SOLVER-CANDIDATE:v1\0";
const PORTFOLIO_POLICY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:SOLVER-PORTFOLIO-POLICY:v1\0";
const PORTFOLIO_REPORT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:SOLVER-PORTFOLIO-REPORT:v1\0";
const CANDIDATE_EVALUATION_DOMAIN: &[u8] = b"CEREBRO:TIDEX:CANDIDATE-EVALUATION:v1\0";
const MAX_EXACT_CANDIDATE_DENSE_ELEMENTS: usize = 16_777_216;
const DEFAULT_MAX_NUMERICAL_WORK_UNITS: u64 = 250_000_000;
const ABSOLUTE_MAX_NUMERICAL_WORK_UNITS: u64 = 2_000_000_000;

/// Exact identity of one numerical problem. Unlike applicability/context
/// identities, this commits to dimensions and every IEEE-754 bit of X and Y.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct ExactSolverProblemDigest(Sha256Digest);

impl ExactSolverProblemDigest {
    pub fn as_digest(&self) -> &Sha256Digest {
        &self.0
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl std::fmt::Display for ExactSolverProblemDigest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Exact identity of one proposed numerical representation, including its
/// representation kind, dimensions and every IEEE-754 bit.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct ExactSolverCandidateDigest(Sha256Digest);

impl ExactSolverCandidateDigest {
    pub fn as_digest(&self) -> &Sha256Digest {
        &self.0
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

/// Exact identity of the complete numerical selection policy.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct SolverPortfolioPolicyDigest(Sha256Digest);

impl SolverPortfolioPolicyDigest {
    pub fn as_digest(&self) -> &Sha256Digest {
        &self.0
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

/// Core-computed identity of a complete solver run. It commits to the exact
/// problem, complete policy, diagnostics, every evaluated candidate's exact
/// representation digest, and the deterministic selection. External backend
/// labels are deliberately not treated as authenticated provenance; that
/// requires an independent execution receipt.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct SolverPortfolioReportDigest(Sha256Digest);

impl SolverPortfolioReportDigest {
    pub fn as_digest(&self) -> &Sha256Digest {
        &self.0
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl std::fmt::Display for SolverPortfolioReportDigest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Rectangular multi-response least-squares contract.
///
/// `inputs` is `[case, input]` and `targets` is `[case, output]`. Construction
/// validates shape and finiteness, while [`PortfolioPolicy`] supplies explicit
/// execution and memory bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct LeastSquaresProblem {
    inputs: Matrix,
    targets: Matrix,
}

impl LeastSquaresProblem {
    pub fn new(inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>) -> BrainResult<Self> {
        if inputs.is_empty() || targets.is_empty() || inputs.len() != targets.len() {
            return Err(BrainError::Invalid(
                "solver_portfolio_problem_case_shape".into(),
            ));
        }
        let inputs = Matrix::from_rows(&inputs)?;
        let targets = Matrix::from_rows(&targets)?;
        if inputs.row_count() == 0
            || inputs.column_count() == 0
            || targets.column_count() == 0
            || inputs.row_count() != targets.row_count()
        {
            return Err(BrainError::Invalid("solver_portfolio_problem_shape".into()));
        }
        Ok(Self { inputs, targets })
    }

    pub fn case_count(&self) -> usize {
        self.inputs.row_count()
    }

    pub fn input_dimension(&self) -> usize {
        self.inputs.column_count()
    }

    pub fn output_dimension(&self) -> usize {
        self.targets.column_count()
    }

    pub fn inputs(&self) -> &Matrix {
        &self.inputs
    }

    pub fn targets(&self) -> &Matrix {
        &self.targets
    }

    pub fn digest(&self) -> BrainResult<ExactSolverProblemDigest> {
        let cases = u64::try_from(self.case_count())
            .map_err(|_| BrainError::Invalid("solver_problem_case_overflow".into()))?;
        let inputs = u64::try_from(self.input_dimension())
            .map_err(|_| BrainError::Invalid("solver_problem_input_overflow".into()))?;
        let outputs = u64::try_from(self.output_dimension())
            .map_err(|_| BrainError::Invalid("solver_problem_output_overflow".into()))?;
        // Stream the exact frame into SHA-256 so even a problem rejected by a
        // resource gate can receive an immutable identity without allocating
        // a second copy of all of its scalar bytes.
        let mut hasher = Sha256::new();
        hasher.update(EXACT_PROBLEM_DOMAIN);
        hasher.update(cases.to_be_bytes());
        hasher.update(inputs.to_be_bytes());
        hasher.update(outputs.to_be_bytes());
        hasher.update([b'X']);
        for value in self.inputs.as_slice() {
            hasher.update(value.to_bits().to_be_bytes());
        }
        hasher.update([b'Y']);
        for value in self.targets.as_slice() {
            hasher.update(value.to_bits().to_be_bytes());
        }
        Ok(ExactSolverProblemDigest(Sha256Digest::parse(format!(
            "{:x}",
            hasher.finalize()
        ))?))
    }
}

/// Hard resource ceilings. Exceeding one yields [`SolverStatus::BoundedUnknown`]
/// rather than silently truncating the problem or claiming failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverResourceLimits {
    max_cases: usize,
    max_input_dimension: usize,
    max_output_dimension: usize,
    max_working_elements: usize,
    max_candidate_parameters: usize,
    max_external_proposals: usize,
    /// Conservative upper bound for scalar work in the direct Jacobi SVD.
    /// This is deliberately a work unit, not a claim about measured FLOPs.
    max_numerical_work_units: u64,
}

impl Default for SolverResourceLimits {
    fn default() -> Self {
        Self {
            max_cases: 64,
            max_input_dimension: 4096,
            max_output_dimension: 4096,
            max_working_elements: 16_777_216,
            max_candidate_parameters: 16_777_216,
            max_external_proposals: 16,
            max_numerical_work_units: DEFAULT_MAX_NUMERICAL_WORK_UNITS,
        }
    }
}

impl SolverResourceLimits {
    pub fn new(
        max_cases: usize,
        max_input_dimension: usize,
        max_output_dimension: usize,
        max_working_elements: usize,
        max_candidate_parameters: usize,
        max_external_proposals: usize,
    ) -> BrainResult<Self> {
        let limits = Self {
            max_cases,
            max_input_dimension,
            max_output_dimension,
            max_working_elements,
            max_candidate_parameters,
            max_external_proposals,
            max_numerical_work_units: DEFAULT_MAX_NUMERICAL_WORK_UNITS,
        };
        limits.validate()?;
        Ok(limits)
    }

    fn validate(&self) -> BrainResult<()> {
        if self.max_cases == 0
            || self.max_input_dimension == 0
            || self.max_output_dimension == 0
            || self.max_working_elements == 0
            || self.max_candidate_parameters == 0
            || self.max_working_elements > MAX_EXACT_CANDIDATE_DENSE_ELEMENTS
            || self.max_candidate_parameters > MAX_EXACT_CANDIDATE_DENSE_ELEMENTS
            || self.max_numerical_work_units == 0
            || self.max_numerical_work_units > ABSOLUTE_MAX_NUMERICAL_WORK_UNITS
        {
            return Err(BrainError::Invalid("solver_resource_limits_invalid".into()));
        }
        Ok(())
    }

    pub fn max_cases(&self) -> usize {
        self.max_cases
    }

    pub fn max_input_dimension(&self) -> usize {
        self.max_input_dimension
    }

    pub fn max_output_dimension(&self) -> usize {
        self.max_output_dimension
    }

    pub fn max_working_elements(&self) -> usize {
        self.max_working_elements
    }

    pub fn max_candidate_parameters(&self) -> usize {
        self.max_candidate_parameters
    }

    pub fn max_external_proposals(&self) -> usize {
        self.max_external_proposals
    }

    pub fn with_max_numerical_work_units(mut self, value: u64) -> BrainResult<Self> {
        self.max_numerical_work_units = value;
        self.validate()?;
        Ok(self)
    }

    pub fn max_numerical_work_units(&self) -> u64 {
        self.max_numerical_work_units
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioPolicy {
    /// Cholesky is forbidden when the unregularized Gram condition exceeds
    /// this value, even though damping could make the regularized system SPD.
    cholesky_max_condition: f64,
    cholesky_damping: f64,
    /// Relative eigenvalue threshold used to determine numerical row rank.
    relative_rank_tolerance: f64,
    /// Minimum Gram energy that an adaptively selected spectral rank must
    /// explain before residual-based expansion begins.
    target_explained_energy: f64,
    minimum_rank: usize,
    maximum_rank: usize,
    relative_residual_tolerance: f64,
    absolute_residual_tolerance: f64,
    svd_orthogonality_tolerance: f64,
    max_svd_sweeps: usize,
    /// Bounds finite but extreme input magnitudes before forming a Gram matrix.
    max_absolute_value: f64,
    limits: SolverResourceLimits,
}

impl Default for PortfolioPolicy {
    fn default() -> Self {
        Self {
            cholesky_max_condition: 1.0e6,
            cholesky_damping: 1.0e-8,
            relative_rank_tolerance: 1.0e-10,
            target_explained_energy: 0.999,
            minimum_rank: 1,
            maximum_rank: 64,
            relative_residual_tolerance: 1.0e-8,
            absolute_residual_tolerance: 1.0e-10,
            svd_orthogonality_tolerance: 1.0e-12,
            max_svd_sweeps: 100,
            max_absolute_value: 1.0e100,
            limits: SolverResourceLimits::default(),
        }
    }
}

impl PortfolioPolicy {
    fn validate(&self) -> BrainResult<()> {
        if !self.cholesky_max_condition.is_finite()
            || self.cholesky_max_condition < 1.0
            || !self.cholesky_damping.is_finite()
            || self.cholesky_damping <= 0.0
            || !self.relative_rank_tolerance.is_finite()
            || !(0.0..1.0).contains(&self.relative_rank_tolerance)
            || !self.target_explained_energy.is_finite()
            || !(0.0..=1.0).contains(&self.target_explained_energy)
            || self.target_explained_energy == 0.0
            || self.minimum_rank == 0
            || self.maximum_rank < self.minimum_rank
            || !self.relative_residual_tolerance.is_finite()
            || self.relative_residual_tolerance < 0.0
            || !self.absolute_residual_tolerance.is_finite()
            || self.absolute_residual_tolerance < 0.0
            || self.relative_residual_tolerance == 0.0 && self.absolute_residual_tolerance == 0.0
            || !self.svd_orthogonality_tolerance.is_finite()
            || !(0.0..1.0).contains(&self.svd_orthogonality_tolerance)
            || self.max_svd_sweeps == 0
            || !self.max_absolute_value.is_finite()
            || self.max_absolute_value <= 0.0
        {
            return Err(BrainError::Invalid(
                "solver_portfolio_policy_invalid".into(),
            ));
        }
        self.limits.validate()?;
        Ok(())
    }

    pub fn with_cholesky_max_condition(mut self, value: f64) -> BrainResult<Self> {
        self.cholesky_max_condition = value;
        self.validate()?;
        Ok(self)
    }

    pub fn with_cholesky_damping(mut self, value: f64) -> BrainResult<Self> {
        self.cholesky_damping = value;
        self.validate()?;
        Ok(self)
    }

    pub fn with_rank_policy(
        mut self,
        relative_tolerance: f64,
        target_explained_energy: f64,
        minimum_rank: usize,
        maximum_rank: usize,
    ) -> BrainResult<Self> {
        self.relative_rank_tolerance = relative_tolerance;
        self.target_explained_energy = target_explained_energy;
        self.minimum_rank = minimum_rank;
        self.maximum_rank = maximum_rank;
        self.validate()?;
        Ok(self)
    }

    pub fn with_residual_tolerances(mut self, relative: f64, absolute: f64) -> BrainResult<Self> {
        self.relative_residual_tolerance = relative;
        self.absolute_residual_tolerance = absolute;
        self.validate()?;
        Ok(self)
    }

    pub fn with_svd_convergence(
        mut self,
        orthogonality_tolerance: f64,
        max_sweeps: usize,
    ) -> BrainResult<Self> {
        self.svd_orthogonality_tolerance = orthogonality_tolerance;
        self.max_svd_sweeps = max_sweeps;
        self.validate()?;
        Ok(self)
    }

    pub fn with_max_absolute_value(mut self, value: f64) -> BrainResult<Self> {
        self.max_absolute_value = value;
        self.validate()?;
        Ok(self)
    }

    pub fn with_limits(mut self, limits: SolverResourceLimits) -> BrainResult<Self> {
        self.limits = limits;
        self.validate()?;
        Ok(self)
    }

    pub fn cholesky_max_condition(&self) -> f64 {
        self.cholesky_max_condition
    }

    pub fn cholesky_damping(&self) -> f64 {
        self.cholesky_damping
    }

    pub fn relative_rank_tolerance(&self) -> f64 {
        self.relative_rank_tolerance
    }

    pub fn target_explained_energy(&self) -> f64 {
        self.target_explained_energy
    }

    pub fn minimum_rank(&self) -> usize {
        self.minimum_rank
    }

    pub fn maximum_rank(&self) -> usize {
        self.maximum_rank
    }

    pub fn relative_residual_tolerance(&self) -> f64 {
        self.relative_residual_tolerance
    }

    pub fn absolute_residual_tolerance(&self) -> f64 {
        self.absolute_residual_tolerance
    }

    pub fn svd_orthogonality_tolerance(&self) -> f64 {
        self.svd_orthogonality_tolerance
    }

    pub fn max_svd_sweeps(&self) -> usize {
        self.max_svd_sweeps
    }

    pub fn max_absolute_value(&self) -> f64 {
        self.max_absolute_value
    }

    pub fn limits(&self) -> &SolverResourceLimits {
        &self.limits
    }

    pub fn digest(&self) -> BrainResult<SolverPortfolioPolicyDigest> {
        self.validate()?;
        let usize_value = |value: usize, label: &str| {
            u64::try_from(value).map_err(|_| {
                BrainError::Invalid(format!("solver_portfolio_policy_{label}_overflow"))
            })
        };
        let mut frame = Vec::with_capacity(16 * std::mem::size_of::<u64>());
        for value in [
            self.cholesky_max_condition,
            self.cholesky_damping,
            self.relative_rank_tolerance,
            self.target_explained_energy,
            self.relative_residual_tolerance,
            self.absolute_residual_tolerance,
            self.svd_orthogonality_tolerance,
            self.max_absolute_value,
        ] {
            frame.extend_from_slice(&value.to_bits().to_be_bytes());
        }
        for value in [
            usize_value(self.minimum_rank, "minimum_rank")?,
            usize_value(self.maximum_rank, "maximum_rank")?,
            usize_value(self.max_svd_sweeps, "svd_sweeps")?,
            usize_value(self.limits.max_cases, "max_cases")?,
            usize_value(self.limits.max_input_dimension, "max_input_dimension")?,
            usize_value(self.limits.max_output_dimension, "max_output_dimension")?,
            usize_value(self.limits.max_working_elements, "max_working_elements")?,
            usize_value(
                self.limits.max_candidate_parameters,
                "max_candidate_parameters",
            )?,
            usize_value(self.limits.max_external_proposals, "max_external_proposals")?,
            self.limits.max_numerical_work_units,
        ] {
            frame.extend_from_slice(&value.to_be_bytes());
        }
        Ok(SolverPortfolioPolicyDigest(Sha256Digest::digest_domain(
            PORTFOLIO_POLICY_DOMAIN,
            &frame,
        )))
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum NumericalBackendRole {
    BuiltInCholeskyRidge,
    BuiltInDirectJacobiSvd,
    /// Neutral role for all untrusted backend proposals. Sparse/block/dense
    /// structure is proven by [`CandidateRepresentation`], never by this label.
    ExternalProposal,
}

impl NumericalBackendRole {
    fn is_external(self) -> bool {
        self == Self::ExternalProposal
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDescriptor {
    role: NumericalBackendRole,
    implementation: String,
}

impl BackendDescriptor {
    fn external(implementation: impl Into<String>) -> BrainResult<Self> {
        let descriptor = Self {
            role: NumericalBackendRole::ExternalProposal,
            implementation: implementation.into(),
        };
        descriptor.validate(true)?;
        Ok(descriptor)
    }

    fn builtin(role: NumericalBackendRole, implementation: &'static str) -> Self {
        Self {
            role,
            implementation: implementation.into(),
        }
    }

    pub fn role(&self) -> NumericalBackendRole {
        self.role
    }

    pub fn implementation(&self) -> &str {
        &self.implementation
    }

    fn validate(&self, external: bool) -> BrainResult<()> {
        let name = self.implementation.trim();
        if name.is_empty()
            || name.len() > MAX_BACKEND_NAME_BYTES
            || name.chars().any(char::is_control)
            || external && !self.role.is_external()
        {
            return Err(BrainError::Invalid(
                "solver_backend_descriptor_invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseEntry {
    row: usize,
    column: usize,
    value: f64,
}

impl SparseEntry {
    pub fn new(row: usize, column: usize, value: f64) -> BrainResult<Self> {
        if !value.is_finite() || value == 0.0 {
            return Err(BrainError::Invalid("solver_sparse_entry_invalid".into()));
        }
        Ok(Self { row, column, value })
    }

    pub fn row(&self) -> usize {
        self.row
    }

    pub fn column(&self) -> usize {
        self.column
    }

    pub fn value(&self) -> f64 {
        self.value
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseBlock {
    row_offset: usize,
    column_offset: usize,
    rows: usize,
    columns: usize,
    values: Vec<f64>,
}

impl DenseBlock {
    pub fn new(
        row_offset: usize,
        column_offset: usize,
        rows: usize,
        columns: usize,
        values: Vec<f64>,
    ) -> BrainResult<Self> {
        let expected = rows
            .checked_mul(columns)
            .ok_or_else(|| BrainError::Invalid("solver_block_shape_overflow".into()))?;
        if rows == 0
            || columns == 0
            || values.len() != expected
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(BrainError::Invalid("solver_block_invalid".into()));
        }
        Ok(Self {
            row_offset,
            column_offset,
            rows,
            columns,
            values,
        })
    }

    pub fn row_offset(&self) -> usize {
        self.row_offset
    }

    pub fn column_offset(&self) -> usize {
        self.column_offset
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// Candidate representation only. It deliberately contains no acceptance,
/// evidence, signature, promotion, or residency field. Sparse and block
/// claims are structural: the authority validates canonical coordinates and
/// materializes their exact dense semantics. A hybrid software/weights system
/// is intentionally absent because it belongs to residency governance, not a
/// pure least-squares solver.
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateRepresentation {
    /// `left` is row-major `[output, rank]`; `right` is `[rank, input]`.
    LowRank {
        rows: usize,
        columns: usize,
        rank: usize,
        left: Vec<f64>,
        right: Vec<f64>,
    },
    /// Row-major `[output, input]` update.
    Dense {
        rows: usize,
        columns: usize,
        weights: Vec<f64>,
    },
    /// Canonically ordered, unique, nonzero coordinates.
    Sparse {
        rows: usize,
        columns: usize,
        entries: Vec<SparseEntry>,
    },
    /// Canonically ordered, non-overlapping dense rectangles.
    Block {
        rows: usize,
        columns: usize,
        blocks: Vec<DenseBlock>,
    },
}

impl CandidateRepresentation {
    pub fn rows(&self) -> usize {
        match self {
            Self::LowRank { rows, .. }
            | Self::Dense { rows, .. }
            | Self::Sparse { rows, .. }
            | Self::Block { rows, .. } => *rows,
        }
    }

    pub fn columns(&self) -> usize {
        match self {
            Self::LowRank { columns, .. }
            | Self::Dense { columns, .. }
            | Self::Sparse { columns, .. }
            | Self::Block { columns, .. } => *columns,
        }
    }

    pub fn rank(&self) -> Option<usize> {
        match self {
            Self::LowRank { rank, .. } => Some(*rank),
            Self::Dense { .. } | Self::Sparse { .. } | Self::Block { .. } => None,
        }
    }

    pub fn stored_parameter_count(&self) -> Option<usize> {
        match self {
            Self::LowRank {
                rows,
                columns,
                rank,
                ..
            } => rows
                .checked_mul(*rank)?
                .checked_add(rank.checked_mul(*columns)?),
            Self::Dense { rows, columns, .. } => rows.checked_mul(*columns),
            Self::Sparse { entries, .. } => Some(entries.len()),
            Self::Block { blocks, .. } => blocks.iter().try_fold(0_usize, |total, block| {
                total.checked_add(block.values.len())
            }),
        }
    }

    pub fn materialize_dense(&self) -> BrainResult<Vec<f64>> {
        let dense_len = self
            .rows()
            .checked_mul(self.columns())
            .ok_or_else(|| BrainError::Invalid("solver_candidate_shape_overflow".into()))?;
        if dense_len > MAX_EXACT_CANDIDATE_DENSE_ELEMENTS {
            return Err(BrainError::Invalid(
                "solver_candidate_dense_materialization_limit".into(),
            ));
        }
        match self {
            Self::Dense {
                rows,
                columns,
                weights,
            } => {
                let expected = rows
                    .checked_mul(*columns)
                    .ok_or_else(|| BrainError::Invalid("solver_candidate_shape_overflow".into()))?;
                if weights.len() != expected || weights.iter().any(|value| !value.is_finite()) {
                    return Err(BrainError::Integrity(
                        "solver_dense_candidate_invalid".into(),
                    ));
                }
                Ok(weights.clone())
            }
            Self::LowRank {
                rows,
                columns,
                rank,
                left,
                right,
            } => {
                if *rank == 0
                    || left.len() != rows.checked_mul(*rank).unwrap_or(usize::MAX)
                    || right.len() != rank.checked_mul(*columns).unwrap_or(usize::MAX)
                    || left.iter().chain(right).any(|value| !value.is_finite())
                {
                    return Err(BrainError::Integrity(
                        "solver_low_rank_candidate_invalid".into(),
                    ));
                }
                let mut dense = vec![0.0; dense_len];
                for row in 0..*rows {
                    for column in 0..*columns {
                        let value = compensated_sum((0..*rank).map(|component| {
                            left[row * *rank + component] * right[component * *columns + column]
                        }))?;
                        if !value.is_finite() {
                            return Err(BrainError::Numerical(
                                "solver_candidate_materialization_nonfinite".into(),
                            ));
                        }
                        dense[row * *columns + column] = value;
                    }
                }
                Ok(dense)
            }
            Self::Sparse {
                rows,
                columns,
                entries,
            } => {
                let mut dense = vec![0.0; dense_len];
                let mut previous = None;
                for entry in entries {
                    let coordinate = (entry.row, entry.column);
                    if entry.row >= *rows
                        || entry.column >= *columns
                        || !entry.value.is_finite()
                        || entry.value == 0.0
                        || previous.is_some_and(|prior| prior >= coordinate)
                    {
                        return Err(BrainError::Integrity(
                            "solver_sparse_candidate_not_canonical".into(),
                        ));
                    }
                    dense[entry.row * *columns + entry.column] = entry.value;
                    previous = Some(coordinate);
                }
                Ok(dense)
            }
            Self::Block {
                rows,
                columns,
                blocks,
            } => {
                let mut dense = vec![0.0; dense_len];
                let mut occupied = vec![false; dense_len];
                let mut previous = None;
                for block in blocks {
                    let coordinate = (block.row_offset, block.column_offset);
                    let row_end = block.row_offset.checked_add(block.rows).ok_or_else(|| {
                        BrainError::Integrity("solver_block_candidate_overflow".into())
                    })?;
                    let column_end =
                        block
                            .column_offset
                            .checked_add(block.columns)
                            .ok_or_else(|| {
                                BrainError::Integrity("solver_block_candidate_overflow".into())
                            })?;
                    if block.rows == 0
                        || block.columns == 0
                        || row_end > *rows
                        || column_end > *columns
                        || block.values.len() != block.rows.saturating_mul(block.columns)
                        || block.values.iter().any(|value| !value.is_finite())
                        || previous.is_some_and(|prior| prior >= coordinate)
                    {
                        return Err(BrainError::Integrity(
                            "solver_block_candidate_not_canonical".into(),
                        ));
                    }
                    for local_row in 0..block.rows {
                        for local_column in 0..block.columns {
                            let dense_index = (block.row_offset + local_row) * *columns
                                + block.column_offset
                                + local_column;
                            if occupied[dense_index] {
                                return Err(BrainError::Integrity(
                                    "solver_block_candidate_overlap".into(),
                                ));
                            }
                            occupied[dense_index] = true;
                            dense[dense_index] =
                                block.values[local_row * block.columns + local_column];
                        }
                    }
                    previous = Some(coordinate);
                }
                Ok(dense)
            }
        }
    }

    pub fn exact_digest(&self) -> BrainResult<ExactSolverCandidateDigest> {
        // Reuse the full structural validator before committing an identity.
        // This also proves that all encoded values are finite and all sparse
        // or block coordinates are canonical.
        let dense_elements = self
            .rows()
            .checked_mul(self.columns())
            .ok_or_else(|| BrainError::Invalid("solver_candidate_shape_overflow".into()))?;
        if dense_elements > MAX_EXACT_CANDIDATE_DENSE_ELEMENTS {
            return Err(BrainError::Invalid(
                "solver_candidate_digest_resource_limit".into(),
            ));
        }
        self.materialize_dense()?;
        let mut frame = Vec::new();
        append_usize(&mut frame, self.rows(), "candidate_rows")?;
        append_usize(&mut frame, self.columns(), "candidate_columns")?;
        match self {
            Self::LowRank {
                rank, left, right, ..
            } => {
                frame.push(0);
                append_usize(&mut frame, *rank, "candidate_rank")?;
                append_f64_slice(&mut frame, left, "candidate_left")?;
                append_f64_slice(&mut frame, right, "candidate_right")?;
            }
            Self::Dense { weights, .. } => {
                frame.push(1);
                append_f64_slice(&mut frame, weights, "candidate_dense")?;
            }
            Self::Sparse { entries, .. } => {
                frame.push(2);
                append_usize(&mut frame, entries.len(), "candidate_sparse_count")?;
                for entry in entries {
                    append_usize(&mut frame, entry.row, "candidate_sparse_row")?;
                    append_usize(&mut frame, entry.column, "candidate_sparse_column")?;
                    frame.extend_from_slice(&entry.value.to_bits().to_be_bytes());
                }
            }
            Self::Block { blocks, .. } => {
                frame.push(3);
                append_usize(&mut frame, blocks.len(), "candidate_block_count")?;
                for block in blocks {
                    append_usize(&mut frame, block.row_offset, "candidate_block_row")?;
                    append_usize(&mut frame, block.column_offset, "candidate_block_column")?;
                    append_usize(&mut frame, block.rows, "candidate_block_rows")?;
                    append_usize(&mut frame, block.columns, "candidate_block_columns")?;
                    append_f64_slice(&mut frame, &block.values, "candidate_block_values")?;
                }
            }
        }
        Ok(ExactSolverCandidateDigest(Sha256Digest::digest_domain(
            EXACT_CANDIDATE_DOMAIN,
            &frame,
        )))
    }
}

fn append_usize(frame: &mut Vec<u8>, value: usize, label: &str) -> BrainResult<()> {
    let value = u64::try_from(value)
        .map_err(|_| BrainError::Invalid(format!("solver_{label}_overflow")))?;
    frame.extend_from_slice(&value.to_be_bytes());
    Ok(())
}

fn append_f64_slice(frame: &mut Vec<u8>, values: &[f64], label: &str) -> BrainResult<()> {
    append_usize(frame, values.len(), label)?;
    for value in values {
        if !value.is_finite() {
            return Err(BrainError::Invalid(format!("solver_{label}_nonfinite")));
        }
        frame.extend_from_slice(&value.to_bits().to_be_bytes());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SolverStatus {
    Accepted,
    Rejected,
    BoundedUnknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateMetrics {
    absolute_residual: f64,
    root_mean_square_residual: f64,
    maximum_absolute_residual: f64,
    /// Undefined when the target norm is zero and the residual is nonzero.
    relative_residual: Option<f64>,
    target_norm: f64,
    frobenius_norm: f64,
    stored_parameter_count: usize,
}

impl CandidateMetrics {
    pub fn absolute_residual(&self) -> f64 {
        self.absolute_residual
    }

    pub fn root_mean_square_residual(&self) -> f64 {
        self.root_mean_square_residual
    }

    pub fn maximum_absolute_residual(&self) -> f64 {
        self.maximum_absolute_residual
    }

    pub fn relative_residual(&self) -> Option<f64> {
        self.relative_residual
    }

    pub fn target_norm(&self) -> f64 {
        self.target_norm
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.frobenius_norm
    }

    pub fn stored_parameter_count(&self) -> usize {
        self.stored_parameter_count
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GramCondition {
    Finite,
    RankDeficient,
    Degenerate,
    Unrepresentable,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NumericalDiagnostics {
    singular_values: Vec<f64>,
    gram_spectrum: Vec<f64>,
    largest_singular_value: f64,
    numerical_rank: usize,
    effective_rank: f64,
    energy_rank: usize,
    singular_value_cutoff: f64,
    condition_kind: GramCondition,
    finite_singular_condition_number: Option<f64>,
    finite_gram_condition_number: Option<f64>,
    direct_svd_sweeps: usize,
    maximum_scaled_column_correlation: f64,
    direct_svd_reconstruction_relative_error: f64,
    direct_svd_vector_orthogonality_error: f64,
}

impl NumericalDiagnostics {
    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    pub fn gram_spectrum(&self) -> &[f64] {
        &self.gram_spectrum
    }

    pub fn largest_singular_value(&self) -> f64 {
        self.largest_singular_value
    }

    pub fn numerical_rank(&self) -> usize {
        self.numerical_rank
    }

    pub fn effective_rank(&self) -> f64 {
        self.effective_rank
    }

    pub fn energy_rank(&self) -> usize {
        self.energy_rank
    }

    pub fn singular_value_cutoff(&self) -> f64 {
        self.singular_value_cutoff
    }

    pub fn condition_kind(&self) -> &GramCondition {
        &self.condition_kind
    }

    pub fn finite_singular_condition_number(&self) -> Option<f64> {
        self.finite_singular_condition_number
    }

    pub fn finite_gram_condition_number(&self) -> Option<f64> {
        self.finite_gram_condition_number
    }

    pub fn direct_svd_sweeps(&self) -> usize {
        self.direct_svd_sweeps
    }

    pub fn maximum_scaled_column_correlation(&self) -> f64 {
        self.maximum_scaled_column_correlation
    }

    pub fn direct_svd_reconstruction_relative_error(&self) -> f64 {
        self.direct_svd_reconstruction_relative_error
    }

    pub fn direct_svd_vector_orthogonality_error(&self) -> f64 {
        self.direct_svd_vector_orthogonality_error
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CholeskyGate {
    Authorized,
    Degenerate,
    RankDeficient,
    ConditionExceedsPolicy,
    CaseCountInsufficient,
    CaseLimitExceeded,
    Float32ConversionUnsafe,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum EvaluationReason {
    ResidualWithinTolerance,
    ResidualExceedsTolerance,
    CholeskyNotAuthorized(CholeskyGate),
    RankBudgetInsufficient,
    ResourceLimit(&'static str),
    BackendNotApplicable(String),
    BackendBoundedUnknown(String),
    BackendFailure(String),
    InvalidCandidate(String),
    DirectSvdDidNotConverge,
    ResidualToleranceUnrepresentable,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateEvaluation {
    backend: BackendDescriptor,
    status: SolverStatus,
    reason: EvaluationReason,
    /// Spectral/component rank used to construct the proposal, including when
    /// dense storage is cheaper than retaining its factors.
    constructed_rank: Option<usize>,
    candidate: Option<CandidateRepresentation>,
    metrics: Option<CandidateMetrics>,
}

impl CandidateEvaluation {
    pub fn backend(&self) -> &BackendDescriptor {
        &self.backend
    }

    pub fn status(&self) -> SolverStatus {
        self.status
    }

    pub fn reason(&self) -> &EvaluationReason {
        &self.reason
    }

    pub fn constructed_rank(&self) -> Option<usize> {
        self.constructed_rank
    }

    pub fn candidate(&self) -> Option<&CandidateRepresentation> {
        self.candidate.as_ref()
    }

    pub fn metrics(&self) -> Option<&CandidateMetrics> {
        self.metrics.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolverPortfolioReport {
    problem_digest: ExactSolverProblemDigest,
    policy_digest: SolverPortfolioPolicyDigest,
    status: SolverStatus,
    reason: EvaluationReason,
    diagnostics: Option<NumericalDiagnostics>,
    cholesky_gate: Option<CholeskyGate>,
    evaluations: Vec<CandidateEvaluation>,
    selected_evaluation: Option<usize>,
    external_proposals_supplied: usize,
    external_proposals_considered: usize,
    report_digest: SolverPortfolioReportDigest,
}

#[derive(Serialize)]
struct CandidateMetricsProjection {
    absolute_residual_bits: u64,
    root_mean_square_residual_bits: u64,
    maximum_absolute_residual_bits: u64,
    relative_residual_bits: Option<u64>,
    target_norm_bits: u64,
    frobenius_norm_bits: u64,
    stored_parameter_count: usize,
}

impl From<&CandidateMetrics> for CandidateMetricsProjection {
    fn from(metrics: &CandidateMetrics) -> Self {
        Self {
            absolute_residual_bits: metrics.absolute_residual.to_bits(),
            root_mean_square_residual_bits: metrics.root_mean_square_residual.to_bits(),
            maximum_absolute_residual_bits: metrics.maximum_absolute_residual.to_bits(),
            relative_residual_bits: metrics.relative_residual.map(f64::to_bits),
            target_norm_bits: metrics.target_norm.to_bits(),
            frobenius_norm_bits: metrics.frobenius_norm.to_bits(),
            stored_parameter_count: metrics.stored_parameter_count,
        }
    }
}

#[derive(Serialize)]
struct NumericalDiagnosticsProjection {
    singular_value_bits: Vec<u64>,
    gram_spectrum_bits: Vec<u64>,
    largest_singular_value_bits: u64,
    numerical_rank: usize,
    effective_rank_bits: u64,
    energy_rank: usize,
    singular_value_cutoff_bits: u64,
    condition_kind: GramCondition,
    finite_singular_condition_number_bits: Option<u64>,
    finite_gram_condition_number_bits: Option<u64>,
    direct_svd_sweeps: usize,
    maximum_scaled_column_correlation_bits: u64,
    direct_svd_reconstruction_relative_error_bits: u64,
    direct_svd_vector_orthogonality_error_bits: u64,
}

impl From<&NumericalDiagnostics> for NumericalDiagnosticsProjection {
    fn from(diagnostics: &NumericalDiagnostics) -> Self {
        Self {
            singular_value_bits: diagnostics
                .singular_values
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            gram_spectrum_bits: diagnostics
                .gram_spectrum
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            largest_singular_value_bits: diagnostics.largest_singular_value.to_bits(),
            numerical_rank: diagnostics.numerical_rank,
            effective_rank_bits: diagnostics.effective_rank.to_bits(),
            energy_rank: diagnostics.energy_rank,
            singular_value_cutoff_bits: diagnostics.singular_value_cutoff.to_bits(),
            condition_kind: diagnostics.condition_kind.clone(),
            finite_singular_condition_number_bits: diagnostics
                .finite_singular_condition_number
                .map(f64::to_bits),
            finite_gram_condition_number_bits: diagnostics
                .finite_gram_condition_number
                .map(f64::to_bits),
            direct_svd_sweeps: diagnostics.direct_svd_sweeps,
            maximum_scaled_column_correlation_bits: diagnostics
                .maximum_scaled_column_correlation
                .to_bits(),
            direct_svd_reconstruction_relative_error_bits: diagnostics
                .direct_svd_reconstruction_relative_error
                .to_bits(),
            direct_svd_vector_orthogonality_error_bits: diagnostics
                .direct_svd_vector_orthogonality_error
                .to_bits(),
        }
    }
}

#[derive(Serialize)]
struct CandidateEvaluationProjection {
    backend_role: NumericalBackendRole,
    status: SolverStatus,
    reason: EvaluationReason,
    constructed_rank: Option<usize>,
    candidate_digest: Option<ExactSolverCandidateDigest>,
    metrics: Option<CandidateMetricsProjection>,
}

impl CandidateEvaluationProjection {
    fn from_evaluation(evaluation: &CandidateEvaluation) -> BrainResult<Self> {
        Ok(Self {
            // The role records which governed path ran. The implementation
            // string of an external backend is only a declaration and cannot
            // become authenticated provenance by entering this digest.
            backend_role: evaluation.backend.role,
            status: evaluation.status,
            reason: evaluation.reason.clone(),
            constructed_rank: evaluation.constructed_rank,
            candidate_digest: evaluation
                .candidate
                .as_ref()
                .map(CandidateRepresentation::exact_digest)
                .transpose()?,
            metrics: evaluation
                .metrics
                .as_ref()
                .map(CandidateMetricsProjection::from),
        })
    }
}

#[derive(Serialize)]
struct SolverPortfolioReportProjection<'a> {
    problem_digest: &'a ExactSolverProblemDigest,
    policy_digest: &'a SolverPortfolioPolicyDigest,
    status: SolverStatus,
    reason: &'a EvaluationReason,
    diagnostics: Option<NumericalDiagnosticsProjection>,
    cholesky_gate: &'a Option<CholeskyGate>,
    evaluation_digests: Vec<Sha256Digest>,
    selected_evaluation_digest: Option<Sha256Digest>,
    external_proposals_supplied: usize,
    external_proposals_considered: usize,
}

impl SolverPortfolioReport {
    pub fn problem_digest(&self) -> &ExactSolverProblemDigest {
        &self.problem_digest
    }

    pub fn policy_digest(&self) -> &SolverPortfolioPolicyDigest {
        &self.policy_digest
    }

    /// Return the exact run digest only after recomputing and authenticating
    /// the complete canonical projection.
    pub fn exact_digest(&self) -> BrainResult<&SolverPortfolioReportDigest> {
        self.authenticate()?;
        Ok(&self.report_digest)
    }

    pub fn status(&self) -> SolverStatus {
        self.status
    }

    pub fn reason(&self) -> &EvaluationReason {
        &self.reason
    }

    pub fn diagnostics(&self) -> Option<&NumericalDiagnostics> {
        self.diagnostics.as_ref()
    }

    pub fn cholesky_gate(&self) -> Option<&CholeskyGate> {
        self.cholesky_gate.as_ref()
    }

    pub fn evaluations(&self) -> &[CandidateEvaluation] {
        &self.evaluations
    }

    pub fn selected_evaluation_index(&self) -> Option<usize> {
        self.selected_evaluation
    }

    pub fn external_proposals_supplied(&self) -> usize {
        self.external_proposals_supplied
    }

    pub fn external_proposals_considered(&self) -> usize {
        self.external_proposals_considered
    }

    pub fn selected(&self) -> Option<&CandidateEvaluation> {
        self.selected_evaluation
            .and_then(|index| self.evaluations.get(index))
    }

    /// Canonical observed-but-unaccepted candidate for negative procedural
    /// evidence. This is not a portfolio selection. It is defined only when
    /// the candidate and its centrally recomputed metrics both exist.
    pub(crate) fn canonical_observed_rejection(&self) -> BrainResult<Option<&CandidateEvaluation>> {
        self.authenticate()?;
        if let Some(direct) = self.evaluations.iter().find(|evaluation| {
            evaluation.status == SolverStatus::Rejected
                && evaluation.backend.role() == NumericalBackendRole::BuiltInDirectJacobiSvd
                && evaluation.candidate.is_some()
                && evaluation.metrics.is_some()
        }) {
            return Ok(Some(direct));
        }
        let mut candidates = self
            .evaluations
            .iter()
            .filter(|evaluation| {
                evaluation.status == SolverStatus::Rejected
                    && evaluation.candidate.is_some()
                    && evaluation.metrics.is_some()
            })
            .map(|evaluation| {
                Ok((
                    evaluation
                        .candidate
                        .as_ref()
                        .ok_or_else(|| {
                            BrainError::Integrity(
                                "solver_observed_rejection_candidate_missing".into(),
                            )
                        })?
                        .exact_digest()?,
                    evaluation,
                ))
            })
            .collect::<BrainResult<Vec<_>>>()?;
        candidates.sort_by(|left, right| left.0.cmp(&right.0));
        Ok(candidates.first().map(|(_, evaluation)| *evaluation))
    }

    fn projection(&self) -> BrainResult<SolverPortfolioReportProjection<'_>> {
        let evaluation_digests = self
            .evaluations
            .iter()
            .map(|evaluation| {
                Ok(Sha256Digest::digest_domain(
                    CANDIDATE_EVALUATION_DOMAIN,
                    &serde_json::to_vec(&CandidateEvaluationProjection::from_evaluation(
                        evaluation,
                    )?)?,
                ))
            })
            .collect::<BrainResult<Vec<_>>>()?;
        let selected_evaluation_digest = self
            .selected_evaluation
            .map(|index| {
                evaluation_digests.get(index).cloned().ok_or_else(|| {
                    BrainError::Integrity("solver_selected_evaluation_out_of_bounds".into())
                })
            })
            .transpose()?;
        let mut canonical_evaluation_digests = evaluation_digests;
        canonical_evaluation_digests.sort();
        Ok(SolverPortfolioReportProjection {
            problem_digest: &self.problem_digest,
            policy_digest: &self.policy_digest,
            status: self.status,
            reason: &self.reason,
            diagnostics: self
                .diagnostics
                .as_ref()
                .map(NumericalDiagnosticsProjection::from),
            cholesky_gate: &self.cholesky_gate,
            evaluation_digests: canonical_evaluation_digests,
            selected_evaluation_digest,
            external_proposals_supplied: self.external_proposals_supplied,
            external_proposals_considered: self.external_proposals_considered,
        })
    }

    pub(crate) fn authenticate(&self) -> BrainResult<()> {
        let calculated = SolverPortfolioReportDigest(Sha256Digest::digest_domain(
            PORTFOLIO_REPORT_DOMAIN,
            &serde_json::to_vec(&self.projection()?)?,
        ));
        if calculated != self.report_digest {
            return Err(BrainError::Integrity(
                "solver_portfolio_report_digest_mismatch".into(),
            ));
        }
        Ok(())
    }
}

struct SolverPortfolioReportDraft {
    status: SolverStatus,
    reason: EvaluationReason,
    diagnostics: Option<NumericalDiagnostics>,
    cholesky_gate: Option<CholeskyGate>,
    evaluations: Vec<CandidateEvaluation>,
    selected_evaluation: Option<usize>,
    external_proposals_supplied: usize,
    external_proposals_considered: usize,
}

fn seal_portfolio_report(
    problem: &LeastSquaresProblem,
    policy: &PortfolioPolicy,
    draft: SolverPortfolioReportDraft,
) -> BrainResult<SolverPortfolioReport> {
    if draft
        .selected_evaluation
        .is_some_and(|index| index >= draft.evaluations.len())
    {
        return Err(BrainError::Integrity(
            "solver_selected_evaluation_out_of_bounds".into(),
        ));
    }
    let mut report = SolverPortfolioReport {
        problem_digest: problem.digest()?,
        policy_digest: policy.digest()?,
        status: draft.status,
        reason: draft.reason,
        diagnostics: draft.diagnostics,
        cholesky_gate: draft.cholesky_gate,
        evaluations: draft.evaluations,
        selected_evaluation: draft.selected_evaluation,
        external_proposals_supplied: draft.external_proposals_supplied,
        external_proposals_considered: draft.external_proposals_considered,
        report_digest: SolverPortfolioReportDigest(Sha256Digest::zero()),
    };
    report.report_digest = SolverPortfolioReportDigest(Sha256Digest::digest_domain(
        PORTFOLIO_REPORT_DOMAIN,
        &serde_json::to_vec(&report.projection()?)?,
    ));
    report.authenticate()?;
    Ok(report)
}

#[derive(Debug, Clone, PartialEq)]
enum ExternalProposalPayload {
    Candidate(CandidateRepresentation),
    NotApplicable(String),
    BoundedUnknown(String),
}

/// Inert data produced by an external runner or sandbox. The core never calls
/// plugin code in-process: it only validates this payload and recomputes all
/// candidate metrics. `declared_implementation` is diagnostic metadata, not
/// authenticated provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalCandidateProposal {
    descriptor: BackendDescriptor,
    payload: ExternalProposalPayload,
}

impl ExternalCandidateProposal {
    pub fn candidate(
        declared_implementation: impl Into<String>,
        candidate: CandidateRepresentation,
    ) -> BrainResult<Self> {
        Ok(Self {
            descriptor: BackendDescriptor::external(declared_implementation)?,
            payload: ExternalProposalPayload::Candidate(candidate),
        })
    }

    pub fn not_applicable(
        declared_implementation: impl Into<String>,
        reason: impl Into<String>,
    ) -> BrainResult<Self> {
        Self::with_message(declared_implementation, reason.into(), false)
    }

    pub fn bounded_unknown(
        declared_implementation: impl Into<String>,
        reason: impl Into<String>,
    ) -> BrainResult<Self> {
        Self::with_message(declared_implementation, reason.into(), true)
    }

    fn with_message(
        declared_implementation: impl Into<String>,
        message: String,
        bounded_unknown: bool,
    ) -> BrainResult<Self> {
        if message.trim().is_empty()
            || message.len() > MAX_BACKEND_REASON_BYTES
            || message.chars().any(char::is_control)
        {
            return Err(BrainError::Invalid(
                "solver_external_proposal_message_invalid".into(),
            ));
        }
        Ok(Self {
            descriptor: BackendDescriptor::external(declared_implementation)?,
            payload: if bounded_unknown {
                ExternalProposalPayload::BoundedUnknown(message)
            } else {
                ExternalProposalPayload::NotApplicable(message)
            },
        })
    }
}

/// Diagnose, generate, verify, and deterministically select bounded solver
/// candidates. This function never promotes or persists its selected proposal.
pub fn solve_with_portfolio(
    problem: &LeastSquaresProblem,
    policy: &PortfolioPolicy,
    external_proposals: &[ExternalCandidateProposal],
) -> BrainResult<SolverPortfolioReport> {
    policy.validate()?;
    problem.inputs.validate("solver_portfolio_inputs")?;
    problem.targets.validate("solver_portfolio_targets")?;

    if let Some(limit) = problem_limit(problem, policy) {
        return seal_portfolio_report(
            problem,
            policy,
            SolverPortfolioReportDraft {
                status: SolverStatus::BoundedUnknown,
                reason: limit.clone(),
                diagnostics: None,
                cholesky_gate: None,
                evaluations: Vec::new(),
                selected_evaluation: None,
                external_proposals_supplied: external_proposals.len(),
                external_proposals_considered: 0,
            },
        );
    }
    if let Some(limit) = external_proposal_limit(external_proposals, policy) {
        return seal_portfolio_report(
            problem,
            policy,
            SolverPortfolioReportDraft {
                status: SolverStatus::BoundedUnknown,
                reason: limit,
                diagnostics: None,
                cholesky_gate: None,
                evaluations: Vec::new(),
                selected_evaluation: None,
                external_proposals_supplied: external_proposals.len(),
                external_proposals_considered: 0,
            },
        );
    }

    if direct_solver_work_upper_bound(problem, policy)
        .is_none_or(|work| work > u128::from(policy.limits.max_numerical_work_units))
    {
        return seal_with_unavailable_direct_solver(
            problem,
            policy,
            external_proposals,
            EvaluationReason::ResourceLimit("solver_max_svd_work_units"),
        );
    }

    let direct_svd = match direct_one_sided_jacobi_svd(problem, policy)? {
        Some(decomposition) => decomposition,
        None => {
            return seal_with_unavailable_direct_solver(
                problem,
                policy,
                external_proposals,
                EvaluationReason::DirectSvdDidNotConverge,
            );
        }
    };
    let gram = row_gram(&problem.inputs)?;
    let diagnostics = diagnose_problem(&gram, &direct_svd, policy)?;
    let mut evaluations = Vec::new();
    let mut cholesky_gate = cholesky_eligibility(problem, &diagnostics, policy);
    if cholesky_gate == CholeskyGate::Authorized && !float32_problem_is_safe(problem) {
        cholesky_gate = CholeskyGate::Float32ConversionUnsafe;
    }
    if cholesky_gate == CholeskyGate::Authorized {
        evaluations.push(run_cholesky(problem, policy));
    } else {
        evaluations.push(CandidateEvaluation {
            backend: BackendDescriptor::builtin(
                NumericalBackendRole::BuiltInCholeskyRidge,
                BUILTIN_CHOLESKY_NAME,
            ),
            status: SolverStatus::Rejected,
            reason: EvaluationReason::CholeskyNotAuthorized(cholesky_gate.clone()),
            constructed_rank: None,
            candidate: None,
            metrics: None,
        });
    }

    evaluations.push(run_direct_svd(problem, &direct_svd, &diagnostics, policy)?);

    let considered = external_proposals
        .len()
        .min(policy.limits.max_external_proposals);
    for proposal in external_proposals.iter().take(considered) {
        evaluations.push(run_external(proposal, problem, policy));
    }

    let selected_evaluation = choose_accepted(&evaluations)?;
    let truncated_proposals = external_proposals.len() > considered;
    let (status, reason) = if let Some(index) = selected_evaluation {
        (SolverStatus::Accepted, evaluations[index].reason.clone())
    } else if truncated_proposals {
        (
            SolverStatus::BoundedUnknown,
            EvaluationReason::ResourceLimit("solver_external_proposal_budget"),
        )
    } else if let Some(unknown) = evaluations
        .iter()
        .find(|evaluation| evaluation.status == SolverStatus::BoundedUnknown)
    {
        (SolverStatus::BoundedUnknown, unknown.reason.clone())
    } else {
        (
            SolverStatus::Rejected,
            aggregate_rejection_reason(&evaluations)?,
        )
    };

    seal_portfolio_report(
        problem,
        policy,
        SolverPortfolioReportDraft {
            status,
            reason,
            diagnostics: Some(diagnostics),
            cholesky_gate: Some(cholesky_gate),
            evaluations,
            selected_evaluation,
            external_proposals_supplied: external_proposals.len(),
            external_proposals_considered: considered,
        },
    )
}

fn external_proposal_limit(
    proposals: &[ExternalCandidateProposal],
    policy: &PortfolioPolicy,
) -> Option<EvaluationReason> {
    if proposals.len() > policy.limits.max_external_proposals {
        return Some(EvaluationReason::ResourceLimit(
            "solver_external_proposal_budget",
        ));
    }
    let total_parameters = proposals.iter().try_fold(0_usize, |total, proposal| {
        let count = match &proposal.payload {
            ExternalProposalPayload::Candidate(candidate) => candidate.stored_parameter_count()?,
            ExternalProposalPayload::NotApplicable(_)
            | ExternalProposalPayload::BoundedUnknown(_) => 0,
        };
        total.checked_add(count)
    });
    match total_parameters {
        Some(total) if total <= policy.limits.max_candidate_parameters => None,
        _ => Some(EvaluationReason::ResourceLimit(
            "solver_max_total_external_parameters",
        )),
    }
}

fn seal_with_unavailable_direct_solver(
    problem: &LeastSquaresProblem,
    policy: &PortfolioPolicy,
    external_proposals: &[ExternalCandidateProposal],
    unavailable_reason: EvaluationReason,
) -> BrainResult<SolverPortfolioReport> {
    let mut evaluations = vec![CandidateEvaluation {
        backend: BackendDescriptor::builtin(
            NumericalBackendRole::BuiltInDirectJacobiSvd,
            BUILTIN_DIRECT_SVD_NAME,
        ),
        status: SolverStatus::BoundedUnknown,
        reason: unavailable_reason.clone(),
        constructed_rank: None,
        candidate: None,
        metrics: None,
    }];
    let considered = external_proposals
        .len()
        .min(policy.limits.max_external_proposals);
    for proposal in external_proposals.iter().take(considered) {
        evaluations.push(run_external(proposal, problem, policy));
    }
    let selected_evaluation = choose_accepted(&evaluations)?;
    let truncated = external_proposals.len() > considered;
    let (status, reason) = if let Some(index) = selected_evaluation {
        (SolverStatus::Accepted, evaluations[index].reason.clone())
    } else if truncated {
        (
            SolverStatus::BoundedUnknown,
            EvaluationReason::ResourceLimit("solver_external_proposal_budget"),
        )
    } else if let Some(unknown) = evaluations
        .iter()
        .find(|evaluation| evaluation.status == SolverStatus::BoundedUnknown)
    {
        (SolverStatus::BoundedUnknown, unknown.reason.clone())
    } else {
        return Err(BrainError::Integrity(
            "solver_unavailable_path_without_unknown".into(),
        ));
    };
    seal_portfolio_report(
        problem,
        policy,
        SolverPortfolioReportDraft {
            status,
            reason,
            diagnostics: None,
            cholesky_gate: None,
            evaluations,
            selected_evaluation,
            external_proposals_supplied: external_proposals.len(),
            external_proposals_considered: considered,
        },
    )
}

fn aggregate_rejection_reason(
    evaluations: &[CandidateEvaluation],
) -> BrainResult<EvaluationReason> {
    // The rank-revealing built-in is the portfolio's canonical fallback. Its
    // verified rejection explains why the portfolio itself has no candidate;
    // an unrelated external malformed proposal must not overwrite that cause.
    evaluations
        .iter()
        .find(|evaluation| {
            evaluation.status == SolverStatus::Rejected
                && evaluation.backend.role() == NumericalBackendRole::BuiltInDirectJacobiSvd
        })
        .or_else(|| {
            evaluations
                .iter()
                .find(|evaluation| evaluation.status == SolverStatus::Rejected)
        })
        .map(|evaluation| evaluation.reason.clone())
        .ok_or_else(|| BrainError::Integrity("solver_rejected_without_rejection_reason".into()))
}

fn problem_limit(
    problem: &LeastSquaresProblem,
    policy: &PortfolioPolicy,
) -> Option<EvaluationReason> {
    let limits = &policy.limits;
    if problem.case_count() > limits.max_cases {
        return Some(EvaluationReason::ResourceLimit("solver_max_cases"));
    }
    if problem.input_dimension() > limits.max_input_dimension {
        return Some(EvaluationReason::ResourceLimit(
            "solver_max_input_dimension",
        ));
    }
    if problem.output_dimension() > limits.max_output_dimension {
        return Some(EvaluationReason::ResourceLimit(
            "solver_max_output_dimension",
        ));
    }
    if problem
        .inputs
        .as_slice()
        .iter()
        .chain(problem.targets.as_slice())
        .any(|value| value.abs() > policy.max_absolute_value)
    {
        return Some(EvaluationReason::ResourceLimit("solver_max_absolute_value"));
    }
    // Bound the complete Frobenius energy, not just one Gram entry. Since the
    // largest squared singular value is bounded by ||X||_F^2, this prevents
    // the audited spectrum and condition diagnostics from overflowing even
    // when every case is aligned with every other case.
    let energy_terms = problem
        .case_count()
        .saturating_mul(problem.input_dimension())
        .max(1) as f64;
    let safe_gram_magnitude = (f64::MAX / energy_terms).sqrt();
    if problem
        .inputs
        .as_slice()
        .iter()
        .any(|value| value.abs() > safe_gram_magnitude)
    {
        return Some(EvaluationReason::ResourceLimit(
            "solver_safe_gram_magnitude",
        ));
    }
    let working = problem
        .case_count()
        .checked_mul(problem.case_count())
        .and_then(|value| {
            value.checked_add(
                problem
                    .case_count()
                    .checked_mul(problem.input_dimension())?,
            )
        })
        .and_then(|value| {
            value.checked_add(
                problem
                    .case_count()
                    .checked_mul(problem.output_dimension())?,
            )
        })
        .and_then(|value| {
            value.checked_add(
                problem
                    .output_dimension()
                    .checked_mul(problem.input_dimension())?,
            )
        });
    match working {
        Some(value) if value <= limits.max_working_elements => None,
        _ => Some(EvaluationReason::ResourceLimit(
            "solver_max_working_elements",
        )),
    }
}

/// Conservative work bound for the complete built-in rank-revealing path:
/// Jacobi sweeps (including convergence checks), decomposition verification,
/// Gram diagnostics, construction of every permitted rank candidate, and
/// independent residual recomputation for each candidate. Units deliberately
/// over-count loop bodies; they are an execution budget, not reported FLOPs.
fn direct_solver_work_upper_bound(
    problem: &LeastSquaresProblem,
    policy: &PortfolioPolicy,
) -> Option<u128> {
    let cases = u128::try_from(problem.case_count()).ok()?;
    let inputs = u128::try_from(problem.input_dimension()).ok()?;
    let outputs = u128::try_from(problem.output_dimension()).ok()?;
    let components = u128::try_from(problem.input_dimension().min(problem.case_count())).ok()?;
    let tall = u128::try_from(problem.input_dimension().max(problem.case_count())).ok()?;
    let sweeps = u128::try_from(policy.max_svd_sweeps).ok()?;
    let pairs = components
        .checked_mul(components.saturating_sub(1))?
        .checked_div(2)?;
    // Three dot products plus two rotations in the Jacobi visit, followed by
    // another three dot products in maximum_column_correlation.
    let per_pair_per_sweep = tall
        .checked_mul(8)?
        .checked_add(components.checked_mul(2)?)?
        .checked_add(64)?;
    let jacobi = sweeps.checked_mul(pairs)?.checked_mul(per_pair_per_sweep)?;

    let verification_pairs = components
        .checked_mul(components.checked_add(1)?)?
        .checked_div(2)?;
    let decomposition_verification = verification_pairs
        .checked_mul(inputs.checked_add(cases)?)?
        .checked_mul(4)?
        .checked_add(
            cases
                .checked_mul(inputs)?
                .checked_mul(components)?
                .checked_mul(4)?,
        )?;

    let gram_diagnostics = cases
        .checked_mul(cases.checked_add(1)?)?
        .checked_div(2)?
        .checked_mul(inputs)?
        .checked_mul(4)?;

    let rank_cap = u128::try_from(
        policy
            .maximum_rank
            .min(problem.case_count().min(problem.input_dimension())),
    )
    .ok()?;
    let rank_sum = rank_cap
        .checked_mul(rank_cap.checked_add(1)?)?
        .checked_div(2)?;
    let factor_construction = rank_sum
        .checked_mul(outputs.checked_mul(cases)?.checked_add(inputs)?)?
        .checked_mul(4)?;
    let candidate_materialization = rank_sum
        .checked_mul(outputs)?
        .checked_mul(inputs)?
        .checked_mul(4)?;
    let candidate_evaluation =
        rank_cap.checked_mul(single_candidate_evaluation_work_upper_bound(problem)?)?;
    let cholesky_path = cases
        .checked_mul(cases)?
        .checked_mul(inputs.checked_add(outputs)?)?
        .checked_mul(8)?
        .checked_add(
            cases
                .checked_mul(cases)?
                .checked_mul(cases)?
                .checked_mul(8)?,
        )?;

    jacobi
        .checked_add(decomposition_verification)?
        .checked_add(gram_diagnostics)?
        .checked_add(factor_construction)?
        .checked_add(candidate_materialization)?
        .checked_add(candidate_evaluation)
        .and_then(|work| work.checked_add(cholesky_path))
}

fn single_candidate_evaluation_work_upper_bound(problem: &LeastSquaresProblem) -> Option<u128> {
    let cases = u128::try_from(problem.case_count()).ok()?;
    let inputs = u128::try_from(problem.input_dimension()).ok()?;
    let outputs = u128::try_from(problem.output_dimension()).ok()?;
    outputs.checked_mul(inputs)?.checked_mul(8)?.checked_add(
        cases
            .checked_mul(outputs)?
            .checked_mul(inputs.checked_add(1)?)?
            .checked_mul(8)?,
    )
}

fn row_gram(inputs: &Matrix) -> BrainResult<Matrix> {
    let mut rows = vec![vec![0.0; inputs.row_count()]; inputs.row_count()];
    let mut row = 0;
    while row < inputs.row_count() {
        let mut column = row;
        while column < inputs.row_count() {
            let value = scaled_compensated_dot(inputs.row(row), inputs.row(column))?;
            rows[row][column] = value;
            rows[column][row] = value;
            column += 1;
        }
        row += 1;
    }
    Matrix::from_rows(&rows)
}

fn compensated_sum(values: impl IntoIterator<Item = f64>) -> BrainResult<f64> {
    let mut sum = 0.0_f64;
    let mut correction = 0.0_f64;
    for value in values {
        if !value.is_finite() {
            return Err(BrainError::Numerical(
                "solver_compensated_sum_input_nonfinite".into(),
            ));
        }
        let updated = sum + value;
        if sum.abs() >= value.abs() {
            correction += (sum - updated) + value;
        } else {
            correction += (value - updated) + sum;
        }
        sum = updated;
    }
    let result = sum + correction;
    if !result.is_finite() {
        return Err(BrainError::Numerical(
            "solver_compensated_sum_nonfinite".into(),
        ));
    }
    Ok(result)
}

/// Scale-normalized Neumaier dot product used at the independent numerical
/// boundary. The common linalg dot remains the general fast primitive; this
/// path additionally protects verification and Jacobi rotations from avoidable
/// overflow and cancellation.
fn scaled_compensated_dot(left: &[f64], right: &[f64]) -> BrainResult<f64> {
    if left.len() != right.len() || left.iter().chain(right).any(|value| !value.is_finite()) {
        return Err(BrainError::Invalid(
            "solver_compensated_dot_shape_or_value".into(),
        ));
    }
    let left_scale = left.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let right_scale = right.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if left_scale == 0.0 || right_scale == 0.0 {
        return Ok(0.0);
    }
    let normalized = compensated_sum(
        left.iter()
            .zip(right)
            .map(|(left, right)| (left / left_scale) * (right / right_scale)),
    )?;
    let value = (normalized * left_scale) * right_scale;
    if !value.is_finite() {
        return Err(BrainError::Numerical(
            "solver_compensated_dot_nonfinite".into(),
        ));
    }
    Ok(value)
}

#[derive(Debug, Clone)]
struct DirectSvdComponent {
    singular_value: f64,
    input_vector: Vec<f64>,
    case_vector: Vec<f64>,
}

#[derive(Debug, Clone)]
struct DirectJacobiSvd {
    input_dimension: usize,
    components: Vec<DirectSvdComponent>,
    sweeps: usize,
    maximum_scaled_column_correlation: f64,
    reconstruction_relative_error: f64,
    vector_orthogonality_error: f64,
}

/// Thin one-sided Jacobi SVD of X^T. It orthogonalizes the case columns
/// directly instead of diagonalizing X X^T, so rank revelation does not first
/// square the condition number. This is a bounded in-tree implementation, not
/// a claim of LAPACK GELSD compatibility.
fn direct_one_sided_jacobi_svd(
    problem: &LeastSquaresProblem,
    policy: &PortfolioPolicy,
) -> BrainResult<Option<DirectJacobiSvd>> {
    let input_dimension = problem.input_dimension();
    let cases = problem.case_count();
    // One-sided Jacobi is applied to the tall orientation. For the common
    // adaptation case d >= n this is X^T; for n > d it is X itself. The
    // resulting component is normalized back to the same (input, case) pair.
    let transposed = input_dimension >= cases;
    let component_count = input_dimension.min(cases);
    let scale = problem
        .inputs
        .as_slice()
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Ok(Some(DirectJacobiSvd {
            input_dimension,
            components: (0..component_count)
                .map(|component| DirectSvdComponent {
                    singular_value: 0.0,
                    input_vector: if transposed {
                        vec![0.0; input_dimension]
                    } else {
                        (0..input_dimension)
                            .map(|index| if index == component { 1.0 } else { 0.0 })
                            .collect()
                    },
                    case_vector: if transposed {
                        (0..cases)
                            .map(|index| if index == component { 1.0 } else { 0.0 })
                            .collect()
                    } else {
                        vec![0.0; cases]
                    },
                })
                .collect(),
            sweeps: 0,
            maximum_scaled_column_correlation: 0.0,
            reconstruction_relative_error: 0.0,
            vector_orthogonality_error: 0.0,
        }));
    }

    let mut columns = (0..component_count)
        .map(|component| {
            if transposed {
                (0..input_dimension)
                    .map(|input| problem.inputs.get(component, input) / scale)
                    .collect::<Vec<_>>()
            } else {
                (0..cases)
                    .map(|case| problem.inputs.get(case, component) / scale)
                    .collect::<Vec<_>>()
            }
        })
        .collect::<Vec<_>>();
    let mut rotations = (0..component_count)
        .map(|column| {
            (0..component_count)
                .map(|row| if row == column { 1.0 } else { 0.0 })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut converged = component_count <= 1;
    let mut sweeps = 0;
    let mut maximum_correlation = 0.0;
    for sweep in 0..policy.max_svd_sweeps {
        for first in 0..component_count {
            for second in (first + 1)..component_count {
                let alpha = scaled_compensated_dot(&columns[first], &columns[first])?;
                let beta = scaled_compensated_dot(&columns[second], &columns[second])?;
                if alpha == 0.0 || beta == 0.0 {
                    continue;
                }
                let gamma = scaled_compensated_dot(&columns[first], &columns[second])?;
                let correlation = gamma.abs() / (alpha.sqrt() * beta.sqrt());
                if !correlation.is_finite() {
                    return Err(BrainError::Numerical(
                        "solver_direct_svd_correlation_nonfinite".into(),
                    ));
                }
                if correlation <= policy.svd_orthogonality_tolerance {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let tangent = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
                let sine = cosine * tangent;
                if !cosine.is_finite() || !sine.is_finite() {
                    return Err(BrainError::Numerical(
                        "solver_direct_svd_rotation_nonfinite".into(),
                    ));
                }
                rotate_columns(&mut columns, first, second, cosine, sine);
                rotate_columns(&mut rotations, first, second, cosine, sine);
            }
        }
        sweeps = sweep + 1;
        maximum_correlation = maximum_column_correlation(&columns)?;
        if maximum_correlation <= policy.svd_orthogonality_tolerance {
            converged = true;
            break;
        }
    }
    if !converged {
        return Ok(None);
    }

    let mut components = Vec::with_capacity(component_count);
    for index in 0..component_count {
        let scaled_sigma = norm(&columns[index])?;
        let singular_value = scaled_sigma * scale;
        if !singular_value.is_finite() {
            return Err(BrainError::Numerical(
                "solver_direct_svd_singular_value_nonfinite".into(),
            ));
        }
        let normalized_column = if scaled_sigma > 0.0 {
            columns[index]
                .iter()
                .map(|value| value / scaled_sigma)
                .collect::<Vec<_>>()
        } else {
            vec![0.0; columns[index].len()]
        };
        let (mut input_vector, mut case_vector) = if transposed {
            (normalized_column, rotations[index].clone())
        } else {
            (rotations[index].clone(), normalized_column)
        };
        if let Some(pivot) = input_vector
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
            .map(|(position, _)| position)
        {
            if input_vector[pivot].is_sign_negative() {
                for value in &mut input_vector {
                    *value = -*value;
                }
                for value in &mut case_vector {
                    *value = -*value;
                }
            }
        }
        components.push((
            index,
            DirectSvdComponent {
                singular_value,
                input_vector,
                case_vector,
            },
        ));
    }
    components.sort_by(|left, right| {
        right
            .1
            .singular_value
            .total_cmp(&left.1.singular_value)
            .then_with(|| left.0.cmp(&right.0))
    });
    let components = components
        .into_iter()
        .map(|(_, component)| component)
        .collect::<Vec<_>>();
    let (reconstruction_relative_error, vector_orthogonality_error) =
        verify_direct_svd(problem, &components)?;
    let verification_tolerance = (policy.svd_orthogonality_tolerance * 100.0)
        .max(f64::EPSILON * input_dimension.max(cases).max(1) as f64 * 100.0);
    if reconstruction_relative_error > verification_tolerance
        || vector_orthogonality_error > verification_tolerance
    {
        return Ok(None);
    }
    Ok(Some(DirectJacobiSvd {
        input_dimension,
        components,
        sweeps,
        maximum_scaled_column_correlation: maximum_correlation,
        reconstruction_relative_error,
        vector_orthogonality_error,
    }))
}

fn verify_direct_svd(
    problem: &LeastSquaresProblem,
    components: &[DirectSvdComponent],
) -> BrainResult<(f64, f64)> {
    let largest = components
        .iter()
        .map(|component| component.singular_value)
        .fold(0.0_f64, f64::max);
    let active_threshold =
        largest * f64::EPSILON * problem.input_dimension().max(problem.case_count()).max(1) as f64;
    let active = components
        .iter()
        .filter(|component| component.singular_value > active_threshold)
        .collect::<Vec<_>>();
    let mut orthogonality_error = 0.0_f64;
    for first in 0..active.len() {
        for second in first..active.len() {
            let expected = if first == second { 1.0 } else { 0.0 };
            let input_dot =
                scaled_compensated_dot(&active[first].input_vector, &active[second].input_vector)?;
            let case_dot =
                scaled_compensated_dot(&active[first].case_vector, &active[second].case_vector)?;
            orthogonality_error = orthogonality_error
                .max((input_dot - expected).abs())
                .max((case_dot - expected).abs());
        }
    }

    let mut residual = Vec::with_capacity(problem.inputs.as_slice().len());
    for case in 0..problem.case_count() {
        for input in 0..problem.input_dimension() {
            let reconstructed = compensated_sum(components.iter().map(|component| {
                component.singular_value
                    * component.case_vector[case]
                    * component.input_vector[input]
            }))?;
            residual.push(reconstructed - problem.inputs.get(case, input));
        }
    }
    let input_norm = norm(problem.inputs.as_slice())?;
    let reconstruction_relative_error = if input_norm == 0.0 {
        if residual.iter().all(|value| *value == 0.0) {
            0.0
        } else {
            return Err(BrainError::Numerical(
                "solver_direct_svd_zero_reconstruction_invalid".into(),
            ));
        }
    } else {
        norm(&residual)? / input_norm
    };
    if !reconstruction_relative_error.is_finite() || !orthogonality_error.is_finite() {
        return Err(BrainError::Numerical(
            "solver_direct_svd_verification_nonfinite".into(),
        ));
    }
    Ok((reconstruction_relative_error, orthogonality_error))
}

fn rotate_columns(columns: &mut [Vec<f64>], first: usize, second: usize, cosine: f64, sine: f64) {
    let (left, right) = columns.split_at_mut(second);
    let first_column = &mut left[first];
    let second_column = &mut right[0];
    for (first_value, second_value) in first_column.iter_mut().zip(second_column) {
        let old_first = *first_value;
        let old_second = *second_value;
        *first_value = cosine * old_first - sine * old_second;
        *second_value = sine * old_first + cosine * old_second;
    }
}

fn maximum_column_correlation(columns: &[Vec<f64>]) -> BrainResult<f64> {
    let mut maximum = 0.0_f64;
    for first in 0..columns.len() {
        for second in (first + 1)..columns.len() {
            let alpha = scaled_compensated_dot(&columns[first], &columns[first])?;
            let beta = scaled_compensated_dot(&columns[second], &columns[second])?;
            if alpha == 0.0 || beta == 0.0 {
                continue;
            }
            let gamma = scaled_compensated_dot(&columns[first], &columns[second])?;
            maximum = maximum.max(gamma.abs() / (alpha.sqrt() * beta.sqrt()));
        }
    }
    if !maximum.is_finite() {
        return Err(BrainError::Numerical(
            "solver_direct_svd_correlation_nonfinite".into(),
        ));
    }
    Ok(maximum)
}

fn diagnose_problem(
    gram: &Matrix,
    direct_svd: &DirectJacobiSvd,
    policy: &PortfolioPolicy,
) -> BrainResult<NumericalDiagnostics> {
    validate_symmetric_psd(gram, "solver_portfolio_gram")?;
    let singular_values = direct_svd
        .components
        .iter()
        .map(|component| component.singular_value)
        .collect::<Vec<_>>();
    let spectrum = singular_values
        .iter()
        .map(|value| value * value)
        .collect::<Vec<_>>();
    if spectrum.iter().any(|value| !value.is_finite()) {
        return Err(BrainError::Numerical(
            "solver_gram_spectrum_nonfinite".into(),
        ));
    }
    let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let automatic_tolerance =
        f64::EPSILON * gram.row_count().max(direct_svd.input_dimension).max(1) as f64;
    let singular_value_cutoff = largest * policy.relative_rank_tolerance.max(automatic_tolerance);
    let numerical_rank = singular_values
        .iter()
        .filter(|value| **value > singular_value_cutoff)
        .count();
    // Entropy and explained-energy ranks are scale-free. Feeding the raw
    // squared singular values into a routine with an absolute zero threshold
    // would incorrectly label a perfectly conditioned, very small matrix as
    // rank zero. Normalize before computing these diagnostics while retaining
    // the physical Gram spectrum above for auditability.
    let normalized_spectrum = if largest == 0.0 {
        vec![0.0; singular_values.len()]
    } else {
        singular_values
            .iter()
            .map(|value| {
                let ratio = value / largest;
                ratio * ratio
            })
            .collect::<Vec<_>>()
    };
    let effective_rank = effective_rank_from_spectrum(&normalized_spectrum)?;
    let energy_rank = if numerical_rank == 0 {
        0
    } else {
        choose_energy_rank(
            &normalized_spectrum,
            policy.target_explained_energy,
            numerical_rank,
            policy.minimum_rank.min(numerical_rank),
        )?
    };
    let (condition_kind, finite_singular_condition_number, finite_gram_condition_number) =
        if numerical_rank == 0 {
            (GramCondition::Degenerate, None, None)
        } else if numerical_rank < gram.row_count() {
            (GramCondition::RankDeficient, None, None)
        } else {
            let smallest = singular_values[numerical_rank - 1];
            let singular_condition = largest / smallest;
            let direct_gram_condition = singular_condition * singular_condition;
            if singular_condition.is_finite() && direct_gram_condition.is_finite() {
                let independently_estimated =
                    symmetric_psd_condition(gram, "solver_portfolio_gram")?;
                // The generic Gram estimator deliberately treats very small
                // absolute spectra as unresolved. The verified direct SVD is
                // scale-normalized, so an infinite auxiliary estimate must
                // not contaminate a finite direct condition number.
                let conservative_gram_condition = if independently_estimated.is_finite() {
                    direct_gram_condition.max(independently_estimated)
                } else {
                    direct_gram_condition
                };
                (
                    GramCondition::Finite,
                    Some(singular_condition),
                    Some(conservative_gram_condition),
                )
            } else {
                (GramCondition::Unrepresentable, None, None)
            }
        };
    Ok(NumericalDiagnostics {
        singular_values,
        gram_spectrum: spectrum,
        largest_singular_value: largest,
        numerical_rank,
        effective_rank,
        energy_rank,
        singular_value_cutoff,
        condition_kind,
        finite_singular_condition_number,
        finite_gram_condition_number,
        direct_svd_sweeps: direct_svd.sweeps,
        maximum_scaled_column_correlation: direct_svd.maximum_scaled_column_correlation,
        direct_svd_reconstruction_relative_error: direct_svd.reconstruction_relative_error,
        direct_svd_vector_orthogonality_error: direct_svd.vector_orthogonality_error,
    })
}

fn cholesky_eligibility(
    problem: &LeastSquaresProblem,
    diagnostics: &NumericalDiagnostics,
    policy: &PortfolioPolicy,
) -> CholeskyGate {
    if problem.case_count() < 2 {
        return CholeskyGate::CaseCountInsufficient;
    }
    if problem.case_count() > MAX_LOW_RANK as usize {
        return CholeskyGate::CaseLimitExceeded;
    }
    match diagnostics.condition_kind {
        GramCondition::Degenerate => CholeskyGate::Degenerate,
        GramCondition::RankDeficient => CholeskyGate::RankDeficient,
        GramCondition::Unrepresentable => CholeskyGate::ConditionExceedsPolicy,
        GramCondition::Finite => match diagnostics.finite_gram_condition_number {
            Some(condition) if condition <= policy.cholesky_max_condition => {
                CholeskyGate::Authorized
            }
            _ => CholeskyGate::ConditionExceedsPolicy,
        },
    }
}

fn float32_problem_is_safe(problem: &LeastSquaresProblem) -> bool {
    problem
        .inputs
        .as_slice()
        .iter()
        .chain(problem.targets.as_slice())
        .all(|value| {
            let converted = *value as f32;
            converted.is_finite() && (*value == 0.0 || converted != 0.0)
        })
}

fn run_cholesky(problem: &LeastSquaresProblem, policy: &PortfolioPolicy) -> CandidateEvaluation {
    let backend = BackendDescriptor::builtin(
        NumericalBackendRole::BuiltInCholeskyRidge,
        BUILTIN_CHOLESKY_NAME,
    );
    let inputs = (0..problem.case_count())
        .map(|row| {
            problem
                .inputs
                .row(row)
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let targets = (0..problem.case_count())
        .map(|row| {
            problem
                .targets
                .row(row)
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    match solve_regularized_multi_case_low_rank(&inputs, &targets, policy.cholesky_damping) {
        Ok(solution) => {
            let rank = solution.rank as usize;
            let candidate = compact_representation(
                problem.output_dimension(),
                problem.input_dimension(),
                rank,
                solution.left.into_iter().map(f64::from).collect(),
                solution.right.into_iter().map(f64::from).collect(),
            );
            evaluate_candidate(backend, Some(rank), candidate, problem, policy)
        }
        Err(error) => CandidateEvaluation {
            backend,
            status: SolverStatus::Rejected,
            reason: EvaluationReason::BackendFailure(error.to_string()),
            constructed_rank: None,
            candidate: None,
            metrics: None,
        },
    }
}

fn run_direct_svd(
    problem: &LeastSquaresProblem,
    direct_svd: &DirectJacobiSvd,
    diagnostics: &NumericalDiagnostics,
    policy: &PortfolioPolicy,
) -> BrainResult<CandidateEvaluation> {
    let backend = BackendDescriptor::builtin(
        NumericalBackendRole::BuiltInDirectJacobiSvd,
        BUILTIN_DIRECT_SVD_NAME,
    );
    if diagnostics.numerical_rank == 0 {
        let candidate = CandidateRepresentation::Dense {
            rows: problem.output_dimension(),
            columns: problem.input_dimension(),
            weights: vec![0.0; problem.output_dimension() * problem.input_dimension()],
        };
        return Ok(evaluate_candidate(
            backend,
            Some(0),
            candidate,
            problem,
            policy,
        ));
    }
    let rank_cap = diagnostics.numerical_rank.min(policy.maximum_rank);
    if rank_cap < diagnostics.energy_rank || rank_cap < policy.minimum_rank {
        return Ok(CandidateEvaluation {
            backend,
            status: SolverStatus::BoundedUnknown,
            reason: EvaluationReason::RankBudgetInsufficient,
            constructed_rank: Some(rank_cap),
            candidate: None,
            metrics: None,
        });
    }
    let mut last = None;
    for rank in diagnostics.energy_rank.max(policy.minimum_rank)..=rank_cap {
        let (left, right) =
            direct_svd_factors(problem, direct_svd, rank, diagnostics.singular_value_cutoff)?;
        let candidate = compact_representation(
            problem.output_dimension(),
            problem.input_dimension(),
            rank,
            left,
            right,
        );
        let evaluated = evaluate_candidate(backend.clone(), Some(rank), candidate, problem, policy);
        if evaluated.status == SolverStatus::Accepted {
            return Ok(evaluated);
        }
        last = Some(evaluated);
    }
    let mut result =
        last.ok_or_else(|| BrainError::Numerical("solver_direct_svd_candidate_missing".into()))?;
    if rank_cap < diagnostics.numerical_rank {
        result.status = SolverStatus::BoundedUnknown;
        result.reason = EvaluationReason::RankBudgetInsufficient;
    }
    Ok(result)
}

fn direct_svd_factors(
    problem: &LeastSquaresProblem,
    direct_svd: &DirectJacobiSvd,
    rank: usize,
    singular_value_cutoff: f64,
) -> BrainResult<(Vec<f64>, Vec<f64>)> {
    if rank == 0
        || rank > direct_svd.components.len()
        || direct_svd.components.iter().take(rank).any(|component| {
            !component.singular_value.is_finite()
                || component.singular_value <= singular_value_cutoff
                || component.case_vector.len() != problem.case_count()
                || component.input_vector.len() != problem.input_dimension()
                || component
                    .case_vector
                    .iter()
                    .chain(&component.input_vector)
                    .any(|entry| !entry.is_finite())
        })
    {
        return Err(BrainError::Numerical(
            "solver_direct_svd_rank_invalid".into(),
        ));
    }
    let mut left = vec![0.0; problem.output_dimension() * rank];
    let mut right = vec![0.0; rank * problem.input_dimension()];
    for (component_index, component) in direct_svd.components.iter().take(rank).enumerate() {
        let inverse_sigma = 1.0 / component.singular_value;
        if !inverse_sigma.is_finite() {
            return Err(BrainError::Numerical(
                "solver_direct_svd_inverse_nonfinite".into(),
            ));
        }
        for output in 0..problem.output_dimension() {
            left[output * rank + component_index] = compensated_sum(
                (0..problem.case_count())
                    .map(|case| problem.targets.get(case, output) * component.case_vector[case]),
            )? * inverse_sigma;
        }
        for input in 0..problem.input_dimension() {
            right[component_index * problem.input_dimension() + input] =
                component.input_vector[input];
        }
    }
    if left.iter().chain(&right).any(|value| !value.is_finite()) {
        return Err(BrainError::Numerical(
            "solver_direct_svd_factor_nonfinite".into(),
        ));
    }
    Ok((left, right))
}

fn compact_representation(
    rows: usize,
    columns: usize,
    rank: usize,
    left: Vec<f64>,
    right: Vec<f64>,
) -> CandidateRepresentation {
    let low_rank = CandidateRepresentation::LowRank {
        rows,
        columns,
        rank,
        left,
        right,
    };
    let factor_parameters = low_rank.stored_parameter_count().unwrap_or(usize::MAX);
    let dense_parameters = rows.saturating_mul(columns);
    if factor_parameters < dense_parameters {
        low_rank
    } else {
        match low_rank.materialize_dense() {
            Ok(weights) => CandidateRepresentation::Dense {
                rows,
                columns,
                weights,
            },
            Err(_) => low_rank,
        }
    }
}

fn run_external(
    proposal: &ExternalCandidateProposal,
    problem: &LeastSquaresProblem,
    policy: &PortfolioPolicy,
) -> CandidateEvaluation {
    let descriptor = proposal.descriptor.clone();
    match &proposal.payload {
        ExternalProposalPayload::Candidate(candidate) => {
            if single_candidate_evaluation_work_upper_bound(problem)
                .is_none_or(|work| work > u128::from(policy.limits.max_numerical_work_units))
            {
                return CandidateEvaluation {
                    backend: descriptor,
                    status: SolverStatus::BoundedUnknown,
                    reason: EvaluationReason::ResourceLimit(
                        "solver_max_candidate_evaluation_work_units",
                    ),
                    constructed_rank: candidate.rank(),
                    candidate: None,
                    metrics: None,
                };
            }
            let rank = candidate.rank();
            evaluate_candidate(descriptor, rank, candidate.clone(), problem, policy)
        }
        ExternalProposalPayload::NotApplicable(reason) => external_message_evaluation(
            descriptor,
            reason.clone(),
            false,
            EvaluationReason::BackendNotApplicable,
        ),
        ExternalProposalPayload::BoundedUnknown(reason) => external_message_evaluation(
            descriptor,
            reason.clone(),
            true,
            EvaluationReason::BackendBoundedUnknown,
        ),
    }
}

fn external_message_evaluation(
    backend: BackendDescriptor,
    message: String,
    bounded_unknown: bool,
    reason: fn(String) -> EvaluationReason,
) -> CandidateEvaluation {
    let valid = !message.trim().is_empty()
        && message.len() <= MAX_BACKEND_REASON_BYTES
        && !message.chars().any(char::is_control);
    CandidateEvaluation {
        backend,
        status: if valid && bounded_unknown {
            SolverStatus::BoundedUnknown
        } else {
            SolverStatus::Rejected
        },
        reason: if valid {
            reason(message)
        } else {
            EvaluationReason::InvalidCandidate("solver_backend_message_invalid".into())
        },
        constructed_rank: None,
        candidate: None,
        metrics: None,
    }
}

fn evaluate_candidate(
    backend: BackendDescriptor,
    constructed_rank: Option<usize>,
    candidate: CandidateRepresentation,
    problem: &LeastSquaresProblem,
    policy: &PortfolioPolicy,
) -> CandidateEvaluation {
    let parameter_count = match candidate.stored_parameter_count() {
        Some(count) => count,
        None => {
            return CandidateEvaluation {
                backend,
                status: SolverStatus::Rejected,
                reason: EvaluationReason::InvalidCandidate(
                    "solver_candidate_parameter_overflow".into(),
                ),
                constructed_rank,
                candidate: None,
                metrics: None,
            };
        }
    };
    if parameter_count > policy.limits.max_candidate_parameters {
        return CandidateEvaluation {
            backend,
            status: SolverStatus::BoundedUnknown,
            reason: EvaluationReason::ResourceLimit("solver_max_candidate_parameters"),
            constructed_rank,
            candidate: None,
            metrics: None,
        };
    }
    match measure_candidate(&candidate, problem) {
        Ok(metrics) => {
            let relative_allowance = policy.relative_residual_tolerance * metrics.target_norm;
            let acceptance_threshold = policy.absolute_residual_tolerance + relative_allowance;
            if !relative_allowance.is_finite() || !acceptance_threshold.is_finite() {
                return CandidateEvaluation {
                    backend,
                    status: SolverStatus::BoundedUnknown,
                    reason: EvaluationReason::ResidualToleranceUnrepresentable,
                    constructed_rank,
                    candidate: Some(candidate),
                    metrics: Some(metrics),
                };
            }
            let accepted = metrics.absolute_residual <= acceptance_threshold;
            CandidateEvaluation {
                backend,
                status: if accepted {
                    SolverStatus::Accepted
                } else {
                    SolverStatus::Rejected
                },
                reason: if accepted {
                    EvaluationReason::ResidualWithinTolerance
                } else {
                    EvaluationReason::ResidualExceedsTolerance
                },
                constructed_rank,
                candidate: Some(candidate),
                metrics: Some(metrics),
            }
        }
        Err(error) => CandidateEvaluation {
            backend,
            status: SolverStatus::Rejected,
            reason: EvaluationReason::InvalidCandidate(error.to_string()),
            constructed_rank,
            candidate: None,
            metrics: None,
        },
    }
}

fn measure_candidate(
    candidate: &CandidateRepresentation,
    problem: &LeastSquaresProblem,
) -> BrainResult<CandidateMetrics> {
    if candidate.rows() != problem.output_dimension()
        || candidate.columns() != problem.input_dimension()
    {
        return Err(BrainError::Integrity(
            "solver_candidate_problem_shape_mismatch".into(),
        ));
    }
    let dense = candidate.materialize_dense()?;
    let mut residuals = Vec::with_capacity(
        problem
            .case_count()
            .checked_mul(problem.output_dimension())
            .ok_or_else(|| BrainError::Invalid("solver_residual_shape_overflow".into()))?,
    );
    for case in 0..problem.case_count() {
        for output in 0..problem.output_dimension() {
            let start = output * problem.input_dimension();
            let predicted = scaled_compensated_dot(
                &dense[start..start + problem.input_dimension()],
                problem.inputs.row(case),
            )?;
            residuals.push(predicted - problem.targets.get(case, output));
        }
    }
    let absolute_residual = norm(&residuals)?;
    let residual_count = residuals.len();
    if residual_count == 0 {
        return Err(BrainError::Integrity(
            "solver_candidate_residual_empty".into(),
        ));
    }
    let root_mean_square_residual = absolute_residual / (residual_count as f64).sqrt();
    let maximum_absolute_residual = residuals
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let target_norm = norm(problem.targets.as_slice())?;
    let frobenius_norm = norm(&dense)?;
    let relative_residual = if target_norm > 0.0 {
        Some(absolute_residual / target_norm)
    } else if absolute_residual == 0.0 {
        Some(0.0)
    } else {
        None
    };
    if !absolute_residual.is_finite()
        || !root_mean_square_residual.is_finite()
        || !maximum_absolute_residual.is_finite()
        || !target_norm.is_finite()
        || !frobenius_norm.is_finite()
        || relative_residual.is_some_and(|value| !value.is_finite())
    {
        return Err(BrainError::Numerical(
            "solver_candidate_metrics_nonfinite".into(),
        ));
    }
    Ok(CandidateMetrics {
        absolute_residual,
        root_mean_square_residual,
        maximum_absolute_residual,
        relative_residual,
        target_norm,
        frobenius_norm,
        stored_parameter_count: candidate
            .stored_parameter_count()
            .ok_or_else(|| BrainError::Invalid("solver_candidate_parameter_overflow".into()))?,
    })
}

/// Recompute candidate metrics on another exact problem under the same hard
/// resource policy. This is the evaluation primitive used by holdout and
/// canary authorities; it never accepts backend-supplied metrics.
pub fn measure_candidate_with_policy(
    candidate: &CandidateRepresentation,
    problem: &LeastSquaresProblem,
    policy: &PortfolioPolicy,
) -> BrainResult<CandidateMetrics> {
    policy.validate()?;
    problem.inputs.validate("solver_measure_inputs")?;
    problem.targets.validate("solver_measure_targets")?;
    if let Some(reason) = problem_limit(problem, policy) {
        return Err(BrainError::Invalid(format!(
            "solver_measurement_resource_limit:{reason:?}"
        )));
    }
    let stored = candidate
        .stored_parameter_count()
        .ok_or_else(|| BrainError::Invalid("solver_candidate_parameter_overflow".into()))?;
    let dense = candidate
        .rows()
        .checked_mul(candidate.columns())
        .ok_or_else(|| BrainError::Invalid("solver_candidate_shape_overflow".into()))?;
    if stored > policy.limits.max_candidate_parameters || dense > policy.limits.max_working_elements
    {
        return Err(BrainError::Invalid(
            "solver_measurement_candidate_resource_limit".into(),
        ));
    }
    if single_candidate_evaluation_work_upper_bound(problem)
        .is_none_or(|work| work > u128::from(policy.limits.max_numerical_work_units))
    {
        return Err(BrainError::Invalid("solver_measurement_work_limit".into()));
    }
    measure_candidate(candidate, problem)
}

fn choose_accepted(evaluations: &[CandidateEvaluation]) -> BrainResult<Option<usize>> {
    // Every survivor already satisfies the same residual contract. Selection
    // therefore prefers the smaller stored representation, then the bounded
    // built-in cost ordering, and only then the residual inside tolerance.
    // Governance remains free to retain all survivors and apply wider
    // functional, causal, and interference evidence before promotion.
    let mut accepted = Vec::new();
    for (index, evaluation) in evaluations.iter().enumerate() {
        if evaluation.status != SolverStatus::Accepted {
            continue;
        }
        let metrics = evaluation
            .metrics
            .as_ref()
            .ok_or_else(|| BrainError::Integrity("solver_accepted_metrics_missing".into()))?;
        let candidate = evaluation
            .candidate
            .as_ref()
            .ok_or_else(|| BrainError::Integrity("solver_accepted_candidate_missing".into()))?;
        accepted.push((index, evaluation, metrics, candidate.exact_digest()?));
    }
    Ok(accepted
        .into_iter()
        .min_by(
            |(left_index, left, left_metrics, left_digest),
             (right_index, right, right_metrics, right_digest)| {
                left_metrics
                    .stored_parameter_count
                    .cmp(&right_metrics.stored_parameter_count)
                    .then_with(|| left.backend.role.cmp(&right.backend.role))
                    .then_with(|| {
                        left_metrics
                            .absolute_residual
                            .total_cmp(&right_metrics.absolute_residual)
                    })
                    .then_with(|| {
                        left_metrics
                            .frobenius_norm
                            .total_cmp(&right_metrics.frobenius_norm)
                    })
                    .then_with(|| left_digest.cmp(right_digest))
                    .then_with(|| left_index.cmp(right_index))
            },
        )
        .map(|(index, _, _, _)| index))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_policy() -> PortfolioPolicy {
        PortfolioPolicy::default()
            .with_rank_policy(1.0e-10, 0.999999, 1, 64)
            .unwrap()
            .with_residual_tolerances(1.0e-9, 1.0e-11)
            .unwrap()
    }

    #[test]
    fn rejects_wrong_shape_and_nonfinite_problem() {
        assert!(LeastSquaresProblem::new(vec![vec![1.0]], vec![vec![1.0], vec![2.0]]).is_err());
        assert!(LeastSquaresProblem::new(
            vec![vec![1.0], vec![1.0, 2.0]],
            vec![vec![1.0], vec![2.0],]
        )
        .is_err());
        assert!(LeastSquaresProblem::new(vec![vec![f64::NAN]], vec![vec![1.0]]).is_err());
    }

    #[test]
    fn exact_problem_digest_commits_to_shape_order_and_ieee_bits() {
        let positive_zero = LeastSquaresProblem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![0.0], vec![2.0]],
        )
        .unwrap();
        let identical = LeastSquaresProblem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![0.0], vec![2.0]],
        )
        .unwrap();
        let negative_zero = LeastSquaresProblem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![-0.0], vec![2.0]],
        )
        .unwrap();
        let reordered = LeastSquaresProblem::new(
            vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            vec![vec![2.0], vec![0.0]],
        )
        .unwrap();
        assert_eq!(positive_zero.digest().unwrap(), identical.digest().unwrap());
        assert_ne!(
            positive_zero.digest().unwrap(),
            negative_zero.digest().unwrap()
        );
        assert_ne!(positive_zero.digest().unwrap(), reordered.digest().unwrap());
    }

    #[test]
    fn exact_candidate_and_policy_digests_commit_to_semantics() {
        let positive_zero = CandidateRepresentation::Dense {
            rows: 1,
            columns: 1,
            weights: vec![0.0],
        };
        let negative_zero = CandidateRepresentation::Dense {
            rows: 1,
            columns: 1,
            weights: vec![-0.0],
        };
        let factored_zero = CandidateRepresentation::LowRank {
            rows: 1,
            columns: 1,
            rank: 1,
            left: vec![0.0],
            right: vec![1.0],
        };
        assert_ne!(
            positive_zero.exact_digest().unwrap(),
            negative_zero.exact_digest().unwrap()
        );
        assert_ne!(
            positive_zero.exact_digest().unwrap(),
            factored_zero.exact_digest().unwrap()
        );
        let first = PortfolioPolicy::default();
        let second = first.clone().with_cholesky_damping(1.0e-7).unwrap();
        assert_ne!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn rank_deficiency_forbids_cholesky_and_uses_minimum_norm_spectral_path() {
        let problem = LeastSquaresProblem::new(
            vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]],
            vec![vec![1.0], vec![2.0], vec![3.0]],
        )
        .unwrap();
        let report = solve_with_portfolio(&problem, &exact_policy(), &[]).unwrap();
        assert_eq!(report.status, SolverStatus::Accepted);
        assert_eq!(report.cholesky_gate, Some(CholeskyGate::RankDeficient));
        let diagnostics = report.diagnostics.as_ref().unwrap();
        assert_eq!(diagnostics.numerical_rank, 1);
        assert_eq!(diagnostics.condition_kind, GramCondition::RankDeficient);
        assert_eq!(report.selected().unwrap().constructed_rank, Some(1));
    }

    #[test]
    fn direct_svd_uses_tall_orientation_for_overdetermined_contracts() {
        let problem = LeastSquaresProblem::new(
            vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
                vec![2.0, -1.0],
            ],
            vec![vec![2.0], vec![3.0], vec![5.0], vec![1.0]],
        )
        .unwrap();
        let report = solve_with_portfolio(&problem, &exact_policy(), &[]).unwrap();
        assert_eq!(report.status, SolverStatus::Accepted);
        assert_eq!(report.cholesky_gate, Some(CholeskyGate::RankDeficient));
        assert_eq!(report.diagnostics.as_ref().unwrap().numerical_rank, 2);
        assert!(
            report
                .selected()
                .unwrap()
                .metrics
                .as_ref()
                .unwrap()
                .absolute_residual
                <= 1.0e-10
        );
    }

    #[test]
    fn ill_conditioned_full_rank_problem_falls_back_to_spectral_backend() {
        let problem = LeastSquaresProblem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0e-3]],
            vec![vec![1.0], vec![1.0e-3]],
        )
        .unwrap();
        let policy = exact_policy().with_cholesky_max_condition(1.0e4).unwrap();
        let report = solve_with_portfolio(&problem, &policy, &[]).unwrap();
        assert_eq!(report.status, SolverStatus::Accepted);
        assert_eq!(
            report.cholesky_gate,
            Some(CholeskyGate::ConditionExceedsPolicy)
        );
        assert_eq!(
            report.selected().unwrap().backend.role(),
            NumericalBackendRole::BuiltInDirectJacobiSvd
        );
    }

    #[test]
    fn spectral_rank_expands_until_the_contract_residual_is_met() {
        let problem = LeastSquaresProblem::new(
            vec![vec![2.0, 0.0], vec![0.0, 1.0]],
            vec![vec![2.0], vec![1.0]],
        )
        .unwrap();
        let policy = exact_policy()
            .with_rank_policy(1.0e-10, 0.75, 1, 64)
            .unwrap();
        let report = solve_with_portfolio(&problem, &policy, &[]).unwrap();
        assert_eq!(report.diagnostics.as_ref().unwrap().energy_rank, 1);
        let spectral = report
            .evaluations
            .iter()
            .find(|evaluation| {
                evaluation.backend.role() == NumericalBackendRole::BuiltInDirectJacobiSvd
            })
            .unwrap();
        assert_eq!(spectral.status, SolverStatus::Accepted);
        assert_eq!(spectral.constructed_rank, Some(2));
    }

    #[test]
    fn repeated_solve_is_deterministic() {
        let problem = LeastSquaresProblem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![2.0, -1.0], vec![3.0, 4.0]],
        )
        .unwrap();
        let first = solve_with_portfolio(&problem, &exact_policy(), &[]).unwrap();
        let second = solve_with_portfolio(&problem, &exact_policy(), &[]).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.exact_digest().unwrap(),
            second.exact_digest().unwrap()
        );

        let mut tampered = first.clone();
        tampered.status = SolverStatus::Rejected;
        assert!(tampered.exact_digest().is_err());
    }

    #[test]
    fn external_declared_name_is_not_authenticated_as_algorithmic_provenance() {
        let problem = LeastSquaresProblem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![1.0], vec![1.0]],
        )
        .unwrap();
        let candidate = CandidateRepresentation::Dense {
            rows: 1,
            columns: 2,
            weights: vec![1.0, 1.0],
        };
        let alpha = external_candidate("declared.alpha", candidate.clone());
        let beta = external_candidate("declared.beta", candidate);
        let first = solve_with_portfolio(&problem, &exact_policy(), &[alpha]).unwrap();
        let second = solve_with_portfolio(&problem, &exact_policy(), &[beta]).unwrap();
        assert_eq!(
            first.exact_digest().unwrap(),
            second.exact_digest().unwrap()
        );
    }

    #[test]
    fn zero_target_has_a_verified_zero_minimum_norm_solution() {
        let problem = LeastSquaresProblem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        )
        .unwrap();
        let report = solve_with_portfolio(&problem, &exact_policy(), &[]).unwrap();
        assert_eq!(report.status, SolverStatus::Accepted);
        let selected = report.selected().unwrap();
        assert_eq!(selected.metrics.as_ref().unwrap().absolute_residual, 0.0);
        assert_eq!(selected.metrics.as_ref().unwrap().target_norm, 0.0);
        assert_eq!(
            selected.metrics.as_ref().unwrap().relative_residual,
            Some(0.0)
        );
        assert!(selected
            .candidate
            .as_ref()
            .unwrap()
            .materialize_dense()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn degenerate_zero_design_still_returns_the_verified_zero_solution() {
        let problem = LeastSquaresProblem::new(
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            vec![vec![0.0], vec![0.0]],
        )
        .unwrap();
        let report = solve_with_portfolio(&problem, &exact_policy(), &[]).unwrap();
        assert_eq!(report.status(), SolverStatus::Accepted);
        assert_eq!(report.cholesky_gate(), Some(&CholeskyGate::Degenerate));
        assert_eq!(report.diagnostics().unwrap().numerical_rank(), 0);
        let selected = report.selected().unwrap();
        assert_eq!(selected.constructed_rank(), Some(0));
        assert_eq!(selected.metrics().unwrap().absolute_residual(), 0.0);
        assert!(selected
            .candidate()
            .unwrap()
            .materialize_dense()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn scale_free_rank_diagnostics_do_not_turn_tiny_data_into_rank_zero() {
        let scale = 1.0e-100;
        let problem = LeastSquaresProblem::new(
            vec![vec![scale, 0.0], vec![0.0, scale]],
            vec![vec![2.0 * scale], vec![-3.0 * scale]],
        )
        .unwrap();
        let report = solve_with_portfolio(&problem, &exact_policy(), &[]).unwrap();
        assert_eq!(report.status(), SolverStatus::Accepted);
        let diagnostics = report.diagnostics().unwrap();
        assert_eq!(diagnostics.numerical_rank(), 2);
        assert!((diagnostics.effective_rank() - 2.0).abs() <= 1.0e-12);
        assert_eq!(diagnostics.finite_singular_condition_number(), Some(1.0));
        assert_eq!(diagnostics.finite_gram_condition_number(), Some(1.0));
    }

    #[test]
    fn overflowing_tolerance_is_unknown_and_never_silently_accepts() {
        let problem = LeastSquaresProblem::new(vec![vec![1.0]], vec![vec![1.0]]).unwrap();
        let policy = PortfolioPolicy::default()
            .with_residual_tolerances(f64::MAX, f64::MAX)
            .unwrap();
        let report = solve_with_portfolio(&problem, &policy, &[]).unwrap();
        assert_eq!(report.status(), SolverStatus::BoundedUnknown);
        assert_eq!(
            report.reason(),
            &EvaluationReason::ResidualToleranceUnrepresentable
        );
        assert!(report.selected().is_none());
    }

    #[test]
    fn direct_svd_and_verifier_handle_large_finite_scaling() {
        let scale = 1.0e75;
        let problem = LeastSquaresProblem::new(
            vec![vec![scale, 0.0], vec![0.0, scale]],
            vec![vec![2.0 * scale], vec![-3.0 * scale]],
        )
        .unwrap();
        let report = solve_with_portfolio(&problem, &exact_policy(), &[]).unwrap();
        assert_eq!(report.status, SolverStatus::Accepted);
        assert_eq!(
            report.cholesky_gate,
            Some(CholeskyGate::Float32ConversionUnsafe)
        );
        assert_eq!(
            report.selected().unwrap().backend.role(),
            NumericalBackendRole::BuiltInDirectJacobiSvd
        );
        assert!(
            report
                .selected()
                .unwrap()
                .metrics
                .as_ref()
                .unwrap()
                .relative_residual
                .unwrap()
                <= 1.0e-14
        );
    }

    #[test]
    fn compensated_verification_preserves_cancellation_residual() {
        let value = scaled_compensated_dot(&[1.0e16, 1.0, -1.0e16], &[1.0, 1.0, 1.0]).unwrap();
        assert!((value - 1.0).abs() <= 1.0e-12);
    }

    #[test]
    fn exhausted_direct_svd_sweeps_are_bounded_unknown() {
        let problem = LeastSquaresProblem::new(
            vec![
                vec![1.0, 2.0, 3.0],
                vec![2.0, 1.0, 4.0],
                vec![3.0, 5.0, 1.0],
            ],
            vec![vec![1.0], vec![2.0], vec![3.0]],
        )
        .unwrap();
        let policy = exact_policy().with_svd_convergence(1.0e-16, 1).unwrap();
        let report = solve_with_portfolio(&problem, &policy, &[]).unwrap();
        assert_eq!(report.status, SolverStatus::BoundedUnknown);
        assert_eq!(report.reason, EvaluationReason::DirectSvdDidNotConverge);
        assert!(report.selected().is_none());
    }

    fn external_candidate(
        implementation: &'static str,
        candidate: CandidateRepresentation,
    ) -> ExternalCandidateProposal {
        ExternalCandidateProposal::candidate(implementation, candidate).unwrap()
    }

    #[test]
    fn external_proposal_cannot_self_report_or_bypass_residual_verification() {
        let problem = LeastSquaresProblem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![1.0], vec![1.0]],
        )
        .unwrap();
        let proposal = external_candidate(
            "test.external.full_rank",
            CandidateRepresentation::Dense {
                rows: 1,
                columns: 2,
                weights: vec![0.0, 0.0],
            },
        );
        let report = solve_with_portfolio(&problem, &exact_policy(), &[proposal]).unwrap();
        let external = report
            .evaluations
            .iter()
            .find(|evaluation| evaluation.backend.role() == NumericalBackendRole::ExternalProposal)
            .unwrap();
        assert_eq!(external.status, SolverStatus::Rejected);
        assert_eq!(external.reason, EvaluationReason::ResidualExceedsTolerance);
        assert!(external.metrics.as_ref().unwrap().absolute_residual > 1.0);
    }

    #[test]
    fn malformed_external_candidates_fail_closed() {
        let problem = LeastSquaresProblem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![1.0], vec![1.0]],
        )
        .unwrap();
        let wrong_shape = external_candidate(
            "test.external.wrong_shape",
            CandidateRepresentation::Dense {
                rows: 2,
                columns: 2,
                weights: vec![0.0; 4],
            },
        );
        let nonfinite = external_candidate(
            "test.external.nonfinite",
            CandidateRepresentation::Dense {
                rows: 1,
                columns: 2,
                weights: vec![f64::INFINITY, 0.0],
            },
        );
        for proposal in [wrong_shape, nonfinite] {
            let report = solve_with_portfolio(&problem, &exact_policy(), &[proposal]).unwrap();
            let external = report
                .evaluations
                .iter()
                .find(|evaluation| {
                    evaluation.backend.role() == NumericalBackendRole::ExternalProposal
                })
                .unwrap();
            assert_eq!(external.status, SolverStatus::Rejected);
            assert!(matches!(
                external.reason,
                EvaluationReason::InvalidCandidate(_)
            ));
        }
    }

    #[test]
    fn external_relabeling_cannot_manipulate_selection() {
        let scale = 1.0e75;
        let problem = LeastSquaresProblem::new(
            vec![vec![scale, 0.0], vec![0.0, scale]],
            vec![vec![scale], vec![scale]],
        )
        .unwrap();
        let candidate = CandidateRepresentation::Dense {
            rows: 1,
            columns: 2,
            weights: vec![1.0, 1.0],
        };
        let alpha = external_candidate("aaa.external", candidate.clone());
        let zulu = external_candidate("zzz.external", candidate);
        let policy = exact_policy().with_rank_policy(1.0e-10, 0.5, 1, 1).unwrap();
        let alpha_first =
            solve_with_portfolio(&problem, &policy, &[alpha.clone(), zulu.clone()]).unwrap();
        let zulu_first = solve_with_portfolio(&problem, &policy, &[zulu, alpha]).unwrap();
        assert_eq!(alpha_first.status, SolverStatus::Accepted);
        assert_eq!(zulu_first.status, SolverStatus::Accepted);
        assert_eq!(
            alpha_first.selected_evaluation,
            zulu_first.selected_evaluation
        );
        assert_eq!(
            alpha_first.selected().unwrap().candidate,
            zulu_first.selected().unwrap().candidate
        );
        assert_eq!(
            alpha_first.selected().unwrap().metrics,
            zulu_first.selected().unwrap().metrics
        );
        assert_eq!(
            alpha_first.selected().unwrap().backend.implementation(),
            "aaa.external"
        );
        assert_eq!(
            zulu_first.selected().unwrap().backend.implementation(),
            "zzz.external"
        );
    }

    #[test]
    fn external_candidate_order_cannot_change_selected_semantics() {
        let problem = LeastSquaresProblem::new(vec![vec![1.0, 1.0]], vec![vec![1.0]]).unwrap();
        let left = external_candidate(
            "left.external",
            CandidateRepresentation::Sparse {
                rows: 1,
                columns: 2,
                entries: vec![SparseEntry::new(0, 0, 1.0).unwrap()],
            },
        );
        let right = external_candidate(
            "right.external",
            CandidateRepresentation::Sparse {
                rows: 1,
                columns: 2,
                entries: vec![SparseEntry::new(0, 1, 1.0).unwrap()],
            },
        );
        let forward =
            solve_with_portfolio(&problem, &exact_policy(), &[left.clone(), right.clone()])
                .unwrap();
        let reverse = solve_with_portfolio(&problem, &exact_policy(), &[right, left]).unwrap();
        assert_eq!(
            forward
                .selected()
                .unwrap()
                .candidate()
                .unwrap()
                .exact_digest()
                .unwrap(),
            reverse
                .selected()
                .unwrap()
                .candidate()
                .unwrap()
                .exact_digest()
                .unwrap()
        );
        assert_eq!(
            forward.exact_digest().unwrap(),
            reverse.exact_digest().unwrap()
        );
    }

    #[test]
    fn gram_overflow_risk_is_a_bounded_unknown() {
        let problem = LeastSquaresProblem::new(vec![vec![1.0e200]], vec![vec![1.0]]).unwrap();
        let policy = exact_policy().with_max_absolute_value(1.0e300).unwrap();
        let report = solve_with_portfolio(&problem, &policy, &[]).unwrap();
        assert_eq!(report.status(), SolverStatus::BoundedUnknown);
        assert_eq!(
            report.reason(),
            &EvaluationReason::ResourceLimit("solver_safe_gram_magnitude")
        );
    }

    #[test]
    fn direct_svd_work_is_bounded_before_execution() {
        let problem = LeastSquaresProblem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![1.0], vec![1.0]],
        )
        .unwrap();
        let limits = SolverResourceLimits::default()
            .with_max_numerical_work_units(100)
            .unwrap();
        let policy = exact_policy().with_limits(limits).unwrap();
        let report = solve_with_portfolio(&problem, &policy, &[]).unwrap();
        assert_eq!(report.status(), SolverStatus::BoundedUnknown);
        assert_eq!(
            report.reason(),
            &EvaluationReason::ResourceLimit("solver_max_svd_work_units")
        );
        assert_eq!(report.evaluations().len(), 1);
        assert_eq!(
            report.evaluations()[0].status(),
            SolverStatus::BoundedUnknown
        );
    }

    #[test]
    fn bounded_builtin_work_does_not_hide_an_exact_external_candidate() {
        let problem = LeastSquaresProblem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![2.0], vec![3.0]],
        )
        .unwrap();
        let limits = SolverResourceLimits::default()
            .with_max_numerical_work_units(100)
            .unwrap();
        let policy = exact_policy().with_limits(limits).unwrap();
        let proposal = ExternalCandidateProposal::candidate(
            "test.external.exact",
            CandidateRepresentation::Dense {
                rows: 1,
                columns: 2,
                weights: vec![2.0, 3.0],
            },
        )
        .unwrap();
        let report = solve_with_portfolio(&problem, &policy, &[proposal]).unwrap();
        assert_eq!(report.status(), SolverStatus::Accepted);
        assert_eq!(
            report.selected().unwrap().backend().role(),
            NumericalBackendRole::ExternalProposal
        );
        assert!(report.diagnostics().is_none());
    }

    #[test]
    fn excess_external_proposals_never_create_an_order_dependent_selection() {
        let problem = LeastSquaresProblem::new(vec![vec![1.0]], vec![vec![1.0]]).unwrap();
        let limits = SolverResourceLimits::new(64, 16, 16, 1_024, 1_024, 1).unwrap();
        let policy = exact_policy().with_limits(limits).unwrap();
        let first = ExternalCandidateProposal::candidate(
            "test.external.first",
            CandidateRepresentation::Dense {
                rows: 1,
                columns: 1,
                weights: vec![1.0],
            },
        )
        .unwrap();
        let second = first.clone();
        for proposals in [vec![first.clone(), second.clone()], vec![second, first]] {
            let report = solve_with_portfolio(&problem, &policy, &proposals).unwrap();
            assert_eq!(report.status(), SolverStatus::BoundedUnknown);
            assert_eq!(
                report.reason(),
                &EvaluationReason::ResourceLimit("solver_external_proposal_budget")
            );
            assert!(report.selected().is_none());
            assert!(report.evaluations().is_empty());
        }
    }

    #[test]
    fn aggregate_external_candidate_storage_is_bounded_before_cloning() {
        let problem = LeastSquaresProblem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![1.0], vec![1.0]],
        )
        .unwrap();
        let limits = SolverResourceLimits::new(64, 16, 16, 1_024, 2, 2).unwrap();
        let policy = exact_policy().with_limits(limits).unwrap();
        let proposal = ExternalCandidateProposal::candidate(
            "test.external.aggregate",
            CandidateRepresentation::Dense {
                rows: 1,
                columns: 2,
                weights: vec![1.0, 1.0],
            },
        )
        .unwrap();
        let report =
            solve_with_portfolio(&problem, &policy, &[proposal.clone(), proposal]).unwrap();
        assert_eq!(report.status(), SolverStatus::BoundedUnknown);
        assert_eq!(
            report.reason(),
            &EvaluationReason::ResourceLimit("solver_max_total_external_parameters")
        );
        assert_eq!(report.external_proposals_considered(), 0);
    }

    #[test]
    fn direct_svd_work_limit_has_an_absolute_ceiling() {
        assert!(SolverResourceLimits::default()
            .with_max_numerical_work_units(ABSOLUTE_MAX_NUMERICAL_WORK_UNITS + 1)
            .is_err());
    }

    #[test]
    fn public_candidate_materialization_enforces_the_absolute_dense_limit() {
        let oversized = CandidateRepresentation::Sparse {
            rows: MAX_EXACT_CANDIDATE_DENSE_ELEMENTS + 1,
            columns: 1,
            entries: Vec::new(),
        };
        assert!(matches!(
            oversized.materialize_dense(),
            Err(BrainError::Invalid(message))
                if message == "solver_candidate_dense_materialization_limit"
        ));
    }

    #[test]
    fn sparse_and_block_claims_are_verified_from_canonical_structure() {
        let problem = LeastSquaresProblem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![2.0, 0.0], vec![0.0, 3.0]],
        )
        .unwrap();
        let sparse = external_candidate(
            "test.external.sparse",
            CandidateRepresentation::Sparse {
                rows: 2,
                columns: 2,
                entries: vec![
                    SparseEntry::new(0, 0, 2.0).unwrap(),
                    SparseEntry::new(1, 1, 3.0).unwrap(),
                ],
            },
        );
        let block = external_candidate(
            "test.external.block",
            CandidateRepresentation::Block {
                rows: 2,
                columns: 2,
                blocks: vec![
                    DenseBlock::new(0, 0, 1, 1, vec![2.0]).unwrap(),
                    DenseBlock::new(1, 1, 1, 1, vec![3.0]).unwrap(),
                ],
            },
        );
        for proposal in [sparse, block] {
            let report = solve_with_portfolio(&problem, &exact_policy(), &[proposal]).unwrap();
            let external = report
                .evaluations
                .iter()
                .find(|evaluation| {
                    evaluation.backend.role() == NumericalBackendRole::ExternalProposal
                })
                .unwrap();
            assert_eq!(external.status, SolverStatus::Accepted);
            assert_eq!(external.metrics.as_ref().unwrap().stored_parameter_count, 2);
            assert_eq!(external.metrics.as_ref().unwrap().absolute_residual, 0.0);
        }

        let duplicate_sparse = external_candidate(
            "test.external.duplicate_sparse",
            CandidateRepresentation::Sparse {
                rows: 2,
                columns: 2,
                entries: vec![
                    SparseEntry::new(0, 0, 1.0).unwrap(),
                    SparseEntry::new(0, 0, 1.0).unwrap(),
                ],
            },
        );
        let report = solve_with_portfolio(&problem, &exact_policy(), &[duplicate_sparse]).unwrap();
        let external = report.evaluations.last().unwrap();
        assert_eq!(external.status, SolverStatus::Rejected);
        assert!(matches!(
            external.reason,
            EvaluationReason::InvalidCandidate(_)
        ));
    }

    #[test]
    fn resource_exhaustion_is_unknown_not_numerical_rejection() {
        let problem = LeastSquaresProblem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![1.0], vec![1.0]],
        )
        .unwrap();
        let limits = SolverResourceLimits::new(1, 4096, 4096, 16_777_216, 16_777_216, 16).unwrap();
        let policy = exact_policy().with_limits(limits).unwrap();
        let report = solve_with_portfolio(&problem, &policy, &[]).unwrap();
        assert_eq!(report.status, SolverStatus::BoundedUnknown);
        assert_eq!(
            report.reason,
            EvaluationReason::ResourceLimit("solver_max_cases")
        );
        assert!(report.diagnostics.is_none());
    }

    #[test]
    fn compact_solution_retains_low_rank_factors_when_they_are_smaller() {
        let mut first = vec![0.0; 8];
        first[0] = 1.0;
        let mut second = vec![0.0; 8];
        second[0] = 2.0;
        let problem =
            LeastSquaresProblem::new(vec![first, second], vec![vec![1.0; 8], vec![2.0; 8]])
                .unwrap();
        let report = solve_with_portfolio(&problem, &exact_policy(), &[]).unwrap();
        assert_eq!(report.status, SolverStatus::Accepted);
        assert!(matches!(
            report.selected().unwrap().candidate,
            Some(CandidateRepresentation::LowRank { rank: 1, .. })
        ));
    }

    #[test]
    fn rank_cap_reports_bounded_unknown_instead_of_false_impossibility() {
        let problem = LeastSquaresProblem::new(
            vec![vec![2.0, 0.0], vec![0.0, 1.0]],
            vec![vec![2.0], vec![1.0]],
        )
        .unwrap();
        let policy = exact_policy()
            .with_rank_policy(1.0e-10, 0.75, 1, 1)
            .unwrap();
        let report = solve_with_portfolio(&problem, &policy, &[]).unwrap();
        let spectral = report
            .evaluations
            .iter()
            .find(|evaluation| {
                evaluation.backend.role() == NumericalBackendRole::BuiltInDirectJacobiSvd
            })
            .unwrap();
        assert_eq!(spectral.status, SolverStatus::BoundedUnknown);
        assert_eq!(spectral.reason, EvaluationReason::RankBudgetInsufficient);
    }
}
