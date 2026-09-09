//! Verified numerical core for the shadow LowRank/LoRA backend.
//!
//! This module factors an already compiled dense tensor delta. It does not
//! define capability identity and has no model-loading or activation API.

use crate::authority::{write_or_verify_immutable, PrivateFileReference};
use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::identity::TensorId;
use crate::lab_isolation::LabRoots;
use crate::linalg::{norm, symmetric_eigen_jacobi, Matrix};
use crate::receiver_layout::{
    ReceiverMaterializationLayout, ReceiverScalarEncoding, ReceiverTensorAlias,
    ReceiverTensorPartitioning,
};
use crate::receiver_profile::MaterializationStrategy;
use crate::solver_portfolio::CandidateRepresentation;
use crate::universal_capability_compiler::{
    replay_universal_capability_shadow_plan, UniversalCapabilityPlanningRequest,
    UniversalCapabilityShadowPlanReceipt,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const MAX_FACTOR_INPUT_ELEMENTS: usize = 16 * 1024 * 1024;
const MAX_FACTOR_GRAM_ELEMENTS: usize = 16 * 1024 * 1024;
const MAX_SHADOW_RANK: usize = 64;
const MAX_LOW_RANK_SHADOW_BYTES: u64 = 256 * 1024 * 1024;
const MAX_SVD_SWEEPS: usize = 1_000;
const MAX_SVD_ROTATIONS: usize = 100_000_000;
const MAX_FACTORIZATION_WORK_UNITS: usize = 250_000_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LowRankShadowPolicy {
    pub schema: String,
    pub maximum_rank: usize,
    pub relative_reconstruction_tolerance: f64,
    pub absolute_reconstruction_tolerance: f64,
    pub minimum_parameter_reduction_ratio: f64,
    pub maximum_svd_sweeps: usize,
}

impl LowRankShadowPolicy {
    pub fn validate(&self) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.low_rank_shadow_policy/v1"
            || self.maximum_rank == 0
            || self.maximum_rank > MAX_SHADOW_RANK
            || !self.relative_reconstruction_tolerance.is_finite()
            || !(0.0..1.0).contains(&self.relative_reconstruction_tolerance)
            || !self.absolute_reconstruction_tolerance.is_finite()
            || self.absolute_reconstruction_tolerance < 0.0
            || self.relative_reconstruction_tolerance == 0.0
                && self.absolute_reconstruction_tolerance == 0.0
            || !self.minimum_parameter_reduction_ratio.is_finite()
            || !(0.0..1.0).contains(&self.minimum_parameter_reduction_ratio)
            || self.maximum_svd_sweeps == 0
            || self.maximum_svd_sweeps > MAX_SVD_SWEEPS
        {
            return Err(BrainError::Invalid("low_rank_shadow_policy_invalid".into()));
        }
        Ok(())
    }

    pub fn digest(&self) -> BrainResult<Sha256Digest> {
        self.validate()?;
        Ok(Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:LOW-RANK-SHADOW-POLICY:v1\0",
            &serde_json::to_vec(self)?,
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VerifiedLowRankFactors {
    pub schema: String,
    pub rows: usize,
    pub columns: usize,
    pub rank: usize,
    pub left: Vec<f64>,
    pub right: Vec<f64>,
    pub absolute_reconstruction_error: f64,
    pub relative_reconstruction_error: f64,
    pub dense_parameter_count: usize,
    pub factor_parameter_count: usize,
    pub parameter_reduction_ratio: f64,
}

impl VerifiedLowRankFactors {
    pub fn materialize_dense(&self) -> BrainResult<Vec<f64>> {
        CandidateRepresentation::LowRank {
            rows: self.rows,
            columns: self.columns,
            rank: self.rank,
            left: self.left.clone(),
            right: self.right.clone(),
        }
        .materialize_dense()
    }
}

pub fn factor_dense_delta_verified(
    rows: usize,
    columns: usize,
    dense: &[f64],
    policy: &LowRankShadowPolicy,
) -> BrainResult<VerifiedLowRankFactors> {
    policy.validate()?;
    let dense_count = rows
        .checked_mul(columns)
        .ok_or_else(|| BrainError::Invalid("low_rank_dense_shape_overflow".into()))?;
    let spectral_dimension = rows.min(columns);
    let gram_count = spectral_dimension
        .checked_mul(spectral_dimension)
        .ok_or_else(|| BrainError::Invalid("low_rank_gram_shape_overflow".into()))?;
    if rows == 0
        || columns == 0
        || dense.len() != dense_count
        || dense_count > MAX_FACTOR_INPUT_ELEMENTS
        || gram_count > MAX_FACTOR_GRAM_ELEMENTS
        || dense.iter().any(|value| !value.is_finite())
    {
        return Err(BrainError::Invalid("low_rank_dense_input_invalid".into()));
    }
    // Jacobi scans the Gram matrix per rotation, with an eight-sweep floor.
    // Bound real work before computing Gram or allocating spectral matrices.
    let rotations = gram_count
        .checked_mul(policy.maximum_svd_sweeps.max(8))
        .ok_or_else(|| BrainError::Invalid("low_rank_svd_work_overflow".into()))?;
    let effective_rank = policy.maximum_rank.min(spectral_dimension);
    let factor_work = effective_rank
        .checked_mul(effective_rank + 1)
        .and_then(|r| r.checked_mul(dense_count));
    let work = spectral_dimension
        .checked_mul(32)
        .and_then(|w| gram_count.checked_add(w))
        .and_then(|w| w.checked_mul(rotations))
        .and_then(|w| {
            gram_count
                .checked_mul(rows.max(columns))
                .and_then(|g| g.checked_add(w))
        })
        .and_then(|w| factor_work.and_then(|f| w.checked_add(f)));
    if rotations > MAX_SVD_ROTATIONS || work.is_none_or(|w| w > MAX_FACTORIZATION_WORK_UNITS) {
        return Err(BrainError::Invalid("low_rank_svd_work_limit".into()));
    }
    let target_norm = norm(dense)?;
    let scale = dense
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(BrainError::Numerical(
            "low_rank_zero_delta_has_no_factors".into(),
        ));
    }
    let mut gram = Matrix::zeros(spectral_dimension, spectral_dimension);
    if rows <= columns {
        for first in 0..rows {
            for second in first..rows {
                let value = (0..columns)
                    .map(|column| {
                        (dense[first * columns + column] / scale)
                            * (dense[second * columns + column] / scale)
                    })
                    .sum::<f64>();
                gram.set(first, second, value);
                gram.set(second, first, value);
            }
        }
    } else {
        for first in 0..columns {
            for second in first..columns {
                let value = (0..rows)
                    .map(|row| {
                        (dense[row * columns + first] / scale)
                            * (dense[row * columns + second] / scale)
                    })
                    .sum::<f64>();
                gram.set(first, second, value);
                gram.set(second, first, value);
            }
        }
    }
    let eigen = symmetric_eigen_jacobi(&gram, 1.0e-12, rotations)?;
    let maximum_rank = policy.maximum_rank.min(eigen.len());
    for rank in 1..=maximum_rank {
        let factor_count = rows
            .checked_mul(rank)
            .and_then(|value| value.checked_add(rank.checked_mul(columns)?))
            .ok_or_else(|| BrainError::Invalid("low_rank_factor_shape_overflow".into()))?;
        let parameter_reduction_ratio = 1.0 - factor_count as f64 / dense_count as f64;
        if parameter_reduction_ratio < policy.minimum_parameter_reduction_ratio {
            continue;
        }
        let mut left = vec![0.0; rows * rank];
        let mut right = vec![0.0; rank * columns];
        for component in 0..rank {
            let vector = &eigen[component].1;
            let scaled_sigma = eigen[component].0.sqrt();
            let balanced_scale = scale.sqrt() * scaled_sigma.sqrt();
            if !scaled_sigma.is_finite()
                || scaled_sigma <= 0.0
                || !balanced_scale.is_finite()
                || balanced_scale <= 0.0
            {
                return Err(BrainError::Numerical(
                    "low_rank_singular_value_invalid".into(),
                ));
            }
            if rows <= columns {
                for row in 0..rows {
                    left[row * rank + component] = vector[row] * balanced_scale;
                }
                for column in 0..columns {
                    right[component * columns + column] = (0..rows)
                        .map(|row| {
                            vector[row] * (dense[row * columns + column] / scale) / scaled_sigma
                        })
                        .sum::<f64>()
                        * balanced_scale;
                }
            } else {
                for row in 0..rows {
                    left[row * rank + component] = (0..columns)
                        .map(|column| {
                            (dense[row * columns + column] / scale) * vector[column] / scaled_sigma
                        })
                        .sum::<f64>()
                        * balanced_scale;
                }
                for column in 0..columns {
                    right[component * columns + column] = vector[column] * balanced_scale;
                }
            }
        }
        let candidate = CandidateRepresentation::LowRank {
            rows,
            columns,
            rank,
            left: left.clone(),
            right: right.clone(),
        };
        let reconstructed = candidate.materialize_dense()?;
        let residual = dense
            .iter()
            .zip(&reconstructed)
            .map(|(expected, actual)| expected - actual)
            .collect::<Vec<_>>();
        let absolute = norm(&residual)?;
        let relative = absolute / target_norm;
        if absolute.is_finite()
            && relative.is_finite()
            && (absolute <= policy.absolute_reconstruction_tolerance
                || relative <= policy.relative_reconstruction_tolerance)
        {
            return Ok(VerifiedLowRankFactors {
                schema: "cerebro.tidex.verified_low_rank_factors/v1".into(),
                rows,
                columns,
                rank,
                left,
                right,
                absolute_reconstruction_error: absolute,
                relative_reconstruction_error: relative,
                dense_parameter_count: dense_count,
                factor_parameter_count: factor_count,
                parameter_reduction_ratio,
            });
        }
    }
    Err(BrainError::Numerical(
        "low_rank_factorization_not_admissible".into(),
    ))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ShadowLowRankTensorDelta {
    pub tensor_id: TensorId,
    pub shape: Vec<usize>,
    pub base_encoding: ReceiverScalarEncoding,
    pub partitioning: ReceiverTensorPartitioning,
    pub dense_values_sha256: Sha256Digest,
    pub factors: VerifiedLowRankFactors,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ShadowLowRankCandidate {
    pub schema: String,
    pub planning_request_sha256: Sha256Digest,
    pub receiver_layout_sha256: Sha256Digest,
    pub policy_sha256: Sha256Digest,
    pub target_delta_sha256: Sha256Digest,
    pub tensors: Vec<ShadowLowRankTensorDelta>,
    pub aliases: Vec<ReceiverTensorAlias>,
    pub manifest_sha256: Sha256Digest,
}

fn values_digest(values: &[f64]) -> BrainResult<Sha256Digest> {
    Ok(Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:SHADOW-LOW-RANK-DENSE-VALUES:v1\0",
        &serde_json::to_vec(values)?,
    ))
}

impl ShadowLowRankCandidate {
    fn calculate_digest(&self) -> BrainResult<Sha256Digest> {
        let mut unsigned = self.clone();
        unsigned.manifest_sha256 = Sha256Digest::zero();
        // Normalize through the actual JSON wire representation. Numerical
        // factors can be rounded to their shortest round-trippable decimal by
        // serde_json; hashing the normalized value keeps identity stable after
        // persistence while semantic validation still checks reconstructed f64s.
        let wire_value: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&unsigned)?)?;
        Ok(Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:SHADOW-LOW-RANK-CANDIDATE:v1\0",
            &serde_json::to_vec(&wire_value)?,
        ))
    }

    pub fn validate(
        &self,
        request: &UniversalCapabilityPlanningRequest,
        receipt: &UniversalCapabilityShadowPlanReceipt,
        layout: &ReceiverMaterializationLayout,
        policy: &LowRankShadowPolicy,
    ) -> BrainResult<()> {
        replay_universal_capability_shadow_plan(request, receipt)?;
        layout.validate_for(&request.receiver_profile, &request.receiver_snapshot)?;
        policy.validate()?;
        let shadow = &receipt.shadow_plan;
        let target = &shadow.compilation_receipt.compilation.receiver.target_delta;
        if shadow.materialization_plan.strategy != MaterializationStrategy::LowRank {
            return Err(BrainError::Invalid(
                "shadow_low_rank_strategy_required".into(),
            ));
        }
        if self.schema != "cerebro.tidex.shadow_low_rank_candidate/v1"
            || self.planning_request_sha256 != receipt.planning_request_sha256
            || self.receiver_layout_sha256 != layout.manifest_sha256
            || self.policy_sha256 != policy.digest()?
            || self.target_delta_sha256 != values_digest(target)?
            || self.aliases != layout.aliases
        {
            return Err(BrainError::Integrity(
                "shadow_low_rank_binding_invalid".into(),
            ));
        }
        if self.manifest_sha256 != self.calculate_digest()? {
            return Err(BrainError::Integrity(
                "shadow_low_rank_manifest_invalid".into(),
            ));
        }
        let affected = shadow
            .materialization_plan
            .affected_regions
            .iter()
            .collect::<BTreeSet<_>>();
        let mut tensors = self.tensors.iter();
        for ((block, physical), region) in layout
            .geometry
            .layout
            .blocks
            .iter()
            .zip(&layout.physical_tensors)
            .zip(&request.receiver_profile.regions)
        {
            let start = usize::try_from(block.offset)
                .map_err(|_| BrainError::Invalid("shadow_low_rank_offset_overflow".into()))?;
            let end = start
                .checked_add(block.count)
                .ok_or_else(|| BrainError::Invalid("shadow_low_rank_range_overflow".into()))?;
            let dense = target
                .get(start..end)
                .ok_or_else(|| BrainError::Integrity("shadow_low_rank_range_invalid".into()))?;
            if !affected.contains(&region.tensor_id) {
                if dense.iter().any(|value| *value != 0.0) {
                    return Err(BrainError::Integrity(
                        "shadow_low_rank_nonzero_outside_planned_regions".into(),
                    ));
                }
                continue;
            }
            let tensor = tensors
                .next()
                .ok_or_else(|| BrainError::Integrity("shadow_low_rank_tensor_missing".into()))?;
            let factors = &tensor.factors;
            let factor_count = factors
                .rows
                .checked_mul(factors.rank)
                .and_then(|count| count.checked_add(factors.rank.checked_mul(factors.columns)?))
                .ok_or_else(|| BrainError::Invalid("low_rank_factor_shape_overflow".into()))?;
            let expected_reduction = 1.0 - factor_count as f64 / block.count as f64;
            if block.shape.len() != 2
                || tensor.tensor_id != region.tensor_id
                || tensor.shape != block.shape
                || tensor.base_encoding != physical.encoding
                || tensor.partitioning != physical.partitioning
                || tensor.dense_values_sha256 != values_digest(dense)?
                || factors.schema != "cerebro.tidex.verified_low_rank_factors/v1"
                || factors.rows != block.shape[0]
                || factors.columns != block.shape[1]
                || factors.rank == 0
                || factors.rank > policy.maximum_rank
                || factors.left.len() != factors.rows * factors.rank
                || factors.right.len() != factors.rank * factors.columns
                || factors
                    .left
                    .iter()
                    .chain(&factors.right)
                    .any(|value| !value.is_finite())
                || factors.dense_parameter_count != block.count
                || factors.factor_parameter_count != factor_count
                || (factors.parameter_reduction_ratio - expected_reduction).abs() > 1e-12
                || factors.parameter_reduction_ratio < policy.minimum_parameter_reduction_ratio
            {
                return Err(BrainError::Integrity(
                    "shadow_low_rank_factor_contract_invalid".into(),
                ));
            }
            let reconstructed = factors.materialize_dense()?;
            let residual = dense
                .iter()
                .zip(&reconstructed)
                .map(|(expected, actual)| expected - actual)
                .collect::<Vec<_>>();
            let absolute = norm(&residual)?;
            let target_norm = norm(dense)?;
            let relative = if target_norm == 0.0 {
                0.0
            } else {
                absolute / target_norm
            };
            let error_slack = 1e-12 * (1.0 + absolute.abs() + relative.abs());
            if (absolute - factors.absolute_reconstruction_error).abs() > error_slack
                || (relative - factors.relative_reconstruction_error).abs() > error_slack
                || !(absolute <= policy.absolute_reconstruction_tolerance
                    || relative <= policy.relative_reconstruction_tolerance)
            {
                return Err(BrainError::Integrity(
                    "shadow_low_rank_reconstruction_invalid".into(),
                ));
            }
        }
        if tensors.next().is_some() {
            return Err(BrainError::Integrity(
                "shadow_low_rank_tensor_surplus".into(),
            ));
        }
        Ok(())
    }
}

fn build_candidate(
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    layout: &ReceiverMaterializationLayout,
    policy: &LowRankShadowPolicy,
) -> BrainResult<ShadowLowRankCandidate> {
    replay_universal_capability_shadow_plan(request, receipt)?;
    layout.validate_for(&request.receiver_profile, &request.receiver_snapshot)?;
    policy.validate()?;
    let shadow = &receipt.shadow_plan;
    if shadow.materialization_plan.strategy != MaterializationStrategy::LowRank {
        return Err(BrainError::Invalid(
            "shadow_low_rank_strategy_required".into(),
        ));
    }
    let target = &shadow.compilation_receipt.compilation.receiver.target_delta;
    if target.len() > MAX_FACTOR_INPUT_ELEMENTS
        || target.iter().any(|value| !value.is_finite())
        || u64::try_from(target.len())
            .map_err(|_| BrainError::Invalid("shadow_low_rank_target_overflow".into()))?
            != layout.geometry.total_parameter_count
    {
        return Err(BrainError::Integrity(
            "shadow_low_rank_target_invalid".into(),
        ));
    }
    let affected = shadow
        .materialization_plan
        .affected_regions
        .iter()
        .collect::<BTreeSet<_>>();
    let mut tensors = Vec::with_capacity(affected.len());
    for ((block, physical), region) in layout
        .geometry
        .layout
        .blocks
        .iter()
        .zip(&layout.physical_tensors)
        .zip(&request.receiver_profile.regions)
    {
        let start = usize::try_from(block.offset)
            .map_err(|_| BrainError::Invalid("shadow_low_rank_offset_overflow".into()))?;
        let end = start
            .checked_add(block.count)
            .ok_or_else(|| BrainError::Invalid("shadow_low_rank_range_overflow".into()))?;
        let values = target
            .get(start..end)
            .ok_or_else(|| BrainError::Integrity("shadow_low_rank_range_invalid".into()))?;
        if affected.contains(&region.tensor_id) {
            if block.shape.len() != 2 {
                return Err(BrainError::Invalid(
                    "shadow_low_rank_tensor_must_be_matrix".into(),
                ));
            }
            tensors.push(ShadowLowRankTensorDelta {
                tensor_id: region.tensor_id.clone(),
                shape: block.shape.clone(),
                base_encoding: physical.encoding.clone(),
                partitioning: physical.partitioning.clone(),
                dense_values_sha256: values_digest(values)?,
                factors: factor_dense_delta_verified(
                    block.shape[0],
                    block.shape[1],
                    values,
                    policy,
                )?,
            });
        } else if values.iter().any(|value| *value != 0.0) {
            return Err(BrainError::Integrity(
                "shadow_low_rank_nonzero_outside_planned_regions".into(),
            ));
        }
    }
    let mut candidate = ShadowLowRankCandidate {
        schema: "cerebro.tidex.shadow_low_rank_candidate/v1".into(),
        planning_request_sha256: receipt.planning_request_sha256.clone(),
        receiver_layout_sha256: layout.manifest_sha256.clone(),
        policy_sha256: policy.digest()?,
        target_delta_sha256: values_digest(target)?,
        tensors,
        aliases: layout.aliases.clone(),
        manifest_sha256: Sha256Digest::zero(),
    };
    // Canonicalize floating-point wire values before assigning artifact
    // identity so a persisted candidate has exactly the same semantics and
    // digest after deserialization.
    candidate = serde_json::from_slice(&serde_json::to_vec(&candidate)?)?;
    candidate.manifest_sha256 = candidate.calculate_digest()?;
    Ok(candidate)
}

pub fn materialize_replayed_low_rank_shadow(
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    layout: &ReceiverMaterializationLayout,
    policy: &LowRankShadowPolicy,
) -> BrainResult<ShadowLowRankCandidate> {
    let candidate = build_candidate(request, receipt, layout, policy)?;
    candidate.validate(request, receipt, layout, policy)?;
    Ok(candidate)
}

pub fn persist_low_rank_shadow(
    roots: &LabRoots,
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    layout: &ReceiverMaterializationLayout,
    policy: &LowRankShadowPolicy,
    candidate: &ShadowLowRankCandidate,
) -> BrainResult<PrivateFileReference> {
    candidate.validate(request, receipt, layout, policy)?;
    let bytes = serde_json::to_vec(candidate)?;
    if u64::try_from(bytes.len())
        .map_err(|_| BrainError::Invalid("shadow_low_rank_size_overflow".into()))?
        > MAX_LOW_RANK_SHADOW_BYTES
    {
        return Err(BrainError::Invalid(
            "shadow_low_rank_candidate_too_large".into(),
        ));
    }
    let destination = roots
        .artifact_root()
        .join("low-rank-shadow-candidates")
        .join(format!("{}.json", candidate.manifest_sha256));
    let sha256 = write_or_verify_immutable(roots.lab_root(), &destination, &bytes)?;
    Ok(PrivateFileReference::new(destination, sha256))
}

pub fn load_low_rank_shadow(
    roots: &LabRoots,
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    layout: &ReceiverMaterializationLayout,
    policy: &LowRankShadowPolicy,
    reference: &PrivateFileReference,
) -> BrainResult<ShadowLowRankCandidate> {
    let bytes = reference.read_verified_bounded(roots.lab_root(), MAX_LOW_RANK_SHADOW_BYTES)?;
    let candidate: ShadowLowRankCandidate = serde_json::from_slice(&bytes)?;
    let expected_path = roots
        .artifact_root()
        .join("low-rank-shadow-candidates")
        .join(format!("{}.json", candidate.manifest_sha256));
    if reference.path != expected_path {
        return Err(BrainError::Integrity(
            "shadow_low_rank_candidate_path_invalid".into(),
        ));
    }
    candidate.validate(request, receipt, layout, policy)?;
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(maximum_rank: usize) -> LowRankShadowPolicy {
        LowRankShadowPolicy {
            schema: "cerebro.tidex.low_rank_shadow_policy/v1".into(),
            maximum_rank,
            relative_reconstruction_tolerance: 1.0e-10,
            absolute_reconstruction_tolerance: 1.0e-10,
            minimum_parameter_reduction_ratio: 0.4,
            maximum_svd_sweeps: 100,
        }
    }

    #[test]
    fn exact_low_rank_delta_is_factored_and_full_rank_delta_is_rejected() {
        let left = [1.0, 2.0, 3.0, 4.0];
        let right = [0.5, -1.0, 2.0, 0.25];
        let dense = left
            .iter()
            .flat_map(|left| right.iter().map(move |right| left * right))
            .collect::<Vec<_>>();
        let factors = factor_dense_delta_verified(4, 4, &dense, &policy(1)).unwrap();
        assert_eq!(factors.rank, 1);
        assert!(factors.parameter_reduction_ratio >= 0.4);
        let reconstructed = factors.materialize_dense().unwrap();
        assert!(dense
            .iter()
            .zip(reconstructed)
            .all(|(expected, actual)| (expected - actual).abs() < 1.0e-10));

        let identity = (0..16)
            .map(|index| f64::from(index / 4 == index % 4))
            .collect::<Vec<_>>();
        assert!(factor_dense_delta_verified(4, 4, &identity, &policy(1)).is_err());
    }

    #[test]
    fn rectangular_orientations_and_finite_scaling_reconstruct() {
        for (rows, columns, scale) in [(3, 8, 1.0e100), (8, 3, 1.0e-100)] {
            let left = (0..rows)
                .map(|index| (index + 1) as f64 * scale)
                .collect::<Vec<_>>();
            let right = (0..columns)
                .map(|index| (index as f64 + 0.5) / scale.sqrt())
                .collect::<Vec<_>>();
            let dense = (0..rows)
                .flat_map(|row| {
                    let left = &left;
                    let right = &right;
                    (0..columns).map(move |column| left[row] * right[column])
                })
                .collect::<Vec<_>>();
            let factors = factor_dense_delta_verified(rows, columns, &dense, &policy(1)).unwrap();
            assert_eq!(factors.rank, 1);
            assert!(factors.relative_reconstruction_error < 1.0e-10);
            assert!(factors
                .materialize_dense()
                .unwrap()
                .iter()
                .all(|value| value.is_finite()));
        }
    }
    #[test]
    fn convergence_regression_factorization_bounds_work_before_gram_construction() {
        let dense = vec![0.0; 512 * 512];
        let error = factor_dense_delta_verified(512, 512, &dense, &policy(1)).unwrap_err();
        assert!(error.to_string().contains("low_rank_svd_work_limit"));
    }
}
