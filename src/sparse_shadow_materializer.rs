//! Replay-bound sparse materialization of a compiler-produced receiver delta.
//!
//! Sparse coordinates are selected deterministically by magnitude, reconstructed
//! independently, and admitted only when both error and storage gates pass.

use crate::authority::{write_or_verify_immutable, PrivateFileReference};
use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::identity::TensorId;
use crate::lab_isolation::LabRoots;
use crate::linalg::norm;
use crate::receiver_layout::{
    ReceiverMaterializationLayout, ReceiverScalarEncoding, ReceiverTensorPartitioning,
};
use crate::receiver_profile::MaterializationStrategy;
use crate::universal_capability_compiler::{
    replay_universal_capability_shadow_plan, UniversalCapabilityPlanningRequest,
    UniversalCapabilityShadowPlanReceipt,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeSet;

const MAX_SPARSE_ELEMENTS: usize = 16 * 1024 * 1024;
const MAX_SPARSE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SparseShadowPolicy {
    pub schema: String,
    pub maximum_nonzero_count: usize,
    pub maximum_density: f64,
    pub absolute_zero_threshold: f64,
    pub relative_reconstruction_tolerance: f64,
    pub absolute_reconstruction_tolerance: f64,
    pub minimum_storage_reduction_ratio: f64,
}

impl SparseShadowPolicy {
    pub fn validate(&self) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.sparse_shadow_policy/v1"
            || self.maximum_nonzero_count == 0
            || self.maximum_nonzero_count > MAX_SPARSE_ELEMENTS
            || !self.maximum_density.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_density)
            || self.maximum_density == 0.0
            || !self.absolute_zero_threshold.is_finite()
            || self.absolute_zero_threshold < 0.0
            || !self.relative_reconstruction_tolerance.is_finite()
            || !(0.0..=1.0).contains(&self.relative_reconstruction_tolerance)
            || !self.absolute_reconstruction_tolerance.is_finite()
            || self.absolute_reconstruction_tolerance < 0.0
            || self.relative_reconstruction_tolerance == 0.0
                && self.absolute_reconstruction_tolerance == 0.0
            || !self.minimum_storage_reduction_ratio.is_finite()
            || !(0.0..1.0).contains(&self.minimum_storage_reduction_ratio)
        {
            return Err(BrainError::Invalid("sparse_shadow_policy_invalid".into()));
        }
        Ok(())
    }

    pub fn digest(&self) -> BrainResult<Sha256Digest> {
        self.validate()?;
        Ok(Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:SPARSE-SHADOW-POLICY:v1\0",
            &serde_json::to_vec(self)?,
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SparseCoordinate {
    pub flat_index: u64,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ShadowSparseTensorDelta {
    pub tensor_id: TensorId,
    pub shape: Vec<usize>,
    pub base_encoding: ReceiverScalarEncoding,
    pub partitioning: ReceiverTensorPartitioning,
    pub dense_values_sha256: Sha256Digest,
    pub coordinates: Vec<SparseCoordinate>,
    pub absolute_reconstruction_error: f64,
    pub relative_reconstruction_error: f64,
    pub retained_energy_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ShadowSparseCandidate {
    pub schema: String,
    pub planning_request_sha256: Sha256Digest,
    pub receiver_layout_sha256: Sha256Digest,
    pub policy_sha256: Sha256Digest,
    pub target_delta_sha256: Sha256Digest,
    pub dense_element_count: usize,
    pub nonzero_count: usize,
    pub density: f64,
    pub estimated_storage_reduction_ratio: f64,
    pub tensors: Vec<ShadowSparseTensorDelta>,
    pub manifest_sha256: Sha256Digest,
}

fn values_digest(values: &[f64]) -> BrainResult<Sha256Digest> {
    Ok(Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:SHADOW-SPARSE-DENSE-VALUES:v1\0",
        &serde_json::to_vec(values)?,
    ))
}

fn sparsify(
    values: &[f64],
    budget: usize,
    policy: &SparseShadowPolicy,
) -> BrainResult<(Vec<SparseCoordinate>, f64, f64, f64)> {
    if values.is_empty()
        || values.len() > MAX_SPARSE_ELEMENTS
        || values.iter().any(|v| !v.is_finite())
    {
        return Err(BrainError::Invalid("sparse_dense_input_invalid".into()));
    }
    let target_norm = norm(values)?;
    let mut ranked = values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.abs() > policy.absolute_zero_threshold)
        .collect::<Vec<_>>();
    ranked.sort_by(|(ia, a), (ib, b)| {
        b.abs()
            .partial_cmp(&a.abs())
            .unwrap_or(Ordering::Equal)
            .then_with(|| ia.cmp(ib))
    });
    ranked.truncate(budget.min(ranked.len()));
    ranked.sort_by_key(|(index, _)| *index);
    let coordinates = ranked
        .into_iter()
        .map(|(index, value)| {
            Ok(SparseCoordinate {
                flat_index: u64::try_from(index)
                    .map_err(|_| BrainError::Invalid("sparse_index_overflow".into()))?,
                value,
            })
        })
        .collect::<BrainResult<Vec<_>>>()?;
    let mut reconstructed = vec![0.0; values.len()];
    for coordinate in &coordinates {
        let index = usize::try_from(coordinate.flat_index)
            .map_err(|_| BrainError::Invalid("sparse_index_overflow".into()))?;
        *reconstructed
            .get_mut(index)
            .ok_or_else(|| BrainError::Integrity("sparse_index_out_of_bounds".into()))? =
            coordinate.value;
    }
    let residual = values
        .iter()
        .zip(&reconstructed)
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    let absolute = norm(&residual)?;
    let relative = if target_norm == 0.0 {
        0.0
    } else {
        absolute / target_norm
    };
    let reconstructed_norm = norm(&reconstructed)?;
    let retained = if target_norm == 0.0 {
        1.0
    } else {
        (reconstructed_norm / target_norm).powi(2).clamp(0.0, 1.0)
    };
    if !(absolute <= policy.absolute_reconstruction_tolerance
        || relative <= policy.relative_reconstruction_tolerance)
    {
        return Err(BrainError::Numerical(
            "sparse_reconstruction_not_admissible".into(),
        ));
    }
    Ok((coordinates, absolute, relative, retained))
}

impl ShadowSparseCandidate {
    fn calculate_digest(&self) -> BrainResult<Sha256Digest> {
        let mut unsigned = self.clone();
        unsigned.manifest_sha256 = Sha256Digest::zero();
        Ok(Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:SHADOW-SPARSE-CANDIDATE:v1\0",
            &serde_json::to_vec(&unsigned)?,
        ))
    }

    pub fn validate(
        &self,
        request: &UniversalCapabilityPlanningRequest,
        receipt: &UniversalCapabilityShadowPlanReceipt,
        layout: &ReceiverMaterializationLayout,
        policy: &SparseShadowPolicy,
    ) -> BrainResult<()> {
        if self != &build_candidate(request, receipt, layout, policy)? {
            return Err(BrainError::Integrity(
                "shadow_sparse_candidate_invalid".into(),
            ));
        }
        Ok(())
    }
}

fn build_candidate(
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    layout: &ReceiverMaterializationLayout,
    policy: &SparseShadowPolicy,
) -> BrainResult<ShadowSparseCandidate> {
    replay_universal_capability_shadow_plan(request, receipt)?;
    layout.validate_for(&request.receiver_profile, &request.receiver_snapshot)?;
    policy.validate()?;
    if receipt.shadow_plan.materialization_plan.strategy != MaterializationStrategy::SparseDelta {
        return Err(BrainError::Invalid(
            "shadow_sparse_strategy_required".into(),
        ));
    }
    let target = &receipt
        .shadow_plan
        .compilation_receipt
        .compilation
        .receiver
        .target_delta;
    if target.len() > MAX_SPARSE_ELEMENTS
        || target.iter().any(|v| !v.is_finite())
        || u64::try_from(target.len()).ok() != Some(layout.geometry.total_parameter_count)
    {
        return Err(BrainError::Integrity("shadow_sparse_target_invalid".into()));
    }
    let affected = receipt
        .shadow_plan
        .materialization_plan
        .affected_regions
        .iter()
        .collect::<BTreeSet<_>>();
    let global_budget = policy
        .maximum_nonzero_count
        .min(((target.len() as f64) * policy.maximum_density).floor() as usize);
    if global_budget == 0 {
        return Err(BrainError::Invalid("sparse_budget_empty".into()));
    }
    let affected_count = request
        .receiver_profile
        .regions
        .iter()
        .filter(|r| affected.contains(&r.tensor_id))
        .map(|r| r.parameter_count as usize)
        .sum::<usize>();
    let mut remaining_budget = global_budget.min(affected_count);
    let mut remaining_elements = affected_count;
    let mut tensors = Vec::new();
    for ((block, physical), region) in layout
        .geometry
        .layout
        .blocks
        .iter()
        .zip(&layout.physical_tensors)
        .zip(&request.receiver_profile.regions)
    {
        let start = usize::try_from(block.offset)
            .map_err(|_| BrainError::Invalid("sparse_offset_overflow".into()))?;
        let end = start
            .checked_add(block.count)
            .ok_or_else(|| BrainError::Invalid("sparse_range_overflow".into()))?;
        let values = target
            .get(start..end)
            .ok_or_else(|| BrainError::Integrity("sparse_range_invalid".into()))?;
        if affected.contains(&region.tensor_id) {
            let budget = if remaining_elements == block.count {
                remaining_budget
            } else {
                ((remaining_budget as u128 * block.count as u128) / remaining_elements as u128)
                    as usize
            }
            .max(usize::from(remaining_budget > 0))
            .min(block.count)
            .min(remaining_budget);
            let (coordinates, absolute, relative, retained) = sparsify(values, budget, policy)?;
            remaining_budget -= budget;
            remaining_elements -= block.count;
            tensors.push(ShadowSparseTensorDelta {
                tensor_id: region.tensor_id.clone(),
                shape: block.shape.clone(),
                base_encoding: physical.encoding.clone(),
                partitioning: physical.partitioning.clone(),
                dense_values_sha256: values_digest(values)?,
                coordinates,
                absolute_reconstruction_error: absolute,
                relative_reconstruction_error: relative,
                retained_energy_ratio: retained,
            });
        } else if values.iter().any(|v| *v != 0.0) {
            return Err(BrainError::Integrity(
                "shadow_sparse_nonzero_outside_planned_regions".into(),
            ));
        }
    }
    let nonzero_count = tensors.iter().map(|t| t.coordinates.len()).sum::<usize>();
    let density = nonzero_count as f64 / target.len() as f64;
    // Each sparse value uses an index and a value; require actual reduction versus f64 dense storage.
    let sparse_bytes = nonzero_count
        .checked_mul(16)
        .ok_or_else(|| BrainError::Invalid("sparse_storage_overflow".into()))?;
    let dense_bytes = target
        .len()
        .checked_mul(8)
        .ok_or_else(|| BrainError::Invalid("sparse_storage_overflow".into()))?;
    let reduction = 1.0 - sparse_bytes as f64 / dense_bytes as f64;
    if density > policy.maximum_density || reduction < policy.minimum_storage_reduction_ratio {
        return Err(BrainError::Numerical(
            "sparse_storage_not_admissible".into(),
        ));
    }
    let mut candidate = ShadowSparseCandidate {
        schema: "cerebro.tidex.shadow_sparse_candidate/v1".into(),
        planning_request_sha256: receipt.planning_request_sha256.clone(),
        receiver_layout_sha256: layout.manifest_sha256.clone(),
        policy_sha256: policy.digest()?,
        target_delta_sha256: values_digest(target)?,
        dense_element_count: target.len(),
        nonzero_count,
        density,
        estimated_storage_reduction_ratio: reduction,
        tensors,
        manifest_sha256: Sha256Digest::zero(),
    };
    candidate.manifest_sha256 = candidate.calculate_digest()?;
    Ok(candidate)
}

pub fn materialize_replayed_sparse_shadow(
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    layout: &ReceiverMaterializationLayout,
    policy: &SparseShadowPolicy,
) -> BrainResult<ShadowSparseCandidate> {
    let candidate = build_candidate(request, receipt, layout, policy)?;
    candidate.validate(request, receipt, layout, policy)?;
    Ok(candidate)
}

pub fn persist_sparse_shadow(
    roots: &LabRoots,
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    layout: &ReceiverMaterializationLayout,
    policy: &SparseShadowPolicy,
    candidate: &ShadowSparseCandidate,
) -> BrainResult<PrivateFileReference> {
    candidate.validate(request, receipt, layout, policy)?;
    let bytes = serde_json::to_vec(candidate)?;
    if u64::try_from(bytes.len())
        .ok()
        .filter(|n| *n <= MAX_SPARSE_BYTES)
        .is_none()
    {
        return Err(BrainError::Invalid(
            "shadow_sparse_candidate_too_large".into(),
        ));
    }
    let path = roots
        .artifact_root()
        .join("sparse-shadow-candidates")
        .join(format!("{}.json", candidate.manifest_sha256));
    let sha256 = write_or_verify_immutable(roots.lab_root(), &path, &bytes)?;
    Ok(PrivateFileReference::new(path, sha256))
}

pub fn load_sparse_shadow(
    roots: &LabRoots,
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    layout: &ReceiverMaterializationLayout,
    policy: &SparseShadowPolicy,
    reference: &PrivateFileReference,
) -> BrainResult<ShadowSparseCandidate> {
    let bytes = reference.read_verified_bounded(roots.lab_root(), MAX_SPARSE_BYTES)?;
    let candidate: ShadowSparseCandidate = serde_json::from_slice(&bytes)?;
    let expected = roots
        .artifact_root()
        .join("sparse-shadow-candidates")
        .join(format!("{}.json", candidate.manifest_sha256));
    if reference.path != expected {
        return Err(BrainError::Integrity(
            "shadow_sparse_candidate_path_invalid".into(),
        ));
    }
    candidate.validate(request, receipt, layout, policy)?;
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> SparseShadowPolicy {
        SparseShadowPolicy {
            schema: "cerebro.tidex.sparse_shadow_policy/v1".into(),
            maximum_nonzero_count: 2,
            maximum_density: 0.25,
            absolute_zero_threshold: 0.0,
            relative_reconstruction_tolerance: 1e-12,
            absolute_reconstruction_tolerance: 1e-12,
            minimum_storage_reduction_ratio: 0.4,
        }
    }

    #[test]
    fn deterministic_top_magnitude_sparse_encoding_and_rejection() {
        let (coordinates, absolute, relative, retained) =
            sparsify(&[0.0, 3.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0], 2, &policy()).unwrap();
        assert_eq!(
            coordinates,
            vec![
                SparseCoordinate {
                    flat_index: 1,
                    value: 3.0
                },
                SparseCoordinate {
                    flat_index: 3,
                    value: -2.0
                }
            ]
        );
        assert_eq!((absolute, relative, retained), (0.0, 0.0, 1.0));
        assert!(sparsify(&[1.0, 2.0, 3.0, 4.0], 2, &policy()).is_err());
    }
}
