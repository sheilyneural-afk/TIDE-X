//! Authenticated activation-steering compilation for shadow execution.
//!
//! Parameter coordinates are never re-labelled as activations. Every hook owns
//! an explicit, bounded linear projection from one receiver tensor to an
//! activation vector, plus runtime placement and token-selection semantics.

use crate::authority::{write_or_verify_immutable, PrivateFileReference};
use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::identity::TensorId;
use crate::lab_isolation::LabRoots;
use crate::linalg::norm;
use crate::receiver_layout::ReceiverMaterializationLayout;
use crate::receiver_profile::MaterializationStrategy;
use crate::universal_capability_compiler::{
    replay_universal_capability_shadow_plan, UniversalCapabilityPlanningRequest,
    UniversalCapabilityShadowPlanReceipt,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const MAX_HOOKS: usize = 16_384;
const MAX_ACTIVATION_WIDTH: usize = 1_048_576;
const MAX_PROJECTION_ELEMENTS: usize = 16 * 1024 * 1024;
const MAX_STEERING_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HookStage {
    PreModule,
    PostModule,
    ResidualStream,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TokenSelection {
    All,
    Last,
    AbsoluteRange {
        start_inclusive: u64,
        end_exclusive: u64,
    },
    GeneratedRange {
        start_offset: u64,
        maximum_tokens: u64,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SteeringNormalization {
    None,
    UnitL2,
    RootMeanSquare,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ActivationHookProjection {
    pub hook_id: String,
    pub module_path: String,
    pub source_tensor_id: TensorId,
    pub stage: HookStage,
    pub activation_width: usize,
    pub token_selection: TokenSelection,
    pub normalization: SteeringNormalization,
    pub projection_rows: Vec<Vec<f64>>,
    pub gain: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ActivationSteeringLayout {
    pub schema: String,
    pub receiver_layout_sha256: Sha256Digest,
    pub hooks: Vec<ActivationHookProjection>,
    pub manifest_sha256: Sha256Digest,
}

impl ActivationSteeringLayout {
    pub fn create(
        receiver_layout: &ReceiverMaterializationLayout,
        hooks: Vec<ActivationHookProjection>,
    ) -> BrainResult<Self> {
        let mut result = Self {
            schema: "cerebro.tidex.activation_steering_layout/v1".into(),
            receiver_layout_sha256: receiver_layout.manifest_sha256.clone(),
            hooks,
            manifest_sha256: Sha256Digest::zero(),
        };
        result.validate_geometry(receiver_layout)?;
        result.manifest_sha256 = result.calculate_digest()?;
        Ok(result)
    }

    fn validate_geometry(
        &self,
        receiver_layout: &ReceiverMaterializationLayout,
    ) -> BrainResult<()> {
        receiver_layout.geometry.validate()?;
        if self.schema != "cerebro.tidex.activation_steering_layout/v1"
            || self.receiver_layout_sha256 != receiver_layout.manifest_sha256
            || self.hooks.is_empty()
            || self.hooks.len() > MAX_HOOKS
        {
            return Err(BrainError::Invalid(
                "activation_steering_layout_invalid".into(),
            ));
        }
        let mut previous = None::<&str>;
        let mut projection_elements = 0usize;
        for hook in &self.hooks {
            if hook.hook_id.trim().is_empty()
                || hook.hook_id != hook.hook_id.trim()
                || hook.hook_id.len() > 4096
                || hook.module_path.trim().is_empty()
                || hook.module_path != hook.module_path.trim()
                || previous.is_some_and(|prior| prior >= hook.hook_id.as_str())
                || hook.activation_width == 0
                || hook.activation_width > MAX_ACTIVATION_WIDTH
                || hook.projection_rows.len() != hook.activation_width
                || !hook.gain.is_finite()
            {
                return Err(BrainError::Invalid(
                    "activation_hook_contract_invalid".into(),
                ));
            }
            let block = receiver_layout
                .geometry
                .layout
                .blocks
                .iter()
                .find(|block| block.name == hook.source_tensor_id.as_str())
                .ok_or_else(|| BrainError::Invalid("activation_hook_source_unknown".into()))?;
            if hook
                .projection_rows
                .iter()
                .any(|row| row.len() != block.count || row.iter().any(|v| !v.is_finite()))
            {
                return Err(BrainError::Invalid(
                    "activation_projection_shape_invalid".into(),
                ));
            }
            projection_elements =
                projection_elements
                    .checked_add(hook.activation_width.checked_mul(block.count).ok_or_else(
                        || BrainError::Invalid("activation_projection_overflow".into()),
                    )?)
                    .ok_or_else(|| BrainError::Invalid("activation_projection_overflow".into()))?;
            if projection_elements > MAX_PROJECTION_ELEMENTS {
                return Err(BrainError::Invalid("activation_projection_limit".into()));
            }
            match hook.token_selection {
                TokenSelection::AbsoluteRange {
                    start_inclusive,
                    end_exclusive,
                } if start_inclusive >= end_exclusive => {
                    return Err(BrainError::Invalid("activation_token_range_invalid".into()))
                }
                TokenSelection::GeneratedRange {
                    maximum_tokens: 0, ..
                } => return Err(BrainError::Invalid("activation_token_range_invalid".into())),
                _ => {}
            }
            previous = Some(&hook.hook_id);
        }
        Ok(())
    }

    pub fn validate_for(&self, receiver_layout: &ReceiverMaterializationLayout) -> BrainResult<()> {
        self.validate_geometry(receiver_layout)?;
        if self.manifest_sha256 != self.calculate_digest()? {
            return Err(BrainError::Integrity(
                "activation_steering_layout_digest_invalid".into(),
            ));
        }
        Ok(())
    }

    fn calculate_digest(&self) -> BrainResult<Sha256Digest> {
        let mut unsigned = self.clone();
        unsigned.manifest_sha256 = Sha256Digest::zero();
        Ok(Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:ACTIVATION-STEERING-LAYOUT:v1\0",
            &serde_json::to_vec(&unsigned)?,
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ActivationSteeringPolicy {
    pub schema: String,
    pub maximum_vector_l2: f64,
    pub maximum_absolute_component: f64,
    pub maximum_gain: f64,
    pub allow_zero_vector: bool,
}

impl ActivationSteeringPolicy {
    fn validate(&self) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.activation_steering_policy/v1"
            || !self.maximum_vector_l2.is_finite()
            || self.maximum_vector_l2 <= 0.0
            || !self.maximum_absolute_component.is_finite()
            || self.maximum_absolute_component <= 0.0
            || !self.maximum_gain.is_finite()
            || self.maximum_gain <= 0.0
        {
            return Err(BrainError::Invalid(
                "activation_steering_policy_invalid".into(),
            ));
        }
        Ok(())
    }
    pub fn digest(&self) -> BrainResult<Sha256Digest> {
        self.validate()?;
        Ok(Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:ACTIVATION-STEERING-POLICY:v1\0",
            &serde_json::to_vec(self)?,
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ShadowActivationIntervention {
    pub hook_id: String,
    pub module_path: String,
    pub stage: HookStage,
    pub token_selection: TokenSelection,
    pub vector: Vec<f64>,
    pub vector_l2: f64,
    pub gain: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ShadowActivationSteeringCandidate {
    pub schema: String,
    pub planning_request_sha256: Sha256Digest,
    pub receiver_layout_sha256: Sha256Digest,
    pub steering_layout_sha256: Sha256Digest,
    pub policy_sha256: Sha256Digest,
    pub target_delta_sha256: Sha256Digest,
    pub interventions: Vec<ShadowActivationIntervention>,
    pub manifest_sha256: Sha256Digest,
}

fn normalize(mut vector: Vec<f64>, normalization: SteeringNormalization) -> BrainResult<Vec<f64>> {
    let magnitude = norm(&vector)?;
    match normalization {
        SteeringNormalization::None => {}
        SteeringNormalization::UnitL2 if magnitude > 0.0 => {
            vector.iter_mut().for_each(|v| *v /= magnitude)
        }
        SteeringNormalization::RootMeanSquare if magnitude > 0.0 => {
            let rms = magnitude / (vector.len() as f64).sqrt();
            vector.iter_mut().for_each(|v| *v /= rms);
        }
        _ => {}
    }
    Ok(vector)
}

fn target_digest(values: &[f64]) -> BrainResult<Sha256Digest> {
    Ok(Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:ACTIVATION-STEERING-TARGET:v1\0",
        &serde_json::to_vec(values)?,
    ))
}

impl ShadowActivationSteeringCandidate {
    fn calculate_digest(&self) -> BrainResult<Sha256Digest> {
        let mut unsigned = self.clone();
        unsigned.manifest_sha256 = Sha256Digest::zero();
        Ok(Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:ACTIVATION-STEERING-CANDIDATE:v1\0",
            &serde_json::to_vec(&unsigned)?,
        ))
    }
    pub fn validate(
        &self,
        request: &UniversalCapabilityPlanningRequest,
        receipt: &UniversalCapabilityShadowPlanReceipt,
        receiver_layout: &ReceiverMaterializationLayout,
        steering_layout: &ActivationSteeringLayout,
        policy: &ActivationSteeringPolicy,
    ) -> BrainResult<()> {
        if self != &build_candidate(request, receipt, receiver_layout, steering_layout, policy)? {
            return Err(BrainError::Integrity(
                "activation_steering_candidate_invalid".into(),
            ));
        }
        Ok(())
    }
}

fn build_candidate(
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    receiver_layout: &ReceiverMaterializationLayout,
    steering_layout: &ActivationSteeringLayout,
    policy: &ActivationSteeringPolicy,
) -> BrainResult<ShadowActivationSteeringCandidate> {
    replay_universal_capability_shadow_plan(request, receipt)?;
    receiver_layout.validate_for(&request.receiver_profile, &request.receiver_snapshot)?;
    steering_layout.validate_for(receiver_layout)?;
    policy.validate()?;
    if receipt.shadow_plan.materialization_plan.strategy
        != MaterializationStrategy::ActivationSteering
    {
        return Err(BrainError::Invalid(
            "activation_steering_strategy_required".into(),
        ));
    }
    let target = &receipt
        .shadow_plan
        .compilation_receipt
        .compilation
        .receiver
        .target_delta;
    let affected = receipt
        .shadow_plan
        .materialization_plan
        .affected_regions
        .iter()
        .collect::<BTreeSet<_>>();
    let mut interventions = Vec::with_capacity(steering_layout.hooks.len());
    for hook in &steering_layout.hooks {
        if !affected.contains(&hook.source_tensor_id) || hook.gain.abs() > policy.maximum_gain {
            return Err(BrainError::Invalid("activation_hook_not_authorized".into()));
        }
        let block = receiver_layout
            .geometry
            .layout
            .blocks
            .iter()
            .find(|b| b.name == hook.source_tensor_id.as_str())
            .ok_or_else(|| BrainError::Integrity("activation_source_block_missing".into()))?;
        let start = usize::try_from(block.offset)
            .map_err(|_| BrainError::Invalid("activation_offset_overflow".into()))?;
        let end = start
            .checked_add(block.count)
            .ok_or_else(|| BrainError::Invalid("activation_range_overflow".into()))?;
        let source = target
            .get(start..end)
            .ok_or_else(|| BrainError::Integrity("activation_target_range_invalid".into()))?;
        let raw = hook
            .projection_rows
            .iter()
            .map(|row| row.iter().zip(source).map(|(a, b)| a * b).sum::<f64>())
            .collect::<Vec<_>>();
        let vector = normalize(raw, hook.normalization)?
            .into_iter()
            .map(|v| v * hook.gain)
            .collect::<Vec<_>>();
        let vector_l2 = norm(&vector)?;
        if (!policy.allow_zero_vector && vector_l2 == 0.0)
            || vector_l2 > policy.maximum_vector_l2
            || vector
                .iter()
                .any(|v| !v.is_finite() || v.abs() > policy.maximum_absolute_component)
        {
            return Err(BrainError::Numerical(
                "activation_steering_vector_not_admissible".into(),
            ));
        }
        interventions.push(ShadowActivationIntervention {
            hook_id: hook.hook_id.clone(),
            module_path: hook.module_path.clone(),
            stage: hook.stage,
            token_selection: hook.token_selection.clone(),
            vector,
            vector_l2,
            gain: hook.gain,
        });
    }
    let mut candidate = ShadowActivationSteeringCandidate {
        schema: "cerebro.tidex.shadow_activation_steering_candidate/v1".into(),
        planning_request_sha256: receipt.planning_request_sha256.clone(),
        receiver_layout_sha256: receiver_layout.manifest_sha256.clone(),
        steering_layout_sha256: steering_layout.manifest_sha256.clone(),
        policy_sha256: policy.digest()?,
        target_delta_sha256: target_digest(target)?,
        interventions,
        manifest_sha256: Sha256Digest::zero(),
    };
    candidate.manifest_sha256 = candidate.calculate_digest()?;
    Ok(candidate)
}

pub fn materialize_replayed_activation_steering_shadow(
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    receiver_layout: &ReceiverMaterializationLayout,
    steering_layout: &ActivationSteeringLayout,
    policy: &ActivationSteeringPolicy,
) -> BrainResult<ShadowActivationSteeringCandidate> {
    let candidate = build_candidate(request, receipt, receiver_layout, steering_layout, policy)?;
    candidate.validate(request, receipt, receiver_layout, steering_layout, policy)?;
    Ok(candidate)
}

pub fn persist_activation_steering_shadow(
    roots: &LabRoots,
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    receiver_layout: &ReceiverMaterializationLayout,
    steering_layout: &ActivationSteeringLayout,
    policy: &ActivationSteeringPolicy,
    candidate: &ShadowActivationSteeringCandidate,
) -> BrainResult<PrivateFileReference> {
    candidate.validate(request, receipt, receiver_layout, steering_layout, policy)?;
    let bytes = serde_json::to_vec(candidate)?;
    if u64::try_from(bytes.len())
        .ok()
        .filter(|n| *n <= MAX_STEERING_BYTES)
        .is_none()
    {
        return Err(BrainError::Invalid(
            "activation_steering_candidate_too_large".into(),
        ));
    }
    let path = roots
        .artifact_root()
        .join("activation-steering-candidates")
        .join(format!("{}.json", candidate.manifest_sha256));
    let sha256 = write_or_verify_immutable(roots.lab_root(), &path, &bytes)?;
    Ok(PrivateFileReference::new(path, sha256))
}

pub fn load_activation_steering_shadow(
    roots: &LabRoots,
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    receiver_layout: &ReceiverMaterializationLayout,
    steering_layout: &ActivationSteeringLayout,
    policy: &ActivationSteeringPolicy,
    reference: &PrivateFileReference,
) -> BrainResult<ShadowActivationSteeringCandidate> {
    let bytes = reference.read_verified_bounded(roots.lab_root(), MAX_STEERING_BYTES)?;
    let candidate: ShadowActivationSteeringCandidate = serde_json::from_slice(&bytes)?;
    let expected = roots
        .artifact_root()
        .join("activation-steering-candidates")
        .join(format!("{}.json", candidate.manifest_sha256));
    if reference.path != expected {
        return Err(BrainError::Integrity(
            "activation_steering_candidate_path_invalid".into(),
        ));
    }
    candidate.validate(request, receipt, receiver_layout, steering_layout, policy)?;
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_is_exact_and_finite() {
        let unit = normalize(vec![3.0, 4.0], SteeringNormalization::UnitL2).unwrap();
        assert!((norm(&unit).unwrap() - 1.0).abs() < 1e-12);
        let rms = normalize(vec![3.0, 4.0], SteeringNormalization::RootMeanSquare).unwrap();
        assert!((norm(&rms).unwrap() - 2.0_f64.sqrt()).abs() < 1e-12);
    }
}
