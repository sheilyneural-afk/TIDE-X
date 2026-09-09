//! Authenticated receiver snapshot bindings.
//!
//! Architecture adapters may inspect a model format, but this core only
//! accepts their resulting exact artifact commitments.  It never treats a
//! human-readable model name as receiver identity.

use crate::block_tomography::ParameterLayoutArtifact;
use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::receiver_profile::ReceiverProfile;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverSnapshotBinding {
    pub schema: String,
    pub receiver_profile_sha256: Sha256Digest,
    pub model_snapshot_sha256: Sha256Digest,
    pub configuration_sha256: Sha256Digest,
    pub tokenizer_sha256: Sha256Digest,
    pub parameter_layout_sha256: Sha256Digest,
    pub manifest_sha256: Sha256Digest,
}

impl ReceiverSnapshotBinding {
    pub fn manifest_digest(&self) -> &Sha256Digest {
        &self.manifest_sha256
    }
    pub fn create(
        profile: &ReceiverProfile,
        model_snapshot_sha256: Sha256Digest,
        configuration_sha256: Sha256Digest,
        tokenizer_sha256: Sha256Digest,
        parameter_layout_sha256: Sha256Digest,
    ) -> BrainResult<Self> {
        profile.validate()?;
        let receiver_profile_sha256 = profile.digest()?;
        let mut binding = Self {
            schema: "cerebro.tidex.receiver_snapshot_binding/v1".into(),
            receiver_profile_sha256,
            model_snapshot_sha256,
            configuration_sha256,
            tokenizer_sha256,
            parameter_layout_sha256,
            manifest_sha256: Sha256Digest::zero(),
        };
        binding.manifest_sha256 = binding.calculate_digest()?;
        binding.validate_for(profile)?;
        Ok(binding)
    }

    pub fn validate_for(&self, profile: &ReceiverProfile) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.receiver_snapshot_binding/v1"
            || self.receiver_profile_sha256 != profile.digest()?
            || self.model_snapshot_sha256 == Sha256Digest::zero()
            || self.configuration_sha256 == Sha256Digest::zero()
            || self.tokenizer_sha256 == Sha256Digest::zero()
            || self.parameter_layout_sha256 == Sha256Digest::zero()
            || self.manifest_sha256 != self.calculate_digest()?
        {
            return Err(BrainError::Integrity(
                "receiver_snapshot_binding_invalid".into(),
            ));
        }
        Ok(())
    }

    fn calculate_digest(&self) -> BrainResult<Sha256Digest> {
        #[derive(Serialize)]
        struct Projection<'a> {
            schema: &'a str,
            receiver_profile_sha256: &'a Sha256Digest,
            model_snapshot_sha256: &'a Sha256Digest,
            configuration_sha256: &'a Sha256Digest,
            tokenizer_sha256: &'a Sha256Digest,
            parameter_layout_sha256: &'a Sha256Digest,
        }
        Ok(Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:RECEIVER-SNAPSHOT-BINDING:v1\0",
            &serde_json::to_vec(&Projection {
                schema: &self.schema,
                receiver_profile_sha256: &self.receiver_profile_sha256,
                model_snapshot_sha256: &self.model_snapshot_sha256,
                configuration_sha256: &self.configuration_sha256,
                tokenizer_sha256: &self.tokenizer_sha256,
                parameter_layout_sha256: &self.parameter_layout_sha256,
            })?,
        ))
    }
}

/// Verify that the reusable parameter-layout authority describes exactly the
/// receiver regions committed by a snapshot. This is the geometry gate used
/// before any structured shadow backend (dense, sparse, or low-rank) may
/// interpret flat receiver coordinates as tensors.
pub fn validate_receiver_parameter_layout(
    profile: &ReceiverProfile,
    snapshot: &ReceiverSnapshotBinding,
    artifact: &ParameterLayoutArtifact,
) -> BrainResult<()> {
    snapshot.validate_for(profile)?;
    artifact.validate()?;
    if artifact.parameter_layout_sha256.as_digest() != &snapshot.parameter_layout_sha256
        || artifact.total_parameter_count != profile.parameter_dimension
        || artifact.layout.blocks.len() != profile.regions.len()
    {
        return Err(BrainError::Integrity(
            "receiver_parameter_layout_binding_mismatch".into(),
        ));
    }
    for (block, region) in artifact.layout.blocks.iter().zip(&profile.regions) {
        if block.name != region.tensor_id.as_str()
            || u64::try_from(block.count)
                .map_err(|_| BrainError::Invalid("receiver_layout_count_overflow".into()))?
                != region.parameter_count
        {
            return Err(BrainError::Integrity(
                "receiver_parameter_layout_region_mismatch".into(),
            ));
        }
    }
    Ok(())
}
