//! Receiver-specific physical layout metadata above the reusable flat geometry.
//!
//! Capability identity never depends on this module. These contracts describe
//! one receiver's machine-level tensor encoding so materialization backends can
//! fail closed instead of guessing dtype, quantization, sharding, or tied weights.

use crate::block_tomography::ParameterLayoutArtifact;
use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::identity::TensorId;
use crate::receiver_profile::ReceiverProfile;
use crate::receiver_profiler::ReceiverSnapshotBinding;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const MAX_ALIASES: usize = 1_048_576;
const MAX_PARTITIONS_PER_TENSOR: usize = 1_048_576;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FloatingScalarType {
    Float64,
    Float32,
    Bfloat16,
    Float16,
    Float8E4m3,
    Float8E5m2,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReceiverScalarEncoding {
    Floating {
        scalar_type: FloatingScalarType,
    },
    Quantized {
        bits: u8,
        group_size: u64,
        scale_count: u64,
        symmetric: bool,
    },
}

impl ReceiverScalarEncoding {
    fn validate(&self, element_count: u64) -> BrainResult<()> {
        match self {
            Self::Floating { .. } => Ok(()),
            Self::Quantized {
                bits,
                group_size,
                scale_count,
                ..
            } => {
                let expected_scales = element_count
                    .checked_add(group_size.saturating_sub(1))
                    .and_then(|value| value.checked_div(*group_size));
                if !(1..=16).contains(bits)
                    || *group_size == 0
                    || *scale_count == 0
                    || expected_scales != Some(*scale_count)
                {
                    return Err(BrainError::Invalid(
                        "receiver_quantization_contract_invalid".into(),
                    ));
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverShardPartition {
    pub ordinal: u32,
    pub start: u64,
    pub length: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReceiverTensorPartitioning {
    Replicated,
    Sharded {
        axis: u32,
        partitions: Vec<ReceiverShardPartition>,
    },
}

impl ReceiverTensorPartitioning {
    fn validate(&self, shape: &[usize]) -> BrainResult<()> {
        if let Self::Sharded { axis, partitions } = self {
            let axis = usize::try_from(*axis)
                .map_err(|_| BrainError::Invalid("receiver_shard_axis_overflow".into()))?;
            if axis >= shape.len()
                || partitions.is_empty()
                || partitions.len() > MAX_PARTITIONS_PER_TENSOR
            {
                return Err(BrainError::Invalid(
                    "receiver_shard_contract_invalid".into(),
                ));
            }
            let mut expected_start = 0u64;
            for (index, partition) in partitions.iter().enumerate() {
                if usize::try_from(partition.ordinal).ok() != Some(index)
                    || partition.length == 0
                    || partition.start != expected_start
                {
                    return Err(BrainError::Invalid(
                        "receiver_shard_partition_invalid".into(),
                    ));
                }
                expected_start = expected_start
                    .checked_add(partition.length)
                    .ok_or_else(|| BrainError::Invalid("receiver_shard_range_overflow".into()))?;
            }
            if usize::try_from(expected_start).ok() != Some(shape[axis]) {
                return Err(BrainError::Invalid(
                    "receiver_shard_coverage_invalid".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverTensorPhysicalSpec {
    pub tensor_id: TensorId,
    pub encoding: ReceiverScalarEncoding,
    pub partitioning: ReceiverTensorPartitioning,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverTensorAlias {
    pub alias_tensor_id: TensorId,
    pub canonical_tensor_id: TensorId,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverMaterializationLayout {
    pub schema: String,
    pub geometry: ParameterLayoutArtifact,
    pub physical_tensors: Vec<ReceiverTensorPhysicalSpec>,
    pub aliases: Vec<ReceiverTensorAlias>,
    pub manifest_sha256: Sha256Digest,
}

impl ReceiverMaterializationLayout {
    pub fn create(
        profile: &ReceiverProfile,
        geometry: ParameterLayoutArtifact,
        physical_tensors: Vec<ReceiverTensorPhysicalSpec>,
        aliases: Vec<ReceiverTensorAlias>,
    ) -> BrainResult<Self> {
        let mut layout = Self {
            schema: "cerebro.tidex.receiver_materialization_layout/v1".into(),
            geometry,
            physical_tensors,
            aliases,
            manifest_sha256: Sha256Digest::zero(),
        };
        layout.validate_geometry(profile)?;
        layout.manifest_sha256 = layout.calculate_digest()?;
        Ok(layout)
    }

    pub fn validate_for(
        &self,
        profile: &ReceiverProfile,
        snapshot: &ReceiverSnapshotBinding,
    ) -> BrainResult<()> {
        self.validate_geometry(profile)?;
        snapshot.validate_for(profile)?;
        if self.manifest_sha256 != self.calculate_digest()?
            || snapshot.parameter_layout_sha256 != self.manifest_sha256
        {
            return Err(BrainError::Integrity(
                "receiver_materialization_layout_binding_invalid".into(),
            ));
        }
        Ok(())
    }

    fn validate_geometry(&self, profile: &ReceiverProfile) -> BrainResult<()> {
        profile.validate()?;
        self.geometry.validate()?;
        if self.schema != "cerebro.tidex.receiver_materialization_layout/v1"
            || self.geometry.total_parameter_count != profile.parameter_dimension
            || self.geometry.layout.blocks.len() != profile.regions.len()
            || self.physical_tensors.len() != profile.regions.len()
            || self.aliases.len() > MAX_ALIASES
        {
            return Err(BrainError::Invalid(
                "receiver_materialization_layout_invalid".into(),
            ));
        }
        let canonical = profile
            .regions
            .iter()
            .map(|region| &region.tensor_id)
            .collect::<BTreeSet<_>>();
        for ((block, physical), region) in self
            .geometry
            .layout
            .blocks
            .iter()
            .zip(&self.physical_tensors)
            .zip(&profile.regions)
        {
            if block.name != region.tensor_id.as_str()
                || physical.tensor_id != region.tensor_id
                || u64::try_from(block.count).ok() != Some(region.parameter_count)
            {
                return Err(BrainError::Integrity(
                    "receiver_materialization_tensor_mismatch".into(),
                ));
            }
            physical.encoding.validate(region.parameter_count)?;
            physical.partitioning.validate(&block.shape)?;
        }
        let mut previous = None::<&TensorId>;
        let mut seen_aliases = BTreeSet::new();
        for alias in &self.aliases {
            if previous.is_some_and(|prior| prior >= &alias.alias_tensor_id)
                || !seen_aliases.insert(&alias.alias_tensor_id)
                || canonical.contains(&alias.alias_tensor_id)
                || !canonical.contains(&alias.canonical_tensor_id)
                || alias.alias_tensor_id == alias.canonical_tensor_id
            {
                return Err(BrainError::Invalid("receiver_tensor_alias_invalid".into()));
            }
            previous = Some(&alias.alias_tensor_id);
        }
        Ok(())
    }

    fn calculate_digest(&self) -> BrainResult<Sha256Digest> {
        let mut unsigned = self.clone();
        unsigned.manifest_sha256 = Sha256Digest::zero();
        Ok(Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:RECEIVER-MATERIALIZATION-LAYOUT:v1\0",
            &serde_json::to_vec(&unsigned)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_tomography::{BlockShapeSpec, ParameterBlockLayout};
    use crate::identity::{ArchitectureId, ModelId};
    use crate::receiver_profile::{
        CapabilityModality, MaterializationStrategy, ReceiverArchitecture, ReceiverRegion,
    };

    fn fixture() -> (ReceiverProfile, ReceiverMaterializationLayout) {
        let embedding = TensorId::parse("embeddings.weight").unwrap();
        let projection = TensorId::parse("layers.0.attn.q_proj.weight").unwrap();
        let profile = ReceiverProfile {
            schema: "cerebro.tidex.receiver_profile/v1".into(),
            model_id: ModelId::parse("receiver.v1").unwrap(),
            architecture_id: ArchitectureId::parse("transformer.v1").unwrap(),
            architecture: ReceiverArchitecture::Transformer,
            modalities: BTreeSet::from([CapabilityModality::Text]),
            supports_persistent_state: false,
            parameter_dimension: 12,
            regions: vec![
                ReceiverRegion {
                    tensor_id: embedding.clone(),
                    parameter_count: 8,
                    supported_strategies: BTreeSet::from([MaterializationStrategy::DenseDelta]),
                },
                ReceiverRegion {
                    tensor_id: projection.clone(),
                    parameter_count: 4,
                    supported_strategies: BTreeSet::from([
                        MaterializationStrategy::DenseDelta,
                        MaterializationStrategy::LowRank,
                    ]),
                },
            ],
        };
        let geometry = ParameterLayoutArtifact::new(
            ParameterBlockLayout::from_shapes(&[
                BlockShapeSpec {
                    name: embedding.as_str().into(),
                    shape: vec![4, 2],
                    count: 8,
                },
                BlockShapeSpec {
                    name: projection.as_str().into(),
                    shape: vec![2, 2],
                    count: 4,
                },
            ])
            .unwrap(),
        )
        .unwrap();
        let layout = ReceiverMaterializationLayout::create(
            &profile,
            geometry,
            vec![
                ReceiverTensorPhysicalSpec {
                    tensor_id: embedding.clone(),
                    encoding: ReceiverScalarEncoding::Floating {
                        scalar_type: FloatingScalarType::Bfloat16,
                    },
                    partitioning: ReceiverTensorPartitioning::Sharded {
                        axis: 0,
                        partitions: vec![
                            ReceiverShardPartition {
                                ordinal: 0,
                                start: 0,
                                length: 2,
                            },
                            ReceiverShardPartition {
                                ordinal: 1,
                                start: 2,
                                length: 2,
                            },
                        ],
                    },
                },
                ReceiverTensorPhysicalSpec {
                    tensor_id: projection,
                    encoding: ReceiverScalarEncoding::Quantized {
                        bits: 8,
                        group_size: 4,
                        scale_count: 1,
                        symmetric: true,
                    },
                    partitioning: ReceiverTensorPartitioning::Replicated,
                },
            ],
            vec![ReceiverTensorAlias {
                alias_tensor_id: TensorId::parse("lm_head.weight").unwrap(),
                canonical_tensor_id: embedding,
            }],
        )
        .unwrap();
        (profile, layout)
    }

    #[test]
    fn layout_binds_geometry_encoding_sharding_and_optional_aliases() {
        let (profile, layout) = fixture();
        let snapshot = ReceiverSnapshotBinding::create(
            &profile,
            Sha256Digest::digest_bytes(b"model"),
            Sha256Digest::digest_bytes(b"config"),
            Sha256Digest::digest_bytes(b"tokenizer"),
            layout.manifest_sha256.clone(),
        )
        .unwrap();
        layout.validate_for(&profile, &snapshot).unwrap();

        let mut gap = layout.clone();
        if let ReceiverTensorPartitioning::Sharded { partitions, .. } =
            &mut gap.physical_tensors[0].partitioning
        {
            partitions[1].start = 3;
        }
        assert!(gap.validate_for(&profile, &snapshot).is_err());

        let mut false_alias = layout.clone();
        false_alias.aliases[0].canonical_tensor_id = TensorId::parse("missing.weight").unwrap();
        assert!(false_alias.validate_for(&profile, &snapshot).is_err());

        let mut invalid_quantization = layout.clone();
        invalid_quantization.physical_tensors[1].encoding = ReceiverScalarEncoding::Quantized {
            bits: 8,
            group_size: 3,
            scale_count: 1,
            symmetric: true,
        };
        assert!(invalid_quantization
            .validate_for(&profile, &snapshot)
            .is_err());

        let mut tampered = layout;
        tampered.physical_tensors[1].encoding = ReceiverScalarEncoding::Floating {
            scalar_type: FloatingScalarType::Float32,
        };
        assert!(tampered.validate_for(&profile, &snapshot).is_err());
    }
}
