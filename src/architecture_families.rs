//! Evidence-based architecture and internal module-family fingerprinting.

use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::receiver_profile::ReceiverArchitecture;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    DecoderTransformer,
    EncoderTransformer,
    EncoderDecoderTransformer,
    MixtureOfExperts,
    StateSpace,
    HybridStateSpaceTransformer,
    Multimodal,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ModuleFamily {
    Embedding,
    Attention,
    FeedForward,
    Normalizer,
    MixtureRouter,
    Expert,
    StateSpaceMixer,
    Convolution,
    Recurrent,
    VisionEncoder,
    AudioEncoder,
    OutputHead,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArchitectureFamilyFingerprint {
    pub schema: String,
    pub configuration_sha256: Sha256Digest,
    pub model_type: Option<String>,
    pub declared_architectures: Vec<String>,
    pub model_family: ModelFamily,
    pub receiver_architecture: ReceiverArchitecture,
    pub module_family_counts: BTreeMap<ModuleFamily, u64>,
    pub evidence_tags: BTreeSet<String>,
    pub manifest_sha256: Sha256Digest,
}

impl ArchitectureFamilyFingerprint {
    fn calculate_digest(&self) -> BrainResult<Sha256Digest> {
        let mut unsigned = self.clone();
        unsigned.manifest_sha256 = Sha256Digest::zero();
        Ok(Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:ARCHITECTURE-FAMILY-FINGERPRINT:v1\0",
            &serde_json::to_vec(&unsigned)?,
        ))
    }

    pub fn validate(&self) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.architecture_family_fingerprint/v1"
            || self.configuration_sha256 == Sha256Digest::zero()
            || self.module_family_counts.is_empty()
            || self.module_family_counts.values().any(|count| *count == 0)
            || self.manifest_sha256 != self.calculate_digest()?
        {
            return Err(BrainError::Integrity(
                "architecture_family_fingerprint_invalid".into(),
            ));
        }
        Ok(())
    }
}

fn classify_module(name: &str) -> ModuleFamily {
    let name = name.to_ascii_lowercase();
    if name.contains("embed") {
        ModuleFamily::Embedding
    } else if name.contains("router") || name.contains("expert_gate") {
        ModuleFamily::MixtureRouter
    } else if name.contains("expert") {
        ModuleFamily::Expert
    } else if name.contains("attn") || name.contains("attention") || name.contains("q_proj") {
        ModuleFamily::Attention
    } else if name.contains("mlp")
        || name.contains("ffn")
        || name.contains("up_proj")
        || name.contains("down_proj")
        || name.contains("gate_proj")
    {
        ModuleFamily::FeedForward
    } else if name.contains("norm") {
        ModuleFamily::Normalizer
    } else if name.contains("ssm") || name.contains("state_space") || name.contains("mixer") {
        ModuleFamily::StateSpaceMixer
    } else if name.contains("conv") {
        ModuleFamily::Convolution
    } else if name.contains("rnn") || name.contains("recurrent") {
        ModuleFamily::Recurrent
    } else if name.contains("vision") || name.contains("visual") {
        ModuleFamily::VisionEncoder
    } else if name.contains("audio") || name.contains("speech") {
        ModuleFamily::AudioEncoder
    } else if name.contains("lm_head") || name.contains("output_head") {
        ModuleFamily::OutputHead
    } else {
        ModuleFamily::Other
    }
}

pub fn fingerprint_architecture(
    config_bytes: &[u8],
    tensor_names: &[String],
) -> BrainResult<ArchitectureFamilyFingerprint> {
    if config_bytes.is_empty() || config_bytes.len() > 256 * 1024 * 1024 || tensor_names.is_empty()
    {
        return Err(BrainError::Invalid(
            "architecture_fingerprint_input_invalid".into(),
        ));
    }
    let config: serde_json::Value = serde_json::from_slice(config_bytes)?;
    let object = config
        .as_object()
        .ok_or_else(|| BrainError::Invalid("model_configuration_not_object".into()))?;
    let model_type = object
        .get("model_type")
        .and_then(|value| value.as_str())
        .map(str::to_string);
    let declared_architectures = object
        .get("architectures")
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let joined = format!(
        "{} {} {}",
        model_type.as_deref().unwrap_or(""),
        declared_architectures.join(" "),
        tensor_names.join(" ")
    )
    .to_ascii_lowercase();
    let has_moe = object
        .get("num_local_experts")
        .and_then(|value| value.as_u64())
        .is_some_and(|count| count > 1)
        || joined.contains("mixtral")
        || joined.contains("moe")
        || joined.contains("experts.");
    let has_ssm = joined.contains("mamba")
        || joined.contains("state_space")
        || joined.contains("ssm")
        || joined.contains("selective_scan");
    let has_attention =
        joined.contains("attention") || joined.contains("attn") || joined.contains("q_proj");
    let multimodal = joined.contains("vision")
        || joined.contains("visual")
        || joined.contains("audio")
        || joined.contains("multimodal");
    let encoder_decoder = object
        .get("is_encoder_decoder")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let encoder_only = declared_architectures
        .iter()
        .any(|value| value.to_ascii_lowercase().contains("maskedlm"))
        && !encoder_decoder;
    let model_family = if multimodal {
        ModelFamily::Multimodal
    } else if has_moe {
        ModelFamily::MixtureOfExperts
    } else if has_ssm && has_attention {
        ModelFamily::HybridStateSpaceTransformer
    } else if has_ssm {
        ModelFamily::StateSpace
    } else if encoder_decoder {
        ModelFamily::EncoderDecoderTransformer
    } else if encoder_only {
        ModelFamily::EncoderTransformer
    } else if has_attention {
        ModelFamily::DecoderTransformer
    } else {
        ModelFamily::Unknown
    };
    let receiver_architecture = match model_family {
        ModelFamily::MixtureOfExperts => ReceiverArchitecture::MixtureOfExperts,
        ModelFamily::StateSpace => ReceiverArchitecture::StateSpace,
        ModelFamily::DecoderTransformer
        | ModelFamily::EncoderTransformer
        | ModelFamily::EncoderDecoderTransformer
        | ModelFamily::Multimodal => ReceiverArchitecture::Transformer,
        ModelFamily::HybridStateSpaceTransformer | ModelFamily::Unknown => {
            ReceiverArchitecture::Unknown
        }
    };
    let mut module_family_counts = BTreeMap::new();
    for name in tensor_names {
        *module_family_counts
            .entry(classify_module(name))
            .or_insert(0) += 1;
    }
    let mut evidence_tags = BTreeSet::new();
    if has_attention {
        evidence_tags.insert("attention_tensors".into());
    }
    if has_ssm {
        evidence_tags.insert("state_space_tensors_or_config".into());
    }
    if has_moe {
        evidence_tags.insert("mixture_experts_config_or_tensors".into());
    }
    if multimodal {
        evidence_tags.insert("multimodal_config_or_tensors".into());
    }
    let mut result = ArchitectureFamilyFingerprint {
        schema: "cerebro.tidex.architecture_family_fingerprint/v1".into(),
        configuration_sha256: Sha256Digest::digest_bytes(config_bytes),
        model_type,
        declared_architectures,
        model_family,
        receiver_architecture,
        module_family_counts,
        evidence_tags,
        manifest_sha256: Sha256Digest::zero(),
    };
    result.manifest_sha256 = result.calculate_digest()?;
    result.validate()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_moe_and_internal_module_families() {
        let config = br#"{"model_type":"mixtral","architectures":["MixtralForCausalLM"],"num_local_experts":8}"#;
        let report = fingerprint_architecture(
            config,
            &[
                "model.layers.0.self_attn.q_proj.weight".into(),
                "model.layers.0.block_sparse_moe.experts.0.w1.weight".into(),
            ],
        )
        .unwrap();
        assert_eq!(report.model_family, ModelFamily::MixtureOfExperts);
        assert_eq!(
            report.receiver_architecture,
            ReceiverArchitecture::MixtureOfExperts
        );
    }
}
