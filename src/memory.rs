use crate::contracts::{DeltaObservation, SkillField};
use crate::digest::{ObservationRecordDigest, ProvenanceDigest, Sha256Digest};
use crate::error::{BrainError, BrainResult};
use crate::identity::{ObservationId, SkillId};
use crate::validation::{source_support_indices, validate_reliability};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceMemoryEntry {
    pub observation_id: ObservationId,
    pub observation_digest: ObservationRecordDigest,
    pub provenance_digest: ProvenanceDigest,
    pub independence_group: String,
    pub generation: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EpisodicSemanticEntry {
    pub episode_id: String,
    pub observation_ids: Vec<ObservationId>,
    pub confounder_names: Vec<String>,
    pub mean_functional_response: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FunctionalMemoryEntry {
    pub skill_id: SkillId,
    pub functional_signature: Vec<f64>,
    pub coherence: f64,
    pub persistence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ParametricMemoryEntry {
    pub skill_id: SkillId,
    pub generation_created: u64,
    pub support: usize,
    pub uncertainty: f64,
    pub promoted: bool,
    pub source_observation_ids: Vec<ObservationId>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LearningGraphRelation {
    #[serde(rename = "supports_parametric_field")]
    SupportsParametricField,
    #[serde(rename = "functional_effect")]
    FunctionalEffect,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LearningGraphEdge {
    pub from: String,
    pub to: String,
    pub relation: LearningGraphRelation,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CausalLearningGraph {
    pub nodes: Vec<String>,
    pub edges: Vec<LearningGraphEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BrainMemorySnapshot {
    pub schema: String,
    pub generation: u64,
    pub evidence: Vec<EvidenceMemoryEntry>,
    pub semantic_episodic: Vec<EpisodicSemanticEntry>,
    pub functional: Vec<FunctionalMemoryEntry>,
    pub parametric: Vec<ParametricMemoryEntry>,
    pub causal_learning_graph: CausalLearningGraph,
}

fn observation_record_digest(
    observation: &DeltaObservation,
) -> BrainResult<ObservationRecordDigest> {
    Ok(ObservationRecordDigest::from(Sha256Digest::digest_bytes(
        &serde_json::to_vec(observation)?,
    )))
}

pub fn build_memory_snapshot(
    observations: &[DeltaObservation],
    generation: u64,
    promoted: bool,
    fields: &[SkillField],
    source_mixtures: &[Vec<f64>],
) -> BrainResult<BrainMemorySnapshot> {
    if observations.is_empty()
        || fields.len() != source_mixtures.len()
        || source_mixtures
            .iter()
            .any(|row| row.len() != observations.len())
        || observations.iter().any(|observation| {
            observation.independence_group.trim().is_empty()
                || validate_reliability(observation.reliability, "memory").is_err()
                || observation
                    .functional_response
                    .iter()
                    .any(|value| !value.is_finite())
        })
        || fields.iter().any(|field| {
            field.skill_id.trim().is_empty()
                || !field.coherence.is_finite()
                || !(0.0..=1.0).contains(&field.coherence)
                || !field.persistence.is_finite()
                || !(0.0..=1.0).contains(&field.persistence)
                || !field.uncertainty.is_finite()
                || field.uncertainty < 0.0
                || field
                    .functional_signature
                    .iter()
                    .any(|value| !value.is_finite())
        })
    {
        return Err(BrainError::Invalid("memory_snapshot_input_shape".into()));
    }
    let evidence = observations
        .iter()
        .map(|observation| {
            Ok(EvidenceMemoryEntry {
                observation_id: observation.observation_id.clone(),
                observation_digest: observation_record_digest(observation)?,
                provenance_digest: observation.provenance_digest.clone(),
                independence_group: observation.independence_group.clone(),
                generation: observation.generation,
            })
        })
        .collect::<BrainResult<Vec<_>>>()?;

    let mut by_episode = BTreeMap::<String, Vec<usize>>::new();
    for (index, observation) in observations.iter().enumerate() {
        by_episode
            .entry(observation.independence_group.clone())
            .or_default()
            .push(index);
    }
    let mut semantic_episodic = Vec::with_capacity(by_episode.len());
    for (episode_id, indices) in by_episode {
        let functional_dim = observations[indices[0]].functional_response.len();
        if indices
            .iter()
            .any(|index| observations[*index].functional_response.len() != functional_dim)
        {
            return Err(BrainError::Invalid(
                "memory_functional_dimension_mismatch".into(),
            ));
        }
        let mut mean = vec![0.0; functional_dim];
        let mut total_weight = 0.0;
        let mut confounders = BTreeSet::new();
        for &index in &indices {
            let weight = observations[index].reliability;
            total_weight += weight;
            for (output, value) in mean.iter_mut().enumerate().take(functional_dim) {
                *value += weight * observations[index].functional_response[output];
            }
            for confounder in &observations[index].confounders {
                confounders.insert(confounder.name.clone());
            }
        }
        for value in &mut mean {
            *value /= total_weight;
        }
        semantic_episodic.push(EpisodicSemanticEntry {
            episode_id,
            observation_ids: indices
                .iter()
                .map(|index| observations[*index].observation_id.clone())
                .collect(),
            confounder_names: confounders.into_iter().collect(),
            mean_functional_response: mean,
        });
    }

    let functional = fields
        .iter()
        .map(|field| FunctionalMemoryEntry {
            skill_id: field.skill_id.clone(),
            functional_signature: field.functional_signature.clone(),
            coherence: field.coherence,
            persistence: field.persistence,
        })
        .collect::<Vec<_>>();
    let mut parametric = Vec::with_capacity(fields.len());
    let mut nodes = BTreeSet::<String>::new();
    let mut edges = Vec::<LearningGraphEdge>::new();
    for observation in observations {
        nodes.insert(format!("evidence:{}", observation.observation_id));
    }
    for (field_index, field) in fields.iter().enumerate() {
        let skill_node = format!("skill:{}", field.skill_id);
        nodes.insert(skill_node.clone());
        let support = source_support_indices(&source_mixtures[field_index])?;
        let source_observation_ids = support
            .iter()
            .map(|index| observations[*index].observation_id.clone())
            .collect::<Vec<_>>();
        for index in support {
            edges.push(LearningGraphEdge {
                from: format!("evidence:{}", observations[index].observation_id),
                to: skill_node.clone(),
                relation: LearningGraphRelation::SupportsParametricField,
                weight: source_mixtures[field_index][index],
            });
        }
        for (output, weight) in field.functional_signature.iter().enumerate() {
            let function_node = format!("function:{output}");
            nodes.insert(function_node.clone());
            edges.push(LearningGraphEdge {
                from: skill_node.clone(),
                to: function_node,
                relation: LearningGraphRelation::FunctionalEffect,
                weight: *weight,
            });
        }
        parametric.push(ParametricMemoryEntry {
            skill_id: field.skill_id.clone(),
            generation_created: field.generation_created,
            support: field.support,
            uncertainty: field.uncertainty,
            promoted,
            source_observation_ids,
        });
    }
    Ok(BrainMemorySnapshot {
        schema: "cerebro.tidex.memory_snapshot/v1".into(),
        generation,
        evidence,
        semantic_episodic,
        functional,
        parametric,
        causal_learning_graph: CausalLearningGraph {
            nodes: nodes.into_iter().collect(),
            edges,
        },
    })
}

pub fn memory_artifact_path(root: &Path, digest: &str) -> std::path::PathBuf {
    root.join("state")
        .join("memory")
        .join("by-sha")
        .join(format!("{digest}.json"))
}

/// Deprecated external persistence entrypoint.
///
/// Memory is authority state: it can only be materialized as part of the
/// receipt-bound engine transaction, where its digest is bound to the report,
/// commit and ledger event.  This compatibility symbol remains solely so
/// quarantined archival code cannot regain a silent writer; it always fails
/// before inspecting or mutating the supplied root.
pub fn persist_memory_snapshot(
    _root: &Path,
    _snapshot: &BrainMemorySnapshot,
) -> BrainResult<String> {
    Err(BrainError::Integrity(
        "memory_persistence_requires_receipt_bound_engine_cycle".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{ConfounderValue, SkillField};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_lineage(index: usize, group: usize) -> crate::contracts::ExperimentLineage {
        crate::contracts::ExperimentLineage {
            run_id: format!("run-{index}"),
            replicate_id: format!("g{group}"),
            randomization_id: format!("rand-{group}"),
            dataset_split_digest: format!("{:064x}", 1),
            initial_checkpoint_digest: format!("{:064x}", 2),
            optimizer_config_digest: format!("{:064x}", 3),
            template_config_digest: format!("{:064x}", 4),
        }
    }

    #[test]
    fn memory_snapshot_links_evidence_to_functional_and_parametric_layers() {
        let observations = (0..3)
            .map(|i| DeltaObservation {
                observation_id: ObservationId::parse(format!("o{i}")).unwrap(),
                from_checkpoint: "a".into(),
                to_checkpoint: format!("b{i}"),
                generation: i as u64 + 1,
                delta: vec![1.0, 0.0],
                functional_response: vec![i as f64],
                confounders: vec![ConfounderValue {
                    name: "seed".into(),
                    value: i as f64,
                }],
                reliability: 1.0,
                independence_group: format!("g{i}"),
                experiment_lineage: test_lineage(i, i),
                dense_artifact: None,
                parameter_layout_sha256: None,
                representation_artifact: None,
                representation_protocol_sha256: None,
                provenance_digest: ProvenanceDigest::from(
                    Sha256Digest::parse(format!("{i:064x}")).unwrap(),
                ),
            })
            .collect::<Vec<_>>();
        let field = SkillField {
            skill_id: SkillId::parse("s").unwrap(),
            reconstruction_id: Default::default(),
            lineage_id: Default::default(),
            generation_created: 1,
            direction: vec![1.0, 0.0],
            structured_geometry: None,
            dense_materialization: None,
            parameter_layout_sha256: None,
            representation_signature: Vec::new(),
            singular_value: 1.0,
            explained_variance: 1.0,
            persistence: 1.0,
            coherence: 1.0,
            uncertainty: 0.1,
            evidence_support_digests: Vec::new(),
            support: 3,
            functional_signature: vec![2.0],
            parent_skill_ids: vec![],
        };
        let snapshot =
            build_memory_snapshot(&observations, 3, true, &[field], &[vec![1.0 / 3.0; 3]]).unwrap();
        assert_eq!(snapshot.evidence.len(), 3);
        assert_eq!(snapshot.semantic_episodic.len(), 3);
        assert_eq!(snapshot.functional.len(), 1);
        assert_eq!(snapshot.parametric.len(), 1);
        assert_eq!(
            snapshot
                .causal_learning_graph
                .edges
                .iter()
                .filter(|edge| edge.relation == LearningGraphRelation::SupportsParametricField)
                .count(),
            3
        );

        let mut invalid = observations;
        invalid[0].reliability = 0.0;
        assert!(build_memory_snapshot(&invalid, 3, true, &[], &[]).is_err());
    }

    #[test]
    fn public_memory_persistence_fails_closed_without_creating_state() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cerebro-memory-public-persistence-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let snapshot = BrainMemorySnapshot {
            schema: "cerebro.tidex.memory_snapshot/v1".into(),
            generation: 0,
            evidence: Vec::new(),
            semantic_episodic: Vec::new(),
            functional: Vec::new(),
            parametric: Vec::new(),
            causal_learning_graph: CausalLearningGraph {
                nodes: Vec::new(),
                edges: Vec::new(),
            },
        };

        assert!(persist_memory_snapshot(&root, &snapshot).is_err());
        assert!(!root.join("state").exists());
        fs::remove_dir_all(root).unwrap();
    }
}
