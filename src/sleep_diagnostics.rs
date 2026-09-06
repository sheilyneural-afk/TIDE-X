use crate::contracts::{SkillBank, SkillField};
use crate::error::BrainResult;
use crate::identity::SkillId;
use crate::linalg::cosine;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ConsolidationEventKind {
    Born,
    Merge,
    Retained,
    Faded,
    Split,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConsolidationEvent {
    pub event: ConsolidationEventKind,
    pub prior_skill_ids: Vec<SkillId>,
    pub reconstructed_skill_ids: Vec<SkillId>,
    pub strongest_similarity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SleepConsolidationDiagnostics {
    pub schema: String,
    pub prior_skill_count: usize,
    pub reconstructed_skill_count: usize,
    pub retained: usize,
    pub merged: usize,
    pub split: usize,
    pub born: usize,
    pub faded: usize,
    pub events: Vec<ConsolidationEvent>,
}

pub fn diagnose_consolidation(
    prior: &SkillBank,
    reconstructed: &[SkillField],
    similarity_threshold: f64,
) -> BrainResult<SleepConsolidationDiagnostics> {
    let mut matches_for_new = Vec::with_capacity(reconstructed.len());
    for new_field in reconstructed {
        let mut matches = Vec::new();
        for (index, old) in prior.fields.iter().enumerate() {
            let similarity = cosine(&old.direction, &new_field.direction)?.abs();
            if similarity >= similarity_threshold {
                matches.push((index, similarity));
            }
        }
        matches_for_new.push(matches);
    }
    let mut matches_for_old = Vec::with_capacity(prior.fields.len());
    for old in &prior.fields {
        let mut matches = Vec::new();
        for (index, new_field) in reconstructed.iter().enumerate() {
            let similarity = cosine(&old.direction, &new_field.direction)?.abs();
            if similarity >= similarity_threshold {
                matches.push((index, similarity));
            }
        }
        matches_for_old.push(matches);
    }

    let mut events = Vec::new();
    let mut retained = 0;
    let mut merged = 0;
    let mut split = 0;
    let mut born = 0;
    let mut faded = 0;
    for (new_index, matches) in matches_for_new.iter().enumerate() {
        if matches.is_empty() {
            born += 1;
            events.push(ConsolidationEvent {
                event: ConsolidationEventKind::Born,
                prior_skill_ids: vec![],
                reconstructed_skill_ids: vec![reconstructed[new_index].skill_id.clone()],
                strongest_similarity: 0.0,
            });
        } else if matches.len() > 1 {
            merged += 1;
            events.push(ConsolidationEvent {
                event: ConsolidationEventKind::Merge,
                prior_skill_ids: matches
                    .iter()
                    .map(|(index, _)| prior.fields[*index].skill_id.clone())
                    .collect(),
                reconstructed_skill_ids: vec![reconstructed[new_index].skill_id.clone()],
                strongest_similarity: matches.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max),
            });
        } else {
            let old_index = matches[0].0;
            if matches_for_old[old_index].len() > 1 {
                // Count the split once at the old-field pass below.
            } else {
                retained += 1;
                events.push(ConsolidationEvent {
                    event: ConsolidationEventKind::Retained,
                    prior_skill_ids: vec![prior.fields[old_index].skill_id.clone()],
                    reconstructed_skill_ids: vec![reconstructed[new_index].skill_id.clone()],
                    strongest_similarity: matches[0].1,
                });
            }
        }
    }
    for (old_index, matches) in matches_for_old.iter().enumerate() {
        if matches.is_empty() {
            faded += 1;
            events.push(ConsolidationEvent {
                event: ConsolidationEventKind::Faded,
                prior_skill_ids: vec![prior.fields[old_index].skill_id.clone()],
                reconstructed_skill_ids: vec![],
                strongest_similarity: 0.0,
            });
        } else if matches.len() > 1 {
            split += 1;
            events.push(ConsolidationEvent {
                event: ConsolidationEventKind::Split,
                prior_skill_ids: vec![prior.fields[old_index].skill_id.clone()],
                reconstructed_skill_ids: matches
                    .iter()
                    .map(|(index, _)| reconstructed[*index].skill_id.clone())
                    .collect(),
                strongest_similarity: matches.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max),
            });
        }
    }
    Ok(SleepConsolidationDiagnostics {
        schema: "cerebro.tidex.sleep_consolidation_diagnostics/v1".into(),
        prior_skill_count: prior.fields.len(),
        reconstructed_skill_count: reconstructed.len(),
        retained,
        merged,
        split,
        born,
        faded,
        events,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn field(id: &str, direction: Vec<f64>) -> SkillField {
        SkillField {
            skill_id: SkillId::parse(id).unwrap(),
            reconstruction_id: Default::default(),
            lineage_id: Default::default(),
            generation_created: 1,
            direction,
            structured_geometry: None,
            dense_materialization: None,
            parameter_layout_sha256: None,
            representation_signature: Vec::new(),
            singular_value: 1.0,
            explained_variance: 1.0,
            persistence: 1.0,
            coherence: 1.0,
            uncertainty: 0.0,
            evidence_support_digests: Vec::new(),
            support: 1,
            functional_signature: vec![],
            parent_skill_ids: vec![],
        }
    }
    #[test]
    fn diagnostics_detect_split_and_birth() {
        let prior = SkillBank {
            generation: 1,
            fields: vec![field("old", vec![1.0, 0.0, 0.0])],
        };
        let new = vec![
            field("a", vec![1.0, 0.01, 0.0]),
            field("b", vec![1.0, -0.01, 0.0]),
            field("c", vec![0.0, 0.0, 1.0]),
        ];
        let d = diagnose_consolidation(&prior, &new, 0.99).unwrap();
        assert_eq!(d.split, 1);
        assert_eq!(d.born, 1);
    }
}
