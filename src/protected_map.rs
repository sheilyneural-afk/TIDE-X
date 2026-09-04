#![allow(clippy::needless_range_loop)]

use crate::artifact::{create_content_addressed_f64, read_f64_artifact, F64ArtifactRef};
use crate::contracts::{ProtectedCortex, ProtectedDirection};
use crate::error::{BrainError, BrainResult};
use crate::linalg::{dot, norm, symmetric_eigen_jacobi, weighted_row_gram, Matrix};
use crate::validation::{choose_energy_rank, effective_rank_from_spectrum};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SensitivityEvidence {
    pub probe_id: String,
    pub sensitivity: Vec<f64>,
    pub reliability: f64,
    pub causal_damage: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProtectedMapReport {
    pub schema: String,
    pub probe_count: usize,
    pub parameter_dimension: usize,
    pub selected_rank: usize,
    pub effective_rank: f64,
    pub retained_sensitivity_energy: f64,
    pub fisher_trace: f64,
    pub causal_damage_supported_probes: usize,
    pub sensitivity_damage_correlation: Option<f64>,
    pub cortex: ProtectedCortex,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProtectedDirectionArtifact {
    pub probe_id: String,
    pub direction: F64ArtifactRef,
    pub importance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProtectedCortexArtifact {
    pub parameter_importance: F64ArtifactRef,
    pub directions: Vec<ProtectedDirectionArtifact>,
    pub max_damage_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProtectedMapArtifactReport {
    pub schema: String,
    pub probe_count: usize,
    pub parameter_dimension: usize,
    pub selected_rank: usize,
    pub effective_rank: f64,
    pub retained_sensitivity_energy: f64,
    pub fisher_trace: f64,
    pub causal_damage_supported_probes: usize,
    pub sensitivity_damage_correlation: Option<f64>,
    pub cortex: ProtectedCortexArtifact,
}

fn confined_artifact_path(root: &Path, reference: &F64ArtifactRef) -> BrainResult<PathBuf> {
    let canonical_root = root.canonicalize()?;
    let canonical = Path::new(&reference.path).canonicalize()?;
    if !canonical.starts_with(&canonical_root) {
        return Err(BrainError::Integrity(
            "protected_map_artifact_outside_root".into(),
        ));
    }
    Ok(canonical)
}

pub fn persist_protected_map(
    root: &Path,
    report: &ProtectedMapReport,
) -> BrainResult<ProtectedMapArtifactReport> {
    if report.parameter_dimension == 0
        || report.cortex.parameter_importance.len() != report.parameter_dimension
        || report.cortex.directions.len() != report.selected_rank
    {
        return Err(BrainError::Invalid("protected_map_persist_shape".into()));
    }
    let parameter_importance =
        create_content_addressed_f64(root, &report.cortex.parameter_importance)?;
    let mut directions = Vec::with_capacity(report.cortex.directions.len());
    for direction in &report.cortex.directions {
        if direction.direction.len() != report.parameter_dimension {
            return Err(BrainError::Invalid(
                "protected_map_persist_direction_shape".into(),
            ));
        }
        directions.push(ProtectedDirectionArtifact {
            probe_id: direction.probe_id.clone(),
            direction: create_content_addressed_f64(root, &direction.direction)?,
            importance: direction.importance,
        });
    }
    Ok(ProtectedMapArtifactReport {
        schema: "cerebro.tidex.protected_cortex_map_artifact/v2".into(),
        probe_count: report.probe_count,
        parameter_dimension: report.parameter_dimension,
        selected_rank: report.selected_rank,
        effective_rank: report.effective_rank,
        retained_sensitivity_energy: report.retained_sensitivity_energy,
        fisher_trace: report.fisher_trace,
        causal_damage_supported_probes: report.causal_damage_supported_probes,
        sensitivity_damage_correlation: report.sensitivity_damage_correlation,
        cortex: ProtectedCortexArtifact {
            parameter_importance,
            directions,
            max_damage_ratio: report.cortex.max_damage_ratio,
        },
    })
}

pub fn load_protected_cortex(
    root: &Path,
    report: &ProtectedMapArtifactReport,
) -> BrainResult<ProtectedCortex> {
    if report.schema != "cerebro.tidex.protected_cortex_map_artifact/v2"
        || report.parameter_dimension == 0
        || report.selected_rank != report.cortex.directions.len()
    {
        return Err(BrainError::Invalid(
            "protected_map_artifact_contract".into(),
        ));
    }
    let _ = confined_artifact_path(root, &report.cortex.parameter_importance)?;
    let parameter_importance = read_f64_artifact(&report.cortex.parameter_importance)?;
    if parameter_importance.len() != report.parameter_dimension {
        return Err(BrainError::Integrity(
            "protected_map_parameter_importance_count_mismatch".into(),
        ));
    }
    let mut directions = Vec::with_capacity(report.cortex.directions.len());
    for stored in &report.cortex.directions {
        let _ = confined_artifact_path(root, &stored.direction)?;
        let direction = read_f64_artifact(&stored.direction)?;
        if direction.len() != report.parameter_dimension {
            return Err(BrainError::Integrity(
                "protected_map_direction_count_mismatch".into(),
            ));
        }
        directions.push(ProtectedDirection {
            probe_id: stored.probe_id.clone(),
            direction,
            importance: stored.importance,
        });
    }
    Ok(ProtectedCortex {
        parameter_importance,
        directions,
        max_damage_ratio: report.cortex.max_damage_ratio,
    })
}

fn validate_evidence(evidence: &[SensitivityEvidence]) -> BrainResult<usize> {
    if evidence.len() < 2 {
        return Err(BrainError::Invalid(
            "protected_map_requires_two_probes".into(),
        ));
    }
    let dim = evidence[0].sensitivity.len();
    if dim == 0 {
        return Err(BrainError::Invalid("protected_map_empty_direction".into()));
    }
    for (index, probe) in evidence.iter().enumerate() {
        if probe.probe_id.trim().is_empty()
            || probe.sensitivity.len() != dim
            || probe.sensitivity.iter().any(|value| !value.is_finite())
            || !probe.reliability.is_finite()
            || !(0.0..=1.0).contains(&probe.reliability)
            || probe.reliability <= 0.0
            || probe
                .causal_damage
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(BrainError::Invalid(format!(
                "protected_map_probe_invalid:{index}"
            )));
        }
    }
    Ok(dim)
}

fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let lm = left.iter().sum::<f64>() / left.len() as f64;
    let rm = right.iter().sum::<f64>() / right.len() as f64;
    let mut numerator = 0.0;
    let mut ld = 0.0;
    let mut rd = 0.0;
    for (l, r) in left.iter().zip(right) {
        numerator += (l - lm) * (r - rm);
        ld += (l - lm).powi(2);
        rd += (r - rm).powi(2);
    }
    let denom = (ld * rd).sqrt();
    (denom > 1e-18).then_some((numerator / denom).clamp(-1.0, 1.0))
}

pub fn build_protected_cortex_map(
    evidence: &[SensitivityEvidence],
    target_explained_sensitivity: f64,
    max_damage_ratio: f64,
) -> BrainResult<ProtectedMapReport> {
    let dim = validate_evidence(evidence)?;
    if !target_explained_sensitivity.is_finite()
        || !(0.0..=1.0).contains(&target_explained_sensitivity)
        || target_explained_sensitivity <= 0.0
        || !max_damage_ratio.is_finite()
        || !(0.0..=1.0).contains(&max_damage_ratio)
    {
        return Err(BrainError::Invalid("protected_map_config_invalid".into()));
    }
    let rows = Matrix::from_rows(
        &evidence
            .iter()
            .map(|probe| probe.sensitivity.clone())
            .collect::<Vec<_>>(),
    )?;
    let weights = evidence
        .iter()
        .map(|probe| probe.reliability)
        .collect::<Vec<_>>();
    let gram = weighted_row_gram(&rows, &weights)?;
    let eigs = symmetric_eigen_jacobi(&gram, 1e-12, gram.rows * gram.rows * 100)?;
    let eigenvalues = eigs.iter().map(|(value, _)| *value).collect::<Vec<_>>();
    let selected_rank = choose_energy_rank(
        &eigenvalues,
        target_explained_sensitivity,
        eigenvalues.len(),
        0,
    )?;
    let total_energy = eigenvalues.iter().sum::<f64>().max(1e-18);
    let retained_sensitivity_energy =
        eigenvalues.iter().take(selected_rank).sum::<f64>() / total_energy;

    let mut parameter_importance = vec![0.0; dim];
    let total_weight = weights.iter().sum::<f64>().max(1e-15);
    for (probe, weight) in evidence.iter().zip(&weights) {
        for parameter in 0..dim {
            parameter_importance[parameter] += weight * probe.sensitivity[parameter].powi(2);
        }
    }
    for value in &mut parameter_importance {
        *value /= total_weight;
    }
    let fisher_trace = parameter_importance.iter().sum::<f64>();
    let max_importance = parameter_importance
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(1e-18);
    for value in &mut parameter_importance {
        *value /= max_importance;
    }

    let mut directions = Vec::with_capacity(selected_rank);
    for (component, (lambda, eigenvector)) in eigs.iter().take(selected_rank).enumerate() {
        let denom = lambda.sqrt().max(1e-15);
        let mut direction = vec![0.0; dim];
        for row in 0..rows.rows {
            let coefficient = weights[row].sqrt() * eigenvector[row] / denom;
            for parameter in 0..dim {
                direction[parameter] += coefficient * rows.get(row, parameter);
            }
        }
        let direction_norm = norm(&direction)?.max(1e-15);
        for value in &mut direction {
            *value /= direction_norm;
        }
        directions.push(ProtectedDirection {
            probe_id: format!("sensitivity-pc-{component:03}"),
            direction,
            importance: (*lambda / eigenvalues[0].max(1e-18)).clamp(0.0, 1.0),
        });
    }

    let causal = evidence
        .iter()
        .filter_map(|probe| probe.causal_damage.map(|damage| (probe, damage)))
        .collect::<Vec<_>>();
    let sensitivity_damage_correlation = if causal.len() >= 2 {
        let sensitivity_energy = causal
            .iter()
            .map(|(probe, _)| Ok(dot(&probe.sensitivity, &probe.sensitivity)?.sqrt()))
            .collect::<BrainResult<Vec<_>>>()?;
        let damage = causal.iter().map(|(_, damage)| *damage).collect::<Vec<_>>();
        pearson(&sensitivity_energy, &damage)
    } else {
        None
    };
    Ok(ProtectedMapReport {
        schema: "cerebro.tidex.protected_cortex_map/v1".into(),
        probe_count: evidence.len(),
        parameter_dimension: dim,
        selected_rank,
        effective_rank: effective_rank_from_spectrum(&eigenvalues)?,
        retained_sensitivity_energy,
        fisher_trace,
        causal_damage_supported_probes: causal.len(),
        sensitivity_damage_correlation,
        cortex: ProtectedCortex {
            parameter_importance,
            directions,
            max_damage_ratio,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protected::project_to_safe_subspace;

    #[test]
    fn protected_map_learns_load_bearing_subspace() {
        let evidence = vec![
            SensitivityEvidence {
                probe_id: "p1".into(),
                sensitivity: vec![3.0, 0.1, 0.0],
                reliability: 1.0,
                causal_damage: Some(3.1),
            },
            SensitivityEvidence {
                probe_id: "p2".into(),
                sensitivity: vec![2.8, -0.1, 0.0],
                reliability: 1.0,
                causal_damage: Some(2.9),
            },
            SensitivityEvidence {
                probe_id: "p3".into(),
                sensitivity: vec![0.0, 0.2, 0.1],
                reliability: 1.0,
                causal_damage: Some(0.2),
            },
        ];
        let map = build_protected_cortex_map(&evidence, 0.90, 0.95).unwrap();
        assert!(map.selected_rank >= 1);
        assert!(map.retained_sensitivity_energy >= 0.90);
        assert!(map.sensitivity_damage_correlation.unwrap() > 0.9);
        let damaging = project_to_safe_subspace(&[1.0, 0.0, 0.0], &map.cortex).unwrap();
        let orthogonal = project_to_safe_subspace(&[0.0, 0.0, 1.0], &map.cortex).unwrap();
        assert!(norm(&damaging.projected).unwrap() < norm(&orthogonal.projected).unwrap());
    }

    #[test]
    fn protected_map_rejects_zero_reliability_instead_of_fabricating_a_weight() {
        let evidence = vec![
            SensitivityEvidence {
                probe_id: "p1".into(),
                sensitivity: vec![1.0, 0.0],
                reliability: 1.0,
                causal_damage: None,
            },
            SensitivityEvidence {
                probe_id: "p2".into(),
                sensitivity: vec![0.0, 1.0],
                reliability: 0.0,
                causal_damage: None,
            },
        ];
        assert!(build_protected_cortex_map(&evidence, 0.9, 0.1).is_err());
    }
}
