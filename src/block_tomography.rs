#![allow(clippy::needless_range_loop)]

use crate::artifact::{inspect_dvec, read_dvec_range, DeltaArtifactRef};
use crate::contracts::{
    BlockSubspaceAxis, BlockSubspaceGeometry, BrainConfig, SkillField, SkillSubspaceGeometry,
};
use crate::error::{BrainError, BrainResult};
use crate::linalg::{dot, norm, symmetric_eigen_jacobi, weighted_row_gram, Matrix};
use crate::validation::{choose_energy_rank, effective_rank_from_spectrum, source_support_indices};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BlockShapeSpec {
    #[serde(rename = "module")]
    pub name: String,
    pub shape: Vec<usize>,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterBlockSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterBlockLayout {
    pub schema: String,
    pub blocks: Vec<ParameterBlockSpec>,
    pub total_parameter_count: u64,
}

impl ParameterBlockLayout {
    pub fn from_shapes(specs: &[BlockShapeSpec]) -> BrainResult<Self> {
        if specs.is_empty() {
            return Err(BrainError::Invalid("block_layout_empty".into()));
        }
        let mut blocks = Vec::with_capacity(specs.len());
        let mut offset = 0u64;
        for (index, spec) in specs.iter().enumerate() {
            if spec.name.trim().is_empty()
                || spec.count == 0
                || spec.shape.is_empty()
                || spec.shape.contains(&0)
            {
                return Err(BrainError::Invalid(format!("block_layout_invalid:{index}")));
            }
            let shape_count = spec.shape.iter().try_fold(1usize, |acc, value| {
                acc.checked_mul(*value)
                    .ok_or_else(|| BrainError::Invalid("block_layout_shape_overflow".into()))
            })?;
            if shape_count != spec.count {
                return Err(BrainError::Invalid(format!(
                    "block_layout_shape_count_mismatch:{}:{}:{}",
                    spec.name, shape_count, spec.count
                )));
            }
            blocks.push(ParameterBlockSpec {
                name: spec.name.clone(),
                shape: spec.shape.clone(),
                offset,
                count: spec.count,
            });
            offset = offset
                .checked_add(spec.count as u64)
                .ok_or_else(|| BrainError::Invalid("block_layout_offset_overflow".into()))?;
        }
        Ok(Self {
            schema: "cerebro.tidex.parameter_block_layout/v1".into(),
            blocks,
            total_parameter_count: offset,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StructuredSource {
    pub observation_id: String,
    pub artifact: DeltaArtifactRef,
    pub reliability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StructuredGeometryReport {
    pub schema: String,
    pub source_count: usize,
    pub block_count: usize,
    pub total_parameter_count: u64,
    pub skills: Vec<SkillSubspaceGeometry>,
}

fn validate_sources(
    fields: &[SkillField],
    source_mixtures: &[Vec<f64>],
    sources: &[StructuredSource],
    layout: &ParameterBlockLayout,
) -> BrainResult<()> {
    if fields.is_empty()
        || fields.len() != source_mixtures.len()
        || source_mixtures.iter().any(|row| row.len() != sources.len())
    {
        return Err(BrainError::Invalid(
            "structured_geometry_source_count_mismatch".into(),
        ));
    }
    for (index, source) in sources.iter().enumerate() {
        if source.observation_id.trim().is_empty()
            || !source.reliability.is_finite()
            || !(0.0..=1.0).contains(&source.reliability)
        {
            return Err(BrainError::Invalid(format!(
                "structured_geometry_source_invalid:{index}"
            )));
        }
        let inspected = inspect_dvec(Path::new(&source.artifact.path))?;
        if inspected.sha256 != source.artifact.sha256
            || inspected.parameter_count != source.artifact.parameter_count
            || inspected.parameter_count != layout.total_parameter_count
        {
            return Err(BrainError::Integrity(format!(
                "structured_geometry_artifact_mismatch:{index}"
            )));
        }
    }
    Ok(())
}

fn reconstruct_block(
    block: &ParameterBlockSpec,
    sources: &[StructuredSource],
    support: &[usize],
    alignment_mixture: &[f64],
    cfg: &BrainConfig,
) -> BrainResult<BlockSubspaceGeometry> {
    if support.is_empty() {
        return Err(BrainError::Invalid(
            "structured_geometry_empty_skill_support".into(),
        ));
    }
    let mut rows = Vec::with_capacity(support.len());
    let mut weights = Vec::with_capacity(support.len());
    for &source_index in support {
        let mut values = read_dvec_range(
            Path::new(&sources[source_index].artifact.path),
            block.offset,
            block.count,
        )?;
        let sign = alignment_mixture[source_index].signum();
        if sign < 0.0 {
            for value in &mut values {
                *value = -*value;
            }
        }
        rows.push(values);
        weights.push(sources[source_index].reliability.clamp(1e-4, 1.0));
    }
    let data = Matrix::from_rows(&rows)?;
    let block_energy = (0..data.rows).try_fold(0.0, |total, row| {
        Ok::<f64, BrainError>(total + weights[row] * dot(data.row(row), data.row(row))?)
    })?;
    if block_energy <= 1e-24 {
        return Ok(BlockSubspaceGeometry {
            block_name: block.name.clone(),
            offset: block.offset,
            count: block.count,
            shape: block.shape.clone(),
            selected_rank: 0,
            effective_rank: 0.0,
            retained_energy: 0.0,
            block_energy: 0.0,
            normalized_block_energy: 0.0,
            reconstruction_rms: 0.0,
            axes: Vec::new(),
        });
    }
    let gram = weighted_row_gram(&data, &weights)?;
    let eigs = symmetric_eigen_jacobi(&gram, 1e-11, data.rows * data.rows * 100)?;
    let eigenvalues = eigs.iter().map(|(value, _)| *value).collect::<Vec<_>>();
    let rank = choose_energy_rank(
        &eigenvalues,
        cfg.target_explained_variance,
        cfg.max_rank.min(support.len()),
        0,
    )?;
    let total_energy = eigenvalues.iter().sum::<f64>().max(1e-18);
    let retained_energy = eigenvalues.iter().take(rank).sum::<f64>() / total_energy;

    let mut basis = Vec::<Vec<f64>>::with_capacity(rank);
    let mut axes = Vec::with_capacity(rank);
    for (lambda, eigenvector) in eigs.iter().take(rank) {
        let denom = lambda.sqrt().max(1e-15);
        let mut axis = vec![0.0; data.cols];
        let mut global_coefficients = vec![0.0; sources.len()];
        for local_row in 0..data.rows {
            let local_coefficient = weights[local_row].sqrt() * eigenvector[local_row] / denom;
            for parameter in 0..data.cols {
                axis[parameter] += local_coefficient * data.get(local_row, parameter);
            }
            let source_index = support[local_row];
            let sign = alignment_mixture[source_index].signum();
            global_coefficients[source_index] =
                local_coefficient * if sign < 0.0 { -1.0 } else { 1.0 };
        }
        let axis_norm = norm(&axis)?.max(1e-15);
        for value in &mut axis {
            *value /= axis_norm;
        }
        for coefficient in &mut global_coefficients {
            *coefficient /= axis_norm;
        }
        basis.push(axis);
        axes.push(BlockSubspaceAxis {
            singular_value: lambda.sqrt(),
            source_coefficients: global_coefficients,
        });
    }

    let mut squared_error = 0.0;
    for row in 0..data.rows {
        let mut reconstructed = vec![0.0; data.cols];
        for axis in &basis {
            let coefficient = dot(data.row(row), axis)?;
            for parameter in 0..data.cols {
                reconstructed[parameter] += coefficient * axis[parameter];
            }
        }
        for parameter in 0..data.cols {
            squared_error += (data.get(row, parameter) - reconstructed[parameter]).powi(2);
        }
    }
    let reconstruction_rms = (squared_error / (data.rows * data.cols).max(1) as f64).sqrt();
    Ok(BlockSubspaceGeometry {
        block_name: block.name.clone(),
        offset: block.offset,
        count: block.count,
        shape: block.shape.clone(),
        selected_rank: rank,
        effective_rank: effective_rank_from_spectrum(&eigenvalues)?,
        retained_energy,
        block_energy,
        normalized_block_energy: 0.0,
        reconstruction_rms,
        axes,
    })
}

pub fn reconstruct_structured_geometry(
    fields: &[SkillField],
    source_mixtures: &[Vec<f64>],
    sources: &[StructuredSource],
    layout: &ParameterBlockLayout,
    cfg: &BrainConfig,
) -> BrainResult<StructuredGeometryReport> {
    validate_sources(fields, source_mixtures, sources, layout)?;
    let mut skills = Vec::with_capacity(fields.len());
    for (skill_index, field) in fields.iter().enumerate() {
        let mixture = source_mixtures
            .get(skill_index)
            .ok_or_else(|| BrainError::Integrity("structured_geometry_mixture_missing".into()))?;
        let support = source_support_indices(mixture)?;
        if support.is_empty() {
            return Err(BrainError::Invalid(format!(
                "structured_geometry_skill_support_empty:{skill_index}"
            )));
        }
        let mut blocks = Vec::with_capacity(layout.blocks.len());
        for block in &layout.blocks {
            blocks.push(reconstruct_block(block, sources, &support, mixture, cfg)?);
        }
        let total_block_energy = blocks.iter().map(|block| block.block_energy).sum::<f64>();
        if total_block_energy > 1e-24 {
            for block in &mut blocks {
                block.normalized_block_energy = block.block_energy / total_block_energy;
            }
        }
        let max_local_rank = blocks
            .iter()
            .map(|block| block.selected_rank)
            .max()
            .unwrap_or(0);
        let active_blocks = blocks
            .iter()
            .filter(|block| block.selected_rank > 0)
            .count();
        let mean_effective_rank = if active_blocks == 0 {
            0.0
        } else {
            blocks
                .iter()
                .filter(|block| block.selected_rank > 0)
                .map(|block| block.effective_rank)
                .sum::<f64>()
                / active_blocks as f64
        };
        skills.push(SkillSubspaceGeometry {
            skill_id: field.skill_id.clone(),
            source_support_indices: support,
            blocks,
            max_local_rank,
            mean_effective_rank,
        });
    }
    Ok(StructuredGeometryReport {
        schema: "cerebro.tidex.structured_geometry/v1".into(),
        source_count: sources.len(),
        block_count: layout.blocks.len(),
        total_parameter_count: layout.total_parameter_count,
        skills,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::create_dvec;
    use crate::contracts::SkillField;
    use std::fs;

    fn minimal_field() -> SkillField {
        SkillField {
            skill_id: "s".into(),
            reconstruction_id: "r".into(),
            lineage_id: "l".into(),
            generation_created: 1,
            direction: vec![1.0, 0.0, 0.0, 0.0],
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
            support: 3,
            functional_signature: vec![],
            parent_skill_ids: vec![],
        }
    }

    #[test]
    fn block_geometry_preserves_local_rank_and_source_coefficients() {
        let root = std::env::temp_dir().join(format!("tidex-block-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let a = create_dvec(&root, "a", &[1.0, 0.0, 1.0, 0.0]).unwrap();
        let b = create_dvec(&root, "b", &[0.0, 1.0, 0.0, 1.0]).unwrap();
        let c = create_dvec(&root, "c", &[1.0, 1.0, 1.0, 1.0]).unwrap();
        let sources = vec![
            StructuredSource {
                observation_id: "a".into(),
                artifact: a,
                reliability: 1.0,
            },
            StructuredSource {
                observation_id: "b".into(),
                artifact: b,
                reliability: 1.0,
            },
            StructuredSource {
                observation_id: "c".into(),
                artifact: c,
                reliability: 1.0,
            },
        ];
        let layout = ParameterBlockLayout::from_shapes(&[
            BlockShapeSpec {
                name: "x".into(),
                shape: vec![2],
                count: 2,
            },
            BlockShapeSpec {
                name: "y".into(),
                shape: vec![2],
                count: 2,
            },
        ])
        .unwrap();
        let fields = vec![minimal_field()];
        let mixtures = vec![vec![1.0 / 3.0; 3]];
        let geometry = reconstruct_structured_geometry(
            &fields,
            &mixtures,
            &sources,
            &layout,
            &BrainConfig {
                target_explained_variance: 0.99,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(geometry.skills.len(), 1);
        assert_eq!(geometry.skills[0].blocks.len(), 2);
        assert!(geometry.skills[0].max_local_rank >= 2);
        assert!(geometry.skills[0].blocks.iter().all(|block| block
            .axes
            .iter()
            .all(|axis| axis.source_coefficients.len() == 3)));
        let _ = fs::remove_dir_all(root);
    }
}
