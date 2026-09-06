#![allow(clippy::needless_range_loop)]

use crate::contracts::SkillField;
use crate::error::{BrainError, BrainResult};
use crate::identity::SkillId;
use crate::linalg::{cosine, inverse_with_ridge, norm, symmetric_eigen_jacobi, Matrix};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldResolution {
    pub skill_id: SkillId,
    pub coefficient_rms: f64,
    pub posterior_std: f64,
    pub signal_to_posterior_noise: f64,
    pub resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolutionMap {
    pub schema: String,
    pub field_count: usize,
    pub field_geometry_numerical_rank: usize,
    pub excitation_numerical_rank: usize,
    pub resolved_rank: usize,
    pub field_geometry_spectrum: Vec<f64>,
    pub excitation_spectrum: Vec<f64>,
    pub field_geometry_condition: f64,
    pub excitation_condition: f64,
    pub min_principal_angle_degrees: f64,
    pub posterior_covariance: Vec<Vec<f64>>,
    pub fields: Vec<FieldResolution>,
    pub all_fields_resolved: bool,
    pub unresolved_field_ids: Vec<SkillId>,
}

fn gram_of_fields(fields: &[SkillField]) -> BrainResult<Matrix> {
    if fields.is_empty() {
        return Err(BrainError::Invalid(
            "identifiability_fields_required".into(),
        ));
    }
    let dim = fields[0].direction.len();
    let mut skill_ids = BTreeSet::new();
    let norm_tolerance = f64::EPSILON.sqrt() * (dim.max(1) as f64).sqrt() * 16.0;
    if dim == 0 {
        return Err(BrainError::Invalid("identifiability_field_shape".into()));
    }
    for field in fields {
        if field.skill_id.trim().is_empty()
            || !skill_ids.insert(field.skill_id.as_str())
            || field.direction.len() != dim
            || field.direction.iter().any(|value| !value.is_finite())
            || (norm(&field.direction)? - 1.0).abs() > norm_tolerance
        {
            return Err(BrainError::Invalid("identifiability_field_shape".into()));
        }
    }
    let mut gram = Matrix::zeros(fields.len(), fields.len());
    for i in 0..fields.len() {
        for j in i..fields.len() {
            let value = crate::linalg::dot(&fields[i].direction, &fields[j].direction)?;
            gram.set(i, j, value);
            gram.set(j, i, value);
        }
    }
    Ok(gram)
}

fn excitation_information(coefficients: &Matrix, weights: &[f64]) -> BrainResult<Matrix> {
    coefficients.validate("identifiability_coefficients")?;
    if coefficients.rows == 0
        || coefficients.cols == 0
        || coefficients.rows != weights.len()
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err(BrainError::Invalid(
            "identifiability_excitation_shape".into(),
        ));
    }
    let mut info = Matrix::zeros(coefficients.cols, coefficients.cols);
    for row in 0..coefficients.rows {
        let weight = weights[row];
        for i in 0..coefficients.cols {
            let left = coefficients.get(row, i);
            for j in 0..coefficients.cols {
                info.data[i * info.cols + j] += weight * left * coefficients.get(row, j);
            }
        }
    }
    info.validate("identifiability_excitation")?;
    Ok(info)
}

fn spectrum_and_rank(matrix: &Matrix) -> BrainResult<(Vec<f64>, usize, f64)> {
    let eigs = symmetric_eigen_jacobi(matrix, 1e-12, matrix.rows * matrix.rows * 100)?;
    let spectrum = eigs
        .iter()
        .map(|(eigenvalue, _)| eigenvalue.max(0.0).sqrt())
        .collect::<Vec<_>>();
    if spectrum.is_empty() {
        return Ok((Vec::new(), 0, f64::INFINITY));
    }
    let largest = spectrum[0].max(1e-18);
    // Numerical rank is derived from floating-point resolution and matrix size,
    // not a task-specific hand-tuned cutoff.
    let tolerance = largest * f64::EPSILON.sqrt() * matrix.rows.max(1) as f64;
    let rank = spectrum.iter().filter(|value| **value > tolerance).count();
    let smallest_resolved = spectrum
        .iter()
        .copied()
        .filter(|value| *value > tolerance)
        .fold(f64::INFINITY, f64::min);
    let condition = if rank == 0 || !smallest_resolved.is_finite() {
        f64::INFINITY
    } else {
        largest / smallest_resolved.max(1e-18)
    };
    Ok((spectrum, rank, condition))
}

fn principal_angle_min(fields: &[SkillField]) -> BrainResult<f64> {
    if fields.len() < 2 {
        return Ok(90.0);
    }
    let mut min_angle = 90.0_f64;
    for i in 0..fields.len() {
        for j in 0..i {
            let c = cosine(&fields[i].direction, &fields[j].direction)?
                .abs()
                .clamp(0.0, 1.0);
            min_angle = min_angle.min(c.acos().to_degrees());
        }
    }
    Ok(min_angle)
}

pub fn resolution_map(
    fields: &[SkillField],
    coefficients: &Matrix,
    observation_weights: &[f64],
    reconstruction_rms: f64,
    ridge: f64,
    minimum_signal_to_noise: f64,
) -> BrainResult<ResolutionMap> {
    if fields.len() != coefficients.cols
        || coefficients.rows != observation_weights.len()
        || !reconstruction_rms.is_finite()
        || reconstruction_rms < 0.0
        || !ridge.is_finite()
        || ridge <= 0.0
        || !minimum_signal_to_noise.is_finite()
        || minimum_signal_to_noise <= 0.0
    {
        return Err(BrainError::Invalid("identifiability_input_contract".into()));
    }
    let geometry = gram_of_fields(fields)?;
    let excitation = excitation_information(coefficients, observation_weights)?;
    let (geometry_spectrum, geometry_rank, geometry_condition) = spectrum_and_rank(&geometry)?;
    let (excitation_spectrum, excitation_rank, excitation_condition) =
        spectrum_and_rank(&excitation)?;
    let resolved_rank = geometry_rank.min(excitation_rank);

    let covariance = inverse_with_ridge(&excitation, ridge)?;
    let noise_variance = reconstruction_rms.powi(2).max(f64::EPSILON);
    let mut field_resolution = Vec::with_capacity(fields.len());
    let mut unresolved = Vec::new();
    for field_index in 0..fields.len() {
        let weighted_squared_coefficient = (0..coefficients.rows)
            .map(|row| observation_weights[row] * coefficients.get(row, field_index).powi(2))
            .sum::<f64>();
        let total_observation_weight = observation_weights.iter().copied().sum::<f64>();
        if !weighted_squared_coefficient.is_finite()
            || !total_observation_weight.is_finite()
            || total_observation_weight <= 0.0
        {
            return Err(BrainError::Numerical(
                "identifiability_coefficient_energy_invalid".into(),
            ));
        }
        let coefficient_rms = (weighted_squared_coefficient / total_observation_weight).sqrt();
        let covariance_diagonal = covariance.get(field_index, field_index);
        let covariance_tolerance = f64::EPSILON.sqrt()
            * covariance
                .as_slice()
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
        if covariance_diagonal < -covariance_tolerance {
            return Err(BrainError::Numerical(
                "identifiability_negative_posterior_variance".into(),
            ));
        }
        let posterior_std = (covariance_diagonal.max(0.0) * noise_variance).sqrt();
        let signal_to_posterior_noise = coefficient_rms / posterior_std.max(1e-15);
        // One-sigma identifiability: the observed field amplitude must exceed
        // its posterior standard deviation, and the joint system must have full
        // numerical rank. "Unresolved" is a valid result, never promoted as truth.
        let resolved = resolved_rank == fields.len()
            && signal_to_posterior_noise.is_finite()
            && signal_to_posterior_noise >= minimum_signal_to_noise;
        if !resolved {
            unresolved.push(fields[field_index].skill_id.clone());
        }
        field_resolution.push(FieldResolution {
            skill_id: fields[field_index].skill_id.clone(),
            coefficient_rms,
            posterior_std,
            signal_to_posterior_noise,
            resolved,
        });
    }
    let posterior_covariance = (0..covariance.rows)
        .map(|row| covariance.row_vec(row))
        .collect::<Vec<_>>();
    Ok(ResolutionMap {
        schema: "cerebro.tidex.resolution_map/v1".into(),
        field_count: fields.len(),
        field_geometry_numerical_rank: geometry_rank,
        excitation_numerical_rank: excitation_rank,
        resolved_rank,
        field_geometry_spectrum: geometry_spectrum,
        excitation_spectrum,
        field_geometry_condition: geometry_condition,
        excitation_condition,
        min_principal_angle_degrees: principal_angle_min(fields)?,
        posterior_covariance,
        fields: field_resolution,
        all_fields_resolved: unresolved.is_empty(),
        unresolved_field_ids: unresolved,
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
            explained_variance: 0.5,
            persistence: 1.0,
            coherence: 1.0,
            uncertainty: 0.0,
            evidence_support_digests: Vec::new(),
            support: 4,
            functional_signature: vec![],
            parent_skill_ids: vec![],
        }
    }

    #[test]
    fn resolution_map_accepts_independent_excited_fields() {
        let fields = vec![field("a", vec![1.0, 0.0]), field("b", vec![0.0, 1.0])];
        let coefficients = Matrix::from_rows(&[
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.2],
            vec![0.2, 1.0],
        ])
        .unwrap();
        let map = resolution_map(&fields, &coefficients, &[1.0; 4], 0.01, 1e-6, 1.0).unwrap();
        assert_eq!(map.resolved_rank, 2);
        assert!(map.all_fields_resolved);
        assert!(map.min_principal_angle_degrees > 89.0);
    }

    #[test]
    fn resolution_map_marks_collinear_fields_unresolved() {
        let fields = vec![field("a", vec![1.0, 0.0]), field("b", vec![1.0, 0.0])];
        let coefficients = Matrix::from_rows(&[
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
        ])
        .unwrap();
        let map = resolution_map(&fields, &coefficients, &[1.0; 4], 0.01, 1e-6, 1.0).unwrap();
        assert!(map.resolved_rank < 2);
        assert!(!map.all_fields_resolved);
        assert_eq!(map.unresolved_field_ids.len(), 2);
    }

    #[test]
    fn resolution_map_rejects_zero_weight_and_noncanonical_field_geometry() {
        let fields = vec![field("a", vec![1.0, 0.0])];
        let coefficients = Matrix::from_rows(&[vec![1.0], vec![2.0]]).unwrap();
        assert!(resolution_map(&fields, &coefficients, &[1.0, 0.0], 0.01, 1e-6, 1.0).is_err());

        let invalid = vec![field("a", vec![2.0, 0.0])];
        assert!(resolution_map(&invalid, &coefficients, &[1.0; 2], 0.01, 1e-6, 1.0).is_err());
    }
}
