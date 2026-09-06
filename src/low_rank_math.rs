//! Architecture-independent low-rank solvers.
//!
//! This module owns only the numerical problem and its validated results. It
//! has no model identity, tensor-file format, evaluation suite, runtime,
//! persistence, or promotion dependency. Integrations may consume these
//! solutions, but their evidence and residency claims remain separate.

use crate::error::{BrainError, BrainResult};

/// Current hard safety ceiling for the in-memory exact solver. This is a
/// resource bound, not a claim that rank 32 is sufficient for a capability.
pub const MAX_LOW_RANK: u64 = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct MinimumNormRankOneSolution {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub predicted_shift: Vec<f32>,
    pub residual_norm: f64,
    pub frobenius_norm: f64,
}

/// A regularized low-rank fit over several independently supplied
/// activation/response pairs. `left` is row-major `[output, rank]` and
/// `right` is row-major `[rank, input]`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiCaseLowRankSolution {
    pub rank: u64,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub residual_norm: f64,
    pub frobenius_norm: f64,
}

/// Solve the damped minimum-norm rank-one update for one observation.
///
/// For an input `x` and requested local output shift `y`, this returns
/// `delta-W = y x^T / (x^T x + damping)`. It is a numerical solution, not a
/// capability, causal, equivalence, or promotion claim.
pub fn solve_minimum_norm_rank_one(
    input_activation: &[f32],
    desired_output_shift: &[f32],
    damping: f64,
) -> BrainResult<MinimumNormRankOneSolution> {
    if input_activation.is_empty()
        || desired_output_shift.is_empty()
        || !damping.is_finite()
        || damping < 0.0
        || input_activation
            .iter()
            .chain(desired_output_shift)
            .any(|value| !value.is_finite())
    {
        return Err(BrainError::Invalid(
            "minimum_norm_rank_one_input_invalid".into(),
        ));
    }
    let input_squared_norm = input_activation
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>();
    if !input_squared_norm.is_finite() || input_squared_norm <= f64::EPSILON {
        return Err(BrainError::Numerical(
            "minimum_norm_rank_one_activation_degenerate".into(),
        ));
    }
    let denominator = input_squared_norm + damping;
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(BrainError::Numerical(
            "minimum_norm_rank_one_denominator_invalid".into(),
        ));
    }
    let right = input_activation
        .iter()
        .map(|value| (f64::from(*value) / denominator) as f32)
        .collect::<Vec<_>>();
    if right.iter().any(|value| !value.is_finite()) {
        return Err(BrainError::Numerical(
            "minimum_norm_rank_one_factor_non_finite".into(),
        ));
    }
    let response_scale = input_activation
        .iter()
        .zip(&right)
        .map(|(input, factor)| f64::from(*input) * f64::from(*factor))
        .sum::<f64>();
    let predicted_shift = desired_output_shift
        .iter()
        .map(|value| (f64::from(*value) * response_scale) as f32)
        .collect::<Vec<_>>();
    let residual_norm = predicted_shift
        .iter()
        .zip(desired_output_shift)
        .map(|(predicted, desired)| f64::from(*predicted - *desired).powi(2))
        .sum::<f64>()
        .sqrt();
    let left_squared_norm = desired_output_shift
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>();
    let right_squared_norm = right
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>();
    let frobenius_norm = (left_squared_norm * right_squared_norm).sqrt();
    if !residual_norm.is_finite() || !frobenius_norm.is_finite() || frobenius_norm == 0.0 {
        return Err(BrainError::Numerical(
            "minimum_norm_rank_one_solution_invalid".into(),
        ));
    }
    Ok(MinimumNormRankOneSolution {
        left: desired_output_shift.to_vec(),
        right,
        predicted_shift,
        residual_norm,
        frobenius_norm,
    })
}

/// Fit the minimum-norm ridge update `Y (X X^T + lambda I)^-1 X`.
///
/// The returned factorization has one component per independently supplied
/// case. A singleton or duplicate activation set is rejected so it cannot be
/// mislabeled as multi-case evidence.
pub fn solve_regularized_multi_case_low_rank(
    input_activations: &[Vec<f32>],
    desired_output_shifts: &[Vec<f32>],
    damping: f64,
) -> BrainResult<MultiCaseLowRankSolution> {
    let cases = input_activations.len();
    if cases < 2
        || cases != desired_output_shifts.len()
        || cases as u64 > MAX_LOW_RANK
        || !damping.is_finite()
        || damping <= 0.0
    {
        return Err(BrainError::Invalid(
            "multi_case_low_rank_input_invalid".into(),
        ));
    }
    let input_dimension = input_activations[0].len();
    let output_dimension = desired_output_shifts[0].len();
    if input_dimension == 0
        || output_dimension == 0
        || input_activations
            .iter()
            .any(|row| row.len() != input_dimension || row.iter().any(|value| !value.is_finite()))
        || desired_output_shifts
            .iter()
            .any(|row| row.len() != output_dimension || row.iter().any(|value| !value.is_finite()))
        || input_activations
            .iter()
            .enumerate()
            .any(|(index, row)| input_activations[..index].contains(row))
    {
        return Err(BrainError::Invalid(
            "multi_case_low_rank_examples_invalid".into(),
        ));
    }

    let mut gram = vec![0.0_f64; cases * cases];
    for row in 0..cases {
        for column in 0..cases {
            gram[row * cases + column] = input_activations[row]
                .iter()
                .zip(&input_activations[column])
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum::<f64>();
        }
        gram[row * cases + row] += damping;
    }
    let cholesky = cholesky_spd(&gram, cases)?;

    let mut coefficients_by_input = vec![0.0_f32; input_dimension * cases];
    for input in 0..input_dimension {
        let rhs = input_activations
            .iter()
            .map(|row| f64::from(row[input]))
            .collect::<Vec<_>>();
        let coefficients = solve_cholesky(&cholesky, &rhs, cases)?;
        for case_index in 0..cases {
            coefficients_by_input[input * cases + case_index] = coefficients[case_index] as f32;
        }
    }
    let mut right = vec![0.0_f32; cases * input_dimension];
    for component in 0..cases {
        for input in 0..input_dimension {
            right[component * input_dimension + input] =
                coefficients_by_input[input * cases + component];
        }
    }
    let mut left = vec![0.0_f32; output_dimension * cases];
    for output in 0..output_dimension {
        for case_index in 0..cases {
            left[output * cases + case_index] = desired_output_shifts[case_index][output];
        }
    }

    let mut residual_squared = 0.0_f64;
    for case_index in 0..cases {
        for output in 0..output_dimension {
            let predicted = (0..cases)
                .map(|component| {
                    f64::from(left[output * cases + component])
                        * input_activations[case_index]
                            .iter()
                            .enumerate()
                            .map(|(input, value)| {
                                f64::from(*value)
                                    * f64::from(right[component * input_dimension + input])
                            })
                            .sum::<f64>()
                })
                .sum::<f64>();
            residual_squared +=
                (predicted - f64::from(desired_output_shifts[case_index][output])).powi(2);
        }
    }
    let frobenius_squared = (0..output_dimension)
        .flat_map(|output| (0..input_dimension).map(move |input| (output, input)))
        .map(|(output, input)| {
            let value = (0..cases)
                .map(|component| {
                    f64::from(left[output * cases + component])
                        * f64::from(right[component * input_dimension + input])
                })
                .sum::<f64>();
            value * value
        })
        .sum::<f64>();
    let residual_norm = residual_squared.sqrt();
    let frobenius_norm = frobenius_squared.sqrt();
    if !residual_norm.is_finite() || !frobenius_norm.is_finite() || frobenius_norm <= 0.0 {
        return Err(BrainError::Numerical(
            "multi_case_low_rank_solution_invalid".into(),
        ));
    }
    Ok(MultiCaseLowRankSolution {
        rank: cases as u64,
        left,
        right,
        residual_norm,
        frobenius_norm,
    })
}

/// Recompute the relative residual of a dense linear update against an exact
/// multi-case contract. Integrations use this independent recomputation rather
/// than trusting the residual reported by a producer.
pub fn dense_multi_case_relative_residual(
    dense: &[f32],
    rows: usize,
    columns: usize,
    inputs: &[Vec<f32>],
    exact_shifts: &[Vec<f64>],
) -> BrainResult<f64> {
    if inputs.len() < 2
        || inputs.len() != exact_shifts.len()
        || dense.len() != rows.saturating_mul(columns)
        || inputs
            .iter()
            .any(|input| input.len() != columns || input.iter().any(|value| !value.is_finite()))
        || exact_shifts
            .iter()
            .any(|shift| shift.len() != rows || shift.iter().any(|value| !value.is_finite()))
    {
        return Err(BrainError::Integrity(
            "multi_case_low_rank_residual_inputs_invalid".into(),
        ));
    }
    let mut residual_squared = 0.0_f64;
    let mut target_squared = 0.0_f64;
    for (input, shift) in inputs.iter().zip(exact_shifts) {
        for row in 0..rows {
            let predicted = input
                .iter()
                .enumerate()
                .map(|(column, input)| f64::from(dense[row * columns + column]) * f64::from(*input))
                .sum::<f64>();
            residual_squared += (predicted - shift[row]).powi(2);
            target_squared += shift[row].powi(2);
        }
    }
    if target_squared <= f64::EPSILON {
        return Err(BrainError::Numerical(
            "multi_case_low_rank_relative_residual_invalid".into(),
        ));
    }
    let relative = residual_squared.sqrt() / target_squared.sqrt();
    if !relative.is_finite() {
        return Err(BrainError::Numerical(
            "multi_case_low_rank_relative_residual_invalid".into(),
        ));
    }
    Ok(relative)
}

fn cholesky_spd(matrix: &[f64], dimension: usize) -> BrainResult<Vec<f64>> {
    let mut lower = vec![0.0; dimension * dimension];
    for row in 0..dimension {
        for column in 0..=row {
            let sum = matrix[row * dimension + column]
                - (0..column)
                    .map(|k| lower[row * dimension + k] * lower[column * dimension + k])
                    .sum::<f64>();
            if row == column {
                if !sum.is_finite() || sum <= f64::EPSILON {
                    return Err(BrainError::Numerical(
                        "multi_case_low_rank_gram_indefinite".into(),
                    ));
                }
                lower[row * dimension + column] = sum.sqrt();
            } else {
                lower[row * dimension + column] = sum / lower[column * dimension + column];
            }
        }
    }
    Ok(lower)
}

fn solve_cholesky(lower: &[f64], rhs: &[f64], dimension: usize) -> BrainResult<Vec<f64>> {
    let mut forward = vec![0.0; dimension];
    for row in 0..dimension {
        forward[row] = (rhs[row]
            - (0..row)
                .map(|k| lower[row * dimension + k] * forward[k])
                .sum::<f64>())
            / lower[row * dimension + row];
    }
    let mut result = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        result[row] = (forward[row]
            - ((row + 1)..dimension)
                .map(|k| lower[k * dimension + row] * result[k])
                .sum::<f64>())
            / lower[row * dimension + row];
    }
    if result.iter().any(|value| !value.is_finite()) {
        return Err(BrainError::Numerical(
            "multi_case_low_rank_linear_solve_invalid".into(),
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_one_solution_is_exact_without_damping() {
        let solution = solve_minimum_norm_rank_one(&[3.0, 4.0], &[2.0, -1.0], 0.0).unwrap();
        assert!(solution.residual_norm <= 1e-6);
        assert_eq!(solution.left, vec![2.0, -1.0]);
        assert_eq!(solution.right, vec![0.12, 0.16]);
    }

    #[test]
    fn degenerate_and_invalid_rank_one_inputs_fail_closed() {
        assert!(solve_minimum_norm_rank_one(&[0.0, 0.0], &[1.0], 0.0).is_err());
        assert!(solve_minimum_norm_rank_one(&[1.0], &[1.0], -1.0).is_err());
        assert!(solve_minimum_norm_rank_one(&[f32::NAN], &[1.0], 0.0).is_err());
    }
}
