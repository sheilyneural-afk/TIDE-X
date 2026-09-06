use cerebro_tidex::low_rank_math::{
    solve_regularized_multi_case_low_rank, MultiCaseLowRankSolution,
};

fn next_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

fn bounded_value(state: &mut u64) -> f32 {
    let integer = (next_u32(state) % 33) as i32 - 16;
    integer as f32 / 4.0
}

fn dense(solution: &MultiCaseLowRankSolution, rows: usize, columns: usize) -> Vec<f64> {
    let rank = solution.rank as usize;
    (0..rows)
        .flat_map(|row| {
            (0..columns).map(move |column| {
                (0..rank)
                    .map(|component| {
                        f64::from(solution.left[row * rank + component])
                            * f64::from(solution.right[component * columns + column])
                    })
                    .sum()
            })
        })
        .collect()
}

#[test]
fn generated_solutions_have_consistent_factors_and_reported_norms() {
    let mut state = 0x5449_4445_585f_5031_u64;
    for sample in 0..256 {
        let cases = 2 + next_u32(&mut state) as usize % 4;
        let input_dimension = cases + next_u32(&mut state) as usize % 4;
        let output_dimension = 1 + next_u32(&mut state) as usize % 5;
        let mut inputs = vec![vec![0.0; input_dimension]; cases];
        for (case, row) in inputs.iter_mut().enumerate() {
            row[case] = 1.0 + case as f32;
            for value in &mut row[cases..] {
                *value = bounded_value(&mut state);
            }
        }
        let shifts = (0..cases)
            .map(|_| {
                (0..output_dimension)
                    .map(|_| bounded_value(&mut state))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if shifts.iter().flatten().all(|value| *value == 0.0) {
            continue;
        }
        let damping = 10.0_f64.powi(-9 + sample % 8);
        let solution = solve_regularized_multi_case_low_rank(&inputs, &shifts, damping)
            .expect("bounded independent examples must be solvable");
        assert_eq!(solution.rank as usize, cases);
        assert_eq!(solution.left.len(), output_dimension * cases);
        assert_eq!(solution.right.len(), cases * input_dimension);

        let matrix = dense(&solution, output_dimension, input_dimension);
        let recomputed_frobenius = matrix.iter().map(|value| value * value).sum::<f64>().sqrt();
        let recomputed_residual = inputs
            .iter()
            .zip(&shifts)
            .flat_map(|(input, desired)| {
                let matrix = &matrix;
                (0..output_dimension).map(move |output| {
                    let predicted = (0..input_dimension)
                        .map(|column| {
                            matrix[output * input_dimension + column] * f64::from(input[column])
                        })
                        .sum::<f64>();
                    (predicted - f64::from(desired[output])).powi(2)
                })
            })
            .sum::<f64>()
            .sqrt();
        assert!(
            (recomputed_frobenius - solution.frobenius_norm).abs()
                <= 1e-5_f64.max(solution.frobenius_norm * 1e-6)
        );
        assert!(
            (recomputed_residual - solution.residual_norm).abs()
                <= 1e-5_f64.max(solution.residual_norm * 1e-6)
        );
    }
}

#[test]
fn generated_dense_solution_is_invariant_to_case_rotation() {
    let mut state = 0x524f_5441_5445_5031_u64;
    for _ in 0..128 {
        let cases = 2 + next_u32(&mut state) as usize % 4;
        let input_dimension = cases + 2;
        let output_dimension = 1 + next_u32(&mut state) as usize % 4;
        let mut inputs = vec![vec![0.0; input_dimension]; cases];
        for (case, row) in inputs.iter_mut().enumerate() {
            row[case] = 2.0;
            row[cases] = bounded_value(&mut state);
            row[cases + 1] = bounded_value(&mut state);
        }
        let shifts = (0..cases)
            .map(|_| {
                (0..output_dimension)
                    .map(|_| bounded_value(&mut state))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if shifts.iter().flatten().all(|value| *value == 0.0) {
            continue;
        }
        let original = solve_regularized_multi_case_low_rank(&inputs, &shifts, 1e-5)
            .expect("independent examples must be solvable");
        let mut rotated_inputs = inputs.clone();
        let mut rotated_shifts = shifts.clone();
        rotated_inputs.rotate_left(1);
        rotated_shifts.rotate_left(1);
        let rotated = solve_regularized_multi_case_low_rank(&rotated_inputs, &rotated_shifts, 1e-5)
            .expect("case rotation must remain solvable");
        for (left, right) in dense(&original, output_dimension, input_dimension)
            .iter()
            .zip(dense(&rotated, output_dimension, input_dimension))
        {
            assert!((*left - right).abs() <= 2e-5_f64.max(left.abs() * 2e-5));
        }
    }
}

#[test]
fn invalid_multi_case_inputs_are_rejected_without_panicking() {
    let valid_inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let valid_shifts = vec![vec![1.0], vec![2.0]];
    for damping in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            solve_regularized_multi_case_low_rank(&valid_inputs, &valid_shifts, damping).is_err()
        );
    }
    assert!(
        solve_regularized_multi_case_low_rank(&valid_inputs[..1], &valid_shifts[..1], 1e-6)
            .is_err()
    );
    assert!(
        solve_regularized_multi_case_low_rank(&[vec![1.0], vec![1.0]], &valid_shifts, 1e-6)
            .is_err()
    );
    assert!(solve_regularized_multi_case_low_rank(
        &[vec![1.0], vec![1.0, 2.0]],
        &valid_shifts,
        1e-6
    )
    .is_err());
    assert!(solve_regularized_multi_case_low_rank(
        &[vec![1.0], vec![f32::NAN]],
        &valid_shifts,
        1e-6
    )
    .is_err());
}
