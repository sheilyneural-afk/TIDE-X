#![no_main]

use cerebro_tidex::solver_portfolio::{
    solve_with_portfolio, LeastSquaresProblem, PortfolioPolicy, SolverStatus,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    if bytes.len() < 5 {
        return;
    }
    let cases = usize::from(bytes[0] % 7) + 2;
    let input_dimension = usize::from(bytes[1] % 16) + 1;
    let output_dimension = usize::from(bytes[2] % 16) + 1;
    let value_count = cases * (input_dimension + output_dimension);
    if bytes.len() < 4 + value_count * 2 {
        return;
    }
    // Map arbitrary input to bounded finite values. Feeding raw f32 bits made
    // a large fraction of cases terminate at the non-finite input guard and
    // left the factorization itself comparatively under-exercised.
    let mut values = bytes[4..]
        .chunks_exact(2)
        .map(|chunk| f64::from(i16::from_le_bytes([chunk[0], chunk[1]])) / 1024.0);
    let inputs = (0..cases)
        .map(|_| values.by_ref().take(input_dimension).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let shifts = (0..cases)
        .map(|_| values.by_ref().take(output_dimension).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let policy = PortfolioPolicy::default()
        .with_cholesky_damping(10.0_f64.powi(-9 + i32::from(bytes[3] % 10)))
        .expect("bounded damping must be valid");
    if let Ok(problem) = LeastSquaresProblem::new(inputs, shifts) {
        let report = solve_with_portfolio(&problem, &policy, &[])
            .expect("bounded finite problem must produce a typed report");
        report
            .exact_digest()
            .expect("portfolio report must authenticate");
        if report.status() == SolverStatus::Accepted {
            let selected = report
                .selected()
                .expect("accepted report must identify its candidate");
            let candidate = selected
                .candidate()
                .expect("selected evaluation must contain its candidate");
            assert_eq!(candidate.rows(), output_dimension);
            assert_eq!(candidate.columns(), input_dimension);
            assert_eq!(
                candidate.materialize_dense().unwrap().len(),
                output_dimension * input_dimension
            );
            candidate.exact_digest().unwrap();
            let metrics = selected
                .metrics()
                .expect("selected candidate must have recomputed metrics");
            assert!(metrics.absolute_residual().is_finite());
            assert!(metrics.frobenius_norm().is_finite());
        } else {
            assert!(report.selected().is_none());
        }
    }
});
