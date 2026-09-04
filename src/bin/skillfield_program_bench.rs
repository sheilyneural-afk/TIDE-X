use cerebro_tidex::contracts::SkillField;
use cerebro_tidex::parametric_program::{
    apply_parametric_transition, compile_operator_to_fields, compose_skill_fields,
    task_arithmetic_merge, ties_merge,
};
use serde_json::json;

fn field(id: &str, direction: Vec<f64>) -> SkillField {
    SkillField {
        skill_id: id.into(),
        reconstruction_id: String::new(),
        lineage_id: String::new(),
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
        support: 3,
        functional_signature: Vec::new(),
        parent_skill_ids: Vec::new(),
    }
}

fn fields() -> Vec<SkillField> {
    vec![
        field("rotated-hold", vec![1.0, 0.35, 0.35, 1.0]),
        field("rotated-toggle", vec![0.25, 1.0, 1.0, 0.25]),
    ]
}

fn operators() -> Vec<Vec<f64>> {
    vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0]]
}

fn parity(tokens: &[usize]) -> usize {
    tokens.iter().filter(|token| **token == 1).count() & 1
}

fn sequences(max_len: usize) -> Vec<Vec<usize>> {
    let mut cases = Vec::new();
    let exhaustive = max_len.min(10);
    for len in 1..=exhaustive {
        for mask in 0usize..(1usize << len) {
            cases.push((0..len).map(|bit| (mask >> bit) & 1).collect());
        }
    }
    let mut rng = 0xD1B5_4A32_D192_ED03u64;
    for len in exhaustive + 1..=max_len {
        for _ in 0..16 {
            let mut row = Vec::with_capacity(len);
            for _ in 0..len {
                rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                row.push(((rng >> 63) & 1) as usize);
            }
            cases.push(row);
        }
    }
    cases
}

fn final_index(state: &[f64]) -> usize {
    state
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn static_execute(operator: &[f64], steps: usize) -> usize {
    let mut state = vec![1.0, 0.0];
    for _ in 0..steps {
        state = apply_parametric_transition(&state, operator, 2).unwrap();
    }
    final_index(&state)
}

fn compiled_dynamic(tokens: &[usize], fields: &[SkillField], operators: &[Vec<f64>]) -> usize {
    let compiled = operators
        .iter()
        .map(|operator| compile_operator_to_fields(fields, operator, 1e-12, 1e-8).unwrap())
        .collect::<Vec<_>>();
    let mut state = vec![1.0, 0.0];
    for token in tokens {
        // Token selection exists only inside this synthetic benchmark oracle;
        // there is no token table in the production controller module.
        let operator = compose_skill_fields(fields, &compiled[*token].coefficients).unwrap();
        state = apply_parametric_transition(&state, &operator, 2).unwrap();
    }
    final_index(&state)
}

fn oracle_dynamic(tokens: &[usize], operators: &[Vec<f64>]) -> usize {
    let mut state = vec![1.0, 0.0];
    for token in tokens {
        state = apply_parametric_transition(&state, &operators[*token], 2).unwrap();
    }
    final_index(&state)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fields = fields();
    let operators = operators();
    let task = task_arithmetic_merge(&operators, 0.5)?;
    let ties = ties_merge(&operators, 1.0)?;
    let cases = sequences(64);
    let mut compiled_correct = 0usize;
    let mut oracle_correct = 0usize;
    let mut task_correct = 0usize;
    let mut ties_correct = 0usize;
    for tokens in &cases {
        let expected = parity(tokens);
        compiled_correct += usize::from(compiled_dynamic(tokens, &fields, &operators) == expected);
        oracle_correct += usize::from(oracle_dynamic(tokens, &operators) == expected);
        task_correct += usize::from(static_execute(&task, tokens.len()) == expected);
        ties_correct += usize::from(static_execute(&ties, tokens.len()) == expected);
    }
    let n = cases.len() as f64;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.skillfield_operator_compilation_benchmark/v3",
            "total_cases":cases.len(),
            "max_sequence_len":64,
            "compiled_skillfield_dynamic_accuracy":compiled_correct as f64/n,
            "oracle_dynamic_accuracy":oracle_correct as f64/n,
            "task_arithmetic_static_accuracy":task_correct as f64/n,
            "ties_static_accuracy":ties_correct as f64/n,
            "production_token_controller_removed":true,
            "interpretation":{
                "compiled_matches_equivalent_dynamic_oracle":compiled_correct==oracle_correct,
                "token_selection_is_benchmark_only":true
            }
        }))?
    );
    Ok(())
}
