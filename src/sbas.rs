#![allow(clippy::needless_range_loop)]
use crate::contracts::DeltaObservation;
use crate::error::{BrainError, BrainResult};
use crate::linalg::{norm, solve, sub, Matrix};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct SbasResult {
    pub checkpoint_order: Vec<String>,
    pub potentials: Vec<Vec<f64>>,
    pub edge_residual_norms: Vec<f64>,
    pub cycle_rms: f64,
    pub max_edge_residual: f64,
}

pub fn reconstruct_trajectory(
    observations: &[DeltaObservation],
    ridge: f64,
) -> BrainResult<SbasResult> {
    if observations.is_empty() {
        return Err(BrainError::Invalid("sbas_no_edges".into()));
    }
    let dim = observations[0].delta.len();
    if dim == 0 || observations.iter().any(|o| o.delta.len() != dim) {
        return Err(BrainError::Invalid("sbas_dimension_mismatch".into()));
    }
    let mut nodes = BTreeSet::new();
    for o in observations {
        nodes.insert(o.from_checkpoint.clone());
        nodes.insert(o.to_checkpoint.clone());
    }
    let checkpoint_order = nodes.into_iter().collect::<Vec<_>>();
    if checkpoint_order.len() < 2 {
        return Err(BrainError::Invalid("sbas_need_two_nodes".into()));
    }
    let index = checkpoint_order
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect::<BTreeMap<_, _>>();
    let n = checkpoint_order.len();
    let mut adjacency = vec![Vec::<usize>::new(); n];
    for observation in observations {
        let from = index[observation.from_checkpoint.as_str()];
        let to = index[observation.to_checkpoint.as_str()];
        adjacency[from].push(to);
        adjacency[to].push(from);
    }
    let mut visited = vec![false; n];
    let mut stack = vec![0usize];
    visited[0] = true;
    while let Some(node) = stack.pop() {
        for &neighbor in &adjacency[node] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                stack.push(neighbor);
            }
        }
    }
    if visited.iter().any(|seen| !*seen) {
        return Err(BrainError::Invalid(
            "sbas_checkpoint_graph_disconnected".into(),
        ));
    }
    let anchor = 0usize;
    let mut lap = Matrix::zeros(n - 1, n - 1);
    let mut rhs = vec![vec![0.0; dim]; n - 1];
    for o in observations {
        let i = index[o.from_checkpoint.as_str()];
        let j = index[o.to_checkpoint.as_str()];
        let w = o.reliability.clamp(0.0, 1.0).max(1e-4);
        let ridx = |node: usize| {
            if node < anchor {
                Some(node)
            } else if node > anchor {
                Some(node - 1)
            } else {
                None
            }
        };
        if let Some(a) = ridx(i) {
            lap.data[a * (n - 1) + a] += w;
            for p in 0..dim {
                rhs[a][p] -= w * o.delta[p];
            }
        }
        if let Some(b) = ridx(j) {
            lap.data[b * (n - 1) + b] += w;
            for p in 0..dim {
                rhs[b][p] += w * o.delta[p];
            }
        }
        if let (Some(a), Some(b)) = (ridx(i), ridx(j)) {
            lap.data[a * (n - 1) + b] -= w;
            lap.data[b * (n - 1) + a] -= w;
        }
    }
    for i in 0..n - 1 {
        lap.data[i * (n - 1) + i] += ridge;
    }
    let mut potentials = vec![vec![0.0; dim]; n];
    for p in 0..dim {
        let b = (0..n - 1).map(|r| rhs[r][p]).collect::<Vec<_>>();
        let x = solve(lap.clone(), b)?;
        for node in 1..n {
            potentials[node][p] = x[node - 1];
        }
    }
    let mut edge_residual_norms = Vec::new();
    let mut sq = 0.0;
    let mut weight_sum = 0.0;
    let mut maxr = 0.0f64;
    for o in observations {
        let i = index[o.from_checkpoint.as_str()];
        let j = index[o.to_checkpoint.as_str()];
        let predicted = sub(&potentials[j], &potentials[i])?;
        let residual = sub(&predicted, &o.delta)?;
        let r = norm(&residual)? / (norm(&o.delta)? + 1e-12);
        let w = o.reliability.clamp(0.0, 1.0).max(1e-4);
        sq += w * r * r;
        weight_sum += w;
        maxr = maxr.max(r);
        edge_residual_norms.push(r);
    }
    Ok(SbasResult {
        checkpoint_order,
        potentials,
        edge_residual_norms,
        cycle_rms: (sq / weight_sum.max(1e-12)).sqrt(),
        max_edge_residual: maxr,
    })
}

#[cfg(test)]
mod autonomous_star_tests {
    use super::*;
    use crate::contracts::{DeltaObservation, ExperimentLineage};

    fn digest(value: u64) -> String {
        format!("{value:064x}")
    }
    fn observation(index: usize, delta: Vec<f64>) -> DeltaObservation {
        DeltaObservation {
            observation_id: format!("star-{index}"),
            from_checkpoint: "base:checkpoint".into(),
            to_checkpoint: format!("variant-{index}"),
            generation: index as u64 + 1,
            delta,
            functional_response: vec![0.0],
            confounders: vec![],
            reliability: 1.0,
            independence_group: format!("g-{index}"),
            experiment_lineage: ExperimentLineage {
                run_id: digest(100 + index as u64),
                replicate_id: format!("rep-{index}"),
                randomization_id: digest(200 + index as u64),
                dataset_split_digest: digest(300 + index as u64),
                initial_checkpoint_digest: digest(400),
                optimizer_config_digest: digest(500),
                template_config_digest: digest(600 + index as u64),
            },
            dense_artifact: None,
            parameter_layout_sha256: None,
            representation_artifact: None,
            representation_protocol_sha256: None,
            provenance_digest: digest(700 + index as u64),
        }
    }

    #[test]
    fn parallel_base_to_variant_edges_have_near_zero_cycle_residual() {
        let observations = vec![
            observation(0, vec![1.0, 0.0, 0.5]),
            observation(1, vec![0.0, 1.0, -0.25]),
            observation(2, vec![0.4, 0.2, 1.0]),
        ];
        let result = reconstruct_trajectory(&observations, 1e-6).unwrap();
        assert!(result.cycle_rms < 1e-4, "cycle={}", result.cycle_rms);
        assert!(
            result.max_edge_residual < 1e-4,
            "max={}",
            result.max_edge_residual
        );
    }
}
