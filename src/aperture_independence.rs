#![allow(clippy::needless_range_loop)]

use crate::contracts::DeltaObservation;
use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::linalg::{symmetric_eigen_jacobi, Matrix};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApertureIndependenceReport {
    pub schema: String,
    pub declared_groups: usize,
    pub design_feature_count: usize,
    pub active_design_features: usize,
    pub effective_group_rank: f64,
    pub effective_independent_groups: f64,
    pub numerical_design_rank: usize,
    pub unique_randomization_count: usize,
    pub unique_replicate_count: usize,
    pub lineage_verified: bool,
    pub max_cross_group_similarity: f64,
    pub mean_cross_group_similarity: f64,
    pub group_similarity_matrix: Vec<Vec<f64>>,
    pub group_ids: Vec<String>,
    pub group_independence_weights: Vec<f64>,
    pub minimum_required_groups: usize,
    pub independent_enough: bool,
}

fn confounder_names(observations: &[DeltaObservation]) -> Vec<String> {
    observations
        .iter()
        .flat_map(|observation| observation.confounders.iter().map(|item| item.name.clone()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn group_design_profiles(
    observations: &[DeltaObservation],
) -> BrainResult<(Vec<String>, Vec<Vec<f64>>)> {
    if observations.is_empty() {
        return Err(BrainError::Invalid(
            "aperture_independence_observations_required".into(),
        ));
    }
    let names = confounder_names(observations);
    let mut members = BTreeMap::<String, Vec<usize>>::new();
    for (index, observation) in observations.iter().enumerate() {
        if observation.independence_group.trim().is_empty() {
            return Err(BrainError::Invalid(
                "aperture_independence_group_required".into(),
            ));
        }
        members
            .entry(observation.independence_group.clone())
            .or_default()
            .push(index);
    }

    let mut group_ids = Vec::with_capacity(members.len());
    let mut profiles = Vec::with_capacity(members.len());
    for (group, rows) in members {
        let mut profile = Vec::with_capacity(names.len() * 3);
        for name in &names {
            let mut values = Vec::<(f64, f64)>::new();
            let mut present = 0usize;
            for &row in &rows {
                if let Some(item) = observations[row]
                    .confounders
                    .iter()
                    .find(|item| item.name == *name)
                {
                    if !item.value.is_finite() {
                        return Err(BrainError::Invalid(
                            "aperture_independence_non_finite_confounder".into(),
                        ));
                    }
                    present += 1;
                    values.push((item.value, observations[row].reliability.clamp(1e-4, 1.0)));
                }
            }
            let presence_fraction = present as f64 / rows.len().max(1) as f64;
            if values.is_empty() {
                profile.extend([0.0, 0.0, presence_fraction]);
                continue;
            }
            let total_weight = values
                .iter()
                .map(|(_, weight)| *weight)
                .sum::<f64>()
                .max(1e-15);
            let mean = values
                .iter()
                .map(|(value, weight)| value * weight)
                .sum::<f64>()
                / total_weight;
            let variance = values
                .iter()
                .map(|(value, weight)| weight * (value - mean).powi(2))
                .sum::<f64>()
                / total_weight;
            profile.extend([mean, variance, presence_fraction]);
        }
        group_ids.push(group);
        profiles.push(profile);
    }
    Ok((group_ids, profiles))
}

fn standardized_profiles(profiles: &[Vec<f64>]) -> BrainResult<(Vec<Vec<f64>>, usize)> {
    if profiles.is_empty() {
        return Err(BrainError::Invalid(
            "aperture_independence_profiles_required".into(),
        ));
    }
    if profiles[0].is_empty() {
        return Ok((vec![Vec::new(); profiles.len()], 0));
    }
    let cols = profiles[0].len();
    if profiles
        .iter()
        .any(|row| row.len() != cols || row.iter().any(|value| !value.is_finite()))
    {
        return Err(BrainError::Invalid(
            "aperture_independence_profile_shape".into(),
        ));
    }
    let mut active = Vec::new();
    let mut means = vec![0.0; cols];
    let mut stds = vec![0.0; cols];
    for col in 0..cols {
        means[col] = profiles.iter().map(|row| row[col]).sum::<f64>() / profiles.len() as f64;
        let variance = profiles
            .iter()
            .map(|row| (row[col] - means[col]).powi(2))
            .sum::<f64>()
            / profiles.len() as f64;
        stds[col] = variance.sqrt();
        let scale = profiles
            .iter()
            .map(|row| row[col].abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        if stds[col] > f64::EPSILON.sqrt() * scale {
            active.push(col);
        }
    }
    let standardized = profiles
        .iter()
        .map(|row| {
            active
                .iter()
                .map(|&col| (row[col] - means[col]) / stds[col].max(1e-15))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok((standardized, active.len()))
}

fn design_numerical_rank(profiles: &[Vec<f64>]) -> BrainResult<usize> {
    if profiles.is_empty() || profiles[0].is_empty() {
        return Ok(0);
    }
    let matrix = Matrix::from_rows(profiles)?;
    let gram = matrix.matmul(&matrix.transpose())?;
    let eigs = symmetric_eigen_jacobi(&gram, 1e-12, gram.rows * gram.rows * 100)?;
    let largest = eigs
        .first()
        .map(|(value, _)| *value)
        .unwrap_or(0.0)
        .max(1e-18);
    let tolerance = largest * f64::EPSILON.sqrt() * gram.rows.max(1) as f64;
    Ok(eigs.iter().filter(|(value, _)| *value > tolerance).count())
}

fn valid_digest(value: &str) -> bool {
    Sha256Digest::is_valid_str(value)
}

#[derive(Debug, Clone)]
struct GroupLineage {
    group_id: String,
    randomization_id: String,
    replicate_id: String,
}

fn group_lineages(observations: &[DeltaObservation]) -> BrainResult<Vec<GroupLineage>> {
    let mut grouped = BTreeMap::<String, Vec<&DeltaObservation>>::new();
    let mut run_ids = BTreeSet::new();
    for observation in observations {
        let lineage = &observation.experiment_lineage;
        if lineage.run_id.trim().is_empty()
            || lineage.replicate_id.trim().is_empty()
            || lineage.randomization_id.trim().is_empty()
            || !valid_digest(&lineage.dataset_split_digest)
            || !valid_digest(&lineage.initial_checkpoint_digest)
            || !valid_digest(&lineage.optimizer_config_digest)
            || !valid_digest(&lineage.template_config_digest)
            || !run_ids.insert(lineage.run_id.clone())
        {
            return Err(BrainError::Invalid(
                "aperture_experiment_lineage_invalid".into(),
            ));
        }
        grouped
            .entry(observation.independence_group.clone())
            .or_default()
            .push(observation);
    }
    let mut result = Vec::with_capacity(grouped.len());
    for (group_id, rows) in grouped {
        let randomizations = rows
            .iter()
            .map(|row| row.experiment_lineage.randomization_id.clone())
            .collect::<BTreeSet<_>>();
        let replicates = rows
            .iter()
            .map(|row| row.experiment_lineage.replicate_id.clone())
            .collect::<BTreeSet<_>>();
        if randomizations.len() != 1 || replicates.len() != 1 {
            return Err(BrainError::Invalid(
                "aperture_group_lineage_inconsistent".into(),
            ));
        }
        result.push(GroupLineage {
            group_id,
            randomization_id: randomizations.into_iter().next().unwrap(),
            replicate_id: replicates.into_iter().next().unwrap(),
        });
    }
    Ok(result)
}

pub fn estimate_aperture_independence(
    observations: &[DeltaObservation],
    minimum_required_groups: usize,
) -> BrainResult<ApertureIndependenceReport> {
    if minimum_required_groups < 2 {
        return Err(BrainError::Invalid(
            "aperture_minimum_required_groups_invalid".into(),
        ));
    }
    let lineages = group_lineages(observations)?;
    let (profile_group_ids, raw_profiles) = group_design_profiles(observations)?;
    let design_feature_count = raw_profiles.first().map(Vec::len).unwrap_or(0);
    let (profiles, active_design_features) = standardized_profiles(&raw_profiles)?;
    let numerical_design_rank = design_numerical_rank(&profiles)?;
    let group_ids = lineages
        .iter()
        .map(|lineage| lineage.group_id.clone())
        .collect::<Vec<_>>();
    if group_ids != profile_group_ids {
        return Err(BrainError::Integrity(
            "aperture_lineage_profile_group_order_mismatch".into(),
        ));
    }

    let mut matrix = Matrix::zeros(lineages.len(), lineages.len());
    for i in 0..lineages.len() {
        matrix.set(i, i, 1.0);
        for j in 0..i {
            // Redundancy is categorical: reusing the same randomization does
            // not become more independent because a seed integer is farther away.
            let same_randomization = lineages[i].randomization_id == lineages[j].randomization_id;
            let similarity = if same_randomization { 1.0 } else { 0.0 };
            matrix.set(i, j, similarity);
            matrix.set(j, i, similarity);
        }
    }
    let mut cross = Vec::new();
    for i in 0..matrix.rows {
        for j in 0..i {
            cross.push(matrix.get(i, j));
        }
    }
    let max_cross_group_similarity = cross.iter().copied().fold(0.0_f64, f64::max);
    let mean_cross_group_similarity = if cross.is_empty() {
        0.0
    } else {
        cross.iter().sum::<f64>() / cross.len() as f64
    };
    let group_independence_weights = (0..matrix.rows)
        .map(|row| {
            let redundancy = (0..matrix.cols)
                .filter(|col| matrix.get(row, *col) > 0.5)
                .count()
                .max(1);
            1.0 / redundancy as f64
        })
        .collect::<Vec<_>>();
    let effective_independent_groups = group_independence_weights.iter().sum::<f64>();
    let unique_randomization_count = lineages
        .iter()
        .map(|lineage| lineage.randomization_id.clone())
        .collect::<BTreeSet<_>>()
        .len();
    let unique_replicate_count = lineages
        .iter()
        .map(|lineage| lineage.replicate_id.clone())
        .collect::<BTreeSet<_>>()
        .len();
    let lineage_verified =
        unique_randomization_count == lineages.len() && unique_replicate_count == lineages.len();

    let eigs = symmetric_eigen_jacobi(&matrix, 1e-12, matrix.rows * matrix.rows * 100)?;
    let eigenvalues = eigs
        .iter()
        .map(|(value, _)| value.max(0.0))
        .collect::<Vec<_>>();
    let total = eigenvalues.iter().sum::<f64>();
    let square_sum = eigenvalues.iter().map(|value| value * value).sum::<f64>();
    let effective_group_rank = if total <= 1e-18 || square_sum <= 1e-18 {
        0.0
    } else {
        total * total / square_sum
    };
    let declared_groups = group_ids.len();
    let independent_enough = declared_groups >= minimum_required_groups
        && lineage_verified
        && effective_independent_groups >= minimum_required_groups as f64;
    let group_similarity_matrix = (0..matrix.rows)
        .map(|row| matrix.row_vec(row))
        .collect::<Vec<_>>();
    Ok(ApertureIndependenceReport {
        schema: "cerebro.tidex.aperture_independence/v4".into(),
        declared_groups,
        design_feature_count,
        active_design_features,
        effective_group_rank,
        effective_independent_groups,
        numerical_design_rank,
        unique_randomization_count,
        unique_replicate_count,
        lineage_verified,
        max_cross_group_similarity,
        mean_cross_group_similarity,
        group_similarity_matrix,
        group_ids,
        group_independence_weights,
        minimum_required_groups,
        independent_enough,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::ConfounderValue;

    fn obs(index: usize, group: &str, axis: f64) -> DeltaObservation {
        DeltaObservation {
            observation_id: format!("o{index}"),
            from_checkpoint: "a".into(),
            to_checkpoint: format!("b{index}"),
            generation: 1,
            delta: vec![1.0, 0.0],
            functional_response: vec![0.0],
            confounders: vec![ConfounderValue {
                name: "design_axis".into(),
                value: axis,
            }],
            reliability: 1.0,
            independence_group: group.into(),
            experiment_lineage: crate::contracts::ExperimentLineage {
                run_id: format!("run-{index}"),
                replicate_id: group.into(),
                randomization_id: format!("randomization-{axis:.6}"),
                dataset_split_digest: format!("{:064x}", 1),
                initial_checkpoint_digest: format!("{:064x}", 2),
                optimizer_config_digest: format!("{:064x}", 3),
                template_config_digest: format!("{:064x}", 4),
            },
            dense_artifact: None,
            parameter_layout_sha256: None,
            representation_artifact: None,
            representation_protocol_sha256: None,
            provenance_digest: format!("{index:064x}"),
        }
    }

    #[test]
    fn identical_design_labels_count_as_one_effective_experiment() {
        let observations = vec![obs(1, "a", 1.0), obs(2, "b", 1.0), obs(3, "c", 1.0)];
        let report = estimate_aperture_independence(&observations, 3).unwrap();
        assert_eq!(report.declared_groups, 3);
        assert!((report.effective_independent_groups - 1.0).abs() < 1e-12);
        assert!(!report.independent_enough);
    }

    #[test]
    fn many_well_separated_designs_exceed_three_effective_apertures() {
        let observations = vec![
            obs(1, "a", -4.0),
            obs(2, "b", -2.0),
            obs(3, "c", 0.0),
            obs(4, "d", 2.0),
            obs(5, "e", 4.0),
        ];
        let report = estimate_aperture_independence(&observations, 3).unwrap();
        assert!(
            report.effective_independent_groups >= 3.0,
            "{}",
            report.effective_independent_groups
        );
        assert!(report.independent_enough);
    }
}
