#![allow(clippy::needless_range_loop)]

use crate::contracts::{DeltaObservation, SkillField};
use crate::error::{BrainError, BrainResult};
use crate::gauge::align_bases;
use crate::identity::{LineageId, ReconstructionId, SkillId};
use crate::linalg::{cosine, dot, norm, solve, symmetric_eigen_jacobi, Matrix};
use crate::validation::{independence_group_folds, validate_reliability};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct PersistentTomographyResult {
    pub fields: Vec<SkillField>,
    pub coefficients: Matrix,
    pub source_mixtures: Vec<Vec<f64>>,
    pub selected_rank: usize,
    pub effective_rank: f64,
    pub condition_estimate: f64,
    pub reconstruction_rms: f64,
    pub normalized_reconstruction_rms: f64,
    pub functional_cv_r2: f64,
    pub coherence_threshold: f64,
    pub coherence_gap: f64,
    pub coverage_ratio: f64,
    pub cluster_stability: f64,
    pub parametric_cluster_stability: f64,
    pub functional_cluster_stability: f64,
    pub cluster_identity_min_margin: f64,
    pub cluster_assignment_consistent: bool,
    pub min_holdout_similarity: f64,
    pub cluster_sizes: Vec<usize>,
}

#[derive(Debug, Clone)]
struct ClusterSolution {
    components: Vec<Vec<usize>>,
    threshold: f64,
    gap: f64,
    recurrence_coverage: f64,
}

#[derive(Debug, Clone)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] = self.rank[ra].saturating_add(1);
        }
    }
}

fn normalized_rows(d: &Matrix) -> BrainResult<Vec<Vec<f64>>> {
    if d.rows == 0 || d.cols == 0 {
        return Err(BrainError::Invalid("persistent_empty_matrix".into()));
    }
    let mut rows = Vec::with_capacity(d.rows);
    for r in 0..d.rows {
        let row = d.row(r);
        let n = norm(row)?;
        if !n.is_finite() || n <= 1e-15 {
            return Err(BrainError::Invalid(format!(
                "persistent_zero_or_invalid_row:{r}"
            )));
        }
        rows.push(row.iter().map(|v| v / n).collect());
    }
    Ok(rows)
}

fn components_at_threshold(scores: &Matrix, threshold: f64) -> Vec<Vec<usize>> {
    let mut uf = UnionFind::new(scores.rows);
    for i in 0..scores.rows {
        for j in 0..i {
            if scores.get(i, j) >= threshold {
                uf.union(i, j);
            }
        }
    }
    let mut grouped = BTreeMap::<usize, Vec<usize>>::new();
    for i in 0..scores.rows {
        let root = uf.find(i);
        grouped.entry(root).or_default().push(i);
    }
    let mut components = grouped.into_values().collect::<Vec<_>>();
    components.sort_by(|left, right| {
        left[0]
            .cmp(&right[0])
            .then_with(|| left.len().cmp(&right.len()))
    });
    components
}

fn clustering_silhouette(scores: &Matrix, components: &[Vec<usize>]) -> f64 {
    if components.len() < 2 || scores.rows == 0 {
        return -1.0;
    }
    let mut labels = vec![0usize; scores.rows];
    for (cluster_index, component) in components.iter().enumerate() {
        for &row in component {
            labels[row] = cluster_index;
        }
    }
    let mut total = 0.0;
    for row in 0..scores.rows {
        let own = &components[labels[row]];
        if own.len() == 1 {
            continue;
        }
        let intra = own
            .iter()
            .copied()
            .filter(|other| *other != row)
            .map(|other| 1.0 - scores.get(row, other))
            .sum::<f64>()
            / (own.len() - 1) as f64;
        let nearest_other = components
            .iter()
            .enumerate()
            .filter(|(cluster_index, _)| *cluster_index != labels[row])
            .map(|(_, component)| {
                component
                    .iter()
                    .map(|other| 1.0 - scores.get(row, *other))
                    .sum::<f64>()
                    / component.len() as f64
            })
            .fold(f64::INFINITY, f64::min);
        let denom = intra.max(nearest_other).max(1e-15);
        total += (nearest_other - intra) / denom;
    }
    total / scores.rows as f64
}

fn recurrence_metrics(components: &[Vec<usize>], groups: &[String]) -> (f64, f64, usize) {
    let all_groups = groups.iter().cloned().collect::<BTreeSet<_>>();
    let group_count = all_groups.len().max(1);
    let minimum_recurrence_groups = group_count;
    let mut recurrent_members = 0usize;
    let mut weighted_persistence = 0.0;
    let mut singletons = 0usize;
    for component in components {
        if component.len() == 1 {
            singletons += 1;
        }
        let seen = component
            .iter()
            .map(|index| groups[*index].clone())
            .collect::<BTreeSet<_>>();
        if seen.len() >= minimum_recurrence_groups {
            recurrent_members += component.len();
        }
        weighted_persistence += component.len() as f64 * seen.len() as f64 / group_count as f64;
    }
    (
        recurrent_members as f64 / groups.len().max(1) as f64,
        weighted_persistence / groups.len().max(1) as f64,
        singletons,
    )
}

type ClusterQuality = (f64, f64, f64, usize, f64, f64);

fn better_cluster_candidate(candidate: ClusterQuality, current: ClusterQuality) -> bool {
    // Lexicographic model selection with no task labels:
    // 1) maximize observations participating in cross-aperture recurrence;
    // 2) maximize geometric/functional silhouette;
    // 3) maximize average aperture persistence;
    // 4) minimize singleton fragments;
    // 5) prefer a larger actual score gap when structure is otherwise equal;
    // 6) then prefer the lower threshold to avoid arbitrary fragmentation.
    const EPS: f64 = 1e-12;
    if (candidate.0 - current.0).abs() > EPS {
        return candidate.0 > current.0;
    }
    if (candidate.1 - current.1).abs() > EPS {
        return candidate.1 > current.1;
    }
    if (candidate.2 - current.2).abs() > EPS {
        return candidate.2 > current.2;
    }
    if candidate.3 != current.3 {
        return candidate.3 < current.3;
    }
    if (candidate.4 - current.4).abs() > EPS {
        return candidate.4 > current.4;
    }
    candidate.5 < current.5
}

fn cluster_rows(rows: &[Vec<f64>], groups: &[String]) -> BrainResult<ClusterSolution> {
    if rows.len() < 2 {
        return Err(BrainError::Invalid(
            "persistent_cluster_minimum_rows".into(),
        ));
    }
    let dim = rows[0].len();
    if dim == 0 || rows.iter().any(|row| row.len() != dim) {
        return Err(BrainError::Invalid(
            "persistent_cluster_dimension_mismatch".into(),
        ));
    }
    if groups.len() != rows.len() || groups.iter().any(|group| group.trim().is_empty()) {
        return Err(BrainError::Invalid(
            "persistent_cluster_group_count_mismatch".into(),
        ));
    }
    let mut scores = Matrix::zeros(rows.len(), rows.len());
    let mut similarities = Vec::new();
    for i in 0..rows.len() {
        scores.set(i, i, 1.0);
        for j in 0..i {
            let score = cosine(&rows[i], &rows[j])?.abs().clamp(0.0, 1.0);
            scores.set(i, j, score);
            scores.set(j, i, score);
            similarities.push(score);
        }
    }
    if similarities.len() < 2 {
        return Err(BrainError::Invalid(
            "persistent_cluster_insufficient_pairs".into(),
        ));
    }
    similarities.sort_by(|a, b| b.total_cmp(a));
    let mut best: Option<(ClusterSolution, ClusterQuality)> = None;
    for index in 0..similarities.len() - 1 {
        let upper = similarities[index];
        let lower = similarities[index + 1];
        let gap = upper - lower;
        if !gap.is_finite() || gap <= 1e-12 {
            continue;
        }
        let threshold = 0.5 * (upper + lower);
        let components = components_at_threshold(&scores, threshold);
        let silhouette = clustering_silhouette(&scores, &components);
        let (recurrence_coverage, mean_persistence, singletons) =
            recurrence_metrics(&components, groups);
        let quality = (
            recurrence_coverage,
            silhouette,
            mean_persistence,
            singletons,
            gap,
            threshold,
        );
        if best
            .as_ref()
            .is_none_or(|(_, current_quality)| better_cluster_candidate(quality, *current_quality))
        {
            best = Some((
                ClusterSolution {
                    components,
                    threshold,
                    gap,
                    recurrence_coverage,
                },
                quality,
            ));
        }
    }
    best.map(|(solution, _)| solution).ok_or_else(|| {
        BrainError::Numerical("persistent_coherence_structure_not_identifiable".into())
    })
}

fn canonical_component_key(members: &[usize], observations: &[DeltaObservation]) -> Vec<String> {
    let mut keys = members
        .iter()
        .map(|index| {
            format!(
                "{}:{}",
                observations[*index].observation_id, observations[*index].provenance_digest
            )
        })
        .collect::<Vec<_>>();
    keys.sort();
    keys
}

fn component_identity(
    members: &[usize],
    observations: &[DeltaObservation],
) -> BrainResult<(String, ReconstructionId, LineageId)> {
    let keys = canonical_component_key(members, observations);
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:CAPABILITY-SUPPORT:v1\0");
    for key in &keys {
        hasher.update((key.len() as u64).to_be_bytes());
        hasher.update(key.as_bytes());
    }
    let digest = format!("{:x}", hasher.finalize());
    Ok((
        format!("cap-{}", &digest[..24]),
        ReconstructionId::parse(format!("recon-{digest}"))?,
        LineageId::parse(format!("lineage-{}", &digest[..32]))?,
    ))
}

fn centroid_for_members(
    rows: &[Vec<f64>],
    members: &[usize],
    reliability: &[f64],
) -> BrainResult<Vec<f64>> {
    if members.is_empty() {
        return Err(BrainError::Invalid("persistent_empty_cluster".into()));
    }
    let dim = rows[0].len();
    let reference = &rows[members[0]];
    let mut pre = vec![0.0; dim];
    let mut total_weight = 0.0;
    for &index in members {
        let sign = if cosine(reference, &rows[index])? >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let weight = validate_reliability(reliability[index], "persistent")?;
        total_weight += weight;
        for p in 0..dim {
            pre[p] += weight * sign * rows[index][p];
        }
    }
    if total_weight <= 1e-15 {
        return Err(BrainError::Numerical(
            "persistent_cluster_weight_zero".into(),
        ));
    }
    for value in &mut pre {
        *value /= total_weight;
    }
    let pre_norm = norm(&pre)?;
    if !pre_norm.is_finite() || pre_norm <= 1e-15 {
        return Err(BrainError::Numerical(
            "persistent_cluster_centroid_degenerate".into(),
        ));
    }
    Ok(pre.iter().map(|value| value / pre_norm).collect())
}

fn functional_dimension(observations: &[DeltaObservation]) -> BrainResult<usize> {
    let dim = observations
        .first()
        .map(|observation| observation.functional_response.len())
        .unwrap_or(0);
    if dim == 0
        || observations.iter().any(|observation| {
            observation.functional_response.len() != dim
                || observation
                    .functional_response
                    .iter()
                    .any(|value| !value.is_finite())
        })
    {
        return Err(BrainError::Invalid(
            "persistent_functional_dimension_invalid".into(),
        ));
    }
    Ok(dim)
}

fn weighted_functional_signature(
    rows: &[Vec<f64>],
    observations: &[DeltaObservation],
    members: &[usize],
) -> BrainResult<Vec<f64>> {
    let output_dim = functional_dimension(observations)?;
    if rows.len() != observations.len() || members.is_empty() {
        return Err(BrainError::Invalid(
            "persistent_functional_signature_shape".into(),
        ));
    }
    let reference = &rows[members[0]];
    let mut signature = vec![0.0; output_dim];
    let mut total = 0.0;
    for &index in members {
        let w = validate_reliability(observations[index].reliability, "persistent")?;
        let sign = if cosine(reference, &rows[index])? >= 0.0 {
            1.0
        } else {
            -1.0
        };
        total += w;
        for out in 0..output_dim {
            signature[out] += w * sign * observations[index].functional_response[out];
        }
    }
    if total <= 1e-15 {
        return Err(BrainError::Numerical(
            "persistent_functional_weight_zero".into(),
        ));
    }
    for value in &mut signature {
        *value /= total;
    }
    Ok(signature)
}

#[derive(Debug, Clone)]
struct CrossApertureValidation {
    functional_cv_r2: f64,
    cluster_stability: f64,
    parametric_cluster_stability: f64,
    functional_cluster_stability: f64,
    cluster_identity_min_margin: f64,
    cluster_assignment_consistent: bool,
    min_holdout_similarity: f64,
}

fn assignment_min_margin(
    reference: &[Vec<f64>],
    candidate: &[Vec<f64>],
    assignment: &[Option<usize>],
) -> BrainResult<f64> {
    if reference.len() != assignment.len() || candidate.is_empty() {
        return Err(BrainError::Invalid(
            "persistent_identity_margin_shape".into(),
        ));
    }
    let mut minimum = f64::INFINITY;
    for (reference_index, matched_candidate) in assignment.iter().enumerate() {
        let matched_candidate = matched_candidate.ok_or_else(|| {
            BrainError::Integrity("persistent_identity_assignment_incomplete".into())
        })?;
        let matched = cosine(&reference[reference_index], &candidate[matched_candidate])?.abs();
        let mut runner_up = 0.0_f64;
        for candidate_index in 0..candidate.len() {
            if candidate_index == matched_candidate {
                continue;
            }
            runner_up = runner_up
                .max(cosine(&reference[reference_index], &candidate[candidate_index])?.abs());
        }
        minimum = minimum.min(matched - runner_up);
    }
    if !minimum.is_finite() {
        return Err(BrainError::Numerical(
            "persistent_identity_margin_nonfinite".into(),
        ));
    }
    Ok(minimum)
}

fn cross_aperture_functional_cv(
    rows: &[Vec<f64>],
    observations: &[DeltaObservation],
    minimum_groups: usize,
) -> BrainResult<CrossApertureValidation> {
    let output_dim = functional_dimension(observations)?;
    let folds = independence_group_folds(observations, minimum_groups)?;
    let mut predictions = vec![vec![0.0; output_dim]; observations.len()];
    let mut predicted = vec![false; observations.len()];
    let mut fold_centroids = Vec::<Vec<Vec<f64>>>::new();
    let mut fold_signatures = Vec::<Vec<Vec<f64>>>::new();
    let mut min_holdout_similarity = f64::INFINITY;

    for fold in &folds {
        let train_indices = &fold.train;
        let test_indices = &fold.test;
        if train_indices.len() < 4 {
            return Err(BrainError::Invalid(
                "persistent_cv_fold_cardinality_invalid".into(),
            ));
        }
        let train_rows = train_indices
            .iter()
            .map(|&index| rows[index].clone())
            .collect::<Vec<_>>();
        let train_groups = train_indices
            .iter()
            .map(|&index| observations[index].independence_group.clone())
            .collect::<Vec<_>>();
        let solution = cluster_rows(&train_rows, &train_groups)?;
        let train_reliability = train_indices
            .iter()
            .map(|&index| observations[index].reliability)
            .collect::<Vec<_>>();
        let mut centroids = Vec::with_capacity(solution.components.len());
        let mut signatures = Vec::with_capacity(solution.components.len());
        for component in &solution.components {
            let centroid = centroid_for_members(&train_rows, component, &train_reliability)?;
            let original_members = component
                .iter()
                .map(|&local| train_indices[local])
                .collect::<Vec<_>>();
            centroids.push(centroid);
            signatures.push(weighted_functional_signature(
                rows,
                observations,
                &original_members,
            )?);
        }
        fold_centroids.push(centroids.clone());
        fold_signatures.push(signatures.clone());
        for &index in test_indices {
            let mut best = None::<(usize, f64)>;
            for (cluster_index, centroid) in centroids.iter().enumerate() {
                let similarity = cosine(&rows[index], centroid)?;
                if best.is_none_or(|(_, current)| similarity.abs() > current.abs()) {
                    best = Some((cluster_index, similarity));
                }
            }
            let (cluster_index, similarity) = best.ok_or_else(|| {
                BrainError::Numerical("persistent_cv_no_cluster_assignment".into())
            })?;
            min_holdout_similarity = min_holdout_similarity.min(similarity.abs());
            let sign = if similarity >= 0.0 { 1.0 } else { -1.0 };
            predictions[index] = signatures[cluster_index]
                .iter()
                .map(|value| sign * value)
                .collect();
            predicted[index] = true;
        }
    }
    if predicted.iter().any(|value| !*value) {
        return Err(BrainError::Integrity(
            "persistent_cv_incomplete_prediction_coverage".into(),
        ));
    }
    let means = (0..output_dim)
        .map(|out| {
            observations
                .iter()
                .map(|observation| observation.functional_response[out])
                .sum::<f64>()
                / observations.len() as f64
        })
        .collect::<Vec<_>>();
    let mut sse = 0.0;
    let mut sst = 0.0;
    for index in 0..observations.len() {
        for out in 0..output_dim {
            let actual = observations[index].functional_response[out];
            let error = actual - predictions[index][out];
            sse += error * error;
            let centered = actual - means[out];
            sst += centered * centered;
        }
    }
    let cv_r2 = if sst <= 1e-18 { 0.0 } else { 1.0 - sse / sst };
    let reference_centroids = fold_centroids
        .first()
        .ok_or_else(|| BrainError::Integrity("persistent_cv_no_fold_centroids".into()))?;
    let reference_signatures = fold_signatures
        .first()
        .ok_or_else(|| BrainError::Integrity("persistent_cv_no_fold_signatures".into()))?;
    let mut parametric_cluster_stability = 1.0_f64;
    let mut functional_cluster_stability = 1.0_f64;
    let mut cluster_identity_min_margin = f64::INFINITY;
    let mut cluster_assignment_consistent = true;
    for fold_index in 1..fold_centroids.len() {
        if fold_centroids[fold_index].len() != reference_centroids.len()
            || fold_signatures[fold_index].len() != reference_signatures.len()
        {
            cluster_assignment_consistent = false;
            parametric_cluster_stability = 0.0;
            functional_cluster_stability = 0.0;
            cluster_identity_min_margin = f64::NEG_INFINITY;
            break;
        }
        let parametric = align_bases(reference_centroids, &fold_centroids[fold_index])?;
        let functional = align_bases(reference_signatures, &fold_signatures[fold_index])?;
        cluster_assignment_consistent &= parametric.assignment == functional.assignment;
        parametric_cluster_stability = parametric_cluster_stability.min(parametric.mean_abs_cosine);
        functional_cluster_stability = functional_cluster_stability.min(functional.mean_abs_cosine);
        cluster_identity_min_margin = cluster_identity_min_margin
            .min(assignment_min_margin(
                reference_centroids,
                &fold_centroids[fold_index],
                &parametric.assignment,
            )?)
            .min(assignment_min_margin(
                reference_signatures,
                &fold_signatures[fold_index],
                &functional.assignment,
            )?);
    }
    let cluster_stability = parametric_cluster_stability.min(functional_cluster_stability);
    Ok(CrossApertureValidation {
        functional_cv_r2: cv_r2,
        cluster_stability,
        parametric_cluster_stability,
        functional_cluster_stability,
        cluster_identity_min_margin,
        cluster_assignment_consistent,
        min_holdout_similarity,
    })
}

fn effective_rank_from_energies(energies: &[f64]) -> f64 {
    let total = energies.iter().copied().sum::<f64>().max(1e-18);
    let entropy = energies
        .iter()
        .filter_map(|energy| {
            let probability = *energy / total;
            (probability > 1e-15).then_some(-probability * probability.ln())
        })
        .sum::<f64>();
    entropy.exp()
}

pub fn reconstruct_persistent_skill_fields(
    d: &Matrix,
    observations: &[DeltaObservation],
    generation: u64,
    minimum_observations: usize,
    minimum_groups: usize,
) -> BrainResult<PersistentTomographyResult> {
    d.validate("persistent_tomography_matrix")?;
    let mut observation_ids = BTreeSet::new();
    let mut provenance_digests = BTreeSet::new();
    if d.rows != observations.len()
        || d.rows < minimum_observations
        || d.cols == 0
        || minimum_observations < 2
        || minimum_groups < 2
        || observations.iter().any(|observation| {
            !observation_ids.insert(observation.observation_id.clone())
                || observation.independence_group.trim().is_empty()
                || validate_reliability(observation.reliability, "persistent").is_err()
                || !provenance_digests.insert(observation.provenance_digest.clone())
        })
    {
        return Err(BrainError::Invalid(
            "persistent_tomography_input_shape".into(),
        ));
    }
    let rows = normalized_rows(d)?;
    let clustering_groups = observations
        .iter()
        .map(|observation| observation.independence_group.clone())
        .collect::<Vec<_>>();
    let mut solution = cluster_rows(&rows, &clustering_groups)?;
    solution.components.sort_by(|left, right| {
        canonical_component_key(left, observations)
            .cmp(&canonical_component_key(right, observations))
    });
    let reliability = observations
        .iter()
        .map(|observation| observation.reliability)
        .collect::<Vec<_>>();
    let all_groups = observations
        .iter()
        .map(|observation| observation.independence_group.clone())
        .collect::<BTreeSet<_>>();
    let total_raw_energy = (0..d.rows).try_fold(0.0, |total, row| {
        Ok::<f64, BrainError>(total + reliability[row] * dot(d.row(row), d.row(row))?)
    })?;
    if total_raw_energy <= 1e-18 {
        return Err(BrainError::Numerical(
            "persistent_total_energy_degenerate".into(),
        ));
    }

    let mut fields = Vec::with_capacity(solution.components.len());
    let mut source_mixtures = Vec::with_capacity(solution.components.len());
    let mut energies = Vec::with_capacity(solution.components.len());

    for members in &solution.components {
        let direction = centroid_for_members(&rows, members, &reliability)?;
        // `direction` is deliberately unit-normalized for geometry, but the
        // materialized SkillField must retain a real update amplitude.  A
        // source mixture that recreates the unit vector would shrink a full
        // dense LoRA by roughly its source norm.  Materialization therefore
        // uses the reliability-weighted aligned mean of the original deltas;
        // geometry and actuator amplitude are separate contracts.
        let reference = &rows[members[0]];
        let total_materialization_weight = members
            .iter()
            .map(|&member| reliability[member])
            .sum::<f64>();
        let mut original_mixture = vec![0.0; d.rows];
        for &member in members {
            let sign = if cosine(reference, &rows[member])? >= 0.0 {
                1.0
            } else {
                -1.0
            };
            original_mixture[member] = sign * reliability[member] / total_materialization_weight;
        }
        let signature = weighted_functional_signature(&rows, observations, members)?;
        let cluster_energy = members.iter().try_fold(0.0, |total, &row| {
            Ok::<f64, BrainError>(total + reliability[row] * dot(d.row(row), d.row(row))?)
        })?;
        if cluster_energy <= 1e-18 {
            return Err(BrainError::Numerical(
                "persistent_cluster_energy_degenerate".into(),
            ));
        }
        energies.push(cluster_energy);
        let support_weight = members.iter().map(|&row| reliability[row]).sum::<f64>();
        let coherence = members.iter().try_fold(0.0, |total, &row| {
            Ok::<f64, BrainError>(total + reliability[row] * cosine(&rows[row], &direction)?.abs())
        })? / support_weight;
        let angular_variance = members.iter().try_fold(0.0, |total, &row| {
            let residual = 1.0 - cosine(&rows[row], &direction)?.abs().clamp(0.0, 1.0);
            Ok::<f64, BrainError>(total + reliability[row] * residual * residual)
        })? / support_weight;
        let seen_groups = members
            .iter()
            .map(|&row| observations[row].independence_group.clone())
            .collect::<BTreeSet<_>>();
        let persistence = seen_groups.len() as f64 / all_groups.len().max(1) as f64;
        let (skill_id, reconstruction_id, lineage_id) = component_identity(members, observations)?;
        fields.push(SkillField {
            skill_id: SkillId::parse(skill_id)?,
            reconstruction_id,
            lineage_id,
            generation_created: generation,
            direction,
            structured_geometry: None,
            dense_materialization: None,
            parameter_layout_sha256: None,
            representation_signature: Vec::new(),
            singular_value: cluster_energy.sqrt(),
            explained_variance: (cluster_energy / total_raw_energy).clamp(0.0, 1.0),
            persistence: persistence.clamp(0.0, 1.0),
            coherence: coherence.clamp(0.0, 1.0),
            uncertainty: angular_variance.sqrt() / (members.len() as f64).sqrt().max(1.0),
            evidence_support_digests: Vec::new(),
            support: members.len(),
            functional_signature: signature,
            parent_skill_ids: Vec::new(),
        });
        source_mixtures.push(original_mixture);
    }

    let directions = fields
        .iter()
        .map(|field| field.direction.clone())
        .collect::<Vec<_>>();
    let mut field_gram = Matrix::zeros(directions.len(), directions.len());
    for left in 0..directions.len() {
        for right in left..directions.len() {
            let value = dot(&directions[left], &directions[right])?;
            field_gram.set(left, right, value);
            field_gram.set(right, left, value);
        }
    }
    let geometry_eigenpairs = symmetric_eigen_jacobi(
        &field_gram,
        1e-12,
        field_gram
            .rows
            .saturating_mul(field_gram.rows)
            .saturating_mul(100),
    )?;
    if geometry_eigenpairs.len() != directions.len() {
        return Err(BrainError::Numerical(
            "persistent_field_geometry_rank_deficient".into(),
        ));
    }
    let largest_geometry_eigenvalue = geometry_eigenpairs[0].0;
    let smallest_geometry_eigenvalue = geometry_eigenpairs
        .last()
        .map(|(value, _)| *value)
        .ok_or_else(|| BrainError::Numerical("persistent_field_geometry_empty".into()))?;
    if smallest_geometry_eigenvalue <= 0.0 {
        return Err(BrainError::Numerical(
            "persistent_field_geometry_rank_deficient".into(),
        ));
    }
    let condition_estimate = largest_geometry_eigenvalue / smallest_geometry_eigenvalue;
    let mut coefficients = Matrix::zeros(d.rows, directions.len());
    let mut residual_energy = 0.0;
    let mut input_energy = 0.0;
    for row in 0..d.rows {
        let right_hand_side = directions
            .iter()
            .map(|direction| dot(d.row(row), direction))
            .collect::<BrainResult<Vec<_>>>()?;
        let row_coefficients = solve(field_gram.clone(), right_hand_side)?;
        let mut reconstruction = vec![0.0; d.cols];
        for (field_index, direction) in directions.iter().enumerate() {
            let coefficient = row_coefficients[field_index];
            coefficients.set(row, field_index, coefficient);
            for parameter in 0..d.cols {
                reconstruction[parameter] += coefficient * direction[parameter];
            }
        }
        for parameter in 0..d.cols {
            let value = d.get(row, parameter);
            let error = value - reconstruction[parameter];
            residual_energy += error * error;
            input_energy += value * value;
        }
    }
    if !residual_energy.is_finite() || !input_energy.is_finite() || input_energy <= 0.0 {
        return Err(BrainError::Numerical(
            "persistent_reconstruction_energy_invalid".into(),
        ));
    }
    let reconstruction_rms = (residual_energy / (d.rows * d.cols).max(1) as f64).sqrt();
    let normalized_reconstruction_rms = (residual_energy / input_energy).sqrt();
    let cross_validation = cross_aperture_functional_cv(&rows, observations, minimum_groups)?;

    Ok(PersistentTomographyResult {
        selected_rank: fields.len(),
        effective_rank: effective_rank_from_energies(&energies),
        condition_estimate,
        reconstruction_rms,
        normalized_reconstruction_rms,
        functional_cv_r2: cross_validation.functional_cv_r2,
        coherence_threshold: solution.threshold,
        coherence_gap: solution.gap,
        coverage_ratio: solution.recurrence_coverage,
        cluster_stability: cross_validation.cluster_stability,
        parametric_cluster_stability: cross_validation.parametric_cluster_stability,
        functional_cluster_stability: cross_validation.functional_cluster_stability,
        cluster_identity_min_margin: cross_validation.cluster_identity_min_margin,
        cluster_assignment_consistent: cross_validation.cluster_assignment_consistent,
        min_holdout_similarity: cross_validation.min_holdout_similarity,
        cluster_sizes: solution.components.iter().map(Vec::len).collect(),
        fields,
        coefficients,
        source_mixtures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::ConfounderValue;
    use crate::linalg::normalize;

    fn observation(
        index: usize,
        aperture: usize,
        delta: Vec<f64>,
        response: Vec<f64>,
    ) -> DeltaObservation {
        DeltaObservation {
            observation_id: crate::identity::ObservationId::parse(format!("obs-{index}")).unwrap(),
            from_checkpoint: "base".into(),
            to_checkpoint: format!("candidate-{index}"),
            generation: index as u64 + 1,
            delta,
            functional_response: response,
            confounders: vec![ConfounderValue {
                name: "aperture".into(),
                value: aperture as f64,
            }],
            reliability: 1.0,
            independence_group: format!("aperture-{aperture}"),
            experiment_lineage: Default::default(),
            dense_artifact: None,
            parameter_layout_sha256: None,
            representation_artifact: None,
            representation_protocol_sha256: None,
            provenance_digest: crate::digest::ProvenanceDigest::from(
                crate::digest::Sha256Digest::parse(format!("{index:064x}")).unwrap(),
            ),
        }
    }

    #[test]
    fn structural_threshold_selection_avoids_largest_gap_fragmentation() {
        let g = 0.3_f64.sqrt();
        let skill = 0.2_f64.sqrt();
        let pair = 0.4_f64.sqrt();
        let unique_pair = 0.1_f64.sqrt();
        let unique_single = 0.5_f64.sqrt();
        let rows = vec![
            // Skill A: first two runs are extremely close (.9), third is only .5.
            vec![g, skill, 0.0, pair, 0.0, unique_pair, 0.0, 0.0, 0.0],
            vec![g, skill, 0.0, pair, 0.0, 0.0, unique_pair, 0.0, 0.0],
            vec![g, skill, 0.0, 0.0, 0.0, 0.0, 0.0, unique_single, 0.0],
            // Skill B has the same internal geometry in an orthogonal skill/pair axis.
            vec![g, 0.0, skill, 0.0, pair, unique_pair, 0.0, 0.0, 0.0],
            vec![g, 0.0, skill, 0.0, pair, 0.0, unique_pair, 0.0, 0.0],
            vec![g, 0.0, skill, 0.0, 0.0, 0.0, 0.0, 0.0, unique_single],
        ];
        let groups = vec![
            "g0".into(),
            "g1".into(),
            "g2".into(),
            "g0".into(),
            "g1".into(),
            "g2".into(),
        ];
        let solution = cluster_rows(&rows, &groups).unwrap();
        assert_eq!(solution.components.len(), 2);
        assert_eq!(
            solution.components.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![3, 3]
        );
        assert!(solution.threshold > 0.3 && solution.threshold < 0.5);
    }

    #[test]
    fn persistent_inverse_discovers_recurrent_skills_without_task_labels() {
        let skill_a = normalize(&[1.0, 0.2, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let skill_b = normalize(&[0.0, 0.0, 1.0, 0.2, 0.0, 0.0]).unwrap();
        let skill_c = normalize(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.2]).unwrap();
        let mut observations = Vec::new();
        let mut index = 1usize;
        for aperture in 0..3 {
            for (skill, response) in [
                (&skill_a, vec![1.0, 0.0, 0.1]),
                (&skill_b, vec![0.0, 1.0, 0.2]),
                (&skill_c, vec![0.1, 0.2, 1.0]),
            ] {
                let mut delta = skill.clone();
                for (parameter, value) in delta.iter_mut().enumerate() {
                    *value += 0.03
                        * (((aperture + 1) * (parameter + 2) * (index + 3)) as f64 * 0.37).sin();
                }
                observations.push(observation(index, aperture, delta, response));
                index += 1;
            }
        }
        let d = Matrix::from_rows(
            &observations
                .iter()
                .map(|observation| observation.delta.clone())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let result = reconstruct_persistent_skill_fields(&d, &observations, 9, 6, 3).unwrap();
        assert_eq!(result.selected_rank, 3);
        assert!(
            result.functional_cv_r2 > 0.95,
            "{}",
            result.functional_cv_r2
        );
        assert!(
            result.cluster_stability > 0.99,
            "{}",
            result.cluster_stability
        );
        assert_eq!(result.coverage_ratio, 1.0);
        assert!(result.fields.iter().all(|field| field.persistence == 1.0));
        assert!(result.fields.iter().all(|field| field.coherence > 0.95));
        for mixture in &result.source_mixtures {
            let nonzero = mixture.iter().filter(|value| value.abs() > 1e-12).count();
            assert_eq!(nonzero, 3);
            assert!((mixture.iter().map(|value| value.abs()).sum::<f64>() - 1.0).abs() < 1e-12);
        }

        let mut altered_outcomes = observations.clone();
        for (index, observation) in altered_outcomes.iter_mut().enumerate() {
            for value in &mut observation.functional_response {
                *value = -*value + index as f64 * 0.17;
            }
        }
        let altered = reconstruct_persistent_skill_fields(&d, &altered_outcomes, 9, 6, 3).unwrap();
        assert_eq!(result.cluster_sizes, altered.cluster_sizes);
        assert_eq!(
            result
                .fields
                .iter()
                .map(|field| field.skill_id.clone())
                .collect::<Vec<_>>(),
            altered
                .fields
                .iter()
                .map(|field| field.skill_id.clone())
                .collect::<Vec<_>>()
        );
        for (left, right) in result.fields.iter().zip(&altered.fields) {
            assert!(cosine(&left.direction, &right.direction).unwrap().abs() > 1.0 - 1e-12);
        }

        let mut invalid_reliability = observations.clone();
        invalid_reliability[0].reliability = 0.0;
        assert!(reconstruct_persistent_skill_fields(&d, &invalid_reliability, 9, 6, 3).is_err());
    }

    #[test]
    fn persistent_inverse_solves_nonorthogonal_field_coefficients_jointly() {
        let first = vec![1.0, 0.0];
        let second = vec![0.5, 0.75_f64.sqrt()];
        let mut observations = Vec::new();
        let mut index = 1usize;
        for aperture in 0..3 {
            observations.push(observation(index, aperture, first.clone(), vec![1.0, 0.0]));
            index += 1;
            observations.push(observation(index, aperture, second.clone(), vec![0.0, 1.0]));
            index += 1;
        }
        let matrix = Matrix::from_rows(
            &observations
                .iter()
                .map(|observation| observation.delta.clone())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let result = reconstruct_persistent_skill_fields(&matrix, &observations, 1, 6, 3).unwrap();
        assert_eq!(result.selected_rank, 2);
        assert!(
            result.reconstruction_rms < 1e-12,
            "{}",
            result.reconstruction_rms
        );
        assert!((result.condition_estimate - 3.0).abs() < 1e-10);
        for row in 0..result.coefficients.row_count() {
            let material = result
                .coefficients
                .row(row)
                .iter()
                .filter(|coefficient| coefficient.abs() > 1e-10)
                .count();
            assert_eq!(material, 1);
        }
    }
}
