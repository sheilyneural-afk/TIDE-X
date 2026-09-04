#![allow(clippy::needless_range_loop)]
use crate::contracts::{BrainConfig, SkillBank, SkillField};
use crate::error::{BrainError, BrainResult};
use crate::gauge::align_bases;
use crate::linalg::{
    cosine, dot, median, normalize, symmetric_eigen_jacobi, weighted_row_gram, Matrix,
};
use crate::validation::{choose_energy_rank, effective_rank_from_spectrum};

#[derive(Debug, Clone)]
pub struct TomographyResult {
    pub fields: Vec<SkillField>,
    pub coefficients: Matrix,
    pub reconstruction_rms: f64,
    pub effective_rank: f64,
    pub selected_rank: usize,
    pub condition_estimate: f64,
    pub robust_weights: Vec<f64>,
    /// For each field k, coefficients over residualized observations such that
    /// h_k = sum_r source_mixtures[k][r] * residual_delta_r.
    pub source_mixtures: Vec<Vec<f64>>,
}

fn reconstruct(d: &Matrix, fields: &[Vec<f64>]) -> BrainResult<(Matrix, Matrix, f64)> {
    let mut coeff = Matrix::zeros(d.rows, fields.len());
    let mut rec = Matrix::zeros(d.rows, d.cols);
    let mut err = 0.0;
    for r in 0..d.rows {
        for (k, h) in fields.iter().enumerate() {
            let a = dot(d.row(r), h)?;
            coeff.set(r, k, a);
            for p in 0..d.cols {
                rec.data[r * d.cols + p] += a * h[p];
            }
        }
        for p in 0..d.cols {
            let e = d.get(r, p) - rec.get(r, p);
            err += e * e;
        }
    }
    Ok((coeff, rec, (err / (d.rows * d.cols).max(1) as f64).sqrt()))
}

pub fn reconstruct_skill_fields(
    d: &Matrix,
    base_weights: &[f64],
    groups: &[String],
    generation: u64,
    cfg: &BrainConfig,
) -> BrainResult<TomographyResult> {
    if d.rows < 3 || d.cols == 0 || base_weights.len() != d.rows || groups.len() != d.rows {
        return Err(BrainError::Invalid("tomography_input_shape".into()));
    }
    let mut weights = base_weights
        .iter()
        .map(|w| w.clamp(1e-4, 1.0))
        .collect::<Vec<_>>();
    let mut final_eigs = Vec::new();
    let mut final_dirs = Vec::new();
    let mut final_mixtures = Vec::new();
    let mut final_coeff = Matrix::zeros(0, 0);
    let mut final_rms = 0.0;
    for _ in 0..cfg.irls_rounds.max(1) {
        let gram = weighted_row_gram(d, &weights)?;
        let eigs = symmetric_eigen_jacobi(&gram, 1e-11, d.rows * d.rows * 80)?;
        if eigs.is_empty() {
            return Err(BrainError::Numerical("tomography_zero_spectrum".into()));
        }
        let eigenvalues = eigs.iter().map(|(value, _)| *value).collect::<Vec<_>>();
        let rank =
            choose_energy_rank(&eigenvalues, cfg.target_explained_variance, cfg.max_rank, 1)?;
        let mut dirs = Vec::new();
        let mut mixtures = Vec::new();
        for (lambda, u) in eigs.iter().take(rank) {
            let mut h = vec![0.0; d.cols];
            let denom = lambda.sqrt().max(1e-12);
            let mix = (0..d.rows)
                .map(|r| weights[r].sqrt() * u[r] / denom)
                .collect::<Vec<_>>();
            for r in 0..d.rows {
                for p in 0..d.cols {
                    h[p] += mix[r] * d.get(r, p);
                }
            }
            dirs.push(normalize(&h)?);
            mixtures.push(mix);
        }
        let (coeff, rec, rms) = reconstruct(d, &dirs)?;
        let residual_norms = (0..d.rows)
            .map(|r| {
                let mut s = 0.0;
                for p in 0..d.cols {
                    let e = d.get(r, p) - rec.get(r, p);
                    s += e * e;
                }
                s.sqrt()
            })
            .collect::<Vec<_>>();
        let med = median(residual_norms.clone());
        let mad = median(residual_norms.iter().map(|x| (x - med).abs()).collect()).max(1e-9);
        let scale = 1.4826 * mad;
        for r in 0..d.rows {
            let z = (residual_norms[r] - med).abs() / scale;
            let huber = if z <= cfg.huber_delta {
                1.0
            } else {
                cfg.huber_delta / z
            };
            weights[r] = (base_weights[r].clamp(1e-4, 1.0) * huber).clamp(1e-4, 1.0);
        }
        final_eigs = eigs;
        final_dirs = dirs;
        final_mixtures = mixtures;
        final_coeff = coeff;
        final_rms = rms;
    }
    let rank = final_dirs.len();
    let total = final_eigs.iter().map(|e| e.0).sum::<f64>().max(1e-18);
    let smallest = final_eigs
        .iter()
        .take(rank)
        .map(|e| e.0)
        .fold(f64::INFINITY, f64::min)
        .max(1e-18);
    let largest = final_eigs[0].0.max(smallest);
    let mut fields = Vec::new();
    for k in 0..rank {
        let h = &final_dirs[k];
        let lambda = final_eigs[k].0;
        let mut support = 0usize;
        let mut aligned = Vec::new();
        let mut seen_groups = std::collections::BTreeSet::new();
        for r in 0..d.rows {
            let a = final_coeff.get(r, k);
            if a.abs() > 0.15 * lambda.sqrt().max(1e-9) {
                support += 1;
                aligned.push((a.signum() * cosine(d.row(r), h)?).abs());
                seen_groups.insert(groups[r].clone());
            }
        }
        let coherence = if aligned.is_empty() {
            0.0
        } else {
            aligned.iter().sum::<f64>() / aligned.len() as f64
        };
        let persistence = (seen_groups.len() as f64
            / groups
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                .max(1) as f64)
            .clamp(0.0, 1.0);
        fields.push(SkillField {
            skill_id: format!("skill-g{generation}-{k:03}"),
            reconstruction_id: String::new(),
            lineage_id: String::new(),
            generation_created: generation,
            direction: h.clone(),
            structured_geometry: None,
            dense_materialization: None,
            parameter_layout_sha256: None,
            representation_signature: Vec::new(),
            singular_value: lambda.sqrt(),
            explained_variance: (lambda / total).clamp(0.0, 1.0),
            persistence,
            coherence,
            uncertainty: 1.0 / (lambda.sqrt() + 1e-9),
            evidence_support_digests: Vec::new(),
            support,
            functional_signature: Vec::new(),
            parent_skill_ids: Vec::new(),
        });
    }
    Ok(TomographyResult {
        fields,
        coefficients: final_coeff,
        reconstruction_rms: final_rms,
        effective_rank: effective_rank_from_spectrum(
            &final_eigs
                .iter()
                .map(|(value, _)| *value)
                .collect::<Vec<_>>(),
        )?,
        selected_rank: rank,
        condition_estimate: largest / smallest,
        robust_weights: weights,
        source_mixtures: final_mixtures,
    })
}

fn global_field_alignment(
    old: &[SkillField],
    incoming: &[SkillField],
    threshold: f64,
) -> BrainResult<Vec<Option<(usize, f64)>>> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(BrainError::Invalid(
            "skill_alignment_threshold_invalid".into(),
        ));
    }
    if incoming.is_empty() {
        return Ok(Vec::new());
    }
    if old.is_empty() {
        return Ok(vec![None; incoming.len()]);
    }
    let reference = old
        .iter()
        .map(|field| field.direction.clone())
        .collect::<Vec<_>>();
    let candidate = incoming
        .iter()
        .map(|field| field.direction.clone())
        .collect::<Vec<_>>();
    let alignment = align_bases(&reference, &candidate)?;
    let mut incoming_to_old = vec![None; incoming.len()];
    for (old_index, candidate_index) in alignment.assignment.iter().enumerate() {
        let Some(incoming_index) = *candidate_index else {
            continue;
        };
        let similarity = cosine(
            &old[old_index].direction,
            &incoming[incoming_index].direction,
        )?;
        if similarity.abs() >= threshold {
            incoming_to_old[incoming_index] =
                Some((old_index, if similarity >= 0.0 { 1.0 } else { -1.0 }));
        }
    }
    Ok(incoming_to_old)
}

fn evidence_set(field: &SkillField) -> BrainResult<std::collections::BTreeSet<String>> {
    let set = field
        .evidence_support_digests
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    if set.len() != field.evidence_support_digests.len()
        || set.iter().any(|digest| {
            digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
    {
        return Err(BrainError::Integrity(
            "skill_evidence_support_digest_invalid".into(),
        ));
    }
    Ok(set)
}

fn durable_identity(prior: &SkillField, incoming: &SkillField) -> (String, String, Vec<String>) {
    let legacy_prior = prior.lineage_id.is_empty();
    let skill_id = if legacy_prior {
        incoming.skill_id.clone()
    } else {
        prior.skill_id.clone()
    };
    let lineage_id = if legacy_prior {
        incoming.lineage_id.clone()
    } else {
        prior.lineage_id.clone()
    };
    let mut parents = prior.parent_skill_ids.clone();
    for id in [&prior.skill_id, &incoming.skill_id] {
        if id != &skill_id && !parents.contains(id) {
            parents.push(id.clone());
        }
    }
    parents.sort();
    parents.dedup();
    (skill_id, lineage_id, parents)
}

pub fn align_incoming_identities(
    bank: &SkillBank,
    incoming: &[SkillField],
    threshold: f64,
) -> BrainResult<Vec<SkillField>> {
    let alignment = global_field_alignment(&bank.fields, incoming, threshold)?;
    let mut aligned = Vec::with_capacity(incoming.len());
    for (incoming_index, field) in incoming.iter().enumerate() {
        let mut current = field.clone();
        if let Some((old_index, sign)) = alignment[incoming_index] {
            let prior = &bank.fields[old_index];
            current.direction = current.direction.iter().map(|value| sign * value).collect();
            let (skill_id, lineage_id, parent_skill_ids) = durable_identity(prior, field);
            current.skill_id = skill_id;
            current.lineage_id = lineage_id;
            current.generation_created = if prior.lineage_id.is_empty() {
                field.generation_created
            } else {
                prior.generation_created
            };
            current.parent_skill_ids = parent_skill_ids;
        }
        aligned.push(current);
    }
    Ok(aligned)
}

pub fn assimilate_bank(
    bank: &mut SkillBank,
    incoming: &[SkillField],
    threshold: f64,
) -> BrainResult<()> {
    let old = bank.fields.clone();
    let alignment = global_field_alignment(&old, incoming, threshold)?;
    let mut next = old.clone();
    for (incoming_index, newf) in incoming.iter().enumerate() {
        if let Some((old_index, sign)) = alignment[incoming_index] {
            let prior = &old[old_index];
            let prior_evidence = evidence_set(prior)?;
            let incoming_evidence = evidence_set(newf)?;
            let exact_evidence = !incoming_evidence.is_empty();
            let overlap = prior_evidence.intersection(&incoming_evidence).count();
            let same_evidence =
                exact_evidence && !prior_evidence.is_empty() && prior_evidence == incoming_evidence;
            if exact_evidence && !prior_evidence.is_empty() && overlap > 0 && !same_evidence {
                return Err(BrainError::Integrity(
                    "skill_evidence_partial_overlap_requires_union_reconstruction".into(),
                ));
            }

            let (skill_id, lineage_id, parent_skill_ids) = durable_identity(prior, newf);
            if exact_evidence && (prior_evidence.is_empty() || same_evidence) {
                // Legacy banks have no exact support set, and replaying exactly
                // the same evidence under a newer analysis is a replacement,
                // not another independent observation.
                let mut replacement = newf.clone();
                replacement.direction = replacement
                    .direction
                    .iter()
                    .map(|value| sign * value)
                    .collect();
                replacement.skill_id = skill_id;
                replacement.lineage_id = lineage_id;
                replacement.generation_created = if prior.lineage_id.is_empty() {
                    newf.generation_created
                } else {
                    prior.generation_created
                };
                replacement.parent_skill_ids = parent_skill_ids;
                replacement.support = incoming_evidence.len();
                replacement.evidence_support_digests = incoming_evidence.into_iter().collect();
                next[old_index] = replacement;
                continue;
            }

            let old_support = if prior_evidence.is_empty() {
                prior.support.max(1)
            } else {
                prior_evidence.len()
            } as f64;
            let new_support = if incoming_evidence.is_empty() {
                newf.support.max(1)
            } else {
                incoming_evidence.len()
            } as f64;
            let mut merged = prior
                .direction
                .iter()
                .zip(&newf.direction)
                .map(|(a, b)| old_support * a + new_support * sign * b)
                .collect::<Vec<_>>();
            merged = normalize(&merged)?;
            let slot = &mut next[old_index];
            slot.skill_id = skill_id;
            slot.reconstruction_id = newf.reconstruction_id.clone();
            slot.lineage_id = lineage_id;
            slot.direction = merged;
            slot.structured_geometry = newf.structured_geometry.clone();
            slot.dense_materialization = newf.dense_materialization.clone();
            slot.parameter_layout_sha256 = newf.parameter_layout_sha256.clone();
            slot.representation_signature = newf.representation_signature.clone();
            slot.parent_skill_ids = parent_skill_ids;
            if exact_evidence {
                let union = prior_evidence
                    .union(&incoming_evidence)
                    .cloned()
                    .collect::<Vec<_>>();
                slot.support = union.len();
                slot.evidence_support_digests = union;
            } else {
                slot.support = prior.support.saturating_add(newf.support);
            }
            slot.persistence = (0.7 * prior.persistence + 0.3 * newf.persistence).clamp(0.0, 1.0);
            slot.coherence = (0.7 * prior.coherence + 0.3 * newf.coherence).clamp(0.0, 1.0);
            slot.uncertainty = prior.uncertainty.min(newf.uncertainty);
            if !newf.functional_signature.is_empty() {
                slot.functional_signature = newf.functional_signature.clone();
            }
        } else {
            let mut field = newf.clone();
            if !field.evidence_support_digests.is_empty() {
                let evidence = evidence_set(&field)?;
                field.support = evidence.len();
                field.evidence_support_digests = evidence.into_iter().collect();
            }
            next.push(field);
        }
    }
    next.sort_by(|left, right| left.skill_id.cmp(&right.skill_id));
    bank.generation = bank.generation.saturating_add(1);
    bank.fields = next;
    Ok(())
}

/// Reconcile a complete-corpus reconstruction against durable capabilities with
/// one global Hungarian assignment. This makes identity independent of incoming
/// field order and prevents two candidates from greedily claiming one prior.
pub fn reconcile_full_corpus(
    bank: &mut SkillBank,
    incoming: &[SkillField],
    threshold: f64,
) -> BrainResult<()> {
    let old = bank.fields.clone();
    let alignment = global_field_alignment(&old, incoming, threshold)?;
    let mut used_old = std::collections::BTreeSet::new();
    let mut next = Vec::new();
    for (incoming_index, newf) in incoming.iter().enumerate() {
        if let Some((old_index, sign)) = alignment[incoming_index] {
            used_old.insert(old_index);
            let prior = &old[old_index];
            let mut field = newf.clone();
            field.direction = field.direction.iter().map(|value| sign * value).collect();
            let (skill_id, lineage_id, parent_skill_ids) = durable_identity(prior, newf);
            field.skill_id = skill_id;
            field.lineage_id = lineage_id;
            field.generation_created = if prior.lineage_id.is_empty() {
                newf.generation_created
            } else {
                prior.generation_created
            };
            field.parent_skill_ids = parent_skill_ids;
            next.push(field);
        } else {
            next.push(newf.clone());
        }
    }
    for (old_index, mut prior) in old.into_iter().enumerate() {
        if used_old.contains(&old_index) {
            continue;
        }
        prior.persistence *= 0.85;
        prior.coherence *= 0.90;
        prior.uncertainty *= 1.15;
        if prior.persistence >= 0.15 && prior.coherence >= 0.15 {
            next.push(prior);
        }
    }
    next.sort_by(|left, right| left.skill_id.cmp(&right.skill_id));
    bank.generation = bank.generation.saturating_add(1);
    bank.fields = next;
    Ok(())
}
