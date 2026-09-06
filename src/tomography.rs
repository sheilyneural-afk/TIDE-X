#![allow(clippy::needless_range_loop)]
use crate::contracts::{BrainConfig, SkillBank, SkillField};
use crate::digest::ObservationRecordDigest;
use crate::error::{BrainError, BrainResult};
use crate::gauge::align_bases;
use crate::identity::{LineageId, ReconstructionId, SkillId};
use crate::linalg::{
    cosine, dot, median, norm, normalize, symmetric_eigen_jacobi, weighted_row_gram, Matrix,
};
use crate::validation::{choose_energy_rank, effective_rank_from_spectrum, validate_reliability};

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
    d.validate("tomography_matrix")?;
    if d.rows < 3
        || d.cols == 0
        || base_weights.len() != d.rows
        || groups.len() != d.rows
        || groups.iter().any(|group| group.trim().is_empty())
        || !cfg.huber_delta.is_finite()
        || cfg.huber_delta <= 0.0
        || cfg.irls_rounds == 0
    {
        return Err(BrainError::Invalid("tomography_input_shape".into()));
    }
    let mut weights = base_weights
        .iter()
        .map(|weight| validate_reliability(*weight, "tomography"))
        .collect::<BrainResult<Vec<_>>>()?;
    let mut final_eigs = Vec::new();
    let mut final_dirs = Vec::new();
    let mut final_mixtures = Vec::new();
    let mut final_coeff = Matrix::zeros(0, 0);
    let mut final_rms = 0.0;
    for round in 0..cfg.irls_rounds {
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
            let denom = lambda.sqrt();
            if denom == 0.0 {
                return Err(BrainError::Numerical(
                    "tomography_zero_singular_value".into(),
                ));
            }
            let mut mix = (0..d.rows)
                .map(|r| weights[r].sqrt() * u[r] / denom)
                .collect::<Vec<_>>();
            for r in 0..d.rows {
                for p in 0..d.cols {
                    h[p] += mix[r] * d.get(r, p);
                }
            }
            let h_norm = norm(&h)?;
            if h_norm == 0.0 {
                return Err(BrainError::Numerical(
                    "tomography_reconstructed_direction_degenerate".into(),
                ));
            }
            for coefficient in &mut mix {
                *coefficient /= h_norm;
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
        if round + 1 < cfg.irls_rounds {
            let med = median(residual_norms.clone())?;
            let mad = median(residual_norms.iter().map(|x| (x - med).abs()).collect())?.max(1e-9);
            let scale = 1.4826 * mad;
            for r in 0..d.rows {
                let z = (residual_norms[r] - med).abs() / scale;
                let huber = if z <= cfg.huber_delta {
                    1.0
                } else {
                    cfg.huber_delta / z
                };
                weights[r] = base_weights[r] * huber;
            }
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
            skill_id: SkillId::parse(format!("skill-g{generation}-{k:03}"))?,
            reconstruction_id: ReconstructionId::unassigned(),
            lineage_id: LineageId::unassigned(),
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

fn evidence_set(
    field: &SkillField,
) -> BrainResult<std::collections::BTreeSet<ObservationRecordDigest>> {
    let set = field
        .evidence_support_digests
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    if set.len() != field.evidence_support_digests.len() {
        return Err(BrainError::Integrity(
            "skill_evidence_support_digest_invalid".into(),
        ));
    }
    Ok(set)
}

fn durable_identity(
    prior: &SkillField,
    incoming: &SkillField,
) -> (SkillId, LineageId, Vec<SkillId>) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tomography_rejects_zero_reliability_and_zero_irls_rounds() {
        let matrix = Matrix::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]).unwrap();
        let groups = vec!["a".into(), "b".into(), "c".into()];
        assert!(reconstruct_skill_fields(
            &matrix,
            &[1.0, 0.0, 1.0],
            &groups,
            1,
            &BrainConfig::default(),
        )
        .is_err());

        let config = BrainConfig {
            irls_rounds: 0,
            ..Default::default()
        };
        assert!(reconstruct_skill_fields(&matrix, &[1.0; 3], &groups, 1, &config).is_err());
    }

    #[test]
    fn tomography_sources_exactly_recreate_normalized_fields_and_report_used_weights() {
        let matrix = Matrix::from_rows(&[vec![2.0, 0.0], vec![0.0, 3.0], vec![1.0, 1.0]]).unwrap();
        let base_weights = [0.25, 0.5, 1.0];
        let groups = vec!["a".into(), "b".into(), "c".into()];
        let config = BrainConfig {
            irls_rounds: 1,
            max_rank: 2,
            target_explained_variance: 1.0,
            ..Default::default()
        };
        let result = reconstruct_skill_fields(&matrix, &base_weights, &groups, 1, &config).unwrap();
        assert_eq!(result.robust_weights, base_weights);
        for (field, mixture) in result.fields.iter().zip(&result.source_mixtures) {
            let mut reconstructed = vec![0.0; matrix.column_count()];
            for row in 0..matrix.row_count() {
                for column in 0..matrix.column_count() {
                    reconstructed[column] += mixture[row] * matrix.get(row, column);
                }
            }
            for (actual, expected) in reconstructed.iter().zip(&field.direction) {
                assert!((actual - expected).abs() < 1e-10);
            }
        }
    }

    fn test_field(
        id: &str,
        dir: Vec<f64>,
        ev_keys: &[&[u8]],
        persistence: f64,
        coherence: f64,
    ) -> SkillField {
        use crate::digest::{ObservationRecordDigest, Sha256Digest};
        use crate::identity::{LineageId, ReconstructionId, SkillId};
        let digests = ev_keys
            .iter()
            .map(|k| ObservationRecordDigest::from(Sha256Digest::digest_bytes(k)))
            .collect::<Vec<_>>();
        SkillField {
            skill_id: SkillId::parse(id).unwrap(),
            reconstruction_id: ReconstructionId::parse("recon-1").unwrap(),
            lineage_id: LineageId::parse(format!("lineage-{id}")).unwrap(),
            generation_created: 1,
            direction: dir,
            structured_geometry: None,
            dense_materialization: None,
            parameter_layout_sha256: None,
            representation_signature: vec![],
            singular_value: 1.0,
            explained_variance: 0.5,
            persistence,
            coherence,
            uncertainty: 0.1,
            evidence_support_digests: digests.clone(),
            support: digests.len(),
            functional_signature: vec![1.0],
            parent_skill_ids: vec![],
        }
    }

    #[test]
    fn align_incoming_identities_preserves_prior_and_aligns_sign() {
        let f1 = test_field("f1", vec![1.0, 0.0], &[b"ev1"], 0.8, 0.9);
        let bank = SkillBank {
            generation: 1,
            fields: vec![f1],
        };
        // Incoming field with opposite sign
        let incoming_neg = test_field("temp-id", vec![-1.0, 0.0], &[b"ev1"], 0.8, 0.9);
        let aligned = align_incoming_identities(&bank, &[incoming_neg], 0.8).unwrap();
        assert_eq!(aligned.len(), 1);
        assert_eq!(aligned[0].skill_id.as_str(), "f1");
        assert!((aligned[0].direction[0] - 1.0).abs() < 1e-10);
        assert!((aligned[0].direction[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn assimilate_bank_replaces_on_exact_evidence_match() {
        let f1 = test_field("f1", vec![1.0, 0.0], &[b"ev1"], 0.8, 0.9);
        let mut bank = SkillBank {
            generation: 1,
            fields: vec![f1],
        };
        let incoming = test_field("f1-new", vec![0.99, 0.1], &[b"ev1"], 0.85, 0.95);
        assimilate_bank(&mut bank, &[incoming], 0.8).unwrap();
        assert_eq!(bank.generation, 2);
        assert_eq!(bank.fields.len(), 1);
        assert_eq!(bank.fields[0].skill_id.as_str(), "f1");
        assert_eq!(bank.fields[0].support, 1);
    }

    #[test]
    fn assimilate_bank_rejects_partial_evidence_overlap() {
        let f1 = test_field("f1", vec![1.0, 0.0], &[b"ev1", b"ev2"], 0.8, 0.9);
        let mut bank = SkillBank {
            generation: 1,
            fields: vec![f1],
        };
        let incoming = test_field("f1-overlap", vec![1.0, 0.0], &[b"ev2", b"ev3"], 0.8, 0.9);
        let err = assimilate_bank(&mut bank, &[incoming], 0.8).unwrap_err();
        assert!(
            matches!(err, BrainError::Integrity(ref m) if m.contains("skill_evidence_partial_overlap"))
        );
    }

    #[test]
    fn assimilate_bank_merges_disjoint_evidence_and_adds_unmatched() {
        let f1 = test_field("f1", vec![1.0, 0.0], &[b"ev1"], 0.8, 0.9);
        let mut bank = SkillBank {
            generation: 1,
            fields: vec![f1],
        };
        let f1_incoming = test_field("f1-incoming", vec![1.0, 0.0], &[b"ev2"], 0.8, 0.9);
        let f2_incoming = test_field("f2", vec![0.0, 1.0], &[b"ev3"], 0.8, 0.9);
        assimilate_bank(&mut bank, &[f1_incoming, f2_incoming], 0.8).unwrap();
        assert_eq!(bank.generation, 2);
        assert_eq!(bank.fields.len(), 2);
        let f1_merged = bank
            .fields
            .iter()
            .find(|f| f.skill_id.as_str() == "f1")
            .unwrap();
        assert_eq!(f1_merged.support, 2);
        assert_eq!(f1_merged.evidence_support_digests.len(), 2);
        let f2_new = bank
            .fields
            .iter()
            .find(|f| f.skill_id.as_str() == "f2")
            .unwrap();
        assert_eq!(f2_new.support, 1);
    }

    #[test]
    fn reconcile_full_corpus_decays_or_drops_unassigned_priors() {
        let f1 = test_field("f1", vec![1.0, 0.0], &[b"ev1"], 0.8, 0.9);
        let f2_decaying = test_field("f2", vec![0.0, 1.0], &[b"ev2"], 0.5, 0.5);
        let f3_dropping = test_field(
            "f3",
            vec![
                -std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            ],
            &[b"ev3"],
            0.15,
            0.15,
        );
        let mut bank = SkillBank {
            generation: 1,
            fields: vec![f1, f2_decaying, f3_dropping],
        };
        // incoming only matches f1
        let incoming = vec![test_field("f1-match", vec![1.0, 0.0], &[b"ev1"], 0.9, 0.9)];
        reconcile_full_corpus(&mut bank, &incoming, 0.8).unwrap();
        assert_eq!(bank.generation, 2);
        // f1 is matched, f2 decayed (0.5 * 0.85 = 0.425 >= 0.15), f3 dropped (0.15 * 0.85 = 0.1275 < 0.15)
        assert!(bank.fields.iter().any(|f| f.skill_id.as_str() == "f1"));
        assert!(bank.fields.iter().any(|f| f.skill_id.as_str() == "f2"));
        assert!(!bank.fields.iter().any(|f| f.skill_id.as_str() == "f3"));
    }
}
