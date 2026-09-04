use crate::error::{BrainError, BrainResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const EFFECT_95_Z: f64 = 1.959_963_984_540_054;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CounterfactualEvaluation {
    pub context_id: String,
    /// Statistical independence unit. Multiple tasks evaluated under the same
    /// randomization/seed share one group and must not be treated as replicates.
    #[serde(default)]
    pub independence_group: String,
    pub active_fields: Vec<String>,
    pub utility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldCausalCredit {
    pub skill_id: String,
    pub matched_pairs: usize,
    pub independent_contexts: usize,
    pub mean_marginal_effect: f64,
    pub standard_error: f64,
    pub lower_confidence_bound: f64,
    pub positive_fraction: f64,
    /// `resolved` means the effect is statistically estimable from enough
    /// independent groups. It does NOT itself claim positive benefit.
    pub resolved: bool,
    pub beneficial: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairInteractionCredit {
    pub left_skill_id: String,
    pub right_skill_id: String,
    pub matched_quads: usize,
    pub independent_contexts: usize,
    pub mean_interaction_effect: f64,
    pub standard_error: f64,
    pub resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CausalCreditReport {
    pub schema: String,
    pub context_count: usize,
    pub independent_group_count: usize,
    pub field_count: usize,
    pub fields: Vec<FieldCausalCredit>,
    pub pair_interactions: Vec<PairInteractionCredit>,
    pub unresolved_fields: Vec<String>,
}

/// Return the conservative, evidence-backed causal utility weight for every
/// runtime field in the caller's canonical order.  The lower confidence bound,
/// rather than the point estimate, is the only quantity that may prioritize
/// preservation inside a governed trust region.  Any missing, unresolved, or
/// non-beneficial field is an authority failure: composition must not silently
/// substitute a heuristic weight.
pub fn certified_causal_priority_weights(
    report: &CausalCreditReport,
    field_ids: &[String],
) -> BrainResult<Vec<f64>> {
    if report.schema != "cerebro.tidex.causal_credit/v3"
        || field_ids.is_empty()
        || report.field_count != field_ids.len()
        || report.independent_group_count < 3
        || !report.unresolved_fields.is_empty()
    {
        return Err(BrainError::Integrity(
            "causal_credit_priority_report_unresolved".into(),
        ));
    }
    let expected = field_ids.iter().cloned().collect::<BTreeSet<_>>();
    if expected.len() != field_ids.len() || expected.iter().any(|id| id.trim().is_empty()) {
        return Err(BrainError::Invalid(
            "causal_credit_priority_field_identity_invalid".into(),
        ));
    }
    let mut by_id = BTreeMap::new();
    for field in &report.fields {
        if field.skill_id.trim().is_empty()
            || !field.resolved
            || !field.beneficial
            || field.matched_pairs == 0
            || field.independent_contexts < 3
            || !field.mean_marginal_effect.is_finite()
            || !field.standard_error.is_finite()
            || field.standard_error < 0.0
            || !field.lower_confidence_bound.is_finite()
            || field.lower_confidence_bound <= 0.0
            || !field.positive_fraction.is_finite()
            || !(0.0..=1.0).contains(&field.positive_fraction)
            || by_id
                .insert(field.skill_id.as_str(), field.lower_confidence_bound)
                .is_some()
        {
            return Err(BrainError::Integrity(
                "causal_credit_priority_field_unresolved".into(),
            ));
        }
    }
    if by_id.len() != expected.len()
        || by_id.keys().any(|id| !expected.contains(*id))
        || report.pair_interactions.len() != expected.len() * expected.len().saturating_sub(1) / 2
    {
        return Err(BrainError::Integrity(
            "causal_credit_priority_identity_mismatch".into(),
        ));
    }
    let mut pairs = BTreeSet::new();
    for pair in &report.pair_interactions {
        let (left, right) = if pair.left_skill_id < pair.right_skill_id {
            (&pair.left_skill_id, &pair.right_skill_id)
        } else {
            (&pair.right_skill_id, &pair.left_skill_id)
        };
        if left == right
            || !expected.contains(left)
            || !expected.contains(right)
            || !pair.resolved
            || pair.matched_quads == 0
            || pair.independent_contexts < 3
            || !pair.mean_interaction_effect.is_finite()
            || !pair.standard_error.is_finite()
            || pair.standard_error < 0.0
            || !pairs.insert((left.clone(), right.clone()))
        {
            return Err(BrainError::Integrity(
                "causal_credit_priority_interaction_unresolved".into(),
            ));
        }
    }
    field_ids
        .iter()
        .map(|field_id| {
            by_id
                .get(field_id.as_str())
                .copied()
                .ok_or_else(|| BrainError::Integrity("causal_credit_priority_field_missing".into()))
        })
        .collect::<BrainResult<Vec<_>>>()
}

fn canonical_set(fields: &[String]) -> BTreeSet<String> {
    fields.iter().cloned().collect()
}

fn mean_and_se(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, f64::INFINITY);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() < 2 {
        return (mean, f64::INFINITY);
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (mean, (variance / values.len() as f64).sqrt())
}

fn independent_means(
    context_values: &[(String, f64)],
    context_groups: &BTreeMap<String, String>,
) -> BrainResult<Vec<f64>> {
    let mut grouped = BTreeMap::<String, Vec<f64>>::new();
    for (context, value) in context_values {
        let group = context_groups
            .get(context)
            .ok_or_else(|| BrainError::Integrity("causal_credit_context_group_missing".into()))?;
        grouped.entry(group.clone()).or_default().push(*value);
    }
    Ok(grouped
        .into_values()
        .map(|values| values.iter().sum::<f64>() / values.len() as f64)
        .collect())
}

pub fn estimate_causal_credit(
    evaluations: &[CounterfactualEvaluation],
) -> BrainResult<CausalCreditReport> {
    if evaluations.is_empty() {
        return Err(BrainError::Invalid("causal_credit_empty".into()));
    }
    let mut contexts = BTreeSet::new();
    let mut independent_groups = BTreeSet::new();
    let mut context_groups = BTreeMap::<String, String>::new();
    let mut fields = BTreeSet::new();
    let mut table = BTreeMap::<(String, BTreeSet<String>), f64>::new();
    for (index, evaluation) in evaluations.iter().enumerate() {
        if evaluation.context_id.trim().is_empty() || !evaluation.utility.is_finite() {
            return Err(BrainError::Invalid(format!(
                "causal_credit_evaluation_invalid:{index}"
            )));
        }
        let group = if evaluation.independence_group.trim().is_empty() {
            evaluation.context_id.clone()
        } else {
            evaluation.independence_group.clone()
        };
        if let Some(existing) = context_groups.insert(evaluation.context_id.clone(), group.clone())
        {
            if existing != group {
                return Err(BrainError::Invalid(
                    "causal_credit_context_independence_group_inconsistent".into(),
                ));
            }
        }
        let active = canonical_set(&evaluation.active_fields);
        if active.len() != evaluation.active_fields.len() {
            return Err(BrainError::Invalid(format!(
                "causal_credit_duplicate_field:{index}"
            )));
        }
        contexts.insert(evaluation.context_id.clone());
        independent_groups.insert(group);
        fields.extend(active.iter().cloned());
        let key = (evaluation.context_id.clone(), active);
        if table.insert(key, evaluation.utility).is_some() {
            return Err(BrainError::Invalid("causal_credit_duplicate_cell".into()));
        }
    }

    let field_ids = fields.iter().cloned().collect::<Vec<_>>();
    let mut field_reports = Vec::with_capacity(field_ids.len());
    let mut unresolved = Vec::new();
    for field in &field_ids {
        let mut raw_pair_count = 0usize;
        let mut context_effects = Vec::<(String, f64)>::new();
        for context in &contexts {
            let cells = table
                .iter()
                .filter(|((ctx, _), _)| ctx == context)
                .map(|((_, set), utility)| (set.clone(), *utility))
                .collect::<Vec<_>>();
            let mut local_effects = Vec::new();
            for (set, utility_without) in &cells {
                if set.contains(field) {
                    continue;
                }
                let mut with = set.clone();
                with.insert(field.clone());
                if let Some(utility_with) = table.get(&(context.clone(), with)) {
                    local_effects.push(*utility_with - *utility_without);
                    raw_pair_count += 1;
                }
            }
            if !local_effects.is_empty() {
                context_effects.push((
                    context.clone(),
                    local_effects.iter().sum::<f64>() / local_effects.len() as f64,
                ));
            }
        }
        let effects = independent_means(&context_effects, &context_groups)?;
        let (mean, se) = mean_and_se(&effects);
        let lower_confidence_bound = if se.is_finite() {
            mean - EFFECT_95_Z * se
        } else {
            f64::NEG_INFINITY
        };
        let positive_fraction = if effects.is_empty() {
            0.0
        } else {
            effects.iter().filter(|value| **value > 0.0).count() as f64 / effects.len() as f64
        };
        let resolved = effects.len() >= 3 && se.is_finite();
        let beneficial = resolved && lower_confidence_bound > 0.0;
        if !resolved {
            unresolved.push(field.clone());
        }
        field_reports.push(FieldCausalCredit {
            skill_id: field.clone(),
            matched_pairs: raw_pair_count,
            independent_contexts: effects.len(),
            mean_marginal_effect: mean,
            standard_error: se,
            lower_confidence_bound,
            positive_fraction,
            resolved,
            beneficial,
        });
    }

    let mut pair_interactions = Vec::new();
    for i in 0..field_ids.len() {
        for j in 0..i {
            let left = &field_ids[j];
            let right = &field_ids[i];
            let mut raw_quad_count = 0usize;
            let mut context_interactions = Vec::<(String, f64)>::new();
            for context in &contexts {
                let cells = table
                    .iter()
                    .filter(|((ctx, _), _)| ctx == context)
                    .map(|((_, set), utility)| (set.clone(), *utility))
                    .collect::<Vec<_>>();
                let mut local_interactions = Vec::new();
                for (base, u00) in &cells {
                    if base.contains(left) || base.contains(right) {
                        continue;
                    }
                    let mut only_left = base.clone();
                    only_left.insert(left.clone());
                    let mut only_right = base.clone();
                    only_right.insert(right.clone());
                    let mut both = only_left.clone();
                    both.insert(right.clone());
                    if let (Some(u10), Some(u01), Some(u11)) = (
                        table.get(&(context.clone(), only_left)),
                        table.get(&(context.clone(), only_right)),
                        table.get(&(context.clone(), both)),
                    ) {
                        local_interactions.push(*u11 - *u10 - *u01 + *u00);
                        raw_quad_count += 1;
                    }
                }
                if !local_interactions.is_empty() {
                    context_interactions.push((
                        context.clone(),
                        local_interactions.iter().sum::<f64>() / local_interactions.len() as f64,
                    ));
                }
            }
            let effects = independent_means(&context_interactions, &context_groups)?;
            let (mean, se) = mean_and_se(&effects);
            pair_interactions.push(PairInteractionCredit {
                left_skill_id: left.clone(),
                right_skill_id: right.clone(),
                matched_quads: raw_quad_count,
                independent_contexts: effects.len(),
                mean_interaction_effect: mean,
                standard_error: se,
                resolved: effects.len() >= 3 && se.is_finite(),
            });
        }
    }
    Ok(CausalCreditReport {
        schema: "cerebro.tidex.causal_credit/v3".into(),
        context_count: contexts.len(),
        independent_group_count: independent_groups.len(),
        field_count: field_ids.len(),
        fields: field_reports,
        pair_interactions,
        unresolved_fields: unresolved,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_credit_recovers_main_and_interaction_effects() {
        let mut evaluations = Vec::new();
        for context in ["c1", "c2", "c3"] {
            for mask in 0..4 {
                let mut active = Vec::new();
                if mask & 1 != 0 {
                    active.push("a".to_string());
                }
                if mask & 2 != 0 {
                    active.push("b".to_string());
                }
                let a = if mask & 1 != 0 { 2.0 } else { 0.0 };
                let b = if mask & 2 != 0 { 1.0 } else { 0.0 };
                let interaction = if mask == 3 { 0.5 } else { 0.0 };
                evaluations.push(CounterfactualEvaluation {
                    context_id: context.into(),
                    independence_group: context.into(),
                    active_fields: active,
                    utility: a + b + interaction,
                });
            }
        }
        let report = estimate_causal_credit(&evaluations).unwrap();
        assert_eq!(report.independent_group_count, 3);
        assert!(report.unresolved_fields.is_empty());
        let a = report
            .fields
            .iter()
            .find(|field| field.skill_id == "a")
            .unwrap();
        let b = report
            .fields
            .iter()
            .find(|field| field.skill_id == "b")
            .unwrap();
        assert!((a.mean_marginal_effect - 2.25).abs() < 1e-12);
        assert!((b.mean_marginal_effect - 1.25).abs() < 1e-12);
        let pair = &report.pair_interactions[0];
        assert!((pair.mean_interaction_effect - 0.5).abs() < 1e-12);
    }

    #[test]
    fn tasks_under_one_randomization_count_as_one_independent_group() {
        let mut evaluations = Vec::new();
        for seed in [1, 2, 3] {
            for task in ["a", "b"] {
                for mask in 0..2 {
                    evaluations.push(CounterfactualEvaluation {
                        context_id: format!("seed-{seed}:{task}"),
                        independence_group: format!("seed-{seed}"),
                        active_fields: if mask == 1 {
                            vec!["skill".into()]
                        } else {
                            vec![]
                        },
                        utility: if mask == 1 { 1.0 } else { 0.0 },
                    });
                }
            }
        }
        let report = estimate_causal_credit(&evaluations).unwrap();
        assert_eq!(report.context_count, 6);
        assert_eq!(report.independent_group_count, 3);
        assert_eq!(report.fields[0].independent_contexts, 3);
        assert!(report.fields[0].beneficial);
    }
}
