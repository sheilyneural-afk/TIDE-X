use crate::causal_credit::CausalCreditReport;
use crate::contracts::SkillField;
use crate::error::{BrainError, BrainResult};
use crate::identity::SkillId;
use crate::linalg::{cosine, Matrix};
use crate::validation::validate_symmetric_psd;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CognitiveFieldConfig {
    pub curvature_weight: f64,
    pub causal_interaction_weight: f64,
    pub functional_compatibility_weight: f64,
    pub causal_bias_weight: f64,
    pub coupling_gain: f64,
    pub laplacian_gain: f64,
    pub evidence_gain: f64,
    pub prediction_error_gain: f64,
    pub inhibition_gain: f64,
    pub risk_gain: f64,
    pub decay: f64,
    pub beta: f64,
    pub dt: f64,
    pub state_limit: f64,
    pub max_steps: usize,
    pub convergence_tolerance: f64,
    pub coalition_activation_threshold: f64,
    pub coalition_coupling_threshold: f64,
    pub ignition_threshold: f64,
    pub minimum_ignition_fields: usize,
}

impl Default for CognitiveFieldConfig {
    fn default() -> Self {
        Self {
            curvature_weight: 0.35,
            causal_interaction_weight: 0.45,
            functional_compatibility_weight: 0.20,
            causal_bias_weight: 0.25,
            coupling_gain: 0.65,
            laplacian_gain: 0.15,
            evidence_gain: 1.0,
            prediction_error_gain: 1.0,
            inhibition_gain: 1.0,
            risk_gain: 1.0,
            decay: 1.0,
            beta: 1.5,
            dt: 0.08,
            state_limit: 4.0,
            max_steps: 1024,
            convergence_tolerance: 1e-9,
            coalition_activation_threshold: 0.25,
            coalition_coupling_threshold: 0.05,
            ignition_threshold: 0.50,
            minimum_ignition_fields: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CognitiveFieldDrive {
    pub evidence: Vec<f64>,
    pub prediction_error: Vec<f64>,
    pub inhibition: Vec<f64>,
    pub risk: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CognitiveCoalition {
    pub field_ids: Vec<SkillId>,
    pub mean_activation: f64,
    pub internal_coherence: f64,
    pub salience: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CognitiveFieldState {
    pub schema: String,
    pub field_ids: Vec<SkillId>,
    pub activations: Vec<f64>,
    pub steps: usize,
    pub converged: bool,
    pub final_max_delta: f64,
    pub attractor_digest: String,
    pub coalitions: Vec<CognitiveCoalition>,
    pub global_ignition: bool,
    pub ignition_coalition: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldRoutingDecision {
    pub schema: String,
    pub field_ids: Vec<SkillId>,
    pub coefficients: Vec<f64>,
    pub selected_field_ids: Vec<SkillId>,
    pub selected_activation_mass: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DynamicCognitiveField {
    pub schema: String,
    pub field_ids: Vec<SkillId>,
    pub coupling_matrix: Vec<Vec<f64>>,
    pub positive_laplacian: Vec<Vec<f64>>,
    pub causal_bias: Vec<f64>,
    pub config: CognitiveFieldConfig,
}

fn validate_config(config: &CognitiveFieldConfig) -> BrainResult<()> {
    let nonnegative = [
        config.curvature_weight,
        config.causal_interaction_weight,
        config.functional_compatibility_weight,
        config.causal_bias_weight,
        config.coupling_gain,
        config.laplacian_gain,
        config.evidence_gain,
        config.prediction_error_gain,
        config.inhibition_gain,
        config.risk_gain,
    ];
    if nonnegative.iter().any(|v| !v.is_finite() || *v < 0.0)
        || config.curvature_weight
            + config.causal_interaction_weight
            + config.functional_compatibility_weight
            <= 0.0
        || !config.decay.is_finite()
        || config.decay <= 0.0
        || !config.beta.is_finite()
        || config.beta <= 0.0
        || !config.dt.is_finite()
        || config.dt <= 0.0
        || config.dt > 1.0
        || !config.state_limit.is_finite()
        || config.state_limit <= 0.0
        || config.max_steps == 0
        || !config.convergence_tolerance.is_finite()
        || config.convergence_tolerance <= 0.0
        || !config.coalition_activation_threshold.is_finite()
        || config.coalition_activation_threshold < -config.state_limit
        || config.coalition_activation_threshold > config.state_limit
        || !config.coalition_coupling_threshold.is_finite()
        || config.coalition_coupling_threshold < 0.0
        || !config.ignition_threshold.is_finite()
        || config.ignition_threshold < 0.0
        || config.minimum_ignition_fields == 0
    {
        return Err(BrainError::Invalid("cognitive_field_config_invalid".into()));
    }
    Ok(())
}

fn matrix_rows(matrix: &Matrix) -> Vec<Vec<f64>> {
    (0..matrix.rows).map(|row| matrix.row_vec(row)).collect()
}

fn curvature_similarity(curvature: &Matrix, left: usize, right: usize) -> f64 {
    let denom = (curvature.get(left, left).max(0.0) * curvature.get(right, right).max(0.0)).sqrt();
    if denom <= 1e-15 {
        0.0
    } else {
        (curvature.get(left, right) / denom).clamp(-1.0, 1.0)
    }
}

fn robust_effect_scale(values: impl Iterator<Item = f64>) -> f64 {
    let mut abs = values
        .map(f64::abs)
        .filter(|value| value.is_finite() && *value > 1e-15)
        .collect::<Vec<_>>();
    if abs.is_empty() {
        return 1.0;
    }
    abs.sort_by(f64::total_cmp);
    let middle = abs.len() / 2;
    if abs.len() % 2 == 0 {
        0.5 * (abs[middle - 1] + abs[middle])
    } else {
        abs[middle]
    }
    .max(1e-12)
}

fn functional_similarity(left: &SkillField, right: &SkillField) -> BrainResult<f64> {
    match (
        left.functional_signature.is_empty(),
        right.functional_signature.is_empty(),
    ) {
        (true, true) | (true, false) | (false, true) => Ok(0.0),
        (false, false) => {
            if left.functional_signature.len() != right.functional_signature.len() {
                return Err(BrainError::Integrity(
                    "cognitive_field_functional_signature_shape".into(),
                ));
            }
            cosine(&left.functional_signature, &right.functional_signature)
        }
    }
}

fn attractor_digest(field_ids: &[SkillId], activations: &[f64]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:COGNITIVE-FIELD-ATTRACTOR:v1\0");
    for (field_id, activation) in field_ids.iter().zip(activations) {
        hasher.update((field_id.len() as u64).to_be_bytes());
        hasher.update(field_id.as_bytes());
        let quantized = (activation * 1e12).round() / 1e12;
        hasher.update(quantized.to_bits().to_be_bytes());
    }
    format!("{:x}", hasher.finalize())
}

impl DynamicCognitiveField {
    pub fn build(
        fields: &[SkillField],
        curvature: &Matrix,
        causal: &CausalCreditReport,
        config: CognitiveFieldConfig,
    ) -> BrainResult<Self> {
        validate_config(&config)?;
        let n = fields.len();
        if n == 0 || curvature.rows != n || curvature.cols != n || causal.field_count != n {
            return Err(BrainError::Invalid("cognitive_field_input_shape".into()));
        }
        let _ = validate_symmetric_psd(curvature, "cognitive_field_curvature")?;
        if config.minimum_ignition_fields > n {
            return Err(BrainError::Invalid(
                "cognitive_field_minimum_ignition_exceeds_field_count".into(),
            ));
        }
        if config.functional_compatibility_weight > 0.0 {
            let dimensions = fields
                .iter()
                .map(|field| field.functional_signature.len())
                .collect::<BTreeSet<_>>();
            if dimensions.len() != 1 || dimensions.contains(&0) {
                return Err(BrainError::Integrity(
                    "cognitive_field_functional_evidence_incomplete".into(),
                ));
            }
        }
        let mut field_ids = Vec::with_capacity(n);
        let mut unique = BTreeSet::new();
        for field in fields {
            if field.skill_id.trim().is_empty() || !unique.insert(field.skill_id.clone()) {
                return Err(BrainError::Invalid("cognitive_field_field_identity".into()));
            }
            field_ids.push(field.skill_id.clone());
        }
        let causal_ids = causal
            .fields
            .iter()
            .map(|field| field.skill_id.clone())
            .collect::<BTreeSet<_>>();
        if causal_ids != unique || causal.fields.iter().any(|field| !field.resolved) {
            return Err(BrainError::Integrity(
                "cognitive_field_causal_identity_unresolved".into(),
            ));
        }

        let marginal_scale =
            robust_effect_scale(causal.fields.iter().map(|row| row.mean_marginal_effect));
        let mut pair_effects = BTreeMap::<(SkillId, SkillId), f64>::new();
        for pair in &causal.pair_interactions {
            if !pair.resolved
                || !pair.mean_interaction_effect.is_finite()
                || !unique.contains(&pair.left_skill_id)
                || !unique.contains(&pair.right_skill_id)
                || pair.left_skill_id == pair.right_skill_id
            {
                return Err(BrainError::Integrity(
                    "cognitive_field_pair_interaction_invalid".into(),
                ));
            }
            let key = if pair.left_skill_id < pair.right_skill_id {
                (pair.left_skill_id.clone(), pair.right_skill_id.clone())
            } else {
                (pair.right_skill_id.clone(), pair.left_skill_id.clone())
            };
            let conservative_effect = pair.lower_confidence_bound()?;
            if pair_effects.insert(key, conservative_effect).is_some() {
                return Err(BrainError::Integrity(
                    "cognitive_field_pair_interaction_duplicate".into(),
                ));
            }
        }

        let pair_scale = robust_effect_scale(pair_effects.values().copied());

        let expected_pairs = n * n.saturating_sub(1) / 2;
        if config.causal_interaction_weight > 0.0 && pair_effects.len() != expected_pairs {
            return Err(BrainError::Integrity(
                "cognitive_field_pair_interaction_evidence_incomplete".into(),
            ));
        }

        let weight_sum = config.curvature_weight
            + config.causal_interaction_weight
            + config.functional_compatibility_weight;
        let mut coupling = Matrix::zeros(n, n);
        for left in 0..n {
            for right in left + 1..n {
                let curvature_term = curvature_similarity(curvature, left, right);
                let key = if field_ids[left] < field_ids[right] {
                    (field_ids[left].clone(), field_ids[right].clone())
                } else {
                    (field_ids[right].clone(), field_ids[left].clone())
                };
                let causal_term = pair_effects
                    .get(&key)
                    .map(|effect| (effect / pair_scale).tanh())
                    .unwrap_or(0.0);
                let functional_term = functional_similarity(&fields[left], &fields[right])?;
                let combined = config.coupling_gain
                    * (config.curvature_weight * curvature_term
                        + config.causal_interaction_weight * causal_term
                        + config.functional_compatibility_weight * functional_term)
                    / weight_sum;
                if !combined.is_finite() {
                    return Err(BrainError::Numerical(
                        "cognitive_field_coupling_non_finite".into(),
                    ));
                }
                coupling.set(left, right, combined);
                coupling.set(right, left, combined);
            }
        }

        let mut positive_laplacian = Matrix::zeros(n, n);
        for row in 0..n {
            let mut degree = 0.0;
            for col in 0..n {
                if row == col {
                    continue;
                }
                let weight = coupling.get(row, col).max(0.0);
                degree += weight;
                positive_laplacian.set(row, col, -weight);
            }
            positive_laplacian.set(row, row, degree);
        }

        let causal_by_id = causal
            .fields
            .iter()
            .map(|row| (row.skill_id.as_str(), row))
            .collect::<BTreeMap<_, _>>();
        let causal_bias = field_ids
            .iter()
            .map(|field_id| {
                let effect = causal_by_id[field_id.as_str()].mean_marginal_effect;
                config.causal_bias_weight * (effect / marginal_scale).tanh()
            })
            .collect::<Vec<_>>();

        Ok(Self {
            schema: "cerebro.tidex.dynamic_cognitive_field/v1".into(),
            field_ids,
            coupling_matrix: matrix_rows(&coupling),
            positive_laplacian: matrix_rows(&positive_laplacian),
            causal_bias,
            config,
        })
    }

    fn matrices(&self) -> BrainResult<(Matrix, Matrix)> {
        let coupling = Matrix::from_rows(&self.coupling_matrix)?;
        let laplacian = Matrix::from_rows(&self.positive_laplacian)?;
        let n = self.field_ids.len();
        if n == 0
            || coupling.rows != n
            || coupling.cols != n
            || laplacian.rows != n
            || laplacian.cols != n
            || self.causal_bias.len() != n
        {
            return Err(BrainError::Integrity("cognitive_field_model_shape".into()));
        }
        Ok((coupling, laplacian))
    }

    pub fn evolve(
        &self,
        initial: &[f64],
        drive: &CognitiveFieldDrive,
    ) -> BrainResult<CognitiveFieldState> {
        validate_config(&self.config)?;
        let n = self.field_ids.len();
        if initial.len() != n
            || drive.evidence.len() != n
            || drive.prediction_error.len() != n
            || drive.inhibition.len() != n
            || drive.risk.len() != n
            || initial
                .iter()
                .chain(&drive.evidence)
                .chain(&drive.prediction_error)
                .chain(&drive.inhibition)
                .chain(&drive.risk)
                .any(|value| !value.is_finite())
            || drive.risk.iter().any(|value| *value < 0.0)
            || drive.inhibition.iter().any(|value| *value < 0.0)
        {
            return Err(BrainError::Invalid(
                "cognitive_field_drive_shape_or_values".into(),
            ));
        }
        let (coupling, laplacian) = self.matrices()?;
        let mut state = initial.to_vec();
        let mut final_max_delta = f64::INFINITY;
        let mut steps = 0usize;
        let mut converged = false;
        for step in 0..self.config.max_steps {
            let phi = state
                .iter()
                .map(|value| (self.config.beta * value).tanh())
                .collect::<Vec<_>>();
            let coupled = coupling.matvec(&phi)?;
            let diffusion = laplacian.matvec(&state)?;
            let mut next = vec![0.0; n];
            final_max_delta = 0.0;
            for index in 0..n {
                let derivative = self.config.evidence_gain * drive.evidence[index]
                    + self.config.prediction_error_gain * drive.prediction_error[index]
                    + self.causal_bias[index]
                    + coupled[index]
                    - self.config.decay * state[index]
                    - self.config.laplacian_gain * diffusion[index]
                    - self.config.inhibition_gain * drive.inhibition[index]
                    - self.config.risk_gain * drive.risk[index];
                let candidate = (state[index] + self.config.dt * derivative)
                    .clamp(-self.config.state_limit, self.config.state_limit);
                final_max_delta = final_max_delta.max((candidate - state[index]).abs());
                next[index] = candidate;
            }
            if next.iter().any(|value| !value.is_finite()) {
                return Err(BrainError::Numerical(
                    "cognitive_field_state_non_finite".into(),
                ));
            }
            state = next;
            steps = step + 1;
            if final_max_delta <= self.config.convergence_tolerance {
                converged = true;
                break;
            }
        }
        let coalitions = self.coalitions(&state)?;
        let ignition_coalition = coalitions.iter().position(|coalition| {
            coalition.field_ids.len() >= self.config.minimum_ignition_fields
                && coalition.salience >= self.config.ignition_threshold
        });
        Ok(CognitiveFieldState {
            schema: "cerebro.tidex.cognitive_field_state/v1".into(),
            field_ids: self.field_ids.clone(),
            activations: state.clone(),
            steps,
            converged,
            final_max_delta,
            attractor_digest: attractor_digest(&self.field_ids, &state),
            coalitions,
            global_ignition: ignition_coalition.is_some(),
            ignition_coalition,
        })
    }

    fn coalitions(&self, activations: &[f64]) -> BrainResult<Vec<CognitiveCoalition>> {
        let (coupling, _) = self.matrices()?;
        if activations.len() != self.field_ids.len() {
            return Err(BrainError::Invalid(
                "cognitive_field_activation_shape".into(),
            ));
        }
        let active = activations
            .iter()
            .enumerate()
            .filter_map(|(index, activation)| {
                (*activation >= self.config.coalition_activation_threshold).then_some(index)
            })
            .collect::<BTreeSet<_>>();
        let mut visited = BTreeSet::new();
        let mut coalitions = Vec::new();
        for &start in &active {
            if !visited.insert(start) {
                continue;
            }
            let mut queue = VecDeque::from([start]);
            let mut component = vec![start];
            while let Some(node) = queue.pop_front() {
                for &candidate in &active {
                    if visited.contains(&candidate) || candidate == node {
                        continue;
                    }
                    if coupling.get(node, candidate) >= self.config.coalition_coupling_threshold {
                        visited.insert(candidate);
                        queue.push_back(candidate);
                        component.push(candidate);
                    }
                }
            }
            component.sort_unstable();
            let mean_activation = component
                .iter()
                .map(|index| activations[*index])
                .sum::<f64>()
                / component.len() as f64;
            let mut internal = Vec::new();
            for left in 0..component.len() {
                for right in 0..left {
                    internal.push(coupling.get(component[left], component[right]).max(0.0));
                }
            }
            let internal_coherence = if internal.is_empty() {
                0.0
            } else {
                internal.iter().sum::<f64>() / internal.len() as f64
            };
            let salience = mean_activation.max(0.0) * (0.5 + 0.5 * internal_coherence);
            coalitions.push(CognitiveCoalition {
                field_ids: component
                    .iter()
                    .map(|index| self.field_ids[*index].clone())
                    .collect(),
                mean_activation,
                internal_coherence,
                salience,
            });
        }
        coalitions.sort_by(|left, right| right.salience.total_cmp(&left.salience));
        Ok(coalitions)
    }

    pub fn route_top_k(
        &self,
        state: &CognitiveFieldState,
        top_k: usize,
        minimum_activation: f64,
    ) -> BrainResult<FieldRoutingDecision> {
        if !state.converged
            || state.field_ids != self.field_ids
            || state.activations.len() != self.field_ids.len()
            || top_k == 0
            || !minimum_activation.is_finite()
        {
            return Err(BrainError::Invalid(
                "cognitive_field_routing_contract".into(),
            ));
        }
        let mut order = state
            .activations
            .iter()
            .enumerate()
            .filter(|(_, activation)| **activation >= minimum_activation)
            .map(|(index, activation)| (index, *activation))
            .collect::<Vec<_>>();
        order.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| self.field_ids[left.0].cmp(&self.field_ids[right.0]))
        });
        order.truncate(top_k.min(order.len()));
        if order.is_empty() {
            return Err(BrainError::Invalid(
                "cognitive_field_no_routable_activation".into(),
            ));
        }
        let mut coefficients = vec![0.0; self.field_ids.len()];
        let selected_activation_mass = order.iter().map(|(_, value)| value.max(0.0)).sum::<f64>();
        if selected_activation_mass <= 0.0 {
            return Err(BrainError::Invalid(
                "cognitive_field_nonpositive_activation_mass".into(),
            ));
        }
        let mut selected_field_ids = Vec::with_capacity(order.len());
        for (index, activation) in order {
            coefficients[index] = activation.max(0.0) / selected_activation_mass;
            selected_field_ids.push(self.field_ids[index].clone());
        }
        Ok(FieldRoutingDecision {
            schema: "cerebro.tidex.cognitive_field_routing/v1".into(),
            field_ids: self.field_ids.clone(),
            coefficients,
            selected_field_ids,
            selected_activation_mass,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_credit::{FieldCausalCredit, PairInteractionCredit};

    fn field(id: &str, functional: Vec<f64>) -> SkillField {
        SkillField {
            skill_id: SkillId::parse(id).unwrap(),
            reconstruction_id: Default::default(),
            lineage_id: Default::default(),
            generation_created: 1,
            direction: vec![1.0, 0.0],
            structured_geometry: None,
            dense_materialization: None,
            parameter_layout_sha256: None,
            representation_signature: Vec::new(),
            singular_value: 1.0,
            explained_variance: 0.5,
            persistence: 1.0,
            coherence: 1.0,
            uncertainty: 0.1,
            evidence_support_digests: Vec::new(),
            support: 3,
            functional_signature: functional,
            parent_skill_ids: Vec::new(),
        }
    }

    fn causal() -> CausalCreditReport {
        CausalCreditReport {
            schema: "cerebro.tidex.causal_credit/v3".into(),
            context_count: 12,
            independent_group_count: 4,
            field_count: 3,
            fields: ["a", "b", "c"]
                .into_iter()
                .map(|id| FieldCausalCredit {
                    skill_id: SkillId::parse(id).unwrap(),
                    matched_pairs: 16,
                    independent_contexts: 4,
                    mean_marginal_effect: 0.2,
                    standard_error: 0.01,
                    lower_confidence_bound: 0.18,
                    positive_fraction: 1.0,
                    resolved: true,
                    beneficial: true,
                })
                .collect(),
            pair_interactions: vec![
                PairInteractionCredit {
                    left_skill_id: SkillId::parse("a").unwrap(),
                    right_skill_id: SkillId::parse("b").unwrap(),
                    matched_quads: 8,
                    independent_contexts: 4,
                    mean_interaction_effect: 0.3,
                    standard_error: 0.01,
                    resolved: true,
                },
                PairInteractionCredit {
                    left_skill_id: SkillId::parse("a").unwrap(),
                    right_skill_id: SkillId::parse("c").unwrap(),
                    matched_quads: 8,
                    independent_contexts: 4,
                    mean_interaction_effect: -0.2,
                    standard_error: 0.01,
                    resolved: true,
                },
                PairInteractionCredit {
                    left_skill_id: SkillId::parse("b").unwrap(),
                    right_skill_id: SkillId::parse("c").unwrap(),
                    matched_quads: 8,
                    independent_contexts: 4,
                    mean_interaction_effect: -0.2,
                    standard_error: 0.01,
                    resolved: true,
                },
            ],
            unresolved_fields: Vec::new(),
        }
    }

    #[test]
    fn cooperative_fields_form_a_coalition_and_ignite() {
        let fields = vec![
            field("a", vec![1.0, 0.0]),
            field("b", vec![0.98, 0.02]),
            field("c", vec![-1.0, 0.0]),
        ];
        let curvature = Matrix::from_rows(&[
            vec![1.0, 0.8, -0.1],
            vec![0.8, 1.0, -0.1],
            vec![-0.1, -0.1, 1.0],
        ])
        .unwrap();
        let config = CognitiveFieldConfig {
            ignition_threshold: 0.20,
            ..CognitiveFieldConfig::default()
        };
        let model = DynamicCognitiveField::build(&fields, &curvature, &causal(), config).unwrap();
        let state = model
            .evolve(
                &[0.0; 3],
                &CognitiveFieldDrive {
                    evidence: vec![1.0, 0.9, 0.0],
                    prediction_error: vec![0.0; 3],
                    inhibition: vec![0.0; 3],
                    risk: vec![0.0; 3],
                },
            )
            .unwrap();
        assert!(state.converged, "delta={}", state.final_max_delta);
        assert!(state.global_ignition);
        assert!(state
            .coalitions
            .iter()
            .any(|coalition| { coalition.field_ids == vec!["a".to_string(), "b".to_string()] }));
    }

    #[test]
    fn cognitive_field_rejects_incomplete_causal_graph() {
        let fields = vec![
            field("a", vec![1.0, 0.0]),
            field("b", vec![0.9, 0.1]),
            field("c", vec![-1.0, 0.0]),
        ];
        let mut causal = causal();
        causal.pair_interactions.pop();
        assert!(DynamicCognitiveField::build(
            &fields,
            &Matrix::identity(3),
            &causal,
            CognitiveFieldConfig::default(),
        )
        .is_err());
    }

    #[test]
    fn uncertain_pairwise_synergy_is_routed_as_possible_interference() {
        let fields = vec![
            field("a", vec![1.0, 0.0]),
            field("b", vec![1.0, 0.0]),
            field("c", vec![1.0, 0.0]),
        ];
        let mut evidence = causal();
        let pair = evidence
            .pair_interactions
            .iter_mut()
            .find(|pair| pair.left_skill_id == "a" && pair.right_skill_id == "b")
            .unwrap();
        pair.mean_interaction_effect = 0.3;
        pair.standard_error = 0.3;
        let config = CognitiveFieldConfig {
            curvature_weight: 0.0,
            causal_interaction_weight: 1.0,
            functional_compatibility_weight: 0.0,
            coupling_gain: 1.0,
            ..CognitiveFieldConfig::default()
        };
        let model =
            DynamicCognitiveField::build(&fields, &Matrix::identity(3), &evidence, config).unwrap();
        assert!(model.coupling_matrix[0][1] < 0.0);
    }

    #[test]
    fn cognitive_field_rejects_indefinite_curvature() {
        let fields = vec![
            field("a", vec![1.0, 0.0]),
            field("b", vec![0.9, 0.1]),
            field("c", vec![-1.0, 0.0]),
        ];
        let curvature = Matrix::from_rows(&[
            vec![1.0, 2.0, 0.0],
            vec![2.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ])
        .unwrap();
        assert!(DynamicCognitiveField::build(
            &fields,
            &curvature,
            &causal(),
            CognitiveFieldConfig::default(),
        )
        .is_err());
    }

    #[test]
    fn top_k_routing_emerges_from_converged_field_state() {
        let fields = vec![
            field("a", vec![1.0, 0.0]),
            field("b", vec![0.9, 0.1]),
            field("c", vec![-1.0, 0.0]),
        ];
        let curvature = Matrix::identity(3);
        let model = DynamicCognitiveField::build(
            &fields,
            &curvature,
            &causal(),
            CognitiveFieldConfig::default(),
        )
        .unwrap();
        let state = model
            .evolve(
                &[0.0; 3],
                &CognitiveFieldDrive {
                    evidence: vec![1.0, 0.5, 0.1],
                    prediction_error: vec![0.0; 3],
                    inhibition: vec![0.0; 3],
                    risk: vec![0.0; 3],
                },
            )
            .unwrap();
        assert!(state.converged);
        let route = model.route_top_k(&state, 2, 0.0).unwrap();
        assert_eq!(route.selected_field_ids.len(), 2);
        assert!((route.coefficients.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
