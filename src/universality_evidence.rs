//! Reproducible measurement of experimental universality N.
//!
//! N counts held-out capabilities that pass receiver/family/seed coverage,
//! zero-target-optimization, preservation, negative-control, and confidence
//! gates. The reducer checks supplied observations and label consistency; it
//! does not authenticate their execution or establish statistical independence.
//! Its confidence bounds are conditional on the declared experimental units.

use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const MAX_TRIALS: usize = 1_000_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalityProtocol {
    pub schema: String,
    pub minimum_calibration_capabilities: usize,
    pub minimum_held_out_capabilities: usize,
    pub minimum_receivers_per_capability: usize,
    pub minimum_receiver_families_per_capability: usize,
    pub minimum_seeds_per_capability: usize,
    pub require_unseen_receiver: bool,
    pub minimum_target_score: f64,
    pub minimum_preservation_score: f64,
    pub minimum_identity_margin: f64,
    pub minimum_success_probability: f64,
    pub confidence_z: f64,
}

impl UniversalityProtocol {
    pub fn validate(&self) -> BrainResult<()> {
        let unit = [
            self.minimum_target_score,
            self.minimum_preservation_score,
            self.minimum_success_probability,
        ];
        if self.schema != "cerebro.tidex.universality_protocol/v1"
            || self.minimum_calibration_capabilities == 0
            || self.minimum_held_out_capabilities == 0
            || self.minimum_receivers_per_capability == 0
            || self.minimum_receiver_families_per_capability == 0
            || self.minimum_seeds_per_capability == 0
            || unit
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || !self.minimum_identity_margin.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_identity_margin)
            || self.minimum_success_probability == 0.0
            || !self.confidence_z.is_finite()
            || !(0.0..=8.0).contains(&self.confidence_z)
            || self.confidence_z == 0.0
        {
            return Err(BrainError::Invalid("universality_protocol_invalid".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalityTrial {
    pub schema: String,
    pub trial_id: String,
    pub capability_id: String,
    pub receiver_id: String,
    pub receiver_family_id: String,
    pub seed: u64,
    pub capability_was_calibration: bool,
    pub receiver_was_calibration: bool,
    pub target_optimizer_steps: u64,
    pub target_score: f64,
    pub preservation_score: f64,
    pub wrong_ir_score: f64,
    pub random_delta_score: f64,
    pub unmodified_receiver_score: f64,
}

impl UniversalityTrial {
    fn validate(&self) -> BrainResult<()> {
        let values = [
            self.target_score,
            self.preservation_score,
            self.wrong_ir_score,
            self.random_delta_score,
            self.unmodified_receiver_score,
        ];
        if self.schema != "cerebro.tidex.universality_trial/v1"
            || [
                &self.trial_id,
                &self.capability_id,
                &self.receiver_id,
                &self.receiver_family_id,
            ]
            .iter()
            .any(|v| {
                v.trim().is_empty()
                    || v.as_str() != v.trim()
                    || v.len() > 4096
                    || v.chars().any(char::is_control)
            })
            || values
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(BrainError::Invalid("universality_trial_invalid".into()));
        }
        Ok(())
    }
    fn passes(&self, protocol: &UniversalityProtocol) -> bool {
        let strongest_control = self
            .wrong_ir_score
            .max(self.random_delta_score)
            .max(self.unmodified_receiver_score);
        !self.capability_was_calibration
            && self.target_optimizer_steps == 0
            && self.target_score >= protocol.minimum_target_score
            && self.preservation_score >= protocol.minimum_preservation_score
            && self.target_score - strongest_control >= protocol.minimum_identity_margin
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CapabilityUniversalityEvidence {
    pub capability_id: String,
    pub trials: usize,
    pub successes: usize,
    pub receivers: usize,
    pub receiver_families: usize,
    pub seeds: usize,
    pub includes_unseen_receiver: bool,
    pub successful_receivers: usize,
    pub successful_receiver_families: usize,
    pub successful_seeds: usize,
    pub unseen_receiver_successes: usize,
    pub minimum_receiver_wilson_lower_bound: f64,
    pub minimum_family_wilson_lower_bound: f64,
    pub success_rate: f64,
    pub wilson_lower_bound: f64,
    pub admitted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalityEvidenceReceipt {
    pub schema: String,
    pub protocol_sha256: Sha256Digest,
    pub trials_sha256: Sha256Digest,
    pub calibration_capability_count: usize,
    pub held_out_capability_count: usize,
    pub universality_n: usize,
    pub global_success_rate: f64,
    pub global_wilson_lower_bound: f64,
    pub capabilities: Vec<CapabilityUniversalityEvidence>,
    pub manifest_sha256: Sha256Digest,
}

impl UniversalityEvidenceReceipt {
    pub fn validate_against(&self, input: &UniversalityEvidenceInput) -> BrainResult<()> {
        if self != &input.execute()? {
            return Err(BrainError::Integrity(
                "universality_evidence_receipt_invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalityEvidenceInput {
    pub schema: String,
    pub calibration_capabilities: BTreeSet<String>,
    pub trials: Vec<UniversalityTrial>,
    pub protocol: UniversalityProtocol,
}

impl UniversalityEvidenceInput {
    pub fn execute(&self) -> BrainResult<UniversalityEvidenceReceipt> {
        if self.schema != "cerebro.tidex.universality_evidence_input/v1" {
            return Err(BrainError::Invalid(
                "universality_evidence_input_invalid".into(),
            ));
        }
        measure_universality_n(&self.calibration_capabilities, &self.trials, &self.protocol)
    }
}

fn wilson_lower(successes: usize, total: usize, z: f64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    ((p + z2 / (2.0 * n) - z * ((p * (1.0 - p) + z2 / (4.0 * n)) / n).sqrt()) / (1.0 + z2 / n))
        .clamp(0.0, 1.0)
}

pub fn measure_universality_n(
    calibration_capabilities: &BTreeSet<String>,
    trials: &[UniversalityTrial],
    protocol: &UniversalityProtocol,
) -> BrainResult<UniversalityEvidenceReceipt> {
    protocol.validate()?;
    if calibration_capabilities.len() < protocol.minimum_calibration_capabilities
        || trials.is_empty()
        || trials.len() > MAX_TRIALS
        || calibration_capabilities.iter().any(|v| v.trim().is_empty())
    {
        return Err(BrainError::Invalid(
            "universality_evidence_cardinality_invalid".into(),
        ));
    }
    let mut ids = BTreeSet::new();
    let mut design_units = BTreeSet::new();
    let mut receiver_labels = BTreeMap::new();
    let mut observed_calibration = BTreeSet::new();
    let mut groups: BTreeMap<&str, Vec<&UniversalityTrial>> = BTreeMap::new();
    for trial in trials {
        trial.validate()?;
        if !ids.insert(&trial.trial_id) {
            return Err(BrainError::Invalid("universality_trial_duplicate".into()));
        }
        let labels = (
            trial.receiver_family_id.as_str(),
            trial.receiver_was_calibration,
        );
        if receiver_labels
            .insert(trial.receiver_id.as_str(), labels)
            .is_some_and(|prior| prior != labels)
        {
            return Err(BrainError::Integrity(
                "universality_receiver_relabelled".into(),
            ));
        }
        if !design_units.insert((
            trial.capability_id.as_str(),
            trial.receiver_id.as_str(),
            trial.seed,
        )) {
            return Err(BrainError::Invalid(
                "universality_design_unit_reused".into(),
            ));
        }
        if trial.capability_was_calibration
            != calibration_capabilities.contains(&trial.capability_id)
        {
            return Err(BrainError::Integrity(
                "universality_calibration_label_mismatch".into(),
            ));
        }
        if trial.capability_was_calibration {
            observed_calibration.insert(&trial.capability_id);
        } else {
            groups.entry(&trial.capability_id).or_default().push(trial);
        }
    }
    if calibration_capabilities
        .iter()
        .any(|capability| !observed_calibration.contains(capability))
    {
        return Err(BrainError::Invalid(
            "universality_calibration_evidence_missing".into(),
        ));
    }
    if groups.len() < protocol.minimum_held_out_capabilities {
        return Err(BrainError::Invalid(
            "universality_held_out_coverage_insufficient".into(),
        ));
    }
    let mut capabilities = Vec::with_capacity(groups.len());
    let mut global_successes = 0usize;
    let mut global_trials = 0usize;
    for (capability_id, group) in groups {
        let successes = group.iter().filter(|trial| trial.passes(protocol)).count();
        let receivers = group
            .iter()
            .map(|t| &t.receiver_id)
            .collect::<BTreeSet<_>>()
            .len();
        let families = group
            .iter()
            .map(|t| &t.receiver_family_id)
            .collect::<BTreeSet<_>>()
            .len();
        let seeds = group.iter().map(|t| t.seed).collect::<BTreeSet<_>>().len();
        let includes_unseen = group.iter().any(|t| !t.receiver_was_calibration);
        let lower = wilson_lower(successes, group.len(), protocol.confidence_z);
        let successful = group
            .iter()
            .copied()
            .filter(|t| t.passes(protocol))
            .collect::<Vec<_>>();
        let successful_receivers = successful
            .iter()
            .map(|t| &t.receiver_id)
            .collect::<BTreeSet<_>>()
            .len();
        let successful_families = successful
            .iter()
            .map(|t| &t.receiver_family_id)
            .collect::<BTreeSet<_>>()
            .len();
        let successful_seeds = successful
            .iter()
            .map(|t| t.seed)
            .collect::<BTreeSet<_>>()
            .len();
        let unseen_successes = successful
            .iter()
            .filter(|t| !t.receiver_was_calibration)
            .count();
        let mut by_receiver = BTreeMap::<&str, (usize, usize)>::new();
        let mut by_family = BTreeMap::<&str, (usize, usize)>::new();
        for trial in &group {
            for entry in [
                by_receiver.entry(&trial.receiver_id).or_default(),
                by_family.entry(&trial.receiver_family_id).or_default(),
            ] {
                entry.0 += usize::from(trial.passes(protocol));
                entry.1 += 1;
            }
        }
        let receiver_lower = by_receiver
            .values()
            .map(|(s, n)| wilson_lower(*s, *n, protocol.confidence_z))
            .fold(1.0, f64::min);
        let family_lower = by_family
            .values()
            .map(|(s, n)| wilson_lower(*s, *n, protocol.confidence_z))
            .fold(1.0, f64::min);
        let admitted = receivers >= protocol.minimum_receivers_per_capability
            && families >= protocol.minimum_receiver_families_per_capability
            && seeds >= protocol.minimum_seeds_per_capability
            && (!protocol.require_unseen_receiver || includes_unseen)
            && successful_receivers >= protocol.minimum_receivers_per_capability
            && successful_families >= protocol.minimum_receiver_families_per_capability
            && successful_seeds >= protocol.minimum_seeds_per_capability
            && (!protocol.require_unseen_receiver || unseen_successes > 0)
            && receiver_lower >= protocol.minimum_success_probability
            && family_lower >= protocol.minimum_success_probability
            && lower >= protocol.minimum_success_probability;
        global_successes += successes;
        global_trials += group.len();
        capabilities.push(CapabilityUniversalityEvidence {
            capability_id: capability_id.into(),
            trials: group.len(),
            successes,
            receivers,
            receiver_families: families,
            seeds,
            includes_unseen_receiver: includes_unseen,
            successful_receivers,
            successful_receiver_families: successful_families,
            successful_seeds,
            unseen_receiver_successes: unseen_successes,
            minimum_receiver_wilson_lower_bound: receiver_lower,
            minimum_family_wilson_lower_bound: family_lower,
            success_rate: successes as f64 / group.len() as f64,
            wilson_lower_bound: lower,
            admitted,
        });
    }
    let universality_n = capabilities.iter().filter(|c| c.admitted).count();
    let protocol_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:UNIVERSALITY-PROTOCOL:v1\0",
        &serde_json::to_vec(protocol)?,
    );
    let mut canonical_trials = trials.iter().collect::<Vec<_>>();
    canonical_trials.sort_by(|a, b| a.trial_id.cmp(&b.trial_id));
    let trials_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:UNIVERSALITY-TRIALS:v2\0",
        &serde_json::to_vec(&(calibration_capabilities, canonical_trials))?,
    );
    let mut receipt = UniversalityEvidenceReceipt {
        schema: "cerebro.tidex.universality_evidence_receipt/v2".into(),
        protocol_sha256,
        trials_sha256,
        calibration_capability_count: calibration_capabilities.len(),
        held_out_capability_count: capabilities.len(),
        universality_n,
        global_success_rate: global_successes as f64 / global_trials as f64,
        global_wilson_lower_bound: wilson_lower(
            global_successes,
            global_trials,
            protocol.confidence_z,
        ),
        capabilities,
        manifest_sha256: Sha256Digest::zero(),
    };
    let mut unsigned = receipt.clone();
    unsigned.manifest_sha256 = Sha256Digest::zero();
    receipt.manifest_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:UNIVERSALITY-EVIDENCE-RECEIPT:v2\0",
        &serde_json::to_vec(&unsigned)?,
    );
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn evidence_fixture() -> (
        BTreeSet<String>,
        Vec<UniversalityTrial>,
        UniversalityProtocol,
    ) {
        let protocol = UniversalityProtocol {
            schema: "cerebro.tidex.universality_protocol/v1".into(),
            minimum_calibration_capabilities: 1,
            minimum_held_out_capabilities: 1,
            minimum_receivers_per_capability: 2,
            minimum_receiver_families_per_capability: 2,
            minimum_seeds_per_capability: 2,
            require_unseen_receiver: true,
            minimum_target_score: 0.8,
            minimum_preservation_score: 0.9,
            minimum_identity_margin: 0.2,
            minimum_success_probability: 0.5,
            confidence_z: 1.0,
        };
        let trial = |id: &str, receiver: &str, family: &str, seed| UniversalityTrial {
            schema: "cerebro.tidex.universality_trial/v1".into(),
            trial_id: id.into(),
            capability_id: "held.out".into(),
            receiver_id: receiver.into(),
            receiver_family_id: family.into(),
            seed,
            capability_was_calibration: false,
            receiver_was_calibration: receiver == "seen",
            target_optimizer_steps: 0,
            target_score: 0.95,
            preservation_score: 0.99,
            wrong_ir_score: 0.1,
            random_delta_score: 0.1,
            unmodified_receiver_score: 0.2,
        };
        let calibration_trial = UniversalityTrial {
            schema: "cerebro.tidex.universality_trial/v1".into(),
            trial_id: "calibration-trial".into(),
            capability_id: "calibration".into(),
            receiver_id: "seen".into(),
            receiver_family_id: "a".into(),
            seed: 0,
            capability_was_calibration: true,
            receiver_was_calibration: true,
            target_optimizer_steps: 0,
            target_score: 1.0,
            preservation_score: 1.0,
            wrong_ir_score: 0.0,
            random_delta_score: 0.0,
            unmodified_receiver_score: 0.0,
        };
        (
            BTreeSet::from(["calibration".into()]),
            vec![
                calibration_trial,
                trial("1", "seen", "a", 1),
                trial("2", "unseen", "b", 2),
                trial("3", "unseen", "b", 3),
            ],
            protocol,
        )
    }

    #[test]
    fn n_is_computed_only_from_held_out_zero_step_evidence() {
        let (calibration, trials, protocol) = evidence_fixture();
        assert_eq!(
            measure_universality_n(&calibration, &trials, &protocol)
                .unwrap()
                .universality_n,
            1
        );
    }

    #[test]
    fn convergence_regression_universality_rejects_receiver_relabeling() {
        let (calibration, trials, protocol) = evidence_fixture();
        let mut wrong = trials.clone();
        wrong[2].receiver_family_id = "manufactured-family".into();
        assert!(measure_universality_n(&calibration, &wrong, &protocol).is_err());
        let mut wrong = trials.clone();
        wrong[2].receiver_was_calibration = true;
        assert!(measure_universality_n(&calibration, &wrong, &protocol).is_err());
        let original = measure_universality_n(&calibration, &trials, &protocol).unwrap();
        let mut reordered = trials;
        reordered.reverse();
        assert_eq!(
            original,
            measure_universality_n(&calibration, &reordered, &protocol).unwrap()
        );
    }

    #[test]
    fn convergence_regression_unseen_receiver_failures_cannot_establish_universality() {
        let (calibration, mut trials, mut protocol) = evidence_fixture();
        protocol.minimum_success_probability = 0.1;
        for trial in &mut trials {
            if !trial.receiver_was_calibration {
                trial.target_score = 0.0;
            }
        }
        let receipt = measure_universality_n(&calibration, &trials, &protocol).unwrap();
        assert_eq!(receipt.universality_n, 0);
        assert_eq!(receipt.capabilities[0].unseen_receiver_successes, 0);
    }
}
