//! Behavioral discovery evidence preceding CapabilityIR construction.

use crate::architecture_families::{ArchitectureFamilyFingerprint, ModuleFamily};
use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::linalg::{cosine, norm};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const MAX_TRIALS: usize = 1_000_000;
const MAX_SIGNATURE_DIMENSION: usize = 1_048_576;
const MAX_TOTAL_SIGNATURE_ELEMENTS: usize = 16 * 1024 * 1024;
const MAX_DISCOVERY_WORK_UNITS: usize = 250_000_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CapabilityDiscoveryPolicy {
    pub schema: String,
    pub minimum_trials: usize,
    pub minimum_seeds: usize,
    pub minimum_consistency: f64,
    pub minimum_control_margin: f64,
    pub maximum_closure_error: f64,
    pub maximum_contraction_ratio: f64,
}

impl CapabilityDiscoveryPolicy {
    pub fn validate(&self) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.capability_discovery_policy/v1"
            || self.minimum_trials < 2
            || self.minimum_seeds < 2
            || !self.minimum_consistency.is_finite()
            || !(-1.0..=1.0).contains(&self.minimum_consistency)
            || !self.minimum_control_margin.is_finite()
            || !(-2.0..=2.0).contains(&self.minimum_control_margin)
            || !self.maximum_closure_error.is_finite()
            || self.maximum_closure_error < 0.0
            || !self.maximum_contraction_ratio.is_finite()
            || self.maximum_contraction_ratio < 0.0
        {
            return Err(BrainError::Invalid(
                "capability_discovery_policy_invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CapabilityProbeTrial {
    pub schema: String,
    pub trial_id: String,
    pub probe_id: String,
    pub seed: u64,
    pub input_sha256: Sha256Digest,
    pub output_sha256: Sha256Digest,
    pub functional_signature: Vec<f64>,
    pub wrong_control_signature: Vec<f64>,
    pub closure_error: f64,
    pub contraction_ratio: f64,
    pub attributed_module_families: BTreeSet<ModuleFamily>,
}

impl CapabilityProbeTrial {
    fn validate(&self) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.capability_probe_trial/v1"
            || self.trial_id.trim().is_empty()
            || self.probe_id.trim().is_empty()
            || self.functional_signature.is_empty()
            || self.functional_signature.len() > MAX_SIGNATURE_DIMENSION
            || self.functional_signature.len() != self.wrong_control_signature.len()
            || self
                .functional_signature
                .iter()
                .chain(&self.wrong_control_signature)
                .any(|value| !value.is_finite())
            || norm(&self.functional_signature)? <= 1e-15
            || norm(&self.wrong_control_signature)? <= 1e-15
            || !self.closure_error.is_finite()
            || self.closure_error < 0.0
            || !self.contraction_ratio.is_finite()
            || self.contraction_ratio < 0.0
            || self.attributed_module_families.is_empty()
        {
            return Err(BrainError::Invalid("capability_probe_trial_invalid".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityDiscoveryDisposition {
    EvidenceReadyForCapabilityIr,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DiscoveredCapabilityEvidence {
    pub probe_id: String,
    pub trial_count: usize,
    pub distinct_seeds: usize,
    pub mean_signature: Vec<f64>,
    pub minimum_consistency: f64,
    pub minimum_control_margin: f64,
    pub maximum_closure_error: f64,
    pub maximum_contraction_ratio: f64,
    pub module_families: BTreeSet<ModuleFamily>,
    pub disposition: CapabilityDiscoveryDisposition,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CapabilityDiscoveryReport {
    pub schema: String,
    pub architecture_fingerprint_sha256: Sha256Digest,
    pub policy_sha256: Sha256Digest,
    pub trials_sha256: Sha256Digest,
    pub capabilities: Vec<DiscoveredCapabilityEvidence>,
    pub manifest_sha256: Sha256Digest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CapabilityDiscoveryRequest {
    pub schema: String,
    pub architecture_fingerprint: ArchitectureFamilyFingerprint,
    pub trials: Vec<CapabilityProbeTrial>,
    pub policy: CapabilityDiscoveryPolicy,
}

impl CapabilityDiscoveryRequest {
    pub fn execute(&self) -> BrainResult<CapabilityDiscoveryReport> {
        if self.schema != "cerebro.tidex.capability_discovery_request/v1" {
            return Err(BrainError::Invalid(
                "capability_discovery_request_invalid".into(),
            ));
        }
        discover_capabilities(&self.architecture_fingerprint, &self.trials, &self.policy)
    }
}

fn analyze_group(
    probe_id: &str,
    group: &[&CapabilityProbeTrial],
    policy: &CapabilityDiscoveryPolicy,
) -> BrainResult<DiscoveredCapabilityEvidence> {
    let dimension = group[0].functional_signature.len();
    if group
        .iter()
        .any(|trial| trial.functional_signature.len() != dimension)
    {
        return Err(BrainError::Invalid(
            "capability_probe_dimension_mismatch".into(),
        ));
    }
    let mut mean = vec![0.0; dimension];
    for trial in group {
        for (aggregate, value) in mean.iter_mut().zip(&trial.functional_signature) {
            *aggregate += value / group.len() as f64;
        }
    }
    let mut consistency = 1.0_f64;
    for first in 0..group.len() {
        for second in first + 1..group.len() {
            consistency = consistency.min(cosine(
                &group[first].functional_signature,
                &group[second].functional_signature,
            )?);
        }
    }
    let control_margin = group
        .iter()
        .map(|trial| {
            Ok(cosine(&trial.functional_signature, &mean)?
                - cosine(&trial.wrong_control_signature, &mean)?)
        })
        .collect::<BrainResult<Vec<_>>>()?
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let closure = group
        .iter()
        .map(|trial| trial.closure_error)
        .fold(0.0_f64, f64::max);
    let contraction = group
        .iter()
        .map(|trial| trial.contraction_ratio)
        .fold(0.0_f64, f64::max);
    let seeds = group
        .iter()
        .map(|trial| trial.seed)
        .collect::<BTreeSet<_>>();
    let module_families = group
        .iter()
        .flat_map(|trial| trial.attributed_module_families.iter().copied())
        .collect::<BTreeSet<_>>();
    let admitted = group.len() >= policy.minimum_trials
        && seeds.len() >= policy.minimum_seeds
        && consistency >= policy.minimum_consistency
        && control_margin >= policy.minimum_control_margin
        && closure <= policy.maximum_closure_error
        && contraction <= policy.maximum_contraction_ratio;
    Ok(DiscoveredCapabilityEvidence {
        probe_id: probe_id.into(),
        trial_count: group.len(),
        distinct_seeds: seeds.len(),
        mean_signature: mean,
        minimum_consistency: consistency,
        minimum_control_margin: control_margin,
        maximum_closure_error: closure,
        maximum_contraction_ratio: contraction,
        module_families,
        disposition: if admitted {
            CapabilityDiscoveryDisposition::EvidenceReadyForCapabilityIr
        } else {
            CapabilityDiscoveryDisposition::Rejected
        },
    })
}

pub fn discover_capabilities(
    fingerprint: &ArchitectureFamilyFingerprint,
    trials: &[CapabilityProbeTrial],
    policy: &CapabilityDiscoveryPolicy,
) -> BrainResult<CapabilityDiscoveryReport> {
    policy.validate()?;
    fingerprint.validate()?;
    if trials.is_empty() || trials.len() > MAX_TRIALS {
        return Err(BrainError::Invalid(
            "capability_discovery_input_invalid".into(),
        ));
    }
    let elements = trials.iter().try_fold(0usize, |total, trial| {
        total
            .checked_add(trial.functional_signature.len())
            .and_then(|n| n.checked_add(trial.wrong_control_signature.len()))
    });
    if elements.is_none_or(|n| n > MAX_TOTAL_SIGNATURE_ELEMENTS) {
        return Err(BrainError::Invalid(
            "capability_discovery_storage_limit".into(),
        ));
    }
    let mut ids = BTreeSet::new();
    let mut groups: BTreeMap<&str, Vec<&CapabilityProbeTrial>> = BTreeMap::new();
    for trial in trials {
        trial.validate()?;
        if !ids.insert(&trial.trial_id) {
            return Err(BrainError::Invalid(
                "capability_probe_trial_duplicate".into(),
            ));
        }
        groups.entry(&trial.probe_id).or_default().push(trial);
    }
    let work = groups.values().try_fold(0usize, |total, group| {
        let n = group.len();
        n.checked_mul(n.saturating_sub(1))
            .map(|pairs| pairs / 2)
            .and_then(|pairs| pairs.checked_add(n * 2))
            .and_then(|comparisons| comparisons.checked_mul(group[0].functional_signature.len()))
            .and_then(|work| work.checked_mul(8))
            .and_then(|work| total.checked_add(work))
    });
    if work.is_none_or(|n| n > MAX_DISCOVERY_WORK_UNITS) {
        return Err(BrainError::Invalid(
            "capability_discovery_work_limit".into(),
        ));
    }
    let capabilities = groups
        .into_iter()
        .map(|(probe, group)| analyze_group(probe, &group, policy))
        .collect::<BrainResult<Vec<_>>>()?;
    let policy_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:CAPABILITY-DISCOVERY-POLICY:v1\0",
        &serde_json::to_vec(policy)?,
    );
    let trials_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:CAPABILITY-PROBE-TRIALS:v1\0",
        &serde_json::to_vec(trials)?,
    );
    let mut report = CapabilityDiscoveryReport {
        schema: "cerebro.tidex.capability_discovery_report/v1".into(),
        architecture_fingerprint_sha256: fingerprint.manifest_sha256.clone(),
        policy_sha256,
        trials_sha256,
        capabilities,
        manifest_sha256: Sha256Digest::zero(),
    };
    let mut unsigned = report.clone();
    unsigned.manifest_sha256 = Sha256Digest::zero();
    report.manifest_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:CAPABILITY-DISCOVERY-REPORT:v1\0",
        &serde_json::to_vec(&unsigned)?,
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture_families::fingerprint_architecture;

    #[test]
    fn repeated_behavior_not_module_name_admits_discovery_evidence() {
        let fingerprint = fingerprint_architecture(
            br#"{"model_type":"llama"}"#,
            &["layers.0.self_attn.q_proj.weight".into()],
        )
        .unwrap();
        let make_trial = |id: &str, seed| CapabilityProbeTrial {
            schema: "cerebro.tidex.capability_probe_trial/v1".into(),
            trial_id: id.into(),
            probe_id: "tool.call:v1".into(),
            seed,
            input_sha256: Sha256Digest::digest_bytes(id.as_bytes()),
            output_sha256: Sha256Digest::digest_bytes(format!("out-{id}").as_bytes()),
            functional_signature: vec![1.0, 0.0],
            wrong_control_signature: vec![0.0, 1.0],
            closure_error: 0.0,
            contraction_ratio: 0.0,
            attributed_module_families: BTreeSet::from([ModuleFamily::Attention]),
        };
        let report = discover_capabilities(
            &fingerprint,
            &[make_trial("one", 1), make_trial("two", 2)],
            &CapabilityDiscoveryPolicy {
                schema: "cerebro.tidex.capability_discovery_policy/v1".into(),
                minimum_trials: 2,
                minimum_seeds: 2,
                minimum_consistency: 0.99,
                minimum_control_margin: 0.5,
                maximum_closure_error: 0.01,
                maximum_contraction_ratio: 0.01,
            },
        )
        .unwrap();
        assert_eq!(
            report.capabilities[0].disposition,
            CapabilityDiscoveryDisposition::EvidenceReadyForCapabilityIr
        );
        let oversized = (0..1024)
            .map(|seed| {
                let mut trial = make_trial(&format!("bounded-{seed}"), seed);
                trial.functional_signature = vec![1.0; 1024];
                trial.wrong_control_signature = vec![0.5; 1024];
                trial
            })
            .collect::<Vec<_>>();
        let policy = CapabilityDiscoveryPolicy {
            schema: "cerebro.tidex.capability_discovery_policy/v1".into(),
            minimum_trials: 2,
            minimum_seeds: 2,
            minimum_consistency: 0.99,
            minimum_control_margin: 0.5,
            maximum_closure_error: 0.01,
            maximum_contraction_ratio: 0.01,
        };
        let error = discover_capabilities(&fingerprint, &oversized, &policy).unwrap_err();
        assert!(error
            .to_string()
            .contains("capability_discovery_work_limit"));
    }
}
