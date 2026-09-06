//! Autonomous, evidence-reducing knowledge engine for one governed capability.
//!
//! This module is the sole executable, revisioned epistemic authority for a
//! governed capability. No caller supplied score,
//! prose status, or generic file hash can promote a claim.  Only a sealed
//! cognitive invocation, an authenticated action receipt, and this module's
//! deterministic reducer can change authoritative knowledge state.

use crate::authority::{
    read_untrusted_private_file_bounded, replace_private_file_atomic, with_private_authority_lock,
    write_or_verify_immutable, PrivateFileReference,
};
use crate::capability_bundle::{authenticate_capability_bundle, CapabilityBundle};
use crate::digest::{CapabilityBundleDigest, Sha256Digest};
use crate::error::{BrainError, BrainResult};
use crate::identity::InquiryId;
use crate::security::verify_private_root;
use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

const STATE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:KNOWLEDGE-STATE:v1\0";
const POLICY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:KNOWLEDGE-POLICY:v1\0";
const ACTION_ID_DOMAIN: &[u8] = b"CEREBRO:TIDEX:KNOWLEDGE-ACTION-ID:v1\0";
const INVOCATION_DOMAIN: &[u8] = b"CEREBRO:TIDEX:COGNITIVE-INVOCATION:v1\0";
const OBSERVATION_DOMAIN: &[u8] = b"CEREBRO:TIDEX:OBSERVATION-SET:v1\0";
const EXACT_BYTES_DOMAIN: &[u8] = b"CEREBRO:TIDEX:OBSERVED-EXACT-BYTES:v1\0";
const EVIDENCE_ID_DOMAIN: &[u8] = b"CEREBRO:TIDEX:KNOWLEDGE-EVIDENCE-ID:v1\0";
const EVIDENCE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:KNOWLEDGE-EVIDENCE:v1\0";
const RECEIPT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:KNOWLEDGE-ACTION-RECEIPT:v1\0";
const TRANSITION_DOMAIN: &[u8] = b"CEREBRO:TIDEX:KNOWLEDGE-TRANSITION:v1\0";
const ROOT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:AUTHORITY-ROOT:v1\0";
const CANONICAL_HEAD_DOMAIN: &[u8] = b"CEREBRO:TIDEX:CANONICAL-INQUIRY-HEAD:v1\0";
const CANONICAL_HEAD_POINTER_MAX_BYTES: u64 = 1 << 20;
const DERIVED_CLAIM_ID_DOMAIN: &[u8] = b"CEREBRO:TIDEX:DERIVED-CLAIM-ID:v1\0";
const DERIVED_OBLIGATION_ID_DOMAIN: &[u8] = b"CEREBRO:TIDEX:DERIVED-OBLIGATION-ID:v1\0";

const HARD_MAX_CLAIMS: usize = 4_096;
const HARD_MAX_OBLIGATIONS: usize = 8_192;
const HARD_MAX_HYPOTHESES: usize = 512;
const HARD_MAX_ACTIONS: u64 = 8_192;
const HARD_MAX_TOTAL_COST: u64 = 1_000_000;
const HARD_MAX_REVISION: u64 = 8_192;
const HARD_MAX_DEPENDENCY_NODES: usize = 4_096;
const HARD_MAX_DEPENDENCY_EDGES: usize = 16_384;
const HARD_MAX_DEPENDENCY_EDGES_PER_ACTION: usize = 256;
const HARD_MAX_DEPENDENCY_DEPTH: u16 = 64;
const HARD_MAX_EVIDENCE_RECORDS: usize = 16_384;
const HARD_MAX_OBSERVATIONS_PER_ACTION: usize = 1_024;
const HARD_MAX_SCENARIO_INPUTS: usize = 256;
const HARD_MAX_TRACE_EVENTS: u32 = 1_000_000;
const HARD_MAX_COUNTEREXAMPLE_CASES: u32 = 1_000_000;
const HARD_MAX_DISCOVERIES_PER_ACTION: usize = 256;
const MAX_STATE_BYTES: u64 = 64 << 20;
const MAX_ACTION_RECEIPT_BYTES: u64 = 64 << 20;
const MAX_EVIDENCE_RECORD_BYTES: u64 = 1 << 20;
const MAX_EVIDENCE_ARTIFACT_BYTES: u64 = 64 << 20;
const MAX_EXECUTOR_IMPLEMENTATION_BYTES: u64 = 64 << 20;
const MAX_IDENTIFIER_BYTES: usize = 160;
const MAX_SYMBOL_BYTES: usize = 256;

fn invalid(code: &str) -> BrainError {
    BrainError::Invalid(code.into())
}

fn integrity(code: &str) -> BrainError {
    BrainError::Integrity(code.into())
}

fn validate_identifier(value: &str, code: &str) -> BrainResult<()> {
    let is_raw_sha256 = value.len() == Sha256Digest::HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if value.is_empty()
        || value.len() > MAX_IDENTIFIER_BYTES
        || is_raw_sha256
        || value.starts_with('.')
        || value.ends_with('.')
        || value.split('.').any(str::is_empty)
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(invalid(code));
    }
    Ok(())
}

macro_rules! local_id {
    ($(#[$metadata:meta])* $name:ident, $code:literal) => {
        $(#[$metadata])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn parse(value: impl AsRef<str>) -> BrainResult<Self> {
                let value = value.as_ref();
                validate_identifier(value, $code)?;
                Ok(Self(value.to_owned()))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(self.as_str())
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::parse(value).map_err(D::Error::custom)
            }
        }
    };
}

local_id!(
    /// Stable identity of one concrete, independently reducible knowledge duty.
    KnowledgeObligationId,
    "knowledge_obligation_id_invalid"
);
local_id!(
    /// Stable identity of one claim. It cannot be interchanged with an obligation.
    KnowledgeClaimId,
    "knowledge_claim_id_invalid"
);
local_id!(KnowledgeHypothesisId, "knowledge_hypothesis_id_invalid");
local_id!(HypothesisFamilyId, "hypothesis_family_id_invalid");
local_id!(ObservationKey, "observation_key_invalid");
local_id!(ScenarioId, "scenario_id_invalid");
local_id!(ScenarioInputId, "scenario_input_id_invalid");
local_id!(CounterexampleCaseId, "counterexample_case_id_invalid");
local_id!(InterventionVariableId, "intervention_variable_id_invalid");
local_id!(DependencyNodeId, "dependency_node_id_invalid");
local_id!(ExecutorId, "executor_id_invalid");
local_id!(DiscoveredConcernId, "discovered_concern_id_invalid");
local_id!(
    ProfiledKnowledgeDomainId,
    "profiled_knowledge_domain_id_invalid"
);
local_id!(
    /// Stable, externally provisioned identity of one authority deployment.
    /// It is deliberately supplied by bootstrap infrastructure rather than
    /// derived from a machine-local directory.
    AuthorityInstanceId,
    "authority_instance_id_invalid"
);

impl KnowledgeClaimId {
    fn derived_discovery(payload: &[u8]) -> Self {
        let digest = Sha256Digest::digest_domain(DERIVED_CLAIM_ID_DOMAIN, payload);
        Self(format!("discovered-claim-{}", digest.as_str()))
    }
}

impl KnowledgeObligationId {
    fn derived_discovery(payload: &[u8]) -> Self {
        let digest = Sha256Digest::digest_domain(DERIVED_OBLIGATION_ID_DOMAIN, payload);
        Self(format!("discovered-obligation-{}", digest.as_str()))
    }
}

macro_rules! semantic_digest {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Sha256Digest);

        impl $name {
            fn computed(domain: &[u8], canonical_payload: &[u8]) -> Self {
                Self(Sha256Digest::digest_domain(domain, canonical_payload))
            }

            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(self.as_str())
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                self.0.serialize(serializer)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                Ok(Self(Sha256Digest::deserialize(deserializer)?))
            }
        }
    };
}

semantic_digest!(KnowledgeStateDigest);
semantic_digest!(KnowledgePolicyDigest);
semantic_digest!(AuthorityRootDigest);
semantic_digest!(KnowledgeActionId);
semantic_digest!(CognitiveInvocationDigest);
semantic_digest!(ObservationSetDigest);
semantic_digest!(KnowledgeEvidenceId);
semantic_digest!(KnowledgeEvidenceDigest);
semantic_digest!(ActionReceiptDigest);
semantic_digest!(TransitionReceiptDigest);
semantic_digest!(ExecutorVersionDigest);
semantic_digest!(CanonicalInquiryHeadDigest);

/// Domain-bound commitment to exact observed bytes, never a generic artifact identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExactBytesDigest(Sha256Digest);

impl ExactBytesDigest {
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(Sha256Digest::digest_domain(EXACT_BYTES_DOMAIN, bytes))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl Serialize for ExactBytesDigest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ExactBytesDigest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self(Sha256Digest::deserialize(deserializer)?))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KnowledgeSymbol(String);

impl KnowledgeSymbol {
    pub fn parse(value: impl AsRef<str>) -> BrainResult<Self> {
        let value = value.as_ref();
        if value.is_empty()
            || value.len() > MAX_SYMBOL_BYTES
            || value.starts_with(|character: char| !character.is_ascii_alphanumeric())
            || value.ends_with(|character: char| !character.is_ascii_alphanumeric())
            || !value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')
            })
        {
            return Err(invalid("knowledge_symbol_invalid"));
        }
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Serialize for KnowledgeSymbol {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for KnowledgeSymbol {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(value).map_err(D::Error::custom)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case", tag = "type", content = "value")]
pub enum ObservedValue {
    Bool(bool),
    I64(i64),
    U64(u64),
    Symbol(KnowledgeSymbol),
    ExactBytesDigest(ExactBytesDigest),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "operator")]
pub enum KnowledgePredicate {
    BoolEquals {
        key: ObservationKey,
        expected: bool,
    },
    I64Equals {
        key: ObservationKey,
        expected: i64,
    },
    U64Equals {
        key: ObservationKey,
        expected: u64,
    },
    U64AtLeast {
        key: ObservationKey,
        minimum: u64,
    },
    U64AtMost {
        key: ObservationKey,
        maximum: u64,
    },
    SymbolEquals {
        key: ObservationKey,
        expected: KnowledgeSymbol,
    },
    ExactBytesDigestEquals {
        key: ObservationKey,
        expected: ExactBytesDigest,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PredicateEvaluation {
    Matches,
    Contradicts,
    Missing,
    TypeMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObservationValueType {
    Bool,
    I64,
    U64,
    Symbol,
    ExactBytesDigest,
}

impl KnowledgePredicate {
    pub fn observation_key(&self) -> &ObservationKey {
        match self {
            Self::BoolEquals { key, .. }
            | Self::I64Equals { key, .. }
            | Self::U64Equals { key, .. }
            | Self::U64AtLeast { key, .. }
            | Self::U64AtMost { key, .. }
            | Self::SymbolEquals { key, .. }
            | Self::ExactBytesDigestEquals { key, .. } => key,
        }
    }

    fn evaluate(
        &self,
        observations: &BTreeMap<ObservationKey, ObservedValue>,
    ) -> PredicateEvaluation {
        let Some(actual) = observations.get(self.observation_key()) else {
            return PredicateEvaluation::Missing;
        };
        if self.value_type() != actual.value_type() {
            return PredicateEvaluation::TypeMismatch;
        }
        let matches = match (self, actual) {
            (Self::BoolEquals { expected, .. }, ObservedValue::Bool(actual)) => actual == expected,
            (Self::I64Equals { expected, .. }, ObservedValue::I64(actual)) => actual == expected,
            (Self::U64Equals { expected, .. }, ObservedValue::U64(actual)) => actual == expected,
            (Self::U64AtLeast { minimum, .. }, ObservedValue::U64(actual)) => actual >= minimum,
            (Self::U64AtMost { maximum, .. }, ObservedValue::U64(actual)) => actual <= maximum,
            (Self::SymbolEquals { expected, .. }, ObservedValue::Symbol(actual)) => {
                actual == expected
            }
            (
                Self::ExactBytesDigestEquals { expected, .. },
                ObservedValue::ExactBytesDigest(actual),
            ) => actual == expected,
            _ => return PredicateEvaluation::TypeMismatch,
        };
        if matches {
            PredicateEvaluation::Matches
        } else {
            PredicateEvaluation::Contradicts
        }
    }

    fn value_type(&self) -> ObservationValueType {
        match self {
            Self::BoolEquals { .. } => ObservationValueType::Bool,
            Self::I64Equals { .. } => ObservationValueType::I64,
            Self::U64Equals { .. } | Self::U64AtLeast { .. } | Self::U64AtMost { .. } => {
                ObservationValueType::U64
            }
            Self::SymbolEquals { .. } => ObservationValueType::Symbol,
            Self::ExactBytesDigestEquals { .. } => ObservationValueType::ExactBytesDigest,
        }
    }
}

impl ObservedValue {
    fn value_type(&self) -> ObservationValueType {
        match self {
            Self::Bool(_) => ObservationValueType::Bool,
            Self::I64(_) => ObservationValueType::I64,
            Self::U64(_) => ObservationValueType::U64,
            Self::Symbol(_) => ObservationValueType::Symbol,
            Self::ExactBytesDigest(_) => ObservationValueType::ExactBytesDigest,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvaluationCriterion {
    predicate: KnowledgePredicate,
}

impl EvaluationCriterion {
    pub fn exact(predicate: KnowledgePredicate) -> Self {
        Self { predicate }
    }

    pub fn predicate(&self) -> &KnowledgePredicate {
        &self.predicate
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case", tag = "domain")]
pub enum KnowledgeDomain {
    TraceBehavior,
    ScenarioBehavior,
    CausalMechanism,
    AdversarialBoundary,
    FormalProperty,
    DependencyClosure,
    /// Authority-profiled domain for a capability-specific concern not in the core taxonomy.
    Profiled {
        profile_id: ProfiledKnowledgeDomainId,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProofSystem {
    Smt,
    ModelChecking,
    TheoremKernel,
    AbstractInterpretation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "method")]
pub enum CognitivePlan {
    ObserveTrace {
        observation_keys: BTreeSet<ObservationKey>,
        max_events: u32,
    },
    ExecuteScenario {
        scenario_id: ScenarioId,
        inputs: BTreeMap<ScenarioInputId, ObservedValue>,
        observation_keys: BTreeSet<ObservationKey>,
    },
    CausalIntervention {
        variable: InterventionVariableId,
        baseline: ObservedValue,
        treatment: ObservedValue,
        effect: KnowledgePredicate,
        predictions: BTreeMap<KnowledgeHypothesisId, KnowledgePredicate>,
    },
    SearchCounterexample {
        target: KnowledgePredicate,
        max_cases: u32,
        predictions: BTreeMap<KnowledgeHypothesisId, KnowledgePredicate>,
    },
    VerifyFormalProperty {
        proof_system: ProofSystem,
        property: KnowledgePredicate,
    },
    ExpandDependencyClosure {
        root: DependencyNodeId,
        depth: u16,
        max_depth: u16,
    },
}

impl CognitivePlan {
    fn domain(&self) -> KnowledgeDomain {
        match self {
            Self::ObserveTrace { .. } => KnowledgeDomain::TraceBehavior,
            Self::ExecuteScenario { .. } => KnowledgeDomain::ScenarioBehavior,
            Self::CausalIntervention { .. } => KnowledgeDomain::CausalMechanism,
            Self::SearchCounterexample { .. } => KnowledgeDomain::AdversarialBoundary,
            Self::VerifyFormalProperty { .. } => KnowledgeDomain::FormalProperty,
            Self::ExpandDependencyClosure { .. } => KnowledgeDomain::DependencyClosure,
        }
    }

    fn cost(&self) -> u64 {
        match self {
            Self::ObserveTrace { .. } => 2,
            Self::ExecuteScenario { .. } => 3,
            Self::CausalIntervention { .. } => 5,
            Self::SearchCounterexample { .. } => 4,
            Self::VerifyFormalProperty { .. } => 8,
            Self::ExpandDependencyClosure { .. } => 3,
        }
    }

    fn predictions(&self) -> Option<&BTreeMap<KnowledgeHypothesisId, KnowledgePredicate>> {
        match self {
            Self::CausalIntervention { predictions, .. }
            | Self::SearchCounterexample { predictions, .. } => Some(predictions),
            _ => None,
        }
    }

    fn validate_bounds(&self, budget: &KnowledgeBudget) -> BrainResult<()> {
        match self {
            Self::ObserveTrace {
                observation_keys,
                max_events,
            } => {
                if observation_keys.is_empty()
                    || observation_keys.len() > budget.max_observations_per_action
                    || *max_events == 0
                    || *max_events > HARD_MAX_TRACE_EVENTS
                {
                    return Err(invalid("trace_plan_bounds_invalid"));
                }
            }
            Self::ExecuteScenario {
                inputs,
                observation_keys,
                ..
            } => {
                if inputs.len() > HARD_MAX_SCENARIO_INPUTS
                    || observation_keys.is_empty()
                    || observation_keys.len() > budget.max_observations_per_action
                {
                    return Err(invalid("scenario_plan_bounds_invalid"));
                }
            }
            Self::CausalIntervention {
                baseline,
                treatment,
                predictions,
                ..
            } => {
                if baseline == treatment || predictions.len() < 2 {
                    return Err(invalid("causal_plan_not_discriminating"));
                }
            }
            Self::SearchCounterexample {
                max_cases,
                predictions,
                ..
            } => {
                if *max_cases == 0
                    || *max_cases > HARD_MAX_COUNTEREXAMPLE_CASES
                    || predictions.len() < 2
                {
                    return Err(invalid("counterexample_plan_bounds_invalid"));
                }
            }
            Self::VerifyFormalProperty { .. } => {}
            Self::ExpandDependencyClosure {
                depth, max_depth, ..
            } => {
                if *max_depth == 0 || *max_depth > budget.max_dependency_depth || depth > max_depth
                {
                    return Err(invalid("dependency_plan_bounds_invalid"));
                }
            }
        }
        Ok(())
    }
}

fn dependency_closure_predicate(root: &DependencyNodeId) -> BrainResult<KnowledgePredicate> {
    let key_digest = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:DEPENDENCY-OBSERVATION-KEY:v1\0",
        &serde_json::to_vec(root)?,
    );
    Ok(KnowledgePredicate::BoolEquals {
        key: ObservationKey::parse(format!("dependency.{}.closed", key_digest.as_str()))?,
        expected: true,
    })
}

fn validate_criterion_plan_binding(
    criterion: &EvaluationCriterion,
    plan: &CognitivePlan,
) -> BrainResult<()> {
    let predicate = criterion.predicate();
    let valid = match plan {
        CognitivePlan::ObserveTrace {
            observation_keys, ..
        }
        | CognitivePlan::ExecuteScenario {
            observation_keys, ..
        } => observation_keys.contains(predicate.observation_key()),
        CognitivePlan::CausalIntervention { effect, .. } => effect == predicate,
        CognitivePlan::SearchCounterexample { target, .. } => target == predicate,
        CognitivePlan::VerifyFormalProperty { property, .. } => property == predicate,
        CognitivePlan::ExpandDependencyClosure { root, .. } => {
            &dependency_closure_predicate(root)? == predicate
        }
    };
    if !valid {
        return Err(invalid("knowledge_criterion_not_bound_to_plan_object"));
    }
    Ok(())
}

fn plan_predicates(plan: &CognitivePlan) -> Vec<&KnowledgePredicate> {
    match plan {
        CognitivePlan::ObserveTrace { .. }
        | CognitivePlan::ExecuteScenario { .. }
        | CognitivePlan::ExpandDependencyClosure { .. } => Vec::new(),
        CognitivePlan::CausalIntervention {
            effect,
            predictions,
            ..
        } => std::iter::once(effect)
            .chain(predictions.values())
            .collect(),
        CognitivePlan::SearchCounterexample {
            target,
            predictions,
            ..
        } => std::iter::once(target)
            .chain(predictions.values())
            .collect(),
        CognitivePlan::VerifyFormalProperty { property, .. } => vec![property],
    }
}

fn record_predicate_type(
    schema: &mut BTreeMap<ObservationKey, ObservationValueType>,
    predicate: &KnowledgePredicate,
) -> BrainResult<()> {
    match schema.insert(predicate.observation_key().clone(), predicate.value_type()) {
        Some(previous) if previous != predicate.value_type() => {
            Err(invalid("observation_key_value_type_conflict"))
        }
        _ => Ok(()),
    }
}

fn observation_schema_from_state(
    state: &KnowledgeState,
) -> BrainResult<BTreeMap<ObservationKey, ObservationValueType>> {
    let mut schema = BTreeMap::new();
    for claim in state.claims.values() {
        record_predicate_type(&mut schema, &claim.predicate)?;
    }
    for obligation in state.obligations.values() {
        record_predicate_type(&mut schema, obligation.criterion.predicate())?;
        for predicate in plan_predicates(&obligation.plan) {
            record_predicate_type(&mut schema, predicate)?;
        }
    }
    for hypothesis in state.hypotheses.values() {
        record_predicate_type(&mut schema, &hypothesis.definition)?;
    }
    Ok(schema)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeBudget {
    max_actions: u64,
    max_total_cost: u64,
    max_revision: u64,
    max_dependency_nodes: usize,
    max_dependency_edges: usize,
    max_dependency_depth: u16,
    max_evidence_records: usize,
    max_observations_per_action: usize,
}

impl KnowledgeBudget {
    #[allow(clippy::too_many_arguments)]
    pub fn bounded(
        max_actions: u64,
        max_total_cost: u64,
        max_revision: u64,
        max_dependency_nodes: usize,
        max_dependency_edges: usize,
        max_dependency_depth: u16,
        max_evidence_records: usize,
        max_observations_per_action: usize,
    ) -> BrainResult<Self> {
        let budget = Self {
            max_actions,
            max_total_cost,
            max_revision,
            max_dependency_nodes,
            max_dependency_edges,
            max_dependency_depth,
            max_evidence_records,
            max_observations_per_action,
        };
        budget.validate()?;
        Ok(budget)
    }

    pub fn conservative_default() -> Self {
        Self {
            max_actions: 512,
            max_total_cost: 16_384,
            max_revision: 512,
            max_dependency_nodes: 512,
            max_dependency_edges: 2_048,
            max_dependency_depth: 16,
            max_evidence_records: 2_048,
            max_observations_per_action: 256,
        }
    }

    fn validate(&self) -> BrainResult<()> {
        if self.max_actions == 0
            || self.max_actions > HARD_MAX_ACTIONS
            || self.max_total_cost == 0
            || self.max_total_cost > HARD_MAX_TOTAL_COST
            || self.max_revision == 0
            || self.max_revision > HARD_MAX_REVISION
            || self.max_dependency_nodes == 0
            || self.max_dependency_nodes > HARD_MAX_DEPENDENCY_NODES
            || self.max_dependency_edges == 0
            || self.max_dependency_edges > HARD_MAX_DEPENDENCY_EDGES
            || self.max_dependency_depth == 0
            || self.max_dependency_depth > HARD_MAX_DEPENDENCY_DEPTH
            || self.max_evidence_records == 0
            || self.max_evidence_records > HARD_MAX_EVIDENCE_RECORDS
            || self.max_observations_per_action == 0
            || self.max_observations_per_action > HARD_MAX_OBSERVATIONS_PER_ACTION
        {
            return Err(invalid("knowledge_budget_invalid"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeClaimDraft {
    pub claim_id: KnowledgeClaimId,
    pub domain: KnowledgeDomain,
    pub predicate: KnowledgePredicate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeObligationDraft {
    pub obligation_id: KnowledgeObligationId,
    pub claim_id: KnowledgeClaimId,
    pub criterion: EvaluationCriterion,
    pub plan: CognitivePlan,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeHypothesisDraft {
    pub hypothesis_id: KnowledgeHypothesisId,
    pub family_id: HypothesisFamilyId,
    pub definition: KnowledgePredicate,
}

/// Untrusted caller proposal. `KnowledgeEngine::initialize` validates and seals it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeContractDraft {
    pub claims: Vec<KnowledgeClaimDraft>,
    pub obligations: Vec<KnowledgeObligationDraft>,
    pub hypotheses: Vec<KnowledgeHypothesisDraft>,
    pub budget: KnowledgeBudget,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum KnowledgeEngineSchema {
    V1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "assessment")]
pub enum ClaimAssessment {
    Unresolved,
    Established {
        witnesses: BTreeSet<ActionReceiptDigest>,
    },
    Contradicted {
        witnesses: BTreeSet<ActionReceiptDigest>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionBlockReason {
    AdapterUnavailable,
    ResourceDenied,
    UnsupportedEnvironment,
    ObservationUnavailable,
    SafetyPolicyDenied,
    ExecutorIntegrityFailure,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "block")]
pub enum KnowledgeBlockReason {
    ContradictoryObservation {
        receipt: ActionReceiptDigest,
    },
    ExecutorBlocked {
        receipt: ActionReceiptDigest,
        reason: ExecutionBlockReason,
    },
    DependencyCycle {
        receipt: ActionReceiptDigest,
        from: DependencyNodeId,
        to: DependencyNodeId,
    },
    HypothesisConflict {
        family_id: HypothesisFamilyId,
        hypothesis_id: KnowledgeHypothesisId,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum KnowledgeObligationState {
    Open {
        attempts: u16,
        witnesses: BTreeSet<ActionReceiptDigest>,
    },
    Satisfied {
        attempts: u16,
        witnesses: BTreeSet<ActionReceiptDigest>,
    },
    Blocked {
        attempts: u16,
        witnesses: BTreeSet<ActionReceiptDigest>,
        reason: KnowledgeBlockReason,
    },
}

impl KnowledgeObligationState {
    fn attempts(&self) -> u16 {
        match self {
            Self::Open { attempts, .. }
            | Self::Satisfied { attempts, .. }
            | Self::Blocked { attempts, .. } => *attempts,
        }
    }

    fn witnesses(&self) -> &BTreeSet<ActionReceiptDigest> {
        match self {
            Self::Open { witnesses, .. }
            | Self::Satisfied { witnesses, .. }
            | Self::Blocked { witnesses, .. } => witnesses,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeClaim {
    claim_id: KnowledgeClaimId,
    domain: KnowledgeDomain,
    predicate: KnowledgePredicate,
    assessment: ClaimAssessment,
}

impl KnowledgeClaim {
    pub fn id(&self) -> &KnowledgeClaimId {
        &self.claim_id
    }

    pub fn domain(&self) -> KnowledgeDomain {
        self.domain.clone()
    }

    pub fn predicate(&self) -> &KnowledgePredicate {
        &self.predicate
    }

    pub fn assessment(&self) -> &ClaimAssessment {
        &self.assessment
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeObligation {
    obligation_id: KnowledgeObligationId,
    claim_id: KnowledgeClaimId,
    domain: KnowledgeDomain,
    criterion: EvaluationCriterion,
    plan: CognitivePlan,
    state: KnowledgeObligationState,
}

impl KnowledgeObligation {
    pub fn id(&self) -> &KnowledgeObligationId {
        &self.obligation_id
    }

    pub fn claim_id(&self) -> &KnowledgeClaimId {
        &self.claim_id
    }

    pub fn domain(&self) -> KnowledgeDomain {
        self.domain.clone()
    }

    pub fn criterion(&self) -> &EvaluationCriterion {
        &self.criterion
    }

    pub fn plan(&self) -> &CognitivePlan {
        &self.plan
    }

    pub fn state(&self) -> &KnowledgeObligationState {
        &self.state
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "assessment")]
pub enum HypothesisAssessment {
    Viable,
    Supported {
        witnesses: BTreeSet<ActionReceiptDigest>,
    },
    Refuted {
        witnesses: BTreeSet<ActionReceiptDigest>,
    },
    Conflicted {
        supporting: BTreeSet<ActionReceiptDigest>,
        refuting: BTreeSet<ActionReceiptDigest>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeHypothesis {
    hypothesis_id: KnowledgeHypothesisId,
    family_id: HypothesisFamilyId,
    definition: KnowledgePredicate,
    assessment: HypothesisAssessment,
}

impl KnowledgeHypothesis {
    pub fn id(&self) -> &KnowledgeHypothesisId {
        &self.hypothesis_id
    }

    pub fn family_id(&self) -> &HypothesisFamilyId {
        &self.family_id
    }

    pub fn definition(&self) -> &KnowledgePredicate {
        &self.definition
    }

    pub fn assessment(&self) -> &HypothesisAssessment {
        &self.assessment
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct BudgetUsage {
    actions: u64,
    total_cost: u64,
    evidence_records: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct DependencyEdge {
    from: DependencyNodeId,
    to: DependencyNodeId,
}

impl DependencyEdge {
    pub fn new(from: DependencyNodeId, to: DependencyNodeId) -> Self {
        Self { from, to }
    }

    pub fn from(&self) -> &DependencyNodeId {
        &self.from
    }

    pub fn to(&self) -> &DependencyNodeId {
        &self.to
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct InvocationBinding {
    authority_instance: AuthorityInstanceId,
    authority_root: AuthorityRootDigest,
    inquiry_id: InquiryId,
    capability_bundle: CapabilityBundleDigest,
    parent_state: KnowledgeStateDigest,
    action_id: KnowledgeActionId,
    obligation_id: KnowledgeObligationId,
    claim_id: KnowledgeClaimId,
    attempt: u16,
    criterion: EvaluationCriterion,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TraceInvocation {
    binding: InvocationBinding,
    observation_keys: BTreeSet<ObservationKey>,
    max_events: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioInvocation {
    binding: InvocationBinding,
    scenario_id: ScenarioId,
    inputs: BTreeMap<ScenarioInputId, ObservedValue>,
    observation_keys: BTreeSet<ObservationKey>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CausalInvocation {
    binding: InvocationBinding,
    variable: InterventionVariableId,
    baseline: ObservedValue,
    treatment: ObservedValue,
    effect: KnowledgePredicate,
    predictions: BTreeMap<KnowledgeHypothesisId, KnowledgePredicate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CounterexampleInvocation {
    binding: InvocationBinding,
    target: KnowledgePredicate,
    max_cases: u32,
    predictions: BTreeMap<KnowledgeHypothesisId, KnowledgePredicate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FormalVerificationInvocation {
    binding: InvocationBinding,
    proof_system: ProofSystem,
    property: KnowledgePredicate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DependencyExpansionInvocation {
    binding: InvocationBinding,
    root: DependencyNodeId,
    depth: u16,
    max_depth: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "invocation")]
pub enum CognitiveInvocation {
    ObserveTrace(TraceInvocation),
    ExecuteScenario(ScenarioInvocation),
    CausalIntervention(CausalInvocation),
    SearchCounterexample(CounterexampleInvocation),
    VerifyFormalProperty(FormalVerificationInvocation),
    ExpandDependencyClosure(DependencyExpansionInvocation),
}

impl CognitiveInvocation {
    fn binding(&self) -> &InvocationBinding {
        match self {
            Self::ObserveTrace(value) => &value.binding,
            Self::ExecuteScenario(value) => &value.binding,
            Self::CausalIntervention(value) => &value.binding,
            Self::SearchCounterexample(value) => &value.binding,
            Self::VerifyFormalProperty(value) => &value.binding,
            Self::ExpandDependencyClosure(value) => &value.binding,
        }
    }

    pub fn action_id(&self) -> &KnowledgeActionId {
        &self.binding().action_id
    }

    pub fn obligation_id(&self) -> &KnowledgeObligationId {
        &self.binding().obligation_id
    }

    pub fn claim_id(&self) -> &KnowledgeClaimId {
        &self.binding().claim_id
    }

    pub fn parent_state(&self) -> &KnowledgeStateDigest {
        &self.binding().parent_state
    }

    pub fn attempt(&self) -> u16 {
        self.binding().attempt
    }

    pub fn digest(&self) -> BrainResult<CognitiveInvocationDigest> {
        Ok(CognitiveInvocationDigest::computed(
            INVOCATION_DOMAIN,
            &serde_json::to_vec(self)?,
        ))
    }

    fn cost(&self) -> u64 {
        match self {
            Self::ObserveTrace(_) => 2,
            Self::ExecuteScenario(_) => 3,
            Self::CausalIntervention(_) => 5,
            Self::SearchCounterexample(_) => 4,
            Self::VerifyFormalProperty(_) => 8,
            Self::ExpandDependencyClosure(_) => 3,
        }
    }

    fn predictions(&self) -> Option<&BTreeMap<KnowledgeHypothesisId, KnowledgePredicate>> {
        match self {
            Self::CausalIntervention(value) => Some(&value.predictions),
            Self::SearchCounterexample(value) => Some(&value.predictions),
            _ => None,
        }
    }

    fn required_observations(&self) -> BTreeSet<ObservationKey> {
        let mut keys = match self {
            Self::ObserveTrace(value) => value.observation_keys.clone(),
            Self::ExecuteScenario(value) => value.observation_keys.clone(),
            Self::CausalIntervention(value) => value
                .predictions
                .values()
                .map(|predicate| predicate.observation_key().clone())
                .collect(),
            Self::SearchCounterexample(value) => value
                .predictions
                .values()
                .map(|predicate| predicate.observation_key().clone())
                .collect(),
            Self::VerifyFormalProperty(value) => {
                BTreeSet::from([value.property.observation_key().clone()])
            }
            Self::ExpandDependencyClosure(_) => BTreeSet::new(),
        };
        if let Self::SearchCounterexample(value) = self {
            keys.insert(value.target.observation_key().clone());
        }
        keys.insert(
            self.binding()
                .criterion
                .predicate()
                .observation_key()
                .clone(),
        );
        keys
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    RuntimeTrace,
    ScenarioResult,
    CausalComparison,
    CounterexampleSearch,
    FormalCertificate,
    FormalSearchLog,
    DependencyGraph,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceArtifactInput {
    pub kind: EvidenceKind,
    pub artifact: PrivateFileReference,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TraceEvidenceProtocol {
    CanonicalEventStreamV1,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioEvidenceProtocol {
    ExactScenarioObservationV1,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CausalEvidenceProtocol {
    PairedInterventionV1,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CounterexampleEvidenceProtocol {
    DeterministicCaseSearchV1,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FormalEvidenceProtocol {
    BackendUnavailableV1,
    ExternalCertificateV1,
}

struct EvidenceVerificationContext<'a> {
    private_root: &'a Path,
    authority_instance: &'a AuthorityInstanceId,
    authority_root: &'a AuthorityRootDigest,
    invocation: &'a CognitiveInvocation,
    executor: &'a ExecutorAttestation,
}

/// Every semantic artifact is bound to both the durable authority-instance
/// identity and its domain-separated root.  The explicit fields are
/// intentionally redundant with the invocation digest: redundancy lets the
/// verifier reject relabelled/replayed envelopes before considering payloads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct EvidenceAuthorityBinding {
    schema: KnowledgeEngineSchema,
    authority_instance: AuthorityInstanceId,
    authority_root: AuthorityRootDigest,
    inquiry_id: InquiryId,
    capability_bundle: CapabilityBundleDigest,
    parent_state: KnowledgeStateDigest,
    action_id: KnowledgeActionId,
    obligation_id: KnowledgeObligationId,
    claim_id: KnowledgeClaimId,
    attempt: u16,
    invocation_digest: CognitiveInvocationDigest,
    executor_id: ExecutorId,
    executor_interface_version: ExecutorInterfaceVersion,
    executor_implementation_digest: Sha256Digest,
    executor_version_digest: ExecutorVersionDigest,
}

impl EvidenceAuthorityBinding {
    fn from_invocation(
        invocation: &CognitiveInvocation,
        executor: &ExecutorIdentityDraft,
    ) -> BrainResult<Self> {
        let invocation_binding = invocation.binding();
        let expected_root = AuthorityRootDigest::computed(
            ROOT_DOMAIN,
            &serde_json::to_vec(&invocation_binding.authority_instance)?,
        );
        if invocation_binding.authority_root != expected_root {
            return Err(integrity("evidence_invocation_authority_instance_mismatch"));
        }
        Ok(Self {
            schema: KnowledgeEngineSchema::V1,
            authority_instance: invocation_binding.authority_instance.clone(),
            authority_root: invocation_binding.authority_root.clone(),
            inquiry_id: invocation_binding.inquiry_id.clone(),
            capability_bundle: invocation_binding.capability_bundle.clone(),
            parent_state: invocation_binding.parent_state.clone(),
            action_id: invocation_binding.action_id.clone(),
            obligation_id: invocation_binding.obligation_id.clone(),
            claim_id: invocation_binding.claim_id.clone(),
            attempt: invocation_binding.attempt,
            invocation_digest: invocation.digest()?,
            executor_id: executor.executor_id.clone(),
            executor_interface_version: executor.interface_version,
            executor_implementation_digest: executor.implementation.sha256.clone(),
            executor_version_digest: calculate_executor_version_digest(
                &executor.executor_id,
                executor.interface_version,
                &executor.implementation.sha256,
                &executor.implementation.sha256,
            )?,
        })
    }

    fn validate(&self, context: &EvidenceVerificationContext<'_>) -> BrainResult<()> {
        let invocation_binding = context.invocation.binding();
        let derived_root = AuthorityRootDigest::computed(
            ROOT_DOMAIN,
            &serde_json::to_vec(context.authority_instance)?,
        );
        if self.schema != KnowledgeEngineSchema::V1
            || derived_root != *context.authority_root
            || self.authority_instance != *context.authority_instance
            || self.authority_instance != invocation_binding.authority_instance
            || self.authority_root != *context.authority_root
            || self.authority_root != invocation_binding.authority_root
            || self.inquiry_id != invocation_binding.inquiry_id
            || self.capability_bundle != invocation_binding.capability_bundle
            || self.parent_state != invocation_binding.parent_state
            || self.action_id != invocation_binding.action_id
            || self.obligation_id != invocation_binding.obligation_id
            || self.claim_id != invocation_binding.claim_id
            || self.attempt != invocation_binding.attempt
            || self.invocation_digest != context.invocation.digest()?
            || self.executor_id != context.executor.executor_id
            || self.executor_interface_version != context.executor.interface_version
            || self.executor_implementation_digest != context.executor.implementation.sha256
            || self.executor_version_digest != context.executor.version_digest
        {
            return Err(integrity("evidence_authority_binding_mismatch"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TraceEvidenceEvent {
    ordinal: u32,
    observations: BTreeMap<ObservationKey, ObservedValue>,
}

impl TraceEvidenceEvent {
    pub fn new(
        ordinal: u32,
        observations: BTreeMap<ObservationKey, ObservedValue>,
    ) -> BrainResult<Self> {
        if observations.is_empty() {
            return Err(invalid("trace_evidence_event_empty"));
        }
        Ok(Self {
            ordinal,
            observations,
        })
    }
}

/// A trace outcome is derived from an ordered stream of raw event fields; it
/// is never copied from an executor-supplied `ActionOutcome`.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TraceEvidenceArtifact {
    binding: EvidenceAuthorityBinding,
    protocol: TraceEvidenceProtocol,
    observation_keys: BTreeSet<ObservationKey>,
    max_events: u32,
    events: Vec<TraceEvidenceEvent>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedTraceEvidenceArtifactDto {
    binding: EvidenceAuthorityBinding,
    protocol: TraceEvidenceProtocol,
    observation_keys: BTreeSet<ObservationKey>,
    max_events: u32,
    events: Vec<TraceEvidenceEvent>,
}

impl TraceEvidenceArtifact {
    pub fn from_invocation(
        invocation: &CognitiveInvocation,
        executor: &ExecutorIdentityDraft,
        events: Vec<TraceEvidenceEvent>,
    ) -> BrainResult<Self> {
        let CognitiveInvocation::ObserveTrace(trace) = invocation else {
            return Err(invalid("trace_evidence_requires_trace_invocation"));
        };
        Ok(Self {
            binding: EvidenceAuthorityBinding::from_invocation(invocation, executor)?,
            protocol: TraceEvidenceProtocol::CanonicalEventStreamV1,
            observation_keys: trace.observation_keys.clone(),
            max_events: trace.max_events,
            events,
        })
    }

    pub fn canonical_bytes(&self) -> BrainResult<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    fn decode_canonical(bytes: &[u8]) -> BrainResult<Self> {
        let dto: UntrustedTraceEvidenceArtifactDto = serde_json::from_slice(bytes)
            .map_err(|_| integrity("trace_evidence_encoding_invalid"))?;
        let artifact = Self {
            binding: dto.binding,
            protocol: dto.protocol,
            observation_keys: dto.observation_keys,
            max_events: dto.max_events,
            events: dto.events,
        };
        if artifact.canonical_bytes()? != bytes {
            return Err(integrity("trace_evidence_not_canonical"));
        }
        Ok(artifact)
    }
}

/// Canonical, typed witness emitted by a scenario runner.  Bytes which merely
/// deserialize as this type have no authority until the registry accepts them.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioEvidenceArtifact {
    binding: EvidenceAuthorityBinding,
    protocol: ScenarioEvidenceProtocol,
    scenario_id: ScenarioId,
    inputs: BTreeMap<ScenarioInputId, ObservedValue>,
    observation_keys: BTreeSet<ObservationKey>,
    observations: BTreeMap<ObservationKey, ObservedValue>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedScenarioEvidenceArtifactDto {
    binding: EvidenceAuthorityBinding,
    protocol: ScenarioEvidenceProtocol,
    scenario_id: ScenarioId,
    inputs: BTreeMap<ScenarioInputId, ObservedValue>,
    observation_keys: BTreeSet<ObservationKey>,
    observations: BTreeMap<ObservationKey, ObservedValue>,
}

impl ScenarioEvidenceArtifact {
    pub fn from_invocation(
        invocation: &CognitiveInvocation,
        executor: &ExecutorIdentityDraft,
        observations: BTreeMap<ObservationKey, ObservedValue>,
    ) -> BrainResult<Self> {
        let CognitiveInvocation::ExecuteScenario(scenario) = invocation else {
            return Err(invalid("scenario_evidence_requires_scenario_invocation"));
        };
        Ok(Self {
            binding: EvidenceAuthorityBinding::from_invocation(invocation, executor)?,
            protocol: ScenarioEvidenceProtocol::ExactScenarioObservationV1,
            scenario_id: scenario.scenario_id.clone(),
            inputs: scenario.inputs.clone(),
            observation_keys: scenario.observation_keys.clone(),
            observations,
        })
    }

    pub fn canonical_bytes(&self) -> BrainResult<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    fn decode_canonical(bytes: &[u8]) -> BrainResult<Self> {
        let dto: UntrustedScenarioEvidenceArtifactDto = serde_json::from_slice(bytes)
            .map_err(|_| integrity("scenario_evidence_encoding_invalid"))?;
        let artifact = Self {
            binding: dto.binding,
            protocol: dto.protocol,
            scenario_id: dto.scenario_id,
            inputs: dto.inputs,
            observation_keys: dto.observation_keys,
            observations: dto.observations,
        };
        if artifact.canonical_bytes()? != bytes {
            return Err(integrity("scenario_evidence_not_canonical"));
        }
        Ok(artifact)
    }
}

/// Paired raw arms for the exact intervention precommitted by the invocation.
/// The canonical outcome is the treatment arm; the proposed outcome is not an
/// input to that derivation.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CausalEvidenceArtifact {
    binding: EvidenceAuthorityBinding,
    protocol: CausalEvidenceProtocol,
    variable: InterventionVariableId,
    baseline: ObservedValue,
    treatment: ObservedValue,
    effect: KnowledgePredicate,
    predictions: BTreeMap<KnowledgeHypothesisId, KnowledgePredicate>,
    baseline_observations: BTreeMap<ObservationKey, ObservedValue>,
    treatment_observations: BTreeMap<ObservationKey, ObservedValue>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedCausalEvidenceArtifactDto {
    binding: EvidenceAuthorityBinding,
    protocol: CausalEvidenceProtocol,
    variable: InterventionVariableId,
    baseline: ObservedValue,
    treatment: ObservedValue,
    effect: KnowledgePredicate,
    predictions: BTreeMap<KnowledgeHypothesisId, KnowledgePredicate>,
    baseline_observations: BTreeMap<ObservationKey, ObservedValue>,
    treatment_observations: BTreeMap<ObservationKey, ObservedValue>,
}

impl CausalEvidenceArtifact {
    pub fn from_invocation(
        invocation: &CognitiveInvocation,
        executor: &ExecutorIdentityDraft,
        baseline_observations: BTreeMap<ObservationKey, ObservedValue>,
        treatment_observations: BTreeMap<ObservationKey, ObservedValue>,
    ) -> BrainResult<Self> {
        let CognitiveInvocation::CausalIntervention(causal) = invocation else {
            return Err(invalid("causal_evidence_requires_causal_invocation"));
        };
        Ok(Self {
            binding: EvidenceAuthorityBinding::from_invocation(invocation, executor)?,
            protocol: CausalEvidenceProtocol::PairedInterventionV1,
            variable: causal.variable.clone(),
            baseline: causal.baseline.clone(),
            treatment: causal.treatment.clone(),
            effect: causal.effect.clone(),
            predictions: causal.predictions.clone(),
            baseline_observations,
            treatment_observations,
        })
    }

    pub fn canonical_bytes(&self) -> BrainResult<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    fn decode_canonical(bytes: &[u8]) -> BrainResult<Self> {
        let dto: UntrustedCausalEvidenceArtifactDto = serde_json::from_slice(bytes)
            .map_err(|_| integrity("causal_evidence_encoding_invalid"))?;
        let artifact = Self {
            binding: dto.binding,
            protocol: dto.protocol,
            variable: dto.variable,
            baseline: dto.baseline,
            treatment: dto.treatment,
            effect: dto.effect,
            predictions: dto.predictions,
            baseline_observations: dto.baseline_observations,
            treatment_observations: dto.treatment_observations,
        };
        if artifact.canonical_bytes()? != bytes {
            return Err(integrity("causal_evidence_not_canonical"));
        }
        Ok(artifact)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CounterexampleCaseEvidence {
    inputs: BTreeMap<ScenarioInputId, ObservedValue>,
    observations: BTreeMap<ObservationKey, ObservedValue>,
}

impl CounterexampleCaseEvidence {
    pub fn new(
        inputs: BTreeMap<ScenarioInputId, ObservedValue>,
        observations: BTreeMap<ObservationKey, ObservedValue>,
    ) -> BrainResult<Self> {
        if inputs.len() > HARD_MAX_SCENARIO_INPUTS || observations.is_empty() {
            return Err(invalid("counterexample_case_bounds_invalid"));
        }
        Ok(Self {
            inputs,
            observations,
        })
    }
}

/// Raw, deterministically ordered search cases.  A completed outcome exists
/// only when the verifier itself locates the first case contradicting `target`.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CounterexampleEvidenceArtifact {
    binding: EvidenceAuthorityBinding,
    protocol: CounterexampleEvidenceProtocol,
    target: KnowledgePredicate,
    max_cases: u32,
    predictions: BTreeMap<KnowledgeHypothesisId, KnowledgePredicate>,
    cases: BTreeMap<CounterexampleCaseId, CounterexampleCaseEvidence>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedCounterexampleEvidenceArtifactDto {
    binding: EvidenceAuthorityBinding,
    protocol: CounterexampleEvidenceProtocol,
    target: KnowledgePredicate,
    max_cases: u32,
    predictions: BTreeMap<KnowledgeHypothesisId, KnowledgePredicate>,
    cases: BTreeMap<CounterexampleCaseId, CounterexampleCaseEvidence>,
}

impl CounterexampleEvidenceArtifact {
    pub fn from_invocation(
        invocation: &CognitiveInvocation,
        executor: &ExecutorIdentityDraft,
        cases: BTreeMap<CounterexampleCaseId, CounterexampleCaseEvidence>,
    ) -> BrainResult<Self> {
        let CognitiveInvocation::SearchCounterexample(search) = invocation else {
            return Err(invalid(
                "counterexample_evidence_requires_search_invocation",
            ));
        };
        Ok(Self {
            binding: EvidenceAuthorityBinding::from_invocation(invocation, executor)?,
            protocol: CounterexampleEvidenceProtocol::DeterministicCaseSearchV1,
            target: search.target.clone(),
            max_cases: search.max_cases,
            predictions: search.predictions.clone(),
            cases,
        })
    }

    pub fn canonical_bytes(&self) -> BrainResult<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    fn decode_canonical(bytes: &[u8]) -> BrainResult<Self> {
        let dto: UntrustedCounterexampleEvidenceArtifactDto = serde_json::from_slice(bytes)
            .map_err(|_| integrity("counterexample_evidence_encoding_invalid"))?;
        let artifact = Self {
            binding: dto.binding,
            protocol: dto.protocol,
            target: dto.target,
            max_cases: dto.max_cases,
            predictions: dto.predictions,
            cases: dto.cases,
        };
        if artifact.canonical_bytes()? != bytes {
            return Err(integrity("counterexample_evidence_not_canonical"));
        }
        Ok(artifact)
    }
}

/// Honest terminal evidence for a formal request when no in-process proof
/// checker is installed.  It can only derive `Blocked::AdapterUnavailable`.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FormalVerificationEvidenceArtifact {
    binding: EvidenceAuthorityBinding,
    protocol: FormalEvidenceProtocol,
    proof_system: ProofSystem,
    property: KnowledgePredicate,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedFormalVerificationEvidenceArtifactDto {
    binding: EvidenceAuthorityBinding,
    protocol: FormalEvidenceProtocol,
    proof_system: ProofSystem,
    property: KnowledgePredicate,
}

impl FormalVerificationEvidenceArtifact {
    pub fn backend_unavailable(
        invocation: &CognitiveInvocation,
        executor: &ExecutorIdentityDraft,
    ) -> BrainResult<Self> {
        let CognitiveInvocation::VerifyFormalProperty(formal) = invocation else {
            return Err(invalid("formal_evidence_requires_formal_invocation"));
        };
        Ok(Self {
            binding: EvidenceAuthorityBinding::from_invocation(invocation, executor)?,
            protocol: FormalEvidenceProtocol::BackendUnavailableV1,
            proof_system: formal.proof_system,
            property: formal.property.clone(),
        })
    }

    pub fn canonical_bytes(&self) -> BrainResult<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    fn decode_canonical(bytes: &[u8]) -> BrainResult<Self> {
        let dto: UntrustedFormalVerificationEvidenceArtifactDto = serde_json::from_slice(bytes)
            .map_err(|_| integrity("formal_evidence_encoding_invalid"))?;
        let artifact = Self {
            binding: dto.binding,
            protocol: dto.protocol,
            proof_system: dto.proof_system,
            property: dto.property,
        };
        if artifact.canonical_bytes()? != bytes {
            return Err(integrity("formal_evidence_not_canonical"));
        }
        Ok(artifact)
    }
}

/// Private DTO for a future external certificate checker.  The current
/// registry authenticates the envelope, then fails closed because no proof
/// kernel is linked; opaque certificate bytes never self-authorize a result.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedFormalCertificateArtifactDto {
    binding: EvidenceAuthorityBinding,
    protocol: FormalEvidenceProtocol,
    proof_system: ProofSystem,
    property: KnowledgePredicate,
    certificate: Vec<u8>,
}

fn calculate_executor_version_digest(
    executor_id: &ExecutorId,
    interface_version: ExecutorInterfaceVersion,
    declared_implementation_digest: &Sha256Digest,
    observed_implementation_digest: &Sha256Digest,
) -> BrainResult<ExecutorVersionDigest> {
    let version_payload = serde_json::to_vec(&(
        executor_id,
        interface_version,
        declared_implementation_digest,
        observed_implementation_digest,
    ))?;
    Ok(ExecutorVersionDigest::computed(
        b"CEREBRO:TIDEX:COGNITIVE-EXECUTOR-VERSION:v1\0",
        &version_payload,
    ))
}

fn canonical_completed(observations: BTreeMap<ObservationKey, ObservedValue>) -> ActionOutcome {
    ActionOutcome::Completed {
        observations,
        dependency_edges: BTreeSet::new(),
        discovered_obligations: BTreeMap::new(),
    }
}

fn validate_exact_observation_keys(
    observations: &BTreeMap<ObservationKey, ObservedValue>,
    expected: &BTreeSet<ObservationKey>,
    code: &str,
) -> BrainResult<()> {
    if observations.keys().cloned().collect::<BTreeSet<_>>() != *expected {
        return Err(integrity(code));
    }
    Ok(())
}

fn validate_predicate_observation_types<'a>(
    observations: &BTreeMap<ObservationKey, ObservedValue>,
    predicates: impl IntoIterator<Item = &'a KnowledgePredicate>,
    code: &str,
) -> BrainResult<()> {
    if predicates.into_iter().any(|predicate| {
        matches!(
            predicate.evaluate(observations),
            PredicateEvaluation::Missing | PredicateEvaluation::TypeMismatch
        )
    }) {
        return Err(integrity(code));
    }
    Ok(())
}

fn read_single_evidence(
    private_root: &Path,
    evidence: &[EvidenceArtifactInput],
    expected_kind: EvidenceKind,
    code: &str,
) -> BrainResult<Vec<u8>> {
    if evidence.len() != 1 || evidence[0].kind != expected_kind {
        return Err(integrity(code));
    }
    evidence[0]
        .artifact
        .read_verified_bounded(private_root, MAX_EVIDENCE_ARTIFACT_BYTES)
}

/// Registry of semantic evidence verifiers.  It is intentionally fail-closed:
/// a kind has no production authority unless a dedicated verifier is present.
#[derive(Debug, Clone, Default)]
pub struct EvidenceVerifierRegistry;

impl EvidenceVerifierRegistry {
    fn derive_verified_outcome(
        &self,
        context: &EvidenceVerificationContext<'_>,
        proposed: &ActionOutcome,
        evidence: &[EvidenceArtifactInput],
    ) -> BrainResult<ActionOutcome> {
        match (context.invocation, proposed) {
            (_, ActionOutcome::Completed { .. }) => {
                self.derive_completed(context, proposed, evidence)
            }
            (CognitiveInvocation::VerifyFormalProperty(formal), ActionOutcome::Blocked { .. }) => {
                self.derive_formal_backend_unavailable(context, formal, proposed, evidence)
            }
            (
                CognitiveInvocation::VerifyFormalProperty(_),
                ActionOutcome::BoundedNoResult { .. },
            ) => Err(integrity("formal_bounded_result_requires_real_backend")),
            _ => Ok(proposed.clone()),
        }
    }

    /// Derive, rather than trust, a completed outcome from typed evidence. The
    /// executor's proposal is consulted only after the canonical result exists.
    fn derive_completed(
        &self,
        context: &EvidenceVerificationContext<'_>,
        proposed: &ActionOutcome,
        evidence: &[EvidenceArtifactInput],
    ) -> BrainResult<ActionOutcome> {
        let ActionOutcome::Completed {
            dependency_edges,
            discovered_obligations,
            ..
        } = proposed
        else {
            return Err(integrity(
                "completed_evidence_derivation_requires_completed_outcome",
            ));
        };
        if !dependency_edges.is_empty() || !discovered_obligations.is_empty() {
            return Err(integrity(
                "semantic_verifier_forbids_unverified_derivations",
            ));
        }

        let derived = match context.invocation {
            CognitiveInvocation::ObserveTrace(trace) => {
                self.derive_trace(context, trace, evidence)?
            }
            CognitiveInvocation::ExecuteScenario(scenario) => {
                self.derive_scenario(context, scenario, evidence)?
            }
            CognitiveInvocation::CausalIntervention(causal) => {
                self.derive_causal(context, causal, evidence)?
            }
            CognitiveInvocation::SearchCounterexample(search) => {
                self.derive_counterexample(context, search, evidence)?
            }
            CognitiveInvocation::VerifyFormalProperty(formal) => {
                return self.reject_unchecked_formal_certificate(context, formal, evidence);
            }
            CognitiveInvocation::ExpandDependencyClosure(_) => {
                return Err(integrity("dependency_evidence_verifier_not_implemented"));
            }
        };
        if derived != *proposed {
            return Err(integrity("executor_outcome_not_canonically_derived"));
        }
        Ok(derived)
    }

    fn derive_trace(
        &self,
        context: &EvidenceVerificationContext<'_>,
        trace: &TraceInvocation,
        evidence: &[EvidenceArtifactInput],
    ) -> BrainResult<ActionOutcome> {
        let bytes = read_single_evidence(
            context.private_root,
            evidence,
            EvidenceKind::RuntimeTrace,
            "trace_evidence_cardinality_or_kind_invalid",
        )?;
        let artifact = TraceEvidenceArtifact::decode_canonical(&bytes)?;
        artifact.binding.validate(context)?;
        if artifact.protocol != TraceEvidenceProtocol::CanonicalEventStreamV1
            || artifact.observation_keys != trace.observation_keys
            || artifact.max_events != trace.max_events
            || artifact.events.is_empty()
            || u32::try_from(artifact.events.len()).map_or(true, |len| len > trace.max_events)
        {
            return Err(integrity("trace_evidence_protocol_or_inputs_mismatch"));
        }
        let mut observations = BTreeMap::new();
        for (index, event) in artifact.events.into_iter().enumerate() {
            if event.ordinal
                != u32::try_from(index).map_err(|_| invalid("trace_ordinal_overflow"))?
                || event.observations.is_empty()
                || event
                    .observations
                    .keys()
                    .any(|key| !trace.observation_keys.contains(key))
            {
                return Err(integrity("trace_evidence_event_invalid"));
            }
            for (key, value) in event.observations {
                // Ordered event replay defines the latest observed value. The
                // ordinal check above makes repeated measurements canonical.
                observations.insert(key, value);
            }
        }
        validate_exact_observation_keys(
            &observations,
            &trace.observation_keys,
            "trace_evidence_observations_incomplete",
        )?;
        validate_predicate_observation_types(
            &observations,
            std::iter::once(trace.binding.criterion.predicate()),
            "trace_evidence_observation_type_mismatch",
        )?;
        Ok(canonical_completed(observations))
    }

    fn derive_scenario(
        &self,
        context: &EvidenceVerificationContext<'_>,
        scenario: &ScenarioInvocation,
        evidence: &[EvidenceArtifactInput],
    ) -> BrainResult<ActionOutcome> {
        let bytes = read_single_evidence(
            context.private_root,
            evidence,
            EvidenceKind::ScenarioResult,
            "scenario_evidence_cardinality_or_kind_invalid",
        )?;
        let artifact = ScenarioEvidenceArtifact::decode_canonical(&bytes)?;
        artifact.binding.validate(context)?;
        if artifact.protocol != ScenarioEvidenceProtocol::ExactScenarioObservationV1
            || artifact.scenario_id != scenario.scenario_id
            || artifact.inputs != scenario.inputs
            || artifact.observation_keys != scenario.observation_keys
        {
            return Err(integrity("scenario_evidence_protocol_or_inputs_mismatch"));
        }
        validate_exact_observation_keys(
            &artifact.observations,
            &scenario.observation_keys,
            "scenario_evidence_observations_incomplete",
        )?;
        validate_predicate_observation_types(
            &artifact.observations,
            std::iter::once(scenario.binding.criterion.predicate()),
            "scenario_evidence_observation_type_mismatch",
        )?;
        Ok(canonical_completed(artifact.observations))
    }

    fn derive_causal(
        &self,
        context: &EvidenceVerificationContext<'_>,
        causal: &CausalInvocation,
        evidence: &[EvidenceArtifactInput],
    ) -> BrainResult<ActionOutcome> {
        let bytes = read_single_evidence(
            context.private_root,
            evidence,
            EvidenceKind::CausalComparison,
            "causal_evidence_cardinality_or_kind_invalid",
        )?;
        let artifact = CausalEvidenceArtifact::decode_canonical(&bytes)?;
        artifact.binding.validate(context)?;
        if artifact.protocol != CausalEvidenceProtocol::PairedInterventionV1
            || artifact.variable != causal.variable
            || artifact.baseline != causal.baseline
            || artifact.treatment != causal.treatment
            || artifact.effect != causal.effect
            || artifact.predictions != causal.predictions
        {
            return Err(integrity("causal_evidence_protocol_or_inputs_mismatch"));
        }
        let required = context.invocation.required_observations();
        validate_exact_observation_keys(
            &artifact.baseline_observations,
            &required,
            "causal_baseline_observations_incomplete",
        )?;
        validate_exact_observation_keys(
            &artifact.treatment_observations,
            &required,
            "causal_treatment_observations_incomplete",
        )?;
        let predicates = std::iter::once(&causal.effect).chain(causal.predictions.values());
        validate_predicate_observation_types(
            &artifact.baseline_observations,
            predicates.clone(),
            "causal_baseline_observation_type_mismatch",
        )?;
        validate_predicate_observation_types(
            &artifact.treatment_observations,
            predicates,
            "causal_treatment_observation_type_mismatch",
        )?;
        Ok(canonical_completed(artifact.treatment_observations))
    }

    fn derive_counterexample(
        &self,
        context: &EvidenceVerificationContext<'_>,
        search: &CounterexampleInvocation,
        evidence: &[EvidenceArtifactInput],
    ) -> BrainResult<ActionOutcome> {
        let bytes = read_single_evidence(
            context.private_root,
            evidence,
            EvidenceKind::CounterexampleSearch,
            "counterexample_evidence_cardinality_or_kind_invalid",
        )?;
        let artifact = CounterexampleEvidenceArtifact::decode_canonical(&bytes)?;
        artifact.binding.validate(context)?;
        if artifact.protocol != CounterexampleEvidenceProtocol::DeterministicCaseSearchV1
            || artifact.target != search.target
            || artifact.max_cases != search.max_cases
            || artifact.predictions != search.predictions
            || artifact.cases.is_empty()
            || u32::try_from(artifact.cases.len()).map_or(true, |len| len > search.max_cases)
        {
            return Err(integrity(
                "counterexample_evidence_protocol_or_inputs_mismatch",
            ));
        }
        let required = context.invocation.required_observations();
        let predicates = std::iter::once(&search.target).chain(search.predictions.values());
        let mut unique_inputs = BTreeSet::new();
        let mut first_counterexample = None;
        for case in artifact.cases.values() {
            if case.inputs.len() > HARD_MAX_SCENARIO_INPUTS
                || !unique_inputs.insert(serde_json::to_vec(&case.inputs)?)
            {
                return Err(integrity("counterexample_case_inputs_invalid"));
            }
            validate_exact_observation_keys(
                &case.observations,
                &required,
                "counterexample_case_observations_incomplete",
            )?;
            validate_predicate_observation_types(
                &case.observations,
                predicates.clone(),
                "counterexample_case_observation_type_mismatch",
            )?;
            if first_counterexample.is_none()
                && search.target.evaluate(&case.observations) == PredicateEvaluation::Contradicts
            {
                first_counterexample = Some(case.observations.clone());
            }
        }
        let observations = first_counterexample
            .ok_or_else(|| integrity("counterexample_evidence_contains_no_counterexample"))?;
        Ok(canonical_completed(observations))
    }

    fn derive_formal_backend_unavailable(
        &self,
        context: &EvidenceVerificationContext<'_>,
        formal: &FormalVerificationInvocation,
        proposed: &ActionOutcome,
        evidence: &[EvidenceArtifactInput],
    ) -> BrainResult<ActionOutcome> {
        let bytes = read_single_evidence(
            context.private_root,
            evidence,
            EvidenceKind::FormalSearchLog,
            "formal_evidence_cardinality_or_kind_invalid",
        )?;
        let artifact = FormalVerificationEvidenceArtifact::decode_canonical(&bytes)?;
        artifact.binding.validate(context)?;
        if artifact.protocol != FormalEvidenceProtocol::BackendUnavailableV1
            || artifact.proof_system != formal.proof_system
            || artifact.property != formal.property
        {
            return Err(integrity("formal_evidence_protocol_or_inputs_mismatch"));
        }
        let derived = ActionOutcome::Blocked {
            reason: ExecutionBlockReason::AdapterUnavailable,
        };
        if derived != *proposed {
            return Err(integrity("formal_blocked_outcome_not_canonically_derived"));
        }
        Ok(derived)
    }

    fn reject_unchecked_formal_certificate(
        &self,
        context: &EvidenceVerificationContext<'_>,
        formal: &FormalVerificationInvocation,
        evidence: &[EvidenceArtifactInput],
    ) -> BrainResult<ActionOutcome> {
        let bytes = read_single_evidence(
            context.private_root,
            evidence,
            EvidenceKind::FormalCertificate,
            "formal_certificate_cardinality_or_kind_invalid",
        )?;
        let artifact: UntrustedFormalCertificateArtifactDto = serde_json::from_slice(&bytes)
            .map_err(|_| integrity("formal_certificate_encoding_invalid"))?;
        if serde_json::to_vec(&artifact)? != bytes {
            return Err(integrity("formal_certificate_not_canonical"));
        }
        artifact.binding.validate(context)?;
        if artifact.protocol != FormalEvidenceProtocol::ExternalCertificateV1
            || artifact.proof_system != formal.proof_system
            || artifact.property != formal.property
            || artifact.certificate.is_empty()
        {
            return Err(integrity("formal_certificate_protocol_or_inputs_mismatch"));
        }
        Err(integrity("formal_certificate_checker_backend_unavailable"))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveredConcernKind {
    Effect,
    Risk,
}

/// A newly observed effect or risk becomes a real obligation, not a prose note.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DiscoveredKnowledgeObligation {
    pub kind: DiscoveredConcernKind,
    pub domain: KnowledgeDomain,
    pub claim_predicate: KnowledgePredicate,
    pub criterion: EvaluationCriterion,
    pub plan: CognitivePlan,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "outcome")]
pub enum ActionOutcome {
    Completed {
        observations: BTreeMap<ObservationKey, ObservedValue>,
        dependency_edges: BTreeSet<DependencyEdge>,
        discovered_obligations: BTreeMap<DiscoveredConcernId, DiscoveredKnowledgeObligation>,
    },
    BoundedNoResult {
        reason: BoundedActionReason,
    },
    Blocked {
        reason: ExecutionBlockReason,
    },
}

impl ActionOutcome {
    fn observations(&self) -> Option<&BTreeMap<ObservationKey, ObservedValue>> {
        match self {
            Self::Completed { observations, .. } => Some(observations),
            Self::BoundedNoResult { .. } | Self::Blocked { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BoundedActionReason {
    SearchSpaceExhausted,
    ProofInconclusive,
    TraceLimitReached,
    ScenarioIndeterminate,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutorInterfaceVersion {
    #[serde(rename = "cerebro.tidex.cognitive_executor/v1")]
    V1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutorIdentityDraft {
    pub executor_id: ExecutorId,
    pub interface_version: ExecutorInterfaceVersion,
    pub implementation: PrivateFileReference,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ExecutorAttestation {
    executor_id: ExecutorId,
    interface_version: ExecutorInterfaceVersion,
    implementation: PrivateFileReference,
    version_digest: ExecutorVersionDigest,
}

#[derive(Debug, Clone)]
pub struct AdapterExecution {
    pub(crate) outcome: ActionOutcome,
    pub(crate) evidence: Vec<EvidenceArtifactInput>,
    #[cfg(test)]
    semantic_verification: SemanticEvidenceVerification,
}

impl AdapterExecution {
    pub fn new(outcome: ActionOutcome, evidence: Vec<EvidenceArtifactInput>) -> Self {
        Self {
            outcome,
            evidence,
            #[cfg(test)]
            semantic_verification: SemanticEvidenceVerification::Unverified,
        }
    }

    #[cfg(test)]
    fn verified_for_test(outcome: ActionOutcome, evidence: Vec<EvidenceArtifactInput>) -> Self {
        Self {
            outcome,
            evidence,
            semantic_verification: SemanticEvidenceVerification::VerifiedByTestOracle,
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SemanticEvidenceVerification {
    Unverified,
    #[cfg(test)]
    VerifiedByTestOracle,
}

#[cfg(test)]
impl SemanticEvidenceVerification {
    fn authorizes_completed_outcome(self) -> bool {
        match self {
            Self::Unverified => false,
            #[cfg(test)]
            Self::VerifiedByTestOracle => true,
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Module-sealed execution boundary. External code cannot mint cognitive receipts.
///
/// `Completed` outcomes additionally require a typed semantic verifier owned by
/// this module. Merely attaching bytes whose digest matches an evidence record
/// never authorizes a strong epistemic transition.
pub trait CognitiveExecutorAdapter: sealed::Sealed {
    fn identity(&self) -> ExecutorIdentityDraft;
    fn invoke(&self, invocation: &CognitiveInvocation) -> BrainResult<AdapterExecution>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct KnowledgeEvidenceRecord {
    schema: KnowledgeEngineSchema,
    authority_root: AuthorityRootDigest,
    evidence_id: KnowledgeEvidenceId,
    invocation_digest: CognitiveInvocationDigest,
    executor_id: ExecutorId,
    kind: EvidenceKind,
    artifact: PrivateFileReference,
    outcome_digest: ObservationSetDigest,
    manifest_digest: KnowledgeEvidenceDigest,
}

#[derive(Serialize)]
struct KnowledgeEvidenceProjection<'a> {
    schema: KnowledgeEngineSchema,
    authority_root: &'a AuthorityRootDigest,
    evidence_id: &'a KnowledgeEvidenceId,
    invocation_digest: &'a CognitiveInvocationDigest,
    executor_id: &'a ExecutorId,
    kind: EvidenceKind,
    artifact: &'a PrivateFileReference,
    outcome_digest: &'a ObservationSetDigest,
}

impl KnowledgeEvidenceRecord {
    fn calculate_digest(&self) -> BrainResult<KnowledgeEvidenceDigest> {
        let projection = KnowledgeEvidenceProjection {
            schema: self.schema,
            authority_root: &self.authority_root,
            evidence_id: &self.evidence_id,
            invocation_digest: &self.invocation_digest,
            executor_id: &self.executor_id,
            kind: self.kind,
            artifact: &self.artifact,
            outcome_digest: &self.outcome_digest,
        };
        Ok(KnowledgeEvidenceDigest::computed(
            EVIDENCE_DOMAIN,
            &serde_json::to_vec(&projection)?,
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AuthenticatedEvidenceReference {
    evidence_id: KnowledgeEvidenceId,
    kind: EvidenceKind,
    record: PrivateFileReference,
    record_digest: KnowledgeEvidenceDigest,
    source_artifact_digest: Sha256Digest,
}

impl AuthenticatedEvidenceReference {
    pub fn evidence_id(&self) -> &KnowledgeEvidenceId {
        &self.evidence_id
    }

    pub fn kind(&self) -> EvidenceKind {
        self.kind
    }

    pub fn record(&self) -> &PrivateFileReference {
        &self.record
    }

    pub fn digest(&self) -> &KnowledgeEvidenceDigest {
        &self.record_digest
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ActionReceipt {
    schema: KnowledgeEngineSchema,
    authority_root: AuthorityRootDigest,
    inquiry_id: InquiryId,
    capability_bundle: CapabilityBundleDigest,
    parent_state: KnowledgeStateDigest,
    action_id: KnowledgeActionId,
    invocation: CognitiveInvocation,
    invocation_digest: CognitiveInvocationDigest,
    executor: ExecutorAttestation,
    outcome: ActionOutcome,
    evidence: BTreeMap<KnowledgeEvidenceId, AuthenticatedEvidenceReference>,
    charged_cost: u64,
    manifest_digest: ActionReceiptDigest,
}

/// Deserialized receipt bytes are not authority. Authentication returns `ActionReceipt`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedActionReceiptDto {
    schema: KnowledgeEngineSchema,
    authority_root: AuthorityRootDigest,
    inquiry_id: InquiryId,
    capability_bundle: CapabilityBundleDigest,
    parent_state: KnowledgeStateDigest,
    action_id: KnowledgeActionId,
    invocation: CognitiveInvocation,
    invocation_digest: CognitiveInvocationDigest,
    executor: ExecutorAttestation,
    outcome: ActionOutcome,
    evidence: BTreeMap<KnowledgeEvidenceId, AuthenticatedEvidenceReference>,
    charged_cost: u64,
    manifest_digest: ActionReceiptDigest,
}

impl ActionReceipt {
    fn from_untrusted(value: UntrustedActionReceiptDto) -> Self {
        Self {
            schema: value.schema,
            authority_root: value.authority_root,
            inquiry_id: value.inquiry_id,
            capability_bundle: value.capability_bundle,
            parent_state: value.parent_state,
            action_id: value.action_id,
            invocation: value.invocation,
            invocation_digest: value.invocation_digest,
            executor: value.executor,
            outcome: value.outcome,
            evidence: value.evidence,
            charged_cost: value.charged_cost,
            manifest_digest: value.manifest_digest,
        }
    }
}

#[derive(Serialize)]
struct ActionReceiptProjection<'a> {
    schema: KnowledgeEngineSchema,
    authority_root: &'a AuthorityRootDigest,
    inquiry_id: &'a InquiryId,
    capability_bundle: &'a CapabilityBundleDigest,
    parent_state: &'a KnowledgeStateDigest,
    action_id: &'a KnowledgeActionId,
    invocation: &'a CognitiveInvocation,
    invocation_digest: &'a CognitiveInvocationDigest,
    executor: &'a ExecutorAttestation,
    outcome: &'a ActionOutcome,
    evidence: &'a BTreeMap<KnowledgeEvidenceId, AuthenticatedEvidenceReference>,
    charged_cost: u64,
}

impl ActionReceipt {
    pub fn digest(&self) -> &ActionReceiptDigest {
        &self.manifest_digest
    }

    pub fn action_id(&self) -> &KnowledgeActionId {
        &self.action_id
    }

    pub fn invocation(&self) -> &CognitiveInvocation {
        &self.invocation
    }

    pub fn outcome(&self) -> &ActionOutcome {
        &self.outcome
    }

    pub fn evidence(&self) -> &BTreeMap<KnowledgeEvidenceId, AuthenticatedEvidenceReference> {
        &self.evidence
    }

    fn calculate_digest(&self) -> BrainResult<ActionReceiptDigest> {
        let projection = ActionReceiptProjection {
            schema: self.schema,
            authority_root: &self.authority_root,
            inquiry_id: &self.inquiry_id,
            capability_bundle: &self.capability_bundle,
            parent_state: &self.parent_state,
            action_id: &self.action_id,
            invocation: &self.invocation,
            invocation_digest: &self.invocation_digest,
            executor: &self.executor,
            outcome: &self.outcome,
            evidence: &self.evidence,
            charged_cost: self.charged_cost,
        };
        Ok(ActionReceiptDigest::computed(
            RECEIPT_DOMAIN,
            &serde_json::to_vec(&projection)?,
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BoundedUnknownReason {
    ActionBudgetExhausted,
    CostBudgetExhausted,
    RevisionLimitReached,
    EvidenceLimitReached,
    AttemptLimitReached {
        obligation_id: KnowledgeObligationId,
    },
    DependencyNodeLimitReached,
    DependencyEdgeLimitReached,
    DependencyDepthLimitReached {
        node: DependencyNodeId,
    },
    NoAdmissibleInvocation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "terminal")]
pub enum KnowledgeTerminal {
    ScopedComplete {
        state: KnowledgeStateDigest,
        revision: u64,
        established_claims: BTreeSet<KnowledgeClaimId>,
    },
    BoundedUnknown {
        state: KnowledgeStateDigest,
        revision: u64,
        reason: BoundedUnknownReason,
        unresolved_obligations: BTreeSet<KnowledgeObligationId>,
    },
    Blocked {
        state: KnowledgeStateDigest,
        revision: u64,
        blockers: BTreeMap<KnowledgeObligationId, KnowledgeBlockReason>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "decision")]
pub enum PlanningDecision {
    Invoke {
        invocation: Box<CognitiveInvocation>,
    },
    Terminal {
        terminal: KnowledgeTerminal,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeState {
    schema: KnowledgeEngineSchema,
    authority_root: AuthorityRootDigest,
    inquiry_id: InquiryId,
    capability_bundle_reference: PrivateFileReference,
    capability_bundle: CapabilityBundleDigest,
    policy_digest: KnowledgePolicyDigest,
    revision: u64,
    parent_digest: Option<KnowledgeStateDigest>,
    parent_reference: Option<PrivateFileReference>,
    last_action_receipt_reference: Option<PrivateFileReference>,
    budget: KnowledgeBudget,
    usage: BudgetUsage,
    claims: BTreeMap<KnowledgeClaimId, KnowledgeClaim>,
    obligations: BTreeMap<KnowledgeObligationId, KnowledgeObligation>,
    hypotheses: BTreeMap<KnowledgeHypothesisId, KnowledgeHypothesis>,
    dependency_parents: BTreeMap<DependencyNodeId, Option<DependencyNodeId>>,
    dependency_depths: BTreeMap<DependencyNodeId, u16>,
    dependency_graph: BTreeMap<DependencyNodeId, BTreeSet<DependencyNodeId>>,
    bounded_reason: Option<BoundedUnknownReason>,
    applied_receipts: BTreeSet<ActionReceiptDigest>,
    used_evidence_artifacts: BTreeSet<Sha256Digest>,
    manifest_digest: KnowledgeStateDigest,
}

/// Persisted bytes are explicitly untrusted until `KnowledgeEngine::authenticate_state`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedKnowledgeStateDto {
    schema: KnowledgeEngineSchema,
    authority_root: AuthorityRootDigest,
    inquiry_id: InquiryId,
    capability_bundle_reference: PrivateFileReference,
    capability_bundle: CapabilityBundleDigest,
    policy_digest: KnowledgePolicyDigest,
    revision: u64,
    parent_digest: Option<KnowledgeStateDigest>,
    parent_reference: Option<PrivateFileReference>,
    last_action_receipt_reference: Option<PrivateFileReference>,
    budget: KnowledgeBudget,
    usage: BudgetUsage,
    claims: BTreeMap<KnowledgeClaimId, KnowledgeClaim>,
    obligations: BTreeMap<KnowledgeObligationId, KnowledgeObligation>,
    hypotheses: BTreeMap<KnowledgeHypothesisId, KnowledgeHypothesis>,
    dependency_parents: BTreeMap<DependencyNodeId, Option<DependencyNodeId>>,
    dependency_depths: BTreeMap<DependencyNodeId, u16>,
    dependency_graph: BTreeMap<DependencyNodeId, BTreeSet<DependencyNodeId>>,
    bounded_reason: Option<BoundedUnknownReason>,
    applied_receipts: BTreeSet<ActionReceiptDigest>,
    used_evidence_artifacts: BTreeSet<Sha256Digest>,
    manifest_digest: KnowledgeStateDigest,
}

impl From<UntrustedKnowledgeStateDto> for KnowledgeState {
    fn from(value: UntrustedKnowledgeStateDto) -> Self {
        Self {
            schema: value.schema,
            authority_root: value.authority_root,
            inquiry_id: value.inquiry_id,
            capability_bundle_reference: value.capability_bundle_reference,
            capability_bundle: value.capability_bundle,
            policy_digest: value.policy_digest,
            revision: value.revision,
            parent_digest: value.parent_digest,
            parent_reference: value.parent_reference,
            last_action_receipt_reference: value.last_action_receipt_reference,
            budget: value.budget,
            usage: value.usage,
            claims: value.claims,
            obligations: value.obligations,
            hypotheses: value.hypotheses,
            dependency_parents: value.dependency_parents,
            dependency_depths: value.dependency_depths,
            dependency_graph: value.dependency_graph,
            bounded_reason: value.bounded_reason,
            applied_receipts: value.applied_receipts,
            used_evidence_artifacts: value.used_evidence_artifacts,
            manifest_digest: value.manifest_digest,
        }
    }
}

#[derive(Serialize)]
struct KnowledgeStateProjection<'a> {
    schema: KnowledgeEngineSchema,
    authority_root: &'a AuthorityRootDigest,
    inquiry_id: &'a InquiryId,
    capability_bundle_reference: &'a PrivateFileReference,
    capability_bundle: &'a CapabilityBundleDigest,
    policy_digest: &'a KnowledgePolicyDigest,
    revision: u64,
    parent_digest: &'a Option<KnowledgeStateDigest>,
    parent_reference: &'a Option<PrivateFileReference>,
    last_action_receipt_reference: &'a Option<PrivateFileReference>,
    budget: &'a KnowledgeBudget,
    usage: &'a BudgetUsage,
    claims: &'a BTreeMap<KnowledgeClaimId, KnowledgeClaim>,
    obligations: &'a BTreeMap<KnowledgeObligationId, KnowledgeObligation>,
    hypotheses: &'a BTreeMap<KnowledgeHypothesisId, KnowledgeHypothesis>,
    dependency_parents: &'a BTreeMap<DependencyNodeId, Option<DependencyNodeId>>,
    dependency_depths: &'a BTreeMap<DependencyNodeId, u16>,
    dependency_graph: &'a BTreeMap<DependencyNodeId, BTreeSet<DependencyNodeId>>,
    bounded_reason: &'a Option<BoundedUnknownReason>,
    applied_receipts: &'a BTreeSet<ActionReceiptDigest>,
    used_evidence_artifacts: &'a BTreeSet<Sha256Digest>,
}

impl KnowledgeState {
    /// Durable authority identity authenticated together with this state.
    pub fn authority_root(&self) -> &AuthorityRootDigest {
        &self.authority_root
    }

    /// Investigation whose canonical lineage contains this state.
    pub fn inquiry_id(&self) -> &InquiryId {
        &self.inquiry_id
    }

    /// Exact capability bundle this state is authorized to describe.
    pub fn capability_bundle_digest(&self) -> &CapabilityBundleDigest {
        &self.capability_bundle
    }

    /// Knowledge policy under which this state was reduced.
    pub fn policy_digest(&self) -> &KnowledgePolicyDigest {
        &self.policy_digest
    }

    pub fn digest(&self) -> &KnowledgeStateDigest {
        &self.manifest_digest
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn parent_digest(&self) -> Option<&KnowledgeStateDigest> {
        self.parent_digest.as_ref()
    }

    pub fn claims(&self) -> &BTreeMap<KnowledgeClaimId, KnowledgeClaim> {
        &self.claims
    }

    pub fn obligations(&self) -> &BTreeMap<KnowledgeObligationId, KnowledgeObligation> {
        &self.obligations
    }

    pub fn hypotheses(&self) -> &BTreeMap<KnowledgeHypothesisId, KnowledgeHypothesis> {
        &self.hypotheses
    }

    fn calculate_digest(&self) -> BrainResult<KnowledgeStateDigest> {
        let projection = KnowledgeStateProjection {
            schema: self.schema,
            authority_root: &self.authority_root,
            inquiry_id: &self.inquiry_id,
            capability_bundle_reference: &self.capability_bundle_reference,
            capability_bundle: &self.capability_bundle,
            policy_digest: &self.policy_digest,
            revision: self.revision,
            parent_digest: &self.parent_digest,
            parent_reference: &self.parent_reference,
            last_action_receipt_reference: &self.last_action_receipt_reference,
            budget: &self.budget,
            usage: &self.usage,
            claims: &self.claims,
            obligations: &self.obligations,
            hypotheses: &self.hypotheses,
            dependency_parents: &self.dependency_parents,
            dependency_depths: &self.dependency_depths,
            dependency_graph: &self.dependency_graph,
            bounded_reason: &self.bounded_reason,
            applied_receipts: &self.applied_receipts,
            used_evidence_artifacts: &self.used_evidence_artifacts,
        };
        Ok(KnowledgeStateDigest::computed(
            STATE_DOMAIN,
            &serde_json::to_vec(&projection)?,
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct TransitionChanges {
    pub satisfied_obligations: BTreeSet<KnowledgeObligationId>,
    pub blocked_obligations: BTreeSet<KnowledgeObligationId>,
    pub established_claims: BTreeSet<KnowledgeClaimId>,
    pub contradicted_claims: BTreeSet<KnowledgeClaimId>,
    pub supported_hypotheses: BTreeSet<KnowledgeHypothesisId>,
    pub refuted_hypotheses: BTreeSet<KnowledgeHypothesisId>,
    pub added_claims: BTreeSet<KnowledgeClaimId>,
    pub added_obligations: BTreeSet<KnowledgeObligationId>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TransitionReceipt {
    schema: KnowledgeEngineSchema,
    authority_root: AuthorityRootDigest,
    inquiry_id: InquiryId,
    capability_bundle: CapabilityBundleDigest,
    from_state: KnowledgeStateDigest,
    to_state: KnowledgeStateDigest,
    action_id: KnowledgeActionId,
    action_receipt: ActionReceiptDigest,
    from_state_reference: PrivateFileReference,
    to_state_reference: PrivateFileReference,
    action_receipt_reference: PrivateFileReference,
    changes: TransitionChanges,
    terminal_after: Option<KnowledgeTerminal>,
    manifest_digest: TransitionReceiptDigest,
}

#[derive(Serialize)]
struct TransitionProjection<'a> {
    schema: KnowledgeEngineSchema,
    authority_root: &'a AuthorityRootDigest,
    inquiry_id: &'a InquiryId,
    capability_bundle: &'a CapabilityBundleDigest,
    from_state: &'a KnowledgeStateDigest,
    to_state: &'a KnowledgeStateDigest,
    action_id: &'a KnowledgeActionId,
    action_receipt: &'a ActionReceiptDigest,
    from_state_reference: &'a PrivateFileReference,
    to_state_reference: &'a PrivateFileReference,
    action_receipt_reference: &'a PrivateFileReference,
    changes: &'a TransitionChanges,
    terminal_after: &'a Option<KnowledgeTerminal>,
}

impl TransitionReceipt {
    pub fn digest(&self) -> &TransitionReceiptDigest {
        &self.manifest_digest
    }

    pub fn from_state(&self) -> &KnowledgeStateDigest {
        &self.from_state
    }

    pub fn to_state(&self) -> &KnowledgeStateDigest {
        &self.to_state
    }

    pub fn changes(&self) -> &TransitionChanges {
        &self.changes
    }

    fn calculate_digest(&self) -> BrainResult<TransitionReceiptDigest> {
        let projection = TransitionProjection {
            schema: self.schema,
            authority_root: &self.authority_root,
            inquiry_id: &self.inquiry_id,
            capability_bundle: &self.capability_bundle,
            from_state: &self.from_state,
            to_state: &self.to_state,
            action_id: &self.action_id,
            action_receipt: &self.action_receipt,
            from_state_reference: &self.from_state_reference,
            to_state_reference: &self.to_state_reference,
            action_receipt_reference: &self.action_receipt_reference,
            changes: &self.changes,
            terminal_after: &self.terminal_after,
        };
        Ok(TransitionReceiptDigest::computed(
            TRANSITION_DOMAIN,
            &serde_json::to_vec(&projection)?,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeAdvance {
    state: KnowledgeState,
    transition: TransitionReceipt,
}

/// One durable, serializable answer to "which state is authoritative now for
/// this inquiry?".  The digest intentionally excludes local file references:
/// those are authenticated locators, while the semantic identity is composed
/// only of authority instance, inquiry, immutable state/receipt identities and
/// revision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CanonicalInquiryHead {
    schema: KnowledgeEngineSchema,
    authority_root: AuthorityRootDigest,
    inquiry_id: InquiryId,
    genesis_state: KnowledgeStateDigest,
    genesis_reference: PrivateFileReference,
    current_state: KnowledgeStateDigest,
    current_reference: PrivateFileReference,
    revision: u64,
    action_receipts: BTreeMap<KnowledgeActionId, ActionReceiptDigest>,
    manifest_digest: CanonicalInquiryHeadDigest,
}

#[derive(Serialize)]
struct CanonicalInquiryHeadProjection<'a> {
    schema: KnowledgeEngineSchema,
    authority_root: &'a AuthorityRootDigest,
    inquiry_id: &'a InquiryId,
    genesis_state: &'a KnowledgeStateDigest,
    current_state: &'a KnowledgeStateDigest,
    revision: u64,
    action_receipts: &'a BTreeMap<KnowledgeActionId, ActionReceiptDigest>,
}

impl CanonicalInquiryHead {
    pub fn digest(&self) -> &CanonicalInquiryHeadDigest {
        &self.manifest_digest
    }

    pub fn inquiry_id(&self) -> &InquiryId {
        &self.inquiry_id
    }

    pub fn genesis_state(&self) -> &KnowledgeStateDigest {
        &self.genesis_state
    }

    pub fn current_state(&self) -> &KnowledgeStateDigest {
        &self.current_state
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn receipt_for_action(
        &self,
        action_id: &KnowledgeActionId,
    ) -> Option<&ActionReceiptDigest> {
        self.action_receipts.get(action_id)
    }

    fn calculate_digest(&self) -> BrainResult<CanonicalInquiryHeadDigest> {
        Ok(CanonicalInquiryHeadDigest::computed(
            CANONICAL_HEAD_DOMAIN,
            &serde_json::to_vec(&CanonicalInquiryHeadProjection {
                schema: self.schema,
                authority_root: &self.authority_root,
                inquiry_id: &self.inquiry_id,
                genesis_state: &self.genesis_state,
                current_state: &self.current_state,
                revision: self.revision,
                action_receipts: &self.action_receipts,
            })?,
        ))
    }
}

impl KnowledgeAdvance {
    pub fn state(&self) -> &KnowledgeState {
        &self.state
    }

    pub fn transition(&self) -> &TransitionReceipt {
        &self.transition
    }

    pub fn into_parts(self) -> (KnowledgeState, TransitionReceipt) {
        (self.state, self.transition)
    }
}

#[derive(Debug, Clone, Serialize)]
struct PolicyProjection {
    schema: KnowledgeEngineSchema,
    profile_version: u16,
    planner_version: u16,
    reducer_version: u16,
    canonical_encoding_version: u16,
    evidence_verifier_profile_version: u16,
    max_claims: usize,
    max_obligations: usize,
    max_hypotheses: usize,
    max_attempts_per_obligation: u16,
}

#[derive(Debug, Clone)]
struct KnowledgePolicy {
    digest: KnowledgePolicyDigest,
    max_attempts_per_obligation: u16,
}

impl KnowledgePolicy {
    fn conservative_v1() -> BrainResult<Self> {
        let projection = PolicyProjection {
            schema: KnowledgeEngineSchema::V1,
            profile_version: 2,
            planner_version: 1,
            reducer_version: 1,
            canonical_encoding_version: 2,
            evidence_verifier_profile_version: 2,
            max_claims: HARD_MAX_CLAIMS,
            max_obligations: HARD_MAX_OBLIGATIONS,
            max_hypotheses: HARD_MAX_HYPOTHESES,
            max_attempts_per_obligation: 3,
        };
        Ok(Self {
            digest: KnowledgePolicyDigest::computed(
                POLICY_DOMAIN,
                &serde_json::to_vec(&projection)?,
            ),
            max_attempts_per_obligation: projection.max_attempts_per_obligation,
        })
    }
}

/// Root-bound authority for planning, sealing observations, reducing and persisting knowledge.
#[derive(Debug, Clone)]
pub struct KnowledgeEngine {
    private_root: PathBuf,
    authority_instance: AuthorityInstanceId,
    authority_root: AuthorityRootDigest,
    policy: KnowledgePolicy,
    evidence_verifiers: EvidenceVerifierRegistry,
}

impl KnowledgeEngine {
    /// Open only when the caller supplies the durable identity provisioned for
    /// this authority instance.  Directory names are deployment details and
    /// are never knowledge-authority identity.
    pub fn open_with_authority_instance(
        private_root: &Path,
        authority_instance: AuthorityInstanceId,
    ) -> BrainResult<Self> {
        let root = verify_private_root(private_root)?;
        Self::from_verified_root(root, authority_instance)
    }

    /// The old root-derived identity has no portable meaning.  Keep the entry
    /// point fail-closed until bootstrap owns persistent instance provisioning.
    pub fn open(private_root: &Path) -> BrainResult<Self> {
        let _ = verify_private_root(private_root)?;
        Err(invalid("authority_instance_required"))
    }

    fn from_verified_root(
        private_root: PathBuf,
        authority_instance: AuthorityInstanceId,
    ) -> BrainResult<Self> {
        let authority_root =
            AuthorityRootDigest::computed(ROOT_DOMAIN, &serde_json::to_vec(&authority_instance)?);
        Ok(Self {
            private_root,
            authority_instance,
            authority_root,
            policy: KnowledgePolicy::conservative_v1()?,
            evidence_verifiers: EvidenceVerifierRegistry,
        })
    }

    #[cfg(test)]
    fn for_test(private_root: &Path) -> BrainResult<Self> {
        Self::from_verified_root(
            private_root.to_path_buf(),
            AuthorityInstanceId::parse("test-authority-instance.v1")?,
        )
    }

    /// Validate and seal a capability-specific contract as revision zero.
    pub fn initialize(
        &self,
        inquiry_id: InquiryId,
        capability_bundle_reference: PrivateFileReference,
        draft: KnowledgeContractDraft,
    ) -> BrainResult<KnowledgeState> {
        let bundle = self.authenticate_bundle(&capability_bundle_reference)?;
        self.validate_contract(&draft)?;

        let mut claims = BTreeMap::new();
        for claim in draft.claims {
            let claim_id = claim.claim_id.clone();
            if claims
                .insert(
                    claim_id.clone(),
                    KnowledgeClaim {
                        claim_id,
                        domain: claim.domain,
                        predicate: claim.predicate,
                        assessment: ClaimAssessment::Unresolved,
                    },
                )
                .is_some()
            {
                return Err(invalid("knowledge_claim_duplicate"));
            }
        }

        let mut obligations = BTreeMap::new();
        let mut dependency_parents = BTreeMap::new();
        let mut dependency_depths = BTreeMap::new();
        let mut dependency_graph = BTreeMap::new();
        for obligation in draft.obligations {
            let claim = claims
                .get(&obligation.claim_id)
                .ok_or_else(|| invalid("knowledge_obligation_claim_missing"))?;
            if let CognitivePlan::ExpandDependencyClosure { root, depth, .. } = &obligation.plan {
                if dependency_parents.insert(root.clone(), None).is_none() {
                    dependency_depths.insert(root.clone(), *depth);
                    dependency_graph.insert(root.clone(), BTreeSet::new());
                }
            }
            let obligation_id = obligation.obligation_id.clone();
            if obligations
                .insert(
                    obligation_id.clone(),
                    KnowledgeObligation {
                        obligation_id,
                        claim_id: obligation.claim_id,
                        domain: claim.domain.clone(),
                        criterion: obligation.criterion,
                        plan: obligation.plan,
                        state: KnowledgeObligationState::Open {
                            attempts: 0,
                            witnesses: BTreeSet::new(),
                        },
                    },
                )
                .is_some()
            {
                return Err(invalid("knowledge_obligation_duplicate"));
            }
        }

        let mut hypotheses = BTreeMap::new();
        for hypothesis in draft.hypotheses {
            let hypothesis_id = hypothesis.hypothesis_id.clone();
            if hypotheses
                .insert(
                    hypothesis_id.clone(),
                    KnowledgeHypothesis {
                        hypothesis_id,
                        family_id: hypothesis.family_id,
                        definition: hypothesis.definition,
                        assessment: HypothesisAssessment::Viable,
                    },
                )
                .is_some()
            {
                return Err(invalid("knowledge_hypothesis_duplicate"));
            }
        }

        let mut state = KnowledgeState {
            schema: KnowledgeEngineSchema::V1,
            authority_root: self.authority_root.clone(),
            inquiry_id,
            capability_bundle_reference,
            capability_bundle: bundle.manifest_digest().clone(),
            policy_digest: self.policy.digest.clone(),
            revision: 0,
            parent_digest: None,
            parent_reference: None,
            last_action_receipt_reference: None,
            budget: draft.budget,
            usage: BudgetUsage {
                actions: 0,
                total_cost: 0,
                evidence_records: 0,
            },
            claims,
            obligations,
            hypotheses,
            dependency_parents,
            dependency_depths,
            dependency_graph,
            bounded_reason: None,
            applied_receipts: BTreeSet::new(),
            used_evidence_artifacts: BTreeSet::new(),
            manifest_digest: KnowledgeStateDigest::computed(STATE_DOMAIN, b"unsealed"),
        };
        state.manifest_digest = state.calculate_digest()?;
        self.validate_state(&state)?;
        Ok(state)
    }

    fn validate_contract(&self, draft: &KnowledgeContractDraft) -> BrainResult<()> {
        draft.budget.validate()?;
        if draft.claims.is_empty()
            || draft.obligations.is_empty()
            || draft.claims.len() > HARD_MAX_CLAIMS
            || draft.obligations.len() > HARD_MAX_OBLIGATIONS
            || draft.hypotheses.len() > HARD_MAX_HYPOTHESES
        {
            return Err(invalid("knowledge_contract_cardinality_invalid"));
        }

        let mut claims = BTreeMap::new();
        let mut observation_schema = BTreeMap::new();
        for claim in &draft.claims {
            if claims.insert(&claim.claim_id, claim).is_some() {
                return Err(invalid("knowledge_claim_duplicate"));
            }
            record_predicate_type(&mut observation_schema, &claim.predicate)?;
        }

        let mut obligation_ids = BTreeSet::new();
        let mut obligations_by_claim: BTreeMap<&KnowledgeClaimId, Vec<&KnowledgeObligationDraft>> =
            BTreeMap::new();
        for obligation in &draft.obligations {
            if !obligation_ids.insert(&obligation.obligation_id) {
                return Err(invalid("knowledge_obligation_duplicate"));
            }
            let claim = claims
                .get(&obligation.claim_id)
                .ok_or_else(|| invalid("knowledge_obligation_claim_missing"))?;
            if !matches!(claim.domain, KnowledgeDomain::Profiled { .. })
                && claim.domain != obligation.plan.domain()
            {
                return Err(invalid("knowledge_obligation_domain_method_mismatch"));
            }
            obligation.plan.validate_bounds(&draft.budget)?;
            validate_criterion_plan_binding(&obligation.criterion, &obligation.plan)?;
            record_predicate_type(&mut observation_schema, obligation.criterion.predicate())?;
            for predicate in plan_predicates(&obligation.plan) {
                record_predicate_type(&mut observation_schema, predicate)?;
            }
            if matches!(
                obligation.plan,
                CognitivePlan::ExpandDependencyClosure { depth, .. } if depth != 0
            ) {
                return Err(invalid("initial_dependency_root_depth_nonzero"));
            }
            obligations_by_claim
                .entry(&obligation.claim_id)
                .or_default()
                .push(obligation);
        }

        for claim in &draft.claims {
            let obligations = obligations_by_claim
                .get(&claim.claim_id)
                .ok_or_else(|| invalid("knowledge_claim_without_obligation"))?;
            if !obligations
                .iter()
                .any(|obligation| obligation.criterion.predicate() == &claim.predicate)
            {
                return Err(invalid("knowledge_claim_without_direct_criterion"));
            }
        }

        let mut hypotheses = BTreeMap::new();
        let mut families: BTreeMap<&HypothesisFamilyId, BTreeSet<&KnowledgeHypothesisId>> =
            BTreeMap::new();
        for hypothesis in &draft.hypotheses {
            if hypotheses
                .insert(&hypothesis.hypothesis_id, hypothesis)
                .is_some()
            {
                return Err(invalid("knowledge_hypothesis_duplicate"));
            }
            families
                .entry(&hypothesis.family_id)
                .or_default()
                .insert(&hypothesis.hypothesis_id);
            record_predicate_type(&mut observation_schema, &hypothesis.definition)?;
        }
        if families.values().any(|family| family.len() < 2) {
            return Err(invalid("hypothesis_family_without_rivals"));
        }

        let mut predicted_families = BTreeSet::new();
        for obligation in &draft.obligations {
            let Some(predictions) = obligation.plan.predictions() else {
                continue;
            };
            let mut family_id: Option<&HypothesisFamilyId> = None;
            for hypothesis_id in predictions.keys() {
                let hypothesis = hypotheses
                    .get(hypothesis_id)
                    .ok_or_else(|| invalid("prediction_hypothesis_missing"))?;
                if let Some(expected_family) = family_id {
                    if expected_family != &hypothesis.family_id {
                        return Err(invalid("prediction_crosses_hypothesis_families"));
                    }
                } else {
                    family_id = Some(&hypothesis.family_id);
                }
            }
            let family_id = family_id.ok_or_else(|| invalid("prediction_family_missing"))?;
            if predictions.len() != families[family_id].len()
                || !predictions
                    .keys()
                    .all(|hypothesis_id| families[family_id].contains(hypothesis_id))
            {
                return Err(invalid("prediction_does_not_cover_rival_family"));
            }
            predicted_families.insert(family_id);
        }
        if predicted_families.len() != families.len() {
            return Err(invalid(
                "hypothesis_family_without_precommitted_predictions",
            ));
        }
        Ok(())
    }

    fn authenticate_bundle(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<CapabilityBundle> {
        let bundle = authenticate_capability_bundle(&self.private_root, reference)?;
        if bundle.manifest_digest().is_draft() {
            return Err(integrity("knowledge_bundle_unsealed"));
        }
        Ok(bundle)
    }

    pub fn plan(&self, state: &KnowledgeState) -> BrainResult<PlanningDecision> {
        self.validate_state(state)?;
        if let Some(terminal) = self.derive_terminal(state)? {
            return Ok(PlanningDecision::Terminal { terminal });
        }

        let mut open: Vec<&KnowledgeObligation> = state
            .obligations
            .values()
            .filter(|obligation| matches!(obligation.state, KnowledgeObligationState::Open { .. }))
            .collect();
        open.sort_by(|left, right| left.obligation_id.cmp(&right.obligation_id));
        let has_open_obligations = !open.is_empty();

        for obligation in open {
            let cost = obligation.plan.cost();
            if state.usage.total_cost.saturating_add(cost) > state.budget.max_total_cost {
                continue;
            }
            let invocation = self.materialize_invocation(state, obligation)?;
            return Ok(PlanningDecision::Invoke {
                invocation: Box::new(invocation),
            });
        }

        let reason = if has_open_obligations {
            BoundedUnknownReason::CostBudgetExhausted
        } else {
            BoundedUnknownReason::NoAdmissibleInvocation
        };
        Ok(PlanningDecision::Terminal {
            terminal: self.bounded_terminal(state, reason),
        })
    }

    fn materialize_invocation(
        &self,
        state: &KnowledgeState,
        obligation: &KnowledgeObligation,
    ) -> BrainResult<CognitiveInvocation> {
        let next_attempt = obligation
            .state
            .attempts()
            .checked_add(1)
            .ok_or_else(|| invalid("knowledge_attempt_overflow"))?;
        let action_payload = serde_json::to_vec(&(
            &state.manifest_digest,
            &obligation.obligation_id,
            next_attempt,
        ))?;
        let action_id = KnowledgeActionId::computed(ACTION_ID_DOMAIN, &action_payload);
        let binding = InvocationBinding {
            authority_instance: self.authority_instance.clone(),
            authority_root: self.authority_root.clone(),
            inquiry_id: state.inquiry_id.clone(),
            capability_bundle: state.capability_bundle.clone(),
            parent_state: state.manifest_digest.clone(),
            action_id,
            obligation_id: obligation.obligation_id.clone(),
            claim_id: obligation.claim_id.clone(),
            attempt: next_attempt,
            criterion: obligation.criterion.clone(),
        };
        Ok(match &obligation.plan {
            CognitivePlan::ObserveTrace {
                observation_keys,
                max_events,
            } => CognitiveInvocation::ObserveTrace(TraceInvocation {
                binding,
                observation_keys: observation_keys.clone(),
                max_events: *max_events,
            }),
            CognitivePlan::ExecuteScenario {
                scenario_id,
                inputs,
                observation_keys,
            } => CognitiveInvocation::ExecuteScenario(ScenarioInvocation {
                binding,
                scenario_id: scenario_id.clone(),
                inputs: inputs.clone(),
                observation_keys: observation_keys.clone(),
            }),
            CognitivePlan::CausalIntervention {
                variable,
                baseline,
                treatment,
                effect,
                predictions,
            } => CognitiveInvocation::CausalIntervention(CausalInvocation {
                binding,
                variable: variable.clone(),
                baseline: baseline.clone(),
                treatment: treatment.clone(),
                effect: effect.clone(),
                predictions: predictions.clone(),
            }),
            CognitivePlan::SearchCounterexample {
                target,
                max_cases,
                predictions,
            } => CognitiveInvocation::SearchCounterexample(CounterexampleInvocation {
                binding,
                target: target.clone(),
                max_cases: *max_cases,
                predictions: predictions.clone(),
            }),
            CognitivePlan::VerifyFormalProperty {
                proof_system,
                property,
            } => CognitiveInvocation::VerifyFormalProperty(FormalVerificationInvocation {
                binding,
                proof_system: *proof_system,
                property: property.clone(),
            }),
            CognitivePlan::ExpandDependencyClosure {
                root,
                depth,
                max_depth,
            } => CognitiveInvocation::ExpandDependencyClosure(DependencyExpansionInvocation {
                binding,
                root: root.clone(),
                depth: *depth,
                max_depth: *max_depth,
            }),
        })
    }

    fn validate_state(&self, state: &KnowledgeState) -> BrainResult<()> {
        if state.schema != KnowledgeEngineSchema::V1
            || state.authority_root != self.authority_root
            || state.policy_digest != self.policy.digest
        {
            return Err(integrity("knowledge_state_authority_or_policy_mismatch"));
        }
        state.budget.validate()?;
        if state.claims.is_empty()
            || state.obligations.is_empty()
            || state.claims.len() > HARD_MAX_CLAIMS
            || state.obligations.len() > HARD_MAX_OBLIGATIONS
            || state.hypotheses.len() > HARD_MAX_HYPOTHESES
            || state.revision > state.budget.max_revision
            || state.usage.actions > state.budget.max_actions
            || state.usage.total_cost > state.budget.max_total_cost
            || state.usage.evidence_records > state.budget.max_evidence_records
            || state.dependency_parents.len() > state.budget.max_dependency_nodes
            || dependency_edge_count(&state.dependency_graph)? > state.budget.max_dependency_edges
        {
            return Err(invalid("knowledge_state_bounds_invalid"));
        }
        if (state.revision == 0) != state.parent_digest.is_none()
            || state.parent_digest.is_none() != state.parent_reference.is_none()
            || (state.revision == 0) != state.last_action_receipt_reference.is_none()
            || state.revision != state.usage.actions
            || state.applied_receipts.len() as u64 != state.usage.actions
            || state.used_evidence_artifacts.len() != state.usage.evidence_records
        {
            return Err(integrity("knowledge_state_revision_accounting_mismatch"));
        }
        if state.revision == 0
            && (state.usage.total_cost != 0
                || state.usage.evidence_records != 0
                || !state.applied_receipts.is_empty()
                || !state.used_evidence_artifacts.is_empty()
                || state.bounded_reason.is_some()
                || state
                    .claims
                    .values()
                    .any(|claim| !matches!(claim.assessment, ClaimAssessment::Unresolved))
                || state.obligations.values().any(|obligation| {
                    !matches!(
                        &obligation.state,
                        KnowledgeObligationState::Open {
                            attempts: 0,
                            witnesses
                        } if witnesses.is_empty()
                    )
                })
                || state.hypotheses.values().any(|hypothesis| {
                    !matches!(hypothesis.assessment, HypothesisAssessment::Viable)
                })
                || state.dependency_parents.values().any(Option::is_some)
                || state.dependency_depths.values().any(|depth| *depth != 0)
                || state
                    .dependency_graph
                    .values()
                    .any(|targets| !targets.is_empty()))
        {
            return Err(integrity("knowledge_state_origin_not_canonical"));
        }
        if let (Some(parent_digest), Some(parent_reference)) =
            (&state.parent_digest, &state.parent_reference)
        {
            let expected_path = self.semantic_path("states", parent_digest.as_str());
            let parent_bytes =
                parent_reference.read_verified_bounded(&self.private_root, MAX_STATE_BYTES)?;
            let parent_dto: UntrustedKnowledgeStateDto = serde_json::from_slice(&parent_bytes)?;
            let parent = KnowledgeState::from(parent_dto);
            if parent_reference.path != expected_path
                || parent.manifest_digest != *parent_digest
                || parent.revision.checked_add(1) != Some(state.revision)
                || parent.inquiry_id != state.inquiry_id
                || parent.capability_bundle != state.capability_bundle
                || parent.calculate_digest()? != *parent_digest
            {
                return Err(integrity("knowledge_state_parent_invalid"));
            }
            let new_receipts: Vec<&ActionReceiptDigest> = state
                .applied_receipts
                .difference(&parent.applied_receipts)
                .collect();
            let last_reference = state
                .last_action_receipt_reference
                .as_ref()
                .ok_or_else(|| integrity("knowledge_state_last_receipt_missing"))?;
            let receipt_bytes = last_reference
                .read_verified_bounded(&self.private_root, MAX_ACTION_RECEIPT_BYTES)?;
            let receipt = ActionReceipt::from_untrusted(serde_json::from_slice::<
                UntrustedActionReceiptDto,
            >(&receipt_bytes)?);
            if new_receipts.len() != 1
                || new_receipts[0] != &receipt.manifest_digest
                || last_reference.path
                    != self.semantic_path("action_receipts", receipt.manifest_digest.as_str())
                || receipt.parent_state != *parent_digest
                || receipt.calculate_digest()? != receipt.manifest_digest
                || state.usage.total_cost
                    != parent.usage.total_cost.saturating_add(receipt.charged_cost)
                || state.usage.evidence_records
                    != parent
                        .usage
                        .evidence_records
                        .saturating_add(receipt.evidence.len())
            {
                return Err(integrity("knowledge_state_last_receipt_invalid"));
            }
        }

        let bundle = self.authenticate_bundle(&state.capability_bundle_reference)?;
        if bundle.manifest_digest() != &state.capability_bundle {
            return Err(integrity("knowledge_state_bundle_mismatch"));
        }

        let mut observation_schema = BTreeMap::new();
        for (claim_id, claim) in &state.claims {
            if claim_id != &claim.claim_id {
                return Err(integrity("knowledge_claim_map_key_mismatch"));
            }
            record_predicate_type(&mut observation_schema, &claim.predicate)?;
        }

        let mut obligations_by_claim: BTreeMap<&KnowledgeClaimId, Vec<&KnowledgeObligation>> =
            BTreeMap::new();
        for (obligation_id, obligation) in &state.obligations {
            if obligation_id != &obligation.obligation_id {
                return Err(integrity("knowledge_obligation_map_key_mismatch"));
            }
            let claim = state
                .claims
                .get(&obligation.claim_id)
                .ok_or_else(|| integrity("knowledge_obligation_claim_missing"))?;
            if claim.domain != obligation.domain
                || (!matches!(claim.domain, KnowledgeDomain::Profiled { .. })
                    && claim.domain != obligation.plan.domain())
            {
                return Err(integrity("knowledge_obligation_domain_mismatch"));
            }
            obligation.plan.validate_bounds(&state.budget)?;
            validate_criterion_plan_binding(&obligation.criterion, &obligation.plan)?;
            record_predicate_type(&mut observation_schema, obligation.criterion.predicate())?;
            for predicate in plan_predicates(&obligation.plan) {
                record_predicate_type(&mut observation_schema, predicate)?;
            }
            if let CognitivePlan::ExpandDependencyClosure { root, depth, .. } = &obligation.plan {
                if state.dependency_depths.get(root) != Some(depth)
                    || !state.dependency_parents.contains_key(root)
                {
                    return Err(integrity("knowledge_dependency_plan_not_indexed"));
                }
            }
            if obligation
                .state
                .witnesses()
                .iter()
                .any(|receipt| !state.applied_receipts.contains(receipt))
            {
                return Err(integrity("knowledge_obligation_unapplied_witness"));
            }
            obligations_by_claim
                .entry(&obligation.claim_id)
                .or_default()
                .push(obligation);
        }

        for claim in state.claims.values() {
            let obligations = obligations_by_claim
                .get(&claim.claim_id)
                .ok_or_else(|| integrity("knowledge_claim_without_obligation"))?;
            if !obligations
                .iter()
                .any(|obligation| obligation.criterion.predicate() == &claim.predicate)
            {
                return Err(integrity("knowledge_claim_without_direct_criterion"));
            }
            let expected = derive_claim_assessment(obligations);
            if claim.assessment != expected {
                return Err(integrity("knowledge_claim_assessment_not_reduced"));
            }
        }

        let mut families: BTreeMap<&HypothesisFamilyId, Vec<&KnowledgeHypothesis>> =
            BTreeMap::new();
        for (hypothesis_id, hypothesis) in &state.hypotheses {
            if hypothesis_id != &hypothesis.hypothesis_id {
                return Err(integrity("knowledge_hypothesis_map_key_mismatch"));
            }
            record_predicate_type(&mut observation_schema, &hypothesis.definition)?;
            for receipt in hypothesis_witnesses(&hypothesis.assessment) {
                if !state.applied_receipts.contains(receipt) {
                    return Err(integrity("knowledge_hypothesis_unapplied_witness"));
                }
            }
            families
                .entry(&hypothesis.family_id)
                .or_default()
                .push(hypothesis);
        }
        if families.values().any(|family| family.len() < 2) {
            return Err(integrity("knowledge_hypothesis_family_without_rivals"));
        }
        let mut predicted_families = BTreeSet::new();
        for obligation in state.obligations.values() {
            let Some(predictions) = obligation.plan.predictions() else {
                continue;
            };
            let mut family_id: Option<&HypothesisFamilyId> = None;
            for hypothesis_id in predictions.keys() {
                let hypothesis = state
                    .hypotheses
                    .get(hypothesis_id)
                    .ok_or_else(|| integrity("state_prediction_hypothesis_missing"))?;
                if let Some(expected_family) = family_id {
                    if expected_family != &hypothesis.family_id {
                        return Err(integrity("state_prediction_crosses_families"));
                    }
                } else {
                    family_id = Some(&hypothesis.family_id);
                }
            }
            let family_id = family_id.ok_or_else(|| integrity("state_prediction_empty"))?;
            let family = families
                .get(family_id)
                .ok_or_else(|| integrity("state_prediction_family_missing"))?;
            if predictions.len() != family.len()
                || !predictions
                    .keys()
                    .all(|hypothesis_id| family.iter().any(|item| item.id() == hypothesis_id))
            {
                return Err(integrity("state_prediction_family_incomplete"));
            }
            predicted_families.insert(family_id);
        }
        if predicted_families.len() != families.len() {
            return Err(integrity("state_hypothesis_family_without_predictions"));
        }

        if state.dependency_parents.len() != state.dependency_depths.len()
            || state.dependency_graph.len() != state.dependency_depths.len()
            || state
                .dependency_graph
                .keys()
                .any(|node| !state.dependency_depths.contains_key(node))
            || state.dependency_graph.values().any(|targets| {
                targets
                    .iter()
                    .any(|target| !state.dependency_graph.contains_key(target))
            })
            || state
                .dependency_depths
                .values()
                .any(|depth| *depth > state.budget.max_dependency_depth)
            || !dependency_graph_is_acyclic(
                &state.dependency_graph,
                state.budget.max_dependency_nodes,
            )?
        {
            return Err(integrity("knowledge_dependency_index_mismatch"));
        }
        for (node, parent) in &state.dependency_parents {
            let depth = state
                .dependency_depths
                .get(node)
                .ok_or_else(|| integrity("knowledge_dependency_depth_missing"))?;
            if let Some(parent) = parent {
                let parent_depth = state
                    .dependency_depths
                    .get(parent)
                    .ok_or_else(|| integrity("knowledge_dependency_parent_missing"))?;
                if *depth != parent_depth.saturating_add(1) {
                    return Err(integrity("knowledge_dependency_depth_invalid"));
                }
                if !state
                    .dependency_graph
                    .get(parent)
                    .is_some_and(|targets| targets.contains(node))
                {
                    return Err(integrity("knowledge_dependency_parent_edge_missing"));
                }
            }
        }

        if state.calculate_digest()? != state.manifest_digest {
            return Err(integrity("knowledge_state_digest_mismatch"));
        }
        Ok(())
    }

    fn derive_terminal(&self, state: &KnowledgeState) -> BrainResult<Option<KnowledgeTerminal>> {
        let mut blockers = BTreeMap::new();
        for (obligation_id, obligation) in &state.obligations {
            if let KnowledgeObligationState::Blocked { reason, .. } = &obligation.state {
                blockers.insert(obligation_id.clone(), reason.clone());
            }
        }
        for hypothesis in state.hypotheses.values() {
            if matches!(
                hypothesis.assessment,
                HypothesisAssessment::Conflicted { .. }
            ) {
                let obligation_id = state
                    .obligations
                    .values()
                    .find(|obligation| {
                        obligation.plan.predictions().is_some_and(|predictions| {
                            predictions.contains_key(&hypothesis.hypothesis_id)
                        })
                    })
                    .map(|obligation| obligation.obligation_id.clone())
                    .ok_or_else(|| integrity("conflicted_hypothesis_without_obligation"))?;
                blockers.insert(
                    obligation_id,
                    KnowledgeBlockReason::HypothesisConflict {
                        family_id: hypothesis.family_id.clone(),
                        hypothesis_id: hypothesis.hypothesis_id.clone(),
                    },
                );
            }
        }
        if !blockers.is_empty() {
            return Ok(Some(KnowledgeTerminal::Blocked {
                state: state.manifest_digest.clone(),
                revision: state.revision,
                blockers,
            }));
        }

        if let Some(reason) = &state.bounded_reason {
            return Ok(Some(self.bounded_terminal(state, reason.clone())));
        }

        let all_satisfied = state.obligations.values().all(|obligation| {
            matches!(obligation.state, KnowledgeObligationState::Satisfied { .. })
        });
        if all_satisfied && hypothesis_families_resolved(&state.hypotheses) {
            return Ok(Some(KnowledgeTerminal::ScopedComplete {
                state: state.manifest_digest.clone(),
                revision: state.revision,
                established_claims: state.claims.keys().cloned().collect(),
            }));
        }

        if state.usage.actions >= state.budget.max_actions {
            return Ok(Some(self.bounded_terminal(
                state,
                BoundedUnknownReason::ActionBudgetExhausted,
            )));
        }
        if state.revision >= state.budget.max_revision {
            return Ok(Some(self.bounded_terminal(
                state,
                BoundedUnknownReason::RevisionLimitReached,
            )));
        }
        if state.usage.evidence_records >= state.budget.max_evidence_records {
            return Ok(Some(self.bounded_terminal(
                state,
                BoundedUnknownReason::EvidenceLimitReached,
            )));
        }
        if let Some(obligation) = state.obligations.values().find(|obligation| {
            matches!(obligation.state, KnowledgeObligationState::Open { .. })
                && obligation.state.attempts() >= self.policy.max_attempts_per_obligation
        }) {
            return Ok(Some(self.bounded_terminal(
                state,
                BoundedUnknownReason::AttemptLimitReached {
                    obligation_id: obligation.obligation_id.clone(),
                },
            )));
        }
        if state.dependency_parents.len() >= state.budget.max_dependency_nodes
            && state.obligations.values().any(|obligation| {
                matches!(obligation.state, KnowledgeObligationState::Open { .. })
                    && matches!(
                        obligation.plan,
                        CognitivePlan::ExpandDependencyClosure { .. }
                    )
            })
        {
            return Ok(Some(self.bounded_terminal(
                state,
                BoundedUnknownReason::DependencyNodeLimitReached,
            )));
        }
        if dependency_edge_count(&state.dependency_graph)? >= state.budget.max_dependency_edges
            && state.obligations.values().any(|obligation| {
                matches!(obligation.state, KnowledgeObligationState::Open { .. })
                    && matches!(
                        obligation.plan,
                        CognitivePlan::ExpandDependencyClosure { .. }
                    )
            })
        {
            return Ok(Some(self.bounded_terminal(
                state,
                BoundedUnknownReason::DependencyEdgeLimitReached,
            )));
        }
        if all_satisfied {
            return Ok(Some(self.bounded_terminal(
                state,
                BoundedUnknownReason::NoAdmissibleInvocation,
            )));
        }
        Ok(None)
    }

    fn bounded_terminal(
        &self,
        state: &KnowledgeState,
        reason: BoundedUnknownReason,
    ) -> KnowledgeTerminal {
        KnowledgeTerminal::BoundedUnknown {
            state: state.manifest_digest.clone(),
            revision: state.revision,
            reason,
            unresolved_obligations: state
                .obligations
                .iter()
                .filter_map(|(id, obligation)| {
                    (!matches!(obligation.state, KnowledgeObligationState::Satisfied { .. }))
                        .then_some(id.clone())
                })
                .collect(),
        }
    }

    /// Execute exactly the invocation selected by the internal planner.
    ///
    /// This remains crate-visible because only crate-sealed adapters are an
    /// execution authority. Public callers may plan, persist and inspect, but
    /// cannot manufacture observations or receipts.
    pub fn execute<A: CognitiveExecutorAdapter>(
        &self,
        state: &KnowledgeState,
        adapter: &A,
    ) -> BrainResult<ActionReceipt> {
        let invocation = match self.plan(state)? {
            PlanningDecision::Invoke { invocation } => *invocation,
            PlanningDecision::Terminal { .. } => {
                return Err(invalid("knowledge_state_has_no_executable_action"));
            }
        };
        let identity = self.authenticate_executor(adapter.identity())?;
        let execution = adapter.invoke(&invocation)?;
        self.issue_action_receipt(state, invocation, identity, execution)
    }

    fn authenticate_executor(
        &self,
        draft: ExecutorIdentityDraft,
    ) -> BrainResult<ExecutorAttestation> {
        let implementation_bytes = draft
            .implementation
            .read_verified_bounded(&self.private_root, MAX_EXECUTOR_IMPLEMENTATION_BYTES)?;
        let observed_implementation_digest = Sha256Digest::digest_bytes(&implementation_bytes);
        let version_digest = calculate_executor_version_digest(
            &draft.executor_id,
            draft.interface_version,
            &draft.implementation.sha256,
            &observed_implementation_digest,
        )?;
        Ok(ExecutorAttestation {
            executor_id: draft.executor_id,
            interface_version: draft.interface_version,
            implementation: draft.implementation,
            version_digest,
        })
    }

    fn issue_action_receipt(
        &self,
        state: &KnowledgeState,
        invocation: CognitiveInvocation,
        executor: ExecutorAttestation,
        execution: AdapterExecution,
    ) -> BrainResult<ActionReceipt> {
        self.validate_state(state)?;
        self.validate_execution_output(state, &invocation, &executor, &execution)?;
        let invocation_digest = invocation.digest()?;
        let outcome_digest = ObservationSetDigest::computed(
            OBSERVATION_DOMAIN,
            &serde_json::to_vec(&execution.outcome)?,
        );

        let mut evidence = BTreeMap::new();
        let mut source_digests = BTreeSet::new();
        for input in execution.evidence {
            let source_bytes = input
                .artifact
                .read_verified_bounded(&self.private_root, MAX_EVIDENCE_ARTIFACT_BYTES)?;
            if source_bytes.is_empty()
                || input.artifact.sha256 == executor.implementation.sha256
                || !source_digests.insert(input.artifact.sha256.clone())
                || state
                    .used_evidence_artifacts
                    .contains(&input.artifact.sha256)
            {
                return Err(integrity("cognitive_evidence_empty_reused_or_executable"));
            }
            let id_payload = serde_json::to_vec(&(
                &invocation_digest,
                &executor.executor_id,
                input.kind,
                &input.artifact.sha256,
            ))?;
            let evidence_id = KnowledgeEvidenceId::computed(EVIDENCE_ID_DOMAIN, &id_payload);
            let mut record = KnowledgeEvidenceRecord {
                schema: KnowledgeEngineSchema::V1,
                authority_root: self.authority_root.clone(),
                evidence_id: evidence_id.clone(),
                invocation_digest: invocation_digest.clone(),
                executor_id: executor.executor_id.clone(),
                kind: input.kind,
                artifact: input.artifact,
                outcome_digest: outcome_digest.clone(),
                manifest_digest: KnowledgeEvidenceDigest::computed(EVIDENCE_DOMAIN, b"unsealed"),
            };
            record.manifest_digest = record.calculate_digest()?;
            let destination = self.semantic_path("evidence", record.manifest_digest.as_str());
            let bytes = serde_json::to_vec(&record)?;
            let raw_digest = write_or_verify_immutable(&self.private_root, &destination, &bytes)?;
            let reference = AuthenticatedEvidenceReference {
                evidence_id: evidence_id.clone(),
                kind: record.kind,
                record: PrivateFileReference::new(destination, raw_digest),
                record_digest: record.manifest_digest,
                source_artifact_digest: record.artifact.sha256,
            };
            if evidence.insert(evidence_id, reference).is_some() {
                return Err(integrity("cognitive_evidence_identity_collision"));
            }
        }

        let binding = invocation.binding().clone();
        let charged_cost = invocation.cost();
        let mut receipt = ActionReceipt {
            schema: KnowledgeEngineSchema::V1,
            authority_root: self.authority_root.clone(),
            inquiry_id: binding.inquiry_id,
            capability_bundle: binding.capability_bundle,
            parent_state: binding.parent_state,
            action_id: binding.action_id,
            invocation,
            invocation_digest,
            executor,
            outcome: execution.outcome,
            evidence,
            charged_cost,
            manifest_digest: ActionReceiptDigest::computed(RECEIPT_DOMAIN, b"unsealed"),
        };
        receipt.manifest_digest = receipt.calculate_digest()?;
        self.validate_action_receipt(state, &receipt)?;
        self.persist_action_receipt(&receipt)?;
        Ok(receipt)
    }

    fn validate_execution_output(
        &self,
        state: &KnowledgeState,
        invocation: &CognitiveInvocation,
        executor: &ExecutorAttestation,
        execution: &AdapterExecution,
    ) -> BrainResult<()> {
        #[cfg(test)]
        let verified_by_test_oracle = execution
            .semantic_verification
            .authorizes_completed_outcome();
        #[cfg(not(test))]
        let verified_by_test_oracle = false;
        let requires_semantic_derivation =
            matches!(execution.outcome, ActionOutcome::Completed { .. })
                || matches!(invocation, CognitiveInvocation::VerifyFormalProperty(_));
        if requires_semantic_derivation && !verified_by_test_oracle {
            let context = EvidenceVerificationContext {
                private_root: &self.private_root,
                authority_instance: &self.authority_instance,
                authority_root: &self.authority_root,
                invocation,
                executor,
            };
            let derived = self.evidence_verifiers.derive_verified_outcome(
                &context,
                &execution.outcome,
                &execution.evidence,
            )?;
            if derived != execution.outcome {
                return Err(integrity("executor_outcome_not_semantically_derived"));
            }
        }
        if execution.evidence.is_empty()
            || state
                .usage
                .evidence_records
                .saturating_add(execution.evidence.len())
                > state.budget.max_evidence_records
        {
            return Err(invalid("cognitive_execution_evidence_cardinality_invalid"));
        }
        let expected_kinds = required_evidence_kinds(invocation, &execution.outcome);
        let actual_kinds: BTreeSet<EvidenceKind> =
            execution.evidence.iter().map(|item| item.kind).collect();
        if actual_kinds.len() != execution.evidence.len() || actual_kinds != expected_kinds {
            return Err(invalid("cognitive_execution_evidence_kind_invalid"));
        }
        executor.implementation.verify(&self.private_root)?;

        self.validate_outcome_shape(state, invocation, &execution.outcome)?;
        Ok(())
    }

    fn validate_outcome_shape(
        &self,
        state: &KnowledgeState,
        invocation: &CognitiveInvocation,
        outcome: &ActionOutcome,
    ) -> BrainResult<()> {
        match outcome {
            ActionOutcome::Completed {
                observations,
                dependency_edges,
                discovered_obligations,
            } => {
                if observations.len() > state.budget.max_observations_per_action
                    || dependency_edges.len() > HARD_MAX_DEPENDENCY_EDGES_PER_ACTION
                    || dependency_edge_count(&state.dependency_graph)?
                        .saturating_add(dependency_edges.len())
                        > state.budget.max_dependency_edges
                    || discovered_obligations.len() > HARD_MAX_DISCOVERIES_PER_ACTION
                    || invocation
                        .required_observations()
                        .iter()
                        .any(|key| !observations.contains_key(key))
                {
                    return Err(invalid("cognitive_execution_observations_incomplete"));
                }
                if !discovered_obligations.is_empty() {
                    return Err(integrity(
                        "discovery_requires_authenticated_derivation_policy",
                    ));
                }
                match invocation {
                    CognitiveInvocation::ExpandDependencyClosure(value) => {
                        if dependency_edges.iter().any(|edge| edge.from != value.root) {
                            return Err(invalid("dependency_evidence_wrong_source"));
                        }
                    }
                    _ if !dependency_edges.is_empty() => {
                        return Err(invalid("dependency_edges_on_wrong_invocation"));
                    }
                    _ => {}
                }
                let mut discovery_schema = observation_schema_from_state(state)?;
                for discovery in discovered_obligations.values() {
                    discovery.plan.validate_bounds(&state.budget)?;
                    validate_criterion_plan_binding(&discovery.criterion, &discovery.plan)?;
                    if discovery.claim_predicate != *discovery.criterion.predicate()
                        || (!matches!(&discovery.domain, KnowledgeDomain::Profiled { .. })
                            && discovery.domain != discovery.plan.domain())
                    {
                        return Err(invalid("discovered_obligation_invalid"));
                    }
                    record_predicate_type(&mut discovery_schema, &discovery.claim_predicate)?;
                    record_predicate_type(&mut discovery_schema, discovery.criterion.predicate())?;
                    for predicate in plan_predicates(&discovery.plan) {
                        record_predicate_type(&mut discovery_schema, predicate)?;
                    }
                }
            }
            ActionOutcome::BoundedNoResult { .. } | ActionOutcome::Blocked { .. } => {}
        }
        Ok(())
    }

    fn validate_action_receipt(
        &self,
        state: &KnowledgeState,
        receipt: &ActionReceipt,
    ) -> BrainResult<()> {
        if receipt.schema != KnowledgeEngineSchema::V1
            || receipt.authority_root != self.authority_root
            || receipt.inquiry_id != state.inquiry_id
            || receipt.capability_bundle != state.capability_bundle
            || receipt.parent_state != state.manifest_digest
            || receipt.action_id != *receipt.invocation.action_id()
            || receipt.invocation_digest != receipt.invocation.digest()?
            || receipt.charged_cost != receipt.invocation.cost()
            || receipt.manifest_digest != receipt.calculate_digest()?
            || state.applied_receipts.contains(&receipt.manifest_digest)
        {
            return Err(integrity("action_receipt_binding_or_digest_mismatch"));
        }
        let expected_invocation = match self.plan(state)? {
            PlanningDecision::Invoke { invocation } => *invocation,
            PlanningDecision::Terminal { .. } => {
                return Err(integrity("action_receipt_for_terminal_state"));
            }
        };
        if receipt.invocation != expected_invocation {
            return Err(integrity("action_receipt_not_current_planned_invocation"));
        }
        self.validate_outcome_shape(state, &receipt.invocation, &receipt.outcome)?;
        let executor = self.authenticate_executor(ExecutorIdentityDraft {
            executor_id: receipt.executor.executor_id.clone(),
            interface_version: receipt.executor.interface_version,
            implementation: receipt.executor.implementation.clone(),
        })?;
        if executor != receipt.executor || receipt.evidence.is_empty() {
            return Err(integrity("action_receipt_executor_invalid"));
        }

        let outcome_digest = ObservationSetDigest::computed(
            OBSERVATION_DOMAIN,
            &serde_json::to_vec(&receipt.outcome)?,
        );
        let expected_kinds = required_evidence_kinds(&receipt.invocation, &receipt.outcome);
        if state
            .usage
            .evidence_records
            .saturating_add(receipt.evidence.len())
            > state.budget.max_evidence_records
        {
            return Err(invalid("action_receipt_evidence_limit_exceeded"));
        }
        let mut source_digests = BTreeSet::new();
        let mut evidence_kinds = BTreeSet::new();
        let mut verified_evidence = Vec::new();
        for (evidence_id, reference) in &receipt.evidence {
            if evidence_id != &reference.evidence_id
                || !expected_kinds.contains(&reference.kind)
                || reference.source_artifact_digest == receipt.executor.implementation.sha256
                || state
                    .used_evidence_artifacts
                    .contains(&reference.source_artifact_digest)
                || !source_digests.insert(reference.source_artifact_digest.clone())
                || !evidence_kinds.insert(reference.kind)
            {
                return Err(integrity("action_receipt_evidence_binding_invalid"));
            }
            let record = self.authenticate_evidence_record(
                reference,
                &receipt.invocation_digest,
                &receipt.executor.executor_id,
                &outcome_digest,
            )?;
            verified_evidence.push(EvidenceArtifactInput {
                kind: record.kind,
                artifact: record.artifact,
            });
        }
        if evidence_kinds != expected_kinds {
            return Err(integrity("action_receipt_evidence_kind_set_invalid"));
        }
        // Unit fixtures may use the module-only test oracle for non-scenario
        // plans.  A normal build has no such escape hatch: persisted completed
        // receipts must pass a registry verifier again on every load.
        #[cfg(not(test))]
        if matches!(receipt.outcome, ActionOutcome::Completed { .. })
            || matches!(
                receipt.invocation,
                CognitiveInvocation::VerifyFormalProperty(_)
            )
        {
            let context = EvidenceVerificationContext {
                private_root: &self.private_root,
                authority_instance: &self.authority_instance,
                authority_root: &self.authority_root,
                invocation: &receipt.invocation,
                executor: &executor,
            };
            let derived = self.evidence_verifiers.derive_verified_outcome(
                &context,
                &receipt.outcome,
                &verified_evidence,
            )?;
            if derived != receipt.outcome {
                return Err(integrity("receipt_outcome_not_semantically_derived"));
            }
        }
        Ok(())
    }

    fn authenticate_evidence_record(
        &self,
        reference: &AuthenticatedEvidenceReference,
        invocation_digest: &CognitiveInvocationDigest,
        executor_id: &ExecutorId,
        outcome_digest: &ObservationSetDigest,
    ) -> BrainResult<KnowledgeEvidenceRecord> {
        let bytes = reference
            .record
            .read_verified_bounded(&self.private_root, MAX_EVIDENCE_RECORD_BYTES)?;
        let record: KnowledgeEvidenceRecord = serde_json::from_slice(&bytes)?;
        let expected_path = self.semantic_path("evidence", record.manifest_digest.as_str());
        if reference.record.path != expected_path
            || record.schema != KnowledgeEngineSchema::V1
            || record.authority_root != self.authority_root
            || record.evidence_id != reference.evidence_id
            || record.invocation_digest != *invocation_digest
            || record.executor_id != *executor_id
            || record.kind != reference.kind
            || record.outcome_digest != *outcome_digest
            || record.artifact.sha256 != reference.source_artifact_digest
            || record.manifest_digest != reference.record_digest
            || record.calculate_digest()? != record.manifest_digest
        {
            return Err(integrity("knowledge_evidence_record_mismatch"));
        }
        let artifact_bytes = record
            .artifact
            .read_verified_bounded(&self.private_root, MAX_EVIDENCE_ARTIFACT_BYTES)?;
        if artifact_bytes.is_empty() {
            return Err(integrity("knowledge_evidence_artifact_empty"));
        }
        Ok(record)
    }

    fn semantic_path(&self, class: &str, semantic_digest: &str) -> PathBuf {
        self.private_root
            .join("state/knowledge_engine")
            .join(class)
            .join("by-sha")
            .join(format!("{semantic_digest}.json"))
    }

    pub fn persist_action_receipt(
        &self,
        receipt: &ActionReceipt,
    ) -> BrainResult<PrivateFileReference> {
        if receipt.authority_root != self.authority_root
            || receipt.manifest_digest != receipt.calculate_digest()?
        {
            return Err(integrity("action_receipt_not_authentic"));
        }
        let destination = self.semantic_path("action_receipts", receipt.manifest_digest.as_str());
        let bytes = serde_json::to_vec(receipt)?;
        let raw_digest = write_or_verify_immutable(&self.private_root, &destination, &bytes)?;
        Ok(PrivateFileReference::new(destination, raw_digest))
    }

    pub fn authenticate_action_receipt(
        &self,
        state: &KnowledgeState,
        reference: &PrivateFileReference,
    ) -> BrainResult<ActionReceipt> {
        let bytes =
            reference.read_verified_bounded(&self.private_root, MAX_ACTION_RECEIPT_BYTES)?;
        let dto: UntrustedActionReceiptDto = serde_json::from_slice(&bytes)?;
        let receipt = ActionReceipt::from_untrusted(dto);
        let expected_path = self.semantic_path("action_receipts", receipt.manifest_digest.as_str());
        if reference.path != expected_path {
            return Err(integrity("action_receipt_content_address_mismatch"));
        }
        self.validate_action_receipt(state, &receipt)?;
        Ok(receipt)
    }

    /// Deterministically reduce one authenticated receipt into exactly one successor revision.
    pub fn advance(
        &self,
        state: &KnowledgeState,
        receipt: &ActionReceipt,
    ) -> BrainResult<KnowledgeAdvance> {
        self.validate_state(state)?;
        self.validate_action_receipt(state, receipt)?;

        let parent_reference = self.state_reference(state)?;
        let action_receipt_reference = self.action_receipt_reference(receipt)?;
        let persisted_parent =
            parent_reference.read_verified_bounded(&self.private_root, MAX_STATE_BYTES)?;
        if persisted_parent != serde_json::to_vec(state)? {
            return Err(integrity("knowledge_parent_state_bytes_mismatch"));
        }

        let receipt_digest = receipt.manifest_digest.clone();
        let obligation_id = receipt.invocation.obligation_id().clone();
        let criterion = receipt.invocation.binding().criterion.clone();
        let mut next = state.clone();
        next.parent_digest = Some(state.manifest_digest.clone());
        next.parent_reference = Some(parent_reference.clone());
        next.last_action_receipt_reference = Some(action_receipt_reference.clone());
        next.revision = state
            .revision
            .checked_add(1)
            .ok_or_else(|| invalid("knowledge_revision_overflow"))?;
        next.usage.actions = next
            .usage
            .actions
            .checked_add(1)
            .ok_or_else(|| invalid("knowledge_action_count_overflow"))?;
        next.usage.total_cost = next
            .usage
            .total_cost
            .checked_add(receipt.charged_cost)
            .ok_or_else(|| invalid("knowledge_cost_overflow"))?;
        next.usage.evidence_records = next
            .usage
            .evidence_records
            .checked_add(receipt.evidence.len())
            .ok_or_else(|| invalid("knowledge_evidence_count_overflow"))?;
        if !next.applied_receipts.insert(receipt_digest.clone()) {
            return Err(integrity("action_receipt_replayed"));
        }
        for evidence in receipt.evidence.values() {
            if !next
                .used_evidence_artifacts
                .insert(evidence.source_artifact_digest.clone())
            {
                return Err(integrity("action_receipt_source_evidence_replayed"));
            }
        }

        let mut changes = TransitionChanges::default();
        let prior_claims: BTreeMap<KnowledgeClaimId, ClaimAssessment> = next
            .claims
            .iter()
            .map(|(id, claim)| (id.clone(), claim.assessment.clone()))
            .collect();
        let prior_obligations: BTreeMap<KnowledgeObligationId, KnowledgeObligationState> = next
            .obligations
            .iter()
            .map(|(id, obligation)| (id.clone(), obligation.state.clone()))
            .collect();
        let prior_hypotheses: BTreeMap<KnowledgeHypothesisId, HypothesisAssessment> = next
            .hypotheses
            .iter()
            .map(|(id, hypothesis)| (id.clone(), hypothesis.assessment.clone()))
            .collect();

        if let (Some(predictions), Some(observations)) = (
            receipt.invocation.predictions(),
            receipt.outcome.observations(),
        ) {
            for (hypothesis_id, prediction) in predictions {
                let hypothesis = next
                    .hypotheses
                    .get_mut(hypothesis_id)
                    .ok_or_else(|| integrity("receipt_prediction_hypothesis_missing"))?;
                hypothesis.assessment = reduce_hypothesis_assessment(
                    &hypothesis.assessment,
                    prediction.evaluate(observations),
                    &receipt_digest,
                )?;
            }
        }

        let prior_obligation = next
            .obligations
            .get(&obligation_id)
            .ok_or_else(|| integrity("action_obligation_missing"))?
            .clone();
        let attempts = prior_obligation
            .state
            .attempts()
            .checked_add(1)
            .ok_or_else(|| invalid("knowledge_attempt_overflow"))?;
        let mut witnesses = prior_obligation.state.witnesses().clone();
        witnesses.insert(receipt_digest.clone());

        let reduced_state = match &receipt.outcome {
            ActionOutcome::Completed { observations, .. } => {
                match criterion.predicate().evaluate(observations) {
                    PredicateEvaluation::Matches => KnowledgeObligationState::Satisfied {
                        attempts,
                        witnesses,
                    },
                    PredicateEvaluation::Contradicts => KnowledgeObligationState::Blocked {
                        attempts,
                        witnesses,
                        reason: KnowledgeBlockReason::ContradictoryObservation {
                            receipt: receipt_digest.clone(),
                        },
                    },
                    PredicateEvaluation::Missing => {
                        return Err(integrity("receipt_missing_precommitted_criterion"));
                    }
                    PredicateEvaluation::TypeMismatch => {
                        return Err(integrity("receipt_criterion_observation_type_mismatch"));
                    }
                }
            }
            ActionOutcome::BoundedNoResult { .. } => KnowledgeObligationState::Open {
                attempts,
                witnesses,
            },
            ActionOutcome::Blocked { reason } => KnowledgeObligationState::Blocked {
                attempts,
                witnesses,
                reason: KnowledgeBlockReason::ExecutorBlocked {
                    receipt: receipt_digest.clone(),
                    reason: reason.clone(),
                },
            },
        };
        next.obligations
            .get_mut(&obligation_id)
            .ok_or_else(|| integrity("action_obligation_missing"))?
            .state = reduced_state;

        if let (
            CognitiveInvocation::ExpandDependencyClosure(invocation),
            ActionOutcome::Completed {
                dependency_edges, ..
            },
        ) = (&receipt.invocation, &receipt.outcome)
        {
            self.reduce_dependency_expansion(
                &mut next,
                invocation,
                dependency_edges,
                &receipt_digest,
                &mut changes,
            )?;
        }
        if let ActionOutcome::Completed {
            discovered_obligations,
            ..
        } = &receipt.outcome
        {
            self.reduce_discovered_obligations(&mut next, discovered_obligations, &mut changes)?;
        }

        recompute_claim_assessments(&mut next.claims, &next.obligations)?;
        record_transition_changes(
            &next,
            &prior_claims,
            &prior_obligations,
            &prior_hypotheses,
            &mut changes,
        );

        next.manifest_digest = next.calculate_digest()?;
        self.validate_state(&next)?;
        let terminal_after = self.derive_terminal(&next)?;
        let next_reference = self.state_reference(&next)?;
        action_receipt_reference
            .read_verified_bounded(&self.private_root, MAX_ACTION_RECEIPT_BYTES)?;
        let mut transition = TransitionReceipt {
            schema: KnowledgeEngineSchema::V1,
            authority_root: self.authority_root.clone(),
            inquiry_id: next.inquiry_id.clone(),
            capability_bundle: next.capability_bundle.clone(),
            from_state: state.manifest_digest.clone(),
            to_state: next.manifest_digest.clone(),
            action_id: receipt.action_id.clone(),
            action_receipt: receipt_digest,
            from_state_reference: parent_reference,
            to_state_reference: next_reference,
            action_receipt_reference,
            changes,
            terminal_after,
            manifest_digest: TransitionReceiptDigest::computed(TRANSITION_DOMAIN, b"unsealed"),
        };
        transition.manifest_digest = transition.calculate_digest()?;
        Ok(KnowledgeAdvance {
            state: next,
            transition,
        })
    }

    fn reduce_dependency_expansion(
        &self,
        state: &mut KnowledgeState,
        invocation: &DependencyExpansionInvocation,
        edges: &BTreeSet<DependencyEdge>,
        receipt_digest: &ActionReceiptDigest,
        changes: &mut TransitionChanges,
    ) -> BrainResult<()> {
        let source_depth = *state
            .dependency_depths
            .get(&invocation.root)
            .ok_or_else(|| integrity("dependency_source_not_indexed"))?;
        if source_depth != invocation.depth {
            return Err(integrity("dependency_invocation_depth_mismatch"));
        }

        for edge in edges {
            if edge.from != invocation.root {
                return Err(integrity("dependency_edge_source_mismatch"));
            }
            if edge.to == edge.from
                || dependency_graph_reachable(
                    &state.dependency_graph,
                    &edge.to,
                    &edge.from,
                    state.budget.max_dependency_nodes,
                )?
            {
                let obligation = state
                    .obligations
                    .get_mut(&invocation.binding.obligation_id)
                    .ok_or_else(|| integrity("dependency_obligation_missing"))?;
                let attempts = obligation.state.attempts();
                let witnesses = obligation.state.witnesses().clone();
                obligation.state = KnowledgeObligationState::Blocked {
                    attempts,
                    witnesses,
                    reason: KnowledgeBlockReason::DependencyCycle {
                        receipt: receipt_digest.clone(),
                        from: edge.from.clone(),
                        to: edge.to.clone(),
                    },
                };
                changes
                    .blocked_obligations
                    .insert(obligation.obligation_id.clone());
                continue;
            }
            if state.dependency_parents.contains_key(&edge.to) {
                state
                    .dependency_graph
                    .get_mut(&edge.from)
                    .ok_or_else(|| integrity("dependency_graph_source_missing"))?
                    .insert(edge.to.clone());
                continue;
            }
            let child_depth = source_depth
                .checked_add(1)
                .ok_or_else(|| invalid("dependency_depth_overflow"))?;
            if child_depth > invocation.max_depth || child_depth > state.budget.max_dependency_depth
            {
                state.bounded_reason = Some(BoundedUnknownReason::DependencyDepthLimitReached {
                    node: edge.to.clone(),
                });
                continue;
            }
            if state.dependency_parents.len() >= state.budget.max_dependency_nodes {
                state.bounded_reason = Some(BoundedUnknownReason::DependencyNodeLimitReached);
                continue;
            }

            let identity_payload =
                serde_json::to_vec(&(&state.inquiry_id, &state.capability_bundle, &edge.to))?;
            let claim_id = KnowledgeClaimId::derived_discovery(&identity_payload);
            let obligation_id = KnowledgeObligationId::derived_discovery(&identity_payload);
            if state.claims.contains_key(&claim_id)
                || state.obligations.contains_key(&obligation_id)
            {
                return Err(integrity("derived_dependency_identity_collision"));
            }
            let predicate = dependency_closure_predicate(&edge.to)?;
            state.claims.insert(
                claim_id.clone(),
                KnowledgeClaim {
                    claim_id: claim_id.clone(),
                    domain: KnowledgeDomain::DependencyClosure,
                    predicate: predicate.clone(),
                    assessment: ClaimAssessment::Unresolved,
                },
            );
            state.obligations.insert(
                obligation_id.clone(),
                KnowledgeObligation {
                    obligation_id: obligation_id.clone(),
                    claim_id: claim_id.clone(),
                    domain: KnowledgeDomain::DependencyClosure,
                    criterion: EvaluationCriterion::exact(predicate),
                    plan: CognitivePlan::ExpandDependencyClosure {
                        root: edge.to.clone(),
                        depth: child_depth,
                        max_depth: invocation.max_depth,
                    },
                    state: KnowledgeObligationState::Open {
                        attempts: 0,
                        witnesses: BTreeSet::new(),
                    },
                },
            );
            state
                .dependency_parents
                .insert(edge.to.clone(), Some(edge.from.clone()));
            state.dependency_depths.insert(edge.to.clone(), child_depth);
            state
                .dependency_graph
                .insert(edge.to.clone(), BTreeSet::new());
            state
                .dependency_graph
                .get_mut(&edge.from)
                .ok_or_else(|| integrity("dependency_graph_source_missing"))?
                .insert(edge.to.clone());
            changes.added_claims.insert(claim_id);
            changes.added_obligations.insert(obligation_id);
        }
        Ok(())
    }

    fn reduce_discovered_obligations(
        &self,
        state: &mut KnowledgeState,
        discoveries: &BTreeMap<DiscoveredConcernId, DiscoveredKnowledgeObligation>,
        changes: &mut TransitionChanges,
    ) -> BrainResult<()> {
        for (concern_id, discovery) in discoveries {
            if state.claims.len() >= HARD_MAX_CLAIMS
                || state.obligations.len() >= HARD_MAX_OBLIGATIONS
            {
                state.bounded_reason = Some(BoundedUnknownReason::NoAdmissibleInvocation);
                break;
            }
            let identity_payload = serde_json::to_vec(&(
                &state.inquiry_id,
                &state.capability_bundle,
                concern_id,
                discovery.kind,
            ))?;
            let claim_id = KnowledgeClaimId::derived_discovery(&identity_payload);
            let obligation_id = KnowledgeObligationId::derived_discovery(&identity_payload);
            if let (Some(existing_claim), Some(existing_obligation)) = (
                state.claims.get(&claim_id),
                state.obligations.get(&obligation_id),
            ) {
                if existing_claim.domain != discovery.domain
                    || existing_claim.predicate != discovery.claim_predicate
                    || existing_obligation.criterion != discovery.criterion
                    || existing_obligation.plan != discovery.plan
                {
                    return Err(integrity("discovered_concern_identity_collision"));
                }
                continue;
            }
            if state.claims.contains_key(&claim_id)
                || state.obligations.contains_key(&obligation_id)
            {
                return Err(integrity("discovered_concern_partial_identity_collision"));
            }
            if let CognitivePlan::ExpandDependencyClosure { root, depth, .. } = &discovery.plan {
                match state.dependency_depths.get(root) {
                    Some(existing_depth) if existing_depth != depth => {
                        return Err(integrity("discovered_dependency_depth_conflict"));
                    }
                    Some(_) => {}
                    None if *depth == 0 => {
                        if state.dependency_parents.len() >= state.budget.max_dependency_nodes {
                            state.bounded_reason =
                                Some(BoundedUnknownReason::DependencyNodeLimitReached);
                            break;
                        }
                        state.dependency_parents.insert(root.clone(), None);
                        state.dependency_depths.insert(root.clone(), 0);
                        state.dependency_graph.insert(root.clone(), BTreeSet::new());
                    }
                    None => {
                        return Err(integrity(
                            "discovered_dependency_without_authenticated_parent",
                        ));
                    }
                }
            }
            state.claims.insert(
                claim_id.clone(),
                KnowledgeClaim {
                    claim_id: claim_id.clone(),
                    domain: discovery.domain.clone(),
                    predicate: discovery.claim_predicate.clone(),
                    assessment: ClaimAssessment::Unresolved,
                },
            );
            state.obligations.insert(
                obligation_id.clone(),
                KnowledgeObligation {
                    obligation_id: obligation_id.clone(),
                    claim_id: claim_id.clone(),
                    domain: discovery.domain.clone(),
                    criterion: discovery.criterion.clone(),
                    plan: discovery.plan.clone(),
                    state: KnowledgeObligationState::Open {
                        attempts: 0,
                        witnesses: BTreeSet::new(),
                    },
                },
            );
            changes.added_claims.insert(claim_id);
            changes.added_obligations.insert(obligation_id);
        }
        Ok(())
    }

    pub fn persist_state(&self, state: &KnowledgeState) -> BrainResult<PrivateFileReference> {
        self.validate_state(state)?;
        if let (Some(parent), Some(parent_reference)) =
            (&state.parent_digest, &state.parent_reference)
        {
            let parent_bytes =
                parent_reference.read_verified_bounded(&self.private_root, MAX_STATE_BYTES)?;
            let parent_dto: UntrustedKnowledgeStateDto = serde_json::from_slice(&parent_bytes)?;
            let parent_state = KnowledgeState::from(parent_dto);
            if parent_reference.path != self.semantic_path("states", parent.as_str())
                || parent_state.manifest_digest != *parent
                || parent_state.revision.checked_add(1) != Some(state.revision)
                || parent_state.calculate_digest()? != *parent
            {
                return Err(integrity("knowledge_state_parent_invalid"));
            }
        }
        let destination = self.semantic_path("states", state.manifest_digest.as_str());
        let bytes = serde_json::to_vec(state)?;
        let raw_digest = write_or_verify_immutable(&self.private_root, &destination, &bytes)?;
        Ok(PrivateFileReference::new(destination, raw_digest))
    }

    pub fn authenticate_state(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<KnowledgeState> {
        let state = self.read_state_reference(reference)?;
        let mut reverse_chain = vec![state.clone()];
        while let Some(parent_reference) = reverse_chain
            .last()
            .and_then(|current| current.parent_reference.as_ref())
        {
            if reverse_chain.len() as u64 > HARD_MAX_REVISION {
                return Err(invalid("knowledge_state_lineage_limit_exceeded"));
            }
            reverse_chain.push(self.read_state_reference(parent_reference)?);
        }
        reverse_chain.reverse();
        if reverse_chain
            .first()
            .is_none_or(|initial| initial.revision != 0)
        {
            return Err(integrity("knowledge_state_lineage_missing_origin"));
        }
        for pair in reverse_chain.windows(2) {
            let parent = &pair[0];
            let child = &pair[1];
            let receipt_reference = child
                .last_action_receipt_reference
                .as_ref()
                .ok_or_else(|| integrity("knowledge_state_last_receipt_missing"))?;
            let receipt = self.authenticate_action_receipt(parent, receipt_reference)?;
            let expected = self.advance(parent, &receipt)?;
            if expected.state != *child {
                return Err(integrity("knowledge_state_not_reducer_output"));
            }
        }
        Ok(state)
    }

    fn read_state_reference(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<KnowledgeState> {
        let bytes = reference.read_verified_bounded(&self.private_root, MAX_STATE_BYTES)?;
        let dto: UntrustedKnowledgeStateDto = serde_json::from_slice(&bytes)?;
        let state = KnowledgeState::from(dto);
        if reference.path != self.semantic_path("states", state.manifest_digest.as_str()) {
            return Err(integrity("knowledge_state_content_address_mismatch"));
        }
        self.validate_state(&state)?;
        Ok(state)
    }

    pub fn persist_transition(
        &self,
        transition: &TransitionReceipt,
    ) -> BrainResult<PrivateFileReference> {
        if transition.schema != KnowledgeEngineSchema::V1
            || transition.authority_root != self.authority_root
            || transition.manifest_digest != transition.calculate_digest()?
        {
            return Err(integrity("knowledge_transition_invalid"));
        }
        let from_bytes = transition
            .from_state_reference
            .read_verified_bounded(&self.private_root, MAX_STATE_BYTES)?;
        let to_bytes = transition
            .to_state_reference
            .read_verified_bounded(&self.private_root, MAX_STATE_BYTES)?;
        let receipt_bytes = transition
            .action_receipt_reference
            .read_verified_bounded(&self.private_root, MAX_ACTION_RECEIPT_BYTES)?;
        let from = KnowledgeState::from(serde_json::from_slice::<UntrustedKnowledgeStateDto>(
            &from_bytes,
        )?);
        let to = KnowledgeState::from(serde_json::from_slice::<UntrustedKnowledgeStateDto>(
            &to_bytes,
        )?);
        let receipt = ActionReceipt::from_untrusted(serde_json::from_slice::<
            UntrustedActionReceiptDto,
        >(&receipt_bytes)?);
        if from.manifest_digest != transition.from_state
            || to.manifest_digest != transition.to_state
            || receipt.manifest_digest != transition.action_receipt
            || transition.from_state_reference.path
                != self.semantic_path("states", transition.from_state.as_str())
            || transition.to_state_reference.path
                != self.semantic_path("states", transition.to_state.as_str())
            || transition.action_receipt_reference.path
                != self.semantic_path("action_receipts", transition.action_receipt.as_str())
        {
            return Err(integrity("knowledge_transition_reference_mismatch"));
        }
        self.validate_state(&from)?;
        self.validate_state(&to)?;
        let expected = self.advance(&from, &receipt)?;
        if expected.state != to || expected.transition != *transition {
            return Err(integrity("knowledge_transition_not_reducer_output"));
        }
        let destination = self.semantic_path("transitions", transition.manifest_digest.as_str());
        let bytes = serde_json::to_vec(transition)?;
        let raw_digest = write_or_verify_immutable(&self.private_root, &destination, &bytes)?;
        Ok(PrivateFileReference::new(destination, raw_digest))
    }

    pub fn persist_advance(
        &self,
        advance: &KnowledgeAdvance,
    ) -> BrainResult<(PrivateFileReference, PrivateFileReference)> {
        let state_reference = self.persist_state(&advance.state)?;
        let transition_reference = self.persist_transition(&advance.transition)?;
        Ok((state_reference, transition_reference))
    }

    /// Install revision zero as the only possible genesis for this inquiry.
    /// Repeating the exact request is safe (including after a crash between
    /// immutable state persistence and pointer publication); a different
    /// genesis for the same inquiry is never accepted.
    pub fn establish_canonical_genesis(
        &self,
        state: &KnowledgeState,
    ) -> BrainResult<CanonicalInquiryHead> {
        if state.revision != 0 || state.parent_digest.is_some() {
            return Err(invalid("canonical_genesis_must_be_revision_zero"));
        }
        let state_reference = self.persist_state(state)?;
        let pointer = self.canonical_head_pointer_path(&state.inquiry_id);
        let lock = self.canonical_head_lock_path(&state.inquiry_id);
        with_private_authority_lock(&self.private_root, &lock, || {
            match self.read_canonical_head_if_present(&pointer)? {
                Some(existing) => {
                    self.authenticate_canonical_head(&existing)?;
                    if existing.genesis_state == *state.digest()
                        && existing.current_state == *state.digest()
                        && existing.revision == 0
                    {
                        return Ok(existing);
                    }
                    Err(integrity("canonical_inquiry_genesis_conflict"))
                }
                None => {
                    let mut head = CanonicalInquiryHead {
                        schema: KnowledgeEngineSchema::V1,
                        authority_root: self.authority_root.clone(),
                        inquiry_id: state.inquiry_id.clone(),
                        genesis_state: state.digest().clone(),
                        genesis_reference: state_reference.clone(),
                        current_state: state.digest().clone(),
                        current_reference: state_reference,
                        revision: 0,
                        action_receipts: BTreeMap::new(),
                        manifest_digest: CanonicalInquiryHeadDigest::computed(
                            CANONICAL_HEAD_DOMAIN,
                            b"unsealed",
                        ),
                    };
                    head.manifest_digest = head.calculate_digest()?;
                    self.publish_canonical_head(&pointer, &head)?;
                    Ok(head)
                }
            }
        })
    }

    /// Compare-and-swap a canonical head after persisting all immutable
    /// objects.  The head is the sole mutable pointer; everything it names is
    /// content-authenticated and reducer-validated.  Therefore interruption at
    /// any point is recoverable by retrying the same advance.
    pub fn commit_canonical_advance(
        &self,
        expected: &CanonicalInquiryHead,
        advance: &KnowledgeAdvance,
    ) -> BrainResult<CanonicalInquiryHead> {
        self.authenticate_canonical_head(expected)?;
        if advance.transition.from_state != expected.current_state
            || advance.state.parent_digest.as_ref() != Some(&expected.current_state)
            || advance.state.inquiry_id != expected.inquiry_id
            || advance.state.revision
                != expected
                    .revision
                    .checked_add(1)
                    .ok_or_else(|| invalid("canonical_revision_overflow"))?
        {
            return Err(integrity("canonical_advance_expected_head_mismatch"));
        }
        let (next_reference, _) = self.persist_advance(advance)?;
        let action_id = advance.transition.action_id.clone();
        let receipt_digest = advance.transition.action_receipt.clone();
        let pointer = self.canonical_head_pointer_path(&expected.inquiry_id);
        let lock = self.canonical_head_lock_path(&expected.inquiry_id);
        with_private_authority_lock(&self.private_root, &lock, || {
            let actual = self
                .read_canonical_head_if_present(&pointer)?
                .ok_or_else(|| integrity("canonical_inquiry_head_missing"))?;
            self.authenticate_canonical_head(&actual)?;

            if let Some(recorded) = actual.action_receipts.get(&action_id) {
                if recorded != &receipt_digest {
                    return Err(integrity("canonical_action_id_receipt_conflict"));
                }
                if actual.current_state == *advance.state.digest()
                    && actual.revision == advance.state.revision
                {
                    return Ok(actual);
                }
                return Err(integrity("canonical_action_replay_not_current"));
            }
            if actual.manifest_digest != expected.manifest_digest {
                return Err(integrity("canonical_head_compare_and_swap_conflict"));
            }

            let mut next = CanonicalInquiryHead {
                schema: KnowledgeEngineSchema::V1,
                authority_root: self.authority_root.clone(),
                inquiry_id: expected.inquiry_id.clone(),
                genesis_state: actual.genesis_state.clone(),
                genesis_reference: actual.genesis_reference.clone(),
                current_state: advance.state.digest().clone(),
                current_reference: next_reference.clone(),
                revision: advance.state.revision,
                action_receipts: actual.action_receipts.clone(),
                manifest_digest: CanonicalInquiryHeadDigest::computed(
                    CANONICAL_HEAD_DOMAIN,
                    b"unsealed",
                ),
            };
            next.action_receipts.insert(action_id, receipt_digest);
            next.manifest_digest = next.calculate_digest()?;
            self.publish_canonical_head(&pointer, &next)?;
            Ok(next)
        })
    }

    /// Recover the current canonical state.  If immutable objects survived but
    /// the pointer write did not, retrying `commit_canonical_advance` is the
    /// idempotent recovery protocol; this method never guesses between forks.
    pub fn load_canonical_head(&self, inquiry_id: &InquiryId) -> BrainResult<CanonicalInquiryHead> {
        let pointer = self.canonical_head_pointer_path(inquiry_id);
        let head = self
            .read_canonical_head_if_present(&pointer)?
            .ok_or_else(|| invalid("canonical_inquiry_head_missing"))?;
        self.authenticate_canonical_head(&head)?;
        Ok(head)
    }

    fn canonical_head_pointer_path(&self, inquiry_id: &InquiryId) -> PathBuf {
        self.private_root
            .join("state/knowledge_engine/canonical_heads")
            .join(format!("{}.json", inquiry_id.as_str()))
    }

    fn canonical_head_lock_path(&self, inquiry_id: &InquiryId) -> PathBuf {
        self.private_root
            .join("state/knowledge_engine/canonical_head_locks")
            .join(format!("{}.lock", inquiry_id.as_str()))
    }

    fn read_canonical_head_if_present(
        &self,
        pointer: &Path,
    ) -> BrainResult<Option<CanonicalInquiryHead>> {
        let bytes = match read_untrusted_private_file_bounded(
            &self.private_root,
            pointer,
            CANONICAL_HEAD_POINTER_MAX_BYTES,
        ) {
            Ok(bytes) => bytes,
            Err(BrainError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(None)
            }
            Err(error) => return Err(error),
        };
        Ok(Some(serde_json::from_slice(&bytes)?))
    }

    fn publish_canonical_head(
        &self,
        pointer: &Path,
        head: &CanonicalInquiryHead,
    ) -> BrainResult<()> {
        self.authenticate_canonical_head(head)?;
        replace_private_file_atomic(
            &self.private_root,
            pointer,
            &serde_json::to_vec(head)?,
            None,
        )?;
        Ok(())
    }

    fn authenticate_canonical_head(&self, head: &CanonicalInquiryHead) -> BrainResult<()> {
        if head.schema != KnowledgeEngineSchema::V1
            || head.authority_root != self.authority_root
            || head.manifest_digest != head.calculate_digest()?
            || head.revision > HARD_MAX_REVISION
        {
            return Err(integrity("canonical_inquiry_head_not_authentic"));
        }
        let genesis = self.authenticate_state(&head.genesis_reference)?;
        let current = self.authenticate_state(&head.current_reference)?;
        if genesis.inquiry_id != head.inquiry_id
            || current.inquiry_id != head.inquiry_id
            || genesis.revision != 0
            || genesis.digest() != &head.genesis_state
            || current.digest() != &head.current_state
            || current.revision != head.revision
        {
            return Err(integrity("canonical_inquiry_head_state_mismatch"));
        }
        let lineage_actions = self.canonical_action_receipts_from_lineage(&current)?;
        if head.action_receipts != lineage_actions
            || head.action_receipts.len() as u64 != head.revision
        {
            return Err(integrity("canonical_inquiry_head_action_count_invalid"));
        }
        Ok(())
    }

    fn canonical_action_receipts_from_lineage(
        &self,
        current: &KnowledgeState,
    ) -> BrainResult<BTreeMap<KnowledgeActionId, ActionReceiptDigest>> {
        let mut cursor = current.clone();
        let mut actions = BTreeMap::new();
        while let Some(parent_reference) = cursor.parent_reference.as_ref() {
            let parent = self.read_state_reference(parent_reference)?;
            let receipt_reference = cursor
                .last_action_receipt_reference
                .as_ref()
                .ok_or_else(|| integrity("canonical_inquiry_head_last_receipt_missing"))?;
            let receipt = self.authenticate_action_receipt(&parent, receipt_reference)?;
            if receipt.parent_state != *parent.digest()
                || actions
                    .insert(receipt.action_id.clone(), receipt.manifest_digest.clone())
                    .is_some()
            {
                return Err(integrity("canonical_inquiry_head_lineage_action_invalid"));
            }
            cursor = parent;
        }
        if cursor.revision != 0 || actions.len() as u64 != current.revision {
            return Err(integrity("canonical_inquiry_head_lineage_invalid"));
        }
        Ok(actions)
    }

    fn state_reference(&self, state: &KnowledgeState) -> BrainResult<PrivateFileReference> {
        let bytes = serde_json::to_vec(state)?;
        Ok(PrivateFileReference::new(
            self.semantic_path("states", state.manifest_digest.as_str()),
            Sha256Digest::digest_bytes(&bytes),
        ))
    }

    fn action_receipt_reference(
        &self,
        receipt: &ActionReceipt,
    ) -> BrainResult<PrivateFileReference> {
        let bytes = serde_json::to_vec(receipt)?;
        Ok(PrivateFileReference::new(
            self.semantic_path("action_receipts", receipt.manifest_digest.as_str()),
            Sha256Digest::digest_bytes(&bytes),
        ))
    }
}

fn required_evidence_kinds(
    invocation: &CognitiveInvocation,
    outcome: &ActionOutcome,
) -> BTreeSet<EvidenceKind> {
    match invocation {
        CognitiveInvocation::ObserveTrace(_) => BTreeSet::from([EvidenceKind::RuntimeTrace]),
        CognitiveInvocation::ExecuteScenario(_) => BTreeSet::from([EvidenceKind::ScenarioResult]),
        CognitiveInvocation::CausalIntervention(_) => {
            BTreeSet::from([EvidenceKind::CausalComparison])
        }
        CognitiveInvocation::SearchCounterexample(_) => {
            BTreeSet::from([EvidenceKind::CounterexampleSearch])
        }
        CognitiveInvocation::VerifyFormalProperty(_)
            if matches!(outcome, ActionOutcome::Completed { .. }) =>
        {
            BTreeSet::from([EvidenceKind::FormalCertificate])
        }
        CognitiveInvocation::VerifyFormalProperty(_) => {
            BTreeSet::from([EvidenceKind::FormalSearchLog])
        }
        CognitiveInvocation::ExpandDependencyClosure(_) => {
            BTreeSet::from([EvidenceKind::DependencyGraph])
        }
    }
}

fn derive_claim_assessment(obligations: &[&KnowledgeObligation]) -> ClaimAssessment {
    let mut all_witnesses = BTreeSet::new();
    let mut contradiction_witnesses = BTreeSet::new();
    let mut all_satisfied = true;
    for obligation in obligations {
        all_witnesses.extend(obligation.state.witnesses().iter().cloned());
        match &obligation.state {
            KnowledgeObligationState::Satisfied { .. } => {}
            KnowledgeObligationState::Blocked {
                witnesses,
                reason: KnowledgeBlockReason::ContradictoryObservation { .. },
                ..
            } => {
                all_satisfied = false;
                contradiction_witnesses.extend(witnesses.iter().cloned());
            }
            KnowledgeObligationState::Open { .. } | KnowledgeObligationState::Blocked { .. } => {
                all_satisfied = false
            }
        }
    }
    if !contradiction_witnesses.is_empty() {
        ClaimAssessment::Contradicted {
            witnesses: contradiction_witnesses,
        }
    } else if all_satisfied {
        ClaimAssessment::Established {
            witnesses: all_witnesses,
        }
    } else {
        ClaimAssessment::Unresolved
    }
}

fn recompute_claim_assessments(
    claims: &mut BTreeMap<KnowledgeClaimId, KnowledgeClaim>,
    obligations: &BTreeMap<KnowledgeObligationId, KnowledgeObligation>,
) -> BrainResult<()> {
    let claim_ids: Vec<KnowledgeClaimId> = claims.keys().cloned().collect();
    for claim_id in claim_ids {
        let related: Vec<&KnowledgeObligation> = obligations
            .values()
            .filter(|obligation| obligation.claim_id == claim_id)
            .collect();
        if related.is_empty() {
            return Err(integrity("knowledge_claim_without_obligation"));
        }
        claims
            .get_mut(&claim_id)
            .ok_or_else(|| integrity("knowledge_claim_missing_during_reduction"))?
            .assessment = derive_claim_assessment(&related);
    }
    Ok(())
}

fn hypothesis_witnesses(assessment: &HypothesisAssessment) -> Vec<&ActionReceiptDigest> {
    match assessment {
        HypothesisAssessment::Viable => Vec::new(),
        HypothesisAssessment::Supported { witnesses }
        | HypothesisAssessment::Refuted { witnesses } => witnesses.iter().collect(),
        HypothesisAssessment::Conflicted {
            supporting,
            refuting,
        } => supporting.iter().chain(refuting.iter()).collect(),
    }
}

fn reduce_hypothesis_assessment(
    assessment: &HypothesisAssessment,
    evaluation: PredicateEvaluation,
    receipt: &ActionReceiptDigest,
) -> BrainResult<HypothesisAssessment> {
    if matches!(
        evaluation,
        PredicateEvaluation::Missing | PredicateEvaluation::TypeMismatch
    ) {
        return Err(integrity(
            "hypothesis_prediction_observation_missing_or_type_mismatch",
        ));
    }
    let result = match (assessment, evaluation) {
        (HypothesisAssessment::Viable, PredicateEvaluation::Matches) => {
            HypothesisAssessment::Supported {
                witnesses: BTreeSet::from([receipt.clone()]),
            }
        }
        (HypothesisAssessment::Viable, PredicateEvaluation::Contradicts) => {
            HypothesisAssessment::Refuted {
                witnesses: BTreeSet::from([receipt.clone()]),
            }
        }
        (HypothesisAssessment::Supported { witnesses }, PredicateEvaluation::Matches) => {
            let mut witnesses = witnesses.clone();
            witnesses.insert(receipt.clone());
            HypothesisAssessment::Supported { witnesses }
        }
        (HypothesisAssessment::Refuted { witnesses }, PredicateEvaluation::Contradicts) => {
            let mut witnesses = witnesses.clone();
            witnesses.insert(receipt.clone());
            HypothesisAssessment::Refuted { witnesses }
        }
        (HypothesisAssessment::Supported { witnesses }, PredicateEvaluation::Contradicts) => {
            HypothesisAssessment::Conflicted {
                supporting: witnesses.clone(),
                refuting: BTreeSet::from([receipt.clone()]),
            }
        }
        (HypothesisAssessment::Refuted { witnesses }, PredicateEvaluation::Matches) => {
            HypothesisAssessment::Conflicted {
                supporting: BTreeSet::from([receipt.clone()]),
                refuting: witnesses.clone(),
            }
        }
        (
            HypothesisAssessment::Conflicted {
                supporting,
                refuting,
            },
            evaluation,
        ) => {
            let mut supporting = supporting.clone();
            let mut refuting = refuting.clone();
            match evaluation {
                PredicateEvaluation::Matches => {
                    supporting.insert(receipt.clone());
                }
                PredicateEvaluation::Contradicts => {
                    refuting.insert(receipt.clone());
                }
                PredicateEvaluation::Missing | PredicateEvaluation::TypeMismatch => {
                    return Err(integrity(
                        "hypothesis_prediction_observation_missing_or_type_mismatch",
                    ));
                }
            }
            HypothesisAssessment::Conflicted {
                supporting,
                refuting,
            }
        }
        (_, PredicateEvaluation::Missing | PredicateEvaluation::TypeMismatch) => {
            return Err(integrity(
                "hypothesis_prediction_observation_missing_or_type_mismatch",
            ));
        }
    };
    Ok(result)
}

fn hypothesis_families_resolved(
    hypotheses: &BTreeMap<KnowledgeHypothesisId, KnowledgeHypothesis>,
) -> bool {
    let mut families: BTreeMap<&HypothesisFamilyId, Vec<&HypothesisAssessment>> = BTreeMap::new();
    for hypothesis in hypotheses.values() {
        families
            .entry(&hypothesis.family_id)
            .or_default()
            .push(&hypothesis.assessment);
    }
    families.values().all(|family| {
        family
            .iter()
            .filter(|assessment| matches!(assessment, HypothesisAssessment::Supported { .. }))
            .count()
            == 1
            && family
                .iter()
                .filter(|assessment| matches!(assessment, HypothesisAssessment::Refuted { .. }))
                .count()
                == family.len().saturating_sub(1)
    })
}

fn dependency_graph_reachable(
    graph: &BTreeMap<DependencyNodeId, BTreeSet<DependencyNodeId>>,
    start: &DependencyNodeId,
    target: &DependencyNodeId,
    max_nodes: usize,
) -> BrainResult<bool> {
    if start == target {
        return Ok(true);
    }
    if !graph.contains_key(start) {
        return Ok(false);
    }
    let mut pending = vec![start.clone()];
    let mut visited = BTreeSet::new();
    while let Some(node) = pending.pop() {
        if !visited.insert(node.clone()) {
            continue;
        }
        if visited.len() > max_nodes {
            return Err(integrity("dependency_graph_traversal_limit_exceeded"));
        }
        let targets = graph
            .get(&node)
            .ok_or_else(|| integrity("dependency_graph_node_missing"))?;
        for next in targets.iter().rev() {
            if next == target {
                return Ok(true);
            }
            pending.push(next.clone());
        }
    }
    Ok(false)
}

fn dependency_graph_is_acyclic(
    graph: &BTreeMap<DependencyNodeId, BTreeSet<DependencyNodeId>>,
    max_nodes: usize,
) -> BrainResult<bool> {
    if graph.len() > max_nodes {
        return Ok(false);
    }
    let mut indegree: BTreeMap<DependencyNodeId, usize> =
        graph.keys().cloned().map(|node| (node, 0)).collect();
    for targets in graph.values() {
        for target in targets {
            let degree = indegree
                .get_mut(target)
                .ok_or_else(|| integrity("dependency_graph_target_missing"))?;
            *degree = degree
                .checked_add(1)
                .ok_or_else(|| invalid("dependency_indegree_overflow"))?;
        }
    }
    let mut ready: BTreeSet<DependencyNodeId> = indegree
        .iter()
        .filter_map(|(node, degree)| (*degree == 0).then_some(node.clone()))
        .collect();
    let mut visited = 0usize;
    while let Some(node) = ready.pop_first() {
        visited = visited
            .checked_add(1)
            .ok_or_else(|| invalid("dependency_visit_overflow"))?;
        for target in &graph[&node] {
            let degree = indegree
                .get_mut(target)
                .ok_or_else(|| integrity("dependency_graph_target_missing"))?;
            *degree = degree
                .checked_sub(1)
                .ok_or_else(|| integrity("dependency_indegree_underflow"))?;
            if *degree == 0 {
                ready.insert(target.clone());
            }
        }
    }
    Ok(visited == graph.len())
}

fn dependency_edge_count(
    graph: &BTreeMap<DependencyNodeId, BTreeSet<DependencyNodeId>>,
) -> BrainResult<usize> {
    graph.values().try_fold(0usize, |total, targets| {
        total
            .checked_add(targets.len())
            .ok_or_else(|| invalid("dependency_edge_count_overflow"))
    })
}

fn record_transition_changes(
    next: &KnowledgeState,
    prior_claims: &BTreeMap<KnowledgeClaimId, ClaimAssessment>,
    prior_obligations: &BTreeMap<KnowledgeObligationId, KnowledgeObligationState>,
    prior_hypotheses: &BTreeMap<KnowledgeHypothesisId, HypothesisAssessment>,
    changes: &mut TransitionChanges,
) {
    for (obligation_id, obligation) in &next.obligations {
        if prior_obligations.get(obligation_id) == Some(&obligation.state) {
            continue;
        }
        match obligation.state {
            KnowledgeObligationState::Satisfied { .. } => {
                changes.satisfied_obligations.insert(obligation_id.clone());
            }
            KnowledgeObligationState::Blocked { .. } => {
                changes.blocked_obligations.insert(obligation_id.clone());
            }
            KnowledgeObligationState::Open { .. } => {}
        }
    }
    for (claim_id, claim) in &next.claims {
        if prior_claims.get(claim_id) == Some(&claim.assessment) {
            continue;
        }
        match claim.assessment {
            ClaimAssessment::Established { .. } => {
                changes.established_claims.insert(claim_id.clone());
            }
            ClaimAssessment::Contradicted { .. } => {
                changes.contradicted_claims.insert(claim_id.clone());
            }
            ClaimAssessment::Unresolved => {}
        }
    }
    for (hypothesis_id, hypothesis) in &next.hypotheses {
        if prior_hypotheses.get(hypothesis_id) == Some(&hypothesis.assessment) {
            continue;
        }
        match hypothesis.assessment {
            HypothesisAssessment::Supported { .. } => {
                changes.supported_hypotheses.insert(hypothesis_id.clone());
            }
            HypothesisAssessment::Refuted { .. } => {
                changes.refuted_hypotheses.insert(hypothesis_id.clone());
            }
            HypothesisAssessment::Viable | HypothesisAssessment::Conflicted { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquisition_contract::{
        AcquisitionBudget, AcquisitionRequest, AcquisitionScope, NoisePolicy, RequestedResidency,
    };
    use crate::capability_bundle::CapabilityBundle;
    use crate::identity::{AcquisitionId, CapabilityId};
    use crate::security::{secure_dir, secure_file};
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
        engine: KnowledgeEngine,
        bundle: PrivateFileReference,
        executor: ExecutorIdentityDraft,
        evidence: VecDeque<PrivateFileReference>,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn fixture() -> Fixture {
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "tidex-knowledge-engine-{}-{sequence}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let donor = root.with_extension("donor");
        let _ = fs::remove_dir_all(&donor);
        fs::create_dir_all(donor.join("src")).unwrap();
        fs::create_dir_all(&root).unwrap();
        secure_dir(&root).unwrap();
        fs::write(
            donor.join("src/capability.rs"),
            b"pub fn capability() -> bool { true }\n",
        )
        .unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse(format!("knowledge-acquisition-{sequence}.v1")).unwrap(),
            AcquisitionScope::WholeProject,
            RequestedResidency::BestVerified,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 32,
                max_total_bytes: 1 << 20,
            },
            vec![],
        )
        .unwrap();
        let capture = crate::content_vault::capture_to_vault(&donor, &root, &request).unwrap();
        fs::remove_dir_all(&donor).unwrap();
        let capture_reference = capture.persist(&root).unwrap();
        let bundle = CapabilityBundle::create_unmapped(
            &root,
            CapabilityId::parse(format!("knowledge.fixture{sequence}:v1")).unwrap(),
            capture_reference,
            BTreeSet::new(),
        )
        .unwrap()
        .persist(&root)
        .unwrap();

        let implementation_path = root.join("executors/deterministic-v1.bin");
        fs::create_dir_all(implementation_path.parent().unwrap()).unwrap();
        secure_dir(implementation_path.parent().unwrap()).unwrap();
        fs::write(&implementation_path, b"deterministic cognitive adapter v1").unwrap();
        secure_file(&implementation_path).unwrap();
        let implementation = PrivateFileReference::new(
            implementation_path,
            Sha256Digest::digest_bytes(b"deterministic cognitive adapter v1"),
        );
        let executor = ExecutorIdentityDraft {
            executor_id: ExecutorId::parse("test.deterministic.executor.v1").unwrap(),
            interface_version: ExecutorInterfaceVersion::V1,
            implementation,
        };

        let evidence_dir = root.join("runtime-evidence");
        fs::create_dir_all(&evidence_dir).unwrap();
        secure_dir(&evidence_dir).unwrap();
        let mut evidence = VecDeque::new();
        for index in 0..32 {
            let bytes = format!("independent evidence record {sequence}:{index}").into_bytes();
            let path = evidence_dir.join(format!("evidence-{index}.bin"));
            fs::write(&path, &bytes).unwrap();
            secure_file(&path).unwrap();
            evidence.push_back(PrivateFileReference::new(
                path,
                Sha256Digest::digest_bytes(&bytes),
            ));
        }

        let engine = KnowledgeEngine::for_test(&root).unwrap();
        Fixture {
            root,
            engine,
            bundle,
            executor,
            evidence,
        }
    }

    #[derive(Debug, Clone)]
    enum TestOutcomeMode {
        Complete {
            dependency_edges: BTreeSet<DependencyEdge>,
        },
        UnverifiedComplete,
        CompleteWithDiscovery {
            discoveries: BTreeMap<DiscoveredConcernId, DiscoveredKnowledgeObligation>,
        },
        Bounded,
        Blocked,
    }

    struct DeterministicAdapter {
        identity: ExecutorIdentityDraft,
        evidence: RefCell<VecDeque<PrivateFileReference>>,
        mode: TestOutcomeMode,
    }

    impl sealed::Sealed for DeterministicAdapter {}

    impl CognitiveExecutorAdapter for DeterministicAdapter {
        fn identity(&self) -> ExecutorIdentityDraft {
            self.identity.clone()
        }

        fn invoke(&self, invocation: &CognitiveInvocation) -> BrainResult<AdapterExecution> {
            let artifact = self
                .evidence
                .borrow_mut()
                .pop_front()
                .ok_or_else(|| invalid("test_evidence_exhausted"))?;
            let outcome = match &self.mode {
                TestOutcomeMode::Complete { dependency_edges } => ActionOutcome::Completed {
                    observations: matching_observations(invocation),
                    dependency_edges: dependency_edges.clone(),
                    discovered_obligations: BTreeMap::new(),
                },
                TestOutcomeMode::UnverifiedComplete => ActionOutcome::Completed {
                    observations: matching_observations(invocation),
                    dependency_edges: BTreeSet::new(),
                    discovered_obligations: BTreeMap::new(),
                },
                TestOutcomeMode::CompleteWithDiscovery { discoveries } => {
                    ActionOutcome::Completed {
                        observations: matching_observations(invocation),
                        dependency_edges: BTreeSet::new(),
                        discovered_obligations: discoveries.clone(),
                    }
                }
                TestOutcomeMode::Bounded => ActionOutcome::BoundedNoResult {
                    reason: BoundedActionReason::SearchSpaceExhausted,
                },
                TestOutcomeMode::Blocked => ActionOutcome::Blocked {
                    reason: ExecutionBlockReason::SafetyPolicyDenied,
                },
            };
            let evidence_kind = *required_evidence_kinds(invocation, &outcome)
                .iter()
                .next()
                .ok_or_else(|| invalid("test_evidence_kind_missing"))?;
            let evidence = vec![EvidenceArtifactInput {
                kind: evidence_kind,
                artifact,
            }];
            if matches!(self.mode, TestOutcomeMode::UnverifiedComplete) {
                Ok(AdapterExecution::new(outcome, evidence))
            } else {
                Ok(AdapterExecution::verified_for_test(outcome, evidence))
            }
        }
    }

    /// Exercises the same semantic-verifier path as a production adapter: no
    /// module-only test oracle is attached to its output.
    struct ProductionEvidenceAdapter {
        identity: ExecutorIdentityDraft,
        expected_invocation: CognitiveInvocationDigest,
        execution: AdapterExecution,
    }

    impl sealed::Sealed for ProductionEvidenceAdapter {}

    impl CognitiveExecutorAdapter for ProductionEvidenceAdapter {
        fn identity(&self) -> ExecutorIdentityDraft {
            self.identity.clone()
        }

        fn invoke(&self, invocation: &CognitiveInvocation) -> BrainResult<AdapterExecution> {
            if invocation.digest()? != self.expected_invocation {
                return Err(integrity("test_adapter_invocation_mismatch"));
            }
            Ok(self.execution.clone())
        }
    }

    fn planned_invocation(engine: &KnowledgeEngine, state: &KnowledgeState) -> CognitiveInvocation {
        match engine.plan(state).unwrap() {
            PlanningDecision::Invoke { invocation } => *invocation,
            PlanningDecision::Terminal { .. } => panic!("expected an invocation"),
        }
    }

    fn persist_artifact(fixture: &mut Fixture, bytes: Vec<u8>) -> PrivateFileReference {
        let reference = fixture.evidence.pop_front().unwrap();
        fs::write(&reference.path, &bytes).unwrap();
        secure_file(&reference.path).unwrap();
        PrivateFileReference::new(reference.path, Sha256Digest::digest_bytes(&bytes))
    }

    fn evidence_input(
        fixture: &mut Fixture,
        kind: EvidenceKind,
        bytes: Vec<u8>,
    ) -> EvidenceArtifactInput {
        EvidenceArtifactInput {
            kind,
            artifact: persist_artifact(fixture, bytes),
        }
    }

    fn verification_context<'a>(
        engine: &'a KnowledgeEngine,
        invocation: &'a CognitiveInvocation,
        executor: &'a ExecutorAttestation,
    ) -> EvidenceVerificationContext<'a> {
        EvidenceVerificationContext {
            private_root: &engine.private_root,
            authority_instance: &engine.authority_instance,
            authority_root: &engine.authority_root,
            invocation,
            executor,
        }
    }

    fn matching_value(predicate: &KnowledgePredicate) -> ObservedValue {
        match predicate {
            KnowledgePredicate::BoolEquals { expected, .. } => ObservedValue::Bool(*expected),
            KnowledgePredicate::I64Equals { expected, .. } => ObservedValue::I64(*expected),
            KnowledgePredicate::U64Equals { expected, .. } => ObservedValue::U64(*expected),
            KnowledgePredicate::U64AtLeast { minimum, .. } => ObservedValue::U64(*minimum),
            KnowledgePredicate::U64AtMost { maximum, .. } => ObservedValue::U64(*maximum),
            KnowledgePredicate::SymbolEquals { expected, .. } => {
                ObservedValue::Symbol(expected.clone())
            }
            KnowledgePredicate::ExactBytesDigestEquals { expected, .. } => {
                ObservedValue::ExactBytesDigest(expected.clone())
            }
        }
    }

    fn matching_observations(
        invocation: &CognitiveInvocation,
    ) -> BTreeMap<ObservationKey, ObservedValue> {
        let mut observations = BTreeMap::new();
        let criterion = invocation.binding().criterion.predicate();
        observations.insert(
            criterion.observation_key().clone(),
            matching_value(criterion),
        );
        if let Some(predictions) = invocation.predictions() {
            // BTree order precommits the selected surviving rival in this test adapter.
            for prediction in predictions.values() {
                observations
                    .entry(prediction.observation_key().clone())
                    .or_insert_with(|| matching_value(prediction));
            }
        }
        for key in invocation.required_observations() {
            observations.entry(key).or_insert(ObservedValue::Bool(true));
        }
        observations
    }

    fn adapter(fixture: &mut Fixture, mode: TestOutcomeMode) -> DeterministicAdapter {
        DeterministicAdapter {
            identity: fixture.executor.clone(),
            evidence: RefCell::new(std::mem::take(&mut fixture.evidence)),
            mode,
        }
    }

    fn bool_predicate(key: &str) -> KnowledgePredicate {
        KnowledgePredicate::BoolEquals {
            key: ObservationKey::parse(key).unwrap(),
            expected: true,
        }
    }

    fn trace_contract(budget: KnowledgeBudget) -> KnowledgeContractDraft {
        let predicate = bool_predicate("trace.result.valid");
        KnowledgeContractDraft {
            claims: vec![KnowledgeClaimDraft {
                claim_id: KnowledgeClaimId::parse("claim.trace.valid").unwrap(),
                domain: KnowledgeDomain::TraceBehavior,
                predicate: predicate.clone(),
            }],
            obligations: vec![KnowledgeObligationDraft {
                obligation_id: KnowledgeObligationId::parse("obligation.trace.valid").unwrap(),
                claim_id: KnowledgeClaimId::parse("claim.trace.valid").unwrap(),
                criterion: EvaluationCriterion::exact(predicate),
                plan: CognitivePlan::ObserveTrace {
                    observation_keys: BTreeSet::from([
                        ObservationKey::parse("trace.result.valid").unwrap()
                    ]),
                    max_events: 128,
                },
            }],
            hypotheses: vec![],
            budget,
        }
    }

    fn two_trace_contract(budget: KnowledgeBudget) -> KnowledgeContractDraft {
        let first = bool_predicate("trace.replay.first");
        let second = bool_predicate("trace.replay.second");
        KnowledgeContractDraft {
            claims: vec![
                KnowledgeClaimDraft {
                    claim_id: KnowledgeClaimId::parse("claim.trace.replay.a").unwrap(),
                    domain: KnowledgeDomain::TraceBehavior,
                    predicate: first.clone(),
                },
                KnowledgeClaimDraft {
                    claim_id: KnowledgeClaimId::parse("claim.trace.replay.b").unwrap(),
                    domain: KnowledgeDomain::TraceBehavior,
                    predicate: second.clone(),
                },
            ],
            obligations: vec![
                KnowledgeObligationDraft {
                    obligation_id: KnowledgeObligationId::parse("obligation.trace.replay.a")
                        .unwrap(),
                    claim_id: KnowledgeClaimId::parse("claim.trace.replay.a").unwrap(),
                    criterion: EvaluationCriterion::exact(first.clone()),
                    plan: CognitivePlan::ObserveTrace {
                        observation_keys: BTreeSet::from([first.observation_key().clone()]),
                        max_events: 8,
                    },
                },
                KnowledgeObligationDraft {
                    obligation_id: KnowledgeObligationId::parse("obligation.trace.replay.b")
                        .unwrap(),
                    claim_id: KnowledgeClaimId::parse("claim.trace.replay.b").unwrap(),
                    criterion: EvaluationCriterion::exact(second.clone()),
                    plan: CognitivePlan::ObserveTrace {
                        observation_keys: BTreeSet::from([second.observation_key().clone()]),
                        max_events: 8,
                    },
                },
            ],
            hypotheses: vec![],
            budget,
        }
    }

    fn scenario_contract(budget: KnowledgeBudget) -> KnowledgeContractDraft {
        let predicate = bool_predicate("scenario.completed");
        KnowledgeContractDraft {
            claims: vec![KnowledgeClaimDraft {
                claim_id: KnowledgeClaimId::parse("claim.scenario.completed").unwrap(),
                domain: KnowledgeDomain::ScenarioBehavior,
                predicate: predicate.clone(),
            }],
            obligations: vec![KnowledgeObligationDraft {
                obligation_id: KnowledgeObligationId::parse("obligation.scenario.completed")
                    .unwrap(),
                claim_id: KnowledgeClaimId::parse("claim.scenario.completed").unwrap(),
                criterion: EvaluationCriterion::exact(predicate.clone()),
                plan: CognitivePlan::ExecuteScenario {
                    scenario_id: ScenarioId::parse("scenario.completed.v1").unwrap(),
                    inputs: BTreeMap::from([(
                        ScenarioInputId::parse("input.case").unwrap(),
                        ObservedValue::Bool(true),
                    )]),
                    observation_keys: BTreeSet::from([predicate.observation_key().clone()]),
                },
            }],
            hypotheses: vec![],
            budget,
        }
    }

    fn causal_contract(budget: KnowledgeBudget) -> KnowledgeContractDraft {
        let criterion = bool_predicate("causal.effect.valid");
        let family_id = HypothesisFamilyId::parse("causal.verifier.family").unwrap();
        let hypothesis_a = KnowledgeHypothesisId::parse("causal.verifier.a").unwrap();
        let hypothesis_b = KnowledgeHypothesisId::parse("causal.verifier.b").unwrap();
        let prediction_key = ObservationKey::parse("causal.verifier.selection").unwrap();
        KnowledgeContractDraft {
            claims: vec![KnowledgeClaimDraft {
                claim_id: KnowledgeClaimId::parse("claim.causal.verifier").unwrap(),
                domain: KnowledgeDomain::CausalMechanism,
                predicate: criterion.clone(),
            }],
            obligations: vec![KnowledgeObligationDraft {
                obligation_id: KnowledgeObligationId::parse("obligation.causal.verifier").unwrap(),
                claim_id: KnowledgeClaimId::parse("claim.causal.verifier").unwrap(),
                criterion: EvaluationCriterion::exact(criterion.clone()),
                plan: CognitivePlan::CausalIntervention {
                    variable: InterventionVariableId::parse("causal.verifier.variable").unwrap(),
                    baseline: ObservedValue::Bool(false),
                    treatment: ObservedValue::Bool(true),
                    effect: criterion,
                    predictions: BTreeMap::from([
                        (
                            hypothesis_a.clone(),
                            KnowledgePredicate::BoolEquals {
                                key: prediction_key.clone(),
                                expected: true,
                            },
                        ),
                        (
                            hypothesis_b.clone(),
                            KnowledgePredicate::BoolEquals {
                                key: prediction_key,
                                expected: false,
                            },
                        ),
                    ]),
                },
            }],
            hypotheses: vec![
                KnowledgeHypothesisDraft {
                    hypothesis_id: hypothesis_a,
                    family_id: family_id.clone(),
                    definition: bool_predicate("causal.verifier.definition.a"),
                },
                KnowledgeHypothesisDraft {
                    hypothesis_id: hypothesis_b,
                    family_id,
                    definition: bool_predicate("causal.verifier.definition.b"),
                },
            ],
            budget,
        }
    }

    fn counterexample_contract(budget: KnowledgeBudget) -> KnowledgeContractDraft {
        let target = bool_predicate("counterexample.target.holds");
        let family_id = HypothesisFamilyId::parse("counterexample.verifier.family").unwrap();
        let hypothesis_a = KnowledgeHypothesisId::parse("counterexample.verifier.a").unwrap();
        let hypothesis_b = KnowledgeHypothesisId::parse("counterexample.verifier.b").unwrap();
        let prediction_key = ObservationKey::parse("counterexample.verifier.selection").unwrap();
        KnowledgeContractDraft {
            claims: vec![KnowledgeClaimDraft {
                claim_id: KnowledgeClaimId::parse("claim.counterexample.verifier").unwrap(),
                domain: KnowledgeDomain::AdversarialBoundary,
                predicate: target.clone(),
            }],
            obligations: vec![KnowledgeObligationDraft {
                obligation_id: KnowledgeObligationId::parse("obligation.counterexample.verifier")
                    .unwrap(),
                claim_id: KnowledgeClaimId::parse("claim.counterexample.verifier").unwrap(),
                criterion: EvaluationCriterion::exact(target.clone()),
                plan: CognitivePlan::SearchCounterexample {
                    target,
                    max_cases: 16,
                    predictions: BTreeMap::from([
                        (
                            hypothesis_a.clone(),
                            KnowledgePredicate::BoolEquals {
                                key: prediction_key.clone(),
                                expected: true,
                            },
                        ),
                        (
                            hypothesis_b.clone(),
                            KnowledgePredicate::BoolEquals {
                                key: prediction_key,
                                expected: false,
                            },
                        ),
                    ]),
                },
            }],
            hypotheses: vec![
                KnowledgeHypothesisDraft {
                    hypothesis_id: hypothesis_a,
                    family_id: family_id.clone(),
                    definition: bool_predicate("counterexample.verifier.definition.a"),
                },
                KnowledgeHypothesisDraft {
                    hypothesis_id: hypothesis_b,
                    family_id,
                    definition: bool_predicate("counterexample.verifier.definition.b"),
                },
            ],
            budget,
        }
    }

    fn formal_contract(budget: KnowledgeBudget) -> KnowledgeContractDraft {
        let property = bool_predicate("formal.property.proven");
        KnowledgeContractDraft {
            claims: vec![KnowledgeClaimDraft {
                claim_id: KnowledgeClaimId::parse("claim.formal.verifier").unwrap(),
                domain: KnowledgeDomain::FormalProperty,
                predicate: property.clone(),
            }],
            obligations: vec![KnowledgeObligationDraft {
                obligation_id: KnowledgeObligationId::parse("obligation.formal.verifier").unwrap(),
                claim_id: KnowledgeClaimId::parse("claim.formal.verifier").unwrap(),
                criterion: EvaluationCriterion::exact(property.clone()),
                plan: CognitivePlan::VerifyFormalProperty {
                    proof_system: ProofSystem::Smt,
                    property,
                },
            }],
            hypotheses: vec![],
            budget,
        }
    }

    fn dependency_contract(root: &str) -> KnowledgeContractDraft {
        let root = DependencyNodeId::parse(root).unwrap();
        let predicate = dependency_closure_predicate(&root).unwrap();
        KnowledgeContractDraft {
            claims: vec![KnowledgeClaimDraft {
                claim_id: KnowledgeClaimId::parse("claim.dependency.root").unwrap(),
                domain: KnowledgeDomain::DependencyClosure,
                predicate: predicate.clone(),
            }],
            obligations: vec![KnowledgeObligationDraft {
                obligation_id: KnowledgeObligationId::parse("obligation.dependency.root").unwrap(),
                claim_id: KnowledgeClaimId::parse("claim.dependency.root").unwrap(),
                criterion: EvaluationCriterion::exact(predicate),
                plan: CognitivePlan::ExpandDependencyClosure {
                    root,
                    depth: 0,
                    max_depth: 8,
                },
            }],
            hypotheses: vec![],
            budget: KnowledgeBudget::conservative_default(),
        }
    }

    fn initialize(
        fixture: &Fixture,
        inquiry: &str,
        contract: KnowledgeContractDraft,
    ) -> KnowledgeState {
        fixture
            .engine
            .initialize(
                InquiryId::parse(inquiry).unwrap(),
                fixture.bundle.clone(),
                contract,
            )
            .unwrap()
    }

    fn advance_once(fixture: &mut Fixture, state: &KnowledgeState) -> KnowledgeAdvance {
        let adapter = DeterministicAdapter {
            identity: fixture.executor.clone(),
            evidence: RefCell::new(VecDeque::from([fixture.evidence.pop_front().unwrap()])),
            mode: TestOutcomeMode::Complete {
                dependency_edges: BTreeSet::new(),
            },
        };
        let receipt = fixture.engine.execute(state, &adapter).unwrap();
        fixture.engine.advance(state, &receipt).unwrap()
    }

    #[test]
    fn canonical_head_is_idempotent_and_recovers_after_immutable_persistence() {
        let mut fixture = fixture();
        let state = initialize(
            &fixture,
            "canonical.recovery.inquiry.v1",
            trace_contract(KnowledgeBudget::conservative_default()),
        );
        let genesis = fixture.engine.establish_canonical_genesis(&state).unwrap();
        let advance = advance_once(&mut fixture, &state);

        // `persist_advance` models a process that reached durable immutable
        // objects and then died before publishing the mutable head pointer.
        fixture.engine.persist_advance(&advance).unwrap();
        assert_eq!(
            fixture
                .engine
                .load_canonical_head(genesis.inquiry_id())
                .unwrap(),
            genesis
        );
        let committed = fixture
            .engine
            .commit_canonical_advance(&genesis, &advance)
            .unwrap();
        let replay = fixture
            .engine
            .commit_canonical_advance(&genesis, &advance)
            .unwrap();
        assert_eq!(committed, replay);
        assert_eq!(committed.revision(), 1);
        assert_eq!(
            fixture
                .engine
                .load_canonical_head(genesis.inquiry_id())
                .unwrap(),
            committed
        );

        // A digest alone is not a signature.  Recompute it after substituting
        // a different action id to prove that the head is checked against the
        // authenticated state lineage, not merely against self-consistent JSON.
        let mut relabeled = committed.clone();
        let (_, receipt) = relabeled.action_receipts.pop_first().unwrap();
        relabeled.action_receipts.insert(
            KnowledgeActionId::computed(ACTION_ID_DOMAIN, b"forged-action-label"),
            receipt,
        );
        relabeled.manifest_digest = relabeled.calculate_digest().unwrap();
        let pointer = fixture
            .engine
            .canonical_head_pointer_path(genesis.inquiry_id());
        fs::write(&pointer, serde_json::to_vec(&relabeled).unwrap()).unwrap();
        secure_file(&pointer).unwrap();
        assert!(fixture
            .engine
            .load_canonical_head(genesis.inquiry_id())
            .is_err());
    }

    #[test]
    fn canonical_head_rejects_double_advance_concurrently() {
        let mut fixture = fixture();
        let state = initialize(
            &fixture,
            "canonical.concurrent.inquiry.v1",
            trace_contract(KnowledgeBudget::conservative_default()),
        );
        let genesis = fixture.engine.establish_canonical_genesis(&state).unwrap();
        let first = advance_once(&mut fixture, &state);
        let second = advance_once(&mut fixture, &state);
        assert_ne!(
            first.transition.action_receipt, second.transition.action_receipt,
            "independent executions must not be silently conflated"
        );

        let barrier = Arc::new(Barrier::new(2));
        let engine_a = fixture.engine.clone();
        let engine_b = fixture.engine.clone();
        let head_a = genesis.clone();
        let head_b = genesis.clone();
        let barrier_a = barrier.clone();
        let left = std::thread::spawn(move || {
            barrier_a.wait();
            engine_a.commit_canonical_advance(&head_a, &first)
        });
        let right = std::thread::spawn(move || {
            barrier.wait();
            engine_b.commit_canonical_advance(&head_b, &second)
        });
        let results = [left.join().unwrap(), right.join().unwrap()];
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert!(results.iter().any(|result| {
            result.as_ref().err().is_some_and(|error| {
                error
                    .to_string()
                    .contains("canonical_action_id_receipt_conflict")
            })
        }));
    }

    #[test]
    fn canonical_head_tampering_and_second_genesis_fail_closed() {
        let fixture = fixture();
        let state = initialize(
            &fixture,
            "canonical.tamper.inquiry.v1",
            trace_contract(KnowledgeBudget::conservative_default()),
        );
        let head = fixture.engine.establish_canonical_genesis(&state).unwrap();
        assert_eq!(
            fixture.engine.establish_canonical_genesis(&state).unwrap(),
            head
        );

        let conflicting = initialize(
            &fixture,
            "canonical.tamper.inquiry.v1",
            scenario_contract(KnowledgeBudget::conservative_default()),
        );
        assert!(fixture
            .engine
            .establish_canonical_genesis(&conflicting)
            .is_err());

        let mut tampered = head.clone();
        tampered.revision = 1;
        let pointer = fixture
            .engine
            .canonical_head_pointer_path(head.inquiry_id());
        fs::write(&pointer, serde_json::to_vec(&tampered).unwrap()).unwrap();
        secure_file(&pointer).unwrap();
        assert!(fixture
            .engine
            .load_canonical_head(head.inquiry_id())
            .is_err());
    }

    #[test]
    fn capability_specific_subset_contract_executes_and_reaches_scoped_complete() {
        let mut fixture = fixture();
        let state = initialize(
            &fixture,
            "subset.inquiry.v1",
            trace_contract(KnowledgeBudget::conservative_default()),
        );
        assert_eq!(state.claims().len(), 1);
        assert_eq!(state.obligations().len(), 1);
        assert!(matches!(
            fixture.engine.plan(&state).unwrap(),
            PlanningDecision::Invoke { .. }
        ));

        fixture.engine.persist_state(&state).unwrap();
        let adapter = adapter(
            &mut fixture,
            TestOutcomeMode::Complete {
                dependency_edges: BTreeSet::new(),
            },
        );
        let receipt = fixture.engine.execute(&state, &adapter).unwrap();
        let advance = fixture.engine.advance(&state, &receipt).unwrap();
        assert_eq!(advance.state().revision(), 1);
        assert_eq!(advance.state().parent_digest(), Some(state.digest()));
        assert!(matches!(
            fixture.engine.plan(advance.state()).unwrap(),
            PlanningDecision::Terminal {
                terminal: KnowledgeTerminal::ScopedComplete { .. }
            }
        ));
        fixture.engine.persist_advance(&advance).unwrap();
    }

    #[test]
    fn discovered_dependency_adds_an_obligation_and_prevents_stale_completion() {
        let mut fixture = fixture();
        let state = initialize(
            &fixture,
            "dependency.reopen.inquiry.v1",
            dependency_contract("component.root"),
        );
        fixture.engine.persist_state(&state).unwrap();
        let edge = DependencyEdge::new(
            DependencyNodeId::parse("component.root").unwrap(),
            DependencyNodeId::parse("component.child").unwrap(),
        );
        let first_adapter = adapter(
            &mut fixture,
            TestOutcomeMode::Complete {
                dependency_edges: BTreeSet::from([edge]),
            },
        );
        let receipt = fixture.engine.execute(&state, &first_adapter).unwrap();
        let first = fixture.engine.advance(&state, &receipt).unwrap();
        assert_eq!(first.state().obligations().len(), 2);
        assert_eq!(first.transition().changes().added_obligations.len(), 1);
        assert!(matches!(
            fixture.engine.plan(first.state()).unwrap(),
            PlanningDecision::Invoke { .. }
        ));
        fixture.engine.persist_advance(&first).unwrap();

        let mut remaining_evidence = first_adapter.evidence.into_inner();
        fixture.evidence.append(&mut remaining_evidence);
        let closing_adapter = adapter(
            &mut fixture,
            TestOutcomeMode::Complete {
                dependency_edges: BTreeSet::new(),
            },
        );
        let child_receipt = fixture
            .engine
            .execute(first.state(), &closing_adapter)
            .unwrap();
        let second = fixture
            .engine
            .advance(first.state(), &child_receipt)
            .unwrap();
        assert!(matches!(
            fixture.engine.plan(second.state()).unwrap(),
            PlanningDecision::Terminal {
                terminal: KnowledgeTerminal::ScopedComplete { .. }
            }
        ));
    }

    #[test]
    fn arbitrary_risk_discovery_is_rejected_without_derivation_policy() {
        let mut fixture = fixture();
        let state = initialize(
            &fixture,
            "risk.reopen.inquiry.v1",
            trace_contract(KnowledgeBudget::conservative_default()),
        );
        fixture.engine.persist_state(&state).unwrap();
        let risk_predicate = bool_predicate("risk.bound.known");
        let discovery = DiscoveredKnowledgeObligation {
            kind: DiscoveredConcernKind::Risk,
            domain: KnowledgeDomain::ScenarioBehavior,
            claim_predicate: risk_predicate.clone(),
            criterion: EvaluationCriterion::exact(risk_predicate),
            plan: CognitivePlan::ExecuteScenario {
                scenario_id: ScenarioId::parse("risk.boundary.scenario").unwrap(),
                inputs: BTreeMap::new(),
                observation_keys: BTreeSet::from([
                    ObservationKey::parse("risk.bound.known").unwrap()
                ]),
            },
        };
        let adapter = adapter(
            &mut fixture,
            TestOutcomeMode::CompleteWithDiscovery {
                discoveries: BTreeMap::from([(
                    DiscoveredConcernId::parse("new.risk.boundary").unwrap(),
                    discovery,
                )]),
            },
        );
        let error = fixture.engine.execute(&state, &adapter).unwrap_err();
        assert!(error
            .to_string()
            .contains("discovery_requires_authenticated_derivation_policy"));
    }

    #[test]
    fn unverified_completed_outcome_is_fail_closed() {
        let mut fixture = fixture();
        let state = initialize(
            &fixture,
            "unverified.outcome.inquiry.v1",
            trace_contract(KnowledgeBudget::conservative_default()),
        );
        fixture.engine.persist_state(&state).unwrap();
        let adapter = adapter(&mut fixture, TestOutcomeMode::UnverifiedComplete);
        let error = fixture.engine.execute(&state, &adapter).unwrap_err();
        assert!(error
            .to_string()
            .contains("trace_evidence_encoding_invalid"));
    }

    #[test]
    fn scenario_verifier_derives_only_exact_root_bound_outcome() {
        let mut fixture = fixture();
        let state = initialize(
            &fixture,
            "scenario.verifier.inquiry.v1",
            scenario_contract(KnowledgeBudget::conservative_default()),
        );
        let invocation = match fixture.engine.plan(&state).unwrap() {
            PlanningDecision::Invoke { invocation } => *invocation,
            PlanningDecision::Terminal { .. } => panic!("expected scenario invocation"),
        };
        let executor = fixture
            .engine
            .authenticate_executor(fixture.executor.clone())
            .unwrap();
        let observations = matching_observations(&invocation);
        let artifact = ScenarioEvidenceArtifact::from_invocation(
            &invocation,
            &fixture.executor,
            observations.clone(),
        )
        .unwrap();
        let artifact_reference = fixture.evidence.pop_front().unwrap();
        let bytes = artifact.canonical_bytes().unwrap();
        fs::write(&artifact_reference.path, &bytes).unwrap();
        secure_file(&artifact_reference.path).unwrap();
        let artifact_reference =
            PrivateFileReference::new(artifact_reference.path, Sha256Digest::digest_bytes(&bytes));
        let proposed = ActionOutcome::Completed {
            observations,
            dependency_edges: BTreeSet::new(),
            discovered_obligations: BTreeMap::new(),
        };
        let evidence = vec![EvidenceArtifactInput {
            kind: EvidenceKind::ScenarioResult,
            artifact: artifact_reference,
        }];
        let derived = fixture
            .engine
            .evidence_verifiers
            .derive_completed(
                &verification_context(&fixture.engine, &invocation, &executor),
                &proposed,
                &evidence,
            )
            .unwrap();
        assert_eq!(derived, proposed);

        let foreign_engine = KnowledgeEngine::for_test(&fixture.root).unwrap();
        let foreign_root = AuthorityRootDigest::computed(ROOT_DOMAIN, b"foreign-root");
        let foreign_context = EvidenceVerificationContext {
            private_root: &foreign_engine.private_root,
            authority_instance: &foreign_engine.authority_instance,
            authority_root: &foreign_root,
            invocation: &invocation,
            executor: &executor,
        };
        assert!(foreign_engine
            .evidence_verifiers
            .derive_completed(&foreign_context, &proposed, &evidence)
            .is_err());
    }

    #[test]
    fn trace_verifier_derives_events_and_rejects_tamper_wrong_kind_and_cross_authority() {
        let mut fixture = fixture();
        let state = initialize(
            &fixture,
            "trace.production.verifier.inquiry.v1",
            trace_contract(KnowledgeBudget::conservative_default()),
        );
        fixture.engine.persist_state(&state).unwrap();
        let invocation = planned_invocation(&fixture.engine, &state);
        let executor = fixture
            .engine
            .authenticate_executor(fixture.executor.clone())
            .unwrap();
        let observations = matching_observations(&invocation);
        let criterion_key = invocation
            .binding()
            .criterion
            .predicate()
            .observation_key()
            .clone();
        let initial_event = TraceEvidenceEvent::new(
            0,
            BTreeMap::from([(criterion_key.clone(), ObservedValue::Bool(false))]),
        )
        .unwrap();
        let final_event = TraceEvidenceEvent::new(1, observations.clone()).unwrap();
        let artifact = TraceEvidenceArtifact::from_invocation(
            &invocation,
            &fixture.executor,
            vec![initial_event, final_event],
        )
        .unwrap();
        let evidence = EvidenceArtifactInput {
            kind: EvidenceKind::RuntimeTrace,
            artifact: persist_artifact(&mut fixture, artifact.canonical_bytes().unwrap()),
        };
        let proposed = canonical_completed(observations.clone());
        let derived = fixture
            .engine
            .evidence_verifiers
            .derive_verified_outcome(
                &verification_context(&fixture.engine, &invocation, &executor),
                &proposed,
                std::slice::from_ref(&evidence),
            )
            .unwrap();
        assert_eq!(derived, proposed);

        let mut forged_outcome_observations = observations.clone();
        forged_outcome_observations.insert(criterion_key, ObservedValue::Bool(false));
        let forged_outcome = canonical_completed(forged_outcome_observations);
        assert!(fixture
            .engine
            .evidence_verifiers
            .derive_verified_outcome(
                &verification_context(&fixture.engine, &invocation, &executor),
                &forged_outcome,
                std::slice::from_ref(&evidence),
            )
            .unwrap_err()
            .to_string()
            .contains("executor_outcome_not_canonically_derived"));

        let wrong_kind = EvidenceArtifactInput {
            kind: EvidenceKind::CausalComparison,
            artifact: evidence.artifact.clone(),
        };
        assert!(fixture
            .engine
            .evidence_verifiers
            .derive_verified_outcome(
                &verification_context(&fixture.engine, &invocation, &executor),
                &proposed,
                &[wrong_kind],
            )
            .is_err());

        let mut tampered_artifact = artifact.clone();
        tampered_artifact.max_events += 1;
        let tampered_evidence = evidence_input(
            &mut fixture,
            EvidenceKind::RuntimeTrace,
            tampered_artifact.canonical_bytes().unwrap(),
        );
        assert!(fixture
            .engine
            .evidence_verifiers
            .derive_verified_outcome(
                &verification_context(&fixture.engine, &invocation, &executor),
                &proposed,
                &[tampered_evidence],
            )
            .is_err());

        let foreign_engine = KnowledgeEngine::from_verified_root(
            fixture.root.clone(),
            AuthorityInstanceId::parse("foreign-trace-authority.v1").unwrap(),
        )
        .unwrap();
        let foreign_executor = foreign_engine
            .authenticate_executor(fixture.executor.clone())
            .unwrap();
        assert!(foreign_engine
            .evidence_verifiers
            .derive_verified_outcome(
                &verification_context(&foreign_engine, &invocation, &foreign_executor),
                &proposed,
                std::slice::from_ref(&evidence),
            )
            .is_err());

        let adapter = ProductionEvidenceAdapter {
            identity: fixture.executor.clone(),
            expected_invocation: invocation.digest().unwrap(),
            execution: AdapterExecution::new(proposed, vec![evidence]),
        };
        let receipt = fixture.engine.execute(&state, &adapter).unwrap();
        let advanced = fixture.engine.advance(&state, &receipt).unwrap();
        assert!(matches!(
            fixture.engine.plan(advanced.state()).unwrap(),
            PlanningDecision::Terminal {
                terminal: KnowledgeTerminal::ScopedComplete { .. }
            }
        ));
        assert!(fixture.engine.advance(advanced.state(), &receipt).is_err());
    }

    #[test]
    fn typed_evidence_artifact_cannot_be_replayed_for_a_later_action() {
        let mut fixture = fixture();
        let state = initialize(
            &fixture,
            "trace.artifact.replay.inquiry.v1",
            two_trace_contract(KnowledgeBudget::conservative_default()),
        );
        fixture.engine.persist_state(&state).unwrap();
        let first_invocation = planned_invocation(&fixture.engine, &state);
        let first_observations = matching_observations(&first_invocation);
        let first_artifact = TraceEvidenceArtifact::from_invocation(
            &first_invocation,
            &fixture.executor,
            vec![TraceEvidenceEvent::new(0, first_observations.clone()).unwrap()],
        )
        .unwrap();
        let evidence = evidence_input(
            &mut fixture,
            EvidenceKind::RuntimeTrace,
            first_artifact.canonical_bytes().unwrap(),
        );
        let first_adapter = ProductionEvidenceAdapter {
            identity: fixture.executor.clone(),
            expected_invocation: first_invocation.digest().unwrap(),
            execution: AdapterExecution::new(
                canonical_completed(first_observations),
                vec![evidence.clone()],
            ),
        };
        let first_receipt = fixture.engine.execute(&state, &first_adapter).unwrap();
        let advanced = fixture.engine.advance(&state, &first_receipt).unwrap();
        let second_invocation = planned_invocation(&fixture.engine, advanced.state());
        let replay_adapter = ProductionEvidenceAdapter {
            identity: fixture.executor.clone(),
            expected_invocation: second_invocation.digest().unwrap(),
            execution: AdapterExecution::new(
                canonical_completed(matching_observations(&second_invocation)),
                vec![evidence],
            ),
        };
        let error = fixture
            .engine
            .execute(advanced.state(), &replay_adapter)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("evidence_authority_binding_mismatch"));
    }

    #[test]
    fn causal_verifier_derives_treatment_arm_and_rejects_protocol_input_tamper() {
        let mut fixture = fixture();
        let state = initialize(
            &fixture,
            "causal.production.verifier.inquiry.v1",
            causal_contract(KnowledgeBudget::conservative_default()),
        );
        fixture.engine.persist_state(&state).unwrap();
        let invocation = planned_invocation(&fixture.engine, &state);
        let executor = fixture
            .engine
            .authenticate_executor(fixture.executor.clone())
            .unwrap();
        let treatment_observations = matching_observations(&invocation);
        let mut baseline_observations = treatment_observations.clone();
        let effect_key = invocation
            .binding()
            .criterion
            .predicate()
            .observation_key()
            .clone();
        baseline_observations.insert(effect_key, ObservedValue::Bool(false));
        let artifact = CausalEvidenceArtifact::from_invocation(
            &invocation,
            &fixture.executor,
            baseline_observations,
            treatment_observations.clone(),
        )
        .unwrap();
        let evidence = evidence_input(
            &mut fixture,
            EvidenceKind::CausalComparison,
            artifact.canonical_bytes().unwrap(),
        );
        let proposed = canonical_completed(treatment_observations);
        assert_eq!(
            fixture
                .engine
                .evidence_verifiers
                .derive_verified_outcome(
                    &verification_context(&fixture.engine, &invocation, &executor),
                    &proposed,
                    std::slice::from_ref(&evidence),
                )
                .unwrap(),
            proposed
        );

        let mut tampered = artifact.clone();
        tampered.baseline = ObservedValue::Bool(true);
        let tampered_evidence = evidence_input(
            &mut fixture,
            EvidenceKind::CausalComparison,
            tampered.canonical_bytes().unwrap(),
        );
        assert!(fixture
            .engine
            .evidence_verifiers
            .derive_verified_outcome(
                &verification_context(&fixture.engine, &invocation, &executor),
                &proposed,
                &[tampered_evidence],
            )
            .unwrap_err()
            .to_string()
            .contains("causal_evidence_protocol_or_inputs_mismatch"));

        let adapter = ProductionEvidenceAdapter {
            identity: fixture.executor.clone(),
            expected_invocation: invocation.digest().unwrap(),
            execution: AdapterExecution::new(proposed, vec![evidence]),
        };
        let receipt = fixture.engine.execute(&state, &adapter).unwrap();
        let advanced = fixture.engine.advance(&state, &receipt).unwrap();
        assert!(matches!(
            fixture.engine.plan(advanced.state()).unwrap(),
            PlanningDecision::Terminal {
                terminal: KnowledgeTerminal::ScopedComplete { .. }
            }
        ));
    }

    #[test]
    fn counterexample_verifier_selects_first_real_counterexample_and_blocks_claim() {
        let mut fixture = fixture();
        let state = initialize(
            &fixture,
            "counterexample.production.verifier.inquiry.v1",
            counterexample_contract(KnowledgeBudget::conservative_default()),
        );
        fixture.engine.persist_state(&state).unwrap();
        let invocation = planned_invocation(&fixture.engine, &state);
        let executor = fixture
            .engine
            .authenticate_executor(fixture.executor.clone())
            .unwrap();
        let matching = matching_observations(&invocation);
        let mut contradicting = matching.clone();
        let target_key = invocation
            .binding()
            .criterion
            .predicate()
            .observation_key()
            .clone();
        contradicting.insert(target_key, ObservedValue::Bool(false));
        let cases = BTreeMap::from([
            (
                CounterexampleCaseId::parse("case.00.matching").unwrap(),
                CounterexampleCaseEvidence::new(
                    BTreeMap::from([(
                        ScenarioInputId::parse("candidate.value").unwrap(),
                        ObservedValue::Bool(false),
                    )]),
                    matching.clone(),
                )
                .unwrap(),
            ),
            (
                CounterexampleCaseId::parse("case.01.counterexample").unwrap(),
                CounterexampleCaseEvidence::new(
                    BTreeMap::from([(
                        ScenarioInputId::parse("candidate.value").unwrap(),
                        ObservedValue::Bool(true),
                    )]),
                    contradicting.clone(),
                )
                .unwrap(),
            ),
        ]);
        let artifact =
            CounterexampleEvidenceArtifact::from_invocation(&invocation, &fixture.executor, cases)
                .unwrap();
        let evidence = evidence_input(
            &mut fixture,
            EvidenceKind::CounterexampleSearch,
            artifact.canonical_bytes().unwrap(),
        );
        let proposed = canonical_completed(contradicting);
        assert_eq!(
            fixture
                .engine
                .evidence_verifiers
                .derive_verified_outcome(
                    &verification_context(&fixture.engine, &invocation, &executor),
                    &proposed,
                    std::slice::from_ref(&evidence),
                )
                .unwrap(),
            proposed
        );

        let no_counterexample = CounterexampleEvidenceArtifact::from_invocation(
            &invocation,
            &fixture.executor,
            BTreeMap::from([(
                CounterexampleCaseId::parse("case.only.matching").unwrap(),
                CounterexampleCaseEvidence::new(BTreeMap::new(), matching).unwrap(),
            )]),
        )
        .unwrap();
        let no_counterexample_evidence = evidence_input(
            &mut fixture,
            EvidenceKind::CounterexampleSearch,
            no_counterexample.canonical_bytes().unwrap(),
        );
        assert!(fixture
            .engine
            .evidence_verifiers
            .derive_verified_outcome(
                &verification_context(&fixture.engine, &invocation, &executor),
                &proposed,
                &[no_counterexample_evidence],
            )
            .unwrap_err()
            .to_string()
            .contains("counterexample_evidence_contains_no_counterexample"));

        let adapter = ProductionEvidenceAdapter {
            identity: fixture.executor.clone(),
            expected_invocation: invocation.digest().unwrap(),
            execution: AdapterExecution::new(proposed, vec![evidence]),
        };
        let receipt = fixture.engine.execute(&state, &adapter).unwrap();
        let advanced = fixture.engine.advance(&state, &receipt).unwrap();
        assert!(matches!(
            fixture.engine.plan(advanced.state()).unwrap(),
            PlanningDecision::Terminal {
                terminal: KnowledgeTerminal::Blocked { .. }
            }
        ));
    }

    #[test]
    fn formal_verifier_is_typed_but_cannot_self_attest_without_a_proof_backend() {
        let mut fixture = fixture();
        let state = initialize(
            &fixture,
            "formal.production.verifier.inquiry.v1",
            formal_contract(KnowledgeBudget::conservative_default()),
        );
        fixture.engine.persist_state(&state).unwrap();
        let invocation = planned_invocation(&fixture.engine, &state);
        let executor = fixture
            .engine
            .authenticate_executor(fixture.executor.clone())
            .unwrap();
        let unavailable =
            FormalVerificationEvidenceArtifact::backend_unavailable(&invocation, &fixture.executor)
                .unwrap();
        let blocked_evidence = evidence_input(
            &mut fixture,
            EvidenceKind::FormalSearchLog,
            unavailable.canonical_bytes().unwrap(),
        );
        let blocked = ActionOutcome::Blocked {
            reason: ExecutionBlockReason::AdapterUnavailable,
        };
        assert_eq!(
            fixture
                .engine
                .evidence_verifiers
                .derive_verified_outcome(
                    &verification_context(&fixture.engine, &invocation, &executor),
                    &blocked,
                    std::slice::from_ref(&blocked_evidence),
                )
                .unwrap(),
            blocked
        );
        let false_block = ActionOutcome::Blocked {
            reason: ExecutionBlockReason::SafetyPolicyDenied,
        };
        assert!(fixture
            .engine
            .evidence_verifiers
            .derive_verified_outcome(
                &verification_context(&fixture.engine, &invocation, &executor),
                &false_block,
                std::slice::from_ref(&blocked_evidence),
            )
            .unwrap_err()
            .to_string()
            .contains("formal_blocked_outcome_not_canonically_derived"));

        let CognitiveInvocation::VerifyFormalProperty(formal) = &invocation else {
            panic!("expected formal invocation");
        };
        let opaque_certificate = UntrustedFormalCertificateArtifactDto {
            binding: EvidenceAuthorityBinding::from_invocation(&invocation, &fixture.executor)
                .unwrap(),
            protocol: FormalEvidenceProtocol::ExternalCertificateV1,
            proof_system: formal.proof_system,
            property: formal.property.clone(),
            certificate: b"executor says this is a proof".to_vec(),
        };
        let certificate_evidence = evidence_input(
            &mut fixture,
            EvidenceKind::FormalCertificate,
            serde_json::to_vec(&opaque_certificate).unwrap(),
        );
        let forged_completed = canonical_completed(matching_observations(&invocation));
        assert!(fixture
            .engine
            .evidence_verifiers
            .derive_verified_outcome(
                &verification_context(&fixture.engine, &invocation, &executor),
                &forged_completed,
                &[certificate_evidence],
            )
            .unwrap_err()
            .to_string()
            .contains("formal_certificate_checker_backend_unavailable"));

        let adapter = ProductionEvidenceAdapter {
            identity: fixture.executor.clone(),
            expected_invocation: invocation.digest().unwrap(),
            execution: AdapterExecution::new(blocked, vec![blocked_evidence]),
        };
        let receipt = fixture.engine.execute(&state, &adapter).unwrap();
        let advanced = fixture.engine.advance(&state, &receipt).unwrap();
        assert!(matches!(
            fixture.engine.plan(advanced.state()).unwrap(),
            PlanningDecision::Terminal {
                terminal: KnowledgeTerminal::Blocked { .. }
            }
        ));
    }

    #[test]
    fn foreign_tampered_and_replayed_receipts_are_rejected() {
        let mut fixture = fixture();
        let state = initialize(
            &fixture,
            "receipt.integrity.inquiry.v1",
            trace_contract(KnowledgeBudget::conservative_default()),
        );
        fixture.engine.persist_state(&state).unwrap();
        let adapter = adapter(
            &mut fixture,
            TestOutcomeMode::Complete {
                dependency_edges: BTreeSet::new(),
            },
        );
        let receipt = fixture.engine.execute(&state, &adapter).unwrap();

        let mut tampered = receipt.clone();
        tampered.parent_state = KnowledgeStateDigest::computed(STATE_DOMAIN, b"foreign-state");
        assert!(fixture.engine.advance(&state, &tampered).is_err());

        let first = fixture.engine.advance(&state, &receipt).unwrap();
        assert!(fixture.engine.advance(first.state(), &receipt).is_err());

        let foreign = initialize(
            &fixture,
            "foreign.inquiry.v1",
            trace_contract(KnowledgeBudget::conservative_default()),
        );
        assert!(fixture.engine.advance(&foreign, &receipt).is_err());
    }

    #[test]
    fn budget_exhaustion_and_dependency_cycle_never_become_complete() {
        let mut bounded_fixture = fixture();
        let tiny_budget = KnowledgeBudget::bounded(1, 100, 4, 8, 16, 4, 8, 8).unwrap();
        let bounded_state = initialize(
            &bounded_fixture,
            "bounded.inquiry.v1",
            trace_contract(tiny_budget),
        );
        bounded_fixture
            .engine
            .persist_state(&bounded_state)
            .unwrap();
        let bounded_adapter = adapter(&mut bounded_fixture, TestOutcomeMode::Bounded);
        let receipt = bounded_fixture
            .engine
            .execute(&bounded_state, &bounded_adapter)
            .unwrap();
        let bounded = bounded_fixture
            .engine
            .advance(&bounded_state, &receipt)
            .unwrap();
        assert!(matches!(
            bounded_fixture.engine.plan(bounded.state()).unwrap(),
            PlanningDecision::Terminal {
                terminal: KnowledgeTerminal::BoundedUnknown {
                    reason: BoundedUnknownReason::ActionBudgetExhausted,
                    ..
                }
            }
        ));

        let mut cycle_fixture = fixture();
        let cycle_state = initialize(
            &cycle_fixture,
            "cycle.inquiry.v1",
            dependency_contract("cycle.root"),
        );
        cycle_fixture.engine.persist_state(&cycle_state).unwrap();
        let self_edge = DependencyEdge::new(
            DependencyNodeId::parse("cycle.root").unwrap(),
            DependencyNodeId::parse("cycle.root").unwrap(),
        );
        let cycle_adapter = adapter(
            &mut cycle_fixture,
            TestOutcomeMode::Complete {
                dependency_edges: BTreeSet::from([self_edge]),
            },
        );
        let cycle_receipt = cycle_fixture
            .engine
            .execute(&cycle_state, &cycle_adapter)
            .unwrap();
        let cycle = cycle_fixture
            .engine
            .advance(&cycle_state, &cycle_receipt)
            .unwrap();
        assert!(matches!(
            cycle_fixture.engine.plan(cycle.state()).unwrap(),
            PlanningDecision::Terminal {
                terminal: KnowledgeTerminal::Blocked { .. }
            }
        ));
    }

    #[test]
    fn causal_predictions_are_precommitted_and_reduce_rivals() {
        let mut fixture = fixture();
        let criterion = bool_predicate("causal.effect.valid");
        let family = HypothesisFamilyId::parse("mechanism.family").unwrap();
        let hypothesis_a = KnowledgeHypothesisId::parse("mechanism.a").unwrap();
        let hypothesis_b = KnowledgeHypothesisId::parse("mechanism.b").unwrap();
        let prediction_key = ObservationKey::parse("causal.mechanism.a.selected").unwrap();
        let contract = KnowledgeContractDraft {
            claims: vec![KnowledgeClaimDraft {
                claim_id: KnowledgeClaimId::parse("claim.causal.effect").unwrap(),
                domain: KnowledgeDomain::CausalMechanism,
                predicate: criterion.clone(),
            }],
            obligations: vec![KnowledgeObligationDraft {
                obligation_id: KnowledgeObligationId::parse("obligation.causal.effect").unwrap(),
                claim_id: KnowledgeClaimId::parse("claim.causal.effect").unwrap(),
                criterion: EvaluationCriterion::exact(criterion),
                plan: CognitivePlan::CausalIntervention {
                    variable: InterventionVariableId::parse("memory.strategy").unwrap(),
                    baseline: ObservedValue::Bool(false),
                    treatment: ObservedValue::Bool(true),
                    effect: bool_predicate("causal.effect.valid"),
                    predictions: BTreeMap::from([
                        (
                            hypothesis_a.clone(),
                            KnowledgePredicate::BoolEquals {
                                key: prediction_key.clone(),
                                expected: true,
                            },
                        ),
                        (
                            hypothesis_b.clone(),
                            KnowledgePredicate::BoolEquals {
                                key: prediction_key,
                                expected: false,
                            },
                        ),
                    ]),
                },
            }],
            hypotheses: vec![
                KnowledgeHypothesisDraft {
                    hypothesis_id: hypothesis_a.clone(),
                    family_id: family.clone(),
                    definition: bool_predicate("definition.mechanism.a"),
                },
                KnowledgeHypothesisDraft {
                    hypothesis_id: hypothesis_b.clone(),
                    family_id: family,
                    definition: bool_predicate("definition.mechanism.b"),
                },
            ],
            budget: KnowledgeBudget::conservative_default(),
        };
        let state = initialize(&fixture, "causal.inquiry.v1", contract);
        fixture.engine.persist_state(&state).unwrap();
        let adapter = adapter(
            &mut fixture,
            TestOutcomeMode::Complete {
                dependency_edges: BTreeSet::new(),
            },
        );
        let receipt = fixture.engine.execute(&state, &adapter).unwrap();
        let next = fixture.engine.advance(&state, &receipt).unwrap();
        assert!(matches!(
            next.state().hypotheses()[&hypothesis_a].assessment(),
            HypothesisAssessment::Supported { .. }
        ));
        assert!(matches!(
            next.state().hypotheses()[&hypothesis_b].assessment(),
            HypothesisAssessment::Refuted { .. }
        ));
        assert!(matches!(
            fixture.engine.plan(next.state()).unwrap(),
            PlanningDecision::Terminal {
                terminal: KnowledgeTerminal::ScopedComplete { .. }
            }
        ));
    }

    #[test]
    fn malformed_ids_and_unproven_contract_omissions_fail_closed() {
        assert!(KnowledgeClaimId::parse("a".repeat(64)).is_err());
        assert!(KnowledgeSymbol::parse("human readable free text").is_err());
        let fixture = fixture();
        let empty = KnowledgeContractDraft {
            claims: vec![],
            obligations: vec![],
            hypotheses: vec![],
            budget: KnowledgeBudget::conservative_default(),
        };
        assert!(fixture
            .engine
            .initialize(
                InquiryId::parse("empty.inquiry.v1").unwrap(),
                fixture.bundle.clone(),
                empty,
            )
            .is_err());
    }

    #[test]
    fn executor_block_is_structured_and_never_satisfies_an_obligation() {
        let mut fixture = fixture();
        let state = initialize(
            &fixture,
            "blocked.inquiry.v1",
            trace_contract(KnowledgeBudget::conservative_default()),
        );
        fixture.engine.persist_state(&state).unwrap();
        let adapter = adapter(&mut fixture, TestOutcomeMode::Blocked);
        let receipt = fixture.engine.execute(&state, &adapter).unwrap();
        let next = fixture.engine.advance(&state, &receipt).unwrap();
        assert!(matches!(
            fixture.engine.plan(next.state()).unwrap(),
            PlanningDecision::Terminal {
                terminal: KnowledgeTerminal::Blocked { .. }
            }
        ));
    }
}
