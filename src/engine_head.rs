//! Canonical engine head, transition journal and recovery outcomes.
//!
//! Live pointers remain for compatibility with existing receipts. This module
//! adds the missing atomic snapshot: one revision that names the corpus, bank,
//! memory, sleep state, evidence and reconstruction together, plus a typed
//! journal so an interrupted corpus transition can be recovered without
//! guessing between forks.

use crate::digest::{
    CanonicalEngineHeadDigest, CorpusDigest, EvidenceBundleDigest, MemoryDigest, ReportDigest,
    Sha256Digest, SkillBankDigest,
};
use crate::error::{BrainError, BrainResult};
use serde::{Deserialize, Serialize};

pub const CANONICAL_ENGINE_HEAD_SCHEMA: &str = "cerebro.tidex.canonical_engine_head/v1";
pub const CANONICAL_ENGINE_HEAD_DOMAIN: &[u8] = b"CEREBRO:TIDEX:CANONICAL-ENGINE-HEAD:v1\0";
pub const CANONICAL_ENGINE_HEAD_MAX_BYTES: u64 = 1 << 20;
pub const CORPUS_TRANSITION_JOURNAL_SCHEMA: &str = "cerebro.tidex.corpus_transition_journal/v1";
pub const CORPUS_TRANSITION_JOURNAL_MAX_BYTES: u64 = 1 << 20;
pub const HARD_MAX_ENGINE_REVISION: u64 = 1_048_576;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CorpusTransitionPhase {
    IntentRecorded,
    PriorArchived,
    NewCorpusStaged,
    NewCorpusPublished,
    CommitSealed,
    ReceiptSealed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CorpusTransitionRecoveryOutcome {
    NoIncompleteTransition,
    ClosedVerifiedReceipt,
    RolledBackIntent,
    RestoredPriorCorpus,
    RequiresOriginalFinalizationReplay,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CorpusTransitionRecovery {
    pub schema: String,
    pub outcome: CorpusTransitionRecoveryOutcome,
    pub operation_key: Option<Sha256Digest>,
    pub phase: Option<CorpusTransitionPhase>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CanonicalEngineHead {
    pub schema: String,
    pub revision: u64,
    pub parent_revision: Option<u64>,
    pub parent_digest: Option<CanonicalEngineHeadDigest>,
    pub corpus_digest: Option<CorpusDigest>,
    pub observation_count: usize,
    pub active_bank_sha256: Option<SkillBankDigest>,
    pub memory_sha256: Option<MemoryDigest>,
    pub sleep_state_sha256: Option<Sha256Digest>,
    pub evidence_bundle_sha256: Option<EvidenceBundleDigest>,
    pub reconstruction_report_sha256: Option<ReportDigest>,
    pub certification_status: Option<String>,
    pub incomplete_transition: Option<Sha256Digest>,
    pub manifest_digest: CanonicalEngineHeadDigest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CorpusTransitionJournal {
    pub schema: String,
    pub operation_key: Sha256Digest,
    pub phase: CorpusTransitionPhase,
    pub parent_head_digest: Option<CanonicalEngineHeadDigest>,
    pub parent_head_revision: u64,
}

impl CanonicalEngineHead {
    pub fn calculate_digest(&self) -> BrainResult<CanonicalEngineHeadDigest> {
        let mut unsigned = self.clone();
        unsigned.manifest_digest = CanonicalEngineHeadDigest::from(Sha256Digest::zero());
        let payload = serde_json::to_vec(&unsigned)?;
        Ok(CanonicalEngineHeadDigest::from(
            Sha256Digest::digest_domain(CANONICAL_ENGINE_HEAD_DOMAIN, &payload),
        ))
    }

    pub fn seal(mut self) -> BrainResult<Self> {
        self.schema = CANONICAL_ENGINE_HEAD_SCHEMA.to_string();
        if self.revision > HARD_MAX_ENGINE_REVISION {
            return Err(BrainError::Integrity(
                "canonical_engine_head_revision_limit_exceeded".into(),
            ));
        }
        if self.observation_count > 0 && self.corpus_digest.is_none() {
            return Err(BrainError::Integrity(
                "canonical_engine_head_corpus_digest_missing".into(),
            ));
        }
        if let (Some(parent_revision), Some(_)) = (self.parent_revision, &self.parent_digest) {
            if parent_revision.checked_add(1) != Some(self.revision) {
                return Err(BrainError::Integrity(
                    "canonical_engine_head_revision_not_monotonic".into(),
                ));
            }
        } else if self.revision != 0
            || self.parent_revision.is_some()
            || self.parent_digest.is_some()
        {
            return Err(BrainError::Integrity(
                "canonical_engine_head_genesis_invalid".into(),
            ));
        }
        self.manifest_digest = self.calculate_digest()?;
        Ok(self)
    }

    pub fn authenticate(&self) -> BrainResult<()> {
        if self.schema != CANONICAL_ENGINE_HEAD_SCHEMA
            || self.manifest_digest != self.calculate_digest()?
        {
            return Err(BrainError::Integrity(
                "canonical_engine_head_not_authentic".into(),
            ));
        }
        let _ = self.clone().seal()?;
        Ok(())
    }
}

impl CorpusTransitionJournal {
    pub fn new(
        operation_key: Sha256Digest,
        phase: CorpusTransitionPhase,
        parent: Option<&CanonicalEngineHead>,
    ) -> BrainResult<Self> {
        Ok(Self {
            schema: CORPUS_TRANSITION_JOURNAL_SCHEMA.to_string(),
            operation_key,
            phase,
            parent_head_digest: parent.map(|head| head.manifest_digest.clone()),
            parent_head_revision: parent.map(|head| head.revision).unwrap_or(0),
        })
    }

    pub fn authenticate(&self, expected_key: Option<&Sha256Digest>) -> BrainResult<()> {
        if self.schema != CORPUS_TRANSITION_JOURNAL_SCHEMA {
            return Err(BrainError::Integrity(
                "corpus_transition_journal_schema_invalid".into(),
            ));
        }
        if expected_key.is_some_and(|expected| expected != &self.operation_key) {
            return Err(BrainError::Integrity(
                "corpus_transition_journal_operation_mismatch".into(),
            ));
        }
        Ok(())
    }
}

impl CorpusTransitionRecovery {
    pub fn new(
        outcome: CorpusTransitionRecoveryOutcome,
        operation_key: Option<Sha256Digest>,
        phase: Option<CorpusTransitionPhase>,
    ) -> Self {
        Self {
            schema: "cerebro.tidex.corpus_transition_recovery/v1".into(),
            outcome,
            operation_key,
            phase,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_head(revision: u64, parent: Option<&CanonicalEngineHead>) -> CanonicalEngineHead {
        CanonicalEngineHead {
            schema: CANONICAL_ENGINE_HEAD_SCHEMA.into(),
            revision,
            parent_revision: parent.map(|head| head.revision),
            parent_digest: parent.map(|head| head.manifest_digest.clone()),
            corpus_digest: None,
            observation_count: 0,
            active_bank_sha256: None,
            memory_sha256: None,
            sleep_state_sha256: None,
            evidence_bundle_sha256: None,
            reconstruction_report_sha256: None,
            certification_status: None,
            incomplete_transition: None,
            manifest_digest: CanonicalEngineHeadDigest::from(Sha256Digest::zero()),
        }
        .seal()
        .unwrap()
    }

    #[test]
    fn sealed_head_digest_covers_every_named_authority() {
        let genesis = sample_head(0, None);
        genesis.authenticate().unwrap();
        let mut child = sample_head(1, Some(&genesis));
        child.corpus_digest = Some(CorpusDigest::from(Sha256Digest::digest_bytes(b"corpus")));
        child.observation_count = 6;
        child = child.seal().unwrap();
        child.authenticate().unwrap();
        assert_ne!(genesis.manifest_digest, child.manifest_digest);
    }

    #[test]
    fn genesis_must_be_revision_zero_without_parent() {
        let err = CanonicalEngineHead {
            schema: CANONICAL_ENGINE_HEAD_SCHEMA.into(),
            revision: 1,
            parent_revision: None,
            parent_digest: None,
            corpus_digest: None,
            observation_count: 0,
            active_bank_sha256: None,
            memory_sha256: None,
            sleep_state_sha256: None,
            evidence_bundle_sha256: None,
            reconstruction_report_sha256: None,
            certification_status: None,
            incomplete_transition: None,
            manifest_digest: CanonicalEngineHeadDigest::from(Sha256Digest::zero()),
        }
        .seal()
        .unwrap_err();
        assert!(matches!(err, BrainError::Integrity(_)));
    }

    #[test]
    fn journal_binds_the_parent_head_that_authorized_the_transition() {
        let genesis = sample_head(0, None);
        let key = Sha256Digest::digest_bytes(b"operation");
        let journal = CorpusTransitionJournal::new(
            key.clone(),
            CorpusTransitionPhase::PriorArchived,
            Some(&genesis),
        )
        .unwrap();
        journal.authenticate(Some(&key)).unwrap();
        assert_eq!(journal.parent_head_digest.unwrap(), genesis.manifest_digest);
        assert_eq!(journal.parent_head_revision, 0);
    }
}
