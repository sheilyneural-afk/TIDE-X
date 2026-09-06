//! Universal, donor-agnostic admission and source-envelope primitives.
//!
//! This module deliberately does **not** know about any donor, provider,
//! language toolchain, model format, or project-specific layout. It is the first
//! authority in the acquisition chain: it seals the requested scope and
//! produces a deterministic, re-verifiable manifest of the source tree.
//!
//! An envelope is a descriptor-bound observation of the requested semantic
//! projection: admitted regular-file bytes, executable bits, directory shape
//! and declared exclusions.  It intentionally does not claim to snapshot
//! ACLs, extended attributes, device state or one atomic filesystem instant.
//! Nor is it evidence that the tree implements a requested capability.  Those
//! claims require retained bytes plus later discovery, execution, causal and
//! translation authorities.
//!
//! # Platform and security contract
//!
//! Capture currently requires Linux `openat2(2)` confinement and `/proc` file
//! descriptors.  If either mechanism is unavailable the operation fails
//! closed.  A pathname-based fallback would reintroduce race and symlink
//! escapes, so portability requires an independently verified descriptor-based
//! backend rather than silently weakening this authority.  The selected tree
//! may itself be a mounted root, but nested mount crossings are rejected: a
//! generic capture cannot prove that a second mount is quiescent or disjoint
//! from the private authority without a source-specific snapshot adapter.

use crate::digest::{AcquisitionRequestDigest, Sha256Digest, SystemEnvelopeDigest};
use crate::error::{BrainError, BrainResult};
use crate::identity::AcquisitionId;
use rustix::fd::OwnedFd;
use rustix::fs::{fstat, open, openat, openat2, Dir, FileType, Mode, OFlags, ResolveFlags, Stat};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::os::fd::AsRawFd;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Component, Path, PathBuf};

const REQUEST_DOMAIN: &[u8] = b"CEREBRO:TIDEX:ACQUISITION-REQUEST:v1\0";
const ENVELOPE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:SYSTEM-ENVELOPE:v1\0";
const FILE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:SOURCE-FILE:v1\0";
const DIRECTORY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:SOURCE-DIRECTORY:v1\0";

// Protocol safety ceilings.  The caller-controlled byte/file budget is still
// authoritative for admitted regular-file contents; these additional limits
// prevent empty-directory, path and manifest amplification from bypassing it.
const MAX_SCOPE_ROOTS: usize = 4_096;
const MAX_EXCLUSIONS: usize = 100_000;
const MAX_RELATIVE_COMPONENTS: usize = 256;
const MAX_RELATIVE_PATH_BYTES: usize = 16 * 1_024;
const MAX_DIRECTORY_CHILDREN: u64 = 100_000;
const MAX_TRAVERSED_ENTRIES: u64 = 1_000_000;
const MAX_TRAVERSED_NAME_BYTES: u64 = 64 * 1_024 * 1_024;
const MAX_CAPTURED_DIRECTORIES: u64 = 100_000;
const MAX_MANIFEST_ENTRIES: u64 = 500_000;
const MAX_MANIFEST_PATH_BYTES: u64 = 64 * 1_024 * 1_024;
const MAX_DESCRIPTOR_ANCESTORS: usize = 16 * 1_024;
const STREAM_BUFFER_BYTES: usize = 64 * 1_024;

const ROOT_RESOLUTION: ResolveFlags = ResolveFlags::BENEATH
    .union(ResolveFlags::NO_SYMLINKS)
    .union(ResolveFlags::NO_MAGICLINKS);
const SOURCE_TREE_RESOLUTION: ResolveFlags = ROOT_RESOLUTION.union(ResolveFlags::NO_XDEV);

/// Lossless and deterministic Unix path wire encoding.  JSON strings cannot
/// represent every valid filesystem name, so sealed artifacts use explicitly
/// tagged lowercase hexadecimal bytes rather than lossy Unicode conversion.
mod path_wire {
    use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};
    use std::ffi::OsString;
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    use std::path::{Path, PathBuf};

    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct EncodedPath {
        pub(super) unix_bytes_hex: String,
    }

    pub fn serialize<S>(path: &Path, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        encode(path).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<PathBuf, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = EncodedPath::deserialize(deserializer)?;
        decode(&encoded.unix_bytes_hex).map_err(D::Error::custom)
    }

    pub(super) fn encode(path: &Path) -> EncodedPath {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let raw = path.as_os_str().as_bytes();
        let mut unix_bytes_hex = String::with_capacity(raw.len() * 2);
        for byte in raw {
            unix_bytes_hex.push(char::from(HEX[(byte >> 4) as usize]));
            unix_bytes_hex.push(char::from(HEX[(byte & 0x0f) as usize]));
        }
        EncodedPath { unix_bytes_hex }
    }

    pub(super) fn decode(value: &str) -> Result<PathBuf, &'static str> {
        if value.len() % 2 != 0
            || value
                .bytes()
                .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
        {
            return Err("unix_path_hex_invalid");
        }
        let mut bytes = Vec::with_capacity(value.len() / 2);
        for pair in value.as_bytes().chunks_exact(2) {
            let high = hex_nibble(pair[0]).ok_or("unix_path_hex_invalid")?;
            let low = hex_nibble(pair[1]).ok_or("unix_path_hex_invalid")?;
            bytes.push((high << 4) | low);
        }
        Ok(PathBuf::from(OsString::from_vec(bytes)))
    }

    fn hex_nibble(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AcquisitionSchema {
    #[serde(rename = "cerebro.tidex.acquisition_request/v1")]
    V1,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SystemEnvelopeSchema {
    #[serde(rename = "cerebro.tidex.system_envelope/v1")]
    V1,
}

/// A user-declared path inside a donor root.
///
/// Unlike an observed manifest path, this type can never represent the empty
/// project root.  Construction validates the same lexical and protocol limits
/// used during capture, so an invalid absolute, parent-relative or oversized
/// path cannot enter a public `DeclaredPaths` scope as an unchecked `PathBuf`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeclaredRelativePath(PathBuf);

impl DeclaredRelativePath {
    pub fn parse(path: impl Into<PathBuf>) -> BrainResult<Self> {
        Self::parse_labeled(path.into(), "acquisition_declared_path")
    }

    fn parse_labeled(path: PathBuf, label: &str) -> BrainResult<Self> {
        validate_declared_relative_path(&path, label)?;
        Ok(Self(path))
    }

    pub fn as_path(&self) -> &Path {
        &self.0
    }

    pub fn into_path_buf(self) -> PathBuf {
        self.0
    }
}

impl AsRef<Path> for DeclaredRelativePath {
    fn as_ref(&self) -> &Path {
        self.as_path()
    }
}

impl Serialize for DeclaredRelativePath {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        path_wire::serialize(self.as_path(), serializer)
    }
}

impl<'de> Deserialize<'de> for DeclaredRelativePath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let path = path_wire::deserialize(deserializer)?;
        Self::parse(path).map_err(serde::de::Error::custom)
    }
}

/// A canonical path observed by the descriptor-bound capture authority.
///
/// This is intentionally distinct from `DeclaredRelativePath`: the synthetic
/// manifest root is represented by an empty observed path, while a path
/// supplied by a user may never be empty.  The inner path is private and every
/// deserialized value is validated before it can enter a manifest.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObservedRelativePath(PathBuf);

impl ObservedRelativePath {
    fn parse(path: impl Into<PathBuf>) -> BrainResult<Self> {
        let path = path.into();
        validate_observed_relative_path(&path)?;
        Ok(Self(path))
    }

    fn root() -> Self {
        Self(PathBuf::new())
    }

    pub fn as_path(&self) -> &Path {
        &self.0
    }

    pub fn is_root(&self) -> bool {
        self.0.as_os_str().is_empty()
    }
}

impl AsRef<Path> for ObservedRelativePath {
    fn as_ref(&self) -> &Path {
        self.as_path()
    }
}

impl Serialize for ObservedRelativePath {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        path_wire::serialize(self.as_path(), serializer)
    }
}

impl<'de> Deserialize<'de> for ObservedRelativePath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let path = path_wire::deserialize(deserializer)?;
        Self::parse(path).map_err(serde::de::Error::custom)
    }
}

/// The user-authorized portion of a donor tree.  `DeclaredPaths` is purposely
/// not called a dependency closure: resolving one is a later, evidence-backed
/// gate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AcquisitionScope {
    WholeProject,
    DeclaredPaths { roots: Vec<DeclaredRelativePath> },
}

impl AcquisitionScope {
    pub fn declared_paths(roots: Vec<PathBuf>) -> BrainResult<Self> {
        let roots = roots
            .into_iter()
            .map(|path| DeclaredRelativePath::parse_labeled(path, "acquisition_scope_root"))
            .collect::<BrainResult<Vec<_>>>()?;
        normalize_scope(Self::DeclaredPaths { roots })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RequestedResidency {
    /// Let later authorities choose between portable IR, runtime primitives and
    /// an eligible target model.
    BestVerified,
    /// A later authority must reject rather than silently fall back from model
    /// weights to an opaque external module.
    WeightsOnly,
    PortableIrOnly,
}

/// Controls whether known generated artifacts are admitted into the semantic
/// source universe.  `ExplicitOnly` is the command's safe default.  The
/// conservative generated-artifact rules are opt-in because a compiled binary
/// or environment may itself be a required implementation, oracle or target.
/// Even the opt-in policy intentionally keeps tests, examples, datasets,
/// model files and generic directories merely because those may be evidence
/// or the requested target itself.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NoisePolicy {
    ExplicitOnly,
    ConservativeGeneratedArtifacts,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AcquisitionBudget {
    pub max_files: u64,
    pub max_total_bytes: u64,
}

impl AcquisitionBudget {
    pub fn validate(&self) -> BrainResult<()> {
        if self.max_files == 0 || self.max_total_bytes == 0 {
            return Err(BrainError::Invalid("acquisition_budget_zero".into()));
        }
        Ok(())
    }
}

/// A sealed request contains no host path.  The caller supplies the physical
/// source root to capture; changing that root changes the envelope, never the
/// request's semantic identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AcquisitionRequest {
    schema: AcquisitionSchema,
    acquisition_id: AcquisitionId,
    scope: AcquisitionScope,
    requested_residency: RequestedResidency,
    noise_policy: NoisePolicy,
    budget: AcquisitionBudget,
    /// Explicit relative paths excluded from capture (for example build
    /// caches).  They are never silently ignored.
    exclusions: Vec<DeclaredRelativePath>,
    manifest_sha256: AcquisitionRequestDigest,
}

impl AcquisitionRequest {
    pub fn new(
        acquisition_id: AcquisitionId,
        scope: AcquisitionScope,
        requested_residency: RequestedResidency,
        noise_policy: NoisePolicy,
        budget: AcquisitionBudget,
        exclusions: Vec<PathBuf>,
    ) -> BrainResult<Self> {
        budget.validate()?;
        let scope = normalize_scope(scope)?;
        let exclusions = exclusions
            .into_iter()
            .map(|path| DeclaredRelativePath::parse_labeled(path, "acquisition_exclusion"))
            .collect::<BrainResult<Vec<_>>>()?;
        let exclusions = canonical_declared_set(&exclusions, "acquisition_exclusion")?;
        validate_non_overlapping_exclusions(&exclusions)?;
        validate_scope_exclusion_separation(&scope, &exclusions)?;
        let mut request = Self {
            schema: AcquisitionSchema::V1,
            acquisition_id,
            scope,
            requested_residency,
            noise_policy,
            budget,
            exclusions: exclusions.into_iter().collect(),
            manifest_sha256: AcquisitionRequestDigest::draft_marker(),
        };
        request.manifest_sha256 = request.calculate_digest()?;
        Ok(request)
    }

    pub fn verify(&self) -> BrainResult<()> {
        self.budget.validate()?;
        validate_scope(&self.scope)?;
        let canonical = canonical_declared_set(&self.exclusions, "acquisition_exclusion")?;
        if canonical.iter().cloned().collect::<Vec<_>>() != self.exclusions {
            return Err(BrainError::Integrity(
                "acquisition_exclusions_not_canonical".into(),
            ));
        }
        validate_non_overlapping_exclusions(&canonical)?;
        validate_scope_exclusion_separation(&self.scope, &canonical)?;
        if self.manifest_sha256.is_draft() || self.calculate_digest()? != self.manifest_sha256 {
            return Err(BrainError::Integrity(
                "acquisition_request_digest_mismatch".into(),
            ));
        }
        Ok(())
    }

    pub fn acquisition_id(&self) -> &AcquisitionId {
        &self.acquisition_id
    }

    pub fn scope(&self) -> &AcquisitionScope {
        &self.scope
    }

    pub fn requested_residency(&self) -> RequestedResidency {
        self.requested_residency
    }

    pub fn noise_policy(&self) -> NoisePolicy {
        self.noise_policy
    }

    pub fn budget(&self) -> &AcquisitionBudget {
        &self.budget
    }

    pub fn exclusions(&self) -> &[DeclaredRelativePath] {
        &self.exclusions
    }

    pub fn manifest_sha256(&self) -> &AcquisitionRequestDigest {
        &self.manifest_sha256
    }

    pub(crate) fn calculate_digest(&self) -> BrainResult<AcquisitionRequestDigest> {
        let mut unsigned = self.clone();
        unsigned.manifest_sha256 = AcquisitionRequestDigest::draft_marker();
        Ok(AcquisitionRequestDigest::from_computed(domain_digest(
            REQUEST_DOMAIN,
            &serde_json::to_vec(&unsigned)?,
        )))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotCompleteness {
    /// Every regular file and directory in the semantic projection below the
    /// supplied root was included except exclusions listed in the envelope.
    /// This does not claim capture of ACLs, xattrs or a filesystem snapshot.
    WholeTree,
    /// Only roots explicitly named by the request were included.  This is not
    /// a statement that their dependency closure is complete.
    DeclaredScopeOnly,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotEntryKind {
    File,
    Directory,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SnapshotEntry {
    relative_path: ObservedRelativePath,
    kind: SnapshotEntryKind,
    byte_len: u64,
    executable: bool,
    sha256: Sha256Digest,
}

impl SnapshotEntry {
    pub fn relative_path(&self) -> &Path {
        self.relative_path.as_path()
    }

    pub fn kind(&self) -> SnapshotEntryKind {
        self.kind
    }

    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub fn executable(&self) -> bool {
        self.executable
    }

    pub fn sha256(&self) -> &Sha256Digest {
        &self.sha256
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExcludedPath {
    relative_path: ObservedRelativePath,
    reason: ExclusionReason,
}

impl ExcludedPath {
    pub fn relative_path(&self) -> &Path {
        self.relative_path.as_path()
    }

    pub fn reason(&self) -> ExclusionReason {
        self.reason
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExclusionReason {
    ExplicitRequest,
    GeneratedArtifactPolicy,
}

/// Whether immutable donor bytes are retained by this artifact.  Version 1 is
/// deliberately observation-only: an envelope alone may not authorize a
/// semantic-equivalence, translation or residency claim.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SourceRetention {
    ManifestOnly,
}

/// Re-verifiable source-tree evidence.  `source_root` is deliberately absent:
/// an envelope can be checked against any candidate root without leaking a
/// machine-local source path into the sealed identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SystemEnvelope {
    schema: SystemEnvelopeSchema,
    acquisition_request_sha256: AcquisitionRequestDigest,
    completeness: SnapshotCompleteness,
    projection_roots: Vec<DeclaredRelativePath>,
    entries: Vec<SnapshotEntry>,
    excluded_paths: Vec<ExcludedPath>,
    total_file_bytes: u64,
    source_retention: SourceRetention,
    manifest_sha256: SystemEnvelopeDigest,
}

impl SystemEnvelope {
    pub fn acquisition_request_sha256(&self) -> &AcquisitionRequestDigest {
        &self.acquisition_request_sha256
    }

    pub fn completeness(&self) -> SnapshotCompleteness {
        self.completeness
    }

    pub fn projection_roots(&self) -> &[DeclaredRelativePath] {
        &self.projection_roots
    }

    pub fn entries(&self) -> &[SnapshotEntry] {
        &self.entries
    }

    pub fn excluded_paths(&self) -> &[ExcludedPath] {
        &self.excluded_paths
    }

    pub fn total_file_bytes(&self) -> u64 {
        self.total_file_bytes
    }

    pub fn source_retention(&self) -> SourceRetention {
        self.source_retention
    }

    pub fn manifest_sha256(&self) -> &SystemEnvelopeDigest {
        &self.manifest_sha256
    }

    /// Capture a complete, deterministic manifest of the selected source tree.
    /// Symlinks and special files are rejected, including those hidden below a
    /// requested scope, because accepting them makes the captured boundary
    /// dependent on host filesystem resolution.
    pub fn capture(source_root: &Path, request: &AcquisitionRequest) -> BrainResult<Self> {
        let session = SourceCaptureSession::open(source_root)?;
        Self::capture_from_session(&session, request)
    }

    /// Capture through an already pinned source-root descriptor.  The same
    /// session can therefore be reused by retention without resolving the
    /// ambient source pathname a second time.
    pub(crate) fn capture_from_session(
        session: &SourceCaptureSession,
        request: &AcquisitionRequest,
    ) -> BrainResult<Self> {
        request.verify()?;
        let source_root_stat = session.current_source_root_stat()?;
        let source_root_fd = session.source_root_fd();
        let selected = selected_roots(&request.scope)?;
        let exclusions: BTreeSet<PathBuf> = request
            .exclusions
            .iter()
            .map(|path| path.as_path().to_path_buf())
            .collect();
        let mut entries = Vec::new();
        let mut excluded_paths = Vec::new();
        let mut counters = SnapshotCounters::default();
        let mut visited_directories = BTreeSet::new();
        let mut context = CaptureContext {
            selected: &selected,
            exclusions: &exclusions,
            entries: &mut entries,
            excluded_paths: &mut excluded_paths,
            counters: &mut counters,
            visited_directories: &mut visited_directories,
            budget: &request.budget,
            noise_policy: request.noise_policy,
        };
        match &request.scope {
            AcquisitionScope::WholeProject => {
                capture_directory(
                    Path::new(""),
                    source_root_fd,
                    &source_root_stat,
                    0,
                    &mut context,
                )?;
            }
            AcquisitionScope::DeclaredPaths { .. } => {
                let mut root_commitments = Vec::with_capacity(selected.len());
                for selected_root in &selected {
                    let inspected = inspect_beneath(source_root_fd, selected_root)?;
                    let file_type = FileType::from_raw_mode(inspected.stat.st_mode);
                    let (kind, digest) = match file_type {
                        FileType::Directory => {
                            let (directory_fd, opened_stat) = open_directory_beneath(
                                source_root_fd,
                                selected_root,
                                &inspected.stat,
                            )?;
                            let digest = capture_directory(
                                selected_root,
                                &directory_fd,
                                &opened_stat,
                                selected_root.components().count(),
                                &mut context,
                            )?;
                            (SnapshotEntryKind::Directory, digest)
                        }
                        FileType::RegularFile => {
                            let digest = capture_file(selected_root, &inspected, &mut context)?;
                            (SnapshotEntryKind::File, digest)
                        }
                        FileType::Symlink => {
                            return Err(BrainError::Integrity(
                                "acquisition_source_symlink_forbidden".into(),
                            ));
                        }
                        _ => {
                            return Err(BrainError::Integrity(
                                "acquisition_source_special_file_forbidden".into(),
                            ));
                        }
                    };
                    root_commitments.push((selected_root.clone(), kind, digest));
                }
                let root_digest = directory_digest(Path::new(""), &root_commitments)?;
                admit_manifest_path(Path::new(""), &mut context)?;
                context.entries.push(SnapshotEntry {
                    relative_path: ObservedRelativePath::root(),
                    kind: SnapshotEntryKind::Directory,
                    byte_len: 0,
                    executable: false,
                    sha256: root_digest,
                });
            }
        }
        session.ensure_source_root_identity()?;
        entries.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        excluded_paths.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        validate_explicit_exclusion_coverage(&excluded_paths, &exclusions)?;
        validate_requested_coverage(&entries, &selected)?;
        let completeness = match request.scope {
            AcquisitionScope::WholeProject => SnapshotCompleteness::WholeTree,
            AcquisitionScope::DeclaredPaths { .. } => SnapshotCompleteness::DeclaredScopeOnly,
        };
        let projection_roots = match &request.scope {
            AcquisitionScope::WholeProject => Vec::new(),
            AcquisitionScope::DeclaredPaths { roots } => roots.clone(),
        };
        let mut envelope = Self {
            schema: SystemEnvelopeSchema::V1,
            acquisition_request_sha256: request.manifest_sha256.clone(),
            completeness,
            projection_roots,
            entries,
            excluded_paths,
            total_file_bytes: counters.total_file_bytes,
            source_retention: SourceRetention::ManifestOnly,
            manifest_sha256: SystemEnvelopeDigest::draft_marker(),
        };
        envelope.manifest_sha256 = envelope.calculate_digest()?;
        envelope.verify_manifest()?;
        Ok(envelope)
    }

    /// Verify that a candidate source root still matches this exact snapshot.
    /// This is intentionally a physical re-capture, not a trust in timestamps
    /// or the donor's own manifests.
    pub fn verify_against(
        &self,
        source_root: &Path,
        request: &AcquisitionRequest,
    ) -> BrainResult<()> {
        let session = SourceCaptureSession::open(source_root)?;
        self.verify_against_session(&session, request)
    }

    /// Re-capture through the same pinned source capability used to build the
    /// original envelope.  Root path replacement cannot redirect this check
    /// to a different donor tree.
    pub(crate) fn verify_against_session(
        &self,
        session: &SourceCaptureSession,
        request: &AcquisitionRequest,
    ) -> BrainResult<()> {
        request.verify()?;
        self.verify_manifest()?;
        if request.manifest_sha256 != self.acquisition_request_sha256 {
            return Err(BrainError::Integrity(
                "system_envelope_request_binding_mismatch".into(),
            ));
        }
        let expected_projection_roots: Vec<DeclaredRelativePath> = match &request.scope {
            AcquisitionScope::WholeProject => Vec::new(),
            AcquisitionScope::DeclaredPaths { roots } => roots.clone(),
        };
        if expected_projection_roots != self.projection_roots {
            return Err(BrainError::Integrity(
                "system_envelope_projection_roots_mismatch".into(),
            ));
        }
        let observed = Self::capture_from_session(session, request)?;
        if observed != *self {
            return Err(BrainError::Integrity(
                "system_envelope_source_tree_mismatch".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn calculate_digest(&self) -> BrainResult<SystemEnvelopeDigest> {
        let mut unsigned = self.clone();
        unsigned.manifest_sha256 = SystemEnvelopeDigest::draft_marker();
        Ok(SystemEnvelopeDigest::from_computed(domain_digest(
            ENVELOPE_DOMAIN,
            &serde_json::to_vec(&unsigned)?,
        )))
    }

    pub fn verify_manifest(&self) -> BrainResult<()> {
        self.validate_structure()?;
        if self.manifest_sha256.is_draft() || self.calculate_digest()? != self.manifest_sha256 {
            return Err(BrainError::Integrity(
                "system_envelope_digest_mismatch".into(),
            ));
        }
        Ok(())
    }

    fn validate_structure(&self) -> BrainResult<()> {
        if self.acquisition_request_sha256.is_draft()
            || self.manifest_sha256.is_draft()
            || self.entries.is_empty()
            || self.entries.len() > MAX_MANIFEST_ENTRIES as usize
            || self.excluded_paths.len() > MAX_EXCLUSIONS
        {
            return Err(BrainError::Integrity(
                "system_envelope_structure_invalid".into(),
            ));
        }
        match self.completeness {
            SnapshotCompleteness::WholeTree if !self.projection_roots.is_empty() => {
                return Err(BrainError::Integrity(
                    "system_envelope_whole_tree_has_projection_roots".into(),
                ));
            }
            SnapshotCompleteness::DeclaredScopeOnly => {
                if self.projection_roots.is_empty()
                    || normalize_declared_roots(&self.projection_roots)? != self.projection_roots
                {
                    return Err(BrainError::Integrity(
                        "system_envelope_projection_roots_not_canonical".into(),
                    ));
                }
            }
            SnapshotCompleteness::WholeTree => {}
        }
        if self.entries.len().saturating_add(self.excluded_paths.len())
            > MAX_MANIFEST_ENTRIES as usize
        {
            return Err(BrainError::Integrity(
                "system_envelope_manifest_entry_limit_exceeded".into(),
            ));
        }
        let root = self
            .entries
            .first()
            .filter(|entry| {
                entry.relative_path.is_root() && entry.kind == SnapshotEntryKind::Directory
            })
            .ok_or_else(|| BrainError::Integrity("system_envelope_root_missing".into()))?;
        if root.byte_len != 0 || root.executable {
            return Err(BrainError::Integrity("system_envelope_root_invalid".into()));
        }

        let zero = Sha256Digest::zero();
        let mut previous_entry: Option<&Path> = None;
        let mut total_file_bytes = 0_u64;
        let mut path_bytes = 0_u64;
        let mut directory_count = 0_u64;
        let mut entry_by_path = BTreeMap::new();
        for entry in &self.entries {
            if let Some(previous) = previous_entry {
                if previous >= entry.relative_path.as_path() {
                    return Err(BrainError::Integrity(
                        "system_envelope_entries_not_canonical".into(),
                    ));
                }
            }
            if !entry.relative_path.is_root() {
                validate_observed_relative_path(entry.relative_path.as_path())?;
            }
            path_bytes = add_manifest_path_bytes(path_bytes, entry.relative_path.as_path())?;
            if entry.sha256 == zero {
                return Err(BrainError::Integrity(
                    "system_envelope_entry_digest_zero".into(),
                ));
            }
            match entry.kind {
                SnapshotEntryKind::Directory if entry.byte_len != 0 || entry.executable => {
                    return Err(BrainError::Integrity(
                        "system_envelope_directory_entry_invalid".into(),
                    ));
                }
                SnapshotEntryKind::Directory => {}
                SnapshotEntryKind::File => {
                    total_file_bytes =
                        total_file_bytes
                            .checked_add(entry.byte_len)
                            .ok_or_else(|| {
                                BrainError::Integrity("system_envelope_byte_count_overflow".into())
                            })?;
                }
            }
            if entry.kind == SnapshotEntryKind::Directory {
                directory_count =
                    checked_increment(directory_count, "system_envelope_directory_count_overflow")?;
                if directory_count > MAX_CAPTURED_DIRECTORIES {
                    return Err(BrainError::Integrity(
                        "system_envelope_directory_limit_exceeded".into(),
                    ));
                }
            }
            entry_by_path.insert(entry.relative_path.as_path(), entry);
            previous_entry = Some(entry.relative_path.as_path());
        }
        if total_file_bytes != self.total_file_bytes {
            return Err(BrainError::Integrity(
                "system_envelope_total_bytes_mismatch".into(),
            ));
        }

        let mut previous_excluded: Option<&Path> = None;
        for excluded in &self.excluded_paths {
            if excluded.relative_path.is_root() {
                return Err(BrainError::Integrity(
                    "system_envelope_excluded_root_invalid".into(),
                ));
            }
            validate_observed_relative_path(excluded.relative_path.as_path())?;
            if let Some(previous) = previous_excluded {
                if previous >= excluded.relative_path.as_path() {
                    return Err(BrainError::Integrity(
                        "system_envelope_exclusions_not_canonical".into(),
                    ));
                }
            }
            if self.entries.iter().any(|entry| {
                !entry.relative_path.is_root()
                    && (entry.relative_path == excluded.relative_path
                        || entry
                            .relative_path
                            .as_path()
                            .starts_with(excluded.relative_path.as_path()))
            }) {
                return Err(BrainError::Integrity(
                    "system_envelope_exclusion_contains_entry".into(),
                ));
            }
            path_bytes = add_manifest_path_bytes(path_bytes, excluded.relative_path.as_path())?;
            previous_excluded = Some(excluded.relative_path.as_path());
        }
        for projection_root in &self.projection_roots {
            path_bytes = add_manifest_path_bytes(path_bytes, projection_root.as_path())?;
        }
        if path_bytes > MAX_MANIFEST_PATH_BYTES {
            return Err(BrainError::Integrity(
                "system_envelope_manifest_path_limit_exceeded".into(),
            ));
        }

        let projection_roots: BTreeSet<&Path> = self
            .projection_roots
            .iter()
            .map(DeclaredRelativePath::as_path)
            .collect();
        for projection_root in &projection_roots {
            if !entry_by_path.contains_key(projection_root) {
                return Err(BrainError::Integrity(
                    "system_envelope_projection_root_missing".into(),
                ));
            }
        }

        let mut children_by_directory: BTreeMap<&Path, Vec<&SnapshotEntry>> = BTreeMap::new();
        for entry in self.entries.iter().skip(1) {
            let is_projection_root = projection_roots.contains(entry.relative_path.as_path());
            if self.completeness == SnapshotCompleteness::DeclaredScopeOnly {
                if !self
                    .projection_roots
                    .iter()
                    .any(|root| entry.relative_path.as_path().starts_with(root.as_path()))
                {
                    return Err(BrainError::Integrity(
                        "system_envelope_entry_outside_projection".into(),
                    ));
                }
                if is_projection_root {
                    continue;
                }
            }
            let parent = entry.relative_path.as_path().parent().ok_or_else(|| {
                BrainError::Integrity("system_envelope_entry_parent_missing".into())
            })?;
            let parent_entry = entry_by_path.get(parent).ok_or_else(|| {
                BrainError::Integrity("system_envelope_entry_parent_not_captured".into())
            })?;
            if parent_entry.kind != SnapshotEntryKind::Directory {
                return Err(BrainError::Integrity(
                    "system_envelope_entry_parent_not_directory".into(),
                ));
            }
            children_by_directory.entry(parent).or_default().push(entry);
        }

        for excluded in &self.excluded_paths {
            if self.completeness == SnapshotCompleteness::DeclaredScopeOnly
                && !self
                    .projection_roots
                    .iter()
                    .any(|root| excluded.relative_path.as_path().starts_with(root.as_path()))
            {
                return Err(BrainError::Integrity(
                    "system_envelope_exclusion_outside_projection".into(),
                ));
            }
        }

        for directory in self
            .entries
            .iter()
            .filter(|entry| entry.kind == SnapshotEntryKind::Directory)
        {
            let child_commitments = if directory.relative_path.is_root()
                && self.completeness == SnapshotCompleteness::DeclaredScopeOnly
            {
                self.projection_roots
                    .iter()
                    .map(|root| {
                        let child = entry_by_path.get(root.as_path()).ok_or_else(|| {
                            BrainError::Integrity("system_envelope_projection_root_missing".into())
                        })?;
                        Ok((
                            root.as_path().to_path_buf(),
                            child.kind,
                            child.sha256.clone(),
                        ))
                    })
                    .collect::<BrainResult<Vec<_>>>()?
            } else {
                children_by_directory
                    .get(directory.relative_path.as_path())
                    .into_iter()
                    .flatten()
                    .map(|child| {
                        let name = child.relative_path.as_path().file_name().ok_or_else(|| {
                            BrainError::Integrity("system_envelope_child_name_missing".into())
                        })?;
                        Ok((PathBuf::from(name), child.kind, child.sha256.clone()))
                    })
                    .collect::<BrainResult<Vec<_>>>()?
            };
            if directory_digest(directory.relative_path.as_path(), &child_commitments)?
                != directory.sha256
            {
                return Err(BrainError::Integrity(
                    "system_envelope_directory_commitment_mismatch".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct SnapshotCounters {
    file_count: u64,
    total_file_bytes: u64,
    traversed_entries: u64,
    traversed_name_bytes: u64,
    captured_directories: u64,
    manifest_entries: u64,
    manifest_path_bytes: u64,
}

struct CaptureContext<'a> {
    selected: &'a BTreeSet<PathBuf>,
    exclusions: &'a BTreeSet<PathBuf>,
    entries: &'a mut Vec<SnapshotEntry>,
    excluded_paths: &'a mut Vec<ExcludedPath>,
    counters: &'a mut SnapshotCounters,
    visited_directories: &'a mut BTreeSet<(u64, u64)>,
    budget: &'a AcquisitionBudget,
    noise_policy: NoisePolicy,
}

struct InspectedObject {
    #[allow(dead_code)]
    fd: OwnedFd,
    stat: Stat,
}

struct PinnedRoot {
    fd: OwnedFd,
    stat: Stat,
}

/// One descriptor capability for the source tree used by every phase of an
/// acquisition.  A vault session additionally pins the private root that was
/// proven disjoint before any staging or CAS write.
pub(crate) struct SourceCaptureSession {
    source: PinnedRoot,
    private: Option<PinnedRoot>,
}

impl SourceCaptureSession {
    fn open(source_root: &Path) -> BrainResult<Self> {
        Ok(Self {
            source: open_confined_root(source_root, "acquisition_source_root")?,
            private: None,
        })
    }

    /// Pin both roots and prove their descriptor identities are disjoint.
    /// This operation is read-only and is deliberately completed before the
    /// caller may create even a staging directory.
    pub(crate) fn open_disjoint(source_root: &Path, private_root: &Path) -> BrainResult<Self> {
        let source = open_confined_root(source_root, "acquisition_source_root")?;
        let private = open_confined_root(private_root, "acquisition_private_root")?;
        ensure_disjoint_descriptors(&source, &private)?;
        Ok(Self {
            source,
            private: Some(private),
        })
    }

    fn source_root_fd(&self) -> &OwnedFd {
        &self.source.fd
    }

    fn current_source_root_stat(&self) -> BrainResult<Stat> {
        let current = fstat(&self.source.fd).map_err(rustix_error)?;
        ensure_same_object(&self.source.stat, &current, FileType::Directory)?;
        Ok(current)
    }

    fn ensure_source_root_identity(&self) -> BrainResult<()> {
        self.current_source_root_stat().map(|_| ())
    }

    /// Re-authenticate the ambient private path against the descriptor pinned
    /// before capture, then recheck actual ancestry.  The authority layer
    /// still performs its own confined open for each write; this check stops a
    /// renamed/replaced private path from silently changing the vault target.
    pub(crate) fn verify_private_root_binding(&self, private_root: &Path) -> BrainResult<()> {
        let expected = self
            .private
            .as_ref()
            .ok_or_else(|| BrainError::Integrity("acquisition_private_root_not_pinned".into()))?;
        let observed = open_confined_root(private_root, "acquisition_private_root")?;
        ensure_same_directory_binding(&expected.stat, &observed.stat)?;
        ensure_disjoint_descriptors(&self.source, &observed)
    }
}

fn open_confined_root(root: &Path, label: &str) -> BrainResult<PinnedRoot> {
    if !root.is_absolute()
        || root == Path::new("/")
        || root
            .components()
            .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        return Err(BrainError::Invalid(format!("{label}_invalid")));
    }
    let filesystem_root = open(
        Path::new("/"),
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(rustix_error)?;
    let relative_root = root
        .strip_prefix(Path::new("/"))
        .map_err(|_| BrainError::Invalid(format!("{label}_invalid")))?;
    let fd = openat2(
        &filesystem_root,
        relative_root,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
        ROOT_RESOLUTION,
    )
    .map_err(rustix_error)?;
    let stat = fstat(&fd).map_err(rustix_error)?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
        return Err(BrainError::Integrity(format!("{label}_invalid")));
    }
    Ok(PinnedRoot { fd, stat })
}

fn ensure_disjoint_descriptors(source: &PinnedRoot, private: &PinnedRoot) -> BrainResult<()> {
    if descriptor_is_ancestor(&source.stat, &private.fd)?
        || descriptor_is_ancestor(&private.stat, &source.fd)?
    {
        return Err(BrainError::Invalid(
            "acquisition_source_private_root_overlap".into(),
        ));
    }
    Ok(())
}

fn descriptor_is_ancestor(ancestor: &Stat, descendant: &OwnedFd) -> BrainResult<bool> {
    let mut current = rustix::io::dup(descendant).map_err(rustix_error)?;
    for _ in 0..MAX_DESCRIPTOR_ANCESTORS {
        let current_stat = fstat(&current).map_err(rustix_error)?;
        ensure_expected_object(&current_stat, FileType::Directory)?;
        if same_object_identity(ancestor, &current_stat) {
            return Ok(true);
        }
        let parent = openat(
            &current,
            Path::new(".."),
            OFlags::PATH | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(rustix_error)?;
        let parent_stat = fstat(&parent).map_err(rustix_error)?;
        ensure_expected_object(&parent_stat, FileType::Directory)?;
        if same_object_identity(&current_stat, &parent_stat) {
            return Ok(false);
        }
        current = parent;
    }
    Err(BrainError::Integrity(
        "acquisition_root_ancestry_limit_exceeded".into(),
    ))
}

fn ensure_same_directory_binding(before: &Stat, after: &Stat) -> BrainResult<()> {
    ensure_same_object(before, after, FileType::Directory)?;
    if before.st_mode != after.st_mode
        || before.st_uid != after.st_uid
        || before.st_gid != after.st_gid
    {
        return Err(BrainError::Integrity(
            "acquisition_private_root_binding_changed".into(),
        ));
    }
    Ok(())
}

fn same_object_identity(left: &Stat, right: &Stat) -> bool {
    left.st_dev == right.st_dev && left.st_ino == right.st_ino
}

fn validate_scope(scope: &AcquisitionScope) -> BrainResult<()> {
    match scope {
        AcquisitionScope::WholeProject => Ok(()),
        AcquisitionScope::DeclaredPaths { roots } => {
            if roots.is_empty() {
                return Err(BrainError::Invalid("acquisition_scope_roots_empty".into()));
            }
            if roots.len() > MAX_SCOPE_ROOTS {
                return Err(BrainError::Invalid(
                    "acquisition_scope_roots_limit_exceeded".into(),
                ));
            }
            let normalized = normalize_declared_roots(roots)?;
            if normalized != *roots {
                return Err(BrainError::Invalid(
                    "acquisition_scope_roots_not_canonical".into(),
                ));
            }
            Ok(())
        }
    }
}

fn normalize_scope(scope: AcquisitionScope) -> BrainResult<AcquisitionScope> {
    match scope {
        AcquisitionScope::WholeProject => Ok(AcquisitionScope::WholeProject),
        AcquisitionScope::DeclaredPaths { roots } => {
            if roots.is_empty() {
                return Err(BrainError::Invalid("acquisition_scope_roots_empty".into()));
            }
            if roots.len() > MAX_SCOPE_ROOTS {
                return Err(BrainError::Invalid(
                    "acquisition_scope_roots_limit_exceeded".into(),
                ));
            }
            Ok(AcquisitionScope::DeclaredPaths {
                roots: normalize_declared_roots(&roots)?,
            })
        }
    }
}

fn normalize_declared_roots(
    roots: &[DeclaredRelativePath],
) -> BrainResult<Vec<DeclaredRelativePath>> {
    let mut unique = BTreeSet::new();
    for root in roots {
        unique.insert(root.clone());
    }
    let mut normalized = Vec::with_capacity(unique.len());
    for root in unique {
        if normalized
            .iter()
            .any(|ancestor: &DeclaredRelativePath| root.as_path().starts_with(ancestor.as_path()))
        {
            continue;
        }
        normalized.push(root);
    }
    Ok(normalized)
}

fn selected_roots(scope: &AcquisitionScope) -> BrainResult<BTreeSet<PathBuf>> {
    match scope {
        AcquisitionScope::WholeProject => Ok(BTreeSet::from([PathBuf::new()])),
        AcquisitionScope::DeclaredPaths { roots } => Ok(roots
            .iter()
            .map(|root| root.as_path().to_path_buf())
            .collect()),
    }
}

fn canonical_declared_set(
    paths: &[DeclaredRelativePath],
    label: &str,
) -> BrainResult<BTreeSet<DeclaredRelativePath>> {
    let item_limit = if label == "acquisition_exclusion" {
        MAX_EXCLUSIONS
    } else {
        MAX_SCOPE_ROOTS
    };
    if paths.len() > item_limit {
        return Err(BrainError::Invalid(format!("{label}_limit_exceeded")));
    }
    let mut result = BTreeSet::new();
    for path in paths {
        if !result.insert(path.clone()) {
            return Err(BrainError::Invalid(format!("{label}_duplicate")));
        }
    }
    Ok(result)
}

fn validate_declared_relative_path(path: &Path, label: &str) -> BrainResult<()> {
    if path.as_os_str().is_empty() || !has_valid_relative_path_shape(path) {
        return Err(BrainError::Invalid(format!("{label}_invalid")));
    }
    Ok(())
}

fn has_valid_relative_path_shape(path: &Path) -> bool {
    if path.is_absolute() {
        return false;
    }
    if path.components().count() > MAX_RELATIVE_COMPONENTS
        || path.as_os_str().as_bytes().len() > MAX_RELATIVE_PATH_BYTES
        || path.as_os_str().as_bytes().contains(&0)
    {
        return false;
    }
    let mut canonical = PathBuf::new();
    for component in path.components() {
        let Component::Normal(name) = component else {
            return false;
        };
        canonical.push(name);
    }
    canonical.as_os_str().as_bytes() == path.as_os_str().as_bytes()
}

fn validate_scope_exclusion_separation(
    scope: &AcquisitionScope,
    exclusions: &BTreeSet<DeclaredRelativePath>,
) -> BrainResult<()> {
    if let AcquisitionScope::DeclaredPaths { roots } = scope {
        if let Some(excluded) = exclusions.iter().next() {
            if roots
                .iter()
                .any(|root| paths_overlap(root.as_path(), excluded.as_path()))
            {
                return Err(BrainError::Invalid(
                    "acquisition_scope_exclusion_overlap".into(),
                ));
            }
            return Err(BrainError::Invalid(
                "acquisition_exclusion_outside_declared_scope".into(),
            ));
        }
    }
    Ok(())
}

fn validate_non_overlapping_exclusions(
    exclusions: &BTreeSet<DeclaredRelativePath>,
) -> BrainResult<()> {
    for exclusion in exclusions {
        let mut parent = exclusion.as_path().parent();
        while let Some(candidate) = parent.filter(|path| !path.as_os_str().is_empty()) {
            if exclusions
                .iter()
                .any(|excluded| excluded.as_path() == candidate)
            {
                return Err(BrainError::Invalid("acquisition_exclusions_overlap".into()));
            }
            parent = candidate.parent();
        }
    }
    Ok(())
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left == right || left.starts_with(right) || right.starts_with(left)
}

fn capture_directory(
    relative: &Path,
    directory_fd: &OwnedFd,
    initial_stat: &Stat,
    depth: usize,
    context: &mut CaptureContext<'_>,
) -> BrainResult<Sha256Digest> {
    if depth > MAX_RELATIVE_COMPONENTS {
        return Err(BrainError::Invalid(
            "acquisition_directory_depth_limit_exceeded".into(),
        ));
    }
    ensure_expected_object(initial_stat, FileType::Directory)?;
    if !context
        .visited_directories
        .insert((initial_stat.st_dev, initial_stat.st_ino))
    {
        return Err(BrainError::Integrity(
            "acquisition_directory_cycle_or_alias".into(),
        ));
    }
    context.counters.captured_directories = checked_increment(
        context.counters.captured_directories,
        "acquisition_directory_count_overflow",
    )?;
    if context.counters.captured_directories > MAX_CAPTURED_DIRECTORIES {
        return Err(BrainError::Invalid(
            "acquisition_directory_limit_exceeded".into(),
        ));
    }
    admit_manifest_path(relative, context)?;

    let mut directory = Dir::read_from(directory_fd).map_err(rustix_error)?;
    let mut names = Vec::new();
    let mut directory_children = 0_u64;
    for entry in &mut directory {
        let entry = entry.map_err(rustix_error)?;
        let raw_name = entry.file_name().to_bytes();
        if raw_name == b"." || raw_name == b".." {
            continue;
        }
        directory_children = checked_increment(
            directory_children,
            "acquisition_directory_child_count_overflow",
        )?;
        context.counters.traversed_entries = checked_increment(
            context.counters.traversed_entries,
            "acquisition_traversed_entry_count_overflow",
        )?;
        context.counters.traversed_name_bytes = context
            .counters
            .traversed_name_bytes
            .checked_add(u64::try_from(raw_name.len()).map_err(|_| {
                BrainError::Integrity("acquisition_traversed_name_size_overflow".into())
            })?)
            .ok_or_else(|| {
                BrainError::Integrity("acquisition_traversed_name_size_overflow".into())
            })?;
        if directory_children > MAX_DIRECTORY_CHILDREN
            || context.counters.traversed_entries > MAX_TRAVERSED_ENTRIES
            || context.counters.traversed_name_bytes > MAX_TRAVERSED_NAME_BYTES
        {
            return Err(BrainError::Invalid(
                "acquisition_traversal_limit_exceeded".into(),
            ));
        }
        let name = std::ffi::OsString::from_vec(raw_name.to_vec());
        let child_relative = relative.join(&name);

        // Scope pruning is performed from the lexical name before any open or
        // metadata lookup.  An unrelated sibling symlink/device therefore has
        // no influence on a DeclaredPaths observation.
        if is_selected(&child_relative, context.selected) {
            validate_observed_relative_path(&child_relative)?;
            names.push(name);
        }
    }
    names.sort();

    let mut child_commitments = Vec::new();
    for name in names {
        let child_relative = relative.join(&name);
        if let Some(reason) = exclusion_reason(
            &child_relative,
            context.selected,
            context.exclusions,
            context.noise_policy,
        ) {
            admit_excluded_path(child_relative, reason, context)?;
            continue;
        }

        let inspected = inspect_beneath(directory_fd, Path::new(&name))?;
        match FileType::from_raw_mode(inspected.stat.st_mode) {
            FileType::Directory => {
                let (child_fd, opened_stat) =
                    open_directory_beneath(directory_fd, Path::new(&name), &inspected.stat)?;
                let digest = capture_directory(
                    &child_relative,
                    &child_fd,
                    &opened_stat,
                    depth + 1,
                    context,
                )?;
                child_commitments.push((PathBuf::from(name), SnapshotEntryKind::Directory, digest));
            }
            FileType::RegularFile => {
                let digest = capture_file(&child_relative, &inspected, context)?;
                child_commitments.push((PathBuf::from(name), SnapshotEntryKind::File, digest));
            }
            FileType::Symlink => {
                return Err(BrainError::Integrity(
                    "acquisition_source_symlink_forbidden".into(),
                ));
            }
            _ => {
                return Err(BrainError::Integrity(
                    "acquisition_source_special_file_forbidden".into(),
                ));
            }
        }
    }

    let final_stat = fstat(directory_fd).map_err(rustix_error)?;
    ensure_stable_object(initial_stat, &final_stat, FileType::Directory)?;
    let digest = directory_digest(relative, &child_commitments)?;
    context.entries.push(SnapshotEntry {
        relative_path: ObservedRelativePath::parse(relative.to_path_buf())?,
        kind: SnapshotEntryKind::Directory,
        byte_len: 0,
        executable: false,
        sha256: digest.clone(),
    });
    Ok(digest)
}

fn capture_file(
    relative: &Path,
    inspected: &InspectedObject,
    context: &mut CaptureContext<'_>,
) -> BrainResult<Sha256Digest> {
    ensure_expected_object(&inspected.stat, FileType::RegularFile)?;
    let next_file_count = checked_increment(
        context.counters.file_count,
        "acquisition_file_count_overflow",
    )?;
    if next_file_count > context.budget.max_files {
        return Err(BrainError::Invalid("acquisition_budget_exceeded".into()));
    }

    // Reopen the already-pinned O_PATH descriptor, rather than resolving the
    // donor pathname a second time.  A concurrent swap to a FIFO or device
    // therefore cannot trigger a blocking or side-effecting open.
    let descriptor_path = PathBuf::from(format!("/proc/self/fd/{}", inspected.fd.as_raw_fd()));
    let data_fd = open(
        &descriptor_path,
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(rustix_error)?;
    let opened_stat = fstat(&data_fd).map_err(rustix_error)?;
    ensure_same_object(&inspected.stat, &opened_stat, FileType::RegularFile)?;
    let expected_len = u64::try_from(opened_stat.st_size)
        .map_err(|_| BrainError::Integrity("acquisition_file_size_invalid".into()))?;
    let remaining = context
        .budget
        .max_total_bytes
        .checked_sub(context.counters.total_file_bytes)
        .ok_or_else(|| BrainError::Integrity("acquisition_byte_budget_underflow".into()))?;
    if expected_len > remaining {
        return Err(BrainError::Invalid("acquisition_budget_exceeded".into()));
    }

    let executable = opened_stat.st_mode & 0o111 != 0;
    let mut hasher = Sha256::new();
    hasher.update(FILE_DOMAIN);
    hash_frame(&mut hasher, relative.as_os_str().as_bytes())?;
    hasher.update([u8::from(executable)]);
    hasher.update(expected_len.to_be_bytes());

    let mut file = fs::File::from(data_fd);
    let mut buffer = [0_u8; STREAM_BUFFER_BYTES];
    let mut actual_len = 0_u64;
    loop {
        // Read at most the remaining declared length plus one sentinel byte.
        // This proves EOF without ever materializing or consuming an
        // unbounded post-stat growth in memory.
        let remaining_declared = expected_len.saturating_sub(actual_len);
        let read_limit = remaining_declared
            .saturating_add(1)
            .min(STREAM_BUFFER_BYTES as u64) as usize;
        let count = file.read(&mut buffer[..read_limit])?;
        if count == 0 {
            break;
        }
        actual_len = actual_len
            .checked_add(count as u64)
            .ok_or_else(|| BrainError::Integrity("acquisition_byte_count_overflow".into()))?;
        if actual_len > remaining {
            return Err(BrainError::Invalid("acquisition_budget_exceeded".into()));
        }
        if actual_len > expected_len {
            return Err(BrainError::Integrity(
                "acquisition_file_length_changed_during_capture".into(),
            ));
        }
        hasher.update(&buffer[..count]);
    }
    let final_stat = fstat(&file).map_err(rustix_error)?;
    ensure_stable_object(&opened_stat, &final_stat, FileType::RegularFile)?;
    if actual_len != expected_len {
        return Err(BrainError::Integrity(
            "acquisition_file_length_changed_during_capture".into(),
        ));
    }

    let digest = finalize_digest(hasher)?;
    context.counters.file_count = next_file_count;
    context.counters.total_file_bytes =
        context
            .counters
            .total_file_bytes
            .checked_add(actual_len)
            .ok_or_else(|| BrainError::Integrity("acquisition_byte_count_overflow".into()))?;
    admit_manifest_path(relative, context)?;
    context.entries.push(SnapshotEntry {
        relative_path: ObservedRelativePath::parse(relative.to_path_buf())?,
        kind: SnapshotEntryKind::File,
        byte_len: actual_len,
        executable,
        sha256: digest.clone(),
    });
    Ok(digest)
}

/// Stream one manifest file through the same descriptor-bound source boundary
/// used by capture.  This deliberately has crate visibility only: consumers
/// receive retained CAS objects, never an ambient donor-file reading API.
///
/// The caller supplies the byte length committed by an already verified
/// envelope.  A replacement, type change, growth, shrink or metadata change
/// while copying is rejected.  The helper does not assert an atomic snapshot
/// across *multiple* files; that stronger property requires a source-specific
/// snapshot adapter and must not be inferred from `VaultRetained`.
pub(crate) fn copy_manifest_file_descriptor_bound(
    session: &SourceCaptureSession,
    relative: &Path,
    expected_len: u64,
    destination: &mut fs::File,
) -> BrainResult<Sha256Digest> {
    session.ensure_source_root_identity()?;
    let inspected = inspect_beneath(session.source_root_fd(), relative)?;
    ensure_expected_object(&inspected.stat, FileType::RegularFile)?;
    let descriptor_path = PathBuf::from(format!("/proc/self/fd/{}", inspected.fd.as_raw_fd()));
    let data_fd = open(
        &descriptor_path,
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(rustix_error)?;
    let opened_stat = fstat(&data_fd).map_err(rustix_error)?;
    ensure_same_object(&inspected.stat, &opened_stat, FileType::RegularFile)?;
    let opened_len = u64::try_from(opened_stat.st_size)
        .map_err(|_| BrainError::Integrity("acquisition_file_size_invalid".into()))?;
    if opened_len != expected_len {
        return Err(BrainError::Integrity(
            "content_vault_source_length_envelope_mismatch".into(),
        ));
    }
    let executable = opened_stat.st_mode & 0o111 != 0;
    let mut entry_hasher = Sha256::new();
    entry_hasher.update(FILE_DOMAIN);
    hash_frame(&mut entry_hasher, relative.as_os_str().as_bytes())?;
    entry_hasher.update([u8::from(executable)]);
    entry_hasher.update(expected_len.to_be_bytes());
    let mut source = fs::File::from(data_fd);
    let mut buffer = [0_u8; STREAM_BUFFER_BYTES];
    let mut copied = 0_u64;
    loop {
        let remaining = expected_len.saturating_sub(copied);
        let read_limit = remaining.saturating_add(1).min(STREAM_BUFFER_BYTES as u64) as usize;
        let count = source.read(&mut buffer[..read_limit])?;
        if count == 0 {
            break;
        }
        copied = copied
            .checked_add(count as u64)
            .ok_or_else(|| BrainError::Integrity("content_vault_byte_count_overflow".into()))?;
        if copied > expected_len {
            return Err(BrainError::Integrity(
                "content_vault_source_length_changed_during_capture".into(),
            ));
        }
        entry_hasher.update(&buffer[..count]);
        use std::io::Write as _;
        destination.write_all(&buffer[..count])?;
    }
    let final_stat = fstat(&source).map_err(rustix_error)?;
    ensure_stable_object(&opened_stat, &final_stat, FileType::RegularFile)?;
    if copied != expected_len {
        return Err(BrainError::Integrity(
            "content_vault_source_length_changed_during_capture".into(),
        ));
    }
    session.ensure_source_root_identity()?;
    finalize_digest(entry_hasher)
}

fn is_selected(path: &Path, selected: &BTreeSet<PathBuf>) -> bool {
    // A path participates in the projection when it is:
    // - below a selected root and must be captured; or
    // - an ancestor of a selected root and must be traversed to reach it.
    // The empty root denotes WholeProject.  Keeping both directions explicit
    // prevents the subtle mistake of either omitting ancestors or admitting
    // unrelated siblings for DeclaredPaths.
    selected.iter().any(|root| {
        root.as_os_str().is_empty()
            || path == root
            || path.starts_with(root)
            || root.starts_with(path)
    })
}

fn exclusion_reason(
    path: &Path,
    selected: &BTreeSet<PathBuf>,
    exclusions: &BTreeSet<PathBuf>,
    noise_policy: NoisePolicy,
) -> Option<ExclusionReason> {
    if exclusions
        .iter()
        .any(|excluded| path == excluded || path.starts_with(excluded))
    {
        return Some(ExclusionReason::ExplicitRequest);
    }
    if noise_policy == NoisePolicy::ConservativeGeneratedArtifacts {
        let policy_relative = selected.iter().find_map(|root| {
            if root.as_os_str().is_empty() {
                Some(path)
            } else if path == root {
                None
            } else {
                path.strip_prefix(root).ok()
            }
        });
        if policy_relative.is_some_and(|candidate| {
            !candidate.as_os_str().is_empty() && is_conservative_generated_artifact(candidate)
        }) {
            return Some(ExclusionReason::GeneratedArtifactPolicy);
        }
    }
    None
}

fn is_conservative_generated_artifact(path: &Path) -> bool {
    const GENERATED_DIRECTORIES: &[&str] = &[
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "target",
        "CMakeFiles",
        "bazel-out",
        "buck-out",
    ];
    const GENERATED_SUFFIXES: &[&str] = &[
        ".pyc", ".pyo", ".class", ".o", ".obj", ".a", ".so", ".dylib", ".dll", ".exe", ".log",
        ".tmp", ".swp", ".bak", "~",
    ];
    path.components().any(|component| match component {
        Component::Normal(name) => GENERATED_DIRECTORIES
            .iter()
            .any(|candidate| name == *candidate),
        _ => false,
    }) || path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            GENERATED_SUFFIXES
                .iter()
                .any(|suffix| name.ends_with(suffix))
        })
}

fn directory_digest(
    relative: &Path,
    children: &[(PathBuf, SnapshotEntryKind, Sha256Digest)],
) -> BrainResult<Sha256Digest> {
    let mut hasher = Sha256::new();
    hasher.update(DIRECTORY_DOMAIN);
    hash_frame(&mut hasher, relative.as_os_str().as_bytes())?;
    hasher.update(
        u64::try_from(children.len())
            .map_err(|_| BrainError::Integrity("acquisition_child_count_overflow".into()))?
            .to_be_bytes(),
    );
    for (name, kind, digest) in children {
        hash_frame(&mut hasher, name.as_os_str().as_bytes())?;
        hasher.update([match kind {
            SnapshotEntryKind::File => 1,
            SnapshotEntryKind::Directory => 2,
        }]);
        hash_frame(&mut hasher, digest.as_str().as_bytes())?;
    }
    finalize_digest(hasher)
}

fn hash_frame(hasher: &mut Sha256, bytes: &[u8]) -> BrainResult<()> {
    let length = u64::try_from(bytes.len())
        .map_err(|_| BrainError::Integrity("acquisition_hash_frame_too_large".into()))?;
    hasher.update(length.to_be_bytes());
    hasher.update(bytes);
    Ok(())
}

fn finalize_digest(hasher: Sha256) -> BrainResult<Sha256Digest> {
    Sha256Digest::parse(format!("{:x}", hasher.finalize()))
}

fn domain_digest(domain: &[u8], bytes: &[u8]) -> Sha256Digest {
    Sha256Digest::digest_domain(domain, bytes)
}

fn inspect_beneath(parent_fd: &OwnedFd, path: &Path) -> BrainResult<InspectedObject> {
    let fd = openat2(
        parent_fd,
        path,
        OFlags::PATH | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
        SOURCE_TREE_RESOLUTION,
    )
    .map_err(rustix_error)?;
    let stat = fstat(&fd).map_err(rustix_error)?;
    Ok(InspectedObject { fd, stat })
}

fn open_directory_beneath(
    parent_fd: &OwnedFd,
    path: &Path,
    inspected_stat: &Stat,
) -> BrainResult<(OwnedFd, Stat)> {
    ensure_expected_object(inspected_stat, FileType::Directory)?;
    let fd = openat2(
        parent_fd,
        path,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
        SOURCE_TREE_RESOLUTION,
    )
    .map_err(rustix_error)?;
    let opened_stat = fstat(&fd).map_err(rustix_error)?;
    ensure_same_object(inspected_stat, &opened_stat, FileType::Directory)?;
    Ok((fd, opened_stat))
}

fn ensure_expected_object(stat: &Stat, expected: FileType) -> BrainResult<()> {
    if FileType::from_raw_mode(stat.st_mode) != expected {
        return Err(BrainError::Integrity(
            "acquisition_source_object_type_changed".into(),
        ));
    }
    Ok(())
}

fn ensure_same_object(before: &Stat, after: &Stat, expected: FileType) -> BrainResult<()> {
    ensure_expected_object(before, expected)?;
    ensure_expected_object(after, expected)?;
    if !same_object_identity(before, after) {
        return Err(BrainError::Integrity(
            "acquisition_source_object_replaced".into(),
        ));
    }
    Ok(())
}

fn ensure_stable_object(before: &Stat, after: &Stat, expected: FileType) -> BrainResult<()> {
    ensure_same_object(before, after, expected)?;
    if before.st_mode != after.st_mode
        || before.st_size != after.st_size
        || before.st_mtime != after.st_mtime
        || before.st_mtime_nsec != after.st_mtime_nsec
        || before.st_ctime != after.st_ctime
        || before.st_ctime_nsec != after.st_ctime_nsec
    {
        return Err(BrainError::Integrity(
            "acquisition_source_object_mutated_during_capture".into(),
        ));
    }
    Ok(())
}

fn admit_manifest_path(path: &Path, context: &mut CaptureContext<'_>) -> BrainResult<()> {
    context.counters.manifest_entries = checked_increment(
        context.counters.manifest_entries,
        "acquisition_manifest_entry_count_overflow",
    )?;
    context.counters.manifest_path_bytes =
        add_manifest_path_bytes(context.counters.manifest_path_bytes, path)?;
    if context.counters.manifest_entries > MAX_MANIFEST_ENTRIES
        || context.counters.manifest_path_bytes > MAX_MANIFEST_PATH_BYTES
    {
        return Err(BrainError::Invalid(
            "acquisition_manifest_limit_exceeded".into(),
        ));
    }
    Ok(())
}

fn add_manifest_path_bytes(current: u64, path: &Path) -> BrainResult<u64> {
    let path_bytes = u64::try_from(path.as_os_str().as_bytes().len())
        .map_err(|_| BrainError::Integrity("acquisition_manifest_path_size_overflow".into()))?;
    current
        .checked_add(path_bytes)
        .ok_or_else(|| BrainError::Integrity("acquisition_manifest_path_size_overflow".into()))
}

fn admit_excluded_path(
    relative_path: PathBuf,
    reason: ExclusionReason,
    context: &mut CaptureContext<'_>,
) -> BrainResult<()> {
    admit_manifest_path(&relative_path, context)?;
    if context.excluded_paths.len() >= MAX_EXCLUSIONS {
        return Err(BrainError::Invalid(
            "acquisition_excluded_path_limit_exceeded".into(),
        ));
    }
    context.excluded_paths.push(ExcludedPath {
        relative_path: ObservedRelativePath::parse(relative_path)?,
        reason,
    });
    Ok(())
}

fn validate_requested_coverage(
    entries: &[SnapshotEntry],
    selected: &BTreeSet<PathBuf>,
) -> BrainResult<()> {
    for requested in selected {
        if !entries
            .iter()
            .any(|entry| entry.relative_path.as_path() == requested)
        {
            return Err(BrainError::Integrity(
                "acquisition_declared_root_not_captured".into(),
            ));
        }
    }
    Ok(())
}

fn validate_explicit_exclusion_coverage(
    observed: &[ExcludedPath],
    requested: &BTreeSet<PathBuf>,
) -> BrainResult<()> {
    let observed_explicit: BTreeSet<PathBuf> = observed
        .iter()
        .filter(|entry| entry.reason == ExclusionReason::ExplicitRequest)
        .map(|entry| entry.relative_path.as_path().to_path_buf())
        .collect();
    if observed_explicit != *requested {
        return Err(BrainError::Integrity(
            "acquisition_explicit_exclusion_not_observed".into(),
        ));
    }
    Ok(())
}

fn validate_observed_relative_path(path: &Path) -> BrainResult<()> {
    if !path.as_os_str().is_empty() && !has_valid_relative_path_shape(path) {
        return Err(BrainError::Invalid(
            "acquisition_observed_path_not_portable".into(),
        ));
    }
    Ok(())
}

fn checked_increment(value: u64, label: &str) -> BrainResult<u64> {
    value
        .checked_add(1)
        .ok_or_else(|| BrainError::Integrity(label.into()))
}

fn rustix_error(error: rustix::io::Errno) -> BrainError {
    BrainError::Io(std::io::Error::from_raw_os_error(error.raw_os_error()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declared(roots: Vec<PathBuf>) -> AcquisitionScope {
        AcquisitionScope::declared_paths(roots).unwrap()
    }

    fn request(scope: AcquisitionScope) -> AcquisitionRequest {
        let exclusions = if matches!(&scope, AcquisitionScope::WholeProject) {
            vec![PathBuf::from("cache")]
        } else {
            vec![]
        };
        AcquisitionRequest::new(
            AcquisitionId::parse("universal-acquisition.v1").unwrap(),
            scope,
            RequestedResidency::BestVerified,
            NoisePolicy::ConservativeGeneratedArtifacts,
            AcquisitionBudget {
                max_files: 32,
                max_total_bytes: 1 << 20,
            },
            exclusions,
        )
        .unwrap()
    }

    #[test]
    fn whole_tree_capture_is_reproducible_and_detects_mutation() {
        let temp = std::env::temp_dir().join(format!("tidex-acquisition-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(temp.join("src")).unwrap();
        fs::create_dir_all(temp.join("cache")).unwrap();
        fs::write(temp.join("src/main.rs"), b"fn main() {}\n").unwrap();
        fs::write(temp.join("Cargo.toml"), b"[package]\nname='x'\n").unwrap();
        fs::write(temp.join("cache/noise"), b"untracked").unwrap();
        let request = request(AcquisitionScope::WholeProject);
        let envelope = SystemEnvelope::capture(&temp, &request).unwrap();
        assert_eq!(envelope.completeness, SnapshotCompleteness::WholeTree);
        assert!(envelope
            .entries
            .iter()
            .any(|entry| entry.relative_path() == Path::new("Cargo.toml")));
        assert!(!envelope
            .entries
            .iter()
            .any(|entry| entry.relative_path() == Path::new("cache/noise")));
        envelope.verify_against(&temp, &request).unwrap();
        fs::write(
            temp.join("src/main.rs"),
            b"fn main() { println!(\"changed\"); }\n",
        )
        .unwrap();
        assert!(envelope.verify_against(&temp, &request).is_err());
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn declared_scope_is_normalized_and_never_claims_dependency_closure() {
        let normalized = AcquisitionRequest::new(
            AcquisitionId::parse("acquisition-2").unwrap(),
            declared(vec![
                PathBuf::from("src/nested"),
                PathBuf::from("src"),
                PathBuf::from("config"),
                PathBuf::from("src"),
            ]),
            RequestedResidency::WeightsOnly,
            NoisePolicy::ConservativeGeneratedArtifacts,
            AcquisitionBudget {
                max_files: 1,
                max_total_bytes: 1,
            },
            vec![],
        )
        .unwrap();
        assert_eq!(
            normalized.scope,
            declared(vec![PathBuf::from("config"), PathBuf::from("src")])
        );
        let temp = std::env::temp_dir().join(format!("tidex-scope-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(temp.join("src")).unwrap();
        fs::create_dir_all(temp.join("cache")).unwrap();
        fs::write(temp.join("src/lib.rs"), b"pub fn x() {}\n").unwrap();
        fs::write(temp.join("outside.txt"), b"outside\n").unwrap();
        let request = request(declared(vec![PathBuf::from("src")]));
        let envelope = SystemEnvelope::capture(&temp, &request).unwrap();
        assert_eq!(
            envelope.completeness,
            SnapshotCompleteness::DeclaredScopeOnly
        );
        assert!(!envelope
            .entries
            .iter()
            .any(|entry| entry.relative_path() == Path::new("outside.txt")));
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn declared_roots_must_exist_and_may_not_overlap_exclusions() {
        let overlap = AcquisitionRequest::new(
            AcquisitionId::parse("scope-overlap.v1").unwrap(),
            declared(vec![PathBuf::from("src")]),
            RequestedResidency::BestVerified,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 8,
                max_total_bytes: 1 << 20,
            },
            vec![PathBuf::from("src/generated")],
        );
        assert!(matches!(overlap, Err(BrainError::Invalid(_))));

        let temp = std::env::temp_dir().join(format!("tidex-missing-scope-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        let request = request(declared(vec![PathBuf::from("does-not-exist")]));
        assert!(SystemEnvelope::capture(&temp, &request).is_err());

        let unmatched_exclusion = AcquisitionRequest::new(
            AcquisitionId::parse("missing-exclusion.v1").unwrap(),
            AcquisitionScope::WholeProject,
            RequestedResidency::BestVerified,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 8,
                max_total_bytes: 1 << 20,
            },
            vec![PathBuf::from("absent-cache")],
        )
        .unwrap();
        assert!(matches!(
            SystemEnvelope::capture(&temp, &unmatched_exclusion),
            Err(BrainError::Integrity(message))
                if message == "acquisition_explicit_exclusion_not_observed"
        ));
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn declared_scope_prunes_unrelated_symlinks_but_rejects_a_selected_one() {
        use std::os::unix::fs::symlink;

        let temp =
            std::env::temp_dir().join(format!("tidex-pruned-symlink-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(temp.join("src")).unwrap();
        fs::write(temp.join("src/lib.rs"), b"pub fn x() {}\n").unwrap();
        symlink("/", temp.join("unrelated-link")).unwrap();

        let selected_source = request(declared(vec![PathBuf::from("src")]));
        SystemEnvelope::capture(&temp, &selected_source).unwrap();

        let selected_link = request(declared(vec![PathBuf::from("unrelated-link")]));
        assert!(SystemEnvelope::capture(&temp, &selected_link).is_err());
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn byte_budget_is_applied_to_the_open_file_stream() {
        let temp = std::env::temp_dir().join(format!("tidex-stream-budget-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        fs::write(temp.join("payload.bin"), b"four").unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse("stream-budget.v1").unwrap(),
            AcquisitionScope::WholeProject,
            RequestedResidency::BestVerified,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 1,
                max_total_bytes: 3,
            },
            vec![],
        )
        .unwrap();
        assert!(matches!(
            SystemEnvelope::capture(&temp, &request),
            Err(BrainError::Invalid(message)) if message == "acquisition_budget_exceeded"
        ));
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn selected_fifo_is_rejected_without_reading_it() {
        let temp = std::env::temp_dir().join(format!("tidex-fifo-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        rustix::fs::mkfifoat(rustix::fs::CWD, temp.join("pipe"), Mode::RWXU).unwrap();
        let request = request(declared(vec![PathBuf::from("pipe")]));
        assert!(SystemEnvelope::capture(&temp, &request).is_err());
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn manifest_verification_rejects_self_consistent_but_false_structure() {
        let temp =
            std::env::temp_dir().join(format!("tidex-manifest-shape-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(temp.join("src")).unwrap();
        fs::create_dir_all(temp.join("cache")).unwrap();
        fs::write(temp.join("src/lib.rs"), b"pub fn x() {}\n").unwrap();
        let request = request(AcquisitionScope::WholeProject);
        let envelope = SystemEnvelope::capture(&temp, &request).unwrap();

        let mut false_total = envelope.clone();
        false_total.total_file_bytes = false_total.total_file_bytes.saturating_add(1);
        false_total.manifest_sha256 = false_total.calculate_digest().unwrap();
        assert!(matches!(
            false_total.verify_manifest(),
            Err(BrainError::Integrity(message)) if message == "system_envelope_total_bytes_mismatch"
        ));

        let mut false_merkle = envelope;
        let root = false_merkle
            .entries
            .iter_mut()
            .find(|entry| entry.relative_path.is_root())
            .unwrap();
        root.sha256 = Sha256Digest::digest_bytes(b"forged-root");
        false_merkle.manifest_sha256 = false_merkle.calculate_digest().unwrap();
        assert!(matches!(
            false_merkle.verify_manifest(),
            Err(BrainError::Integrity(message))
                if message == "system_envelope_directory_commitment_mismatch"
        ));
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn file_commitment_is_one_framed_sha256_round() {
        let temp = std::env::temp_dir().join(format!("tidex-file-digest-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        fs::write(temp.join("a.bin"), b"abc").unwrap();
        let request = request(declared(vec![PathBuf::from("a.bin")]));
        let envelope = SystemEnvelope::capture(&temp, &request).unwrap();
        let observed = envelope
            .entries
            .iter()
            .find(|entry| entry.relative_path() == Path::new("a.bin"))
            .unwrap();

        let mut hasher = Sha256::new();
        hasher.update(FILE_DOMAIN);
        hasher.update(5_u64.to_be_bytes());
        hasher.update(b"a.bin");
        hasher.update([0]);
        hasher.update(3_u64.to_be_bytes());
        hasher.update(b"abc");
        let first_round = hasher.finalize();
        let expected = Sha256Digest::parse(format!("{first_round:x}")).unwrap();
        assert_eq!(observed.sha256, expected);
        assert_ne!(observed.sha256, Sha256Digest::digest_bytes(&first_round));
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn source_root_rejects_a_symlink_in_any_path_component() {
        use std::os::unix::fs::symlink;

        let temp = std::env::temp_dir().join(format!("tidex-root-link-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(temp.join("real/source")).unwrap();
        symlink(temp.join("real"), temp.join("alias")).unwrap();
        let request = request(AcquisitionScope::WholeProject);
        assert!(SystemEnvelope::capture(&temp.join("alias/source"), &request).is_err());
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn sealed_paths_round_trip_non_utf8_without_loss() {
        let temp = std::env::temp_dir().join(format!("tidex-non-utf8-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        let raw_name = std::ffi::OsString::from_vec(vec![b'm', 0xff, b'd']);
        let relative = PathBuf::from(&raw_name);
        fs::write(temp.join(&relative), b"bytes").unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse("non-utf8-path.v1").unwrap(),
            declared(vec![relative.clone()]),
            RequestedResidency::BestVerified,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 1,
                max_total_bytes: 16,
            },
            vec![],
        )
        .unwrap();
        let request_round_trip: AcquisitionRequest =
            serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap();
        request_round_trip.verify().unwrap();
        assert_eq!(request_round_trip.scope, request.scope);

        let envelope = SystemEnvelope::capture(&temp, &request_round_trip).unwrap();
        let envelope_wire = serde_json::to_value(&envelope).unwrap();
        assert_eq!(
            envelope_wire.pointer("/projection_roots/0"),
            Some(&serde_json::json!({ "unix_bytes_hex": "6dff64" }))
        );
        assert_eq!(
            envelope_wire.pointer("/entries/0/relative_path"),
            Some(&serde_json::json!({ "unix_bytes_hex": "" }))
        );
        let envelope_round_trip: SystemEnvelope =
            serde_json::from_slice(&serde_json::to_vec(&envelope).unwrap()).unwrap();
        envelope_round_trip.verify_manifest().unwrap();
        assert!(envelope_round_trip
            .entries
            .iter()
            .any(|entry| entry.relative_path() == relative));
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn path_wire_is_canonical_and_rejects_malformed_hex() {
        assert_eq!(
            path_wire::decode("7372632f6c69622e7273").unwrap(),
            PathBuf::from("src/lib.rs")
        );
        for malformed in ["0", "GG", "2F", "ffFf", "xy"] {
            assert_eq!(path_wire::decode(malformed), Err("unix_path_hex_invalid"));
        }
        let encoded = path_wire::encode(Path::new("src/lib.rs"));
        let declared_path = DeclaredRelativePath::parse(PathBuf::from("src/lib.rs")).unwrap();
        let observed = ObservedRelativePath::parse(PathBuf::from("src/lib.rs")).unwrap();
        assert_eq!(
            serde_json::to_vec(&declared_path).unwrap(),
            serde_json::to_vec(&encoded).unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&observed).unwrap(),
            serde_json::to_vec(&encoded).unwrap()
        );

        let request = AcquisitionRequest::new(
            AcquisitionId::parse("wire-contract.v1").unwrap(),
            declared(vec![PathBuf::from("src")]),
            RequestedResidency::BestVerified,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 1,
                max_total_bytes: 1,
            },
            vec![],
        )
        .unwrap();
        let mut value = serde_json::to_value(&request).unwrap();
        assert_eq!(
            value.pointer("/scope/roots/0"),
            Some(&serde_json::json!({ "unix_bytes_hex": "737263" }))
        );

        value["scope"]["roots"][0]["unix_bytes_hex"] = serde_json::json!("73726C");
        assert!(serde_json::from_value::<AcquisitionRequest>(value.clone()).is_err());
        value["scope"]["roots"][0]["unix_bytes_hex"] = serde_json::json!("737263");
        value["scope"]["roots"][0]["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<AcquisitionRequest>(value).is_err());
    }

    #[test]
    fn declared_relative_path_rejects_lexical_aliases_at_construction_and_serde() {
        for invalid in ["", "/absolute", "../escape", "./alias", "a//b", "a/"] {
            assert!(
                DeclaredRelativePath::parse(PathBuf::from(invalid)).is_err(),
                "unexpectedly admitted {invalid:?}"
            );
        }
        assert_eq!(
            DeclaredRelativePath::parse(PathBuf::from("a/b"))
                .unwrap()
                .as_path(),
            Path::new("a/b")
        );

        let request = AcquisitionRequest::new(
            AcquisitionId::parse("typed-path-serde.v1").unwrap(),
            AcquisitionScope::WholeProject,
            RequestedResidency::BestVerified,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 1,
                max_total_bytes: 1,
            },
            vec![PathBuf::from("cache")],
        )
        .unwrap();
        let mut value = serde_json::to_value(request).unwrap();
        value["exclusions"][0]["unix_bytes_hex"] = serde_json::json!("2f6162736f6c757465");
        assert!(serde_json::from_value::<AcquisitionRequest>(value).is_err());
    }

    #[test]
    fn hash_protocol_separates_path_name_kind_and_order() {
        let first = Sha256Digest::digest_bytes(b"first");
        let second = Sha256Digest::digest_bytes(b"second");
        let baseline = directory_digest(
            Path::new("root"),
            &[
                (PathBuf::from("ab"), SnapshotEntryKind::File, first.clone()),
                (
                    PathBuf::from("c"),
                    SnapshotEntryKind::Directory,
                    second.clone(),
                ),
            ],
        )
        .unwrap();
        let changed_framing = directory_digest(
            Path::new("root"),
            &[
                (PathBuf::from("a"), SnapshotEntryKind::File, first.clone()),
                (
                    PathBuf::from("bc"),
                    SnapshotEntryKind::Directory,
                    second.clone(),
                ),
            ],
        )
        .unwrap();
        let changed_kind = directory_digest(
            Path::new("root"),
            &[
                (
                    PathBuf::from("ab"),
                    SnapshotEntryKind::Directory,
                    first.clone(),
                ),
                (
                    PathBuf::from("c"),
                    SnapshotEntryKind::Directory,
                    second.clone(),
                ),
            ],
        )
        .unwrap();
        let changed_order = directory_digest(
            Path::new("root"),
            &[
                (PathBuf::from("c"), SnapshotEntryKind::Directory, second),
                (PathBuf::from("ab"), SnapshotEntryKind::File, first),
            ],
        )
        .unwrap();
        let changed_path = directory_digest(
            Path::new("other-root"),
            &[
                (
                    PathBuf::from("ab"),
                    SnapshotEntryKind::File,
                    Sha256Digest::digest_bytes(b"first"),
                ),
                (
                    PathBuf::from("c"),
                    SnapshotEntryKind::Directory,
                    Sha256Digest::digest_bytes(b"second"),
                ),
            ],
        )
        .unwrap();
        for distinct in [changed_framing, changed_kind, changed_order, changed_path] {
            assert_ne!(baseline, distinct);
        }
    }

    #[test]
    fn declared_selection_includes_only_ancestors_and_descendants() {
        let selected = BTreeSet::from([PathBuf::from("systems/memory/short_term")]);
        assert!(is_selected(Path::new("systems"), &selected));
        assert!(is_selected(Path::new("systems/memory"), &selected));
        assert!(is_selected(
            Path::new("systems/memory/short_term"),
            &selected
        ));
        assert!(is_selected(
            Path::new("systems/memory/short_term/store.rs"),
            &selected
        ));
        assert!(!is_selected(Path::new("systems/vision"), &selected));
        assert!(!is_selected(Path::new("system"), &selected));

        let whole_project = BTreeSet::from([PathBuf::new()]);
        assert!(is_selected(Path::new("anything/at/all"), &whole_project));
    }

    #[test]
    fn conservative_noise_matching_has_no_substring_false_positives() {
        for retained in [
            "my_target/source.rs",
            "targeted/source.rs",
            ".venv_source/config.toml",
            "node_modules_backup/manifest.json",
            "artifact.so.source",
            "backup.bakery",
            "logistics",
        ] {
            assert!(
                !is_conservative_generated_artifact(Path::new(retained)),
                "unexpectedly excluded {retained}"
            );
        }
        for generated in [
            "target/debug/object.o",
            "nested/.venv/bin/python",
            "node_modules/pkg/index.js",
            "src/cache.pyc",
            "build/output.log",
        ] {
            assert!(
                is_conservative_generated_artifact(Path::new(generated)),
                "expected generated artifact {generated}"
            );
        }
    }

    #[test]
    fn path_and_budget_boundaries_are_exact() {
        let mut maximum_depth = PathBuf::new();
        for _ in 0..MAX_RELATIVE_COMPONENTS {
            maximum_depth.push("x");
        }
        assert!(validate_declared_relative_path(&maximum_depth, "boundary").is_ok());
        maximum_depth.push("overflow");
        assert!(validate_declared_relative_path(&maximum_depth, "boundary").is_err());
        assert!(validate_observed_relative_path(Path::new("")).is_ok());
        assert!(validate_declared_relative_path(Path::new(""), "boundary").is_err());

        let temp = std::env::temp_dir().join(format!("tidex-exact-budget-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        fs::write(temp.join("exact.bin"), b"four").unwrap();
        let exact = AcquisitionRequest::new(
            AcquisitionId::parse("exact-budget.v1").unwrap(),
            AcquisitionScope::WholeProject,
            RequestedResidency::BestVerified,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 1,
                max_total_bytes: 4,
            },
            vec![],
        )
        .unwrap();
        let envelope = SystemEnvelope::capture(&temp, &exact).unwrap();
        assert_eq!(envelope.total_file_bytes(), 4);
        fs::write(temp.join("second.bin"), b"").unwrap();
        assert!(matches!(
            SystemEnvelope::capture(&temp, &exact),
            Err(BrainError::Invalid(message)) if message == "acquisition_budget_exceeded"
        ));
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn hardlinked_files_remain_distinct_path_bound_evidence() {
        let temp = std::env::temp_dir().join(format!("tidex-hardlinks-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        fs::write(temp.join("first.bin"), b"same inode").unwrap();
        fs::hard_link(temp.join("first.bin"), temp.join("second.bin")).unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse("hardlinks.v1").unwrap(),
            AcquisitionScope::WholeProject,
            RequestedResidency::BestVerified,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 2,
                max_total_bytes: 20,
            },
            vec![],
        )
        .unwrap();
        let envelope = SystemEnvelope::capture(&temp, &request).unwrap();
        let first = envelope
            .entries()
            .iter()
            .find(|entry| entry.relative_path() == Path::new("first.bin"))
            .unwrap();
        let second = envelope
            .entries()
            .iter()
            .find(|entry| entry.relative_path() == Path::new("second.bin"))
            .unwrap();
        assert_eq!(first.byte_len(), second.byte_len());
        assert_ne!(first.sha256(), second.sha256());
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn conservative_policy_excludes_generated_noise_but_keeps_tests_models_and_examples() {
        let temp = std::env::temp_dir().join(format!("tidex-noise-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(temp.join("src")).unwrap();
        fs::create_dir_all(temp.join(".venv/lib")).unwrap();
        fs::create_dir_all(temp.join("target/debug")).unwrap();
        fs::create_dir_all(temp.join("tests")).unwrap();
        fs::create_dir_all(temp.join("examples")).unwrap();
        fs::create_dir_all(temp.join("models")).unwrap();
        fs::write(temp.join("src/lib.rs"), b"pub fn x() {}\n").unwrap();
        fs::write(temp.join(".venv/lib/x.py"), b"cache").unwrap();
        fs::write(temp.join("target/debug/x.o"), b"object").unwrap();
        fs::write(temp.join("tests/behavior.rs"), b"#[test] fn x() {}\n").unwrap();
        fs::write(temp.join("examples/main.rs"), b"fn main() {}\n").unwrap();
        fs::write(temp.join("models/weights.bin"), b"weights").unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse("noise-policy.v1").unwrap(),
            AcquisitionScope::WholeProject,
            RequestedResidency::BestVerified,
            NoisePolicy::ConservativeGeneratedArtifacts,
            AcquisitionBudget {
                max_files: 32,
                max_total_bytes: 1 << 20,
            },
            vec![],
        )
        .unwrap();
        let envelope = SystemEnvelope::capture(&temp, &request).unwrap();
        assert!(envelope
            .excluded_paths
            .iter()
            .any(|entry| entry.relative_path() == Path::new(".venv")
                && entry.reason == ExclusionReason::GeneratedArtifactPolicy));
        assert!(envelope
            .excluded_paths
            .iter()
            .any(|entry| entry.relative_path() == Path::new("target")
                && entry.reason == ExclusionReason::GeneratedArtifactPolicy));
        for retained in [
            "tests/behavior.rs",
            "examples/main.rs",
            "models/weights.bin",
        ] {
            assert!(envelope
                .entries
                .iter()
                .any(|entry| entry.relative_path() == Path::new(retained)));
        }
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn an_explicitly_selected_generated_named_root_is_not_silently_excluded() {
        let temp =
            std::env::temp_dir().join(format!("tidex-explicit-target-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(temp.join("target")).unwrap();
        fs::write(
            temp.join("target/capability.bin"),
            b"required implementation",
        )
        .unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse("selected-target.v1").unwrap(),
            declared(vec![PathBuf::from("target")]),
            RequestedResidency::BestVerified,
            NoisePolicy::ConservativeGeneratedArtifacts,
            AcquisitionBudget {
                max_files: 8,
                max_total_bytes: 1 << 20,
            },
            vec![],
        )
        .unwrap();
        let envelope = SystemEnvelope::capture(&temp, &request).unwrap();
        assert!(envelope
            .entries
            .iter()
            .any(|entry| entry.relative_path() == Path::new("target/capability.bin")));
        assert!(envelope.excluded_paths.is_empty());
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn conservative_policy_is_relative_to_each_declared_root() {
        let temp =
            std::env::temp_dir().join(format!("tidex-relative-noise-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(temp.join("project/src")).unwrap();
        fs::create_dir_all(temp.join("project/target/debug")).unwrap();
        fs::write(temp.join("project/src/lib.rs"), b"pub fn x() {}\n").unwrap();
        fs::write(temp.join("project/target/debug/x.o"), b"generated").unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse("relative-noise.v1").unwrap(),
            declared(vec![PathBuf::from("project")]),
            RequestedResidency::BestVerified,
            NoisePolicy::ConservativeGeneratedArtifacts,
            AcquisitionBudget {
                max_files: 8,
                max_total_bytes: 1 << 20,
            },
            vec![],
        )
        .unwrap();
        let envelope = SystemEnvelope::capture(&temp, &request).unwrap();
        assert!(envelope
            .entries
            .iter()
            .any(|entry| entry.relative_path() == Path::new("project/src/lib.rs")));
        assert!(envelope.excluded_paths.iter().any(|entry| {
            entry.relative_path() == Path::new("project/target")
                && entry.reason == ExclusionReason::GeneratedArtifactPolicy
        }));
        let _ = fs::remove_dir_all(temp);
    }
}
