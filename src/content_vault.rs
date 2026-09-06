//! Immutable, private retention of bytes admitted by an acquisition envelope.
//!
//! A [`SystemEnvelope`](crate::acquisition_contract::SystemEnvelope) is a
//! descriptor-bound observation, not a retained source snapshot.  The one
//! current capture receipt embeds that request and envelope and adds the CAS
//! retention authority, so they cannot drift as separately active manifests.
//! Callers may only claim `VaultRetained` after every admitted regular file
//! has been copied through one pinned source-root session into the private CAS
//! and that same pinned tree still verifies against the embedded envelope.
//!
//! `VaultRetained` is intentionally weaker than an atomic/quiescent snapshot:
//! a generic POSIX tree has no transaction spanning all files.  Adapters which
//! can obtain such a snapshot must introduce a separate, stronger receipt;
//! this authority never overclaims it.

use crate::acquisition_contract::{
    copy_manifest_file_descriptor_bound, AcquisitionRequest, SnapshotEntryKind,
    SourceCaptureSession, SystemEnvelope,
};
use crate::authority::{
    ensure_private_directory, install_private_immutable_file, remove_private_staging_file,
    stage_private_file, stage_private_file_with_identity, write_or_verify_immutable,
    PrivateFileReference, PrivateStagingIdentity,
};
use crate::digest::{CaptureReceiptDigest, Sha256Digest, SystemEnvelopeDigest};
use crate::error::{BrainError, BrainResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const RECEIPT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:CAPTURE-RECEIPT:v2\0";
const MAX_RETAINED_OBJECTS: usize = 500_000;
const MAX_CAPTURE_RECEIPT_BYTES: u64 = 512 << 20;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CaptureReceiptSchema {
    #[serde(rename = "cerebro.tidex.capture_receipt/v2")]
    Current,
}

/// Retention quality established by a capture receipt.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RetainedSourceKind {
    /// Exact bytes are retained privately and can be re-read by content hash.
    /// This is not an assertion that all source files came from one atomic
    /// instant.
    VaultRetained,
}

/// One admitted source file, linked to the envelope's path-bound file digest
/// and the raw content identity used by the vault.  No machine-local donor
/// path is sealed here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RetainedSourceObject {
    source_relative_path: PathBuf,
    source_entry_sha256: Sha256Digest,
    content_sha256: Sha256Digest,
    byte_len: u64,
}

impl RetainedSourceObject {
    pub fn source_relative_path(&self) -> &Path {
        &self.source_relative_path
    }

    pub fn source_entry_sha256(&self) -> &Sha256Digest {
        &self.source_entry_sha256
    }

    pub fn content_sha256(&self) -> &Sha256Digest {
        &self.content_sha256
    }

    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// The single current authority for one retained capture.  It embeds the
/// acquisition request and its envelope instead of requiring parallel
/// manifests that could drift or remain active after a schema change.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CaptureReceipt {
    schema: CaptureReceiptSchema,
    request: AcquisitionRequest,
    envelope: SystemEnvelope,
    retention: RetainedSourceKind,
    objects: Vec<RetainedSourceObject>,
    total_file_bytes: u64,
    manifest_sha256: CaptureReceiptDigest,
}

impl CaptureReceipt {
    pub fn request(&self) -> &AcquisitionRequest {
        &self.request
    }

    pub fn envelope(&self) -> &SystemEnvelope {
        &self.envelope
    }

    pub fn system_envelope_sha256(&self) -> &SystemEnvelopeDigest {
        self.envelope.manifest_sha256()
    }

    pub fn retention(&self) -> RetainedSourceKind {
        self.retention
    }

    pub fn objects(&self) -> &[RetainedSourceObject] {
        &self.objects
    }

    pub fn total_file_bytes(&self) -> u64 {
        self.total_file_bytes
    }

    pub fn manifest_sha256(&self) -> &CaptureReceiptDigest {
        &self.manifest_sha256
    }

    /// Verify seal, exact request/envelope binding and every retained content
    /// object.  This is the gate callers must use before relying on retained
    /// bytes for later analysis or translation.
    pub fn verify(&self, private_root: &Path) -> BrainResult<()> {
        self.request.verify()?;
        self.envelope.verify_manifest()?;
        if self.schema != CaptureReceiptSchema::Current
            || self.retention != RetainedSourceKind::VaultRetained
            || self.manifest_sha256.is_draft()
            || self.envelope.acquisition_request_sha256() != self.request.manifest_sha256()
            || self.objects.len() > MAX_RETAINED_OBJECTS
            || self.calculate_digest()? != self.manifest_sha256
        {
            return Err(BrainError::Integrity(
                "capture_receipt_structure_invalid".into(),
            ));
        }

        let expected = expected_objects(&self.envelope)?;
        if expected.len() != self.objects.len() {
            return Err(BrainError::Integrity(
                "capture_receipt_object_count_mismatch".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        let mut total = 0_u64;
        for object in &self.objects {
            let (expected_digest, expected_len) = expected
                .get(&object.source_relative_path)
                .ok_or_else(|| BrainError::Integrity("capture_receipt_entry_unknown".into()))?;
            if !seen.insert(object.source_relative_path.clone())
                || *expected_digest != object.source_entry_sha256
                || *expected_len != object.byte_len
            {
                return Err(BrainError::Integrity(
                    "capture_receipt_entry_binding_invalid".into(),
                ));
            }
            let reference = PrivateFileReference::new(
                vault_object_path(private_root, &object.content_sha256),
                object.content_sha256.clone(),
            );
            let bytes = reference.read_verified_bounded(private_root, object.byte_len)?;
            if u64::try_from(bytes.len())
                .map_err(|_| BrainError::Integrity("capture_receipt_length_invalid".into()))?
                != object.byte_len
            {
                return Err(BrainError::Integrity(
                    "capture_receipt_content_length_mismatch".into(),
                ));
            }
            total = total
                .checked_add(object.byte_len)
                .ok_or_else(|| BrainError::Integrity("capture_receipt_total_overflow".into()))?;
        }
        if total != self.total_file_bytes || total != self.envelope.total_file_bytes() {
            return Err(BrainError::Integrity(
                "capture_receipt_total_bytes_mismatch".into(),
            ));
        }
        Ok(())
    }

    /// Persist an already verified receipt independently of its contents.
    pub fn persist(&self, private_root: &Path) -> BrainResult<PrivateFileReference> {
        self.verify(private_root)?;
        let destination = receipt_path(private_root, &self.manifest_sha256);
        let bytes = serde_json::to_vec(self)?;
        let sha256 = write_or_verify_immutable(private_root, &destination, &bytes)?;
        Ok(PrivateFileReference::new(destination, sha256))
    }

    fn calculate_digest(&self) -> BrainResult<CaptureReceiptDigest> {
        let mut unsigned = self.clone();
        unsigned.manifest_sha256 = CaptureReceiptDigest::draft_marker();
        Ok(CaptureReceiptDigest::from_computed(
            Sha256Digest::digest_domain(RECEIPT_DOMAIN, &serde_json::to_vec(&unsigned)?),
        ))
    }
}

/// Verified, vault-derived source for authorities that need a filesystem view
/// of retained bytes.  It deliberately has no donor path: its only possible
/// materialization is reconstructed from the content-addressed vault after
/// the request, envelope, receipt and every object have been re-verified.
#[derive(Debug, Clone)]
pub struct RetainedSourceAdapter {
    private_root: PathBuf,
    receipt: CaptureReceipt,
}

impl RetainedSourceAdapter {
    pub fn new(private_root: &Path, receipt: &CaptureReceipt) -> BrainResult<Self> {
        receipt.verify(private_root)?;
        Ok(Self {
            private_root: private_root.to_path_buf(),
            receipt: receipt.clone(),
        })
    }

    pub fn capture_receipt_sha256(&self) -> &CaptureReceiptDigest {
        self.receipt.manifest_sha256()
    }

    pub fn system_envelope_sha256(&self) -> &SystemEnvelopeDigest {
        self.receipt.envelope().manifest_sha256()
    }

    /// Reconstruct a private, read-only analysis projection from verified CAS
    /// objects.  A pre-existing projection is never overwritten: it must
    /// still reproduce the exact envelope or discovery fails closed.
    /// Materialize this already-authenticated retained capture into its
    /// private, content-addressed projection. Source frontends may inspect
    /// only this projection; they never receive the mutable donor path.
    pub fn materialize(&self) -> BrainResult<PathBuf> {
        self.receipt.verify(&self.private_root)?;
        let root = projection_path(&self.private_root, self.receipt.manifest_sha256());
        if root.exists() {
            self.receipt
                .envelope()
                .verify_against(&root, self.receipt.request())?;
            return Ok(root);
        }

        let parent = root.parent().ok_or_else(|| {
            BrainError::Invalid("retained_source_projection_parent_missing".into())
        })?;
        ensure_private_directory(&self.private_root, parent)?;
        ensure_private_directory(&self.private_root, &root)?;

        let objects = self
            .receipt
            .objects()
            .iter()
            .map(|object| (object.source_relative_path().to_path_buf(), object))
            .collect::<BTreeMap<_, _>>();
        let result = (|| -> BrainResult<()> {
            for entry in self.receipt.envelope().entries() {
                let destination = root.join(entry.relative_path());
                match entry.kind() {
                    SnapshotEntryKind::Directory => {
                        if !entry.relative_path().as_os_str().is_empty() {
                            ensure_private_directory(&self.private_root, &destination)?;
                        }
                    }
                    SnapshotEntryKind::File => {
                        let object = objects.get(entry.relative_path()).ok_or_else(|| {
                            BrainError::Integrity(
                                "retained_source_projection_object_missing".into(),
                            )
                        })?;
                        let bytes = PrivateFileReference::new(
                            vault_object_path(&self.private_root, object.content_sha256()),
                            object.content_sha256().clone(),
                        )
                        .read_verified_bounded(&self.private_root, entry.byte_len())?;
                        let (staging, digest) =
                            stage_private_file(&self.private_root, &destination, |file| {
                                file.write_all(&bytes)?;
                                Ok(())
                            })?;
                        if digest != *object.content_sha256()
                            || !install_private_immutable_file(
                                &self.private_root,
                                &staging,
                                &destination,
                                &digest,
                            )?
                        {
                            return Err(BrainError::Integrity(
                                "retained_source_projection_install_collision".into(),
                            ));
                        }
                    }
                }
            }
            self.receipt
                .envelope()
                .verify_against(&root, self.receipt.request())?;
            Ok(())
        })();
        if let Err(error) = result {
            let _ = fs::remove_dir_all(&root);
            return Err(error);
        }
        Ok(root)
    }

    /// Recheck the projection after a consumer has read it.  This detects a
    /// mutable-private-root race rather than silently treating it as retained
    /// source authority.
    /// Reauthenticate every projected object and its topology against the
    /// capture receipt. Callers must repeat this after analysis to detect a
    /// concurrently modified projection.
    pub fn verify_materialized(&self, root: &Path) -> BrainResult<()> {
        if root != projection_path(&self.private_root, self.receipt.manifest_sha256()) {
            return Err(BrainError::Integrity(
                "retained_source_projection_identity_mismatch".into(),
            ));
        }
        self.receipt.verify(&self.private_root)?;
        self.receipt
            .envelope()
            .verify_against(root, self.receipt.request())
    }
}

/// Construct an authenticated source adapter.  Consumers must not receive a
/// donor filesystem path; this adapter is the bridge from retained bytes to
/// read-only analysis.
pub fn retained_source_adapter(
    private_root: &Path,
    receipt: &CaptureReceipt,
) -> BrainResult<RetainedSourceAdapter> {
    RetainedSourceAdapter::new(private_root, receipt)
}

/// Retain a descriptor-bound acquisition in the private content-addressed
/// vault. The embedded envelope remains an exact observation while this one
/// current receipt is the authority for retained-byte claims.
pub fn capture_to_vault(
    source_root: &Path,
    private_root: &Path,
    request: &AcquisitionRequest,
) -> BrainResult<CaptureReceipt> {
    capture_to_vault_inner(source_root, private_root, request, || Ok(()))
}

fn capture_to_vault_inner<F>(
    source_root: &Path,
    private_root: &Path,
    request: &AcquisitionRequest,
    after_envelope: F,
) -> BrainResult<CaptureReceipt>
where
    F: FnOnce() -> BrainResult<()>,
{
    request.verify()?;
    // Both roots are opened and compared by descriptor before any authority,
    // staging directory or CAS object can be written.
    let session = SourceCaptureSession::open_disjoint(source_root, private_root)?;
    let envelope = SystemEnvelope::capture_from_session(&session, request)?;
    after_envelope()?;

    // The callback is only a deterministic unit-test seam.  In production
    // this immediate re-authentication is the final read-only boundary before
    // the first private write.
    session.verify_private_root_binding(private_root)?;
    let mut staged = Vec::new();
    let mut total = 0_u64;
    for entry in envelope.entries() {
        if entry.kind() != SnapshotEntryKind::File {
            continue;
        }
        let next_total = total
            .checked_add(entry.byte_len())
            .ok_or_else(|| BrainError::Integrity("content_vault_total_overflow".into()))?;
        if next_total > request.budget().max_total_bytes {
            return Err(BrainError::Invalid("content_vault_budget_exceeded".into()));
        }
        session.verify_private_root_binding(private_root)?;
        let destination = vault_object_path_for_pending(private_root, entry.sha256());
        let mut captured_entry_sha256 = None;
        let (staging, content_sha256, staging_identity) =
            stage_private_file_with_identity(private_root, &destination, |file| {
                let digest = copy_manifest_file_descriptor_bound(
                    &session,
                    entry.relative_path(),
                    entry.byte_len(),
                    file,
                )?;
                captured_entry_sha256 = Some(digest);
                Ok(())
            })?;
        if captured_entry_sha256.as_ref() != Some(entry.sha256()) {
            remove_private_staging_file(private_root, &staging, &staging_identity)?;
            return Err(BrainError::Integrity(
                "content_vault_source_entry_digest_mismatch".into(),
            ));
        }
        total = next_total;
        staged.push(StagedRetainedSourceObject {
            private_root: private_root.to_path_buf(),
            staging,
            staging_identity,
            retained: RetainedSourceObject {
                source_relative_path: entry.relative_path().to_path_buf(),
                source_entry_sha256: entry.sha256().clone(),
                content_sha256,
                byte_len: entry.byte_len(),
            },
        });
    }

    // No object becomes visible in the CAS unless all source bytes were staged
    // and the exact same pinned tree still reproduces the envelope.
    envelope.verify_against_session(&session, request)?;
    session.verify_private_root_binding(private_root)?;

    let mut objects = Vec::with_capacity(staged.len());
    for pending in &staged {
        session.verify_private_root_binding(private_root)?;
        let destination = vault_object_path(private_root, &pending.retained.content_sha256);
        if !install_private_immutable_file(
            private_root,
            &pending.staging,
            &destination,
            &pending.retained.content_sha256,
        )? {
            let reference =
                PrivateFileReference::new(destination, pending.retained.content_sha256.clone());
            let existing =
                reference.read_verified_bounded(private_root, pending.retained.byte_len)?;
            if u64::try_from(existing.len())
                .map_err(|_| BrainError::Integrity("content_vault_length_invalid".into()))?
                != pending.retained.byte_len
            {
                return Err(BrainError::Integrity(
                    "content_vault_digest_length_collision".into(),
                ));
            }
        }
        objects.push(pending.retained.clone());
    }
    objects.sort_by(|left, right| left.source_relative_path.cmp(&right.source_relative_path));

    // A namespace reparent or donor mutation concurrent with CAS publication
    // cannot produce a receipt: recheck both pinned roots and the semantic
    // envelope after installation.  CAS blobs alone are never authorities.
    session.verify_private_root_binding(private_root)?;
    envelope.verify_against_session(&session, request)?;
    let mut receipt = CaptureReceipt {
        schema: CaptureReceiptSchema::Current,
        request: request.clone(),
        envelope,
        retention: RetainedSourceKind::VaultRetained,
        objects,
        total_file_bytes: total,
        manifest_sha256: CaptureReceiptDigest::draft_marker(),
    };
    receipt.manifest_sha256 = receipt.calculate_digest()?;
    session.verify_private_root_binding(private_root)?;
    receipt.verify(private_root)?;
    Ok(receipt)
}

struct StagedRetainedSourceObject {
    private_root: PathBuf,
    staging: PathBuf,
    staging_identity: PrivateStagingIdentity,
    retained: RetainedSourceObject,
}

impl Drop for StagedRetainedSourceObject {
    fn drop(&mut self) {
        // Installation removes this name on every terminal outcome. If an
        // earlier phase aborts, only remove the exact inode that was staged;
        // a replaced name is left untouched and the operation remains closed.
        let _ =
            remove_private_staging_file(&self.private_root, &self.staging, &self.staging_identity);
    }
}

/// Reopen the one current capture authority by its semantic identity.  The
/// canonical location is derived internally, so callers cannot substitute an
/// obsolete or parallel manifest path.
pub fn load_capture_receipt(
    private_root: &Path,
    digest: &CaptureReceiptDigest,
) -> BrainResult<CaptureReceipt> {
    let path = receipt_path(private_root, digest);
    let bytes = crate::authority::read_untrusted_private_file_bounded(
        private_root,
        &path,
        MAX_CAPTURE_RECEIPT_BYTES,
    )?;
    let receipt: CaptureReceipt = serde_json::from_slice(&bytes)?;
    if receipt.manifest_sha256() != digest {
        return Err(BrainError::Integrity(
            "capture_receipt_identity_mismatch".into(),
        ));
    }
    receipt.verify(private_root)?;
    Ok(receipt)
}

/// Authenticate an exact persisted receipt reference and require its one
/// canonical current location.  This is the boundary used by downstream
/// capability authorities; alternate copies cannot become active manifests.
pub fn authenticate_capture_receipt(
    private_root: &Path,
    reference: &PrivateFileReference,
) -> BrainResult<CaptureReceipt> {
    let bytes = reference.read_verified_bounded(private_root, MAX_CAPTURE_RECEIPT_BYTES)?;
    let receipt: CaptureReceipt = serde_json::from_slice(&bytes)?;
    if reference.path != receipt_path(private_root, receipt.manifest_sha256()) {
        return Err(BrainError::Integrity(
            "capture_receipt_noncanonical_location".into(),
        ));
    }
    receipt.verify(private_root)?;
    Ok(receipt)
}

/// Parse an untrusted textual identity only inside the loading boundary.  A
/// caller never receives a minted semantic digest merely for presenting 64
/// hexadecimal characters: the corresponding canonical receipt must exist
/// and pass full CAS reauthentication first.
pub fn load_capture_receipt_by_id(
    private_root: &Path,
    untrusted_digest: &str,
) -> BrainResult<CaptureReceipt> {
    let digest = CaptureReceiptDigest::from_computed(Sha256Digest::parse(untrusted_digest)?);
    load_capture_receipt(private_root, &digest)
}

fn expected_objects(
    envelope: &SystemEnvelope,
) -> BrainResult<BTreeMap<PathBuf, (Sha256Digest, u64)>> {
    let mut expected = BTreeMap::new();
    for entry in envelope.entries() {
        if entry.kind() == SnapshotEntryKind::File
            && expected
                .insert(
                    entry.relative_path().to_path_buf(),
                    (entry.sha256().clone(), entry.byte_len()),
                )
                .is_some()
        {
            return Err(BrainError::Integrity(
                "capture_receipt_envelope_file_path_duplicate".into(),
            ));
        }
    }
    Ok(expected)
}

fn vault_object_path_for_pending(private_root: &Path, entry_sha256: &Sha256Digest) -> PathBuf {
    // The final location is keyed by raw content, unknown until streaming has
    // completed.  Stage in a dedicated safe directory rather than beside an
    // untrusted donor path.
    private_root
        .join("state/acquisitions/content-vault/staging")
        .join(format!("{}.pending", entry_sha256.as_str()))
}

fn vault_object_path(private_root: &Path, content_sha256: &Sha256Digest) -> PathBuf {
    let digest = content_sha256.as_str();
    private_root
        .join("state/acquisitions/content-vault/sha256")
        .join(&digest[..2])
        .join(digest)
}

fn receipt_path(private_root: &Path, digest: &CaptureReceiptDigest) -> PathBuf {
    private_root
        .join("state/acquisitions/capture-receipts/by-sha")
        .join(format!("{}.json", digest.as_str()))
}

fn projection_path(private_root: &Path, digest: &CaptureReceiptDigest) -> PathBuf {
    private_root
        .join("state/acquisitions/content-vault/projections")
        .join(digest.as_str())
        .join("source")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquisition_contract::{
        AcquisitionBudget, AcquisitionScope, NoisePolicy, RequestedResidency,
    };
    use crate::identity::AcquisitionId;
    use crate::security::secure_dir;
    use std::fs;

    fn roots(label: &str) -> (PathBuf, PathBuf, PathBuf) {
        let base = std::env::temp_dir().join(format!(
            "tidex-content-vault-{label}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&base);
        let private = base.join("private");
        let source = base.join("source");
        fs::create_dir_all(&private).unwrap();
        fs::create_dir_all(&source).unwrap();
        secure_dir(&private).unwrap();
        (base, private, source)
    }

    fn request() -> AcquisitionRequest {
        AcquisitionRequest::new(
            AcquisitionId::parse("content-vault-test-v1").unwrap(),
            AcquisitionScope::WholeProject,
            RequestedResidency::BestVerified,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 8,
                max_total_bytes: 1024 * 1024,
            },
            vec![],
        )
        .unwrap()
    }

    #[test]
    fn retains_reproducible_content_in_one_current_capture_authority() {
        let (base, root, source) = roots("repro");
        fs::write(source.join("a.txt"), b"same bytes").unwrap();
        let receipt = capture_to_vault(&source, &root, &request()).unwrap();
        assert_eq!(
            receipt.envelope().source_retention(),
            crate::acquisition_contract::SourceRetention::ManifestOnly
        );
        receipt.verify(&root).unwrap();
        let reference = receipt.persist(&root).unwrap();
        assert_eq!(
            load_capture_receipt_by_id(&root, receipt.manifest_sha256().as_str())
                .unwrap()
                .manifest_sha256(),
            receipt.manifest_sha256()
        );
        assert_eq!(
            reference.path,
            receipt_path(&root, receipt.manifest_sha256())
        );
        let repeated = capture_to_vault(&source, &root, &request()).unwrap();
        assert_eq!(receipt.manifest_sha256(), repeated.manifest_sha256());
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn retains_distinct_paths_with_identical_content() {
        let (base, root, source) = roots("duplicate-content");
        fs::write(source.join("a.txt"), b"same bytes").unwrap();
        fs::write(source.join("b.txt"), b"same bytes").unwrap();
        let receipt = capture_to_vault(&source, &root, &request()).unwrap();
        receipt.verify(&root).unwrap();
        assert_eq!(receipt.objects().len(), 2);
        assert_ne!(
            receipt.objects()[0].source_relative_path(),
            receipt.objects()[1].source_relative_path()
        );
        assert_eq!(
            receipt.objects()[0].content_sha256(),
            receipt.objects()[1].content_sha256()
        );
        let adapter = retained_source_adapter(&root, &receipt).unwrap();
        let projection = adapter.materialize().unwrap();
        assert_eq!(fs::read(projection.join("a.txt")).unwrap(), b"same bytes");
        assert_eq!(fs::read(projection.join("b.txt")).unwrap(), b"same bytes");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn rejects_tamper_cross_root_symlink_and_cross_envelope_replay() {
        let (base, root, source) = roots("attack");
        fs::write(source.join("a.txt"), b"content").unwrap();
        let request = request();
        let receipt = capture_to_vault(&source, &root, &request).unwrap();
        let mut tampered_json = serde_json::to_value(&receipt).unwrap();
        tampered_json["total_file_bytes"] = serde_json::json!(999_u64);
        let tampered: CaptureReceipt = serde_json::from_value(tampered_json).unwrap();
        assert!(tampered.verify(&root).is_err());
        let mut obsolete_json = serde_json::to_value(&receipt).unwrap();
        obsolete_json["schema"] = serde_json::json!("cerebro.tidex.capture_receipt/v1");
        assert!(serde_json::from_value::<CaptureReceipt>(obsolete_json).is_err());
        let object = &receipt.objects()[0];
        let path = vault_object_path(&root, object.content_sha256());
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(source.join("a.txt"), &path).unwrap();
        assert!(receipt.verify(&root).is_err());
        fs::remove_file(&path).unwrap();
        let fresh = capture_to_vault(&source, &root, &request).unwrap();
        let (other_base, other_root, _) = roots("other");
        assert!(fresh.verify(&other_root).is_err());
        fs::write(source.join("a.txt"), b"changed").unwrap();
        assert_ne!(
            SystemEnvelope::capture(&source, &request)
                .unwrap()
                .manifest_sha256(),
            fresh.envelope().manifest_sha256()
        );
        fresh.verify(&root).unwrap();
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(other_base);
    }

    #[test]
    fn respects_capture_budget_before_retention() {
        let (base, root, source) = roots("budget");
        fs::write(source.join("large"), vec![1_u8; 32]).unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse("content-vault-budget-v1").unwrap(),
            AcquisitionScope::WholeProject,
            RequestedResidency::BestVerified,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 1,
                max_total_bytes: 16,
            },
            vec![],
        )
        .unwrap();
        assert!(capture_to_vault(&source, &root, &request).is_err());
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn rejects_both_root_overlap_directions_before_any_vault_write() {
        let base = std::env::temp_dir().join(format!(
            "tidex-content-vault-overlap-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&base);

        let private_parent = base.join("private-parent");
        let nested_source = private_parent.join("source");
        fs::create_dir_all(&nested_source).unwrap();
        secure_dir(&private_parent).unwrap();
        fs::write(nested_source.join("a.txt"), b"bytes").unwrap();
        assert!(matches!(
            capture_to_vault(&nested_source, &private_parent, &request()),
            Err(BrainError::Invalid(message))
                if message == "acquisition_source_private_root_overlap"
        ));
        assert!(!private_parent.join("state").exists());

        let source_parent = base.join("source-parent");
        let nested_private = source_parent.join("private");
        fs::create_dir_all(&nested_private).unwrap();
        secure_dir(&nested_private).unwrap();
        fs::write(source_parent.join("b.txt"), b"bytes").unwrap();
        assert!(matches!(
            capture_to_vault(&source_parent, &nested_private, &request()),
            Err(BrainError::Invalid(message))
                if message == "acquisition_source_private_root_overlap"
        ));
        assert!(!nested_private.join("state").exists());
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn source_path_replacement_cannot_redirect_the_pinned_capture() {
        let (base, root, source) = roots("root-replacement");
        fs::write(source.join("a.txt"), b"pinned bytes").unwrap();
        let pinned_location = base.join("pinned-source");
        let receipt = capture_to_vault_inner(&source, &root, &request(), || {
            fs::rename(&source, &pinned_location)?;
            fs::create_dir(&source)?;
            fs::write(source.join("a.txt"), b"replacement bytes")?;
            Ok(())
        })
        .unwrap();
        receipt.verify(&root).unwrap();
        let projection = retained_source_adapter(&root, &receipt)
            .unwrap()
            .materialize()
            .unwrap();
        assert_eq!(fs::read(projection.join("a.txt")).unwrap(), b"pinned bytes");
        assert_eq!(
            fs::read(source.join("a.txt")).unwrap(),
            b"replacement bytes"
        );
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn file_replacement_before_retention_leaves_no_cas_or_receipt_authority() {
        let (base, root, source) = roots("file-replacement");
        fs::write(source.join("a.txt"), b"manifest bytes").unwrap();
        let replacement = source.join("replacement.tmp");
        let result = capture_to_vault_inner(&source, &root, &request(), || {
            fs::write(&replacement, b"changed bytes")?;
            fs::rename(&replacement, source.join("a.txt"))?;
            Ok(())
        });
        assert!(matches!(result, Err(BrainError::Integrity(_))));
        assert!(!root
            .join("state/acquisitions/content-vault/sha256")
            .exists());
        assert!(!root.join("state/acquisitions/capture-receipts").exists());
        let staging = root.join("state/acquisitions/content-vault/staging");
        if staging.exists() {
            assert_eq!(fs::read_dir(staging).unwrap().count(), 0);
        }
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn reparenting_source_under_private_before_staging_fails_without_state() {
        let (base, root, source) = roots("reparent-overlap");
        fs::write(source.join("a.txt"), b"bytes").unwrap();
        let nested = root.join("moved-source");
        let result = capture_to_vault_inner(&source, &root, &request(), || {
            fs::rename(&source, &nested)?;
            Ok(())
        });
        assert!(matches!(
            result,
            Err(BrainError::Invalid(message))
                if message == "acquisition_source_private_root_overlap"
        ));
        assert!(!root.join("state").exists());
        let _ = fs::remove_dir_all(base);
    }
}
