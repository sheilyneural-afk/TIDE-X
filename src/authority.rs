use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use rustix::fs::{
    fchmod, flock, fstat, linkat, mkdirat, open, openat, openat2, renameat, renameat_with,
    unlinkat, AtFlags, Dir, FileType, FlockOperation, Mode, OFlags, RenameFlags, ResolveFlags,
    Stat,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::io::AsRawFd;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_IMMUTABLE_TEMPORARY: AtomicU64 = AtomicU64::new(0);
const PRIVATE_READ_BUFFER_BYTES: usize = 64 * 1024;
const PRIVATE_TREE_MAX_DEPTH: usize = 256;
const PRIVATE_RESOLUTION: ResolveFlags = ResolveFlags::BENEATH
    .union(ResolveFlags::NO_SYMLINKS)
    .union(ResolveFlags::NO_MAGICLINKS);

/// Reserve a unique private staging file beside an immutable destination.
///
/// The returned descriptor is opened with `CREAT|EXCL` relative to a verified
/// parent directory fd, so no pathname exists to race between check and create.
/// [`install_private_immutable_file`] performs the final sync, digest check and
/// no-overwrite installation.
pub fn create_private_staging_file(
    root: &Path,
    destination: &Path,
) -> BrainResult<(PathBuf, File)> {
    ensure_private_parent(root, destination)?;
    let relative = root_relative_path(root, destination)?;
    let parent_relative = relative
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?
        .to_path_buf();
    let file_name = destination
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| BrainError::Invalid("private_file_name_invalid".into()))?;
    let root_fd = open_private_root_fd(root)?;
    let parent_fd = if parent_relative.as_os_str().is_empty() {
        open_private_root_fd(root)?
    } else {
        open_private_dir_fd(&root_fd, &parent_relative)?
    };
    loop {
        let sequence = NEXT_IMMUTABLE_TEMPORARY.fetch_add(1, Ordering::Relaxed);
        let temp_name = format!(
            ".{file_name}.{}.{}.immutable.tmp",
            std::process::id(),
            sequence
        );
        let temporary = root.join(&parent_relative).join(&temp_name);
        root_relative_path(root, &temporary)?;
        match openat(
            &parent_fd,
            temp_name.as_str(),
            OFlags::RDWR | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC,
            Mode::from_bits_truncate(0o600),
        ) {
            Ok(raw_fd) => return Ok((temporary, File::from(raw_fd))),
            Err(e) if e == rustix::io::Errno::EXIST => continue,
            Err(e) => return Err(rustix_error(e)),
        }
    }
}

/// Produce a complete private staging file through a streaming callback.
/// Failed producers never leave a partial staging object behind.
/// The file is secured and hashed through its open descriptor; no pathname
/// reopening occurs after the initial creation.
pub fn stage_private_file<F>(
    root: &Path,
    destination: &Path,
    produce: F,
) -> BrainResult<(PathBuf, Sha256Digest)>
where
    F: FnOnce(&mut File) -> BrainResult<()>,
{
    let (path, digest, _) = stage_private_file_with_identity(root, destination, produce)?;
    Ok((path, digest))
}

pub(crate) fn stage_private_file_with_identity<F>(
    root: &Path,
    destination: &Path,
    produce: F,
) -> BrainResult<(PathBuf, Sha256Digest, PrivateStagingIdentity)>
where
    F: FnOnce(&mut File) -> BrainResult<()>,
{
    let (temporary, mut file) = create_private_staging_file(root, destination)?;
    let created_stat = fstat(&file).map_err(rustix_error)?;
    let created_identity = PrivateStagingIdentity::from_stat(&created_stat);
    let result = (|| -> BrainResult<Sha256Digest> {
        produce(&mut file)?;
        fd_secure_file(&file)?;
        file.sync_all()?;
        sha256_fd(&mut file)
    })();
    drop(file);
    match result {
        Ok(digest) => Ok((temporary, digest, created_identity)),
        Err(error) => {
            match unlink_private_file_if_same_inode(
                root,
                &temporary,
                &created_identity,
                "private_file_staging_replaced_before_cleanup",
            ) {
                Ok(()) => Err(error),
                Err(cleanup_error) => Err(cleanup_error),
            }
        }
    }
}

/// Content identity of an immutable file under a verified CEREBRO private root.
///
/// This is deliberately generic only over file identity. It does not erase the
/// semantic type of the payload: Delta/F64 artifacts, receipts and JSON records
/// keep their own domain structures on top of this reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrivateStagingIdentity {
    device: u64,
    inode: u64,
}

impl PrivateStagingIdentity {
    fn from_stat(stat: &Stat) -> Self {
        Self {
            device: stat.st_dev,
            inode: stat.st_ino,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrivateFileReference {
    pub path: PathBuf,
    pub sha256: Sha256Digest,
}

/// Path-independent identity of a complete private directory tree.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrivateDirectoryIdentity {
    pub tree_sha256: Sha256Digest,
    pub entry_count: u64,
    pub regular_file_count: u64,
    pub total_file_bytes: u64,
}

#[derive(Debug)]
struct PrivateTreeEntry {
    relative_path: PathBuf,
    kind: u8,
    file_size: u64,
    file_sha256: Option<Sha256Digest>,
}

impl PrivateFileReference {
    pub fn new(path: impl Into<PathBuf>, sha256: Sha256Digest) -> Self {
        Self {
            path: path.into(),
            sha256,
        }
    }

    /// Verify path confinement and exact current file content through one
    /// descriptor. The returned path is only the confined lexical name; it is
    /// not a capability and must not be reopened by a new security boundary.
    pub fn verify(&self, root: &Path) -> BrainResult<PathBuf> {
        let opened = open_private_reference(root, &self.path)?;
        verify_opened_private_file(opened.file, &opened.stat, &self.sha256, None, false)
            .map(|_| opened.path)
    }

    /// Legacy compatibility reader for payloads whose size is already bounded
    /// by an authenticated enclosing contract. New authority boundaries must
    /// use [`Self::read_verified_bounded`] with their protocol-specific limit.
    pub fn read_verified(&self, root: &Path) -> BrainResult<Vec<u8>> {
        Ok(self.read_verified_with_path(root)?.1)
    }

    /// Read and authenticate at most `max_bytes`, consuming at most one byte
    /// beyond the limit as an oversize sentinel. Resolution, metadata checks,
    /// hashing and reading all use the same descriptor.
    pub fn read_verified_bounded(&self, root: &Path, max_bytes: u64) -> BrainResult<Vec<u8>> {
        self.read_verified_bounded_after_open(root, max_bytes, || {})
    }

    /// Legacy unbounded counterpart returning the confined lexical path and
    /// exact bytes from one descriptor. The path must not be reopened as an
    /// authority capability; new boundaries should retain the bytes returned
    /// by [`Self::read_verified_bounded`] instead.
    pub fn read_verified_with_path(&self, root: &Path) -> BrainResult<(PathBuf, Vec<u8>)> {
        let opened = open_private_reference(root, &self.path)?;
        let path = opened.path;
        let bytes =
            verify_opened_private_file(opened.file, &opened.stat, &self.sha256, None, true)?;
        Ok((path, bytes))
    }

    fn read_verified_bounded_after_open<F>(
        &self,
        root: &Path,
        max_bytes: u64,
        after_open: F,
    ) -> BrainResult<Vec<u8>>
    where
        F: FnOnce(),
    {
        let opened = open_private_reference(root, &self.path)?;
        after_open();
        verify_opened_private_file(
            opened.file,
            &opened.stat,
            &self.sha256,
            Some(max_bytes),
            true,
        )
    }
}

/// Read an untrusted command input from the configured private authority.
///
/// Unlike [`PrivateFileReference`], this boundary deliberately has no
/// pre-declared content digest: command input is untrusted until its caller
/// parses and validates it.  It still receives the same descriptor-bound
/// confinement, regular-file, ownership, stability, and byte-limit checks as
/// authenticated private files.  This is the only appropriate reader for a
/// bounded, ephemeral CLI input that has not yet been admitted as an
/// immutable artifact.
pub fn read_existing_private_file_bounded(
    root: &Path,
    raw: &Path,
    max_bytes: u64,
) -> BrainResult<Vec<u8>> {
    let opened = open_private_reference(root, raw)?;
    read_opened_private_file_bounded(opened.file, &opened.stat, max_bytes)
}

/// Open one existing private regular file as a descriptor-bound read capability.
/// Path confinement, root identity, symlink rejection, owner and permission
/// checks are completed before the `File` is returned. Consumers that need to
/// stream or seek large artifacts must retain this descriptor rather than
/// validating a pathname and reopening it later.
pub(crate) fn open_existing_private_file(root: &Path, raw: &Path) -> BrainResult<File> {
    Ok(open_private_reference(root, raw)?.file)
}

pub fn read_untrusted_private_file_bounded(
    root: &Path,
    raw: &Path,
    max_bytes: u64,
) -> BrainResult<Vec<u8>> {
    read_existing_private_file_bounded(root, raw, max_bytes)
}

struct OpenedPrivateFile {
    path: PathBuf,
    file: File,
    stat: Stat,
}

fn open_confined_regular_file(root: &Path, raw: &Path) -> BrainResult<OpenedPrivateFile> {
    let relative = root_relative_path(root, raw)?;
    let root_fd = open_private_root_fd(root)?;
    let fd = openat2(
        &root_fd,
        &relative,
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
        PRIVATE_RESOLUTION,
    )
    .map_err(rustix_error)?;
    let stat = fstat(&fd).map_err(rustix_error)?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        return Err(BrainError::Integrity("private_file_not_regular".into()));
    }
    Ok(OpenedPrivateFile {
        path: root.join(relative),
        file: File::from(fd),
        stat,
    })
}

fn open_private_reference(root: &Path, raw: &Path) -> BrainResult<OpenedPrivateFile> {
    let relative = root_relative_path(root, raw)?;
    if root == Path::new("/")
        || root
            .components()
            .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        return Err(BrainError::Invalid("private_root_path_invalid".into()));
    }
    let filesystem_root = open(
        Path::new("/"),
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(rustix_error)?;
    let relative_root = root
        .strip_prefix(Path::new("/"))
        .map_err(|_| BrainError::Invalid("private_root_path_invalid".into()))?;
    let root_fd = openat2(
        &filesystem_root,
        relative_root,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
        PRIVATE_RESOLUTION,
    )
    .map_err(rustix_error)?;
    let root_stat = fstat(&root_fd).map_err(rustix_error)?;
    validate_private_root_stat(&root_stat)?;

    // NONBLOCK ensures that a type race to a FIFO or device cannot hang this
    // authority before fstat rejects it.
    let fd = openat2(
        &root_fd,
        &relative,
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
        PRIVATE_RESOLUTION,
    )
    .map_err(rustix_error)?;
    let stat = fstat(&fd).map_err(rustix_error)?;
    validate_private_file_stat(&root_stat, &stat)?;
    Ok(OpenedPrivateFile {
        path: root.join(relative),
        file: File::from(fd),
        stat,
    })
}

fn validate_private_root_stat(stat: &Stat) -> BrainResult<()> {
    if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
        return Err(BrainError::Integrity(
            "private_root_descriptor_invalid".into(),
        ));
    }
    Ok(())
}

fn validate_private_file_stat(root: &Stat, file: &Stat) -> BrainResult<()> {
    if FileType::from_raw_mode(file.st_mode) != FileType::RegularFile {
        return Err(BrainError::Integrity("private_file_not_regular".into()));
    }
    // Files must share the root owner, cannot carry special bits or be
    // world-writable. Group-write is accepted only when the root itself denies
    // every group permission, so it remains unreachable through this tree.
    let group_write_exposed = file.st_mode & 0o020 != 0 && root.st_mode & 0o070 != 0;
    if file.st_uid != root.st_uid || file.st_mode & 0o7002 != 0 || group_write_exposed {
        return Err(BrainError::Integrity(
            "private_file_permissions_or_owner_invalid".into(),
        ));
    }
    Ok(())
}

fn verify_opened_private_file(
    mut file: File,
    opened: &Stat,
    expected: &Sha256Digest,
    max_bytes: Option<u64>,
    retain_bytes: bool,
) -> BrainResult<Vec<u8>> {
    let opened_len = u64::try_from(opened.st_size)
        .map_err(|_| BrainError::Integrity("private_file_size_invalid".into()))?;
    if max_bytes.is_some_and(|limit| opened_len > limit) {
        return Err(BrainError::Invalid(
            "private_file_read_limit_exceeded".into(),
        ));
    }
    let capacity = if retain_bytes {
        usize::try_from(opened_len)
            .map_err(|_| BrainError::Invalid("private_file_read_limit_exceeded".into()))?
    } else {
        0
    };
    let mut bytes = Vec::with_capacity(capacity);
    let mut hasher = Sha256::new();
    let mut actual_len = 0_u64;
    let mut buffer = [0_u8; PRIVATE_READ_BUFFER_BYTES];
    loop {
        let read_limit = max_bytes
            .map(|limit| {
                limit
                    .saturating_sub(actual_len)
                    .saturating_add(1)
                    .min(PRIVATE_READ_BUFFER_BYTES as u64) as usize
            })
            .unwrap_or(PRIVATE_READ_BUFFER_BYTES);
        let count = file.read(&mut buffer[..read_limit])?;
        if count == 0 {
            break;
        }
        actual_len = actual_len
            .checked_add(count as u64)
            .ok_or_else(|| BrainError::Integrity("private_file_size_overflow".into()))?;
        if max_bytes.is_some_and(|limit| actual_len > limit) {
            return Err(BrainError::Invalid(
                "private_file_read_limit_exceeded".into(),
            ));
        }
        hasher.update(&buffer[..count]);
        if retain_bytes {
            bytes.extend_from_slice(&buffer[..count]);
        }
    }
    let final_stat = fstat(&file).map_err(rustix_error)?;
    ensure_private_file_stable(opened, &final_stat)?;
    if actual_len != opened_len {
        return Err(BrainError::Integrity(
            "private_file_mutated_during_read".into(),
        ));
    }
    let actual = Sha256Digest::parse(format!("{:x}", hasher.finalize()))?;
    if actual != *expected {
        return Err(BrainError::Integrity(
            "private_file_reference_digest_mismatch".into(),
        ));
    }
    Ok(bytes)
}

fn read_opened_private_file_bounded(
    mut file: File,
    opened: &Stat,
    max_bytes: u64,
) -> BrainResult<Vec<u8>> {
    let opened_len = u64::try_from(opened.st_size)
        .map_err(|_| BrainError::Integrity("private_file_size_invalid".into()))?;
    if opened_len > max_bytes {
        return Err(BrainError::Invalid(
            "private_file_read_limit_exceeded".into(),
        ));
    }
    let capacity = usize::try_from(opened_len)
        .map_err(|_| BrainError::Invalid("private_file_read_limit_exceeded".into()))?;
    let mut bytes = Vec::with_capacity(capacity);
    let mut actual_len = 0_u64;
    let mut buffer = [0_u8; PRIVATE_READ_BUFFER_BYTES];
    loop {
        let remaining_plus_sentinel = max_bytes
            .saturating_sub(actual_len)
            .saturating_add(1)
            .min(PRIVATE_READ_BUFFER_BYTES as u64) as usize;
        let count = file.read(&mut buffer[..remaining_plus_sentinel])?;
        if count == 0 {
            break;
        }
        actual_len = actual_len
            .checked_add(count as u64)
            .ok_or_else(|| BrainError::Integrity("private_file_size_overflow".into()))?;
        if actual_len > max_bytes {
            return Err(BrainError::Invalid(
                "private_file_read_limit_exceeded".into(),
            ));
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    let final_stat = fstat(&file).map_err(rustix_error)?;
    ensure_private_file_stable(opened, &final_stat)?;
    if actual_len != opened_len {
        return Err(BrainError::Integrity(
            "private_file_mutated_during_read".into(),
        ));
    }
    Ok(bytes)
}

fn ensure_private_file_stable(before: &Stat, after: &Stat) -> BrainResult<()> {
    if FileType::from_raw_mode(after.st_mode) != FileType::RegularFile
        || before.st_dev != after.st_dev
        || before.st_ino != after.st_ino
        || before.st_mode != after.st_mode
        || before.st_size != after.st_size
        || before.st_mtime != after.st_mtime
        || before.st_mtime_nsec != after.st_mtime_nsec
        || before.st_ctime != after.st_ctime
        || before.st_ctime_nsec != after.st_ctime_nsec
    {
        return Err(BrainError::Integrity(
            "private_file_mutated_during_read".into(),
        ));
    }
    Ok(())
}

fn rustix_error(error: rustix::io::Errno) -> BrainError {
    BrainError::Io(std::io::Error::from_raw_os_error(error.raw_os_error()))
}

/// Open the private authority root as a verified directory descriptor.
/// This root fd anchors all subsequent fd-relative write operations, eliminating
/// any pathname reopening window between verification and mutation.
fn open_private_root_fd(root: &Path) -> BrainResult<File> {
    if root == Path::new("/")
        || root
            .components()
            .any(|c| !matches!(c, Component::RootDir | Component::Normal(_)))
    {
        return Err(BrainError::Invalid("private_root_path_invalid".into()));
    }
    let filesystem_root = open(
        Path::new("/"),
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(rustix_error)?;
    let relative_root = root
        .strip_prefix(Path::new("/"))
        .map_err(|_| BrainError::Invalid("private_root_path_invalid".into()))?;
    let root_fd = openat2(
        &filesystem_root,
        relative_root,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
        PRIVATE_RESOLUTION,
    )
    .map_err(rustix_error)?;
    let stat = fstat(&root_fd).map_err(rustix_error)?;
    validate_private_root_stat(&stat)?;
    Ok(File::from(root_fd))
}

/// Open a private directory relative to an already-verified root descriptor
/// using the same BENEATH/NO_SYMLINKS/NO_MAGICLINKS resolution policy as reads.
fn open_private_dir_fd(root_fd: &File, relative: &Path) -> BrainResult<File> {
    let fd = openat2(
        root_fd,
        relative,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
        PRIVATE_RESOLUTION,
    )
    .map_err(rustix_error)?;
    let stat = fstat(&fd).map_err(rustix_error)?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
        return Err(BrainError::Integrity("private_dir_fd_not_directory".into()));
    }
    Ok(File::from(fd))
}

/// Set file permissions to 0o600 through an open descriptor, eliminating the
/// TOCTOU window between a pathname check and a pathname chmod.
fn unlink_name_if_same_inode(
    parent: &File,
    name: &std::ffi::OsStr,
    expected: &PrivateStagingIdentity,
    replacement_error: &str,
) -> BrainResult<()> {
    let current_fd = match openat2(
        parent,
        Path::new(name),
        OFlags::PATH | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
        PRIVATE_RESOLUTION,
    ) {
        Ok(fd) => fd,
        Err(error) if error == rustix::io::Errno::NOENT => return Ok(()),
        Err(error) => return Err(rustix_error(error)),
    };
    let current = fstat(&current_fd).map_err(rustix_error)?;
    if current.st_dev != expected.device || current.st_ino != expected.inode {
        return Err(BrainError::Integrity(replacement_error.into()));
    }
    unlinkat(parent, name, AtFlags::empty()).map_err(rustix_error)?;
    parent.sync_all()?;
    Ok(())
}

fn unlink_private_file_if_same_inode(
    root: &Path,
    path: &Path,
    expected: &PrivateStagingIdentity,
    replacement_error: &str,
) -> BrainResult<()> {
    let relative = root_relative_path(root, path)?;
    let parent_relative = relative
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?;
    let name = relative
        .file_name()
        .ok_or_else(|| BrainError::Invalid("private_file_name_invalid".into()))?;
    let root_fd = open_private_root_fd(root)?;
    let parent_fd = if parent_relative.as_os_str().is_empty() {
        root_fd
    } else {
        open_private_dir_fd(&root_fd, parent_relative)?
    };
    unlink_name_if_same_inode(&parent_fd, name, expected, replacement_error)
}

pub(crate) fn remove_private_staging_file(
    root: &Path,
    path: &Path,
    identity: &PrivateStagingIdentity,
) -> BrainResult<()> {
    unlink_private_file_if_same_inode(
        root,
        path,
        identity,
        "private_file_staging_replaced_before_cleanup",
    )
}

fn fd_secure_file(fd: &File) -> BrainResult<()> {
    fchmod(fd, Mode::from_bits_truncate(0o600)).map_err(rustix_error)
}

/// Set directory permissions to 0o700 through an open descriptor.
fn fd_secure_dir(fd: &File) -> BrainResult<()> {
    fchmod(fd, Mode::from_bits_truncate(0o700)).map_err(rustix_error)
}

/// Compute SHA-256 of a file's full content from position 0 using only the
/// already-open descriptor. No pathname resolution occurs after the caller
/// has opened the fd.
fn sha256_fd(file: &mut File) -> BrainResult<Sha256Digest> {
    file.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; PRIVATE_READ_BUFFER_BYTES];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Sha256Digest::parse(format!("{:x}", hasher.finalize()))
}

/// Convert an absolute private path into a component-safe relative path.
/// Parent components, root prefixes and other non-normal components are never
/// accepted, so later joins cannot escape the already verified root.
pub fn root_relative_path(root: &Path, raw: &Path) -> BrainResult<PathBuf> {
    if !raw.is_absolute() {
        return Err(BrainError::Invalid(
            "private_file_path_must_be_absolute".into(),
        ));
    }
    let relative = raw
        .strip_prefix(root)
        .map_err(|_| BrainError::Integrity("private_file_path_outside_root".into()))?;
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(BrainError::Invalid("private_file_path_not_confined".into()));
    }
    Ok(relative.to_path_buf())
}

/// Return the confined lexical name of an existing private regular file.
/// The validation itself is descriptor-bound; the returned `PathBuf` is not a
/// capability and callers that consume bytes must open/read through authority.
pub fn existing_regular_file_under_root(root: &Path, raw: &Path) -> BrainResult<PathBuf> {
    Ok(open_confined_regular_file(root, raw)?.path)
}

/// Descriptor-bound optional regular-file preflight. Missing path components
/// are a normal `None`; symlinks, special files and invalid permissions fail.
pub fn existing_regular_file_if_present(root: &Path, raw: &Path) -> BrainResult<Option<PathBuf>> {
    match open_confined_regular_file(root, raw) {
        Ok(opened) => Ok(Some(opened.path)),
        Err(BrainError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// Return the confined lexical name of an existing private directory. The
/// directory is opened with `openat2` under the verified root before success.
pub fn existing_directory_under_root(root: &Path, raw: &Path) -> BrainResult<PathBuf> {
    let relative = root_relative_path(root, raw)?;
    let root_fd = open_private_root_fd(root)?;
    let _directory_fd = open_private_dir_fd(&root_fd, &relative)?;
    Ok(root.join(relative))
}

/// Descriptor-bound optional directory preflight.
pub fn existing_directory_if_present(root: &Path, raw: &Path) -> BrainResult<Option<PathBuf>> {
    let relative = root_relative_path(root, raw)?;
    let root_fd = open_private_root_fd(root)?;
    match open_private_dir_fd(&root_fd, &relative) {
        Ok(_) => Ok(Some(root.join(relative))),
        Err(BrainError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn ensure_same_private_tree_object(
    before: &Stat,
    after: &Stat,
    expected_type: FileType,
    label: &str,
) -> BrainResult<()> {
    if FileType::from_raw_mode(before.st_mode) != expected_type
        || FileType::from_raw_mode(after.st_mode) != expected_type
        || before.st_dev != after.st_dev
        || before.st_ino != after.st_ino
    {
        return Err(BrainError::Integrity(label.into()));
    }
    Ok(())
}

fn ensure_private_directory_stable(before: &Stat, after: &Stat) -> BrainResult<()> {
    ensure_same_private_tree_object(
        before,
        after,
        FileType::Directory,
        "private_directory_tree_directory_replaced",
    )?;
    if before.st_mode != after.st_mode
        || before.st_size != after.st_size
        || before.st_mtime != after.st_mtime
        || before.st_mtime_nsec != after.st_mtime_nsec
        || before.st_ctime != after.st_ctime
        || before.st_ctime_nsec != after.st_ctime_nsec
    {
        return Err(BrainError::Integrity(
            "private_directory_tree_mutated_during_inspection".into(),
        ));
    }
    Ok(())
}

fn private_tree_child_names(directory: &File) -> BrainResult<Vec<OsString>> {
    let mut stream = Dir::read_from(directory).map_err(rustix_error)?;
    let mut names = Vec::new();
    for entry in &mut stream {
        let entry = entry.map_err(rustix_error)?;
        let raw = entry.file_name().to_bytes();
        if raw == b"." || raw == b".." {
            continue;
        }
        names.push(OsString::from_vec(raw.to_vec()));
    }
    names.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    Ok(names)
}

/// Enumerate one existing private directory from a held descriptor and return
/// only its confined lexical child names. A concurrent namespace mutation
/// changes directory metadata and causes the enumeration to fail closed.
pub(crate) fn list_existing_private_directory(
    root: &Path,
    directory: &Path,
) -> BrainResult<Vec<PathBuf>> {
    let relative = root_relative_path(root, directory)?;
    let root_fd = open_private_root_fd(root)?;
    let directory_fd = open_private_dir_fd(&root_fd, &relative)?;
    let before = fstat(&directory_fd).map_err(rustix_error)?;
    let names = private_tree_child_names(&directory_fd)?;
    let after = fstat(&directory_fd).map_err(rustix_error)?;
    ensure_private_directory_stable(&before, &after)?;
    Ok(names
        .into_iter()
        .map(|name| root.join(&relative).join(name))
        .collect())
}

fn collect_private_tree_entries_descriptor_bound(
    root_stat: &Stat,
    directory: File,
    opened_directory_stat: Stat,
    relative_directory: &Path,
    entries: &mut Vec<PrivateTreeEntry>,
    visited_directories: &mut BTreeSet<(u64, u64)>,
    depth: usize,
) -> BrainResult<()> {
    if depth > PRIVATE_TREE_MAX_DEPTH {
        return Err(BrainError::Invalid(
            "private_directory_tree_depth_limit_exceeded".into(),
        ));
    }
    ensure_same_private_tree_object(
        &opened_directory_stat,
        &opened_directory_stat,
        FileType::Directory,
        "private_directory_tree_directory_invalid",
    )?;
    if !visited_directories.insert((opened_directory_stat.st_dev, opened_directory_stat.st_ino)) {
        return Err(BrainError::Integrity(
            "private_directory_tree_cycle_or_alias".into(),
        ));
    }

    let names = private_tree_child_names(&directory)?;
    for name in names {
        let child_relative = relative_directory.join(&name);
        let inspected_fd = openat2(
            &directory,
            Path::new(&name),
            OFlags::PATH | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
            PRIVATE_RESOLUTION,
        )
        .map_err(rustix_error)?;
        let inspected_stat = fstat(&inspected_fd).map_err(rustix_error)?;
        match FileType::from_raw_mode(inspected_stat.st_mode) {
            FileType::Directory => {
                let opened_fd = openat2(
                    &directory,
                    Path::new(&name),
                    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    Mode::empty(),
                    PRIVATE_RESOLUTION,
                )
                .map_err(rustix_error)?;
                let opened_stat = fstat(&opened_fd).map_err(rustix_error)?;
                ensure_same_private_tree_object(
                    &inspected_stat,
                    &opened_stat,
                    FileType::Directory,
                    "private_directory_tree_child_replaced",
                )?;
                entries.push(PrivateTreeEntry {
                    relative_path: child_relative.clone(),
                    kind: b'd',
                    file_size: 0,
                    file_sha256: None,
                });
                collect_private_tree_entries_descriptor_bound(
                    root_stat,
                    File::from(opened_fd),
                    opened_stat,
                    &child_relative,
                    entries,
                    visited_directories,
                    depth + 1,
                )?;
            }
            FileType::RegularFile => {
                let data_fd = openat2(
                    &directory,
                    Path::new(&name),
                    OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    Mode::empty(),
                    PRIVATE_RESOLUTION,
                )
                .map_err(rustix_error)?;
                let opened_stat = fstat(&data_fd).map_err(rustix_error)?;
                ensure_same_private_tree_object(
                    &inspected_stat,
                    &opened_stat,
                    FileType::RegularFile,
                    "private_directory_tree_child_replaced",
                )?;
                validate_private_file_stat(root_stat, &opened_stat)?;
                let file_size = u64::try_from(opened_stat.st_size).map_err(|_| {
                    BrainError::Integrity("private_directory_tree_file_size_invalid".into())
                })?;
                let mut file = File::from(data_fd);
                let digest = sha256_fd(&mut file)?;
                let final_stat = fstat(&file).map_err(rustix_error)?;
                ensure_private_file_stable(&opened_stat, &final_stat)?;
                entries.push(PrivateTreeEntry {
                    relative_path: child_relative,
                    kind: b'f',
                    file_size,
                    file_sha256: Some(digest),
                });
            }
            FileType::Symlink => {
                return Err(BrainError::Integrity(
                    "private_directory_tree_symlink_forbidden".into(),
                ));
            }
            _ => {
                return Err(BrainError::Integrity(
                    "private_directory_tree_special_file_forbidden".into(),
                ));
            }
        }
    }
    let final_directory_stat = fstat(&directory).map_err(rustix_error)?;
    ensure_private_directory_stable(&opened_directory_stat, &final_directory_stat)
}

/// Authenticate the complete contents and topology of a private directory.
/// The digest excludes the directory's own name, so it remains stable across
/// an authorized move from an inflight location to an archive location.
pub fn inspect_private_directory(
    root: &Path,
    directory: &Path,
) -> BrainResult<PrivateDirectoryIdentity> {
    let relative = root_relative_path(root, directory)?;
    let root_fd = open_private_root_fd(root)?;
    let root_stat = fstat(&root_fd).map_err(rustix_error)?;
    let directory_fd = open_private_dir_fd(&root_fd, &relative)?;
    let directory_stat = fstat(&directory_fd).map_err(rustix_error)?;
    let mut entries = Vec::new();
    let mut visited_directories = BTreeSet::new();
    collect_private_tree_entries_descriptor_bound(
        &root_stat,
        directory_fd,
        directory_stat,
        Path::new(""),
        &mut entries,
        &mut visited_directories,
        0,
    )?;
    entries.sort_by(|left, right| {
        left.relative_path
            .as_os_str()
            .as_bytes()
            .cmp(right.relative_path.as_os_str().as_bytes())
    });
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:PRIVATE-DIRECTORY-TREE:v1\0");
    let mut regular_file_count = 0u64;
    let mut total_file_bytes = 0u64;
    for entry in &entries {
        let path = entry.relative_path.as_os_str().as_bytes();
        hasher.update([entry.kind]);
        hasher.update((path.len() as u64).to_be_bytes());
        hasher.update(path);
        hasher.update(entry.file_size.to_be_bytes());
        if let Some(digest) = &entry.file_sha256 {
            hasher.update(digest.as_bytes());
            regular_file_count = regular_file_count.checked_add(1).ok_or_else(|| {
                BrainError::Invalid("private_directory_tree_count_overflow".into())
            })?;
            total_file_bytes = total_file_bytes
                .checked_add(entry.file_size)
                .ok_or_else(|| {
                    BrainError::Invalid("private_directory_tree_size_overflow".into())
                })?;
        }
    }
    Ok(PrivateDirectoryIdentity {
        tree_sha256: Sha256Digest::parse(format!("{:x}", hasher.finalize()))?,
        entry_count: entries.len() as u64,
        regular_file_count,
        total_file_bytes,
    })
}

/// Atomically move a complete private directory to a vacant destination.
///
/// Linux `renameat2(RENAME_NOREPLACE)` closes the preflight/rename overwrite
/// race. Parent directories are opened via `openat2` with BENEATH resolution
/// before the rename so the kernel holds the inodes independently of any
/// subsequent pathname change.  The post-rename destination open uses
/// `openat` relative to the held parent fd with `O_NOFOLLOW`, so no new
/// pathname traversal occurs after the rename.
pub fn move_private_directory_transactional(
    root: &Path,
    source: &Path,
    destination: &Path,
    expected: &PrivateDirectoryIdentity,
) -> BrainResult<()> {
    if source == destination {
        return Err(BrainError::Invalid(
            "private_directory_move_source_equals_destination".into(),
        ));
    }
    let Some(source) = existing_directory_if_present(root, source)? else {
        let destination_identity = inspect_private_directory(root, destination)?;
        if destination_identity != *expected {
            return Err(BrainError::Integrity(
                "private_directory_move_completed_identity_mismatch".into(),
            ));
        }
        return Ok(());
    };
    if inspect_private_directory(root, &source)? != *expected {
        return Err(BrainError::Integrity(
            "private_directory_move_source_identity_mismatch".into(),
        ));
    }
    if existing_directory_if_present(root, destination)?.is_some() {
        return Err(BrainError::Integrity(
            "private_directory_move_destination_already_exists".into(),
        ));
    }
    ensure_private_parent(root, destination)?;
    let source_parent = source.parent().ok_or_else(|| {
        BrainError::Invalid("private_directory_move_source_parent_missing".into())
    })?;
    let destination_parent = destination.parent().ok_or_else(|| {
        BrainError::Invalid("private_directory_move_destination_parent_missing".into())
    })?;
    let source_parent = existing_directory_under_root(root, source_parent)?;
    let destination_parent = existing_directory_under_root(root, destination_parent)?;
    let source_name = source
        .file_name()
        .ok_or_else(|| BrainError::Invalid("private_directory_move_source_name_missing".into()))?;
    let destination_name = destination.file_name().ok_or_else(|| {
        BrainError::Invalid("private_directory_move_destination_name_missing".into())
    })?;
    // Open parent directories as descriptors before the rename so they anchor
    // the operation independently of any subsequent pathname change.
    let root_fd = open_private_root_fd(root)?;
    let source_parent_relative = root_relative_path(root, &source_parent)?;
    let destination_parent_relative = root_relative_path(root, &destination_parent)?;
    let source_parent_file = if source_parent_relative.as_os_str().is_empty() {
        open_private_root_fd(root)?
    } else {
        open_private_dir_fd(&root_fd, &source_parent_relative)?
    };
    let destination_parent_file = if destination_parent_relative.as_os_str().is_empty() {
        open_private_root_fd(root)?
    } else {
        open_private_dir_fd(&root_fd, &destination_parent_relative)?
    };
    renameat_with(
        &source_parent_file,
        source_name,
        &destination_parent_file,
        destination_name,
        RenameFlags::NOREPLACE,
    )
    .map_err(|error| std::io::Error::from_raw_os_error(error.raw_os_error()))?;
    // Open the moved directory relative to the held destination parent fd
    // with O_NOFOLLOW, eliminating the post-rename pathname-open window.
    let dest_moved_fd = File::from(
        openat(
            &destination_parent_file,
            destination_name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(rustix_error)?,
    );
    fd_secure_dir(&dest_moved_fd)?;
    dest_moved_fd.sync_all()?;
    destination_parent_file.sync_all()?;
    if source_parent != destination_parent {
        source_parent_file.sync_all()?;
    }
    let installed = inspect_private_directory(root, destination)?;
    if installed != *expected {
        return Err(BrainError::Integrity(
            "private_directory_move_postinstall_identity_mismatch".into(),
        ));
    }
    Ok(())
}

/// Securely create only missing directory components below the private root.
/// Uses `mkdirat` relative to a running parent descriptor and `fchmod` through
/// that same descriptor, closing the create→chmod TOCTOU window per component.
pub fn ensure_private_parent(root: &Path, destination: &Path) -> BrainResult<()> {
    let relative = root_relative_path(root, destination)?;
    let parent = relative
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?;
    let mut current_fd = open_private_root_fd(root)?;
    for component in parent.components() {
        let Component::Normal(name) = component else {
            return Err(BrainError::Invalid("private_file_path_not_confined".into()));
        };
        let child_fd = match openat(
            &current_fd,
            name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        ) {
            Ok(raw_fd) => File::from(raw_fd),
            Err(e) if e == rustix::io::Errno::NOENT => {
                match mkdirat(&current_fd, name, Mode::from_bits_truncate(0o700)) {
                    Ok(()) => {}
                    Err(e) if e == rustix::io::Errno::EXIST => {}
                    Err(e) => return Err(rustix_error(e)),
                }
                let raw_fd = openat(
                    &current_fd,
                    name,
                    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(rustix_error)?;
                File::from(raw_fd)
            }
            Err(e) if e == rustix::io::Errno::NOTDIR || e == rustix::io::Errno::LOOP => {
                return Err(BrainError::Integrity(
                    "private_file_parent_not_directory".into(),
                ));
            }
            Err(e) => return Err(rustix_error(e)),
        };
        let stat = fstat(&child_fd).map_err(rustix_error)?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
            return Err(BrainError::Integrity(
                "private_file_parent_not_directory".into(),
            ));
        }
        fd_secure_dir(&child_fd)?;
        current_fd = child_fd;
    }
    Ok(())
}

/// Create a private directory tree only through non-symlink components, then
/// return the exact directory path.  Uses `mkdirat` relative to a running
/// parent descriptor and `fchmod` through that descriptor, closing the
/// create→chmod TOCTOU window per component.
pub fn ensure_private_directory(root: &Path, directory: &Path) -> BrainResult<PathBuf> {
    let relative = root_relative_path(root, directory)?;
    let mut current_fd = open_private_root_fd(root)?;
    let mut cursor = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(name) = component else {
            return Err(BrainError::Invalid(
                "private_directory_path_not_confined".into(),
            ));
        };
        cursor.push(name);
        let child_fd = match openat(
            &current_fd,
            name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        ) {
            Ok(raw_fd) => File::from(raw_fd),
            Err(e) if e == rustix::io::Errno::NOENT => {
                match mkdirat(&current_fd, name, Mode::from_bits_truncate(0o700)) {
                    Ok(()) => {}
                    Err(e) if e == rustix::io::Errno::EXIST => {}
                    Err(e) => return Err(rustix_error(e)),
                }
                let raw_fd = openat(
                    &current_fd,
                    name,
                    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(rustix_error)?;
                File::from(raw_fd)
            }
            Err(e) if e == rustix::io::Errno::NOTDIR || e == rustix::io::Errno::LOOP => {
                return Err(BrainError::Integrity(
                    "private_directory_target_invalid".into(),
                ));
            }
            Err(e) => return Err(rustix_error(e)),
        };
        let stat = fstat(&child_fd).map_err(rustix_error)?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
            return Err(BrainError::Integrity(
                "private_directory_target_invalid".into(),
            ));
        }
        fd_secure_dir(&child_fd)?;
        current_fd = child_fd;
    }
    Ok(cursor)
}

/// Install immutable bytes or verify the exact existing content. No overwrite
/// path exists: a content mismatch is an integrity failure.
pub fn write_or_verify_immutable(
    root: &Path,
    path: &Path,
    bytes: &[u8],
) -> BrainResult<Sha256Digest> {
    let expected = Sha256Digest::digest_bytes(bytes);
    if existing_regular_file_if_present(root, path)?.is_some() {
        verify_immutable_content(root, path, &expected)?;
        return Ok(expected);
    }
    let temporary = stage_private_bytes(root, path, bytes, &expected)?;
    if !install_private_immutable_file(root, &temporary, path, &expected)? {
        verify_immutable_content(root, path, &expected)?;
    }
    Ok(expected)
}

/// Create a new immutable file.  Unlike `write_or_verify_immutable`, any
/// existing leaf is an integrity error even when its bytes happen to match.
/// This prevents callers that reserve a fresh receipt or evidence identity
/// from silently accepting a pre-existing authority object.
pub fn create_private_immutable(
    root: &Path,
    path: &Path,
    bytes: &[u8],
) -> BrainResult<Sha256Digest> {
    if existing_regular_file_if_present(root, path)?.is_some() {
        return Err(BrainError::Integrity(
            "private_file_immutable_target_already_exists".into(),
        ));
    }
    let expected = Sha256Digest::digest_bytes(bytes);
    let temporary = stage_private_bytes(root, path, bytes, &expected)?;
    if install_private_immutable_file(root, &temporary, path, &expected)? {
        Ok(expected)
    } else {
        Err(BrainError::Integrity(
            "private_file_immutable_target_already_exists".into(),
        ))
    }
}

/// Durably install a fully-written private staging file as an immutable object.
///
/// The staging file is synced and checked against `expected` before it becomes
/// visible. Installation uses a hard link, which is atomic and cannot replace
/// an existing leaf. The destination file and affected parent directory are
/// synced before success is returned, then the installed bytes are verified
/// again. `Ok(false)` means that another object already occupied the target;
/// the caller must decide whether that existing object is an idempotent match
/// or a collision. The staging file is removed on every terminal outcome.
pub fn install_private_immutable_file(
    root: &Path,
    temporary: &Path,
    destination: &Path,
    expected: &Sha256Digest,
) -> BrainResult<bool> {
    if temporary == destination {
        return Err(BrainError::Invalid(
            "private_file_staging_equals_destination".into(),
        ));
    }
    ensure_private_parent(root, destination)?;
    let root_fd = open_private_root_fd(root)?;

    // Staging file: descriptor-bound open + verify
    let staging_relative = root_relative_path(root, temporary)?;
    let staging_parent_relative = staging_relative
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?
        .to_path_buf();
    let staging_name = temporary
        .file_name()
        .ok_or_else(|| BrainError::Invalid("private_file_name_invalid".into()))?;
    let staging_parent_fd = if staging_parent_relative.as_os_str().is_empty() {
        open_private_root_fd(root)?
    } else {
        open_private_dir_fd(&root_fd, &staging_parent_relative)?
    };
    let opened_staging = open_private_reference(root, temporary)?;
    let staging_identity = PrivateStagingIdentity::from_stat(&opened_staging.stat);
    let mut staging_fd = opened_staging.file;

    // Destination parent fd
    let dest_relative = root_relative_path(root, destination)?;
    let dest_parent_relative = dest_relative
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?
        .to_path_buf();
    let dest_name = destination
        .file_name()
        .ok_or_else(|| BrainError::Invalid("private_file_name_invalid".into()))?;
    let dest_parent_fd = if dest_parent_relative.as_os_str().is_empty() {
        open_private_root_fd(root)?
    } else {
        open_private_dir_fd(&root_fd, &dest_parent_relative)?
    };

    // Secure, sync and verify staging content entirely through the open descriptor.
    let prepared = (|| -> BrainResult<()> {
        fd_secure_file(&staging_fd)?;
        staging_fd.sync_all()?;
        if sha256_fd(&mut staging_fd)? != *expected {
            return Err(BrainError::Integrity(
                "private_file_staged_content_mismatch".into(),
            ));
        }
        Ok(())
    })();
    if let Err(error) = prepared {
        return match unlink_name_if_same_inode(
            &staging_parent_fd,
            staging_name,
            &staging_identity,
            "private_file_staging_replaced_before_cleanup",
        ) {
            Ok(()) => Err(error),
            Err(cleanup_error) => Err(cleanup_error),
        };
    }

    // Create hard link via /proc/self/fd/<n> so the verified staging inode —
    // not the staging pathname — becomes the link target, eliminating the
    // staging-path TOCTOU window entirely.
    let proc_path = format!("/proc/self/fd/{}", staging_fd.as_raw_fd());
    let installed = match linkat(
        &staging_parent_fd,
        proc_path.as_str(),
        &dest_parent_fd,
        dest_name,
        AtFlags::SYMLINK_FOLLOW,
    ) {
        Ok(()) => true,
        Err(e) if e == rustix::io::Errno::EXIST => {
            if let Err(validation_error) = open_existing_private_file(root, destination) {
                return match unlink_name_if_same_inode(
                    &staging_parent_fd,
                    staging_name,
                    &staging_identity,
                    "private_file_staging_replaced_before_cleanup",
                ) {
                    Ok(()) => Err(validation_error),
                    Err(cleanup_error) => Err(cleanup_error),
                };
            }
            false
        }
        Err(e) => {
            let operation_error = rustix_error(e);
            return match unlink_name_if_same_inode(
                &staging_parent_fd,
                staging_name,
                &staging_identity,
                "private_file_staging_replaced_before_cleanup",
            ) {
                Ok(()) => Err(operation_error),
                Err(cleanup_error) => Err(cleanup_error),
            };
        }
    };

    // Post-link: open destination relative to the held parent fd with O_NOFOLLOW.
    let finalize = (|| -> BrainResult<()> {
        if installed {
            let dest_fd = File::from(
                openat(
                    &dest_parent_fd,
                    dest_name,
                    OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(rustix_error)?,
            );
            fd_secure_file(&dest_fd)?;
            dest_fd.sync_all()?;
            dest_parent_fd.sync_all()?;
        }
        Ok(())
    })();
    let cleanup = (|| -> BrainResult<()> {
        unlink_name_if_same_inode(
            &staging_parent_fd,
            staging_name,
            &staging_identity,
            "private_file_staging_replaced_before_cleanup",
        )?;
        if dest_parent_relative != staging_parent_relative && installed {
            dest_parent_fd.sync_all()?;
        }
        Ok(())
    })();
    finalize?;
    cleanup?;

    if installed {
        verify_immutable_content(root, destination, expected)?;
    }
    Ok(installed)
}

/// Move a private regular file to a vacant private destination without an
/// overwrite window, with crash-recoverable link-then-unlink semantics.
///
/// Source is opened via descriptor-bound resolution and content-verified by fd.
/// The hard link is created via `/proc/self/fd/<n>` so the verified source inode
/// becomes the link target.  Post-link operations use `openat` relative to the
/// held destination parent fd with `O_NOFOLLOW`; source removal uses `unlinkat`
/// relative to the held source parent fd.
///
/// If a crash leaves both names visible, a retry completes the transaction
/// only when both names identify the same inode. A distinct pre-existing
/// destination is always an integrity failure, even when its bytes match.
pub fn move_private_file_transactional(
    root: &Path,
    source: &Path,
    destination: &Path,
    expected: &Sha256Digest,
) -> BrainResult<()> {
    if source == destination {
        return Err(BrainError::Invalid(
            "private_file_move_source_equals_destination".into(),
        ));
    }
    let Some(source_path) = existing_regular_file_if_present(root, source)? else {
        // Completed crash/retry state: the source name was already durably
        // removed. Exact destination identity is the only accepted witness.
        verify_immutable_content(root, destination, expected)?;
        return Ok(());
    };
    // Open and verify source content through descriptor-bound resolution.
    let opened_source = open_private_reference(root, &source_path)?;
    let mut source_fd = opened_source.file;
    if sha256_fd(&mut source_fd)? != *expected {
        return Err(BrainError::Integrity(
            "private_file_move_source_digest_mismatch".into(),
        ));
    }
    let source_stat = fstat(&source_fd).map_err(rustix_error)?;
    ensure_private_parent(root, destination)?;
    let root_fd = open_private_root_fd(root)?;
    let source_relative = root_relative_path(root, &source_path)?;
    let source_parent_relative = source_relative
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?
        .to_path_buf();
    let source_name = source_path
        .file_name()
        .ok_or_else(|| BrainError::Invalid("private_file_name_invalid".into()))?;
    let source_parent_fd = if source_parent_relative.as_os_str().is_empty() {
        open_private_root_fd(root)?
    } else {
        open_private_dir_fd(&root_fd, &source_parent_relative)?
    };
    let dest_relative = root_relative_path(root, destination)?;
    let dest_parent_relative = dest_relative
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?
        .to_path_buf();
    let dest_name = destination
        .file_name()
        .ok_or_else(|| BrainError::Invalid("private_file_name_invalid".into()))?;
    let dest_parent_fd = if dest_parent_relative.as_os_str().is_empty() {
        open_private_root_fd(root)?
    } else {
        open_private_dir_fd(&root_fd, &dest_parent_relative)?
    };
    // Link via /proc/self/fd/<n> to bind the verified source inode to dest.
    let proc_path = format!("/proc/self/fd/{}", source_fd.as_raw_fd());
    let dest_fd = match linkat(
        &source_parent_fd,
        proc_path.as_str(),
        &dest_parent_fd,
        dest_name,
        AtFlags::SYMLINK_FOLLOW,
    ) {
        Ok(()) => File::from(
            openat(
                &dest_parent_fd,
                dest_name,
                OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(rustix_error)?,
        ),
        Err(e) if e == rustix::io::Errno::EXIST => {
            let dest_fd = File::from(
                openat(
                    &dest_parent_fd,
                    dest_name,
                    OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(rustix_error)?,
            );
            let dest_stat = fstat(&dest_fd).map_err(rustix_error)?;
            if source_stat.st_dev != dest_stat.st_dev || source_stat.st_ino != dest_stat.st_ino {
                return Err(BrainError::Integrity(
                    "private_file_move_destination_already_exists".into(),
                ));
            }
            dest_fd
        }
        Err(e) => return Err(rustix_error(e)),
    };
    fd_secure_file(&dest_fd)?;
    dest_fd.sync_all()?;
    dest_parent_fd.sync_all()?;
    // Revalidate source inode before unlinking: if source_name was replaced
    // between linkat and here, we must not remove the intruder's file.
    let current_source = File::from(
        openat(
            &source_parent_fd,
            source_name,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(rustix_error)?,
    );
    let current_stat = fstat(&current_source).map_err(rustix_error)?;
    drop(current_source);
    if current_stat.st_dev != source_stat.st_dev || current_stat.st_ino != source_stat.st_ino {
        return Err(BrainError::Integrity(
            "private_file_move_source_replaced_before_unlink".into(),
        ));
    }
    unlinkat(&source_parent_fd, source_name, AtFlags::empty()).map_err(rustix_error)?;
    source_parent_fd.sync_all()?;
    verify_immutable_content(root, destination, expected)
}

fn stage_private_bytes(
    root: &Path,
    path: &Path,
    bytes: &[u8],
    expected: &Sha256Digest,
) -> BrainResult<PathBuf> {
    let (temporary, actual, created_identity) =
        stage_private_file_with_identity(root, path, |file| {
            file.write_all(bytes)?;
            Ok(())
        })?;
    if actual != *expected {
        unlink_private_file_if_same_inode(
            root,
            &temporary,
            &created_identity,
            "private_file_staging_replaced_before_cleanup",
        )?;
        return Err(BrainError::Integrity(
            "private_file_staged_content_mismatch".into(),
        ));
    }
    Ok(temporary)
}

fn verify_immutable_content(root: &Path, path: &Path, expected: &Sha256Digest) -> BrainResult<()> {
    let relative = root_relative_path(root, path)?;
    let root_fd = open_private_root_fd(root)?;
    let root_stat = fstat(&root_fd).map_err(rustix_error)?;
    let mut file_fd = File::from(
        openat2(
            &root_fd,
            &relative,
            OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
            PRIVATE_RESOLUTION,
        )
        .map_err(rustix_error)?,
    );
    let file_stat = fstat(&file_fd).map_err(rustix_error)?;
    validate_private_file_stat(&root_stat, &file_stat)?;
    if sha256_fd(&mut file_fd)? != *expected {
        return Err(BrainError::Integrity(
            "private_file_immutable_content_mismatch".into(),
        ));
    }
    fd_secure_file(&file_fd)
}

/// Atomically replace a mutable private file after validating its entire path.
/// Immutable receipts and artifacts must use an immutable writer instead.  The
/// caller remains responsible for deciding which pointer paths are mutable.
///
/// The entire write chain is descriptor-bound:
/// - Temp file is created with `openat(parent_fd, …, CREAT|EXCL)`, relative to
///   the verified parent directory descriptor.
/// - Rename uses `renameat(parent_fd, temp, parent_fd, dest)` — no pathname.
/// - Destination is re-opened with `openat(parent_fd, dest, O_NOFOLLOW)` so
///   an adversary cannot redirect the post-rename open to a symlink.
/// - All chmod, hash and sync operations use the destination fd directly.
pub fn replace_private_file_atomic(
    root: &Path,
    path: &Path,
    bytes: &[u8],
    expected_sha256: Option<&Sha256Digest>,
) -> BrainResult<Sha256Digest> {
    replace_private_file_atomic_inner(root, path, bytes, expected_sha256, || Ok(()))
}

fn replace_private_file_atomic_inner<F>(
    root: &Path,
    path: &Path,
    bytes: &[u8],
    expected_sha256: Option<&Sha256Digest>,
    after_rename: F,
) -> BrainResult<Sha256Digest>
where
    F: FnOnce() -> BrainResult<()>,
{
    let digest = Sha256Digest::digest_bytes(bytes);
    if expected_sha256.is_some_and(|expected| expected != &digest) {
        return Err(BrainError::Integrity(
            "private_file_atomic_expected_digest_mismatch".into(),
        ));
    }
    // Preflight path validation via pathname (read-only, not security-critical
    // for TOCTOU since all subsequent writes are anchored to the parent fd).
    let _ = existing_regular_file_if_present(root, path)?;
    ensure_private_parent(root, path)?;
    let file_name = path
        .file_name()
        .and_then(|v| v.to_str())
        .ok_or_else(|| BrainError::Invalid("private_file_name_invalid".into()))?;
    let relative = root_relative_path(root, path)?;
    let parent_relative = relative
        .parent()
        .ok_or_else(|| BrainError::Invalid("private_file_parent_missing".into()))?
        .to_path_buf();
    // Open root and parent via descriptor-bound resolution.  All subsequent
    // operations use fds anchored to these inodes; no pathname is reopened.
    let root_fd = open_private_root_fd(root)?;
    let parent_fd = if parent_relative.as_os_str().is_empty() {
        open_private_root_fd(root)?
    } else {
        open_private_dir_fd(&root_fd, &parent_relative)?
    };
    // Create temp relative to the verified parent descriptor (CREAT|EXCL).
    let sequence = NEXT_IMMUTABLE_TEMPORARY.fetch_add(1, Ordering::Relaxed);
    let temp_name = format!(
        ".{file_name}.{}.{}.atomic.tmp",
        std::process::id(),
        sequence
    );
    let mut temp_file = File::from(
        openat(
            &parent_fd,
            temp_name.as_str(),
            OFlags::RDWR | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC,
            Mode::from_bits_truncate(0o600),
        )
        .map_err(|e| {
            if e == rustix::io::Errno::EXIST {
                BrainError::Integrity("private_file_atomic_temporary_exists".into())
            } else {
                rustix_error(e)
            }
        })?,
    );
    // Write, secure and sync through the open descriptor.
    let write_result = (|| -> BrainResult<()> {
        temp_file.write_all(bytes)?;
        fd_secure_file(&temp_file)?;
        temp_file.sync_all()?;
        Ok(())
    })();
    if let Err(e) = write_result {
        let _ = unlinkat(&parent_fd, temp_name.as_str(), AtFlags::empty());
        let _ = parent_fd.sync_all();
        return Err(e);
    }
    let temp_stat = fstat(&temp_file).map_err(rustix_error)?;
    // Atomic rename via fd-relative renameat — replaces any existing dest.
    // For mutable pointer files this is intentional; immutable objects use
    // NOREPLACE via install_private_immutable_file instead.
    if let Err(e) = renameat(&parent_fd, temp_name.as_str(), &parent_fd, file_name) {
        let _ = unlinkat(&parent_fd, temp_name.as_str(), AtFlags::empty());
        let _ = parent_fd.sync_all();
        return Err(BrainError::Io(std::io::Error::from_raw_os_error(
            e.raw_os_error(),
        )));
    }
    after_rename()?;
    // Re-open destination relative to the same parent descriptor with O_NOFOLLOW.
    // The parent_fd holds the inode that just accepted the rename, so no
    // adversarial substitution can redirect this open to a different file.
    let mut dest_fd = File::from(
        openat(
            &parent_fd,
            file_name,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(rustix_error)?,
    );
    // Verify permissions, ownership and content through the descriptor.
    let root_stat = fstat(&root_fd).map_err(rustix_error)?;
    let dest_stat = fstat(&dest_fd).map_err(rustix_error)?;
    validate_private_file_stat(&root_stat, &dest_stat)?;
    if dest_stat.st_dev != temp_stat.st_dev || dest_stat.st_ino != temp_stat.st_ino {
        return Err(BrainError::Integrity(
            "private_file_atomic_postrename_identity_mismatch".into(),
        ));
    }
    fd_secure_file(&dest_fd)?;
    if sha256_fd(&mut dest_fd)? != digest {
        return Err(BrainError::Integrity(
            "private_file_atomic_postwrite_mismatch".into(),
        ));
    }
    dest_fd.sync_all()?;
    let rebound_fd = File::from(
        openat(
            &parent_fd,
            file_name,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(rustix_error)?,
    );
    let rebound_stat = fstat(&rebound_fd).map_err(rustix_error)?;
    if rebound_stat.st_dev != temp_stat.st_dev || rebound_stat.st_ino != temp_stat.st_ino {
        return Err(BrainError::Integrity(
            "private_file_atomic_destination_replaced_after_verification".into(),
        ));
    }
    parent_fd.sync_all()?;
    Ok(digest)
}

/// Execute a critical section guarded by a descriptor-bound, process-safe lock
/// under the private authority root.  The lock object is immutable and carries
/// no semantic state; the advisory lock is released automatically if a process
/// terminates, so a crash cannot leave a stale ownership marker behind.
///
/// This primitive is intentionally separate from mutable payload writes.  A
/// caller must still read and compare the current authoritative value while
/// holding the lock, then use [`replace_private_file_atomic`] to publish it.
pub fn with_private_authority_lock<T, F>(
    root: &Path,
    lock_path: &Path,
    operation: F,
) -> BrainResult<T>
where
    F: FnOnce() -> BrainResult<T>,
{
    write_or_verify_immutable(root, lock_path, b"")?;
    let opened = open_private_reference(root, lock_path)?;
    flock(&opened.file, FlockOperation::LockExclusive).map_err(rustix_error)?;
    let result = operation();
    // Explicit release makes failures observable on platforms where close is
    // delayed.  The descriptor's drop remains the crash-safe fallback.
    let unlock = flock(&opened.file, FlockOperation::Unlock).map_err(rustix_error);
    match (result, unlock) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::secure_dir;
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cerebro-authority-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        secure_dir(&root).unwrap();
        root
    }

    #[test]
    fn private_reference_verifies_exact_bytes() {
        let root = root("reference");
        let path = root.join("state/evidence.bin");
        let digest = write_or_verify_immutable(&root, &path, b"evidence").unwrap();
        let reference = PrivateFileReference::new(path.clone(), digest);
        assert_eq!(reference.read_verified(&root).unwrap(), b"evidence");
        fs::write(&path, b"tampered").unwrap();
        assert!(reference.read_verified(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bounded_reference_read_rejects_oversize_without_allocating_the_payload() {
        let root = root("bounded-reference");
        let path = root.join("state/evidence.bin");
        let payload = vec![b'x'; PRIVATE_READ_BUFFER_BYTES + 1];
        let digest = write_or_verify_immutable(&root, &path, &payload).unwrap();
        let reference = PrivateFileReference::new(path, digest);

        assert!(reference
            .read_verified_bounded(&root, PRIVATE_READ_BUFFER_BYTES as u64)
            .is_err());
        assert_eq!(
            reference
                .read_verified_bounded(&root, payload.len() as u64)
                .unwrap(),
            payload
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn descriptor_resolution_rejects_leaf_and_parent_symlinks() {
        let root = root("descriptor-symlinks");
        let target = root.join("state/target.bin");
        let target_digest = write_or_verify_immutable(&root, &target, b"target").unwrap();

        let leaf = root.join("state/leaf.bin");
        symlink(&target, &leaf).unwrap();
        let leaf_reference = PrivateFileReference::new(leaf, target_digest.clone());
        assert!(leaf_reference.read_verified_bounded(&root, 64).is_err());

        let real_parent = root.join("real-parent");
        ensure_private_directory(&root, &real_parent).unwrap();
        let nested = real_parent.join("nested.bin");
        write_or_verify_immutable(&root, &nested, b"nested").unwrap();
        let linked_parent = root.join("linked-parent");
        symlink(&real_parent, &linked_parent).unwrap();
        let parent_reference = PrivateFileReference::new(
            linked_parent.join("nested.bin"),
            Sha256Digest::digest_bytes(b"nested"),
        );
        assert!(parent_reference.read_verified_bounded(&root, 64).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn replacement_after_open_fails_closed_without_accepting_replacement_bytes() {
        let root = root("descriptor-replacement");
        let path = root.join("state/object.bin");
        let replacement = root.join("state/replacement.bin");
        let digest = write_or_verify_immutable(&root, &path, b"original").unwrap();
        write_or_verify_immutable(&root, &replacement, b"hostile!").unwrap();
        let reference = PrivateFileReference::new(path.clone(), digest);

        let result = reference.read_verified_bounded_after_open(&root, 64, || {
            fs::rename(&replacement, &path).unwrap();
        });
        assert!(result.is_err());
        assert_eq!(fs::read(path).unwrap(), b"hostile!");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mutation_race_on_the_open_inode_fails_closed() {
        let root = root("descriptor-mutation");
        let path = root.join("state/object.bin");
        let digest = write_or_verify_immutable(&root, &path, b"original").unwrap();
        let reference = PrivateFileReference::new(path.clone(), digest);

        assert!(reference
            .read_verified_bounded_after_open(&root, 64, || {
                fs::write(&path, b"mutated!").unwrap();
            })
            .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn writable_by_other_users_is_not_private_authority_data() {
        let root = root("descriptor-permissions");
        let path = root.join("state/object.bin");
        let digest = write_or_verify_immutable(&root, &path, b"payload").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o622)).unwrap();
        let reference = PrivateFileReference::new(path, digest);
        assert!(reference.read_verified_bounded(&root, 64).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fresh_immutable_write_never_accepts_a_preexisting_leaf() {
        let root = root("fresh-immutable");
        let path = root.join("state/receipt.json");
        create_private_immutable(&root, &path, b"first").unwrap();
        assert!(create_private_immutable(&root, &path, b"first").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn streamed_staging_install_is_synced_private_verified_and_clean() {
        let root = root("streamed-install");
        let path = root.join("state/objects/payload.bin");
        let (temporary, mut file) = create_private_staging_file(&root, &path).unwrap();
        file.write_all(b"streamed immutable payload").unwrap();
        drop(file);
        let expected = Sha256Digest::digest_bytes(b"streamed immutable payload");

        assert!(install_private_immutable_file(&root, &temporary, &path, &expected).unwrap());
        assert!(!temporary.exists());
        assert_eq!(fs::read(&path).unwrap(), b"streamed immutable payload");
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        verify_immutable_content(&root, &path, &expected).unwrap();
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn staged_digest_mismatch_fails_closed_and_removes_staging() {
        let root = root("staged-mismatch");
        let path = root.join("state/object.bin");
        let (temporary, mut file) = create_private_staging_file(&root, &path).unwrap();
        file.write_all(b"actual").unwrap();
        drop(file);
        let wrong = Sha256Digest::digest_bytes(b"expected");

        assert!(install_private_immutable_file(&root, &temporary, &path, &wrong).is_err());
        assert!(!temporary.exists());
        assert!(!path.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_streaming_producer_leaves_no_partial_file() {
        let root = root("streaming-failure");
        let path = root.join("state/object.bin");
        let result = stage_private_file(&root, &path, |file| {
            file.write_all(b"partial")?;
            Err(BrainError::Integrity("injected_stream_failure".into()))
        });
        assert!(result.is_err());
        assert!(!path.exists());
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn staging_installer_rejects_symlink_collision_without_touching_target() {
        let root = root("installer-symlink");
        let outside = root.join("outside.bin");
        fs::write(&outside, b"outside").unwrap();
        let destination = root.join("state/object.bin");
        let (temporary, mut file) = create_private_staging_file(&root, &destination).unwrap();
        file.write_all(b"inside").unwrap();
        drop(file);
        symlink(&outside, &destination).unwrap();
        let expected = Sha256Digest::digest_bytes(b"inside");

        assert!(
            install_private_immutable_file(&root, &temporary, &destination, &expected).is_err()
        );
        assert!(!temporary.exists());
        assert_eq!(fs::read(outside).unwrap(), b"outside");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn transactional_move_never_overwrites_and_recovers_its_own_link() {
        let root = root("transactional-move");
        let source = root.join("live/object.bin");
        let destination = root.join("archive/object.bin");
        ensure_private_parent(&root, &source).unwrap();
        fs::write(&source, b"payload").unwrap();
        let expected = Sha256Digest::digest_bytes(b"payload");

        // Reproduce the only intermediate crash state: destination linked and
        // synced, but source name not unlinked yet.
        ensure_private_parent(&root, &destination).unwrap();
        fs::hard_link(&source, &destination).unwrap();
        move_private_file_transactional(&root, &source, &destination, &expected).unwrap();
        assert!(!source.exists());
        assert_eq!(fs::read(&destination).unwrap(), b"payload");
        move_private_file_transactional(&root, &source, &destination, &expected).unwrap();

        let second_source = root.join("live/second.bin");
        fs::write(&second_source, b"other").unwrap();
        let other = Sha256Digest::digest_bytes(b"other");
        assert!(
            move_private_file_transactional(&root, &second_source, &destination, &other).is_err()
        );
        assert_eq!(fs::read(&second_source).unwrap(), b"other");
        assert_eq!(fs::read(&destination).unwrap(), b"payload");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn directory_move_preserves_complete_identity_and_retry_is_idempotent() {
        let root = root("directory-move-retry");
        let source = root.join("inflight/operation");
        let destination = root.join("archive/operation");
        ensure_private_directory(&root, &source.join("nested")).unwrap();
        fs::write(source.join("intent.json"), b"intent").unwrap();
        fs::write(source.join("nested/evidence.bin"), b"evidence").unwrap();
        let expected = inspect_private_directory(&root, &source).unwrap();

        move_private_directory_transactional(&root, &source, &destination, &expected).unwrap();
        assert!(!source.exists());
        assert_eq!(
            inspect_private_directory(&root, &destination).unwrap(),
            expected
        );
        move_private_directory_transactional(&root, &source, &destination, &expected).unwrap();
        assert_eq!(
            fs::read(destination.join("nested/evidence.bin")).unwrap(),
            b"evidence"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn directory_move_rejects_collision_and_tree_tampering_without_mutation() {
        let root = root("directory-move-collision");
        let source = root.join("inflight/operation");
        let destination = root.join("archive/operation");
        ensure_private_directory(&root, &source).unwrap();
        ensure_private_directory(&root, &destination).unwrap();
        fs::write(source.join("intent.json"), b"source").unwrap();
        fs::write(destination.join("intent.json"), b"destination").unwrap();
        let expected = inspect_private_directory(&root, &source).unwrap();

        assert!(
            move_private_directory_transactional(&root, &source, &destination, &expected).is_err()
        );
        assert_eq!(fs::read(source.join("intent.json")).unwrap(), b"source");
        assert_eq!(
            fs::read(destination.join("intent.json")).unwrap(),
            b"destination"
        );

        fs::remove_dir_all(&destination).unwrap();
        fs::write(source.join("intent.json"), b"tampered").unwrap();
        assert!(
            move_private_directory_transactional(&root, &source, &destination, &expected).is_err()
        );
        assert!(source.exists());
        assert!(!destination.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn directory_identity_and_move_reject_nested_symlinks() {
        let root = root("directory-move-symlink");
        let source = root.join("inflight/operation");
        let destination = root.join("archive/operation");
        ensure_private_directory(&root, &source).unwrap();
        let outside = root.join("outside.bin");
        fs::write(&outside, b"outside").unwrap();
        symlink(&outside, source.join("redirect.bin")).unwrap();

        assert!(inspect_private_directory(&root, &source).is_err());
        assert!(!destination.exists());
        assert_eq!(fs::read(outside).unwrap(), b"outside");

        fs::remove_file(source.join("redirect.bin")).unwrap();
        fs::write(source.join("intent.json"), b"intent").unwrap();
        let expected = inspect_private_directory(&root, &source).unwrap();
        let redirected_parent = root.join("redirected-archive");
        fs::create_dir(&redirected_parent).unwrap();
        symlink(&redirected_parent, root.join("archive")).unwrap();
        assert!(
            move_private_directory_transactional(&root, &source, &destination, &expected).is_err()
        );
        assert!(source.exists());
        assert_eq!(fs::read_dir(redirected_parent).unwrap().count(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn concurrent_directory_moves_never_replace_the_winner() {
        let root = Arc::new(root("directory-move-race"));
        let destination = root.join("archive/operation");
        let barrier = Arc::new(Barrier::new(2));
        let mut moves = Vec::new();
        for index in 0..2 {
            let source = root.join(format!("inflight/operation-{index}"));
            ensure_private_directory(&root, &source).unwrap();
            fs::write(source.join("intent.json"), format!("intent-{index}")).unwrap();
            let expected = inspect_private_directory(&root, &source).unwrap();
            let root = Arc::clone(&root);
            let destination = destination.clone();
            let barrier = Arc::clone(&barrier);
            moves.push(thread::spawn(move || {
                barrier.wait();
                move_private_directory_transactional(&root, &source, &destination, &expected)
            }));
        }
        let successes = moves
            .into_iter()
            .map(|operation| usize::from(operation.join().unwrap().is_ok()))
            .sum::<usize>();
        assert_eq!(successes, 1);
        let installed = fs::read(destination.join("intent.json")).unwrap();
        assert!(installed == b"intent-0" || installed == b"intent-1");
        fs::remove_dir_all(root.as_ref()).unwrap();
    }

    #[test]
    fn concurrent_idempotent_writers_install_one_complete_object() {
        let root = Arc::new(root("concurrent-same-content"));
        let path = root.join("state/object.bin");
        let barrier = Arc::new(Barrier::new(8));
        let writers = (0..8)
            .map(|_| {
                let root = Arc::clone(&root);
                let path = path.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    write_or_verify_immutable(&root, &path, b"complete immutable object")
                })
            })
            .collect::<Vec<_>>();
        for writer in writers {
            assert!(writer.join().unwrap().is_ok());
        }
        assert_eq!(fs::read(&path).unwrap(), b"complete immutable object");
        assert_eq!(
            fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1,
            "no staging file may survive a successful race"
        );
        fs::remove_dir_all(root.as_ref()).unwrap();
    }

    #[test]
    fn concurrent_conflicting_writers_never_replace_the_winner() {
        let root = Arc::new(root("concurrent-conflicting-content"));
        let path = root.join("state/object.bin");
        let barrier = Arc::new(Barrier::new(2));
        let writers = [b"first".as_slice(), b"second".as_slice()]
            .into_iter()
            .map(|payload| {
                let root = Arc::clone(&root);
                let path = path.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    write_or_verify_immutable(&root, &path, payload)
                })
            })
            .collect::<Vec<_>>();
        let successes = writers
            .into_iter()
            .map(|writer| usize::from(writer.join().unwrap().is_ok()))
            .sum::<usize>();
        assert_eq!(successes, 1);
        let installed = fs::read(&path).unwrap();
        assert!(installed == b"first" || installed == b"second");
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
        fs::remove_dir_all(root.as_ref()).unwrap();
    }

    #[test]
    fn optional_resolver_rejects_a_symlinked_missing_prefix() {
        let root = root("optional-symlink");
        let outside =
            std::env::temp_dir().join(format!("cerebro-authority-outside-{}", std::process::id()));
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, root.join("state")).unwrap();
        assert!(existing_regular_file_if_present(&root, &root.join("state/future.json")).is_err());
        fs::remove_file(root.join("state")).unwrap();
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn atomic_pointer_replacement_rejects_postrename_inode_substitution() {
        let root = root("atomic-pointer-postrename-race");
        let path = root.join("state/current.json");
        let hostile = root.join("state/hostile.json");
        ensure_private_parent(&root, &path).unwrap();
        fs::write(&hostile, b"trusted").unwrap();
        let expected = Sha256Digest::digest_bytes(b"trusted");

        let result =
            replace_private_file_atomic_inner(&root, &path, b"trusted", Some(&expected), || {
                fs::rename(&hostile, &path)?;
                Ok(())
            });
        assert!(matches!(
            result,
            Err(BrainError::Integrity(message))
                if message == "private_file_atomic_postrename_identity_mismatch"
        ));
        assert_eq!(fs::read(&path).unwrap(), b"trusted");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn atomic_pointer_replacement_verifies_requested_bytes() {
        let root = root("atomic-pointer");
        let path = root.join("state/current.json");
        let first = Sha256Digest::digest_bytes(b"first");
        replace_private_file_atomic(&root, &path, b"first", Some(&first)).unwrap();
        let second = Sha256Digest::digest_bytes(b"second");
        replace_private_file_atomic(&root, &path, b"second", Some(&second)).unwrap();
        assert_eq!(fs::read(path).unwrap(), b"second");
        assert!(replace_private_file_atomic(
            &root,
            &root.join("state/bad.json"),
            b"x",
            Some(&second)
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
