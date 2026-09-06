//! Fail-closed execution of an authenticated program without exposing TIDE-X state.
//!
//! This module intentionally accepts bytes, not paths. Before launch it verifies
//! their declared SHA-256 identities and writes exactly one program and one input
//! to sealed anonymous memory files. Bubblewrap receives individual read-only FD
//! mounts; it never receives a source tree, CAS, private root, or workspace.

use crate::digest::Sha256Digest;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const BWRAP: &str = "/usr/bin/bwrap";
const PRLIMIT: &str = "/usr/bin/prlimit";
const BACKEND_CONTRACT: &str = "linux_bubblewrap_sealed_memfd/v2";
const REQUEST_DIGEST_DOMAIN: &[u8] = b"tidex.isolated_execution.request.v2\0";
const VIRTUAL_PROGRAM: &str = "/tidex/program";
const VIRTUAL_INPUT: &str = "/tidex/input";
const MAX_PROGRAM_BYTES: usize = 64 * 1024 * 1024;
const MAX_INPUT_BYTES: usize = 256 * 1024 * 1024;
const MAX_STAGING_BYTES: u64 = 320 * 1024 * 1024;
const MAX_OUTPUT_BYTES: usize = 64 * 1024 * 1024;
const MAX_ARGUMENT_COUNT: usize = 1_024;
const MAX_ARGUMENT_BYTES: usize = 1024 * 1024;
const MAX_ADDRESS_SPACE_BYTES: u64 = 1 << 40;
const MAX_CPU_SECONDS: u64 = 86_400;
const MAX_WALL_MILLIS: u64 = 86_400_000;
const MAX_PROCESS_COUNT: u64 = 4_096;
const MAX_TEMPORARY_STORAGE_BYTES: u64 = 1024 * 1024 * 1024;
const STDERR_DETAIL_BYTES: usize = 4_096;
const MONITOR_INTERVAL: Duration = Duration::from_millis(5);

/// Exact bytes accompanied by the identity asserted by their authority.
///
/// Construction authenticates immediately. Execution authenticates again
/// before staging so an accepted value cannot silently change its meaning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticatedBytes {
    bytes: Vec<u8>,
    digest: Sha256Digest,
}

impl AuthenticatedBytes {
    pub fn authenticate(bytes: Vec<u8>, expected: Sha256Digest) -> IsolationResult<Self> {
        let observed = Sha256Digest::digest_bytes(&bytes);
        if observed != expected {
            return Err(IsolationError::new(
                IsolationErrorKind::AuthenticationFailed,
                "authenticated_bytes_digest_mismatch",
            ));
        }
        Ok(Self {
            bytes,
            digest: observed,
        })
    }

    pub fn from_trusted_bytes(bytes: Vec<u8>) -> Self {
        let digest = Sha256Digest::digest_bytes(&bytes);
        Self { bytes, digest }
    }

    pub fn digest(&self) -> &Sha256Digest {
        &self.digest
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn verify(&self, mismatch: &'static str) -> IsolationResult<()> {
        if Sha256Digest::digest_bytes(&self.bytes) != self.digest {
            return Err(IsolationError::new(
                IsolationErrorKind::AuthenticationFailed,
                mismatch,
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsolationLimits {
    pub address_space_bytes: u64,
    pub cpu_seconds: u64,
    pub wall_millis: u64,
    pub process_count: u64,
    pub output_bytes: usize,
    /// Aggregate kernel-enforced size of the sandbox's only writable tmpfs.
    pub temporary_storage_bytes: u64,
    /// Per-request admission ceiling for program plus input memfd bytes.
    /// This does not claim admission control across concurrent requests.
    pub staging_bytes: u64,
}

impl IsolationLimits {
    fn validate(self) -> IsolationResult<()> {
        if !(16 * 1024 * 1024..=MAX_ADDRESS_SPACE_BYTES).contains(&self.address_space_bytes)
            || !(1..=MAX_CPU_SECONDS).contains(&self.cpu_seconds)
            || !(1..=MAX_WALL_MILLIS).contains(&self.wall_millis)
            || !(1..=MAX_PROCESS_COUNT).contains(&self.process_count)
            || !(1..=MAX_OUTPUT_BYTES).contains(&self.output_bytes)
            || !(4_096..=MAX_TEMPORARY_STORAGE_BYTES).contains(&self.temporary_storage_bytes)
            || self.temporary_storage_bytes > self.address_space_bytes
            || !(1..=MAX_STAGING_BYTES).contains(&self.staging_bytes)
            || self.staging_bytes > self.address_space_bytes
        {
            return Err(IsolationError::new(
                IsolationErrorKind::InvalidContract,
                "isolation_limits_out_of_range",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsolationRequirements {
    pub require_seccomp_filter: bool,
    pub require_cgroup_limits: bool,
    pub require_global_staging_admission: bool,
}

/// A universal process request. Arguments are passed directly to `execve`;
/// neither a shell nor environment interpolation is involved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsolatedExecutionRequest {
    pub program: AuthenticatedBytes,
    pub input: AuthenticatedBytes,
    pub arguments: Vec<String>,
    pub limits: IsolationLimits,
    pub requirements: IsolationRequirements,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionTermination {
    ExitedSuccessfully,
    /// Bubblewrap produced a non-zero exit before this module had independent
    /// evidence that the payload reached `execve`. It may be setup or payload.
    NonzeroBeforePayloadConfirmation,
    Signaled,
    TimedOut,
    OutputLimitExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptionalIsolationControl {
    Enforced,
    NotProvided,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PayloadExecutionEvidence {
    /// A successful bwrap exit proves setup completed, the payload was exec'd,
    /// and the payload itself returned success.
    EstablishedBySuccessfulExit,
    NotEstablished,
}

/// Deterministic execution observation. This is not a signature or promotion
/// authority; a higher layer must place it in authenticated evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsolatedExecutionReport {
    pub schema_version: String,
    pub backend: String,
    /// Canonical commitment to program, input, arguments, limits,
    /// requirements, and the exact backend contract.
    pub request_digest: Sha256Digest,
    pub program_digest: Sha256Digest,
    pub input_digest: Sha256Digest,
    pub termination: ExecutionTermination,
    pub exit_code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub payload_execution_evidence: PayloadExecutionEvidence,
    pub seccomp_filter: OptionalIsolationControl,
    pub cgroup_limits: OptionalIsolationControl,
    pub global_staging_admission: OptionalIsolationControl,
}

impl IsolatedExecutionReport {
    pub fn succeeded(&self) -> bool {
        self.termination == ExecutionTermination::ExitedSuccessfully
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IsolationErrorKind {
    BackendUnavailable,
    InvalidContract,
    AuthenticationFailed,
    StagingFailed,
    LaunchFailed,
    MonitorFailed,
    RequiredControlUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsolationError {
    pub kind: IsolationErrorKind,
    pub detail: String,
}

impl IsolationError {
    fn new(kind: IsolationErrorKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for IsolationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}:{}", self.kind, self.detail)
    }
}

impl std::error::Error for IsolationError {}

pub type IsolationResult<T> = Result<T, IsolationError>;

/// Executes one authenticated program in a fresh Linux namespace sandbox.
///
/// Absence or inability to launch bubblewrap is an error, never a request to
/// fall back to direct host execution.
pub fn run_isolated(
    request: &IsolatedExecutionRequest,
) -> IsolationResult<IsolatedExecutionReport> {
    validate_request(request)?;
    require_backend(Path::new(BWRAP))?;
    require_runtime_helper(Path::new(PRLIMIT))?;

    let program = create_sealed_payload(
        "program",
        &request.program,
        0o500,
        "staged_program_digest_mismatch",
    )?;
    let input = create_sealed_payload(
        "input",
        &request.input,
        0o400,
        "staged_input_digest_mismatch",
    )?;
    let request_digest = canonical_request_digest(request)?;
    let arguments = build_bwrap_arguments(request, program.as_raw_fd(), input.as_raw_fd())?;

    let child = Command::new(BWRAP)
        .args(&arguments)
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            IsolationError::new(
                IsolationErrorKind::LaunchFailed,
                format!("bubblewrap_spawn_failed:{error}"),
            )
        })?;
    let mut child = ChildGuard::new(child);
    monitor_child(&mut child, request, request_digest)
}

/// Owns a launched process until an observed wait or forced kill+wait. This is
/// the last line of defence for reader-spawn errors and unwinding after spawn.
struct ChildGuard {
    child: Child,
    reaped: bool,
}

impl ChildGuard {
    fn new(child: Child) -> Self {
        Self {
            child,
            reaped: false,
        }
    }

    fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        let status = self.child.try_wait()?;
        if status.is_some() {
            self.reaped = true;
        }
        Ok(status)
    }

    fn kill_and_reap(&mut self) -> IsolationResult<()> {
        if self.reaped {
            return Ok(());
        }
        match self.child.kill() {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => {}
            Err(error) => {
                return Err(IsolationError::new(
                    IsolationErrorKind::MonitorFailed,
                    format!("sandbox_kill_failed:{error}"),
                ));
            }
        }
        self.child.wait().map_err(|error| {
            IsolationError::new(
                IsolationErrorKind::MonitorFailed,
                format!("sandbox_reap_failed:{error}"),
            )
        })?;
        self.reaped = true;
        Ok(())
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.child.kill();
            let _ = self.child.wait();
            self.reaped = true;
        }
    }
}

#[derive(Serialize)]
struct RequestCommitment<'a> {
    schema_version: &'static str,
    backend_contract: &'static str,
    program_digest: &'a Sha256Digest,
    input_digest: &'a Sha256Digest,
    arguments: &'a [String],
    limits: IsolationLimits,
    requirements: IsolationRequirements,
}

fn canonical_request_digest(request: &IsolatedExecutionRequest) -> IsolationResult<Sha256Digest> {
    let commitment = RequestCommitment {
        schema_version: "isolated_execution_request/v2",
        backend_contract: BACKEND_CONTRACT,
        program_digest: request.program.digest(),
        input_digest: request.input.digest(),
        arguments: &request.arguments,
        limits: request.limits,
        requirements: request.requirements,
    };
    let bytes = serde_json::to_vec(&commitment).map_err(|error| {
        IsolationError::new(
            IsolationErrorKind::InvalidContract,
            format!("request_commitment_serialization_failed:{error}"),
        )
    })?;
    Ok(Sha256Digest::digest_domain(REQUEST_DIGEST_DOMAIN, &bytes))
}

fn validate_request(request: &IsolatedExecutionRequest) -> IsolationResult<()> {
    request.limits.validate()?;
    if request.requirements.require_seccomp_filter {
        return Err(IsolationError::new(
            IsolationErrorKind::RequiredControlUnavailable,
            "seccomp_filter_not_configured",
        ));
    }
    if request.requirements.require_cgroup_limits {
        return Err(IsolationError::new(
            IsolationErrorKind::RequiredControlUnavailable,
            "cgroup_limits_not_configured",
        ));
    }
    if request.requirements.require_global_staging_admission {
        return Err(IsolationError::new(
            IsolationErrorKind::RequiredControlUnavailable,
            "global_staging_admission_not_configured",
        ));
    }
    request
        .program
        .verify("execution_program_digest_mismatch")?;
    request.input.verify("execution_input_digest_mismatch")?;
    if request.program.is_empty() || request.program.len() > MAX_PROGRAM_BYTES {
        return Err(IsolationError::new(
            IsolationErrorKind::InvalidContract,
            "execution_program_size_invalid",
        ));
    }
    if request.input.len() > MAX_INPUT_BYTES {
        return Err(IsolationError::new(
            IsolationErrorKind::InvalidContract,
            "execution_input_size_invalid",
        ));
    }
    let staged_bytes = u64::try_from(request.program.len())
        .ok()
        .and_then(|program| {
            u64::try_from(request.input.len())
                .ok()
                .and_then(|input| program.checked_add(input))
        })
        .ok_or_else(|| {
            IsolationError::new(
                IsolationErrorKind::InvalidContract,
                "execution_staging_size_overflow",
            )
        })?;
    if staged_bytes > request.limits.staging_bytes {
        return Err(IsolationError::new(
            IsolationErrorKind::InvalidContract,
            "execution_staging_budget_exceeded",
        ));
    }
    if request.arguments.len() > MAX_ARGUMENT_COUNT {
        return Err(IsolationError::new(
            IsolationErrorKind::InvalidContract,
            "execution_argument_count_exceeded",
        ));
    }
    let mut total_argument_bytes = 0usize;
    for argument in &request.arguments {
        if argument.as_bytes().contains(&0) {
            return Err(IsolationError::new(
                IsolationErrorKind::InvalidContract,
                "execution_argument_contains_nul",
            ));
        }
        total_argument_bytes = total_argument_bytes
            .checked_add(argument.len())
            .ok_or_else(|| {
                IsolationError::new(
                    IsolationErrorKind::InvalidContract,
                    "execution_argument_size_overflow",
                )
            })?;
    }
    if total_argument_bytes > MAX_ARGUMENT_BYTES {
        return Err(IsolationError::new(
            IsolationErrorKind::InvalidContract,
            "execution_argument_bytes_exceeded",
        ));
    }
    Ok(())
}

fn require_backend(path: &Path) -> IsolationResult<()> {
    let metadata = fs::symlink_metadata(path).map_err(|_| {
        IsolationError::new(
            IsolationErrorKind::BackendUnavailable,
            "bubblewrap_backend_unavailable",
        )
    })?;
    if !metadata.file_type().is_file() || metadata.permissions().mode() & 0o111 == 0 {
        return Err(IsolationError::new(
            IsolationErrorKind::BackendUnavailable,
            "bubblewrap_backend_not_executable_regular_file",
        ));
    }
    Ok(())
}

fn require_runtime_helper(path: &Path) -> IsolationResult<()> {
    let metadata = fs::symlink_metadata(path).map_err(|_| {
        IsolationError::new(
            IsolationErrorKind::BackendUnavailable,
            "resource_limit_helper_unavailable",
        )
    })?;
    if !metadata.file_type().is_file() || metadata.permissions().mode() & 0o111 == 0 {
        return Err(IsolationError::new(
            IsolationErrorKind::BackendUnavailable,
            "resource_limit_helper_not_executable_regular_file",
        ));
    }
    Ok(())
}

fn build_bwrap_arguments(
    request: &IsolatedExecutionRequest,
    program_fd: std::os::unix::io::RawFd,
    input_fd: std::os::unix::io::RawFd,
) -> IsolationResult<Vec<String>> {
    if program_fd < 0 || input_fd < 0 || program_fd == input_fd {
        return Err(IsolationError::new(
            IsolationErrorKind::StagingFailed,
            "isolation_payload_descriptors_invalid",
        ));
    }
    let mut arguments = vec![
        "--die-with-parent".into(),
        "--new-session".into(),
        "--unshare-user".into(),
        "--unshare-net".into(),
        "--unshare-pid".into(),
        "--unshare-ipc".into(),
        "--unshare-uts".into(),
        "--clearenv".into(),
        "--cap-drop".into(),
        "ALL".into(),
        "--ro-bind".into(),
        "/usr".into(),
        "/usr".into(),
    ];
    for system_directory in ["/lib", "/lib64"] {
        if Path::new(system_directory).is_dir() {
            arguments.extend([
                "--ro-bind".into(),
                system_directory.into(),
                system_directory.into(),
            ]);
        }
    }
    arguments.extend([
        "--dir".into(),
        "/tidex".into(),
        "--ro-bind-fd".into(),
        program_fd.to_string(),
        VIRTUAL_PROGRAM.into(),
        "--ro-bind-fd".into(),
        input_fd.to_string(),
        VIRTUAL_INPUT.into(),
        "--proc".into(),
        "/proc".into(),
        "--dev".into(),
        "/dev".into(),
        "--remount-ro".into(),
        "/proc".into(),
        "--remount-ro".into(),
        "/dev".into(),
        "--size".into(),
        request.limits.temporary_storage_bytes.to_string(),
        "--tmpfs".into(),
        "/tmp".into(),
        "--remount-ro".into(),
        "/".into(),
        "--setenv".into(),
        "HOME".into(),
        "/tmp".into(),
        "--setenv".into(),
        "TMPDIR".into(),
        "/tmp".into(),
        "--setenv".into(),
        "LANG".into(),
        "C.UTF-8".into(),
        "--setenv".into(),
        "TIDEX_INPUT_PATH".into(),
        VIRTUAL_INPUT.into(),
        "--chdir".into(),
        "/tidex".into(),
        "--".into(),
        PRLIMIT.into(),
        format!("--as={}", request.limits.address_space_bytes),
        format!("--cpu={}", request.limits.cpu_seconds),
        format!("--nproc={}", request.limits.process_count),
        "--nofile=32".into(),
        "--core=0".into(),
        format!("--fsize={}", request.limits.temporary_storage_bytes),
        "--".into(),
        VIRTUAL_PROGRAM.into(),
    ]);
    arguments.extend(request.arguments.iter().cloned());
    Ok(arguments)
}

fn staging_error(error: impl std::fmt::Display) -> IsolationError {
    IsolationError::new(
        IsolationErrorKind::StagingFailed,
        format!("isolation_staging_io:{error}"),
    )
}

fn create_sealed_payload(
    name: &str,
    payload: &AuthenticatedBytes,
    final_mode: u32,
    mismatch: &'static str,
) -> IsolationResult<File> {
    let owned = rustix::fs::memfd_create(name, rustix::fs::MemfdFlags::ALLOW_SEALING)
        .map_err(staging_error)?;
    let mut file = File::from(owned);
    file.write_all(&payload.bytes).map_err(staging_error)?;
    file.sync_all().map_err(staging_error)?;
    file.set_permissions(fs::Permissions::from_mode(final_mode))
        .map_err(staging_error)?;
    let required_seals = rustix::fs::SealFlags::SHRINK
        | rustix::fs::SealFlags::GROW
        | rustix::fs::SealFlags::WRITE
        | rustix::fs::SealFlags::SEAL;
    rustix::fs::fcntl_add_seals(&file, required_seals).map_err(staging_error)?;
    let observed_seals = rustix::fs::fcntl_get_seals(&file).map_err(staging_error)?;
    if !observed_seals.contains(required_seals) {
        return Err(IsolationError::new(
            IsolationErrorKind::StagingFailed,
            "memfd_required_seals_not_enforced",
        ));
    }
    let observed = digest_file_bounded(&mut file, payload.len())?;
    if observed != payload.digest {
        return Err(IsolationError::new(
            IsolationErrorKind::AuthenticationFailed,
            mismatch,
        ));
    }
    // Bubblewrap's `--ro-bind-fd` consumes these two explicitly named FDs. It
    // closes them after constructing mounts and before executing the payload.
    rustix::io::fcntl_setfd(&file, rustix::io::FdFlags::empty()).map_err(staging_error)?;
    Ok(file)
}

fn digest_file_bounded(file: &mut File, expected_len: usize) -> IsolationResult<Sha256Digest> {
    let metadata = file.metadata().map_err(staging_error)?;
    if !metadata.file_type().is_file() || usize::try_from(metadata.len()).ok() != Some(expected_len)
    {
        return Err(IsolationError::new(
            IsolationErrorKind::AuthenticationFailed,
            "staged_payload_metadata_mismatch",
        ));
    }
    file.seek(SeekFrom::Start(0)).map_err(staging_error)?;
    let mut hasher = Sha256::new();
    let mut observed_len = 0usize;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(staging_error)?;
        if count == 0 {
            break;
        }
        observed_len = observed_len.checked_add(count).ok_or_else(|| {
            IsolationError::new(
                IsolationErrorKind::AuthenticationFailed,
                "staged_payload_length_overflow",
            )
        })?;
        if observed_len > expected_len {
            return Err(IsolationError::new(
                IsolationErrorKind::AuthenticationFailed,
                "staged_payload_length_mismatch",
            ));
        }
        hasher.update(&buffer[..count]);
    }
    if observed_len != expected_len {
        return Err(IsolationError::new(
            IsolationErrorKind::AuthenticationFailed,
            "staged_payload_length_mismatch",
        ));
    }
    file.seek(SeekFrom::Start(0)).map_err(staging_error)?;
    Sha256Digest::parse(format!("{:x}", hasher.finalize())).map_err(|error| {
        IsolationError::new(
            IsolationErrorKind::AuthenticationFailed,
            format!("staged_payload_digest_invalid:{error}"),
        )
    })
}

fn monitor_child(
    child: &mut ChildGuard,
    request: &IsolatedExecutionRequest,
    request_digest: Sha256Digest,
) -> IsolationResult<IsolatedExecutionReport> {
    let stdout = child.child.stdout.take().ok_or_else(|| {
        IsolationError::new(IsolationErrorKind::MonitorFailed, "sandbox_stdout_missing")
    })?;
    let stderr = child.child.stderr.take().ok_or_else(|| {
        IsolationError::new(IsolationErrorKind::MonitorFailed, "sandbox_stderr_missing")
    })?;
    let overflow = Arc::new(AtomicBool::new(false));
    let retained_bytes = Arc::new(AtomicUsize::new(0));
    let stdout_reader = spawn_bounded_reader(
        "tidex-isolated-stdout",
        stdout,
        request.limits.output_bytes,
        retained_bytes.clone(),
        overflow.clone(),
    )?;
    let stderr_reader = spawn_bounded_reader(
        "tidex-isolated-stderr",
        stderr,
        request.limits.output_bytes,
        retained_bytes,
        overflow.clone(),
    )?;
    let started = Instant::now();
    let wall_limit = Duration::from_millis(request.limits.wall_millis);

    let (status, forced_termination) = loop {
        if overflow.load(Ordering::Acquire) {
            child.kill_and_reap()?;
            break (None, Some(ExecutionTermination::OutputLimitExceeded));
        }
        if started.elapsed() >= wall_limit {
            child.kill_and_reap()?;
            break (None, Some(ExecutionTermination::TimedOut));
        }
        match child.try_wait() {
            Ok(Some(status)) => break (Some(status), None),
            Ok(None) => thread::sleep(MONITOR_INTERVAL),
            Err(error) => {
                let _ = child.kill_and_reap();
                return Err(IsolationError::new(
                    IsolationErrorKind::MonitorFailed,
                    format!("sandbox_wait_failed:{error}"),
                ));
            }
        }
    };

    let stdout = join_reader(stdout_reader)?;
    let stderr = join_reader(stderr_reader)?;
    let overflowed = overflow.load(Ordering::Acquire);
    let exit_code = status.as_ref().and_then(ExitStatus::code);
    let termination = match (forced_termination, status, overflowed) {
        (Some(termination), _, _) => termination,
        (None, _, true) => ExecutionTermination::OutputLimitExceeded,
        (None, Some(status), false) => classify_status(status),
        (None, None, false) => {
            return Err(IsolationError::new(
                IsolationErrorKind::MonitorFailed,
                "sandbox_monitor_ended_without_status",
            ));
        }
    };
    Ok(IsolatedExecutionReport {
        schema_version: "isolated_execution_report/v2".into(),
        backend: BACKEND_CONTRACT.into(),
        request_digest,
        program_digest: request.program.digest.clone(),
        input_digest: request.input.digest.clone(),
        termination,
        exit_code,
        stdout: stdout.bytes,
        stderr: stderr.bytes,
        stdout_truncated: stdout.truncated,
        stderr_truncated: stderr.truncated,
        payload_execution_evidence: if termination == ExecutionTermination::ExitedSuccessfully {
            PayloadExecutionEvidence::EstablishedBySuccessfulExit
        } else {
            PayloadExecutionEvidence::NotEstablished
        },
        seccomp_filter: OptionalIsolationControl::NotProvided,
        cgroup_limits: OptionalIsolationControl::NotProvided,
        global_staging_admission: OptionalIsolationControl::NotProvided,
    })
}

fn classify_status(status: ExitStatus) -> ExecutionTermination {
    if status.success() {
        ExecutionTermination::ExitedSuccessfully
    } else if status.code().is_some() {
        ExecutionTermination::NonzeroBeforePayloadConfirmation
    } else {
        ExecutionTermination::Signaled
    }
}

struct BoundedRead {
    bytes: Vec<u8>,
    truncated: bool,
}

fn spawn_bounded_reader<R: Read + Send + 'static>(
    thread_name: &'static str,
    mut reader: R,
    limit: usize,
    retained_bytes: Arc<AtomicUsize>,
    overflow: Arc<AtomicBool>,
) -> IsolationResult<thread::JoinHandle<std::io::Result<BoundedRead>>> {
    thread::Builder::new()
        .name(thread_name.into())
        .spawn(move || {
            let mut bytes = Vec::with_capacity(limit.min(64 * 1024));
            let mut buffer = [0u8; 8192];
            let mut truncated = false;
            loop {
                let count = reader.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                let retained = claim_output_budget(&retained_bytes, limit, count);
                bytes.extend_from_slice(&buffer[..retained]);
                if retained != count {
                    truncated = true;
                    overflow.store(true, Ordering::Release);
                }
            }
            Ok(BoundedRead { bytes, truncated })
        })
        .map_err(|error| reader_spawn_error(thread_name, error))
}

fn reader_spawn_error(thread_name: &str, error: std::io::Error) -> IsolationError {
    IsolationError::new(
        IsolationErrorKind::MonitorFailed,
        format!("sandbox_output_reader_spawn_failed:{thread_name}:{error}"),
    )
}

fn claim_output_budget(retained_bytes: &AtomicUsize, limit: usize, requested: usize) -> usize {
    let mut current = retained_bytes.load(Ordering::Acquire);
    loop {
        let retained = limit.saturating_sub(current).min(requested);
        if retained == 0 {
            return 0;
        }
        match retained_bytes.compare_exchange_weak(
            current,
            current + retained,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return retained,
            Err(observed) => current = observed,
        }
    }
}

fn join_reader(
    handle: thread::JoinHandle<std::io::Result<BoundedRead>>,
) -> IsolationResult<BoundedRead> {
    handle
        .join()
        .map_err(|_| {
            IsolationError::new(
                IsolationErrorKind::MonitorFailed,
                "sandbox_output_reader_panicked",
            )
        })?
        .map_err(|error| {
            IsolationError::new(
                IsolationErrorKind::MonitorFailed,
                format!("sandbox_output_read_failed:{error}"),
            )
        })
}

/// Bounded diagnostic text for callers that need to record a failed launch.
pub fn bounded_stderr_detail(report: &IsolatedExecutionReport) -> String {
    String::from_utf8_lossy(&report.stderr)
        .chars()
        .take(STDERR_DETAIL_BYTES)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> IsolationLimits {
        IsolationLimits {
            address_space_bytes: 256 * 1024 * 1024,
            cpu_seconds: 2,
            wall_millis: 2_000,
            process_count: 4,
            output_bytes: 4_096,
            temporary_storage_bytes: 8 * 1024 * 1024,
            staging_bytes: 8 * 1024 * 1024,
        }
    }

    fn request() -> IsolatedExecutionRequest {
        IsolatedExecutionRequest {
            program: AuthenticatedBytes::from_trusted_bytes(b"program".to_vec()),
            input: AuthenticatedBytes::from_trusted_bytes(b"input".to_vec()),
            arguments: vec!["--mode".into(), "exact".into()],
            limits: limits(),
            requirements: IsolationRequirements {
                require_seccomp_filter: false,
                require_cgroup_limits: false,
                require_global_staging_admission: false,
            },
        }
    }

    #[test]
    fn authentication_rejects_a_false_identity() {
        let error = AuthenticatedBytes::authenticate(
            b"actual".to_vec(),
            Sha256Digest::digest_bytes(b"different"),
        )
        .unwrap_err();
        assert_eq!(error.kind, IsolationErrorKind::AuthenticationFailed);
    }

    #[test]
    fn invalid_and_unbounded_contracts_are_rejected() {
        let mut invalid = request();
        invalid.limits.wall_millis = 0;
        assert_eq!(
            validate_request(&invalid).unwrap_err().kind,
            IsolationErrorKind::InvalidContract
        );

        let mut nul = request();
        nul.arguments = vec!["bad\0argument".into()];
        assert_eq!(
            validate_request(&nul).unwrap_err().kind,
            IsolationErrorKind::InvalidContract
        );

        let mut staging = request();
        staging.limits.staging_bytes = 1;
        assert_eq!(
            validate_request(&staging).unwrap_err().kind,
            IsolationErrorKind::InvalidContract
        );
    }

    #[test]
    fn backend_absence_fails_closed() {
        let error = require_backend(Path::new("/definitely/not/a/tidex/backend")).unwrap_err();
        assert_eq!(error.kind, IsolationErrorKind::BackendUnavailable);
    }

    #[test]
    fn argument_manifest_has_only_individual_payload_mounts() {
        let request = request();
        let program =
            create_sealed_payload("program", &request.program, 0o500, "program_mismatch").unwrap();
        let input =
            create_sealed_payload("input", &request.input, 0o400, "input_mismatch").unwrap();
        let arguments =
            build_bwrap_arguments(&request, program.as_raw_fd(), input.as_raw_fd()).unwrap();

        for required in [
            "--unshare-user",
            "--unshare-net",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--clearenv",
            "--cap-drop",
            "--ro-bind-fd",
            "--size",
            "--remount-ro",
            VIRTUAL_PROGRAM,
            VIRTUAL_INPUT,
        ] {
            assert!(arguments.iter().any(|argument| argument == required));
        }
        assert!(!arguments.iter().any(|argument| {
            argument.contains("/home/")
                || argument.contains("/workspace")
                || argument.contains("/source")
                || argument.contains("/cas")
        }));
        assert_eq!(arguments.iter().filter(|value| *value == "--").count(), 2);
        assert_eq!(arguments.last().map(String::as_str), Some("exact"));
    }

    #[test]
    fn memfd_payload_is_immutable_after_authentication() {
        let request = request();
        let mut program =
            create_sealed_payload("program", &request.program, 0o500, "program_mismatch").unwrap();
        let seals = rustix::fs::fcntl_get_seals(&program).unwrap();
        assert!(seals.contains(rustix::fs::SealFlags::WRITE));
        assert!(seals.contains(rustix::fs::SealFlags::GROW));
        assert!(seals.contains(rustix::fs::SealFlags::SHRINK));
        assert!(seals.contains(rustix::fs::SealFlags::SEAL));
        assert!(program.write_all(b"tamper").is_err());
    }

    #[test]
    fn unsupported_mandatory_controls_fail_before_execution() {
        let mut seccomp = request();
        seccomp.requirements.require_seccomp_filter = true;
        assert_eq!(
            validate_request(&seccomp).unwrap_err().kind,
            IsolationErrorKind::RequiredControlUnavailable
        );
        let mut cgroup = request();
        cgroup.requirements.require_cgroup_limits = true;
        assert_eq!(
            validate_request(&cgroup).unwrap_err().kind,
            IsolationErrorKind::RequiredControlUnavailable
        );
        let mut global_admission = request();
        global_admission
            .requirements
            .require_global_staging_admission = true;
        assert_eq!(
            validate_request(&global_admission).unwrap_err().kind,
            IsolationErrorKind::RequiredControlUnavailable
        );
    }

    #[test]
    fn request_digest_commits_arguments_limits_and_backend_contract() {
        let original = request();
        let original_digest = canonical_request_digest(&original).unwrap();

        let mut relabeled_argument = original.clone();
        relabeled_argument.arguments[1] = "approximate".into();
        assert_ne!(
            canonical_request_digest(&relabeled_argument).unwrap(),
            original_digest
        );

        let mut relabeled_limit = original.clone();
        relabeled_limit.limits.cpu_seconds += 1;
        assert_ne!(
            canonical_request_digest(&relabeled_limit).unwrap(),
            original_digest
        );

        assert_eq!(
            canonical_request_digest(&original).unwrap(),
            original_digest
        );
    }

    #[test]
    fn reader_spawn_failure_is_typed() {
        let error = reader_spawn_error(
            "test-reader",
            std::io::Error::other("injected_thread_spawn_failure"),
        );
        assert_eq!(error.kind, IsolationErrorKind::MonitorFailed);
        assert!(error.detail.contains("injected_thread_spawn_failure"));
    }

    #[test]
    fn child_guard_kills_and_reaps_on_early_drop() {
        let child = Command::new("/usr/bin/sleep").arg("60").spawn().unwrap();
        let pid = child.id();
        {
            let _guard = ChildGuard::new(child);
        }
        assert!(!Path::new(&format!("/proc/{pid}")).exists());
    }

    #[test]
    fn bounded_reader_never_retains_more_than_the_limit() {
        let overflow = Arc::new(AtomicBool::new(false));
        let retained = Arc::new(AtomicUsize::new(0));
        let reader = spawn_bounded_reader(
            "test-bounded-reader",
            std::io::Cursor::new(vec![7u8; 10_000]),
            97,
            retained,
            overflow.clone(),
        )
        .unwrap();
        let output = join_reader(reader).unwrap();
        assert_eq!(output.bytes.len(), 97);
        assert!(output.truncated);
        assert!(overflow.load(Ordering::Acquire));
    }

    #[test]
    fn stdout_and_stderr_share_one_aggregate_budget() {
        let overflow = Arc::new(AtomicBool::new(false));
        let retained = Arc::new(AtomicUsize::new(0));
        let first = spawn_bounded_reader(
            "test-first-reader",
            std::io::Cursor::new(vec![1u8; 80]),
            100,
            retained.clone(),
            overflow.clone(),
        )
        .unwrap();
        let second = spawn_bounded_reader(
            "test-second-reader",
            std::io::Cursor::new(vec![2u8; 80]),
            100,
            retained,
            overflow.clone(),
        )
        .unwrap();
        let first = join_reader(first).unwrap();
        let second = join_reader(second).unwrap();
        assert_eq!(first.bytes.len() + second.bytes.len(), 100);
        assert!(first.truncated || second.truncated);
        assert!(overflow.load(Ordering::Acquire));
    }
}
