#![allow(clippy::needless_range_loop)]

use crate::artifact::{inspect_dvec, read_dvec_range, DeltaArtifactRef};
use crate::authority::{write_or_verify_immutable, PrivateFileReference};
use crate::contracts::{
    BlockSubspaceAxis, BlockSubspaceGeometry, BrainConfig, SkillField, SkillSubspaceGeometry,
};
use crate::digest::{ParameterLayoutDigest, Sha256Digest};
use crate::error::{BrainError, BrainResult};
use crate::identity::ObservationId;
use crate::linalg::{dot, norm, symmetric_eigen_jacobi, weighted_row_gram, Matrix};
use crate::validation::{
    choose_energy_rank, effective_rank_from_spectrum, source_support_indices, validate_reliability,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::security::{verify_internal_private_root, verify_private_root};

pub const MAX_PARAMETER_LAYOUT_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_PARAMETER_BLOCKS: usize = 1_048_576;
pub const MAX_PARAMETER_BLOCK_RANK: usize = 64;
pub const MAX_PARAMETER_BLOCK_NAME_BYTES: usize = 16 * 1024;
pub const MAX_LAYOUT_PARAMETER_COUNT: u64 = 1 << 50;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BlockShapeSpec {
    #[serde(rename = "module")]
    pub name: String,
    pub shape: Vec<usize>,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ParameterBlockSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ParameterBlockLayout {
    pub schema: String,
    pub blocks: Vec<ParameterBlockSpec>,
    pub total_parameter_count: u64,
}

impl ParameterBlockLayout {
    pub fn from_shapes(specs: &[BlockShapeSpec]) -> BrainResult<Self> {
        if specs.is_empty() || specs.len() > MAX_PARAMETER_BLOCKS {
            return Err(BrainError::Invalid(
                "block_layout_cardinality_invalid".into(),
            ));
        }
        let mut blocks = Vec::with_capacity(specs.len());
        let mut names = BTreeSet::new();
        let mut offset = 0u64;
        for (index, spec) in specs.iter().enumerate() {
            if spec.name.trim().is_empty()
                || spec.name != spec.name.trim()
                || !names.insert(spec.name.clone())
                || spec.name.len() > MAX_PARAMETER_BLOCK_NAME_BYTES
                || spec.count == 0
                || spec.shape.is_empty()
                || spec.shape.len() > MAX_PARAMETER_BLOCK_RANK
                || spec.shape.contains(&0)
            {
                return Err(BrainError::Invalid(format!("block_layout_invalid:{index}")));
            }
            let shape_count = spec.shape.iter().try_fold(1usize, |acc, value| {
                acc.checked_mul(*value)
                    .ok_or_else(|| BrainError::Invalid("block_layout_shape_overflow".into()))
            })?;
            if shape_count != spec.count {
                return Err(BrainError::Invalid(format!(
                    "block_layout_shape_count_mismatch:{}:{}:{}",
                    spec.name, shape_count, spec.count
                )));
            }
            blocks.push(ParameterBlockSpec {
                name: spec.name.clone(),
                shape: spec.shape.clone(),
                offset,
                count: spec.count,
            });
            offset = offset
                .checked_add(
                    u64::try_from(spec.count)
                        .map_err(|_| BrainError::Invalid("block_layout_count_overflow".into()))?,
                )
                .ok_or_else(|| BrainError::Invalid("block_layout_offset_overflow".into()))?;
            if offset > MAX_LAYOUT_PARAMETER_COUNT {
                return Err(BrainError::Invalid("block_layout_parameter_limit".into()));
            }
        }
        Ok(Self {
            schema: "cerebro.tidex.parameter_block_layout/v1".into(),
            blocks,
            total_parameter_count: offset,
        })
    }

    pub fn validate(&self) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.parameter_block_layout/v1"
            || self.blocks.is_empty()
            || self.blocks.len() > MAX_PARAMETER_BLOCKS
            || self.total_parameter_count > MAX_LAYOUT_PARAMETER_COUNT
        {
            return Err(BrainError::Invalid("block_layout_contract_invalid".into()));
        }
        let mut expected_offset = 0u64;
        let mut names = BTreeSet::new();
        for (index, block) in self.blocks.iter().enumerate() {
            let shape_count = block.shape.iter().try_fold(1usize, |acc, value| {
                acc.checked_mul(*value)
                    .ok_or_else(|| BrainError::Invalid("block_layout_shape_overflow".into()))
            })?;
            if block.name.trim().is_empty()
                || block.name != block.name.trim()
                || !names.insert(block.name.clone())
                || block.name.len() > MAX_PARAMETER_BLOCK_NAME_BYTES
                || block.shape.is_empty()
                || block.shape.len() > MAX_PARAMETER_BLOCK_RANK
                || block.shape.contains(&0)
                || block.count == 0
                || shape_count != block.count
                || block.offset != expected_offset
            {
                return Err(BrainError::Invalid(format!(
                    "block_layout_contract_invalid:{index}"
                )));
            }
            expected_offset = expected_offset
                .checked_add(
                    u64::try_from(block.count)
                        .map_err(|_| BrainError::Invalid("block_layout_count_overflow".into()))?,
                )
                .ok_or_else(|| BrainError::Invalid("block_layout_offset_overflow".into()))?;
        }
        if expected_offset != self.total_parameter_count {
            return Err(BrainError::Invalid(
                "block_layout_total_parameter_count_mismatch".into(),
            ));
        }
        Ok(())
    }
}

/// Version of the durable parameter-layout envelope. The layout itself keeps
/// its own schema because it is also used as an in-memory reconstruction
/// contract; this outer schema versions persistence and authentication.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ParameterLayoutArtifactSchema {
    #[serde(rename = "cerebro.tidex.parameter_layout_artifact/v1")]
    V1,
}

/// Durable semantic description of one exact flat parameter space.
///
/// `parameter_layout_sha256` identifies the validated layout semantics. It is
/// intentionally distinct from the [`PrivateFileReference`] returned by the
/// authority, whose SHA-256 identifies the exact serialized byte sequence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ParameterLayoutArtifact {
    pub schema: ParameterLayoutArtifactSchema,
    pub parameter_layout_sha256: ParameterLayoutDigest,
    pub total_parameter_count: u64,
    pub layout: ParameterBlockLayout,
}

impl ParameterLayoutArtifact {
    pub fn new(layout: ParameterBlockLayout) -> BrainResult<Self> {
        layout.validate()?;
        let parameter_layout_sha256 = parameter_layout_digest(&layout)?;
        Ok(Self {
            schema: ParameterLayoutArtifactSchema::V1,
            total_parameter_count: layout.total_parameter_count,
            parameter_layout_sha256,
            layout,
        })
    }

    pub fn validate(&self) -> BrainResult<()> {
        if self.schema != ParameterLayoutArtifactSchema::V1 {
            return Err(BrainError::Invalid(
                "parameter_layout_artifact_schema_invalid".into(),
            ));
        }
        self.layout.validate()?;
        if self.total_parameter_count != self.layout.total_parameter_count {
            return Err(BrainError::Integrity(
                "parameter_layout_artifact_total_mismatch".into(),
            ));
        }
        if self.parameter_layout_sha256 != parameter_layout_digest(&self.layout)? {
            return Err(BrainError::Integrity(
                "parameter_layout_artifact_semantic_digest_mismatch".into(),
            ));
        }
        Ok(())
    }
}

/// Hash the canonical semantic projection of a validated layout. Serde field
/// order is fixed by the Rust structs, while offsets and totals are required to
/// be contiguous and exact by `validate`; insignificant artifact whitespace is
/// therefore outside this identity.
pub fn parameter_layout_digest(
    layout: &ParameterBlockLayout,
) -> BrainResult<ParameterLayoutDigest> {
    layout.validate()?;
    Ok(ParameterLayoutDigest::from(Sha256Digest::digest_bytes(
        &serde_json::to_vec(layout)?,
    )))
}

#[derive(Debug, Clone, PartialEq)]
pub struct AuthenticatedParameterLayout {
    pub source: PrivateFileReference,
    pub artifact: ParameterLayoutArtifact,
}

/// Root-bound authority for immutable layout artifacts. Persisting always
/// emits canonical compact JSON; authenticating also accepts a different JSON
/// encoding but preserves its exact byte identity in `source`.
#[derive(Debug, Clone)]
pub struct ParameterLayoutAuthority {
    root: PathBuf,
}

impl ParameterLayoutAuthority {
    pub fn open(root: impl AsRef<Path>) -> BrainResult<Self> {
        Ok(Self {
            root: verify_private_root(root.as_ref())?,
        })
    }

    pub(crate) fn for_internal_root(root: &Path) -> BrainResult<Self> {
        Ok(Self {
            root: verify_internal_private_root(root)?,
        })
    }

    pub fn persist(&self, layout: ParameterBlockLayout) -> BrainResult<PrivateFileReference> {
        let artifact = ParameterLayoutArtifact::new(layout)?;
        let bytes = serde_json::to_vec(&artifact)?;
        if u64::try_from(bytes.len())
            .map_err(|_| BrainError::Invalid("parameter_layout_size_overflow".into()))?
            > MAX_PARAMETER_LAYOUT_BYTES
        {
            return Err(BrainError::Invalid("parameter_layout_too_large".into()));
        }
        let byte_sha256 = Sha256Digest::digest_bytes(&bytes);
        let path = self
            .root
            .join("state/parameter_layouts/by-sha")
            .join(format!("{byte_sha256}.json"));
        let stored = write_or_verify_immutable(&self.root, &path, &bytes)?;
        if stored != byte_sha256 {
            return Err(BrainError::Integrity(
                "parameter_layout_artifact_write_mismatch".into(),
            ));
        }
        let reference = PrivateFileReference::new(path, byte_sha256);
        self.authenticate(reference.clone())?;
        Ok(reference)
    }

    pub fn authenticate(
        &self,
        source: PrivateFileReference,
    ) -> BrainResult<AuthenticatedParameterLayout> {
        let bytes = source.read_verified_bounded(&self.root, MAX_PARAMETER_LAYOUT_BYTES)?;
        let artifact: ParameterLayoutArtifact = serde_json::from_slice(&bytes)?;
        artifact.validate()?;
        Ok(AuthenticatedParameterLayout { source, artifact })
    }

    /// Authenticate the one canonical, content-addressed layout envelope and
    /// bind it to the semantic identity and flat dimension asserted by a
    /// consuming authority receipt. Neither value is trusted from that
    /// receipt: both are checked against the reopened envelope.
    pub fn authenticate_canonical_binding(
        &self,
        source: PrivateFileReference,
        expected_semantic_sha256: &ParameterLayoutDigest,
        expected_total_parameter_count: u64,
    ) -> BrainResult<AuthenticatedParameterLayout> {
        let expected_path = self
            .root
            .join("state/parameter_layouts/by-sha")
            .join(format!("{}.json", source.sha256));
        if source.path != expected_path {
            return Err(BrainError::Integrity(
                "parameter_layout_artifact_canonical_path_mismatch".into(),
            ));
        }
        let authenticated = self.authenticate(source)?;
        if &authenticated.artifact.parameter_layout_sha256 != expected_semantic_sha256 {
            return Err(BrainError::Integrity(
                "parameter_layout_artifact_bound_semantic_digest_mismatch".into(),
            ));
        }
        if authenticated.artifact.total_parameter_count != expected_total_parameter_count {
            return Err(BrainError::Integrity(
                "parameter_layout_artifact_bound_total_mismatch".into(),
            ));
        }
        Ok(authenticated)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StructuredSource {
    pub observation_id: ObservationId,
    pub artifact: DeltaArtifactRef,
    pub reliability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StructuredGeometryReport {
    pub schema: String,
    pub source_count: usize,
    pub block_count: usize,
    pub total_parameter_count: u64,
    pub skills: Vec<SkillSubspaceGeometry>,
}

fn validate_sources(
    root: &Path,
    fields: &[SkillField],
    source_mixtures: &[Vec<f64>],
    sources: &[StructuredSource],
    layout: &ParameterBlockLayout,
) -> BrainResult<()> {
    layout.validate()?;
    if fields.is_empty()
        || fields.len() != source_mixtures.len()
        || source_mixtures.iter().any(|row| row.len() != sources.len())
    {
        return Err(BrainError::Invalid(
            "structured_geometry_source_count_mismatch".into(),
        ));
    }
    let mut observation_ids = BTreeSet::new();
    let mut artifact_digests = BTreeSet::new();
    for (index, source) in sources.iter().enumerate() {
        if !observation_ids.insert(source.observation_id.clone())
            || !artifact_digests.insert(source.artifact.sha256.clone())
            || !source.reliability.is_finite()
            || source.reliability <= 0.0
            || source.reliability > 1.0
        {
            return Err(BrainError::Invalid(format!(
                "structured_geometry_source_invalid:{index}"
            )));
        }
        let inspected = inspect_dvec(root, Path::new(&source.artifact.path))?;
        if inspected.sha256 != source.artifact.sha256
            || inspected.parameter_count != source.artifact.parameter_count
            || inspected.parameter_count != layout.total_parameter_count
        {
            return Err(BrainError::Integrity(format!(
                "structured_geometry_artifact_mismatch:{index}"
            )));
        }
    }
    Ok(())
}

fn reconstruct_block(
    root: &Path,
    block: &ParameterBlockSpec,
    sources: &[StructuredSource],
    support: &[usize],
    alignment_mixture: &[f64],
    cfg: &BrainConfig,
) -> BrainResult<BlockSubspaceGeometry> {
    if support.is_empty() {
        return Err(BrainError::Invalid(
            "structured_geometry_empty_skill_support".into(),
        ));
    }
    let mut rows = Vec::with_capacity(support.len());
    let mut weights = Vec::with_capacity(support.len());
    for &source_index in support {
        let mut values = read_dvec_range(
            root,
            Path::new(&sources[source_index].artifact.path),
            block.offset,
            block.count,
        )?;
        let sign = alignment_mixture[source_index].signum();
        if sign < 0.0 {
            for value in &mut values {
                *value = -*value;
            }
        }
        rows.push(values);
        weights.push(validate_reliability(
            sources[source_index].reliability,
            "structured_geometry",
        )?);
    }
    let data = Matrix::from_rows(&rows)?;
    let block_energy = (0..data.rows).try_fold(0.0, |total, row| {
        Ok::<f64, BrainError>(total + weights[row] * dot(data.row(row), data.row(row))?)
    })?;
    if block_energy <= 1e-24 {
        return Ok(BlockSubspaceGeometry {
            block_name: block.name.clone(),
            offset: block.offset,
            count: block.count,
            shape: block.shape.clone(),
            selected_rank: 0,
            effective_rank: 0.0,
            retained_energy: 0.0,
            block_energy: 0.0,
            normalized_block_energy: 0.0,
            reconstruction_rms: 0.0,
            axes: Vec::new(),
        });
    }
    let gram = weighted_row_gram(&data, &weights)?;
    let eigs = symmetric_eigen_jacobi(&gram, 1e-11, data.rows * data.rows * 100)?;
    let eigenvalues = eigs.iter().map(|(value, _)| *value).collect::<Vec<_>>();
    let rank = choose_energy_rank(
        &eigenvalues,
        cfg.target_explained_variance,
        cfg.max_rank.min(support.len()),
        0,
    )?;
    let total_energy = eigenvalues.iter().sum::<f64>().max(1e-18);
    let retained_energy = eigenvalues.iter().take(rank).sum::<f64>() / total_energy;

    let mut basis = Vec::<Vec<f64>>::with_capacity(rank);
    let mut axes = Vec::with_capacity(rank);
    for (lambda, eigenvector) in eigs.iter().take(rank) {
        let denom = lambda.sqrt().max(1e-15);
        let mut axis = vec![0.0; data.cols];
        let mut global_coefficients = vec![0.0; sources.len()];
        for local_row in 0..data.rows {
            let local_coefficient = weights[local_row].sqrt() * eigenvector[local_row] / denom;
            for parameter in 0..data.cols {
                axis[parameter] += local_coefficient * data.get(local_row, parameter);
            }
            let source_index = support[local_row];
            let sign = alignment_mixture[source_index].signum();
            global_coefficients[source_index] =
                local_coefficient * if sign < 0.0 { -1.0 } else { 1.0 };
        }
        let axis_norm = norm(&axis)?;
        if axis_norm <= 1e-15 {
            return Err(BrainError::Numerical(
                "structured_geometry_axis_degenerate".into(),
            ));
        }
        for value in &mut axis {
            *value /= axis_norm;
        }
        for coefficient in &mut global_coefficients {
            *coefficient /= axis_norm;
        }
        basis.push(axis);
        axes.push(BlockSubspaceAxis {
            singular_value: lambda.sqrt(),
            source_coefficients: global_coefficients,
        });
    }

    let mut squared_error = 0.0;
    for row in 0..data.rows {
        let mut reconstructed = vec![0.0; data.cols];
        for axis in &basis {
            let coefficient = dot(data.row(row), axis)?;
            for parameter in 0..data.cols {
                reconstructed[parameter] += coefficient * axis[parameter];
            }
        }
        for parameter in 0..data.cols {
            squared_error += (data.get(row, parameter) - reconstructed[parameter]).powi(2);
        }
    }
    let reconstruction_rms = (squared_error / (data.rows * data.cols).max(1) as f64).sqrt();
    Ok(BlockSubspaceGeometry {
        block_name: block.name.clone(),
        offset: block.offset,
        count: block.count,
        shape: block.shape.clone(),
        selected_rank: rank,
        effective_rank: effective_rank_from_spectrum(&eigenvalues)?,
        retained_energy,
        block_energy,
        normalized_block_energy: 0.0,
        reconstruction_rms,
        axes,
    })
}

pub fn reconstruct_structured_geometry(
    root: &Path,
    fields: &[SkillField],
    source_mixtures: &[Vec<f64>],
    sources: &[StructuredSource],
    layout: &ParameterBlockLayout,
    cfg: &BrainConfig,
) -> BrainResult<StructuredGeometryReport> {
    validate_sources(root, fields, source_mixtures, sources, layout)?;
    let mut skills = Vec::with_capacity(fields.len());
    for (skill_index, field) in fields.iter().enumerate() {
        let mixture = source_mixtures
            .get(skill_index)
            .ok_or_else(|| BrainError::Integrity("structured_geometry_mixture_missing".into()))?;
        let support = source_support_indices(mixture)?;
        if support.is_empty() {
            return Err(BrainError::Invalid(format!(
                "structured_geometry_skill_support_empty:{skill_index}"
            )));
        }
        let mut blocks = Vec::with_capacity(layout.blocks.len());
        for block in &layout.blocks {
            blocks.push(reconstruct_block(
                root, block, sources, &support, mixture, cfg,
            )?);
        }
        let total_block_energy = blocks.iter().map(|block| block.block_energy).sum::<f64>();
        if total_block_energy > 1e-24 {
            for block in &mut blocks {
                block.normalized_block_energy = block.block_energy / total_block_energy;
            }
        }
        let max_local_rank = blocks
            .iter()
            .map(|block| block.selected_rank)
            .max()
            .unwrap_or(0);
        let active_blocks = blocks
            .iter()
            .filter(|block| block.selected_rank > 0)
            .count();
        let mean_effective_rank = if active_blocks == 0 {
            0.0
        } else {
            blocks
                .iter()
                .filter(|block| block.selected_rank > 0)
                .map(|block| block.effective_rank)
                .sum::<f64>()
                / active_blocks as f64
        };
        skills.push(SkillSubspaceGeometry {
            skill_id: field.skill_id.clone(),
            source_support_indices: support,
            blocks,
            max_local_rank,
            mean_effective_rank,
        });
    }
    Ok(StructuredGeometryReport {
        schema: "cerebro.tidex.structured_geometry/v1".into(),
        source_count: sources.len(),
        block_count: layout.blocks.len(),
        total_parameter_count: layout.total_parameter_count,
        skills,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::create_dvec;
    use crate::contracts::SkillField;
    use crate::security::secure_dir;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn private_test_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tidex-layout-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        secure_dir(&root).unwrap();
        root
    }

    fn minimal_layout() -> ParameterBlockLayout {
        ParameterBlockLayout::from_shapes(&[
            BlockShapeSpec {
                name: "attention.q".into(),
                shape: vec![2, 3],
                count: 6,
            },
            BlockShapeSpec {
                name: "mlp.down".into(),
                shape: vec![2],
                count: 2,
            },
        ])
        .unwrap()
    }

    fn minimal_field() -> SkillField {
        SkillField {
            skill_id: crate::identity::SkillId::parse("s").unwrap(),
            reconstruction_id: "r".into(),
            lineage_id: "l".into(),
            generation_created: 1,
            direction: vec![1.0, 0.0, 0.0, 0.0],
            structured_geometry: None,
            dense_materialization: None,
            parameter_layout_sha256: None,
            representation_signature: Vec::new(),
            singular_value: 1.0,
            explained_variance: 1.0,
            persistence: 1.0,
            coherence: 1.0,
            uncertainty: 0.0,
            evidence_support_digests: Vec::new(),
            support: 3,
            functional_signature: vec![],
            parent_skill_ids: vec![],
        }
    }

    #[test]
    fn block_geometry_preserves_local_rank_and_source_coefficients() {
        let root = std::env::temp_dir().join(format!("tidex-block-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let a = create_dvec(&root, "a", &[1.0, 0.0, 1.0, 0.0]).unwrap();
        let b = create_dvec(&root, "b", &[0.0, 1.0, 0.0, 1.0]).unwrap();
        let c = create_dvec(&root, "c", &[1.0, 1.0, 1.0, 1.0]).unwrap();
        let sources = vec![
            StructuredSource {
                observation_id: ObservationId::parse("a").unwrap(),
                artifact: a,
                reliability: 1.0,
            },
            StructuredSource {
                observation_id: ObservationId::parse("b").unwrap(),
                artifact: b,
                reliability: 1.0,
            },
            StructuredSource {
                observation_id: ObservationId::parse("c").unwrap(),
                artifact: c,
                reliability: 1.0,
            },
        ];
        let layout = ParameterBlockLayout::from_shapes(&[
            BlockShapeSpec {
                name: "x".into(),
                shape: vec![2],
                count: 2,
            },
            BlockShapeSpec {
                name: "y".into(),
                shape: vec![2],
                count: 2,
            },
        ])
        .unwrap();
        let fields = vec![minimal_field()];
        let mixtures = vec![vec![1.0 / 3.0; 3]];
        let geometry = reconstruct_structured_geometry(
            &root,
            &fields,
            &mixtures,
            &sources,
            &layout,
            &BrainConfig {
                target_explained_variance: 0.99,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(geometry.skills.len(), 1);
        assert_eq!(geometry.skills[0].blocks.len(), 2);
        assert!(geometry.skills[0].max_local_rank >= 2);
        assert!(geometry.skills[0].blocks.iter().all(|block| block
            .axes
            .iter()
            .all(|axis| axis.source_coefficients.len() == 3)));

        let mut zero_reliability = sources.clone();
        zero_reliability[1].reliability = 0.0;
        assert!(reconstruct_structured_geometry(
            &root,
            &fields,
            &mixtures,
            &zero_reliability,
            &layout,
            &BrainConfig::default(),
        )
        .is_err());

        let mut invalid_layout = layout.clone();
        invalid_layout.blocks[1].offset = 0;
        assert!(reconstruct_structured_geometry(
            &root,
            &fields,
            &mixtures,
            &sources,
            &invalid_layout,
            &BrainConfig::default(),
        )
        .is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn exact_byte_identity_is_distinct_from_layout_semantics() {
        let root = private_test_root("semantic-vs-bytes");
        let authority = ParameterLayoutAuthority { root: root.clone() };
        let canonical = authority.persist(minimal_layout()).unwrap();
        let canonical_layout = authority.authenticate(canonical.clone()).unwrap();

        let pretty_bytes = serde_json::to_vec_pretty(&canonical_layout.artifact).unwrap();
        let pretty_sha256 = Sha256Digest::digest_bytes(&pretty_bytes);
        let pretty_path = root
            .join("state/parameter_layouts/test-encodings")
            .join(format!("{pretty_sha256}.json"));
        write_or_verify_immutable(&root, &pretty_path, &pretty_bytes).unwrap();
        let pretty = authority
            .authenticate(PrivateFileReference::new(pretty_path, pretty_sha256))
            .unwrap();

        assert_ne!(canonical.sha256, pretty.source.sha256);
        assert_eq!(
            canonical_layout.artifact.parameter_layout_sha256,
            pretty.artifact.parameter_layout_sha256
        );
        assert_eq!(canonical_layout.artifact.layout, pretty.artifact.layout);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn authentication_rejects_exact_byte_tampering() {
        let root = private_test_root("tamper");
        let authority = ParameterLayoutAuthority { root: root.clone() };
        let reference = authority.persist(minimal_layout()).unwrap();
        let mut bytes = fs::read(&reference.path).unwrap();
        bytes.push(b'\n');
        fs::write(&reference.path, bytes).unwrap();

        assert!(authority.authenticate(reference).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn authentication_rejects_semantic_tampering_with_a_matching_byte_hash() {
        let root = private_test_root("semantic-tamper");
        let authority = ParameterLayoutAuthority { root: root.clone() };
        let mut artifact = ParameterLayoutArtifact::new(minimal_layout()).unwrap();
        artifact.layout.blocks[0].name = "attention.k".into();
        let bytes = serde_json::to_vec(&artifact).unwrap();
        let byte_sha256 = Sha256Digest::digest_bytes(&bytes);
        let path = root
            .join("state/parameter_layouts/test-tampering")
            .join(format!("{byte_sha256}.json"));
        write_or_verify_immutable(&root, &path, &bytes).unwrap();

        assert!(authority
            .authenticate(PrivateFileReference::new(path, byte_sha256))
            .is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn authentication_rejects_a_reference_from_another_root() {
        let first_root = private_test_root("first-root");
        let second_root = private_test_root("second-root");
        let first = ParameterLayoutAuthority {
            root: first_root.clone(),
        };
        let second = ParameterLayoutAuthority {
            root: second_root.clone(),
        };
        let reference = first.persist(minimal_layout()).unwrap();
        let authenticated = first.authenticate(reference.clone()).unwrap();

        assert!(second
            .authenticate_canonical_binding(
                reference,
                &authenticated.artifact.parameter_layout_sha256,
                authenticated.artifact.total_parameter_count,
            )
            .is_err());
        let _ = fs::remove_dir_all(first_root);
        let _ = fs::remove_dir_all(second_root);
    }

    #[test]
    fn canonical_binding_rejects_semantic_count_and_path_mismatch() {
        let root = private_test_root("canonical-binding");
        let authority = ParameterLayoutAuthority { root: root.clone() };
        let reference = authority.persist(minimal_layout()).unwrap();
        let authenticated = authority.authenticate(reference.clone()).unwrap();
        let semantic = authenticated.artifact.parameter_layout_sha256.clone();
        let total = authenticated.artifact.total_parameter_count;

        let mut different_layout = minimal_layout();
        different_layout.blocks[0].name = "attention.k".into();
        let different_semantic = parameter_layout_digest(&different_layout).unwrap();
        assert!(authority
            .authenticate_canonical_binding(reference.clone(), &different_semantic, total)
            .is_err());
        assert!(authority
            .authenticate_canonical_binding(reference.clone(), &semantic, total + 1)
            .is_err());

        let alternate_path = root
            .join("state/parameter_layouts/aliases")
            .join(format!("{}.json", reference.sha256));
        let bytes = fs::read(&reference.path).unwrap();
        write_or_verify_immutable(&root, &alternate_path, &bytes).unwrap();
        assert!(authority
            .authenticate_canonical_binding(
                PrivateFileReference::new(alternate_path, reference.sha256),
                &semantic,
                total,
            )
            .is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn artifact_rejects_noncanonical_layout_total_and_unknown_fields() {
        let layout = minimal_layout();
        let mut wrong_total = ParameterLayoutArtifact::new(layout.clone()).unwrap();
        wrong_total.total_parameter_count += 1;
        assert!(wrong_total.validate().is_err());

        let mut noncanonical = layout;
        noncanonical.blocks[1].offset -= 1;
        assert!(ParameterLayoutArtifact::new(noncanonical).is_err());

        let valid = ParameterLayoutArtifact::new(minimal_layout()).unwrap();
        let mut value = serde_json::to_value(valid).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("unrecognized".into(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<ParameterLayoutArtifact>(value).is_err());
    }

    #[test]
    fn layout_complexity_limits_fail_closed_before_geometry_work() {
        let mut oversized_name = minimal_layout();
        oversized_name.blocks[0].name = "x".repeat(MAX_PARAMETER_BLOCK_NAME_BYTES + 1);
        assert!(oversized_name.validate().is_err());

        let mut excessive_rank = minimal_layout();
        excessive_rank.blocks[0].shape = vec![1; MAX_PARAMETER_BLOCK_RANK + 1];
        excessive_rank.blocks[0].count = 1;
        excessive_rank.blocks[1].offset = 1;
        excessive_rank.total_parameter_count = 3;
        assert!(excessive_rank.validate().is_err());

        let mut excessive_total = minimal_layout();
        excessive_total.total_parameter_count = MAX_LAYOUT_PARAMETER_COUNT + 1;
        assert!(excessive_total.validate().is_err());
    }
}
