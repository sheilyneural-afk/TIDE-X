use cerebro_tidex::artifact::{read_dvec_f32, DeltaArtifactRef};
use cerebro_tidex::authority::{
    ensure_private_parent, existing_regular_file_under_root, root_relative_path,
    write_or_verify_immutable,
};
use cerebro_tidex::protected_map::{
    build_protected_cortex_map, load_protected_cortex, persist_protected_map, SensitivityEvidence,
};
use cerebro_tidex::security::{verify_private_root, PRIVATE_ROOT};
use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct EvidenceRow {
    probe_id: String,
    artifact: DeltaArtifactRef,
    causal_damage_per_parameter_norm: f64,
    reliability: f64,
}
#[derive(Debug, Deserialize)]
struct Payload {
    schema: String,
    task_labels_used: bool,
    evidence: Vec<EvidenceRow>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = verify_private_root(Path::new(PRIVATE_ROOT))?;
    let args = std::env::args().collect::<Vec<_>>();
    let evidence_path = args
        .get(1)
        .ok_or("usage: protected_map_bench <evidence.json> <output.json>")?;
    let output_path = args
        .get(2)
        .ok_or("usage: protected_map_bench <evidence.json> <output.json>")?;
    let output_path = Path::new(output_path);
    // Validate lexical confinement before creating a single parent directory.
    // This benchmark may emit evidence, but it never writes outside CEREBRO's
    // authenticated private root.
    root_relative_path(&root, output_path)?;
    ensure_private_parent(&root, output_path)?;
    if fs::symlink_metadata(output_path).is_ok() {
        return Err("protected map output already exists".into());
    }
    let evidence_path = existing_regular_file_under_root(&root, Path::new(evidence_path))?;
    let payload: Payload = serde_json::from_slice(&fs::read(evidence_path)?)?;
    if payload.schema != "cerebro.tidex.protected_sensitivity_evidence/v1"
        || payload.task_labels_used
        || payload.evidence.len() < 2
    {
        return Err("protected sensitivity contract invalid".into());
    }
    let evidence = payload
        .evidence
        .iter()
        .map(|row| {
            Ok(SensitivityEvidence {
                probe_id: row.probe_id.clone(),
                sensitivity: read_dvec_f32(&row.artifact)?
                    .into_iter()
                    .map(f64::from)
                    .collect(),
                reliability: row.reliability,
                causal_damage: Some(row.causal_damage_per_parameter_norm),
            })
        })
        .collect::<cerebro_tidex::BrainResult<Vec<_>>>()?;
    let dense = build_protected_cortex_map(&evidence, 0.92, 1.0)?;
    let stored = persist_protected_map(&root, &dense)?;
    let loaded = load_protected_cortex(&root, &stored)?;
    if loaded != dense.cortex {
        return Err("protected map persisted roundtrip mismatch".into());
    }
    let output = json!({
        "schema":"cerebro.tidex.protected_map_benchmark/v2",
        "task_labels_used":false,
        "map":stored,
    });
    let mut bytes = serde_json::to_vec_pretty(&output)?;
    bytes.push(b'\n');
    let _ = write_or_verify_immutable(&root, output_path, &bytes)?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
