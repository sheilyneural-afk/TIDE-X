use cerebro_tidex::artifact::DeltaArtifactRef;
use cerebro_tidex::block_tomography::{
    reconstruct_structured_geometry, BlockShapeSpec, ParameterBlockLayout, StructuredSource,
};
use cerebro_tidex::contracts::{BrainConfig, DeltaObservation};
use cerebro_tidex::engine::ReconstructionReport;
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct AdapterRow {
    observation_id: String,
    artifact: DeltaArtifactRef,
}

#[derive(Debug, Deserialize)]
struct TrainingManifest {
    adapters: Vec<AdapterRow>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 5 {
        return Err("usage: structured_geometry_bench <observations.json> <training_manifest.json> <module_specs.json> <tidex_report.json>".into());
    }
    let observations: Vec<DeltaObservation> =
        serde_json::from_slice(&fs::read(Path::new(&args[1]))?)?;
    let manifest: TrainingManifest = serde_json::from_slice(&fs::read(Path::new(&args[2]))?)?;
    let shapes: Vec<BlockShapeSpec> = serde_json::from_slice(&fs::read(Path::new(&args[3]))?)?;
    let report: ReconstructionReport = serde_json::from_slice(&fs::read(Path::new(&args[4]))?)?;
    let artifacts = manifest
        .adapters
        .into_iter()
        .map(|row| (row.observation_id, row.artifact))
        .collect::<BTreeMap<_, _>>();
    let sources = observations
        .iter()
        .map(|observation| {
            Ok(StructuredSource {
                observation_id: observation.observation_id.clone(),
                artifact: artifacts
                    .get(&observation.observation_id)
                    .cloned()
                    .ok_or_else(|| {
                        format!("artifact missing for {}", observation.observation_id)
                    })?,
                reliability: observation.reliability,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let layout = ParameterBlockLayout::from_shapes(&shapes)?;
    let geometry = reconstruct_structured_geometry(
        &report.fields,
        &report.skill_source_mixtures,
        &sources,
        &layout,
        &BrainConfig::default(),
    )?;

    let skill_summaries = geometry
        .skills
        .iter()
        .map(|skill| {
            let rank_histogram = skill.blocks.iter().fold(BTreeMap::<usize, usize>::new(), |mut map, block| {
                *map.entry(block.selected_rank).or_default() += 1;
                map
            });
            let top_blocks = {
                let mut rows = skill
                    .blocks
                    .iter()
                    .map(|block| (block.block_name.clone(), block.normalized_block_energy, block.selected_rank, block.effective_rank))
                    .collect::<Vec<_>>();
                rows.sort_by(|left, right| right.1.total_cmp(&left.1));
                rows.into_iter().take(5).map(|(name, energy, rank, effective_rank)| {
                    json!({"block":name,"energy_fraction":energy,"rank":rank,"effective_rank":effective_rank})
                }).collect::<Vec<_>>()
            };
            json!({
                "skill_id":skill.skill_id,
                "source_support_indices":skill.source_support_indices,
                "max_local_rank":skill.max_local_rank,
                "mean_effective_rank":skill.mean_effective_rank,
                "rank_histogram":rank_histogram,
                "top_blocks":top_blocks,
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.structured_geometry_benchmark/v1",
            "source_count":geometry.source_count,
            "block_count":geometry.block_count,
            "total_parameter_count":geometry.total_parameter_count,
            "skills":skill_summaries,
            "full_geometry":geometry,
        }))?
    );
    Ok(())
}
