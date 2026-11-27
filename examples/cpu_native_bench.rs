//! Canonical current-recipe CPU benchmark with independent optimization flags.
use clap::{Parser, ValueEnum};
use quip_protocol::diversity::diversity_milli;
use quip_solver_conformance::families::{export_instances, reference_revision};
use quip_solver_core::ising_generators::family::BaseTopology;
use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_core::SampleParams;
#[cfg(feature = "zephyr-spine-polish")]
use quip_solver_core::SamplerResult;
use quip_solver_cpu::families::native::{instance_energy, prepared, tuned_reads, NativeConfig};
use quip_solver_cpu::families::otsuka::{anneal_population_observed, PopulationOptions, Schedule};
#[cfg(feature = "zephyr-spine-polish")]
use quip_solver_cpu::families::{native::polish_zephyr, otsuka::SpinePolishOptions};
#[cfg(feature = "zephyr-spine-polish")]
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::Deserialize;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Kernel,
    Pipeline,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Preset {
    Baseline,
    Cache,
    Copy,
    Union,
    Fusion,
    Core,
    Archive,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum Backend {
    #[default]
    Native,
    #[cfg(feature = "zephyr-spine-polish")]
    ZephyrSpine,
}

impl Backend {
    const fn name(self) -> &'static str {
        match self {
            Self::Native => "native",
            #[cfg(feature = "zephyr-spine-polish")]
            Self::ZephyrSpine => "zephyr-spine",
        }
    }
}

#[derive(Parser)]
struct Args {
    #[arg(
        long,
        default_value = "tools/cpu-native/inputs/current-topology.spec.json"
    )]
    topology: PathBuf,
    #[arg(long, default_value = "tools/cpu-native/inputs/current-family.params")]
    params: PathBuf,
    #[arg(long, value_enum, default_value = "kernel")]
    mode: Mode,
    #[arg(long, value_enum, default_value = "baseline")]
    preset: Preset,
    #[arg(long, value_enum, default_value = "native")]
    backend: Backend,
    #[cfg(feature = "zephyr-spine-polish")]
    #[arg(long)]
    spine_blocks: Option<usize>,
    #[cfg(feature = "zephyr-spine-polish")]
    #[arg(long, default_value_t = 1)]
    spine_span: usize,
    #[cfg(feature = "zephyr-spine-polish")]
    #[arg(long, requires = "spine_blocks")]
    spine_batch: bool,
    #[cfg(feature = "zephyr-spine-polish")]
    #[arg(long, requires = "spine_blocks")]
    spine_cache_plans: bool,
    #[cfg(feature = "zephyr-spine-polish")]
    #[arg(long, requires = "spine_blocks")]
    spine_snapshot: Option<PathBuf>,
    #[arg(long, default_value_t = 1001)]
    seed: u64,
    #[arg(long, default_value_t = 16)]
    reads: usize,
    #[arg(long, default_value_t = 1024)]
    sweeps: usize,
    #[arg(long, default_value_t = 1.0, conflicts_with = "weak_theta")]
    beta_min: f64,
    #[arg(long, default_value_t = 12.0)]
    beta_max: f64,
    #[arg(long)]
    weak_theta: Option<f64>,
    #[arg(long, default_value_t = 20.0)]
    polish_beta: f64,
    #[arg(long, default_value_t = 256)]
    polish_sweeps: usize,
    #[arg(long, default_value_t = 1)]
    single_every: usize,
    #[arg(long, default_value_t = 250)]
    tile_milli: i32,
    #[arg(long, default_value_t = 250)]
    pair_milli: i32,
    #[arg(long, default_value_t = 1.5)]
    weak_scale: f64,
    #[arg(long, default_value_t = 512)]
    resamples: usize,
    #[arg(long, default_value_t = 0)]
    houdayer_every: usize,
    #[arg(long, default_value_t = 8192)]
    cache_slots: usize,
    #[arg(long)]
    disable_cache: bool,
    #[arg(long)]
    disable_motif_score: bool,
    #[cfg(feature = "tree-polish")]
    #[arg(long)]
    tree_polish: bool,
    #[cfg(feature = "pair-worm-polish")]
    #[arg(long)]
    pair_worm_polish: bool,
}

#[derive(Deserialize)]
struct Topology {
    nodes: Vec<u32>,
    edges: Vec<[u32; 2]>,
}

#[expect(clippy::print_stdout, reason = "benchmark emits one JSON result")]
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if !args.beta_min.is_finite() || args.beta_min <= 0.0 {
        return Err("initial beta must be finite and positive".into());
    }
    if matches!(args.mode, Mode::Kernel) && !matches!(args.backend, Backend::Native) {
        return Err("the alternate backend requires pipeline mode".into());
    }
    #[cfg(feature = "zephyr-spine-polish")]
    if args.spine_blocks.is_some()
        && (!matches!(args.mode, Mode::Pipeline) || args.spine_blocks == Some(0))
    {
        return Err("spine polishing requires pipeline mode and a positive block count".into());
    }
    #[cfg(feature = "zephyr-spine-polish")]
    if !(1..=4).contains(&args.spine_span)
        || (args.spine_blocks.is_none() && matches!(args.backend, Backend::ZephyrSpine))
    {
        return Err("Zephyr spine search requires --spine-blocks and a span in 1..=4".into());
    }
    #[cfg(feature = "zephyr-spine-polish")]
    if args.spine_snapshot.is_some() && (args.spine_blocks != Some(1) || args.reads != 1) {
        return Err("a spine snapshot requires one read and one block".into());
    }
    let spec: Topology = serde_json::from_slice(&std::fs::read(&args.topology)?)?;
    let edges: Vec<_> = spec.edges.into_iter().map(|[u, v]| (u, v)).collect();
    let topology = BaseTopology::new(spec.nodes, &edges).map_err(|e| format!("topology: {e:?}"))?;
    let params = std::fs::read(&args.params)?;
    let revision =
        reference_revision(TWISTED_DIMER_V1.family_id, 1).ok_or("family revision missing")?;
    let nonce = *blake3::hash(&args.seed.to_le_bytes()).as_bytes();
    let instances = export_instances(revision.contract, &params, &topology, &[nonce])?;
    let instance = instances
        .into_iter()
        .next()
        .ok_or("no instance")?
        .map_err(|e| format!("salt generation: {e:?}"))?;
    let all = matches!(args.preset, Preset::Core | Preset::Archive);
    let config = NativeConfig {
        shared_cache: !args.disable_cache && (all || matches!(args.preset, Preset::Cache)),
        copy_resampling: all || matches!(args.preset, Preset::Copy),
        inline_union: all || matches!(args.preset, Preset::Union),
        fused_updates: all || matches!(args.preset, Preset::Fusion),
        archive: matches!(args.preset, Preset::Archive),
        weak_beta_end: args.weak_theta,
        polish_beta: args.polish_beta,
        polish_sweeps: args.polish_sweeps,
        single_every: args.single_every,
        tile_milli: args.tile_milli,
        pair_milli: args.pair_milli,
        weak_scale: args.weak_scale,
        resamples: args.resamples,
        houdayer_every: args.houdayer_every,
        #[cfg(feature = "tree-polish")]
        tree_polish: args.tree_polish,
        #[cfg(feature = "pair-worm-polish")]
        pair_worm_polish: args.pair_worm_polish,
        cache_slots: args.cache_slots,
        motif_score: !args.disable_motif_score,
        ..NativeConfig::default()
    };
    let sample_params = SampleParams {
        num_reads: args.reads,
        num_sweeps: args.sweeps,
        seed: args.seed ^ 0x4350_5542_454e_4348,
        beta_range: args
            .weak_theta
            .is_none()
            .then_some((args.beta_min, args.beta_max)),
        ..SampleParams::default()
    };
    let effective_beta_max = if let Some(theta) = args.weak_theta {
        instance
            .j()
            .iter()
            .map(|v| v.abs())
            .filter(|&v| v > 0 && v < 1000)
            .max()
            .map(|weak| {
                (theta * 1000.0
                    / f64::from(((f64::from(weak) * config.weak_scale) as i32).min(999)))
                .clamp(1.0, 128.0)
            })
            .unwrap_or(12.0)
    } else {
        args.beta_max
    };
    let mut fingerprint = blake3::Hasher::new();
    let (seconds, energies, metrics, diversity) = match args.mode {
        Mode::Kernel => {
            let graph = prepared(&instance, &config, true).map_err(|e| format!("graph: {e:?}"))?;
            let betas: Vec<f64> = (0..args.sweeps)
                .map(|i| {
                    let x = if args.sweeps > 1 {
                        i as f64 / (args.sweeps - 1) as f64
                    } else {
                        1.0
                    };
                    args.beta_min * (effective_beta_max / args.beta_min).powf(x)
                })
                .collect();
            let schedule = Schedule::new(&betas, args.single_every)?;
            let started = Instant::now();
            let outcome = anneal_population_observed(
                &graph,
                &schedule,
                args.reads,
                sample_params.seed,
                PopulationOptions {
                    resamples: config.resamples,
                    houdayer_every: config.houdayer_every,
                    shared_cache: config.shared_cache,
                    cache_slots: config.cache_slots,
                    copy_resampling: config.copy_resampling,
                    inline_union: config.inline_union,
                    fused_updates: config.fused_updates,
                },
                || false,
                |_, _| {},
            )?;
            let seconds = started.elapsed().as_secs_f64();
            let energies: Vec<_> = outcome
                .samples
                .iter()
                .map(|r| {
                    let original = instance_energy(&instance, &r.spins);
                    fingerprint.update(&original.to_le_bytes());
                    fingerprint.update(&r.spins.iter().map(|&s| s as u8).collect::<Vec<_>>());
                    original
                })
                .collect();
            let m = outcome.metrics;
            let c = m.cache;
            (
                seconds,
                energies,
                serde_json::json!({
                    "replica_sweeps":m.replica_sweeps,"resamples":m.resamples,
                    "field_rebuilds":m.field_rebuilds,"field_copies":m.field_copies,
                    "parent_snapshots":m.parent_snapshots,
                    "local_field_bytes":m.local_field_bytes,"snapshot_bytes":m.snapshot_bytes,
                    "coefficient_visits_avoided":m.coefficient_visits_avoided,"cache_bytes":m.cache_bytes,
                    "heat_hits":c.heat_hits,"heat_misses":c.heat_misses,
                    "metropolis_hits":c.metropolis_hits,"metropolis_misses":c.metropolis_misses,
                    "fallbacks":c.fallbacks,"temperatures":c.temperatures,
                    "epoch_wraps":c.epoch_wraps,
                }),
                None,
            )
        }
        Mode::Pipeline => {
            let started = Instant::now();
            let reads = match args.backend {
                Backend::Native => tuned_reads(&instance, &sample_params, &config, || false)
                    .map_err(|e| format!("pipeline: {e:?}"))?
                    .ok_or("unexpected cancellation")?,
                #[cfg(feature = "zephyr-spine-polish")]
                Backend::ZephyrSpine => {
                    let mut rng =
                        SmallRng::seed_from_u64(sample_params.seed ^ 0xfc39_1569_7ab3_f8aa);
                    (0..args.reads)
                        .map(|_| {
                            let spins: Vec<i8> = (0..instance.node_ids().len())
                                .map(|_| if rng.gen() { 1 } else { -1 })
                                .collect();
                            SamplerResult {
                                energy_milli: instance_energy(&instance, &spins)
                                    - instance.offset_milli(),
                                spins,
                            }
                        })
                        .collect()
                }
            };
            #[cfg(feature = "zephyr-spine-polish")]
            let mut reads = reads;
            #[cfg(feature = "zephyr-spine-polish")]
            let spine_stats = args
                .spine_blocks
                .map(|blocks| {
                    let spine_started = Instant::now();
                    let initial_energies: Vec<_> = reads
                        .iter()
                        .map(|read| read.energy_milli + instance.offset_milli())
                        .collect();
                    let snapshot = args.spine_snapshot.as_ref().map(|_| reads[0].spins.clone());
                    let stats = polish_zephyr(
                        &instance,
                        &params,
                        &config,
                        &mut reads,
                        SpinePolishOptions {
                            blocks,
                            span: args.spine_span,
                            batch_reads: args.spine_batch,
                            cache_plans: args.spine_cache_plans,
                        },
                        sample_params.seed,
                        &|| false,
                    )
                    .map_err(|error| format!("spine elimination: {error:?}"))?
                    .ok_or("unexpected cancellation")?;
                    let seconds = spine_started.elapsed().as_secs_f64();
                    if let Some((path, before)) = args.spine_snapshot.as_ref().zip(snapshot) {
                        let (m, t) = quip_solver_cpu::families::native::zephyr_shape(&params)
                            .ok_or("missing Zephyr shape")?;
                        let data = serde_json::json!({
                            "seed": args.seed, "m": m, "t": t, "spines": stats.first_spines,
                            "ids": instance.node_ids(), "edges": instance.edges(), "j": instance.j(),
                            "h": instance.h(), "offset_milli": instance.offset_milli(),
                            "before": before, "after": reads[0].spins,
                            "before_energy_milli": initial_energies[0],
                            "after_energy_milli": reads[0].energy_milli + instance.offset_milli(),
                            "free": stats.max_free, "width": stats.max_width, "seconds": seconds,
                        });
                        serde_json::to_writer(std::fs::File::create_new(path)?, &data)?;
                    }
                    Ok::<_, Box<dyn Error>>((stats, seconds, initial_energies))
                })
                .transpose()?;
            let seconds = started.elapsed().as_secs_f64();
            let diversity = diversity_milli(
                &reads
                    .iter()
                    .map(|read| read.spins.as_slice())
                    .collect::<Vec<_>>(),
            );
            let energies: Vec<_> = reads
                .iter()
                .map(|r| {
                    let original = instance_energy(&instance, &r.spins);
                    if r.energy_milli + instance.offset_milli() != original {
                        return Err::<i64, Box<dyn Error>>(
                            "Original-energy rescoring mismatch".into(),
                        );
                    }
                    fingerprint.update(&original.to_le_bytes());
                    fingerprint.update(&r.spins.iter().map(|&s| s as u8).collect::<Vec<_>>());
                    Ok(original)
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            #[cfg(feature = "zephyr-spine-polish")]
            let metrics =
                spine_stats.map_or(serde_json::Value::Null, |(s, spine_seconds, initial)| {
                    serde_json::json!({
                        "spine_blocks": s.blocks,
                        "spine_span": args.spine_span,
                        "spine_batch": args.spine_batch,
                        "spine_cache_plans": args.spine_cache_plans,
                        "spine_plans_built": s.plans_built,
                        "spine_cached_plan_bytes": s.cached_plan_bytes,
                        "spine_seconds": spine_seconds,
                        "spine_initial_energies_milli": initial,
                        "spine_min_free": s.min_free,
                        "spine_max_free": s.max_free,
                        "spine_max_width": s.max_width,
                        "spine_accepted": s.accepted,
                        "spine_gain_milli": s.gain_milli,
                    })
                });
            #[cfg(not(feature = "zephyr-spine-polish"))]
            let metrics = serde_json::Value::Null;
            (seconds, energies, metrics, Some(diversity))
        }
    };
    let best = energies.iter().min().copied().ok_or("no reads")?;
    let mut row = serde_json::json!({"mode":format!("{:?}",args.mode),"preset":format!("{:?}",args.preset),
        "backend":args.backend.name(),
        "seed":args.seed,"nonce":blake3::Hash::from(nonce).to_hex().to_string(),
        "reads":args.reads,"sweeps":args.sweeps,"beta_min":args.beta_min,"beta_max":effective_beta_max,
        "weak_theta":args.weak_theta,"polish_beta":args.polish_beta,"cache_slots":args.cache_slots,
        "cache_disabled":args.disable_cache,"motif_score_disabled":args.disable_motif_score,
        "seconds":seconds,"best_energy_milli":best,"diversity_milli":diversity,
        "energies_milli":energies,"trajectory_digest":fingerprint.finalize().to_hex().to_string(),"metrics":metrics});
    let build = serde_json::json!({
        "row_grouped_residual_deltas":cfg!(feature="row-grouped-residual-deltas"),
        "masked_residual_deltas":cfg!(feature="masked-residual-deltas"),
        "table_tile_bonds":cfg!(feature="table-tile-bonds"),
        "shallow_union_roots":cfg!(feature="shallow-union-roots"),
        "linked_component_lists":cfg!(feature="linked-component-lists"),
        "packed_residual_fields":cfg!(feature="packed-residual-fields"),
    });
    let schedule = serde_json::json!({
        "polish_sweeps":args.polish_sweeps,
        "single_every":args.single_every,
        "weak_scale":args.weak_scale,
        "tile_milli":args.tile_milli,"pair_milli":args.pair_milli,"resamples":args.resamples,
        "houdayer_every":args.houdayer_every,
    });
    let row_fields = row
        .as_object_mut()
        .ok_or("benchmark output must be an object")?;
    #[cfg(feature = "tree-polish")]
    row_fields.insert("tree_polish".into(), config.tree_polish.into());
    #[cfg(feature = "pair-worm-polish")]
    row_fields.insert("pair_worm_polish".into(), config.pair_worm_polish.into());
    for fields in [build, schedule] {
        if let serde_json::Value::Object(fields) = fields {
            row_fields.extend(fields);
        }
    }
    println!("{row}");
    Ok(())
}
