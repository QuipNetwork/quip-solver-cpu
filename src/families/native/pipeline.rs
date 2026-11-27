//! Population annealing and canonical-objective candidate preservation.

#[cfg(feature = "tree-polish")]
use super::super::otsuka::TreePolish;
use super::super::otsuka::{
    anneal_population_observed, AnnealError, Chain, Model, Motifs, OtsukaGraph, PopulationOptions,
    Schedule, SweepCache, Weights,
};
#[cfg(feature = "zephyr-spine-polish")]
use super::super::otsuka::{SpinePolishOptions, SpinePolishStats, ZephyrSpinePolish};
use super::super::otsuka_sampler::{parse_pairs, section, PAIRS_SECTION};
#[cfg(feature = "pair-worm-polish")]
use crate::postprocess::{
    greedy::{DescentError, Ising},
    pair_worm::{PairWorm, WormOptions},
};
use crate::sampler_core::{build_beta_schedule, effective_sweeps_per_beta};
#[cfg(feature = "zephyr-spine-polish")]
use quip_solver_core::ising_generators::descriptor::{Params, Value};
#[cfg(any(test, feature = "zephyr-spine-polish"))]
use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_core::ising_generators::{ice::twisted_dimer::local_gauges, instance::Instance};
use quip_solver_core::{IsingGraph, SampleError, SampleParams, SamplerResult};
use rand::{rngs::SmallRng, SeedableRng};

use super::NativeConfig;

/// Exact original energy, with the family offset. Useful to independently
/// check every returned spin vector, including weak-biased anneals.
#[must_use]
pub fn instance_energy(instance: &Instance, spins: &[i8]) -> i64 {
    instance
        .h()
        .iter()
        .zip(spins)
        .map(|(&h, &s)| i64::from(h) * i64::from(s))
        .sum::<i64>()
        + instance
            .edges()
            .iter()
            .zip(instance.j())
            .map(|(&(u, v), &j)| {
                i64::from(j) * i64::from(spins[u as usize]) * i64::from(spins[v as usize])
            })
            .sum::<i64>()
        + instance.offset_milli()
}

pub(super) fn graph_of(instance: &Instance) -> IsingGraph {
    IsingGraph::new(
        instance
            .h()
            .iter()
            .map(|&v| f64::from(v) / 1000.0)
            .collect(),
        instance
            .j()
            .iter()
            .map(|&v| f64::from(v) / 1000.0)
            .collect(),
        instance
            .edges()
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect(),
    )
}

/// Prepare a CPU graph from a canonical instance and explicit search weights.
/// `bias` scales weak couplings in the internal search model only.
///
/// # Errors
/// Returns capacity when the metadata or weights do not fit the kernel.
pub fn prepared(
    instance: &Instance,
    config: &NativeConfig,
    bias: bool,
) -> Result<OtsukaGraph, SampleError> {
    let (tiles, gauges) = local_gauges(instance).ok_or(SampleError::Capacity)?;
    let pairs =
        parse_pairs(section(instance.metadata(), PAIRS_SECTION).ok_or(SampleError::Capacity)?)
            .ok_or(SampleError::Capacity)?;
    let j: Vec<i32> = instance
        .j()
        .iter()
        .map(|&v| {
            if bias && v.abs() < 1000 {
                ((f64::from(v) * config.weak_scale) as i32).clamp(-999, 999)
            } else {
                v
            }
        })
        .collect();
    OtsukaGraph::new(
        &Model {
            h_milli: instance.h(),
            edges: instance.edges(),
            j_milli: &j,
        },
        &Motifs {
            tiles: &tiles,
            gauges: &gauges,
            pairs: &pairs,
        },
        Weights {
            tile_milli: config.tile_milli,
            pair_milli: config.pair_milli,
        },
    )
    .map_err(|_| SampleError::Capacity)
}

fn greedy(instance: &Instance, spins: &mut [i8], stopped: &impl Fn() -> bool) -> bool {
    let n = spins.len();
    let mut adj = vec![Vec::new(); n];
    for (&(u, v), &j) in instance.edges().iter().zip(instance.j()) {
        adj[u as usize].push((v as usize, i64::from(j)));
        adj[v as usize].push((u as usize, i64::from(j)));
    }
    let mut fields: Vec<i64> = (0..n)
        .map(|u| {
            i64::from(instance.h()[u])
                + adj[u]
                    .iter()
                    .map(|&(v, j)| j * i64::from(spins[v]))
                    .sum::<i64>()
        })
        .collect();
    loop {
        let mut changed = false;
        for u in 0..n {
            if u.is_multiple_of(256) && stopped() {
                return false;
            }
            let old = i64::from(spins[u]);
            if -2 * old * fields[u] < 0 {
                spins[u] = -spins[u];
                for &(v, j) in &adj[u] {
                    fields[v] -= 2 * old * j;
                }
                changed = true;
            }
        }
        if !changed {
            return true;
        }
    }
}

/// Run the tuned CPU kernel on a canonical instance. Returns `None` on
/// cancellation. Energies exclude the offset, as `Sampler::sample_family`
/// requires; callers managing a lease add it before pushing the salt.
///
/// # Errors
/// Returns capacity when the motifs or budget do not fit this kernel.
pub fn tuned_reads(
    instance: &Instance,
    params: &SampleParams,
    config: &NativeConfig,
    stopped: impl Fn() -> bool,
) -> Result<Option<Vec<SamplerResult>>, SampleError> {
    if stopped() {
        return Ok(None);
    }
    if params.num_reads == 0 || params.num_reads > 128 {
        return Err(SampleError::Capacity);
    }
    let biased = prepared(instance, config, true)?;
    let truth = prepared(instance, config, false)?;
    let graph = graph_of(instance);
    let params = SampleParams {
        beta_range: Some(params.beta_range.unwrap_or_else(|| {
            let weak = instance
                .j()
                .iter()
                .map(|v| v.abs())
                .filter(|&v| v > 0 && v < 1000)
                .max();
            match (config.weak_beta_end, weak) {
                (Some(target), Some(weak)) => {
                    let biased_weak = ((f64::from(weak) * config.weak_scale) as i32).min(999);
                    (
                        1.0,
                        (target * 1000.0 / f64::from(biased_weak)).clamp(1.0, 128.0),
                    )
                }
                _ => (1.0, 12.0),
            }
        })),
        ..params.clone()
    };
    let per_beta = effective_sweeps_per_beta(params.num_sweeps, params.sweeps_per_beta);
    let betas: Vec<f64> = build_beta_schedule(&graph, &params)
        .into_iter()
        .flat_map(|beta| std::iter::repeat_n(beta, per_beta))
        .collect();
    let schedule = Schedule::new(&betas, config.single_every).map_err(|_| SampleError::Capacity)?;
    let weak_levels = config
        .motif_score
        .then(|| uniform_weak_levels(instance, config))
        .flatten();
    let mut archive: Option<SamplerResult> = None;
    let mut last_scored: Option<SamplerResult> = None;
    let outcome = match anneal_population_observed(
        &biased,
        &schedule,
        params.num_reads,
        params.seed,
        PopulationOptions {
            resamples: config.resamples,
            houdayer_every: config.houdayer_every,
            shared_cache: config.shared_cache,
            cache_slots: config.cache_slots,
            copy_resampling: config.copy_resampling,
            inline_union: config.inline_union,
            fused_updates: config.fused_updates,
        },
        &stopped,
        |biased_energy, spins| {
            if !config.archive {
                return;
            }
            let energy_milli = if let Some(previous) = &last_scored {
                if previous.spins == spins {
                    previous.energy_milli
                } else {
                    original_record_energy(&biased, &truth, weak_levels, biased_energy, spins)
                }
            } else {
                original_record_energy(&biased, &truth, weak_levels, biased_energy, spins)
            };
            if last_scored.as_ref().is_none_or(|r| r.spins != spins) {
                last_scored = Some(SamplerResult {
                    spins: spins.to_vec(),
                    energy_milli,
                });
            }
            if archive
                .as_ref()
                .is_none_or(|r| energy_milli < r.energy_milli)
            {
                archive = Some(SamplerResult {
                    spins: spins.to_vec(),
                    energy_milli,
                });
            }
        },
    ) {
        Ok(outcome) => outcome,
        Err(AnnealError::Cancelled) => return Ok(None),
    };
    tracing::debug!(metrics=?outcome.metrics, "CPU native population metrics");
    let mut reads: Vec<SamplerResult> = outcome
        .samples
        .into_iter()
        .map(|sample| {
            let energy_milli = truth.energy(&sample.spins);
            SamplerResult {
                spins: sample.spins,
                energy_milli,
            }
        })
        .collect();
    if config.polish_sweeps == 0 {
        if let Some(record) = archive {
            retain_record(&mut reads, record, params.num_reads);
        }
        #[cfg(feature = "tree-polish")]
        if config.tree_polish && !polish_trees(&truth, &mut reads, params.seed, &stopped) {
            return Ok(None);
        }
        #[cfg(feature = "pair-worm-polish")]
        if config.pair_worm_polish
            && !polish_pair_worm(instance, &mut reads, params.seed, &stopped)?
        {
            return Ok(None);
        }
        return Ok(Some(reads));
    }
    for read in &mut reads {
        if !greedy(instance, &mut read.spins, &stopped) {
            return Ok(None);
        }
        read.energy_milli = truth.energy(&read.spins);
    }
    let mut order: Vec<usize> = (0..reads.len()).collect();
    order.sort_unstable_by_key(|&i| reads[i].energy_milli);
    let mut rng = SmallRng::seed_from_u64(params.seed ^ 0x713c_0145_504f_4c49);
    let mut polish_cache = SweepCache::new(config.cache_slots);
    for i in order.into_iter().take(4) {
        let read = &mut reads[i];
        let mut chain =
            Chain::new(&truth, read.spins.clone()).map_err(|_| SampleError::Capacity)?;
        let mut energy = read.energy_milli;
        for _ in 0..config.polish_sweeps {
            if stopped() {
                return Ok(None);
            }
            energy += chain.cluster_sweep_optimized(
                config.polish_beta,
                config.shared_cache.then_some(&mut polish_cache),
                config.inline_union,
                config.fused_updates,
                &mut rng,
            );
            energy += if config.shared_cache {
                chain.single_sweep_cached(config.polish_beta, &mut polish_cache, &mut rng)
            } else {
                chain.single_sweep(config.polish_beta, &mut rng)
            };
            if energy < read.energy_milli {
                read.spins.copy_from_slice(chain.spins());
                read.energy_milli = energy;
            }
        }
        if !greedy(instance, &mut read.spins, &stopped) {
            return Ok(None);
        }
        read.energy_milli = truth.energy(&read.spins);
    }
    if let Some(mut record) = archive {
        if !reads.iter().any(|r| r.spins == record.spins) {
            if !greedy(instance, &mut record.spins, &stopped) {
                return Ok(None);
            }
            record.energy_milli = truth.energy(&record.spins);
            let mut chain =
                Chain::new(&truth, record.spins.clone()).map_err(|_| SampleError::Capacity)?;
            let mut rng = SmallRng::seed_from_u64(params.seed ^ 0x4350_555f_4152_4348);
            let mut energy = record.energy_milli;
            let mut cache = SweepCache::new(config.cache_slots);
            for _ in 0..config.polish_sweeps {
                if stopped() {
                    return Ok(None);
                }
                energy += chain.cluster_sweep_optimized(
                    config.polish_beta,
                    config.shared_cache.then_some(&mut cache),
                    config.inline_union,
                    config.fused_updates,
                    &mut rng,
                );
                energy += if config.shared_cache {
                    chain.single_sweep_cached(config.polish_beta, &mut cache, &mut rng)
                } else {
                    chain.single_sweep(config.polish_beta, &mut rng)
                };
                if energy < record.energy_milli {
                    record.spins.copy_from_slice(chain.spins());
                    record.energy_milli = energy;
                }
            }
            if !greedy(instance, &mut record.spins, &stopped) {
                return Ok(None);
            }
            record.energy_milli = truth.energy(&record.spins);
        }
        retain_record(&mut reads, record, params.num_reads);
    }
    #[cfg(feature = "tree-polish")]
    if config.tree_polish && !polish_trees(&truth, &mut reads, params.seed, &stopped) {
        return Ok(None);
    }
    #[cfg(feature = "pair-worm-polish")]
    if config.pair_worm_polish && !polish_pair_worm(instance, &mut reads, params.seed, &stopped)? {
        return Ok(None);
    }
    Ok(Some(reads))
}

#[cfg(feature = "pair-worm-polish")]
fn polish_pair_worm(
    instance: &Instance,
    reads: &mut [SamplerResult],
    seed: u64,
    stopped: &impl Fn() -> bool,
) -> Result<bool, SampleError> {
    let (tiles, _) = local_gauges(instance).ok_or(SampleError::Capacity)?;
    let pairs =
        parse_pairs(section(instance.metadata(), PAIRS_SECTION).ok_or(SampleError::Capacity)?)
            .ok_or(SampleError::Capacity)?;
    let h: Vec<_> = instance.h().iter().map(|&value| i64::from(value)).collect();
    let j: Vec<_> = instance.j().iter().map(|&value| i64::from(value)).collect();
    let ising = Ising::new(&h, instance.edges(), &j).map_err(|_| SampleError::Capacity)?;
    let worm = PairWorm::new(&ising, &tiles, &pairs).map_err(|_| SampleError::Capacity)?;
    let mut order: Vec<_> = (0..reads.len()).collect();
    order.sort_by_key(|&index| reads[index].energy_milli);
    let settings = WormOptions {
        max_sweeps: 8,
        max_len: 64,
        ..WormOptions::default()
    };
    for index in order.into_iter().take(8) {
        let read = &mut reads[index];
        let local_seed = seed ^ 0xa076_1d64_78bd_642f_u64.wrapping_mul(index as u64 + 1);
        match worm.descend_cancellable(&ising, &mut read.spins, &settings, local_seed, stopped) {
            Ok(_) => read.energy_milli = ising.energy(&read.spins),
            Err(DescentError::Cancelled) => return Ok(false),
            Err(_) => return Err(SampleError::Capacity),
        }
    }
    Ok(true)
}

#[cfg(feature = "tree-polish")]
fn polish_trees(
    graph: &OtsukaGraph,
    reads: &mut [SamplerResult],
    seed: u64,
    stopped: &impl Fn() -> bool,
) -> bool {
    TreePolish::new(graph, seed, stopped)
        .and_then(|mut polish| polish.improve(reads, stopped))
        .is_some()
}

/// Decode the Zephyr size and lane count from a twisted-dimer descriptor.
#[cfg(feature = "zephyr-spine-polish")]
#[must_use]
pub fn zephyr_shape(descriptor: &[u8]) -> Option<(u16, u8)> {
    let params = Params::parse(TWISTED_DIMER_V1.schema, descriptor).ok()?;
    let (Some(Value::Unsigned(m)), Some(Value::Unsigned(t))) = (params.get(1), params.get(2))
    else {
        return None;
    };
    Some((u16::try_from(m).ok()?, u8::try_from(t).ok()?))
}

/// Optimize fixed-boundary Zephyr neighborhoods on canonical coefficients.
///
/// # Errors
/// Returns capacity when the shape or a neighborhood falls outside the
/// bounded kernel. Returns `Ok(None)` when the caller cancels the pass.
#[cfg(feature = "zephyr-spine-polish")]
pub fn polish_zephyr(
    instance: &Instance,
    descriptor: &[u8],
    config: &NativeConfig,
    reads: &mut [SamplerResult],
    options: SpinePolishOptions,
    seed: u64,
    stopped: &impl Fn() -> bool,
) -> Result<Option<SpinePolishStats>, SampleError> {
    if stopped() {
        return Ok(None);
    }
    let (m, t) = zephyr_shape(descriptor).ok_or(SampleError::Capacity)?;
    let graph = prepared(instance, config, false)?;
    let polish =
        ZephyrSpinePolish::new(&graph, instance.node_ids(), m, t).ok_or(SampleError::Capacity)?;
    match polish.improve(reads, options, seed, stopped) {
        None if !stopped() => Err(SampleError::Capacity),
        stats => Ok(stats),
    }
}

fn retain_record(reads: &mut Vec<SamplerResult>, record: SamplerResult, limit: usize) {
    if reads.iter().any(|r| r.spins == record.spins) {
        return;
    }
    reads.push(record);
    reads.sort_by_key(|r| r.energy_milli);
    reads.truncate(limit);
}

fn uniform_weak_levels(instance: &Instance, config: &NativeConfig) -> Option<(i64, i64)> {
    let weak = instance
        .j()
        .iter()
        .map(|j| j.abs())
        .filter(|&j| j < 1000)
        .max()?;
    if instance
        .j()
        .iter()
        .any(|j| j.abs() < 1000 && j.abs() != weak)
    {
        return None;
    }
    Some((
        i64::from(weak),
        i64::from(((f64::from(weak) * config.weak_scale) as i32).min(999)),
    ))
}

fn original_record_energy(
    biased: &OtsukaGraph,
    truth: &OtsukaGraph,
    levels: Option<(i64, i64)>,
    energy: i64,
    spins: &[i8],
) -> i64 {
    let Some((original_weak, search_weak)) = levels else {
        return truth.energy(spins);
    };
    if search_weak == 0 {
        return energy;
    }
    let strong = biased.full_strong_energy(spins);
    let weak_energy = energy - strong;
    if weak_energy % search_weak != 0 {
        return truth.energy(spins);
    }
    strong + (weak_energy / search_weak) * original_weak
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_solver_conformance::families::{export_instances, reference_revision};
    use quip_solver_core::ising_generators::family::BaseTopology;

    fn best_energy(reads: &[SamplerResult]) -> i64 {
        reads.iter().map(|read| read.energy_milli).min().unwrap()
    }

    fn reference() -> Vec<Instance> {
        let revision =
            reference_revision(TWISTED_DIMER_V1.family_id, TWISTED_DIMER_V1.revision).unwrap();
        let stored = quip_solver_conformance::topology(revision.topology).unwrap();
        let topology = BaseTopology::new(stored.nodes, &stored.edges).unwrap();
        let nonces: Vec<_> = (0..64_u8).map(|n| [n; 32]).collect();
        for params in revision.params {
            if let Ok(instances) = export_instances(revision.contract, params, &topology, &nonces) {
                let instances: Vec<_> = instances.into_iter().filter_map(Result::ok).collect();
                if instances.len() == nonces.len() {
                    return instances;
                }
            }
        }
        unreachable!("the reference has a feasible parameter set")
    }

    #[test]
    fn biased_anneals_report_original_energies_and_polish_never_raises_them() {
        let instances = reference();
        let config = NativeConfig {
            polish_sweeps: 0,
            ..NativeConfig::default()
        };
        for instance in instances.iter().take(8) {
            let params = SampleParams {
                num_reads: 8,
                num_sweeps: 128,
                seed: 731,
                ..SampleParams::default()
            };
            let before = tuned_reads(instance, &params, &config, || false)
                .unwrap()
                .unwrap();
            let config = NativeConfig {
                polish_sweeps: 32,
                ..config.clone()
            };
            let after = tuned_reads(instance, &params, &config, || false)
                .unwrap()
                .unwrap();
            assert_eq!(before.len(), after.len());
            for (raw, polished) in before.iter().zip(&after) {
                assert!(polished.energy_milli <= raw.energy_milli);
                assert_eq!(
                    instance_energy(instance, &polished.spins),
                    polished.energy_milli + instance.offset_milli()
                );
            }
        }
    }

    #[test]
    fn native_execution_preserves_pipeline_reads_and_archive_never_loses_the_best() {
        let instances = reference();
        for instance in instances.iter().take(4) {
            let params = SampleParams {
                num_reads: 4,
                num_sweeps: 96,
                seed: 3311,
                ..SampleParams::default()
            };
            let baseline = NativeConfig {
                polish_sweeps: 16,
                ..NativeConfig::default()
            };
            let native = NativeConfig {
                shared_cache: true,
                copy_resampling: true,
                inline_union: true,
                fused_updates: true,
                ..baseline.clone()
            };
            let a = tuned_reads(instance, &params, &baseline, || false)
                .unwrap()
                .unwrap();
            let b = tuned_reads(instance, &params, &native, || false)
                .unwrap()
                .unwrap();
            assert_eq!(a.len(), b.len());
            for (left, right) in a.iter().zip(&b) {
                assert_eq!(left.spins, right.spins);
                assert_eq!(left.energy_milli, right.energy_milli);
            }
            let archive = NativeConfig {
                archive: true,
                ..native
            };
            let c = tuned_reads(instance, &params, &archive, || false)
                .unwrap()
                .unwrap();
            let scalar_archive = NativeConfig {
                motif_score: false,
                ..archive.clone()
            };
            let scalar = tuned_reads(instance, &params, &scalar_archive, || false)
                .unwrap()
                .unwrap();
            for (left, right) in c.iter().zip(&scalar) {
                assert_eq!(left.spins, right.spins);
                assert_eq!(left.energy_milli, right.energy_milli);
            }
            assert_eq!(c.len(), params.num_reads);
            assert!(best_energy(&c) <= best_energy(&b));
            for read in c {
                assert_eq!(
                    read.energy_milli + instance.offset_milli(),
                    instance_energy(instance, &read.spins)
                );
            }
        }
    }

    #[cfg(feature = "pair-worm-polish")]
    #[test]
    fn pair_worm_polish_keeps_candidates_and_lowers_original_energy() {
        let instances = reference();
        let instance = &instances[0];
        for polish_sweeps in [0, 4] {
            let config = NativeConfig {
                archive: true,
                polish_sweeps,
                ..NativeConfig::default()
            };
            let mut polished = config.clone();
            polished
                .apply(
                    &"native_pair_worm_polish = true"
                        .parse::<toml::Table>()
                        .unwrap(),
                )
                .unwrap();
            let params = SampleParams {
                num_reads: 4,
                num_sweeps: 32,
                seed: 5281,
                ..SampleParams::default()
            };
            let before = tuned_reads(instance, &params, &config, || false)
                .unwrap()
                .unwrap();
            let after = tuned_reads(instance, &params, &polished, || false)
                .unwrap()
                .unwrap();
            assert_eq!(after.len(), params.num_reads);
            assert_eq!(before.len(), after.len());
            for (original, polished) in before.iter().zip(&after) {
                assert!(polished.energy_milli <= original.energy_milli);
                assert_eq!(
                    polished.energy_milli + instance.offset_milli(),
                    instance_energy(instance, &polished.spins)
                );
            }
        }
    }

    #[cfg(feature = "tree-polish")]
    #[test]
    fn configured_tree_polish_preserves_candidate_count_and_canonical_energy() {
        let instances = reference();
        let config = NativeConfig {
            polish_sweeps: 0,
            archive: true,
            ..NativeConfig::default()
        };
        let mut polished = config.clone();
        let settings = "native_tree_polish = true".parse::<toml::Table>().unwrap();
        polished.apply(&settings).unwrap();
        let params = SampleParams {
            num_reads: 4,
            num_sweeps: 16,
            seed: 5309,
            ..SampleParams::default()
        };
        let instance = &instances[0];
        let baseline = tuned_reads(instance, &params, &config, || false)
            .unwrap()
            .unwrap();
        let candidate = tuned_reads(instance, &params, &polished, || false)
            .unwrap()
            .unwrap();
        assert_eq!(baseline.len(), candidate.len());
        assert_eq!(candidate.len(), params.num_reads);
        assert!(best_energy(&candidate) <= best_energy(&baseline));
        for read in candidate {
            assert_eq!(
                read.energy_milli + instance.offset_milli(),
                instance_energy(instance, &read.spins)
            );
        }
    }

    #[test]
    fn motif_original_energy_conversion_matches_full_canonical_rescoring() {
        use rand::Rng;
        let instances = reference();
        let mut rng = SmallRng::seed_from_u64(87142);
        for instance in instances.iter().take(4) {
            for scale in [1.0, 1.25, 1.5] {
                let config = NativeConfig {
                    weak_scale: scale,
                    ..NativeConfig::default()
                };
                let biased = prepared(instance, &config, true).unwrap();
                let truth = prepared(instance, &config, false).unwrap();
                let levels = uniform_weak_levels(instance, &config);
                assert!(levels.is_some());
                for _ in 0..32 {
                    let spins: Vec<i8> = (0..instance.h().len())
                        .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
                        .collect();
                    let actual = original_record_energy(
                        &biased,
                        &truth,
                        levels,
                        biased.energy(&spins),
                        &spins,
                    );
                    assert_eq!(
                        actual + instance.offset_milli(),
                        instance_energy(instance, &spins)
                    );
                }
            }
        }
    }
}
