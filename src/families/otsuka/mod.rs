//! Gauge-aware Otsuka cluster annealing for corner-sharing ice.
//!
//! Port of the tailored research kernel `fast_bench.rs` (research inventory
//! `cpu-fast-bench`, BLAKE3 `338910fe2d252a16`). A tile is four nodes with a
//! local gauge and strong couplings [`STRONG_MILLI`]` * g[a] * g[b]`. A pair is
//! an extra [`PAIR_MILLI`] edge. Every other edge is weak. Each sweep samples
//! the strong layer with auxiliary bonds and flips the resulting components by
//! heat bath against the weak layer.
//!
//! The kernel supports zero fields only, and soft pair and tile weights. The
//! source's replica exchange, restarts, strip polish, and repeated heat-bath
//! passes are not part of this port.
//!
//! [`OtsukaGraph::new`] checks what the update needs: edge-disjoint tiles whose
//! couplings match their gauges, node-disjoint pairs on non-tile edges, and no
//! other strong edge. It does not check that the input is a canonical
//! `quip/twisted-ice-dimer` instance. The family's metadata check does that
//! with the declared parameters, before preparation.

mod graph;
#[cfg(feature = "zephyr-spine-polish")]
mod spine;
#[cfg(feature = "tree-polish")]
mod tree;
mod update;

use quip_solver_core::CancelToken;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::fmt;
use update::LocalField;

pub use graph::{
    GraphError, Model, Motifs, OtsukaGraph, Weights, MAX_NODES, PAIR_MILLI, STRONG_MILLI,
};
#[cfg(feature = "zephyr-spine-polish")]
pub use spine::{SpinePolishOptions, SpinePolishStats, ZephyrSpinePolish};
#[cfg(feature = "tree-polish")]
pub(crate) use tree::TreePolish;
pub use update::{CacheStats, Chain, SpinError, SweepCache};

/// Why a schedule cannot drive an anneal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    /// The schedule has no sweeps.
    Empty,
    /// A beta is not finite and positive.
    Beta {
        /// The sweep position.
        index: usize,
    },
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "the schedule has no sweeps"),
            Self::Beta { index } => write!(f, "beta {index} is not finite and positive"),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// A checked anneal schedule: one beta per sweep, in inverse units of 1000
/// milli, and a Metropolis sweep after every `single_every` cluster sweeps.
#[derive(Clone, Copy, Debug)]
pub struct Schedule<'a> {
    betas: &'a [f64],
    single_every: usize,
}

impl<'a> Schedule<'a> {
    /// Check `betas`. `single_every = 0` runs no Metropolis sweep.
    ///
    /// # Errors
    ///
    /// [`ScheduleError::Empty`] for no betas, [`ScheduleError::Beta`] for the
    /// first beta that is not finite and positive.
    pub fn new(betas: &'a [f64], single_every: usize) -> Result<Self, ScheduleError> {
        if betas.is_empty() {
            return Err(ScheduleError::Empty);
        }
        if let Some(index) = betas.iter().position(|&b| !(b.is_finite() && b > 0.0)) {
            return Err(ScheduleError::Beta { index });
        }
        Ok(Self {
            betas,
            single_every,
        })
    }
}

/// The lowest-energy state an anneal visited.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sample {
    /// One spin per node, `1` or `-1`.
    pub spins: Vec<i8>,
    /// The model energy of `spins`, in milli.
    pub energy_milli: i64,
    /// The sweeps run.
    pub sweeps: usize,
}

/// Why an anneal stopped without a sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnnealError {
    /// The cancel token abandoned this job's watermark.
    Cancelled,
}

impl fmt::Display for AnnealError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the anneal was cancelled")
    }
}

impl std::error::Error for AnnealError {}

/// Anneal `graph` from a random state drawn from `seed`, one cluster sweep
/// per beta. The cancel token is checked before each sweep, so one sweep
/// bounds the time a cancel waits.
///
/// # Errors
///
/// [`AnnealError::Cancelled`] when the token abandons the watermark.
pub fn anneal(
    graph: &OtsukaGraph,
    schedule: &Schedule<'_>,
    seed: u64,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Sample, AnnealError> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut chain = Chain::random(graph, &mut rng);
    let mut energy = graph.energy(chain.spins());
    let mut best = (chain.spins().to_vec(), energy);
    for (sweep, &beta) in schedule.betas.iter().enumerate() {
        if cancel.is_some_and(|(token, watermark)| token.is_cancelled(watermark)) {
            return Err(AnnealError::Cancelled);
        }
        energy += chain.cluster_sweep(beta, &mut rng);
        if schedule.single_every > 0 && (sweep + 1).is_multiple_of(schedule.single_every) {
            energy += chain.single_sweep(beta, &mut rng);
        }
        if energy < best.1 {
            best.0.copy_from_slice(chain.spins());
            best.1 = energy;
        }
    }
    Ok(Sample {
        spins: best.0,
        energy_milli: best.1,
        sweeps: schedule.betas.len(),
    })
}

/// How many times a population anneal resamples over its schedule.
const RESAMPLES: usize = 1024;

/// The beta, per full coupler, from which a population anneal pairs its
/// replicas for Houdayer moves. Below it the replicas differ on most nodes,
/// so a cluster spans nearly all of them and the move only swaps the pair.
const HOUDAYER_BETA: f64 = 3.0;

/// Cadence of population resampling and replica-cluster swaps.
#[derive(Clone, Copy, Debug)]
pub struct PopulationOptions {
    /// Resampling checkpoints over the anneal; zero disables resampling.
    pub resamples: usize,
    /// Sweeps between Houdayer moves; zero uses the resampling cadence.
    pub houdayer_every: usize,
    /// Share temperature-local transition probabilities across replicas.
    pub shared_cache: bool,
    /// Positive half-delta slots in the bounded probability table.
    pub cache_slots: usize,
    /// Transfer spins and their exact local fields during resampling.
    pub copy_resampling: bool,
    /// Inline the monotone union-find path in the cluster draw.
    pub inline_union: bool,
    /// Patch residual deltas and full local fields in one neighbor pass.
    pub fused_updates: bool,
}

impl Default for PopulationOptions {
    fn default() -> Self {
        Self {
            resamples: RESAMPLES,
            houdayer_every: 1,
            shared_cache: false,
            cache_slots: 8192,
            copy_resampling: false,
            inline_union: false,
            fused_updates: false,
        }
    }
}

/// Operation counts for one population anneal.
#[derive(Clone, Copy, Debug, Default)]
pub struct PopulationMetrics {
    /// Cluster sweeps executed across replicas.
    pub replica_sweeps: usize,
    /// Population-resampling checkpoints executed.
    pub resamples: usize,
    /// Local-field rebuilds from adjacency, including initial states.
    pub field_rebuilds: usize,
    /// Replica fields restored from a captured parent state.
    pub field_copies: usize,
    /// Parent rows captured before restoring resampled children.
    pub parent_snapshots: usize,
    /// Field payload allocated across the live replicas.
    pub local_field_bytes: usize,
    /// Payload reserved for parent snapshots, excluding small vectors.
    pub snapshot_bytes: usize,
    /// Adjacency entries not revisited because parent fields were copied.
    pub coefficient_visits_avoided: usize,
    /// Record-improvement states passed to the observer.
    pub observer_records: usize,
    /// Probability table size in bytes.
    pub cache_bytes: usize,
    /// Temperature-local probability counters.
    pub cache: CacheStats,
}

/// Population samples and the operation counts of their anneal.
#[derive(Debug)]
pub struct PopulationOutcome {
    /// Best retained state of each replica.
    pub samples: Vec<Sample>,
    /// Executed work and cache activity.
    pub metrics: PopulationMetrics,
}

/// Fixed-size storage of parent spins and fields, reused at every checkpoint.
struct PopulationSnapshot {
    nodes: usize,
    spins: Vec<i8>,
    fields: Vec<LocalField>,
}

impl PopulationSnapshot {
    fn new(replicas: usize, nodes: usize) -> Self {
        Self {
            nodes,
            spins: vec![0; replicas * nodes],
            fields: vec![0; replicas * nodes],
        }
    }

    fn capture(&mut self, chains: &[Chain<'_>]) {
        for (i, chain) in chains.iter().enumerate() {
            let row = i * self.nodes..(i + 1) * self.nodes;
            chain.save_state(&mut self.spins[row.clone()], &mut self.fields[row]);
        }
    }

    fn restore(&self, chain: &mut Chain<'_>, parent: usize) {
        let row = parent * self.nodes..(parent + 1) * self.nodes;
        chain.restore_state(&self.spins[row.clone()], &self.fields[row]);
    }
}

/// Anneal `replicas` states of `graph` together over one schedule, the
/// population annealing of Hukushima and Iba (2003) with Otsuka sweeps.
///
/// Each replica starts at a random state drawn from `seed`. Every
/// `max(1, sweeps / 1024)` sweeps the population is resampled in proportion
/// to `exp(-(beta - beta_last) E)`, with systematic resampling, so the count
/// stays `replicas` and low-energy states take over from high-energy ones.
/// A copy keeps its parent's lowest state. From `beta` 3, after every sweep
/// the replicas are paired at random and each pair takes one Houdayer move.
/// The result holds each replica's lowest state, in replica order. The
/// cancel token is checked before each sweep of each replica.
///
/// On the registered twisted-dimer cells, at equal sweeps, this reaches
/// lower energies than independent anneals: the time goes to the replicas
/// that are already low.
///
/// # Errors
///
/// [`AnnealError::Cancelled`] when the token abandons the watermark.
pub fn anneal_population(
    graph: &OtsukaGraph,
    schedule: &Schedule<'_>,
    replicas: usize,
    seed: u64,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<Sample>, AnnealError> {
    anneal_population_configured(
        graph,
        schedule,
        replicas,
        seed,
        PopulationOptions::default(),
        || cancel.is_some_and(|(token, watermark)| token.is_cancelled(watermark)),
    )
}

/// Population annealing with an explicit cadence and cancellation callback.
/// The default options preserve the stock kernel's draws and trajectories.
///
/// # Errors
/// Returns [`AnnealError::Cancelled`] as soon as `stopped` becomes true.
pub fn anneal_population_configured(
    graph: &OtsukaGraph,
    schedule: &Schedule<'_>,
    replicas: usize,
    seed: u64,
    options: PopulationOptions,
    stopped: impl Fn() -> bool,
) -> Result<Vec<Sample>, AnnealError> {
    anneal_population_observed(graph, schedule, replicas, seed, options, stopped, |_, _| {})
        .map(|outcome| outcome.samples)
}

/// Population annealing with CPU update options and a record-state observer.
/// The observer sees initial states and replica record improvements,
/// including cluster states before a following Metropolis sweep.
/// It does not change replica ancestry or random draws.
///
/// # Errors
/// Returns [`AnnealError::Cancelled`] when `stopped` becomes true.
pub fn anneal_population_observed(
    graph: &OtsukaGraph,
    schedule: &Schedule<'_>,
    replicas: usize,
    seed: u64,
    options: PopulationOptions,
    stopped: impl Fn() -> bool,
    mut observe: impl FnMut(i64, &[i8]),
) -> Result<PopulationOutcome, AnnealError> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut chains: Vec<Chain<'_>> = (0..replicas)
        .map(|_| Chain::random(graph, &mut rng))
        .collect();
    let mut energy: Vec<i64> = chains.iter().map(|c| graph.energy(c.spins())).collect();
    let mut best: Vec<(Vec<i8>, i64)> = chains
        .iter()
        .zip(&energy)
        .map(|(c, &e)| (c.spins().to_vec(), e))
        .collect();
    let mut metrics = PopulationMetrics {
        field_rebuilds: replicas,
        local_field_bytes: replicas * graph.n * size_of::<LocalField>(),
        snapshot_bytes: if options.copy_resampling {
            replicas * graph.n * (size_of::<i8>() + size_of::<LocalField>())
        } else {
            0
        },
        ..PopulationMetrics::default()
    };
    for (chain, &e) in chains.iter().zip(&energy) {
        observe(e, chain.spins());
        metrics.observer_records += 1;
    }
    let mut cache = options
        .shared_cache
        .then(|| SweepCache::new(options.cache_slots));
    let mut snapshot = options
        .copy_resampling
        .then(|| PopulationSnapshot::new(replicas, graph.n));
    let every = schedule
        .betas
        .len()
        .checked_div(options.resamples)
        .map_or(usize::MAX, |every| every.max(1));
    let houdayer_every = if options.houdayer_every == 0 {
        every
    } else {
        options.houdayer_every
    };
    let mut order: Vec<usize> = (0..replicas).collect();
    let mut scratch = Houdayer::new(graph.n);
    let mut last_beta = schedule.betas[0];
    for (sweep, &beta) in schedule.betas.iter().enumerate() {
        if sweep > 0 && sweep % every == 0 && replicas > 1 {
            let picks = resample(&energy, beta - last_beta, &mut rng);
            if let Some(snapshot) = &mut snapshot {
                snapshot.capture(&chains);
                metrics.parent_snapshots += replicas;
                for (chain, &parent) in chains.iter_mut().zip(&picks) {
                    snapshot.restore(chain, parent);
                }
                metrics.field_copies += replicas;
                metrics.coefficient_visits_avoided += replicas * graph.all.link_count();
            } else {
                let states: Vec<Vec<i8>> =
                    picks.iter().map(|&i| chains[i].spins().to_vec()).collect();
                for (chain, state) in chains.iter_mut().zip(&states) {
                    chain.reset(state).expect("a state of the same graph");
                }
                metrics.field_rebuilds += replicas;
            }
            metrics.resamples += 1;
            energy = picks.iter().map(|&i| energy[i]).collect();
            best = picks.iter().map(|&i| best[i].clone()).collect();
            last_beta = beta;
        }
        for ((chain, e), lowest) in chains.iter_mut().zip(&mut energy).zip(&mut best) {
            if stopped() {
                return Err(AnnealError::Cancelled);
            }
            metrics.replica_sweeps += 1;
            if options.shared_cache || options.inline_union || options.fused_updates {
                *e += chain.cluster_sweep_optimized(
                    beta,
                    cache.as_mut(),
                    options.inline_union,
                    options.fused_updates,
                    &mut rng,
                );
            } else {
                *e += chain.cluster_sweep(beta, &mut rng);
            }
            if *e < lowest.1 {
                observe(*e, chain.spins());
                metrics.observer_records += 1;
            }
            if schedule.single_every > 0 && (sweep + 1).is_multiple_of(schedule.single_every) {
                *e += if let Some(cache) = &mut cache {
                    chain.single_sweep_cached(beta, cache, &mut rng)
                } else {
                    chain.single_sweep(beta, &mut rng)
                };
                if *e < lowest.1 {
                    observe(*e, chain.spins());
                    metrics.observer_records += 1;
                }
            }
            if *e < lowest.1 {
                lowest.0.copy_from_slice(chain.spins());
                lowest.1 = *e;
            }
        }
        if replicas > 1 && beta >= HOUDAYER_BETA && (sweep + 1).is_multiple_of(houdayer_every) {
            order.shuffle(&mut rng);
            for pair in order.chunks_exact(2) {
                let (a, b) = (pair[0].min(pair[1]), pair[0].max(pair[1]));
                let (left, right) = chains.split_at_mut(b);
                let (ca, cb) = (&mut left[a], &mut right[0]);
                if let Some((da, db)) = houdayer(graph, ca, cb, &mut scratch, &mut rng) {
                    energy[a] += da;
                    energy[b] += db;
                    for i in [a, b] {
                        if energy[i] < best[i].1 {
                            observe(energy[i], chains[i].spins());
                            metrics.observer_records += 1;
                            best[i].0.copy_from_slice(chains[i].spins());
                            best[i].1 = energy[i];
                        }
                    }
                }
            }
        }
    }
    if let Some(cache) = cache {
        metrics.cache = cache.stats();
        metrics.cache_bytes = cache.table_bytes();
    }
    Ok(PopulationOutcome {
        samples: best
            .into_iter()
            .map(|(spins, energy_milli)| Sample {
                spins,
                energy_milli,
                sweeps: schedule.betas.len(),
            })
            .collect(),
        metrics,
    })
}

/// Scratch space of a Houdayer move.
struct Houdayer {
    seen: Vec<bool>,
    differ: Vec<u32>,
    cluster: Vec<u32>,
}

impl Houdayer {
    fn new(n: usize) -> Self {
        Self {
            seen: vec![false; n],
            differ: Vec::new(),
            cluster: Vec::new(),
        }
    }
}

/// A Houdayer move: pick a node where `a` and `b` differ, grow its connected
/// cluster of differing nodes over every coupling, and swap the cluster
/// between the replicas. Every edge that leaves the cluster ends on a node
/// where the replicas agree, so the sum of the two energies is unchanged.
/// Returns the energy change of each, or `None` when they agree everywhere.
fn houdayer(
    graph: &OtsukaGraph,
    a: &mut Chain<'_>,
    b: &mut Chain<'_>,
    scratch: &mut Houdayer,
    rng: &mut SmallRng,
) -> Option<(i64, i64)> {
    let (sa, sb) = (a.spins(), b.spins());
    scratch.differ.clear();
    scratch
        .differ
        .extend((0..graph.n as u32).filter(|&u| sa[u as usize] != sb[u as usize]));
    if scratch.differ.is_empty() {
        return None;
    }
    let root = scratch.differ[rng.gen_range(0..scratch.differ.len())];
    scratch.cluster.clear();
    scratch.cluster.push(root);
    scratch.seen[root as usize] = true;
    let mut at = 0;
    while at < scratch.cluster.len() {
        let u = scratch.cluster[at] as usize;
        at += 1;
        for link in graph.all.row(u) {
            let v = link.node();
            if !scratch.seen[v] && sa[v] != sb[v] {
                scratch.seen[v] = true;
                scratch.cluster.push(v as u32);
            }
        }
    }
    for &u in &scratch.cluster {
        scratch.seen[u as usize] = false;
    }
    let change = (
        a.flip_nodes(&scratch.cluster),
        b.flip_nodes(&scratch.cluster),
    );
    debug_assert_eq!(change.0 + change.1, 0, "a swap keeps the summed energy");
    Some(change)
}

/// Systematic resampling: the replica each slot copies, drawn in proportion
/// to `exp(-step * (E - E_min))` with one uniform offset.
fn resample(energy: &[i64], step: f64, rng: &mut SmallRng) -> Vec<usize> {
    let low = energy.iter().copied().min().unwrap_or(0);
    let weights: Vec<f64> = energy
        .iter()
        .map(|&e| (-step * (e - low) as f64 / 1000.0).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let stride = total / energy.len() as f64;
    let mut mark = rng.gen::<f64>() * stride;
    let mut picks = Vec::with_capacity(energy.len());
    let mut reach = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        reach += w;
        while mark < reach && picks.len() < energy.len() {
            picks.push(i);
            mark += stride;
        }
    }
    // Rounding can leave the last slot short; it takes the last replica.
    picks.resize(energy.len(), energy.len() - 1);
    picks
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;

    #[test]
    fn snapshots_preserve_cycles_repeated_parents_and_fields() {
        let graph = OtsukaGraph::new(
            &Model {
                h_milli: &[0; 3],
                edges: &[(0, 1), (1, 2), (0, 2)],
                j_milli: &[120, -210, 330],
            },
            &Motifs {
                tiles: &[],
                gauges: &[],
                pairs: &[],
            },
            Weights::FULL,
        )
        .unwrap();
        let states = [vec![1, 1, -1], vec![-1, 1, 1], vec![1, -1, 1]];
        let mut snapshot = PopulationSnapshot::new(3, 3);
        for picks in [[1, 0, 2], [1, 1, 1], [2, 0, 1], [0, 1, 2]] {
            let mut chains: Vec<_> = states
                .iter()
                .map(|s| Chain::new(&graph, s.clone()).unwrap())
                .collect();
            snapshot.capture(&chains);
            for (chain, &parent) in chains.iter_mut().zip(&picks) {
                snapshot.restore(chain, parent);
            }
            for (child, chain) in chains.iter_mut().enumerate() {
                assert_eq!(chain.spins(), states[picks[child]]);
                let before = graph.energy(chain.spins());
                let delta = chain.single_sweep(0.9, &mut SmallRng::seed_from_u64(child as u64));
                assert_eq!(graph.energy(chain.spins()) - before, delta);
            }
        }
    }
}
