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
mod update;

use quip_solver_core::CancelToken;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::fmt;

pub use graph::{
    GraphError, Model, Motifs, OtsukaGraph, Weights, MAX_NODES, PAIR_MILLI, STRONG_MILLI,
};
pub use update::{Chain, SpinError};

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
    let every = (schedule.betas.len() / RESAMPLES).max(1);
    let mut order: Vec<usize> = (0..replicas).collect();
    let mut scratch = Houdayer::new(graph.n);
    let mut last_beta = schedule.betas[0];
    for (sweep, &beta) in schedule.betas.iter().enumerate() {
        if sweep > 0 && sweep % every == 0 && replicas > 1 {
            let picks = resample(&energy, beta - last_beta, &mut rng);
            let states: Vec<Vec<i8>> = picks.iter().map(|&i| chains[i].spins().to_vec()).collect();
            for (chain, state) in chains.iter_mut().zip(&states) {
                chain.reset(state).expect("a state of the same graph");
            }
            energy = picks.iter().map(|&i| energy[i]).collect();
            best = picks.iter().map(|&i| best[i].clone()).collect();
            last_beta = beta;
        }
        for ((chain, e), lowest) in chains.iter_mut().zip(&mut energy).zip(&mut best) {
            if cancel.is_some_and(|(token, watermark)| token.is_cancelled(watermark)) {
                return Err(AnnealError::Cancelled);
            }
            *e += chain.cluster_sweep(beta, &mut rng);
            if schedule.single_every > 0 && (sweep + 1).is_multiple_of(schedule.single_every) {
                *e += chain.single_sweep(beta, &mut rng);
            }
            if *e < lowest.1 {
                lowest.0.copy_from_slice(chain.spins());
                lowest.1 = *e;
            }
        }
        if replicas > 1 && beta >= HOUDAYER_BETA {
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
                            best[i].0.copy_from_slice(chains[i].spins());
                            best[i].1 = energy[i];
                        }
                    }
                }
            }
        }
    }
    Ok(best
        .into_iter()
        .map(|(spins, energy_milli)| Sample {
            spins,
            energy_milli,
            sweeps: schedule.betas.len(),
        })
        .collect())
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
