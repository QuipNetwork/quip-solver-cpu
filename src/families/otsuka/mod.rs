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
use rand::SeedableRng;
use std::fmt;

pub use graph::{GraphError, Model, Motifs, OtsukaGraph, Weights, PAIR_MILLI, STRONG_MILLI};
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
    let mut weak = graph.weak_energy(chain.spins());
    let mut best = (chain.spins().to_vec(), energy);
    for (sweep, &beta) in schedule.betas.iter().enumerate() {
        if cancel.is_some_and(|(token, watermark)| token.is_cancelled(watermark)) {
            return Err(AnnealError::Cancelled);
        }
        weak += chain.cluster_sweep(beta, &mut rng);
        energy = weak + graph.strong_energy(chain.spins());
        if schedule.single_every > 0 && (sweep + 1).is_multiple_of(schedule.single_every) {
            energy += chain.single_sweep(beta, &mut rng);
            weak = energy - graph.strong_energy(chain.spins());
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
