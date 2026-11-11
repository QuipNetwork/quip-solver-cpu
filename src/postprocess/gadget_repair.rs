//! The routing gadget-repair postprocess policy `gadget-repair/1`.
//!
//! Port of the research constraints repair for the two-commodity routing
//! cell: the gadget extraction of `gadgets.py` for `two-commodity-routing`,
//! `repair_init` and `repair_run` of `repair.c` (unit moves, ejection chains
//! over the gadget neighbourhood, and tabu search, all with incremental local
//! fields), and the schedule of `one_model` in `run.py` with its default
//! options. The routing cell has no auxiliary spins, so the closed-form
//! auxiliary minimization of the source is not ported. Energies are integer
//! milli.
//!
//! A routing instance gives the chain and commodity mapping. Arc `a` owns the
//! commodity 1 spin `x1` and the two commodity 2 chain spins. A unit is one
//! spin, one chain, or a pair of units that share a gadget. The extraction
//! checks the mapping and the chain couplers before any state is touched.
//!
//! The source reads the thread CPU clock at fixed places, to stop each repair.
//! [`Work::Budget`] reads the elapsed time on the job thread against the
//! declared 0.05 s. [`Work::Fixed`] replaces the clock by counts of clock
//! reads, so the result depends only on the inputs and the seed. A repair
//! never proves feasibility: [`check_routing`] counts the conservation and
//! capacity defects of a state directly.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::time::{Duration, Instant};

use quip_solver_core::ising_generators::descriptor::Params;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::ising_generators::routing::{
    routing_defects, RoutingDefects, ROUTING_SCHEMA,
};
use quip_solver_core::CancelToken;

use super::greedy::{DescentError, Greedy, Ising, MoveError};
use crate::families::otsuka_sampler::section;

/// The postprocess policy: natural-move descent, then the gadget repair.
/// Versioned apart from any family revision.
pub const GADGET_POLICY: &str = "gadget-repair/1";

/// The declared budget of one job, from the `two-commodity-routing` entry of
/// `targets-pp.json`.
pub const GADGET_BUDGET_SECONDS: f64 = 0.05;

/// The reads the repair starts from, besides the majority vote.
const READS_USED: usize = 4;
/// The most reads that vote.
const VOTERS: usize = 8;
/// The xorshift64 state for a zero seed.
const ZERO_SEED: u64 = 0x9e37_79b9_7f4a_7c15;
/// The metadata section of the routing instance.
const ROUTING_SECTION: u16 = 7;
/// Chains kept apart from the pair units, in the source's gadget size cap.
const PAIR_CAP: usize = 40_000;

/// Why a routing instance or a model cannot run the repair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GadgetError {
    /// The instance has no routing section.
    NoSection,
    /// The routing section does not follow its layout.
    Layout(&'static str),
    /// The model has no arcs.
    NoArcs,
    /// An arc names a node that the section does not list.
    ArcNode {
        /// The arc position.
        arc: usize,
        /// The node it names.
        node: u32,
    },
    /// An arc starts and ends at one node.
    SelfLoop {
        /// The arc position.
        arc: usize,
    },
    /// An arc names a spin outside the model.
    SpinOutside {
        /// The arc position.
        arc: usize,
        /// The spin it names.
        spin: u32,
    },
    /// The two chain spins of an arc are one spin.
    ChainRepeat {
        /// The arc position.
        arc: usize,
    },
    /// A spin serves two roles: two arcs, or a chain and a commodity 1 spin.
    SpinReused {
        /// The arc position of the second use.
        arc: usize,
        /// The spin.
        spin: u32,
    },
    /// The arcs do not map every spin of the model once.
    SpinCount {
        /// The spins the arcs map.
        mapped: usize,
        /// The spins in the model.
        spins: usize,
    },
    /// The two chain spins of an arc share no ferromagnetic coupler.
    ChainCoupler {
        /// The arc position.
        arc: usize,
    },
    /// The commodity 1 spin of an arc and its first chain spin share no
    /// coupler, so the capacity term has no place.
    CapacityCoupler {
        /// The arc position.
        arc: usize,
    },
    /// The instance fails the family validation.
    Invalid,
    /// The instance does not fit the job graph.
    Fit,
    /// A move list or a model is not valid for the descent.
    Moves(MoveError),
}

impl fmt::Display for GadgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSection => write!(f, "the instance has no routing section"),
            Self::Layout(why) => write!(f, "the routing section is malformed: {why}"),
            Self::NoArcs => write!(f, "the routing model has no arcs"),
            Self::ArcNode { arc, node } => {
                write!(
                    f,
                    "arc {arc} names node {node}, which the model does not list"
                )
            }
            Self::SelfLoop { arc } => write!(f, "arc {arc} starts and ends at one node"),
            Self::SpinOutside { arc, spin } => {
                write!(f, "arc {arc} names spin {spin} outside the model")
            }
            Self::ChainRepeat { arc } => {
                write!(f, "the two chain spins of arc {arc} are one spin")
            }
            Self::SpinReused { arc, spin } => {
                write!(
                    f,
                    "arc {arc} maps spin {spin}, which another role already holds"
                )
            }
            Self::SpinCount { mapped, spins } => {
                write!(f, "the arcs map {mapped} spins of a model with {spins}")
            }
            Self::ChainCoupler { arc } => {
                write!(f, "the chain of arc {arc} has no ferromagnetic coupler")
            }
            Self::CapacityCoupler { arc } => {
                write!(f, "arc {arc} has no coupler between its commodity spins")
            }
            Self::Invalid => write!(f, "the instance fails the routing validation"),
            Self::Fit => write!(f, "the routing instance does not fit the job graph"),
            Self::Moves(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for GadgetError {}

/// One arc of the routing model: its end nodes, its commodity 1 spin, and its
/// commodity 2 chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArcMap {
    /// The tail node.
    pub tail: u32,
    /// The head node.
    pub head: u32,
    /// The commodity 1 spin.
    pub x1: u32,
    /// The two commodity 2 chain spins.
    pub chain: [u32; 2],
}

/// The arcs of a routing instance, from its metadata section.
///
/// The layout is a `u32` node count, six bytes per node, a `u32` arc count,
/// 24 bytes per arc (five `u32` and two `u16`), and a `u16` chain strength.
/// This checks the layout and the node bounds only. [`check_routing`]
/// validates the instance in full.
///
/// # Errors
///
/// [`GadgetError::NoSection`] or [`GadgetError::Layout`] for a missing or
/// short section, and [`GadgetError::ArcNode`] for an arc that names an
/// unlisted node.
pub fn arcs_of(instance: &Instance) -> Result<Vec<ArcMap>, GadgetError> {
    let payload = section(instance.metadata(), ROUTING_SECTION).ok_or(GadgetError::NoSection)?;
    let word = |at: usize| -> Result<u32, GadgetError> {
        payload
            .get(at..at + 4)
            .and_then(|b| b.try_into().ok())
            .map(u32::from_le_bytes)
            .ok_or(GadgetError::Layout("the section ends early"))
    };
    let to_len = |v: u32| {
        usize::try_from(v).map_err(|_| GadgetError::Layout("a count exceeds the address space"))
    };
    let nodes = to_len(word(0)?)?;
    let arcs_at = nodes
        .checked_mul(6)
        .and_then(|n| n.checked_add(4))
        .ok_or(GadgetError::Layout("the node count is too large"))?;
    let count = to_len(word(arcs_at)?)?;
    let first = arcs_at + 4;
    let end = count
        .checked_mul(24)
        .and_then(|n| n.checked_add(first))
        .and_then(|n| n.checked_add(2))
        .ok_or(GadgetError::Layout("the arc count is too large"))?;
    if payload.len() != end {
        return Err(GadgetError::Layout("the length does not match the counts"));
    }
    let mut arcs = Vec::with_capacity(count);
    for arc in 0..count {
        let at = first + 24 * arc;
        let [tail, head, x1, c0, c1] = [0, 4, 8, 12, 16].map(|k| word(at + k));
        let (tail, head, x1, c0, c1) = (tail?, head?, x1?, c0?, c1?);
        for node in [tail, head] {
            if to_len(node)? >= nodes {
                return Err(GadgetError::ArcNode { arc, node });
            }
        }
        arcs.push(ArcMap {
            tail,
            head,
            x1,
            chain: [c0, c1],
        });
    }
    Ok(arcs)
}

/// The gadget structure of a routing model: the units the repair flips, in
/// the source order, and the gadget neighbourhood of every unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Structure {
    /// The units: singles and chains first (the chain starts), then the
    /// pairs. Each unit is a sorted spin list.
    pub units: Vec<Vec<u32>>,
    /// The gadgets: spin sets whose energy a move can violate.
    pub gadgets: Vec<Vec<u32>>,
    /// The units before the first pair unit.
    pub n_starts: usize,
    /// CSR offsets of the gadget neighbours of each unit.
    pub gm_ptr: Vec<usize>,
    /// CSR members: the units that share a gadget with a unit.
    pub gm_idx: Vec<u32>,
}

impl Structure {
    /// The structure of `arcs` over `n` spins, as `gadgets.extract` reads the
    /// routing cell.
    ///
    /// The commodity 1 spins and the commodity 2 chain spins of every arc
    /// that ends at a node make one gadget per node and commodity. Each arc
    /// makes a gadget of its three spins, and each chain a unit.
    ///
    /// # Errors
    ///
    /// [`GadgetError`] for no arcs, a self-loop, a spin outside the model, a
    /// chain of one spin, or a spin that two roles share. A spin that no arc
    /// maps stays a single-spin unit: [`check_spin_cover`] rejects it for an
    /// instance.
    pub fn extract(n: usize, arcs: &[ArcMap]) -> Result<Self, GadgetError> {
        if arcs.is_empty() {
            return Err(GadgetError::NoArcs);
        }
        let nodes = arcs
            .iter()
            .flat_map(|a| [a.tail, a.head])
            .max()
            .map_or(0, |m| m as usize + 1);
        let mut role = vec![false; n];
        for (arc, a) in arcs.iter().enumerate() {
            if a.tail == a.head {
                return Err(GadgetError::SelfLoop { arc });
            }
            if a.chain[0] == a.chain[1] {
                return Err(GadgetError::ChainRepeat { arc });
            }
            for spin in [a.x1, a.chain[0], a.chain[1]] {
                let slot = role
                    .get_mut(spin as usize)
                    .ok_or(GadgetError::SpinOutside { arc, spin })?;
                if *slot {
                    return Err(GadgetError::SpinReused { arc, spin });
                }
                *slot = true;
            }
        }

        let chains: BTreeSet<Vec<u32>> = arcs
            .iter()
            .map(|a| sorted_unique(a.chain.iter().copied()))
            .collect();
        let mut per_node: Vec<[Vec<u32>; 2]> = vec![[Vec::new(), Vec::new()]; nodes];
        for a in arcs {
            for end in [a.tail, a.head] {
                let slot = &mut per_node[end as usize];
                slot[0].push(a.x1);
                slot[1].extend(a.chain);
            }
        }
        let mut gadgets: Vec<Vec<u32>> = Vec::new();
        for node in &per_node {
            for commodity in node {
                if commodity.len() >= 2 {
                    gadgets.push(sorted_unique(commodity.iter().copied()));
                }
            }
        }
        for a in arcs {
            gadgets.push(sorted_unique([a.x1, a.chain[0], a.chain[1]]));
        }
        gadgets.retain(|g| g.len() >= 2);

        // The atoms of a gadget are its spins that sit in no chain, and the
        // chains that touch it.
        let mut spin_chain: BTreeMap<u32, &Vec<u32>> = BTreeMap::new();
        for chain in &chains {
            for &x in chain {
                spin_chain.insert(x, chain);
            }
        }
        let mut units: BTreeSet<Vec<u32>> = (0..n)
            .map(|i| vec![u32::try_from(i).unwrap_or(u32::MAX)])
            .collect();
        units.extend(chains.iter().cloned());
        let want_pairs: usize = gadgets.iter().map(|g| g.len() * (g.len() - 1) / 2).sum();
        let want_pairs = want_pairs <= PAIR_CAP;
        let mut pair_units: BTreeSet<Vec<u32>> = BTreeSet::new();
        let mut gadget_units: Vec<Vec<Vec<u32>>> = Vec::with_capacity(gadgets.len());
        for g in &gadgets {
            let mut atoms: Vec<Vec<u32>> = Vec::new();
            let mut seen: BTreeSet<Vec<u32>> = BTreeSet::new();
            for &x in g {
                let unit = spin_chain.get(&x).map_or_else(|| vec![x], |c| (*c).clone());
                if seen.insert(unit.clone()) {
                    atoms.push(unit);
                }
            }
            let mut members = atoms.clone();
            if want_pairs {
                for (i, a) in atoms.iter().enumerate() {
                    for b in &atoms[i + 1..] {
                        let pair = sorted_unique(a.iter().chain(b).copied());
                        pair_units.insert(pair.clone());
                        members.push(pair);
                    }
                }
            }
            gadget_units.push(members);
        }
        // Singles and chains first, ordered by size then spins, then the pairs.
        let order = |set: BTreeSet<Vec<u32>>| -> Vec<Vec<u32>> {
            let mut list: Vec<Vec<u32>> = set.into_iter().collect();
            list.sort_by(|a, b| (a.len(), a).cmp(&(b.len(), b)));
            list
        };
        let extra: BTreeSet<Vec<u32>> = pair_units.difference(&units).cloned().collect();
        let mut ordered = order(units);
        let n_starts = ordered.len();
        ordered.extend(order(extra));

        let index: BTreeMap<&Vec<u32>, u32> = ordered
            .iter()
            .enumerate()
            .map(|(k, u)| (u, u32::try_from(k).unwrap_or(u32::MAX)))
            .collect();
        let mut links: BTreeSet<(u32, u32)> = BTreeSet::new();
        for members in &gadget_units {
            let ids: BTreeSet<u32> = members
                .iter()
                .filter_map(|u| index.get(u).copied())
                .collect();
            if ids.len() < 2 {
                continue;
            }
            for &a in &ids {
                for &b in &ids {
                    if a != b {
                        links.insert((a, b));
                    }
                }
            }
        }
        let mut gm_ptr = vec![0_usize; ordered.len() + 1];
        for &(a, _) in &links {
            gm_ptr[a as usize + 1] += 1;
        }
        for k in 0..ordered.len() {
            gm_ptr[k + 1] += gm_ptr[k];
        }
        let gm_idx = links.iter().map(|&(_, b)| b).collect();
        Ok(Self {
            units: ordered,
            gadgets,
            n_starts,
            gm_ptr,
            gm_idx,
        })
    }
}

/// Check that the arcs map every spin of an `n`-spin model once.
///
/// # Errors
///
/// [`GadgetError::SpinCount`] when three spins per arc do not make `n`. The
/// roles of the spins are checked by [`Structure::extract`].
pub fn check_spin_cover(n: usize, arcs: &[ArcMap]) -> Result<(), GadgetError> {
    let mapped = 3 * arcs.len();
    if mapped == n {
        Ok(())
    } else {
        Err(GadgetError::SpinCount { mapped, spins: n })
    }
}

fn sorted_unique(items: impl IntoIterator<Item = u32>) -> Vec<u32> {
    let set: BTreeSet<u32> = items.into_iter().collect();
    set.into_iter().collect()
}

/// The direct constraint check of a state: arcs where both commodities run,
/// nodes and commodities whose balance differs from the supply, the arcs each
/// commodity uses, and broken chains. This validates the instance and the
/// state, and never reads an energy.
///
/// # Errors
///
/// [`GadgetError::Invalid`] when `params` or `instance` fail the routing
/// validation, or `spins` does not hold one `1` or `-1` per node.
pub fn check_routing(
    params: &[u8],
    instance: &Instance,
    spins: &[i8],
) -> Result<RoutingDefects, GadgetError> {
    let params = Params::parse(ROUTING_SCHEMA, params).map_err(|_| GadgetError::Invalid)?;
    routing_defects(&params, instance, spins).ok_or(GadgetError::Invalid)
}

/// Whether `defects` show a feasible routing: no capacity conflict, no
/// conservation violation, and no broken chain.
#[must_use]
pub fn is_feasible(defects: &RoutingDefects) -> bool {
    defects.capacity_conflicts == 0
        && defects.conservation_violations == 0
        && defects.broken_chains == 0
}

/// How much work the schedule does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Work {
    /// Run against [`GADGET_BUDGET_SECONDS`] of elapsed time on the calling
    /// thread, as the research ran against process CPU time on one core. The
    /// clock decides the read count and where each repair stops, so the
    /// result depends on the host and its load.
    Budget,
    /// Descend `descended` reads (always at least one), and give the repair
    /// of start `k` `ticks[k]` clock reads that report time left. The result
    /// depends only on the inputs and the seed.
    Fixed {
        /// The reads that get the natural-move descent.
        descended: usize,
        /// The clock reads each start survives. A start with no entry is
        /// skipped, as the source skips a start with no time left.
        ticks: Vec<u64>,
    },
}

/// The statistics of one repair, as `repair_run` returns them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RepairStats {
    /// Tabu iterations.
    pub iterations: i64,
    /// Ejection chains kept.
    pub chains_kept: i64,
    /// Chain sweeps.
    pub sweeps: i64,
    /// Tabu steps.
    pub steps: i64,
    /// New best states in the tabu search.
    pub new_bests: i64,
}

/// One repaired start, for the job log.
#[derive(Clone, Debug, PartialEq)]
pub struct StartReport {
    /// The input position of the read, or `None` for the majority vote.
    pub read: Option<usize>,
    /// The energy of the start, rescored from the model.
    pub start_energy: i64,
    /// The energy after the repair, rescored from the model.
    pub energy: i64,
    /// What the repair did.
    pub stats: RepairStats,
    /// The clock reads the repair made. A [`Work::Fixed`] run replays them.
    pub ticks: u64,
    /// The measured time of the repair, in seconds.
    pub seconds: f64,
}

/// What a schedule did, for the job log.
#[derive(Clone, Debug, PartialEq)]
pub struct GadgetReport {
    /// [`GADGET_POLICY`].
    pub policy: &'static str,
    /// The declared budget, [`GADGET_BUDGET_SECONDS`], whatever the work mode.
    pub budget_seconds: f64,
    /// The work mode of this run.
    pub work: Work,
    /// The reads that got the natural-move descent.
    pub descended: usize,
    /// The best energy after the descent.
    pub descent_best: i64,
    /// Each repaired start, in order.
    pub starts: Vec<StartReport>,
    /// The final energy, rescored from the model.
    pub energy: i64,
    /// The measured time of the whole schedule, in seconds.
    pub elapsed_seconds: f64,
}

/// The result of a schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct GadgetOutcome {
    /// The final state: the lowest energy over the descended reads and the
    /// repaired starts.
    pub state: Vec<i8>,
    /// The repaired state of each start, in order.
    pub pool: Vec<Vec<i8>>,
    /// What the schedule did.
    pub report: GadgetReport,
}

/// The clock of a repair: the elapsed time against a deadline, or a count of
/// clock reads. Each cancel check rides on a clock read.
struct Clock<'a> {
    deadline: Option<Instant>,
    limit: u64,
    reads: u64,
    cancel: Option<(&'a CancelToken, Option<u64>)>,
    cancelled: bool,
}

impl<'a> Clock<'a> {
    fn deadline(seconds: f64, cancel: Option<(&'a CancelToken, Option<u64>)>) -> Self {
        let span = Duration::try_from_secs_f64(seconds.max(0.0)).unwrap_or(Duration::ZERO);
        Self {
            deadline: Some(Instant::now() + span),
            limit: 0,
            reads: 0,
            cancel,
            cancelled: false,
        }
    }

    fn ticks(limit: u64, cancel: Option<(&'a CancelToken, Option<u64>)>) -> Self {
        Self {
            deadline: None,
            limit,
            reads: 0,
            cancel,
            cancelled: false,
        }
    }

    /// One clock read: whether time is left.
    fn read(&mut self) -> bool {
        self.reads += 1;
        if self
            .cancel
            .is_some_and(|(token, mark)| token.is_cancelled(mark))
        {
            self.cancelled = true;
            return false;
        }
        match self.deadline {
            Some(deadline) => Instant::now() <= deadline,
            None => self.reads <= self.limit,
        }
    }

    /// `cpu_now() > deadline` of the source.
    fn expired(&mut self) -> bool {
        !self.read()
    }

    /// `cpu_now() < deadline` of the source.
    fn live(&mut self) -> bool {
        self.read()
    }
}

fn xs64(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

/// The tables of the repair: the model, the unit moves with their internal
/// edges, the spin to move map, and the gadget neighbours.
#[derive(Debug)]
pub struct Repair {
    greedy: Greedy,
    n_moves: usize,
    sm_ptr: Vec<usize>,
    sm_idx: Vec<u32>,
    gm_ptr: Vec<usize>,
    gm_idx: Vec<u32>,
    n_starts: usize,
}

/// The options of one repair. The default is the selected policy: the
/// defaults of `run.py`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepairOptions {
    /// The ejection chain depth. Chains run when it is at least 2.
    pub depth: i32,
    /// The base tabu tenure.
    pub tenure: i64,
    /// The random extra tenure is drawn from `0..=tenure_rand`.
    pub tenure_rand: u64,
    /// The tabu steps without a new best before the walk returns to the best
    /// state. Zero never returns.
    pub stall: i64,
    /// Which units start an ejection chain.
    pub starts: ChainStarts,
}

impl Default for RepairOptions {
    fn default() -> Self {
        Self {
            depth: 8,
            tenure: 12,
            tenure_rand: 6,
            stall: 2000,
            starts: ChainStarts::All,
        }
    }
}

/// Which units start an ejection chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainStarts {
    /// Every unit, pairs included: the `all` of `run.py`, the selected one.
    All,
    /// The singles and the chains only.
    Singles,
}

impl Repair {
    /// The tables for `ising` and `structure`.
    ///
    /// # Errors
    ///
    /// [`GadgetError::Moves`] when a unit names a spin outside the model.
    pub fn new(ising: Ising, structure: &Structure) -> Result<Self, GadgetError> {
        let n = ising.num_nodes();
        let greedy = Greedy::new(ising, &structure.units, false).map_err(GadgetError::Moves)?;
        let n_moves = greedy.num_moves();
        let mut sm: Vec<Vec<u32>> = vec![Vec::new(); n];
        for m in 0..n_moves {
            for &i in greedy.members(m) {
                sm[i as usize]
                    .push(u32::try_from(m).map_err(|_| GadgetError::Layout("too many moves"))?);
            }
        }
        let mut sm_ptr = vec![0];
        let mut sm_idx = Vec::new();
        for list in sm {
            sm_idx.extend(list);
            sm_ptr.push(sm_idx.len());
        }
        Ok(Self {
            greedy,
            n_moves,
            sm_ptr,
            sm_idx,
            gm_ptr: structure.gm_ptr.clone(),
            gm_idx: structure.gm_idx.clone(),
            n_starts: structure.n_starts,
        })
    }

    /// Check that every chain is a ferromagnetic coupler and that every arc
    /// couples its commodity 1 spin to its first chain spin.
    ///
    /// # Errors
    ///
    /// [`GadgetError::ChainCoupler`] or [`GadgetError::CapacityCoupler`].
    pub fn check_couplers(ising: &Ising, arcs: &[ArcMap]) -> Result<(), GadgetError> {
        for (arc, a) in arcs.iter().enumerate() {
            match ising.coupling(a.chain[0], a.chain[1]) {
                Some(j) if j < 0 => {}
                _ => return Err(GadgetError::ChainCoupler { arc }),
            }
            if ising.coupling(a.x1, a.chain[0]).is_none() {
                return Err(GadgetError::CapacityCoupler { arc });
            }
        }
        Ok(())
    }

    /// The model.
    #[must_use]
    pub fn ising(&self) -> &Ising {
        self.greedy.ising()
    }

    /// The number of unit moves.
    #[must_use]
    pub fn num_moves(&self) -> usize {
        self.n_moves
    }

    fn starts_for(&self, mode: ChainStarts) -> usize {
        match mode {
            ChainStarts::All => self.n_moves,
            ChainStarts::Singles => self.n_starts,
        }
    }
}

/// The working state of one repair, `repair_run` of the source.
struct Run<'r, 'c> {
    repair: &'r Repair,
    ising: &'r Ising,
    s: Vec<i8>,
    f: Vec<i64>,
    eff: Vec<i64>,
    stamp: Vec<i32>,
    dirty: Vec<u32>,
    path: Vec<u32>,
    order: Vec<u32>,
    inpath: Vec<i32>,
    tabu_until: Vec<i64>,
    gen: i32,
    chain_id: i32,
    n_starts: usize,
    depth: i32,
    rng: u64,
    clock: Clock<'c>,
    chains_kept: i64,
    sweeps: i64,
}

impl Run<'_, '_> {
    fn flip_spin(&mut self, u: usize) -> i64 {
        let old = i64::from(self.s[u]);
        let d = -2 * old * self.f[u];
        self.s[u] = -self.s[u];
        for &(v, j) in self.ising.row(u) {
            self.f[v as usize] -= 2 * j * old;
        }
        d
    }

    /// The effective energy change of move `m` at the current state.
    fn eff_delta(&self, m: usize) -> i64 {
        let mut d = 0;
        for &i in self.repair.greedy.members(m) {
            d += -2 * i64::from(self.s[i as usize]) * self.f[i as usize];
        }
        for &(a, b, j) in self.repair.greedy.inner(m) {
            d += 4 * j * i64::from(self.s[a as usize]) * i64::from(self.s[b as usize]);
        }
        d
    }

    /// Flip the members of `m`, push each on the undo path, and return the
    /// exact energy change.
    fn apply_move(&mut self, m: usize) -> i64 {
        let repair = self.repair;
        let mut d = 0;
        for &i in repair.greedy.members(m) {
            d += self.flip_spin(i as usize);
            self.path.push(i);
        }
        d
    }

    fn undo_all(&mut self) {
        while let Some(i) = self.path.pop() {
            self.flip_spin(i as usize);
        }
    }

    fn mark_spin(&mut self, x: usize) {
        let r = &self.repair;
        for &m in &r.sm_idx[r.sm_ptr[x]..r.sm_ptr[x + 1]] {
            if self.stamp[m as usize] != self.gen {
                self.stamp[m as usize] = self.gen;
                self.dirty.push(m);
            }
        }
    }

    /// Mark every move whose effective change may have moved after the spins
    /// of the undo path flipped, and refresh them.
    fn refresh_path(&mut self) {
        self.gen += 1;
        self.dirty.clear();
        let ising = self.ising;
        for t in 0..self.path.len() {
            let x = self.path[t] as usize;
            self.mark_spin(x);
            for &(y, _) in ising.row(x) {
                self.mark_spin(y as usize);
            }
        }
        for t in 0..self.dirty.len() {
            let m = self.dirty[t] as usize;
            self.eff[m] = self.eff_delta(m);
        }
    }

    fn is_tabu(&self, m: usize, it: i64) -> bool {
        self.repair
            .greedy
            .members(m)
            .iter()
            .any(|&i| self.tabu_until[i as usize] > it)
    }

    fn reload_fields(&mut self) {
        for i in 0..self.s.len() {
            self.f[i] = self.ising.field(&self.s, i);
        }
        for m in 0..self.repair.n_moves {
            self.eff[m] = self.eff_delta(m);
        }
    }

    /// One sweep of plain descent plus ejection chains over all moves in
    /// random order. Returns the energy change and whether a move or chain
    /// was kept.
    fn chain_sweep(&mut self) -> (i64, bool) {
        let n_moves = self.repair.n_moves;
        let mut total = 0;
        let mut improved = false;
        self.order.clear();
        self.order.extend((0..n_moves).map(|m| m as u32));
        for i in (1..n_moves).rev() {
            let k = (xs64(&mut self.rng) % (i as u64 + 1)) as usize;
            self.order.swap(i, k);
        }
        for oi in 0..n_moves {
            if (oi & 255) == 0 && self.clock.expired() {
                break;
            }
            let m0 = self.order[oi] as usize;
            self.path.clear();
            if self.eff[m0] < 0 {
                total += self.apply_move(m0);
                self.refresh_path();
                improved = true;
                continue;
            }
            if self.depth < 2 || m0 >= self.n_starts {
                continue;
            }
            self.chain_id += 1;
            let mut cum = 0;
            let mut cur = m0;
            let mut kept = false;
            for step in 0..self.depth {
                cum += self.apply_move(cur);
                self.inpath[cur] = self.chain_id;
                if cum < 0 {
                    kept = true;
                    break;
                }
                let mut next: Option<usize> = None;
                let mut dnext = 0;
                let mut ties = 0_u64;
                let (lo, hi) = (self.repair.gm_ptr[cur], self.repair.gm_ptr[cur + 1]);
                for q in lo..hi {
                    let v = self.repair.gm_idx[q] as usize;
                    if self.inpath[v] == self.chain_id {
                        continue;
                    }
                    let d = self.eff_delta(v);
                    if next.is_none() || d < dnext {
                        next = Some(v);
                        dnext = d;
                        ties = 1;
                    } else if d == dnext {
                        ties += 1;
                        if xs64(&mut self.rng).is_multiple_of(ties) {
                            next = Some(v);
                        }
                    }
                }
                match next {
                    Some(v) if !(step + 2 >= self.depth && cum + dnext >= 0) => cur = v,
                    _ => break,
                }
            }
            if kept {
                total += cum;
                self.chains_kept += 1;
                self.refresh_path();
                improved = true;
            } else {
                self.undo_all();
            }
        }
        (total, improved)
    }
}

impl Repair {
    /// Run the repair on one state in place: unit moves, ejection chains over
    /// the gadget neighbourhood, and a tabu search. Returns the energy change
    /// of the best state found against the input state, and the statistics.
    /// The clock decides when to stop. The state holds the best state found.
    ///
    /// # Errors
    ///
    /// [`DescentError::Length`] or [`DescentError::Value`] when `s` does not
    /// fit the model, and [`DescentError::Cancelled`] when the token abandons
    /// the watermark.
    pub fn run(
        &self,
        s: &mut [i8],
        options: &RepairOptions,
        seed: u64,
        clock: ClockSpec<'_>,
    ) -> Result<(i64, RepairStats, u64), DescentError> {
        self.greedy.check(s)?;
        let n = s.len();
        let n_moves = self.n_moves;
        let clock = clock.build();
        let mut r = Run {
            repair: self,
            ising: self.greedy.ising(),
            s: s.to_vec(),
            f: vec![0; n],
            eff: vec![0; n_moves],
            stamp: vec![0; n_moves],
            dirty: Vec::new(),
            path: Vec::new(),
            order: Vec::with_capacity(n_moves),
            inpath: vec![0; n_moves],
            tabu_until: vec![0; n],
            gen: 0,
            chain_id: 0,
            n_starts: self.starts_for(options.starts),
            depth: options.depth,
            rng: if seed == 0 { ZERO_SEED } else { seed },
            clock,
            chains_kept: 0,
            sweeps: 0,
        };
        r.reload_fields();
        let mut e: i64 = 0;
        let mut best_e: i64 = 0;
        let mut best = r.s.clone();
        let (mut steps, mut new_bests) = (0_i64, 0_i64);
        // Phase 1: zero-temperature descent with chains.
        loop {
            let (d, improved) = r.chain_sweep();
            e += d;
            r.sweeps += 1;
            if e < best_e {
                best_e = e;
                best.clone_from(&r.s);
            }
            if !improved || r.clock.expired() {
                break;
            }
        }
        // Phase 2: tabu search over the effective moves.
        let (tenure, tenure_rand) = (options.tenure, options.tenure_rand);
        let mut it: i64 = 0;
        let mut since_best: i64 = 0;
        while r.clock.live() {
            for _ in 0..16 {
                let mut pick: Option<usize> = None;
                let mut dpick = 0;
                let mut ties = 0_u64;
                for m in 0..n_moves {
                    let d = r.eff[m];
                    if pick.is_some() && d > dpick {
                        continue;
                    }
                    if e + d >= best_e && r.is_tabu(m, it) {
                        continue;
                    }
                    if pick.is_none() || d < dpick {
                        pick = Some(m);
                        dpick = d;
                        ties = 1;
                    } else {
                        ties += 1;
                        if xs64(&mut r.rng).is_multiple_of(ties) {
                            pick = Some(m);
                        }
                    }
                }
                let Some(pick) = pick else { break };
                r.path.clear();
                e += r.apply_move(pick);
                it += 1;
                steps += 1;
                for &i in self.greedy.members(pick) {
                    let i = i as usize;
                    r.tabu_until[i] = it + tenure + (xs64(&mut r.rng) % (tenure_rand + 1)) as i64;
                }
                r.refresh_path();
                if e < best_e {
                    best_e = e;
                    best.clone_from(&r.s);
                    new_bests += 1;
                    since_best = 0;
                    // Polish the new best with chain sweeps.
                    if r.depth >= 2 {
                        let mut improved = true;
                        while improved && r.clock.live() {
                            let (d, did) = r.chain_sweep();
                            e += d;
                            improved = did;
                            r.sweeps += 1;
                        }
                        if e < best_e {
                            best_e = e;
                            best.clone_from(&r.s);
                        }
                    }
                } else {
                    since_best += 1;
                    if options.stall > 0 && since_best >= options.stall {
                        // Return to the best state with fresh tabu tenures.
                        r.s.clone_from(&best);
                        r.reload_fields();
                        e = best_e;
                        since_best = 0;
                        for _ in 0..tenure {
                            let x = (xs64(&mut r.rng) % n as u64) as usize;
                            r.tabu_until[x] = it + tenure;
                        }
                    }
                }
            }
        }
        if r.clock.cancelled {
            return Err(DescentError::Cancelled);
        }
        s.copy_from_slice(&best);
        let stats = RepairStats {
            iterations: it,
            chains_kept: r.chains_kept,
            sweeps: r.sweeps,
            steps,
            new_bests,
        };
        Ok((best_e, stats, r.clock.reads))
    }
}

/// How a repair stops.
#[derive(Clone, Copy)]
pub enum ClockSpec<'a> {
    /// Stop after this many seconds of elapsed time.
    Seconds(f64, Option<(&'a CancelToken, Option<u64>)>),
    /// Stop at the clock read after this many reads.
    Ticks(u64, Option<(&'a CancelToken, Option<u64>)>),
}

impl<'a> ClockSpec<'a> {
    fn build(self) -> Clock<'a> {
        match self {
            Self::Seconds(seconds, cancel) => Clock::deadline(seconds, cancel),
            Self::Ticks(limit, cancel) => Clock::ticks(limit, cancel),
        }
    }
}

/// A start state of the repair: a descended read, or the vote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Start {
    /// The input position of the read, or `None` for the majority vote.
    pub read: Option<usize>,
    /// The descended state.
    pub state: Vec<i8>,
    /// Its energy, rescored from the model.
    pub energy: i64,
}

/// The reads after the natural-move descent, and the starts chosen from them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prepared {
    /// The reads that got the descent.
    pub descended: usize,
    /// The lowest-energy read after the descent (the first on a tie).
    pub best: Start,
    /// The starts in repair order: the best descended reads, with the vote
    /// second.
    pub starts: Vec<Start>,
}

/// The first part of the schedule: natural-move descent on the reads in raw
/// energy order, then the start list. `started` is the start of the
/// schedule, for the half-budget stop of [`Work::Budget`].
///
/// # Errors
///
/// [`DescentError::Empty`] for no reads, [`DescentError::Length`] or
/// [`DescentError::Value`] for a read that does not fit the model, and
/// [`DescentError::Cancelled`] when the token abandons the watermark.
pub fn prepare_starts(
    nat: &Greedy,
    reads: &[Vec<i8>],
    work: &Work,
    started: Instant,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Prepared, DescentError> {
    let ising = nat.ising();
    if reads.is_empty() {
        return Err(DescentError::Empty);
    }
    for read in reads {
        nat.check(read)?;
    }
    let raw: Vec<i64> = reads.iter().map(|r| ising.energy(r)).collect();
    let mut order: Vec<usize> = (0..reads.len()).collect();
    order.sort_by_key(|&r| raw[r]);
    let mut s_nat: Vec<Vec<i8>> = reads.to_vec();
    let mut descended = 0;
    for &r in &order {
        let stop = match work {
            Work::Fixed {
                descended: most, ..
            } => descended >= (*most).max(1),
            Work::Budget => {
                descended >= 1 && started.elapsed().as_secs_f64() >= GADGET_BUDGET_SECONDS / 2.0
            }
        };
        if stop {
            break;
        }
        nat.descend(&mut s_nat[r], cancel)?;
        descended += 1;
    }
    let e_nat: Vec<i64> = s_nat.iter().map(|s| ising.energy(s)).collect();
    let mut by_energy: Vec<usize> = (0..reads.len()).collect();
    by_energy.sort_by_key(|&r| e_nat[r]);
    let first_best = by_energy[0];
    let best = Start {
        read: Some(first_best),
        state: s_nat[first_best].clone(),
        energy: e_nat[first_best],
    };
    let mut starts: Vec<Start> = by_energy
        .iter()
        .take(READS_USED)
        .map(|&r| Start {
            read: Some(r),
            state: s_nat[r].clone(),
            energy: e_nat[r],
        })
        .collect();
    if descended >= 2 {
        // The vote of the best reads, ties to the best read, then
        // natural-move descent.
        let top = &by_energy[..VOTERS.min(descended)];
        let mut vote: Vec<i8> = (0..ising.num_nodes())
            .map(|i| {
                let sum: i64 = top.iter().map(|&r| i64::from(s_nat[r][i])).sum();
                match sum.signum() {
                    0 => s_nat[top[0]][i],
                    v => i8::try_from(v).unwrap_or(1),
                }
            })
            .collect();
        nat.descend(&mut vote, cancel)?;
        let energy = ising.energy(&vote);
        starts.insert(
            1,
            Start {
                read: None,
                state: vote,
                energy,
            },
        );
    }
    Ok(Prepared {
        descended,
        best,
        starts,
    })
}

/// The research `one_model` schedule of the constraints repair with default
/// options and the majority-vote start.
///
/// The reads get the natural-move descent in raw energy order, until half the
/// budget is used or every read is done. The best four descended reads and
/// the vote of the best eight, put second, each get the repair. The best
/// start gets half of the remaining time and the others share the rest.
///
/// # Errors
///
/// [`DescentError::Empty`] for no reads, [`DescentError::Length`] or
/// [`DescentError::Value`] for a read that does not fit the model, and
/// [`DescentError::Cancelled`] when the token abandons the watermark.
pub fn schedule_gadget(
    nat: &Greedy,
    repair: &Repair,
    reads: &[Vec<i8>],
    seed: u64,
    work: &Work,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<GadgetOutcome, DescentError> {
    let ising = repair.ising();
    let started = Instant::now();
    let used = || started.elapsed().as_secs_f64();
    let prepared = prepare_starts(nat, reads, work, started, cancel)?;
    let descent_best = prepared.best.energy;
    let (mut best_e, mut best_s) = (prepared.best.energy, prepared.best.state);

    let mut report_starts = Vec::new();
    let mut pool = Vec::new();
    let count = prepared.starts.len();
    for (k, start) in prepared.starts.into_iter().enumerate() {
        let spec = match work {
            Work::Fixed { ticks, .. } => match ticks.get(k) {
                Some(&limit) => ClockSpec::Ticks(limit, cancel),
                None => break,
            },
            Work::Budget => {
                let remaining = GADGET_BUDGET_SECONDS - used();
                if remaining <= 0.0 {
                    break;
                }
                let slice = if k == 0 && count > 1 {
                    remaining / 2.0
                } else {
                    remaining / (count - k) as f64
                };
                ClockSpec::Seconds(slice, cancel)
            }
        };
        let t0 = used();
        let mut s = start.state;
        let (d, stats, ticks) = repair.run(
            &mut s,
            &RepairOptions::default(),
            seed.wrapping_add(1000 * k as u64),
            spec,
        )?;
        let e = start.energy + d;
        report_starts.push(StartReport {
            read: start.read,
            start_energy: start.energy,
            energy: e,
            stats,
            ticks,
            seconds: used() - t0,
        });
        if e < best_e {
            best_e = e;
            best_s.clone_from(&s);
        }
        pool.push(s);
    }
    let report = GadgetReport {
        policy: GADGET_POLICY,
        budget_seconds: GADGET_BUDGET_SECONDS,
        work: work.clone(),
        descended: prepared.descended,
        descent_best,
        starts: report_starts,
        energy: ising.energy(&best_s),
        elapsed_seconds: used(),
    };
    Ok(GadgetOutcome {
        state: best_s,
        pool,
        report,
    })
}
