//! Exact structural baselines for the square-link families: a min-cost
//! circulation for `quip/flow-static` revision 1 (F-static) and a
//! compatible-charge construction for `quip/polariton-static` revision 1
//! (adapted P).
//!
//! Both families put one spin on each link of an open square-link lattice
//! (`docs/square-links.md` of quip-solver-core). Spin `+1` points from the
//! tail of a link to its head. The charge of a vertex is the signed
//! incidence times the spin, summed over its links: arrows in minus arrows
//! out. A labeled vertex must hold its label.
//!
//! The reduction starts from a baseline orientation: the requested direction
//! of every link for F-static, and spin `+1` for adapted P. Reversing a link
//! moves 2 units of charge from one end to the other, so reversals are a unit
//! flow on a network with one arc per link. A labeled vertex `v` with
//! baseline charge `Q0_v` and label `q_v` has the supply `(Q0_v - q_v) / 2`.
//! The free vertices of adapted P connect to one hub node that absorbs their
//! imbalance. The cost of an arc is the cost of reversing its link, so the
//! min-cost flow is the orientation with the lowest objective that meets
//! every label.
//!
//! [`ResidualNetwork`] adapts `Flow::add` and `Flow::solve` of the research
//! two-commodity routing solver: saturate every arc of negative cost, then
//! route each excess along a shortest path of reduced cost with Dijkstra. This
//! port bounds every capacity, checks every sum, and replaces the source
//! assertion with an infeasibility cut. A solved network is certified twice:
//! the flow conserves every supply within the capacities, and the final
//! potentials give every residual arc a nonnegative reduced cost, which proves
//! the cost optimal. A failed route returns the set of nodes the excess can
//! reach. Those nodes must send more flow out than their outgoing arcs carry,
//! which proves that no feasible orientation exists.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;

use quip_solver_core::families::FamilyContext;
use quip_solver_core::ising_generators::descriptor::{Contract, Params, Value};
use quip_solver_core::ising_generators::flow::FLOW_V1;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::ising_generators::polariton::POLARITON_V1;
use quip_solver_core::ising_generators::square::{charges, Boundary, SquareLattice};

use super::otsuka_sampler::section;

/// Metadata kind of the vertex labels: `L` as `u16`, then one `i8` label per
/// vertex in vertex order.
pub const KIND_CHARGES: u16 = 4;
/// Metadata kind of the window and the link positions: `c0` and `r0` as
/// `u16`, then the dense instance position of each link as `u32`.
pub const KIND_INCIDENCE: u16 = 5;
/// Metadata kind of the F-static request: the weight `U` as `u64`, then for
/// each link the target as `i8` and the cost as `u32`.
pub const KIND_FLOW: u16 = 9;
/// The label value of a free vertex in the `charges` section.
pub const FREE_LABEL: i8 = 127;
/// The parameter tag of the lattice side `L` in both families.
const TAG_SIDE: u16 = 3;

/// Which baseline a task runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `quip/flow-static` revision 1. Every vertex is labeled, and each link
    /// has a requested direction and a cost. The baseline is an exact
    /// min-cost circulation.
    FlowStatic,
    /// `quip/polariton-static` revision 1. Interior vertices are labeled, the
    /// boundary is free, and no link has a cost. The baseline constructs an
    /// orientation that meets every label, which has the lowest charge loss,
    /// 0.
    AdaptedP,
}

impl Mode {
    /// The mode of a contract, or `None` for any other family or revision.
    #[must_use]
    pub fn of(contract: &Contract) -> Option<Self> {
        let is =
            |c: &Contract| c.family_id == contract.family_id && c.revision == contract.revision;
        if is(&FLOW_V1) {
            Some(Self::FlowStatic)
        } else if is(&POLARITON_V1) {
            Some(Self::AdaptedP)
        } else {
            None
        }
    }

    /// The name of the mode in logs.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::FlowStatic => "f-static-min-cost-circulation",
            Self::AdaptedP => "adapted-p-compatible-charges",
        }
    }
}

/// Why an instance gives no link task.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaskError {
    /// The family or revision has no baseline here.
    Unsupported {
        /// The family id of the instance.
        family_id: &'static str,
        /// The revision of the instance.
        revision: u32,
    },
    /// The parameters do not parse against the schema, or have no side.
    Params,
    /// The canonical metadata check of the contract fails.
    Metadata,
    /// A metadata section is missing or does not parse.
    Section {
        /// The metadata kind.
        kind: u16,
        /// What is wrong with it.
        why: &'static str,
    },
    /// The lattice, the labels, the positions, or the costs disagree in size.
    Shape(&'static str),
}

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported {
                family_id,
                revision,
            } => write!(f, "{family_id} revision {revision} has no flow baseline"),
            Self::Params => f.write_str("the family parameters do not parse"),
            Self::Metadata => f.write_str("the instance fails the canonical metadata check"),
            Self::Section { kind, why } => write!(f, "metadata section {kind}: {why}"),
            Self::Shape(why) => write!(f, "the link task is malformed: {why}"),
        }
    }
}

impl std::error::Error for TaskError {}

/// The requested direction and the cost of every link, in link order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkCosts {
    /// The requested spin of each link, `+1` or `-1`.
    pub targets: Vec<i8>,
    /// The cost of each link when its spin differs from the target.
    pub costs: Vec<u32>,
}

/// The window origin and the link positions of an `incidence` section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Incidence {
    /// The cell column of lattice vertex `(0, 0)`.
    pub c0: u16,
    /// The cell row of lattice vertex `(0, 0)`.
    pub r0: u16,
    /// The dense instance position of each link, in link order.
    pub positions: Vec<u32>,
}

/// The labels of a `charges` payload for a lattice of side `side`: `None`
/// for [`FREE_LABEL`], the label otherwise. The family rules on which
/// vertices carry which labels are the canonical metadata check's business.
///
/// # Errors
///
/// [`TaskError::Section`] when the side differs or there is not exactly one
/// byte per vertex.
pub fn parse_charges(payload: &[u8], side: u16) -> Result<Vec<Option<i8>>, TaskError> {
    let bad = |why| TaskError::Section {
        kind: KIND_CHARGES,
        why,
    };
    let (head, raw) = payload
        .split_first_chunk::<2>()
        .ok_or_else(|| bad("truncated"))?;
    if u16::from_le_bytes(*head) != side {
        return Err(bad("the side differs from the parameters"));
    }
    let width = usize::from(side) + 1;
    if raw.len() != width * width {
        return Err(bad("not one label per vertex"));
    }
    Ok(raw
        .iter()
        .map(|&byte| {
            let label = i8::from_le_bytes([byte]);
            (label != FREE_LABEL).then_some(label)
        })
        .collect())
}

/// The window and link positions of an `incidence` payload for `links`
/// links on an instance of `nodes` nodes.
///
/// # Errors
///
/// [`TaskError::Section`] unless the payload holds exactly `links`
/// positions and they are a permutation of `0..nodes`.
pub fn parse_incidence(payload: &[u8], links: usize, nodes: usize) -> Result<Incidence, TaskError> {
    let bad = |why| TaskError::Section {
        kind: KIND_INCIDENCE,
        why,
    };
    let (window, rest) = payload
        .split_first_chunk::<4>()
        .ok_or_else(|| bad("truncated"))?;
    let (entries, tail) = rest.as_chunks::<4>();
    if entries.len() != links || !tail.is_empty() {
        return Err(bad("not one position per link"));
    }
    if nodes != links {
        return Err(bad("the links are not exactly the instance nodes"));
    }
    let mut seen = vec![false; nodes];
    let mut positions = Vec::with_capacity(links);
    for entry in entries {
        let position = u32::from_le_bytes(*entry);
        let slot = usize::try_from(position)
            .ok()
            .and_then(|at| seen.get_mut(at))
            .ok_or_else(|| bad("a position is out of range"))?;
        if std::mem::replace(slot, true) {
            return Err(bad("a position repeats"));
        }
        positions.push(position);
    }
    Ok(Incidence {
        c0: u16::from_le_bytes([window[0], window[1]]),
        r0: u16::from_le_bytes([window[2], window[3]]),
        positions,
    })
}

/// The targets and costs of a `flow` payload for `links` links. The weight
/// `U` is skipped: the canonical metadata check covers it.
///
/// # Errors
///
/// [`TaskError::Section`] unless the payload holds one entry per link and
/// every target is `+1` or `-1`.
pub fn parse_link_costs(payload: &[u8], links: usize) -> Result<LinkCosts, TaskError> {
    let bad = |why| TaskError::Section {
        kind: KIND_FLOW,
        why,
    };
    let (_weight, rest) = payload
        .split_first_chunk::<8>()
        .ok_or_else(|| bad("truncated"))?;
    let (entries, tail) = rest.as_chunks::<5>();
    if entries.len() != links || !tail.is_empty() {
        return Err(bad("not one request per link"));
    }
    let mut targets = Vec::with_capacity(links);
    let mut costs = Vec::with_capacity(links);
    for entry in entries {
        let target = i8::from_le_bytes([entry[0]]);
        if target != 1 && target != -1 {
            return Err(bad("a target is not +1 or -1"));
        }
        targets.push(target);
        costs.push(u32::from_le_bytes([entry[1], entry[2], entry[3], entry[4]]));
    }
    Ok(LinkCosts { targets, costs })
}

/// The direct check of one orientation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WitnessCheck {
    /// The direct charge of every vertex, in vertex order.
    pub charges: Vec<i8>,
    /// The number of labeled vertices whose charge differs from the label.
    pub violations: u32,
    /// The cost of the links whose spin differs from the target. Always 0
    /// for a task without costs.
    pub objective: u64,
    /// The number of links whose spin differs from the target.
    pub reversed: u32,
}

impl WitnessCheck {
    /// Whether every labeled vertex holds its label.
    #[must_use]
    pub const fn feasible(&self) -> bool {
        self.violations == 0
    }
}

/// A charge request on an open square-link lattice: the labels, the
/// instance position of each link, and the optional link costs.
#[derive(Clone, Debug)]
pub struct LinkTask {
    lattice: SquareLattice,
    positions: Vec<u32>,
    labels: Vec<Option<i8>>,
    costs: Option<LinkCosts>,
}

impl LinkTask {
    /// A task on `lattice`. `positions` maps link `e` to instance node
    /// `positions[e]` and must be a permutation of `0..links`. `labels` has
    /// one entry per vertex. `costs`, when present, has one target and one
    /// cost per link.
    ///
    /// # Errors
    ///
    /// [`TaskError::Shape`] when a length disagrees with the lattice, the
    /// positions are not a permutation, or a target is not `+1` or `-1`.
    pub fn new(
        lattice: SquareLattice,
        positions: Vec<u32>,
        labels: Vec<Option<i8>>,
        costs: Option<LinkCosts>,
    ) -> Result<Self, TaskError> {
        let links = lattice.link_count() as usize;
        if labels.len() != lattice.vertex_count() as usize {
            return Err(TaskError::Shape("not one label per vertex"));
        }
        if positions.len() != links {
            return Err(TaskError::Shape("not one position per link"));
        }
        let mut seen = vec![false; links];
        for &p in &positions {
            match usize::try_from(p).ok().and_then(|at| seen.get_mut(at)) {
                Some(slot) if !*slot => *slot = true,
                _ => return Err(TaskError::Shape("the positions are not a permutation")),
            }
        }
        if let Some(costs) = &costs {
            if costs.targets.len() != links || costs.costs.len() != links {
                return Err(TaskError::Shape("not one target and one cost per link"));
            }
            if costs.targets.iter().any(|&t| t != 1 && t != -1) {
                return Err(TaskError::Shape("a target is not +1 or -1"));
            }
        }
        Ok(Self {
            lattice,
            positions,
            labels,
            costs,
        })
    }

    /// The task of a validated instance of `contract` with canonical
    /// parameters `params`. The canonical metadata check of the contract runs
    /// first, so bad metadata is rejected before any section is read.
    ///
    /// # Errors
    ///
    /// [`TaskError::Unsupported`] for a family without a baseline here,
    /// [`TaskError::Params`] or [`TaskError::Metadata`] for parameters or
    /// metadata the contract refuses, and [`TaskError::Section`] or
    /// [`TaskError::Shape`] for sections this parser refuses.
    pub fn from_instance(
        contract: &Contract,
        params: &[u8],
        instance: &Instance,
    ) -> Result<(Mode, Self), TaskError> {
        let mode = Mode::of(contract).ok_or(TaskError::Unsupported {
            family_id: contract.family_id,
            revision: contract.revision,
        })?;
        let parsed = Params::parse(contract.schema, params).map_err(|_| TaskError::Params)?;
        let construction = contract.construction.ok_or(TaskError::Metadata)?;
        if !(construction.check_metadata)(&parsed, instance) {
            return Err(TaskError::Metadata);
        }
        let side = match parsed.get(TAG_SIDE) {
            Some(Value::Unsigned(side)) => u16::try_from(side).map_err(|_| TaskError::Params)?,
            _ => return Err(TaskError::Params),
        };
        let lattice = SquareLattice::new(side, Boundary::Open).map_err(|_| TaskError::Params)?;
        let links = lattice.link_count() as usize;
        let metadata = instance.metadata();
        let payload = |kind| {
            section(metadata, kind).ok_or(TaskError::Section {
                kind,
                why: "missing",
            })
        };
        let labels = parse_charges(payload(KIND_CHARGES)?, side)?;
        let incidence =
            parse_incidence(payload(KIND_INCIDENCE)?, links, instance.node_ids().len())?;
        let costs = match mode {
            Mode::FlowStatic => Some(parse_link_costs(payload(KIND_FLOW)?, links)?),
            Mode::AdaptedP => None,
        };
        Ok((
            mode,
            Self::new(lattice, incidence.positions, labels, costs)?,
        ))
    }

    /// The task of a family job the session validated.
    ///
    /// # Errors
    ///
    /// As [`LinkTask::from_instance`].
    pub fn from_family(family: &FamilyContext) -> Result<(Mode, Self), TaskError> {
        Self::from_instance(family.contract(), family.params(), family.instance())
    }

    /// The lattice.
    #[must_use]
    pub const fn lattice(&self) -> &SquareLattice {
        &self.lattice
    }

    /// The instance position of each link, in link order.
    #[must_use]
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }

    /// The label of each vertex, `None` when free.
    #[must_use]
    pub fn labels(&self) -> &[Option<i8>] {
        &self.labels
    }

    /// The link targets and costs, when the task has an objective.
    #[must_use]
    pub const fn costs(&self) -> Option<&LinkCosts> {
        self.costs.as_ref()
    }

    /// The spin of each link, in link order, from one spin per instance
    /// node. `None` when the length differs.
    #[must_use]
    pub fn link_spins(&self, node_spins: &[i8]) -> Option<Vec<i8>> {
        if node_spins.len() != self.positions.len() {
            return None;
        }
        self.positions
            .iter()
            .map(|&p| node_spins.get(p as usize).copied())
            .collect()
    }

    /// The spin of each instance node from one spin per link. `None` when
    /// the length differs.
    #[must_use]
    pub fn node_spins(&self, link_spins: &[i8]) -> Option<Vec<i8>> {
        if link_spins.len() != self.positions.len() {
            return None;
        }
        let mut nodes = vec![0_i8; link_spins.len()];
        for (&p, &spin) in self.positions.iter().zip(link_spins) {
            *nodes.get_mut(p as usize)? = spin;
        }
        Some(nodes)
    }

    /// The direct check of `link_spins`, one spin per link in link order.
    /// `None` when the length differs or a spin is not `+1` or `-1`.
    #[must_use]
    pub fn check(&self, link_spins: &[i8]) -> Option<WitnessCheck> {
        let got = charges(&self.lattice, link_spins)?;
        let mut violations = 0_u32;
        for (&q, label) in got.iter().zip(&self.labels) {
            if label.is_some_and(|want| want != q) {
                violations = violations.checked_add(1)?;
            }
        }
        let (mut objective, mut reversed) = (0_u64, 0_u32);
        if let Some(costs) = &self.costs {
            for ((&spin, &target), &cost) in link_spins.iter().zip(&costs.targets).zip(&costs.costs)
            {
                if spin != target {
                    objective = objective.checked_add(u64::from(cost))?;
                    reversed = reversed.checked_add(1)?;
                }
            }
        }
        Some(WitnessCheck {
            charges: got,
            violations,
            objective,
            reversed,
        })
    }
}

/// Why a flow computation stopped without a result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlowError {
    /// A sum, a distance, or a potential does not fit `i64`, or an index
    /// does not fit `u32`.
    Overflow,
    /// The cancel check fired between two graph operations.
    Cancelled,
    /// An arc or a supply names a node the network does not have.
    NodeOutOfRange,
    /// The supplies do not sum to 0.
    Unbalanced,
    /// A result failed its own certificate. This is a defect of the solver.
    Certificate(&'static str),
}

impl fmt::Display for FlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => f.write_str("a flow quantity overflows"),
            Self::Cancelled => f.write_str("cancelled"),
            Self::NodeOutOfRange => f.write_str("a node is out of range"),
            Self::Unbalanced => f.write_str("the supplies do not sum to 0"),
            Self::Certificate(why) => write!(f, "the flow certificate fails: {why}"),
        }
    }
}

impl std::error::Error for FlowError {}

/// An arc of a [`ResidualNetwork`], as [`ResidualNetwork::add_arc`] returns it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArcId(u32);

/// The optimality certificate of a solved network.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Certificate {
    /// The total cost of the flow.
    pub cost: i64,
    /// A node potential that gives every residual arc a nonnegative reduced
    /// cost.
    pub potentials: Vec<i64>,
    /// The number of shortest-path augmentations.
    pub augmentations: u64,
}

/// An infeasibility certificate: a set of nodes whose supply exceeds the
/// capacity of the arcs that leave it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cut {
    /// The nodes of the set, ascending.
    pub nodes: Vec<u32>,
    /// The sum of their supplies: the net flow that must leave the set.
    pub required: i64,
    /// The total capacity of the arcs from the set to the other nodes.
    pub capacity: i64,
}

/// The result of [`ResidualNetwork::solve`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Solved {
    /// A feasible flow of minimum cost.
    Optimal(Certificate),
    /// No feasible flow exists.
    Infeasible(Cut),
}

/// A residual network with bounded integral capacities and integer costs.
/// Each arc is stored with its reverse: arc `2k` and arc `2k + 1`.
#[derive(Clone, Debug)]
pub struct ResidualNetwork {
    head: Vec<u32>,
    cap: Vec<u32>,
    cost: Vec<i64>,
    out: Vec<Vec<u32>>,
    supply: Vec<i64>,
}

/// The sentinel of an unreached node or an empty predecessor.
const NONE: u32 = u32::MAX;

impl ResidualNetwork {
    /// A network of `nodes` nodes with no arcs and no supply.
    ///
    /// # Errors
    ///
    /// [`FlowError::Overflow`] when `nodes` does not fit `u32`.
    pub fn new(nodes: usize) -> Result<Self, FlowError> {
        if u32::try_from(nodes).map_or(true, |n| n == NONE) {
            return Err(FlowError::Overflow);
        }
        Ok(Self {
            head: Vec::new(),
            cap: Vec::new(),
            cost: Vec::new(),
            out: vec![Vec::new(); nodes],
            supply: vec![0; nodes],
        })
    }

    /// The number of nodes.
    #[must_use]
    pub fn nodes(&self) -> usize {
        self.out.len()
    }

    /// The number of arcs, not counting reverses.
    #[must_use]
    pub fn arcs(&self) -> usize {
        self.head.len() / 2
    }

    /// Add an arc from `tail` to `head` with `capacity` and a cost per unit.
    ///
    /// # Errors
    ///
    /// [`FlowError::NodeOutOfRange`] for an unknown node and
    /// [`FlowError::Overflow`] when the arc count or the negated cost does
    /// not fit.
    pub fn add_arc(
        &mut self,
        tail: u32,
        head: u32,
        capacity: u32,
        cost: i64,
    ) -> Result<ArcId, FlowError> {
        let nodes = self.out.len();
        if tail as usize >= nodes || head as usize >= nodes {
            return Err(FlowError::NodeOutOfRange);
        }
        let k = u32::try_from(self.head.len()).map_err(|_| FlowError::Overflow)?;
        if k.checked_add(2).is_none_or(|end| end == NONE) {
            return Err(FlowError::Overflow);
        }
        let back = cost.checked_neg().ok_or(FlowError::Overflow)?;
        self.head.extend([head, tail]);
        self.cap.extend([capacity, 0]);
        self.cost.extend([cost, back]);
        self.out[tail as usize].push(k);
        self.out[head as usize].push(k + 1);
        Ok(ArcId(k / 2))
    }

    /// Add `amount` to the supply of `node`: the net flow that must leave
    /// it. A negative supply is a demand.
    ///
    /// # Errors
    ///
    /// [`FlowError::NodeOutOfRange`] or [`FlowError::Overflow`].
    pub fn add_supply(&mut self, node: u32, amount: i64) -> Result<(), FlowError> {
        let slot = self
            .supply
            .get_mut(node as usize)
            .ok_or(FlowError::NodeOutOfRange)?;
        *slot = slot.checked_add(amount).ok_or(FlowError::Overflow)?;
        Ok(())
    }

    /// The flow on `arc`.
    #[must_use]
    pub fn flow(&self, arc: ArcId) -> u32 {
        self.cap.get(arc.0 as usize * 2 + 1).copied().unwrap_or(0)
    }

    /// The tail of directed residual arc `k`.
    fn tail(&self, k: usize) -> u32 {
        self.head[k ^ 1]
    }

    /// Solve for a minimum-cost flow that meets every supply. The network
    /// must carry no flow yet. `cancelled` runs before each augmentation, and
    /// a `true` stops the solve.
    ///
    /// # Errors
    ///
    /// [`FlowError::Unbalanced`], [`FlowError::Overflow`],
    /// [`FlowError::Cancelled`], or [`FlowError::Certificate`] when a result
    /// fails its own check.
    pub fn solve(&mut self, cancelled: &mut dyn FnMut() -> bool) -> Result<Solved, FlowError> {
        let total = self
            .supply
            .iter()
            .try_fold(0_i64, |acc, &s| acc.checked_add(s))
            .ok_or(FlowError::Overflow)?;
        if total != 0 {
            return Err(FlowError::Unbalanced);
        }
        let nodes = self.out.len();
        let mut excess = self.supply.clone();
        // Saturate every arc of negative cost, so that every residual arc
        // starts with a nonnegative cost and the potentials can start at 0.
        for k in (0..self.head.len()).step_by(2) {
            if self.cost[k] < 0 && self.cap[k] > 0 {
                let units = self.cap[k];
                self.cap[k] = 0;
                self.cap[k + 1] = self.cap[k + 1]
                    .checked_add(units)
                    .ok_or(FlowError::Overflow)?;
                let (tail, head) = (self.tail(k) as usize, self.head[k] as usize);
                excess[tail] = excess[tail]
                    .checked_sub(i64::from(units))
                    .ok_or(FlowError::Overflow)?;
                excess[head] = excess[head]
                    .checked_add(i64::from(units))
                    .ok_or(FlowError::Overflow)?;
            }
        }
        let mut potentials = vec![0_i64; nodes];
        let mut dist = vec![i64::MAX; nodes];
        let mut prev = vec![NONE; nodes];
        let mut augmentations = 0_u64;
        while excess.iter().any(|&x| x > 0) {
            if cancelled() {
                return Err(FlowError::Cancelled);
            }
            dist.fill(i64::MAX);
            prev.fill(NONE);
            let mut heap = BinaryHeap::new();
            for (v, &x) in excess.iter().enumerate() {
                if x > 0 {
                    dist[v] = 0;
                    heap.push(Reverse((0_i64, v)));
                }
            }
            let mut target = None;
            while let Some(Reverse((d, u))) = heap.pop() {
                if d > dist[u] {
                    continue;
                }
                if excess[u] < 0 {
                    target = Some(u);
                    break;
                }
                for &k in &self.out[u] {
                    let k = k as usize;
                    if self.cap[k] == 0 {
                        continue;
                    }
                    let v = self.head[k] as usize;
                    let step = self.cost[k]
                        .checked_add(potentials[u])
                        .and_then(|c| c.checked_sub(potentials[v]))
                        .ok_or(FlowError::Overflow)?;
                    let nd = d.checked_add(step).ok_or(FlowError::Overflow)?;
                    if nd < dist[v] {
                        dist[v] = nd;
                        prev[v] = k as u32;
                        heap.push(Reverse((nd, v)));
                    }
                }
            }
            let Some(target) = target else {
                return self.cut(&dist).map(Solved::Infeasible);
            };
            let reach = dist[target];
            for (p, &d) in potentials.iter_mut().zip(&dist) {
                *p = p.checked_add(d.min(reach)).ok_or(FlowError::Overflow)?;
            }
            // Walk back to the source of the path for the bottleneck.
            let mut units = u32::MAX;
            let mut v = target;
            while prev[v] != NONE {
                let k = prev[v] as usize;
                units = units.min(self.cap[k]);
                v = self.tail(k) as usize;
            }
            let source = v;
            let room = |x: i64| u32::try_from(x).unwrap_or(u32::MAX);
            let deficit = excess[target].checked_neg().ok_or(FlowError::Overflow)?;
            units = units.min(room(excess[source])).min(room(deficit));
            let mut v = target;
            while prev[v] != NONE {
                let k = prev[v] as usize;
                self.cap[k] -= units;
                self.cap[k ^ 1] = self.cap[k ^ 1]
                    .checked_add(units)
                    .ok_or(FlowError::Overflow)?;
                v = self.tail(k) as usize;
            }
            excess[source] -= i64::from(units);
            excess[target] += i64::from(units);
            augmentations += 1;
        }
        let cost = self.certify(&potentials)?;
        Ok(Solved::Optimal(Certificate {
            cost,
            potentials,
            augmentations,
        }))
    }

    /// The cut of the nodes that `dist` reached, checked to prove
    /// infeasibility.
    fn cut(&self, dist: &[i64]) -> Result<Cut, FlowError> {
        let inside: Vec<bool> = dist.iter().map(|&d| d != i64::MAX).collect();
        let nodes: Vec<u32> = (0_u32..)
            .zip(&inside)
            .filter_map(|(v, &is)| is.then_some(v))
            .collect();
        let (required, capacity) = self.cut_balance(&inside)?;
        if required <= capacity {
            return Err(FlowError::Certificate(
                "the cut does not prove infeasibility",
            ));
        }
        Ok(Cut {
            nodes,
            required,
            capacity,
        })
    }

    /// The supply of the nodes marked in `inside`, and the total capacity of
    /// the arcs from them to the other nodes. A set whose supply exceeds the
    /// capacity proves that no feasible flow exists.
    ///
    /// # Errors
    ///
    /// [`FlowError::NodeOutOfRange`] when `inside` is not one entry per
    /// node, and [`FlowError::Overflow`].
    pub fn cut_balance(&self, inside: &[bool]) -> Result<(i64, i64), FlowError> {
        if inside.len() != self.out.len() {
            return Err(FlowError::NodeOutOfRange);
        }
        let mut required = 0_i64;
        for (&s, &is) in self.supply.iter().zip(inside) {
            if is {
                required = required.checked_add(s).ok_or(FlowError::Overflow)?;
            }
        }
        let mut capacity = 0_i64;
        for k in (0..self.head.len()).step_by(2) {
            let (tail, head) = (self.tail(k) as usize, self.head[k] as usize);
            if inside[tail] && !inside[head] {
                let full = i64::from(self.cap[k]) + i64::from(self.cap[k + 1]);
                capacity = capacity.checked_add(full).ok_or(FlowError::Overflow)?;
            }
        }
        Ok((required, capacity))
    }

    /// Check the current flow against `potentials` and return its cost: every
    /// node sends out exactly its supply, and every residual arc has a
    /// nonnegative reduced cost, which proves the flow has the least cost.
    ///
    /// # Errors
    ///
    /// [`FlowError::Certificate`] when a check fails, and
    /// [`FlowError::Overflow`].
    pub fn certify(&self, potentials: &[i64]) -> Result<i64, FlowError> {
        if potentials.len() != self.out.len() {
            return Err(FlowError::Certificate("not one potential per node"));
        }
        let mut net = vec![0_i64; self.out.len()];
        let mut cost = 0_i64;
        for k in (0..self.head.len()).step_by(2) {
            let flow = i64::from(self.cap[k + 1]);
            let (tail, head) = (self.tail(k) as usize, self.head[k] as usize);
            net[tail] = net[tail].checked_add(flow).ok_or(FlowError::Overflow)?;
            net[head] = net[head].checked_sub(flow).ok_or(FlowError::Overflow)?;
            cost = flow
                .checked_mul(self.cost[k])
                .and_then(|c| c.checked_add(cost))
                .ok_or(FlowError::Overflow)?;
        }
        if net != self.supply {
            return Err(FlowError::Certificate("a supply is not met"));
        }
        for k in 0..self.head.len() {
            if self.cap[k] == 0 {
                continue;
            }
            let reduced = i128::from(self.cost[k]) + i128::from(potentials[self.tail(k) as usize])
                - i128::from(potentials[self.head[k] as usize]);
            if reduced < 0 {
                return Err(FlowError::Certificate(
                    "a residual arc has a negative reduced cost",
                ));
            }
        }
        Ok(cost)
    }
}

/// An orientation that meets every label with the lowest objective.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Orientation {
    /// The spin of each link, in link order.
    pub link_spins: Vec<i8>,
    /// The objective: the cost of the links against their target, 0 without
    /// costs.
    pub objective: u64,
    /// The number of shortest-path augmentations the solve took.
    pub augmentations: u64,
}

/// Why no orientation meets the labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Infeasibility {
    /// A label has the wrong parity for the degree of its vertex, or a size
    /// above the degree.
    Local {
        /// The vertex.
        vertex: u32,
        /// Its label.
        label: i8,
    },
    /// Every vertex is labeled and the labels do not sum to 0.
    Total {
        /// The sum of the labels.
        sum: i64,
    },
    /// A set of network nodes must send more flow out than its arcs carry.
    /// Node `vertex_count` is the hub of the free vertices.
    Cut(Cut),
}

impl fmt::Display for Infeasibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Local { vertex, label } => {
                write!(f, "label {label} cannot hold at vertex {vertex}")
            }
            Self::Total { sum } => write!(f, "the labels sum to {sum}, not 0"),
            Self::Cut(cut) => write!(
                f,
                "a set of {} nodes needs {} reversals out but its arcs carry {}",
                cut.nodes.len(),
                cut.required,
                cut.capacity
            ),
        }
    }
}

/// The status of one baseline run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// A certified optimum: the orientation meets every label, and the flow
    /// certificate proves no orientation that meets every label has a lower
    /// objective.
    Optimal(Orientation),
    /// A certified infeasibility: no orientation meets every label.
    Infeasible(Infeasibility),
}

impl Outcome {
    /// The status name in logs.
    #[must_use]
    pub const fn status(&self) -> &'static str {
        match self {
            Self::Optimal(_) => "certified-optimum",
            Self::Infeasible(_) => "infeasible",
        }
    }
}

/// Orient every link of `task` to meet every label with the lowest
/// objective. `cancelled` runs between graph operations.
///
/// # Errors
///
/// [`FlowError::Cancelled`] when `cancelled` fires, [`FlowError::Overflow`],
/// and [`FlowError::Certificate`] when the result fails the direct check.
pub fn orient(task: &LinkTask, cancelled: &mut dyn FnMut() -> bool) -> Result<Outcome, FlowError> {
    let lattice = task.lattice();
    let vertices = lattice.vertex_count();
    let links = lattice.link_count();
    let baseline: Vec<i8> = task
        .costs()
        .map_or_else(|| vec![1; links as usize], |c| c.targets.clone());
    let start = charges(lattice, &baseline).ok_or(FlowError::Certificate("the baseline"))?;
    let mut free = false;
    let mut sum = 0_i64;
    for (vertex, label) in (0_u32..).zip(task.labels()) {
        let Some(label) = *label else {
            free = true;
            continue;
        };
        let degree = i16::from(lattice.degree(vertex).ok_or(FlowError::NodeOutOfRange)?);
        let q = i16::from(label);
        if q.abs() > degree || (q - degree) % 2 != 0 {
            return Ok(Outcome::Infeasible(Infeasibility::Local { vertex, label }));
        }
        sum += i64::from(label);
    }
    if !free && sum != 0 {
        return Ok(Outcome::Infeasible(Infeasibility::Total { sum }));
    }
    if cancelled() {
        return Err(FlowError::Cancelled);
    }
    let hub = vertices;
    let nodes = vertices as usize + usize::from(free);
    let mut network = ResidualNetwork::new(nodes)?;
    let mut arcs = Vec::with_capacity(links as usize);
    for e in 0..links {
        let link = lattice.link(e).ok_or(FlowError::NodeOutOfRange)?;
        // The baseline sends the unit from `from` to `to`, and a reversal
        // sends it back: one unit of flow from `to` to `from`.
        let (from, to) = if baseline[e as usize] == 1 {
            (link.tail, link.head)
        } else {
            (link.head, link.tail)
        };
        let cost = task.costs().map_or(0, |c| i64::from(c.costs[e as usize]));
        arcs.push(network.add_arc(to, from, 1, cost)?);
    }
    let mut hub_supply = 0_i64;
    for (vertex, label) in (0_u32..).zip(task.labels()) {
        match *label {
            Some(q) => {
                let supply = (i64::from(start[vertex as usize]) - i64::from(q)) / 2;
                network.add_supply(vertex, supply)?;
                hub_supply -= supply;
            }
            None => {
                let degree = u32::from(lattice.degree(vertex).ok_or(FlowError::NodeOutOfRange)?);
                network.add_arc(hub, vertex, degree, 0)?;
                network.add_arc(vertex, hub, degree, 0)?;
            }
        }
    }
    if free {
        network.add_supply(hub, hub_supply)?;
    }
    let certificate = match network.solve(cancelled)? {
        Solved::Optimal(certificate) => certificate,
        Solved::Infeasible(cut) => return Ok(Outcome::Infeasible(Infeasibility::Cut(cut))),
    };
    let link_spins: Vec<i8> = baseline
        .iter()
        .zip(&arcs)
        .map(|(&b, &arc)| if network.flow(arc) == 1 { -b } else { b })
        .collect();
    let check = task
        .check(&link_spins)
        .ok_or(FlowError::Certificate("the orientation does not check"))?;
    if !check.feasible() {
        return Err(FlowError::Certificate("the orientation breaks a label"));
    }
    if i64::try_from(check.objective).ok() != Some(certificate.cost) {
        return Err(FlowError::Certificate(
            "the objective differs from the flow cost",
        ));
    }
    Ok(Outcome::Optimal(Orientation {
        link_spins,
        objective: check.objective,
        augmentations: certificate.augmentations,
    }))
}
