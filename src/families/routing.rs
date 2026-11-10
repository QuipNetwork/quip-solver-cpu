//! The original two-commodity routing attack on `quip/two-commodity-routing`
//! revision 1: a literal port of the research solver `routesolve`
//! (`new-pow/two-commodity-routing/solver/src/main.rs`).
//!
//! With one commodity fixed, the energy of the other is a minimum-cost
//! circulation. Each arc costs its own cost plus the capacity penalty when the
//! fixed commodity uses it. A hub node absorbs the imbalance of every node
//! through unit pieces of the convex conservation penalty. The circulation is
//! solved exactly: saturate the negative arcs, then route the excess along
//! shortest paths. The two commodities alternate, and a single-flip descent on
//! the full Ising model finishes each start order.
//!
//! The policy ([`POLICY`]) is the alternation of exact single-commodity
//! responses from both start orders, then descent. Each response is optimal
//! with the other commodity fixed. The alternation as a whole is a heuristic,
//! and nothing here certifies a global optimum.
//!
//! The Ising energy convention is the original one: `E = sum w s_u s_v + sum
//! h s` in milli, with `x = (1 - s) / 2`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;

use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::CancelToken;

/// The policy name, versioned apart from the family revision.
pub const POLICY: &str = "routing-alt/1";

/// The weight of a conservation square, in milli. It is the value of
/// revision 1 of the family.
pub const CONSERVE: i64 = 248;
/// The weight of the capacity coupler, in milli. Revision 1 value.
pub const CAPACITY: i64 = 496;
/// The most alternation steps of one start order, the research default.
pub const MAX_ALTERNATIONS: usize = 50;
/// The metadata section kind of the routing section.
pub const ROUTING_SECTION: u16 = 7;
/// The two arc costs the family draws, in milli.
const COSTS: [i64; 2] = [62, 124];
/// The most spins of a model, the family instance limit.
pub const MAX_SPINS: usize = 131_072;
/// The most arcs of a model. Every arc owns three spins.
pub const MAX_ARCS: usize = MAX_SPINS / 3;
/// The largest field or coupling magnitude in milli, the `i32` wire range.
const MAX_MILLI: i64 = i32::MAX as i64;
/// The bytes of one node record: `c`, `r`, and two supplies.
const NODE_RECORD: usize = 6;
/// The bytes of one arc record: five `u32` and two `u16`.
const ARC_RECORD: usize = 24;

/// The cancel token and watermark of a job, if it has them.
pub type Guard<'a> = Option<(&'a CancelToken, Option<u64>)>;

/// Why a model or a run is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingError {
    /// The model or its metadata is malformed, or beyond a bound.
    Model(String),
    /// No path joins an excess node to a deficit node. The hub joins every
    /// node, so this signals a broken residual graph.
    NoPath,
    /// The cancel token abandoned the job's watermark.
    Cancelled,
}

impl fmt::Display for RoutingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(why) => write!(f, "invalid routing model: {why}"),
            Self::NoPath => write!(f, "no path from an excess node to a deficit node"),
            Self::Cancelled => write!(f, "the routing run was cancelled"),
        }
    }
}

impl std::error::Error for RoutingError {}

fn invalid(why: impl Into<String>) -> RoutingError {
    RoutingError::Model(why.into())
}

/// The routing structure next to the Ising arrays, as the research model file
/// stores it. Node and spin positions are indices.
#[derive(Debug, Clone, Default)]
pub struct Parts {
    /// The field of each spin, in milli. Its length is the spin count.
    pub h: Vec<i64>,
    /// The couplings `(u, v, w)`, in milli. Zero couplings are dropped.
    pub edges: Vec<(u32, u32, i64)>,
    /// The supply of each node for each commodity.
    pub supply: Vec<[i64; 2]>,
    /// The `(tail, head)` node of each arc.
    pub arcs: Vec<(u32, u32)>,
    /// The commodity 1 spin of each arc.
    pub spin1: Vec<u32>,
    /// The two commodity 2 chain spins of each arc.
    pub chain2: Vec<[u32; 2]>,
    /// The cost of each arc for each commodity, in milli.
    pub cost: Vec<[i64; 2]>,
}

/// A validated routing model.
#[derive(Debug, Clone)]
pub struct Routing {
    n: usize,
    h: Vec<i64>,
    edges: Vec<(usize, usize, i64)>,
    adj: Vec<Vec<(usize, i64)>>,
    supply: Vec<[i64; 2]>,
    arcs: Vec<(usize, usize)>,
    spin1: Vec<usize>,
    chain2: Vec<[usize; 2]>,
    cost: Vec<[i64; 2]>,
}

/// One start order's final state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// The start order: the commodity solved first.
    pub first: usize,
    /// The energy of `spins`, in milli, from the kernel's own scoring.
    pub energy: i64,
    /// The full ordered spin vector. A spin no arc uses is `+1`.
    pub spins: Vec<i8>,
    /// The exact single-commodity solves of the alternation.
    pub solves: usize,
    /// Whether the alternation stopped because a response did not improve,
    /// and not because it reached [`MAX_ALTERNATIONS`].
    pub settled: bool,
    /// The energy after the alternation and before the descent.
    pub before_descent: i64,
    /// The spin flips of the final descent.
    pub flips: u64,
}

fn convert(value: u32, bound: usize, what: &str) -> Result<usize, RoutingError> {
    let index = usize::try_from(value).map_err(|_| invalid(format!("{what} does not fit")))?;
    if index < bound {
        Ok(index)
    } else {
        Err(invalid(format!("{what} {index} is not below {bound}")))
    }
}

impl Routing {
    /// Check `parts` against every bound and build the model.
    ///
    /// # Errors
    /// [`RoutingError::Model`] when an array length, index, or magnitude is
    /// out of bounds, when a coupling is a self-loop, when two arcs share a
    /// spin, or when an arc has the wrong shape.
    pub fn new(parts: Parts) -> Result<Self, RoutingError> {
        let Parts {
            h,
            edges,
            supply,
            arcs,
            spin1,
            chain2,
            cost,
        } = parts;
        let n = h.len();
        let m = arcs.len();
        if n == 0 || n > MAX_SPINS {
            return Err(invalid(format!("{n} spins is outside 1..={MAX_SPINS}")));
        }
        if m == 0 || m > MAX_ARCS || supply.is_empty() || supply.len() > MAX_SPINS {
            return Err(invalid("the node or arc count is outside its bounds"));
        }
        if spin1.len() != m || chain2.len() != m || cost.len() != m {
            return Err(invalid("the arc arrays differ in length"));
        }
        let magnitude = |v: i64| v.unsigned_abs() <= MAX_MILLI.unsigned_abs();
        if !h.iter().all(|&v| magnitude(v)) || !cost.iter().flatten().all(|&v| magnitude(v)) {
            return Err(invalid("a field or cost is beyond the milli range"));
        }
        if !supply.iter().flatten().all(|&b| (-1..=1).contains(&b)) {
            return Err(invalid("a supply is not -1, 0, or 1"));
        }
        let nodes = supply.len();
        let mut kept = Vec::with_capacity(edges.len());
        let mut adj = vec![Vec::new(); n];
        for &(u, v, w) in &edges {
            let (u, v) = (
                convert(u, n, "a coupling end")?,
                convert(v, n, "a coupling end")?,
            );
            if u == v {
                return Err(invalid("a coupling joins a spin to itself"));
            }
            if !magnitude(w) {
                return Err(invalid("a coupling is beyond the milli range"));
            }
            if w != 0 {
                kept.push((u, v, w));
                adj[u].push((v, w));
                adj[v].push((u, w));
            }
        }
        let mut used = vec![false; n];
        let mut take = |spin: usize| -> Result<usize, RoutingError> {
            if std::mem::replace(&mut used[spin], true) {
                Err(invalid(format!("spin {spin} belongs to two arcs")))
            } else {
                Ok(spin)
            }
        };
        let mut arc_nodes = Vec::with_capacity(m);
        let mut s1 = Vec::with_capacity(m);
        let mut ch = Vec::with_capacity(m);
        for a in 0..m {
            let (tail, head) = arcs[a];
            let (tail, head) = (
                convert(tail, nodes, "an arc end")?,
                convert(head, nodes, "an arc end")?,
            );
            if tail == head {
                return Err(invalid("an arc joins a node to itself"));
            }
            arc_nodes.push((tail, head));
            s1.push(take(convert(spin1[a], n, "a commodity 1 spin")?)?);
            let [c0, c1] = chain2[a];
            ch.push([
                take(convert(c0, n, "a chain spin")?)?,
                take(convert(c1, n, "a chain spin")?)?,
            ]);
        }
        Ok(Self {
            n,
            h,
            edges: kept,
            adj,
            supply,
            arcs: arc_nodes,
            spin1: s1,
            chain2: ch,
            cost,
        })
    }

    /// Decode and check the routing section of a validated instance.
    ///
    /// The metadata region must hold exactly one section, of kind
    /// [`ROUTING_SECTION`], with no byte left over. Unknown metadata is an
    /// error.
    ///
    /// # Errors
    /// [`RoutingError::Model`] when the region or the section is malformed,
    /// the counts disagree with the instance, or an arc cost is not one the
    /// family draws.
    pub fn from_instance(instance: &Instance) -> Result<Self, RoutingError> {
        let payload = only_section(instance.metadata())?;
        let (nodes, rest) = decode_nodes(payload)?;
        let (arcs, strength) = decode_arcs(rest)?;
        if !(1..=1000).contains(&strength) {
            return Err(invalid("the chain strength is outside 1..=1000"));
        }
        let n = instance.node_ids().len();
        if arcs.len().checked_mul(3) != Some(n) {
            return Err(invalid("the instance does not have three spins per arc"));
        }
        let edges = instance
            .edges()
            .iter()
            .zip(instance.j())
            .map(|(&(u, v), &j)| (u, v, i64::from(j)))
            .collect();
        Self::new(Parts {
            h: instance.h().iter().map(|&v| i64::from(v)).collect(),
            edges,
            supply: nodes,
            arcs: arcs.iter().map(|a| (a.tail, a.head)).collect(),
            spin1: arcs.iter().map(|a| a.x1).collect(),
            chain2: arcs.iter().map(|a| a.chain).collect(),
            cost: arcs.iter().map(|a| a.cost).collect(),
        })
    }

    /// The spin count, active or not.
    #[must_use]
    pub fn spins(&self) -> usize {
        self.n
    }

    /// The arc count.
    #[must_use]
    pub fn arc_count(&self) -> usize {
        self.arcs.len()
    }

    /// The spin vector of the arc bits `x`. A spin no arc uses has no coupler
    /// and no field, so `+1` is as good as `-1`, and it keeps its position.
    #[must_use]
    pub fn to_spins(&self, x: &[Vec<u8>; 2]) -> Vec<i8> {
        let mut s = vec![1i8; self.n];
        for a in 0..self.arcs.len() {
            s[self.spin1[a]] = spin_of(x[0][a]);
            for &q in &self.chain2[a] {
                s[q] = spin_of(x[1][a]);
            }
        }
        s
    }

    /// The energy of `s` in milli: `sum w s_u s_v + sum h s`.
    #[must_use]
    pub fn energy(&self, s: &[i8]) -> i64 {
        let coupling: i64 = self
            .edges
            .iter()
            .map(|&(u, v, w)| w * i64::from(s[u]) * i64::from(s[v]))
            .sum();
        let field: i64 = self.h.iter().zip(s).map(|(&h, &s)| h * i64::from(s)).sum();
        coupling + field
    }

    /// The exact best flow of commodity `c` with the other commodity fixed to
    /// `other`.
    ///
    /// # Errors
    /// [`RoutingError::NoPath`] or [`RoutingError::Cancelled`].
    pub fn solve_commodity(
        &self,
        c: usize,
        other: &[u8],
        cancel: Guard<'_>,
    ) -> Result<Vec<u8>, RoutingError> {
        let nodes = self.supply.len();
        let hub = nodes;
        let mut f = Flow::new(nodes + 1);
        let real: Vec<usize> = self
            .arcs
            .iter()
            .enumerate()
            .map(|(a, &(tail, head))| {
                f.add(tail, head, self.cost[a][c] + CAPACITY * i64::from(other[a]))
            })
            .collect();
        let mut outdeg = vec![0i64; nodes];
        let mut indeg = vec![0i64; nodes];
        for &(tail, head) in &self.arcs {
            outdeg[tail] += 1;
            indeg[head] += 1;
        }
        for v in 0..nodes {
            let b = self.supply[v][c];
            let pen = |g: i64| CONSERVE * (g - b) * (g - b);
            // g = out - in = (hub -> v) - (v -> hub): unit pieces of the convex penalty.
            for k in 1..=outdeg[v] {
                f.add(hub, v, pen(k) - pen(k - 1));
            }
            for k in 1..=indeg[v] {
                f.add(v, hub, pen(-k) - pen(-k + 1));
            }
        }
        f.solve(cancel)?;
        Ok(real.iter().map(|&k| u8::from(f.cap[k] == 0)).collect())
    }

    /// Single-flip descent on the full Ising model: sweep the spins in order
    /// and flip each one whose flip lowers the energy, until a sweep moves
    /// none. Returns the number of flips.
    ///
    /// # Errors
    /// [`RoutingError::Cancelled`], checked before each sweep.
    pub fn descend(&self, s: &mut [i8], cancel: Guard<'_>) -> Result<u64, RoutingError> {
        let mut flips = 0u64;
        loop {
            check(cancel)?;
            let mut moved = false;
            for i in 0..self.n {
                let g: i64 = self.h[i]
                    + self.adj[i]
                        .iter()
                        .map(|&(v, w)| w * i64::from(s[v]))
                        .sum::<i64>();
                if i64::from(s[i]) * g > 0 {
                    s[i] = -s[i];
                    flips += 1;
                    moved = true;
                }
            }
            if !moved {
                return Ok(flips);
            }
        }
    }

    /// The final state of one start order: alternate exact responses from
    /// commodity `first`, then descend.
    ///
    /// # Errors
    /// Any [`RoutingError`] of a response or the descent.
    pub fn candidate(&self, first: usize, cancel: Guard<'_>) -> Result<Candidate, RoutingError> {
        let zeros = vec![0u8; self.arcs.len()];
        let mut x = [zeros.clone(), zeros];
        let mut last = i64::MAX;
        let mut c = first;
        let mut solves = 0;
        let mut settled = false;
        for _ in 0..MAX_ALTERNATIONS {
            check(cancel)?;
            x[c] = self.solve_commodity(c, &x[1 - c], cancel)?;
            solves += 1;
            let e = self.energy(&self.to_spins(&x));
            if c != first && e >= last {
                settled = true;
                break;
            }
            last = last.min(e);
            c = 1 - c;
        }
        let mut spins = self.to_spins(&x);
        let before_descent = self.energy(&spins);
        let flips = self.descend(&mut spins, cancel)?;
        Ok(Candidate {
            first,
            energy: self.energy(&spins),
            spins,
            solves,
            settled,
            before_descent,
            flips,
        })
    }
}

fn spin_of(bit: u8) -> i8 {
    1 - 2 * i8::from(bit != 0)
}

fn check(cancel: Guard<'_>) -> Result<(), RoutingError> {
    if cancel.is_some_and(|(token, watermark)| token.is_cancelled(watermark)) {
        Err(RoutingError::Cancelled)
    } else {
        Ok(())
    }
}

/// Residual graph with unit-capacity arcs stored in pairs: arc `k` and its
/// reverse `k ^ 1`.
struct Flow {
    head: Vec<usize>,
    cap: Vec<i32>,
    cost: Vec<i64>,
    out: Vec<Vec<usize>>,
}

impl Flow {
    fn new(nodes: usize) -> Self {
        Self {
            head: Vec::new(),
            cap: Vec::new(),
            cost: Vec::new(),
            out: vec![Vec::new(); nodes],
        }
    }

    fn add(&mut self, u: usize, v: usize, cost: i64) -> usize {
        let k = self.head.len();
        self.head.extend([v, u]);
        self.cap.extend([1, 0]);
        self.cost.extend([cost, -cost]);
        self.out[u].push(k);
        self.out[v].push(k + 1);
        k
    }

    /// Minimum-cost circulation: saturate every negative arc, then route the
    /// imbalances along shortest paths with Dijkstra on reduced costs. The
    /// cancel token is checked before each shortest path.
    fn solve(&mut self, cancel: Guard<'_>) -> Result<(), RoutingError> {
        let nn = self.out.len();
        let mut excess = vec![0i64; nn];
        for k in (0..self.head.len()).step_by(2) {
            if self.cost[k] < 0 {
                self.cap[k] = 0;
                self.cap[k + 1] = 1;
                excess[self.head[k]] += 1;
                excess[self.head[k + 1]] -= 1;
            }
        }
        let mut pot = vec![0i64; nn];
        let mut dist = vec![i64::MAX; nn];
        let mut prev = vec![usize::MAX; nn];
        while excess.iter().any(|&x| x > 0) {
            check(cancel)?;
            dist.fill(i64::MAX);
            prev.fill(usize::MAX);
            let mut heap = BinaryHeap::new();
            for v in 0..nn {
                if excess[v] > 0 {
                    dist[v] = 0;
                    heap.push(Reverse((0i64, v)));
                }
            }
            let mut target = usize::MAX;
            while let Some(Reverse((d, u))) = heap.pop() {
                if d > dist[u] {
                    continue;
                }
                if excess[u] < 0 {
                    target = u;
                    break;
                }
                for &k in &self.out[u] {
                    if self.cap[k] == 0 {
                        continue;
                    }
                    let v = self.head[k];
                    let nd = d + self.cost[k] + pot[u] - pot[v];
                    if nd < dist[v] {
                        dist[v] = nd;
                        prev[v] = k;
                        heap.push(Reverse((nd, v)));
                    }
                }
            }
            if target == usize::MAX {
                return Err(RoutingError::NoPath);
            }
            let dt = dist[target];
            for v in 0..nn {
                pot[v] += dist[v].min(dt);
            }
            let mut v = target;
            while prev[v] != usize::MAX {
                let k = prev[v];
                self.cap[k] -= 1;
                self.cap[k ^ 1] += 1;
                v = self.head[k ^ 1];
            }
            excess[v] -= 1;
            excess[target] += 1;
        }
        Ok(())
    }
}

/// One arc of the metadata section.
struct ArcRecord {
    tail: u32,
    head: u32,
    x1: u32,
    chain: [u32; 2],
    cost: [i64; 2],
}

/// A cursor over little-endian fields.
struct Cursor<'a>(&'a [u8]);

impl<'a> Cursor<'a> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], RoutingError> {
        let (head, rest) = self
            .0
            .split_first_chunk::<N>()
            .ok_or_else(|| invalid("the routing section is truncated"))?;
        self.0 = rest;
        Ok(*head)
    }

    fn u16(&mut self) -> Result<u16, RoutingError> {
        Ok(u16::from_le_bytes(self.take()?))
    }

    fn u32(&mut self) -> Result<u32, RoutingError> {
        Ok(u32::from_le_bytes(self.take()?))
    }

    fn i8(&mut self) -> Result<i8, RoutingError> {
        Ok(i8::from_le_bytes(self.take()?))
    }

    fn rest(&self) -> &'a [u8] {
        self.0
    }
}

/// The payload of the one section of a metadata region: a `u16` count of 1,
/// a `u16` kind, a `u32` length, and the payload, with no byte after it.
fn only_section(metadata: &[u8]) -> Result<&[u8], RoutingError> {
    let mut cursor = Cursor(metadata);
    if cursor.u16()? != 1 {
        return Err(invalid(
            "the metadata region does not hold exactly one section",
        ));
    }
    let kind = cursor.u16()?;
    if kind != ROUTING_SECTION {
        return Err(invalid(format!("unknown metadata section kind {kind}")));
    }
    let len = usize::try_from(cursor.u32()?).map_err(|_| invalid("the section is too long"))?;
    let rest = cursor.rest();
    if rest.len() != len {
        return Err(invalid("the section length disagrees with the region"));
    }
    Ok(rest)
}

/// The node supplies of the section, and the bytes after them.
fn decode_nodes(payload: &[u8]) -> Result<(Vec<[i64; 2]>, &[u8]), RoutingError> {
    let mut cursor = Cursor(payload);
    let count = usize::try_from(cursor.u32()?).map_err(|_| invalid("too many nodes"))?;
    if count == 0 || count > MAX_SPINS || cursor.rest().len() < count.saturating_mul(NODE_RECORD) {
        return Err(invalid("the node count disagrees with the section"));
    }
    let mut supply = Vec::with_capacity(count);
    for _ in 0..count {
        cursor.u16()?;
        cursor.u16()?;
        supply.push([i64::from(cursor.i8()?), i64::from(cursor.i8()?)]);
    }
    Ok((supply, cursor.rest()))
}

/// The arcs and the chain strength, with no byte left over.
fn decode_arcs(rest: &[u8]) -> Result<(Vec<ArcRecord>, u16), RoutingError> {
    let mut cursor = Cursor(rest);
    let count = usize::try_from(cursor.u32()?).map_err(|_| invalid("too many arcs"))?;
    let wanted = count.saturating_mul(ARC_RECORD).saturating_add(2);
    if count == 0 || count > MAX_ARCS || cursor.rest().len() != wanted {
        return Err(invalid("the arc count disagrees with the section"));
    }
    let mut arcs = Vec::with_capacity(count);
    for _ in 0..count {
        let tail = cursor.u32()?;
        let head = cursor.u32()?;
        let x1 = cursor.u32()?;
        let chain = [cursor.u32()?, cursor.u32()?];
        let cost = [i64::from(cursor.u16()?), i64::from(cursor.u16()?)];
        if !cost.iter().all(|c| COSTS.contains(c)) {
            return Err(invalid("an arc cost is not one the family draws"));
        }
        arcs.push(ArcRecord {
            tail,
            head,
            x1,
            chain,
            cost,
        });
    }
    let strength = cursor.u16()?;
    Ok((arcs, strength))
}
