//! The cluster update and the single-spin sweep.
//!
//! A cluster sweep draws auxiliary bonds on every tile and pair, so the strong
//! layer is sampled exactly, then flips each connected component by heat bath
//! against the weak layer. The tile bond probabilities follow Otsuka,
//! Phys. Rev. B 90, 220406 (2014).

use rand::rngs::SmallRng;
use rand::Rng;
use std::fmt;

#[cfg(feature = "packed-residual-fields")]
use super::graph::FusedCsr;
use super::graph::{OtsukaGraph, Weights};

pub(super) type LocalField = i64;

// With unique, non-self edges, zero fields, and |J| <= 1000, all local
// field sums fit i32. The full energy and its changes still require i64.
#[cfg(feature = "packed-residual-fields")]
const _: () = assert!(
    (super::graph::MAX_NODES as i64 - 1) * super::graph::STRONG_MILLI as i64 <= i32::MAX as i64
);

#[inline(always)]
fn field_energy(value: LocalField) -> i64 {
    if cfg!(feature = "packed-residual-fields") {
        value >> 32
    } else {
        value
    }
}

#[cfg(feature = "packed-residual-fields")]
const RESIDUAL_FIELD_BIAS: i64 = 1 << 31;

#[cfg(feature = "packed-residual-fields")]
#[inline]
fn residual_field(value: LocalField) -> i64 {
    i64::from(value as u32) - RESIDUAL_FIELD_BIAS
}

type CacheEpoch = u32;

const END_OF_COMPONENT: u32 = u32::MAX;

/// Bond probabilities at one inverse temperature.
#[derive(Clone, Copy, Debug)]
struct Breakup {
    /// An ice tile (two spins up in the gauge) takes no bond below this.
    zero_ice: f64,
    /// An ice tile takes one bond below this and two above it.
    single_ice: f64,
    /// A defect tile (one or three up) takes no bond below this.
    zero_defect: f64,
    /// An aligned pair takes its bond below this.
    pair_link: f64,
}

impl Breakup {
    fn new(beta: f64, weights: Weights) -> Self {
        let tile_beta = beta * f64::from(weights.tile_milli) / 1000.0;
        let z = (-2.0 * tile_beta).exp();
        let w0 = z.powi(4);
        let w1 = (z - w0) / 3.0;
        Self {
            zero_ice: w0,
            single_ice: w0 + 4.0 * w1,
            zero_defect: (-6.0 * tile_beta).exp(),
            pair_link: -(-2.0 * beta * f64::from(weights.pair_milli) / 1000.0).exp_m1(),
        }
    }
}

#[cfg(feature = "table-tile-bonds")]
#[derive(Clone, Copy)]
struct TilePattern {
    up: [u8; 4],
    down: [u8; 4],
    ups: u8,
}

/// Positions within the tile in the same order as the scalar scan.
#[cfg(feature = "table-tile-bonds")]
const TILE_PATTERNS: [TilePattern; 16] = {
    let mut patterns = [TilePattern {
        up: [0; 4],
        down: [0; 4],
        ups: 0,
    }; 16];
    let mut mask = 0;
    while mask < patterns.len() {
        let mut ups = 0;
        let mut downs = 0;
        let mut slot = 0;
        while slot < 4 {
            if mask & (1 << slot) != 0 {
                patterns[mask].up[ups] = slot;
                ups += 1;
            } else {
                patterns[mask].down[downs] = slot;
                downs += 1;
            }
            slot += 1;
        }
        patterns[mask].ups = ups as u8;
        mask += 1;
    }
    patterns
};

/// Counters for the temperature-local transition cache.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheStats {
    /// Heat-bath lookups served without evaluating an exponential.
    pub heat_hits: u64,
    /// Heat-bath probabilities computed and retained.
    pub heat_misses: u64,
    /// Metropolis lookups served from the shared table.
    pub metropolis_hits: u64,
    /// Metropolis probabilities computed and retained.
    pub metropolis_misses: u64,
    /// Energy changes outside the bounded table, evaluated directly.
    pub fallbacks: u64,
    /// Temperature or bond-weight changes that rebuilt the breakup values.
    pub temperatures: u64,
    /// Stamp wraps that cleared the table before an epoch was reused.
    pub epoch_wraps: u64,
}

/// One bounded probability table shared by the replicas at a temperature.
///
/// Integer energy changes index the table without hash collisions. Epoch
/// tags invalidate entries when beta changes, without clearing every slot.
/// The scalar expressions and random draws remain identical to the stock
/// update. Large or odd energy changes take the scalar path.
#[derive(Debug)]
pub struct SweepCache {
    beta: f64,
    weights: Weights,
    breakup: Breakup,
    epoch: CacheEpoch,
    slots: usize,
    heat: Vec<f64>,
    heat_epoch: Vec<CacheEpoch>,
    metropolis: Vec<f64>,
    metropolis_epoch: Vec<CacheEpoch>,
    stats: CacheStats,
}

impl SweepCache {
    /// Allocate a bounded table. `slots` is clamped to `1..=65536`.
    /// Heat-bath entries cover signed half-deltas; Metropolis entries
    /// cover positive half-deltas. No graph-size allocation is required.
    #[must_use]
    pub fn new(slots: usize) -> Self {
        let slots = slots.clamp(1, 65_536);
        Self {
            beta: f64::NAN,
            weights: Weights::FULL,
            breakup: Breakup::new(1.0, Weights::FULL),
            epoch: 0,
            slots,
            heat: vec![0.0; 2 * slots + 1],
            heat_epoch: vec![0; 2 * slots + 1],
            metropolis: vec![0.0; slots + 1],
            metropolis_epoch: vec![0; slots + 1],
            stats: CacheStats::default(),
        }
    }

    /// Aggregate lookup counters for this table's lifetime.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Bytes occupied by table payloads, excluding the small struct.
    #[must_use]
    pub fn table_bytes(&self) -> usize {
        (self.heat.len() + self.metropolis.len()) * (size_of::<f64>() + size_of::<CacheEpoch>())
    }

    fn prepare(&mut self, beta: f64, weights: Weights) {
        if beta.to_bits() == self.beta.to_bits() && weights == self.weights {
            return;
        }
        self.beta = beta;
        self.weights = weights;
        self.breakup = Breakup::new(beta, weights);
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.heat_epoch.fill(0);
            self.metropolis_epoch.fill(0);
            self.epoch = 1;
            self.stats.epoch_wraps += 1;
        }
        self.stats.temperatures += 1;
    }

    fn heat_probability(&mut self, delta: i64) -> f64 {
        let slot = delta / 2 + self.slots as i64;
        if delta % 2 != 0 || !(0..self.heat.len() as i64).contains(&slot) {
            self.stats.fallbacks += 1;
            return heat_probability(self.beta * delta as f64 / 1000.0);
        }
        let slot = slot as usize;
        if self.heat_epoch[slot] == self.epoch {
            self.stats.heat_hits += 1;
            return self.heat[slot];
        }
        let probability = heat_probability(self.beta * delta as f64 / 1000.0);
        self.heat[slot] = probability;
        self.heat_epoch[slot] = self.epoch;
        self.stats.heat_misses += 1;
        probability
    }

    fn metropolis_probability(&mut self, delta: i64) -> f64 {
        let slot = delta / 2;
        if delta % 2 != 0 || !(0..self.metropolis.len() as i64).contains(&slot) {
            self.stats.fallbacks += 1;
            return (-self.beta * delta as f64 / 1000.0).exp();
        }
        let slot = slot as usize;
        if self.metropolis_epoch[slot] == self.epoch {
            self.stats.metropolis_hits += 1;
            return self.metropolis[slot];
        }
        let probability = (-self.beta * delta as f64 / 1000.0).exp();
        self.metropolis[slot] = probability;
        self.metropolis_epoch[slot] = self.epoch;
        self.stats.metropolis_misses += 1;
        probability
    }

    fn heat_bath(&mut self, delta: i64, rng: &mut SmallRng) -> bool {
        rng.gen::<f64>() < self.heat_probability(delta)
    }
}

/// Union-find and component lists, reused across sweeps. Components are
/// numbered densely in the order of their smallest node.
#[derive(Debug)]
struct Workspace {
    parent: Vec<u32>,
    /// The component of each node.
    label: Vec<u32>,
    /// Offsets into `members`, or heads with `linked-component-lists`.
    start: Vec<u32>,
    /// Consecutive nodes, or a next-node index for each node with linked lists.
    members: Vec<u32>,
    /// The weak-energy change of flipping each component.
    delta: Vec<i64>,
    /// Boltzmann factors of the current Metropolis sweep.
    boltzmann: Boltzmann,
    /// The full local field, or full and residual fields in one word.
    field: Vec<LocalField>,
}

/// A direct-mapped cache of `exp(-beta * delta / 1000)` at one beta. The
/// moves of a sweep take few distinct energy changes, so most lookups hit,
/// and a hit returns the same value the call would.
#[derive(Debug)]
struct Boltzmann {
    keys: [i64; Self::SLOTS],
    values: [f64; Self::SLOTS],
}

impl Boltzmann {
    const SLOTS: usize = 256;

    fn new() -> Self {
        Self {
            keys: [i64::MIN; Self::SLOTS],
            values: [0.0; Self::SLOTS],
        }
    }

    /// Forget every entry, before a sweep at a new beta.
    fn clear(&mut self) {
        self.keys.fill(i64::MIN);
    }

    fn get(&mut self, delta: i64, beta: f64) -> f64 {
        // Energy changes are even, so the low bit carries no information.
        let slot = ((delta >> 1) as usize) % Self::SLOTS;
        if self.keys[slot] != delta {
            self.keys[slot] = delta;
            self.values[slot] = (-beta * delta as f64 / 1000.0).exp();
        }
        self.values[slot]
    }
}

impl Workspace {
    fn new(n: usize) -> Self {
        Self {
            parent: vec![0; n],
            label: vec![0; n],
            start: Vec::with_capacity(n + 1),
            members: vec![0; n],
            delta: Vec::with_capacity(n),
            boltzmann: Boltzmann::new(),
            field: vec![0; n],
        }
    }

    /// Recompute every local field from `s`.
    fn set_fields(&mut self, graph: &OtsukaGraph, s: &[i8]) {
        #[cfg(feature = "packed-residual-fields")]
        {
            let rows = graph.fused();
            for (u, field) in self.field.iter_mut().enumerate() {
                *field = rows
                    .row(u)
                    .iter()
                    .map(|&link| link.field_adjustment() * i64::from(s[link.node()]))
                    .fold(RESIDUAL_FIELD_BIAS, i64::wrapping_add);
            }
        }
        #[cfg(not(feature = "packed-residual-fields"))]
        for (u, field) in self.field.iter_mut().enumerate() {
            *field = graph
                .all
                .row(u)
                .iter()
                .map(|&link| i64::from(link.j()) * LocalField::from(s[link.node()]))
                .sum();
        }
    }

    /// Flip node `u`, patch the local fields of its neighbours, and return
    /// the model energy change in milli.
    #[inline]
    fn flip(&mut self, graph: &OtsukaGraph, s: &mut [i8], u: usize) -> i64 {
        #[cfg(feature = "packed-residual-fields")]
        {
            Self::flip_packed(graph.fused(), &mut self.field, s, u)
        }
        #[cfg(not(feature = "packed-residual-fields"))]
        {
            let twice = 2 * LocalField::from(s[u]);
            for &link in graph.all.row(u) {
                self.field[link.node()] -= twice * i64::from(link.j());
            }
            s[u] = -s[u];
            -field_energy(twice) * field_energy(self.field[u])
        }
    }

    #[cfg(feature = "packed-residual-fields")]
    #[inline]
    fn flip_packed(rows: &FusedCsr, fields: &mut [LocalField], s: &mut [i8], u: usize) -> i64 {
        let twice = 2 * i64::from(s[u]);
        for &link in rows.row(u) {
            let field = &mut fields[link.node()];
            // The completed residual field fits i32, so its biased value
            // fits the low lane and never carries into the full field.
            *field = field.wrapping_sub(twice * link.field_adjustment());
        }
        s[u] = -s[u];
        -twice * field_energy(fields[u])
    }

    fn find(&mut self, mut u: u32) -> u32 {
        loop {
            let up = self.parent[u as usize];
            if up == u {
                return u;
            }
            let grand = self.parent[up as usize];
            self.parent[u as usize] = grand;
            u = grand;
        }
    }

    /// Join the sets of `a` and `b`. The smaller root becomes the parent,
    /// so `parent[u] <= u` holds for every node, which the labelling in
    /// [`Self::partition`] relies on.
    fn join(&mut self, a: u32, b: u32) {
        let a = self.find(a);
        let b = self.find(b);
        if a < b {
            self.parent[b as usize] = a;
        } else if b < a {
            self.parent[a as usize] = b;
        }
    }

    #[inline(always)]
    fn find_inline(&mut self, mut u: u32) -> u32 {
        loop {
            let up = self.parent[u as usize];
            if !cfg!(feature = "shallow-union-roots") && up == u {
                return u;
            }
            let grand = self.parent[up as usize];
            if cfg!(feature = "shallow-union-roots") && grand == up {
                return up;
            }
            self.parent[u as usize] = grand;
            u = grand;
        }
    }

    #[inline(always)]
    fn join_selected<const INLINE: bool>(&mut self, a: u32, b: u32) {
        if !INLINE {
            self.join(a, b);
            return;
        }
        let a = self.find_inline(a);
        let b = self.find_inline(b);
        if cfg!(feature = "shallow-union-roots") {
            // A repeated root writes its own index back unchanged.
            self.parent[a.max(b) as usize] = a.min(b);
        } else if a < b {
            self.parent[b as usize] = a;
        } else if b < a {
            self.parent[a as usize] = b;
        }
    }

    /// Draw the bonds of every tile and pair, then list the components.
    fn partition<const INLINE: bool>(
        &mut self,
        graph: &OtsukaGraph,
        s: &[i8],
        p: &Breakup,
        rng: &mut SmallRng,
    ) {
        for (u, parent) in self.parent.iter_mut().enumerate() {
            *parent = u as u32;
        }
        for tile in &graph.tiles {
            #[cfg(not(feature = "table-tile-bonds"))]
            let (up, down, ups, downs) = {
                let mut up = [0; 4];
                let mut down = [0; 4];
                let (mut ups, mut downs) = (0, 0);
                for k in 0..4 {
                    if s[tile.nodes[k] as usize] * tile.gauge[k] > 0 {
                        up[ups] = tile.nodes[k];
                        ups += 1;
                    } else {
                        down[downs] = tile.nodes[k];
                        downs += 1;
                    }
                }
                (
                    move |slot: usize| up[slot],
                    move |slot: usize| down[slot],
                    ups,
                    downs,
                )
            };
            #[cfg(feature = "table-tile-bonds")]
            let (up, down, ups, downs) = {
                let [a, b, c, d] = tile.nodes.map(|node| s[node as usize]);
                let [ga, gb, gc, gd] = tile.gauge;
                // Spins and gauges are validated signs. Matching signs are up.
                let mask = usize::from(a == ga)
                    | (usize::from(b == gb) << 1)
                    | (usize::from(c == gc) << 2)
                    | (usize::from(d == gd) << 3);
                let pattern = &TILE_PATTERNS[mask];
                let ups = usize::from(pattern.ups);
                (
                    move |slot: usize| tile.nodes[usize::from(pattern.up[slot])],
                    move |slot: usize| tile.nodes[usize::from(pattern.down[slot])],
                    ups,
                    4 - ups,
                )
            };
            if ups == 0 || ups == 4 {
                continue;
            }
            // One draw per tile: the top 53 bits make the uniform, and the
            // two lowest bits pick the spins of an ice tile's bonds.
            let x = rng.gen::<u64>();
            let r = (x >> 11) as f64 * (1.0 / (1_u64 << 53) as f64);
            if ups == 2 {
                if r < p.zero_ice {
                    continue;
                }
                if r < p.single_ice {
                    self.join_selected::<INLINE>(
                        up((x & 1) as usize),
                        down(((x >> 1) & 1) as usize),
                    );
                } else {
                    let c = (x & 1) as usize;
                    self.join_selected::<INLINE>(up(0), down(c));
                    self.join_selected::<INLINE>(up(1), down(1 - c));
                }
            } else if r >= p.zero_defect {
                // The source took the partner from the low bits modulo three,
                // which favours the first. A separate range draw is uniform.
                self.join_selected::<INLINE>(
                    up(rng.gen_range(0..ups)),
                    down(rng.gen_range(0..downs)),
                );
            }
        }
        for &(u, v) in &graph.pairs {
            if s[u as usize] == s[v as usize] && rng.gen::<f64>() < p.pair_link {
                self.join_selected::<INLINE>(u, v);
            }
        }
        // Number the components, count their nodes, then list the nodes.
        // A join hooks the larger root under the smaller, and path halving
        // only moves a node closer to its root, so `parent[u] <= u` holds
        // and every root is the smallest node of its component. One pass in
        // node order then numbers the components by their smallest node and
        // finds each label at the parent, which the pass has already seen.
        // Linked lists prepend later members and preserve the component order.
        let linked = cfg!(feature = "linked-component-lists");
        self.start.clear();
        if !linked {
            self.start.push(0);
        }
        for u in 0..self.parent.len() {
            let up = self.parent[u] as usize;
            let c = if up == u {
                if linked {
                    let c = self.start.len() as u32;
                    self.start.push(u as u32);
                    self.members[u] = END_OF_COMPONENT;
                    c
                } else {
                    self.start.push(0);
                    (self.start.len() - 2) as u32
                }
            } else {
                let c = self.label[up];
                if linked {
                    self.members[u] = std::mem::replace(&mut self.start[c as usize], u as u32);
                }
                c
            };
            self.label[u] = c;
            if !linked {
                self.start[c as usize + 1] += 1;
            }
        }
        if linked {
            self.delta.clear();
            self.delta.resize(self.start.len(), 0);
            return;
        }
        for c in 1..self.start.len() {
            self.start[c] += self.start[c - 1];
        }
        let components = self.start.len() - 1;
        // `delta` doubles as the fill cursor of each component's list.
        self.delta.clear();
        self.delta
            .extend(self.start[..components].iter().map(|&at| i64::from(at)));
        for (u, &c) in self.label.iter().enumerate() {
            let at = &mut self.delta[c as usize];
            self.members[*at as usize] = u as u32;
            *at += 1;
        }
    }

    #[inline]
    fn component_range(&self, c: usize) -> (u32, u32) {
        let end = if cfg!(feature = "linked-component-lists") {
            END_OF_COMPONENT
        } else {
            self.start[c + 1]
        };
        (self.start[c], end)
    }

    #[inline]
    fn next_member(&self, at: &mut u32) -> usize {
        if cfg!(feature = "linked-component-lists") {
            let node = *at as usize;
            *at = self.members[node];
            node
        } else {
            let node = self.members[*at as usize] as usize;
            *at += 1;
            node
        }
    }

    /// One cluster sweep. Returns the model energy change in milli.
    fn cluster_sweep(
        &mut self,
        graph: &OtsukaGraph,
        s: &mut [i8],
        beta: f64,
        rng: &mut SmallRng,
    ) -> i64 {
        let p = Breakup::new(beta, graph.weights);
        self.cluster_sweep_using::<false, false>(graph, s, &p, rng, |d, rng| {
            heat_bath(beta * d as f64 / 1000.0, rng)
        })
    }

    #[cfg(not(feature = "packed-residual-fields"))]
    fn cluster_sweep_using<const INLINE: bool, const FUSED: bool>(
        &mut self,
        graph: &OtsukaGraph,
        s: &mut [i8],
        p: &Breakup,
        rng: &mut SmallRng,
        mut draw: impl FnMut(i64, &mut SmallRng) -> bool,
    ) -> i64 {
        self.partition::<INLINE>(graph, s, p, rng);
        let fused = FUSED.then(|| graph.fused());
        // delta[c] = -2 * sum over the weak edges that leave c of J s_u s_v.
        if !cfg!(feature = "linked-component-lists") {
            self.delta.fill(0);
        }
        #[cfg(not(feature = "row-grouped-residual-deltas"))]
        {
            let (label, delta) = (&self.label, &mut self.delta);
            for &(u, link) in &graph.weak_edges {
                let v = link.node();
                let (lu, lv) = (label[u as usize], label[v]);
                if lu != lv {
                    let t = -2 * i64::from(link.j()) * i64::from(s[u as usize] * s[v]);
                    delta[lu as usize] += t;
                    delta[lv as usize] += t;
                }
            }
        }
        #[cfg(feature = "row-grouped-residual-deltas")]
        {
            let rows = graph.weak_forward();
            let (label, delta) = (&self.label, &mut self.delta);
            for (u, &spin) in s.iter().enumerate() {
                let row = rows.row(u);
                if row.is_empty() {
                    continue;
                }
                let lu = label[u] as usize;
                let factor = -2 * i32::from(spin);
                let mut row_delta = 0_i64;
                for &link in row {
                    let v = link.node();
                    let lv = label[v] as usize;
                    let outside = lu != lv;
                    if !cfg!(feature = "masked-residual-deltas") && !outside {
                        continue;
                    }
                    let product = factor * link.j() * i32::from(s[v]);
                    let change = if cfg!(feature = "masked-residual-deltas") {
                        i64::from(product & -i32::from(outside))
                    } else {
                        i64::from(product)
                    };
                    row_delta += change;
                    delta[lv] += change;
                }
                delta[lu] += row_delta;
            }
        }
        let mut change = 0;
        // Heat bath in the order of the components' smallest nodes. A flip
        // changes the sign of every edge that leaves the component, so the
        // deltas of its neighbours are patched before the next one draws.
        for c in 0..self.delta.len() {
            let d = self.delta[c];
            if !draw(d, rng) {
                continue;
            }
            let (mut at, end) = self.component_range(c);
            while at != end {
                let u = self.next_member(&mut at);
                let su = i64::from(s[u]);
                if let Some(rows) = fused {
                    let twice = 2 * su;
                    change -= twice * field_energy(self.field[u]);
                    for &link in rows.row(u) {
                        let v = link.node();
                        let residual = link.residual();
                        let lv = self.label[v] as usize;
                        if residual != 0 && lv != c {
                            self.delta[lv] += 4 * i64::from(residual) * su * i64::from(s[v]);
                        }
                        self.field[v] -= 2 * LocalField::from(s[u]) * i64::from(link.full());
                    }
                    s[u] = -s[u];
                } else {
                    for &link in graph.weak.row(u) {
                        let v = link.node();
                        let lv = self.label[v] as usize;
                        if lv != c {
                            self.delta[lv] += 4 * i64::from(link.j()) * su * i64::from(s[v]);
                        }
                    }
                    change += self.flip(graph, s, u);
                }
            }
            self.delta[c] = -d;
        }
        change
    }

    #[cfg(feature = "packed-residual-fields")]
    fn cluster_sweep_using<const INLINE: bool, const FUSED: bool>(
        &mut self,
        graph: &OtsukaGraph,
        s: &mut [i8],
        p: &Breakup,
        rng: &mut SmallRng,
        mut draw: impl FnMut(i64, &mut SmallRng) -> bool,
    ) -> i64 {
        self.partition::<INLINE>(graph, s, p, rng);
        let rows = graph.fused();
        let forward = graph.weak_forward();
        let tile_energy = i64::from(graph.weights.tile_milli - super::graph::STRONG_MILLI);
        let pair_energy = i64::from(graph.weights.pair_milli - super::graph::STRONG_MILLI);
        let mut change = 0;
        for c in 0..self.delta.len() {
            let (start, end) = self.component_range(c);
            let mut at = start;
            let first = self.next_member(&mut at);
            if at == end {
                let delta = -2 * i64::from(s[first]) * residual_field(self.field[first]);
                if draw(delta, rng) {
                    change += Self::flip_packed(rows, &mut self.field, s, first);
                }
                continue;
            }
            let second = self.next_member(&mut at);
            if at == end {
                // Exactly one sampled bond connected these two nodes. A
                // dimer joins equal spins and a tile joins opposite gauged
                // spins, so either bond's residual energy is fixed.
                let bond_energy = if graph.pair_partner[first] == second as u32 {
                    pair_energy
                } else {
                    tile_energy
                };
                let field_sum = i64::from(s[first]) * residual_field(self.field[first])
                    + i64::from(s[second]) * residual_field(self.field[second]);
                if draw(4 * bond_energy - 2 * field_sum, rng) {
                    change += Self::flip_packed(rows, &mut self.field, s, first);
                    change += Self::flip_packed(rows, &mut self.field, s, second);
                }
                continue;
            }

            // Summed residual fields count each interior edge twice. Restore
            // that contribution once per interior edge using the forward rows.
            let mut delta = 0;
            let mut at = start;
            while at != end {
                let u = self.next_member(&mut at);
                let mut interior = 0_i32;
                for &link in forward.row(u) {
                    let v = link.node();
                    let inside = -i32::from(self.label[v] as usize == c);
                    interior += (link.j() * i32::from(s[v])) & inside;
                }
                delta +=
                    i64::from(s[u]) * (4 * i64::from(interior) - 2 * residual_field(self.field[u]));
            }
            if draw(delta, rng) {
                let mut at = start;
                while at != end {
                    let u = self.next_member(&mut at);
                    change += Self::flip_packed(rows, &mut self.field, s, u);
                }
            }
        }
        change
    }
}

/// Accept a move that costs `x = beta * delta` with probability
/// `1 / (1 + e^x)`, written so neither branch overflows.
fn heat_bath(x: f64, rng: &mut SmallRng) -> bool {
    if x <= 0.0 {
        rng.gen::<f64>() < 1.0 / (1.0 + x.exp())
    } else {
        let e = (-x).exp();
        rng.gen::<f64>() < e / (1.0 + e)
    }
}

fn heat_probability(x: f64) -> f64 {
    if x <= 0.0 {
        1.0 / (1.0 + x.exp())
    } else {
        let e = (-x).exp();
        e / (1.0 + e)
    }
}

/// Why a spin state does not fit a graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpinError {
    /// The state has a different number of spins than the graph has nodes.
    Length,
    /// A spin is not `1` or `-1`.
    Value {
        /// The node.
        node: usize,
    },
}

impl fmt::Display for SpinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length => write!(f, "the spin count differs from the node count"),
            Self::Value { node } => write!(f, "spin {node} is not 1 or -1"),
        }
    }
}

impl std::error::Error for SpinError {}

fn check_spins(graph: &OtsukaGraph, spins: &[i8]) -> Result<(), SpinError> {
    if spins.len() != graph.n {
        return Err(SpinError::Length);
    }
    match spins.iter().position(|&x| x != 1 && x != -1) {
        Some(node) => Err(SpinError::Value { node }),
        None => Ok(()),
    }
}

/// A spin state on one graph with the workspace its moves reuse. Each move
/// takes its bond probabilities from the graph's own weights.
#[derive(Debug)]
pub struct Chain<'g> {
    graph: &'g OtsukaGraph,
    spins: Vec<i8>,
    work: Workspace,
}

impl<'g> Chain<'g> {
    /// A chain on `graph` that starts at `spins`.
    ///
    /// # Errors
    ///
    /// [`SpinError`] when `spins` does not fit the graph.
    pub fn new(graph: &'g OtsukaGraph, spins: Vec<i8>) -> Result<Self, SpinError> {
        check_spins(graph, &spins)?;
        let mut work = Workspace::new(graph.n);
        work.set_fields(graph, &spins);
        Ok(Self { graph, spins, work })
    }

    /// A chain on `graph` at a uniformly random state drawn from `rng`.
    pub(crate) fn random(graph: &'g OtsukaGraph, rng: &mut SmallRng) -> Self {
        let spins = (0..graph.n)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect::<Vec<i8>>();
        let mut work = Workspace::new(graph.n);
        work.set_fields(graph, &spins);
        Self { graph, spins, work }
    }

    /// Move the chain to `spins`.
    ///
    /// # Errors
    ///
    /// [`SpinError`] when `spins` does not fit the graph. The state is then
    /// unchanged.
    pub fn reset(&mut self, spins: &[i8]) -> Result<(), SpinError> {
        check_spins(self.graph, spins)?;
        self.spins.copy_from_slice(spins);
        self.work.set_fields(self.graph, spins);
        Ok(())
    }

    /// The current state.
    #[must_use]
    pub fn spins(&self) -> &[i8] {
        &self.spins
    }

    /// One cluster sweep at `beta`, in inverse units of 1000 milli. `beta`
    /// must be finite and positive. Returns the model energy change in milli.
    pub fn cluster_sweep(&mut self, beta: f64, rng: &mut SmallRng) -> i64 {
        self.work
            .cluster_sweep(self.graph, &mut self.spins, beta, rng)
    }

    /// A cluster sweep using a probability table shared with other replicas.
    /// The table is prepared for this graph's weights and `beta` on demand.
    pub fn cluster_sweep_cached(
        &mut self,
        beta: f64,
        cache: &mut SweepCache,
        rng: &mut SmallRng,
    ) -> i64 {
        cache.prepare(beta, self.graph.weights);
        let breakup = cache.breakup;
        self.work.cluster_sweep_using::<false, false>(
            self.graph,
            &mut self.spins,
            &breakup,
            rng,
            |delta, rng| cache.heat_bath(delta, rng),
        )
    }

    /// A cluster sweep with independently selectable CPU update layouts.
    /// Component numbering, random draws, and full-model energy changes
    /// remain identical to the scalar path.
    pub fn cluster_sweep_optimized(
        &mut self,
        beta: f64,
        cache: Option<&mut SweepCache>,
        inline_union: bool,
        fused_updates: bool,
        rng: &mut SmallRng,
    ) -> i64 {
        macro_rules! run {
            ($inline:literal, $fused:literal) => {
                if let Some(cache) = cache {
                    cache.prepare(beta, self.graph.weights);
                    let p = cache.breakup;
                    self.work.cluster_sweep_using::<$inline, $fused>(
                        self.graph,
                        &mut self.spins,
                        &p,
                        rng,
                        |d, rng| cache.heat_bath(d, rng),
                    )
                } else {
                    let p = Breakup::new(beta, self.graph.weights);
                    self.work.cluster_sweep_using::<$inline, $fused>(
                        self.graph,
                        &mut self.spins,
                        &p,
                        rng,
                        |d, rng| heat_bath(beta * d as f64 / 1000.0, rng),
                    )
                }
            };
        }
        match (inline_union, fused_updates) {
            (false, false) => run!(false, false),
            (true, false) => run!(true, false),
            (false, true) => run!(false, true),
            (true, true) => run!(true, true),
        }
    }

    pub(super) fn save_state(&self, spins: &mut [i8], fields: &mut [LocalField]) {
        spins.copy_from_slice(&self.spins);
        fields.copy_from_slice(&self.work.field);
    }

    pub(super) fn restore_state(&mut self, spins: &[i8], fields: &[LocalField]) {
        self.spins.copy_from_slice(spins);
        self.work.field.copy_from_slice(fields);
    }

    /// Flip every node of `nodes` once. Returns the model energy change in
    /// milli.
    pub(crate) fn flip_nodes(&mut self, nodes: &[u32]) -> i64 {
        let (graph, s, work) = (self.graph, &mut self.spins, &mut self.work);
        nodes.iter().map(|&u| work.flip(graph, s, u as usize)).sum()
    }

    /// One Metropolis sweep over every node in index order at `beta`.
    /// Returns the model energy change in milli.
    pub fn single_sweep(&mut self, beta: f64, rng: &mut SmallRng) -> i64 {
        let (graph, s, work) = (self.graph, &mut self.spins, &mut self.work);
        let mut change = 0;
        work.boltzmann.clear();
        for u in 0..graph.n {
            let delta = -2 * i64::from(s[u]) * field_energy(work.field[u]);
            if delta <= 0 || rng.gen::<f64>() < work.boltzmann.get(delta, beta) {
                change += work.flip(graph, s, u);
            }
        }
        change
    }

    /// A Metropolis sweep using the population's shared probability table.
    /// It uses the same acceptance expression and random draws as the scalar
    /// sweep, including the unconditional acceptance of nonpositive deltas.
    pub fn single_sweep_cached(
        &mut self,
        beta: f64,
        cache: &mut SweepCache,
        rng: &mut SmallRng,
    ) -> i64 {
        cache.prepare(beta, self.graph.weights);
        let (graph, s, work) = (self.graph, &mut self.spins, &mut self.work);
        let mut change = 0;
        for u in 0..graph.n {
            let delta = -2 * i64::from(s[u]) * field_energy(work.field[u]);
            if delta <= 0 || rng.gen::<f64>() < cache.metropolis_probability(delta) {
                change += work.flip(graph, s, u);
            }
        }
        change
    }
}

#[cfg(all(test, feature = "table-tile-bonds"))]
#[test]
fn tile_table_preserves_position_order_for_every_pattern() {
    for (mask, pattern) in TILE_PATTERNS.iter().enumerate() {
        let (up, down): (Vec<_>, Vec<_>) = (0_u8..4).partition(|&slot| mask & (1 << slot) != 0);
        assert_eq!(usize::from(pattern.ups), up.len());
        assert_eq!(&pattern.up[..up.len()], up.as_slice());
        assert_eq!(&pattern.down[..down.len()], down.as_slice());
    }
}

#[cfg(all(test, feature = "packed-residual-fields"))]
#[test]
fn packed_fields_preserve_both_signed_sums_through_flips() {
    use super::graph::{Csr, Model, Motifs};

    let graph = OtsukaGraph::new(
        &Model {
            h_milli: &[0; 3],
            edges: &[(0, 1), (0, 2)],
            j_milli: &[-1000, 999],
        },
        &Motifs {
            tiles: &[],
            gauges: &[],
            pairs: &[(0, 1)],
        },
        Weights {
            pair_milli: 999,
            tile_milli: 1000,
        },
    )
    .unwrap();
    let mut spins = [1; 3];
    let mut work = Workspace::new(graph.n);
    work.set_fields(&graph, &spins);
    for flip in [1, 2, 0, 1, 2, 0] {
        for (u, &packed) in work.field.iter().enumerate() {
            let sum = |rows: &Csr| {
                rows.row(u)
                    .iter()
                    .map(|&link| i64::from(link.j()) * i64::from(spins[link.node()]))
                    .sum::<i64>()
            };
            assert_eq!(field_energy(packed), sum(&graph.all));
            assert_eq!(residual_field(packed), sum(&graph.weak));
        }
        let before = graph.energy(&spins);
        let change = work.flip(&graph, &mut spins, flip);
        assert_eq!(change, graph.energy(&spins) - before);
    }
}

#[cfg(all(test, feature = "packed-residual-fields"))]
#[test]
fn packed_heat_bath_deltas_match_rescoring_at_each_component() {
    use super::graph::{Model, Motifs};
    use rand::SeedableRng;

    let model = Model {
        h_milli: &[0; 7],
        edges: &[
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (5, 6),
            (0, 4),
            (1, 5),
            (2, 6),
            (0, 6),
            (2, 4),
        ],
        j_milli: &[
            -1000, 1000, -1000, -1000, 1000, -1000, -1000, -1000, -321, 200, -230, 43, 98,
        ],
    };
    let motifs = Motifs {
        tiles: &[[0, 1, 2, 3]],
        gauges: &[[1, -1, 1, -1]],
        pairs: &[(3, 4), (5, 6)],
    };
    for weights in [
        Weights::FULL,
        Weights {
            pair_milli: 825,
            tile_milli: 375,
        },
    ] {
        let graph = OtsukaGraph::new(&model, &motifs, weights).unwrap();
        let mut covered = [false; 4];
        for seed in 0..128 {
            let mut spins: [i8; 7] =
                std::array::from_fn(|i| if seed & (1 << i) == 0 { 1 } else { -1 });
            let mut expected = spins;
            let mut rng = SmallRng::seed_from_u64(seed);
            let weights_at_beta = Breakup::new([0.2, 1.0, 4.0, 8.0][seed as usize % 4], weights);
            let mut partition_rng = rng.clone();
            let mut partition = Workspace::new(graph.n);
            partition.partition::<true>(&graph, &spins, &weights_at_beta, &mut partition_rng);
            let mut work = Workspace::new(graph.n);
            work.set_fields(&graph, &spins);
            let before = graph.energy(&spins);
            let mut component = 0;
            let change = work.cluster_sweep_using::<true, true>(
                &graph,
                &mut spins,
                &weights_at_beta,
                &mut rng,
                |delta, _| {
                    let members: Vec<_> = (0..graph.n)
                        .filter(|&u| partition.label[u] == component)
                        .collect();
                    let kind = match members.as_slice() {
                        [_] => 0,
                        &[u, v] if graph.pair_partner[u] == v as u32 => 1,
                        [_, _] => 2,
                        _ => 3,
                    };
                    covered[kind] = true;
                    let mut flipped = expected;
                    for u in members {
                        flipped[u] = -flipped[u];
                    }
                    assert_eq!(
                        delta,
                        graph.weak_energy(&flipped) - graph.weak_energy(&expected)
                    );
                    let accept = (u64::from(component) + seed).is_multiple_of(2);
                    if accept {
                        expected = flipped;
                    }
                    component += 1;
                    accept
                },
            );
            assert_eq!(component as usize, partition.delta.len());
            assert_eq!(spins, expected);
            assert_eq!(change, graph.energy(&spins) - before);
            assert_eq!(rng.gen::<u64>(), partition_rng.gen::<u64>());
        }
        assert!(covered.into_iter().all(|seen| seen));
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn epoch_wrap_clears_entries_before_reusing_the_first_stamp() {
        let mut cache = SweepCache::new(16);
        cache.prepare(1.0, Weights::FULL);
        cache.heat_probability(1000);
        cache.metropolis_probability(1000);
        cache.heat_probability(4);
        cache.metropolis_probability(4);
        cache.epoch = CacheEpoch::MAX;
        cache.prepare(2.0, Weights::FULL);
        assert_eq!(cache.epoch, 1);
        assert_eq!(cache.stats.epoch_wraps, 1);
        assert_eq!(cache.heat_probability(4), heat_probability(0.008));
        assert_eq!(cache.metropolis_probability(4), (-0.008_f64).exp());
    }

    #[test]
    fn cached_probabilities_match_scalar_bits_and_epoch_changes() {
        let mut cache = SweepCache::new(64);
        for beta in [0.001, 0.5, 1.0, 12.0, 48.0, 700.0, 1.0] {
            cache.prepare(beta, Weights::FULL);
            for delta in (-300..300).chain([i64::MIN, i64::MAX]) {
                let expected = heat_probability(beta * delta as f64 / 1000.0);
                for _ in 0..2 {
                    assert_eq!(cache.heat_probability(delta).to_bits(), expected.to_bits());
                }
                if delta > 0 {
                    let expected = (-beta * delta as f64 / 1000.0).exp();
                    for _ in 0..2 {
                        assert_eq!(
                            cache.metropolis_probability(delta).to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        }
        cache.epoch = CacheEpoch::MAX;
        cache.prepare(0.002, Weights::FULL);
        assert_eq!(cache.epoch, 1);
        assert_eq!(
            cache.heat_probability(2).to_bits(),
            heat_probability(0.002 * 2.0 / 1000.0).to_bits()
        );
        assert!(
            cache.stats.heat_hits > 0
                && cache.stats.metropolis_hits > 0
                && cache.stats.fallbacks > 0
        );
    }
}
