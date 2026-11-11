//! The cluster update and the single-spin sweep.
//!
//! A cluster sweep draws auxiliary bonds on every tile and pair, so the strong
//! layer is sampled exactly, then flips each connected component by heat bath
//! against the weak layer. The tile bond probabilities follow Otsuka,
//! Phys. Rev. B 90, 220406 (2014).

use rand::rngs::SmallRng;
use rand::Rng;
use std::fmt;

use super::graph::{OtsukaGraph, Weights};

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

/// Union-find and component lists, reused across sweeps. Components are
/// numbered densely in the order of their smallest node.
#[derive(Debug)]
struct Workspace {
    parent: Vec<u32>,
    /// The component of each node.
    label: Vec<u32>,
    /// The nodes of component `c` are `members[start[c]..start[c + 1]]`.
    start: Vec<u32>,
    members: Vec<u32>,
    /// The weak-energy change of flipping each component.
    delta: Vec<i64>,
    /// Boltzmann factors of the current Metropolis sweep.
    boltzmann: Boltzmann,
    /// The local field of each node over every coupling, `sum J s_v`,
    /// patched at each flip.
    field: Vec<i64>,
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
        for (u, field) in self.field.iter_mut().enumerate() {
            *field = graph
                .all
                .row(u)
                .iter()
                .map(|&link| i64::from(link.j()) * i64::from(s[link.node()]))
                .sum();
        }
    }

    /// Flip node `u`, patch the local fields of its neighbours, and return
    /// the model energy change in milli.
    #[inline]
    fn flip(&mut self, graph: &OtsukaGraph, s: &mut [i8], u: usize) -> i64 {
        let twice = 2 * i64::from(s[u]);
        for &link in graph.all.row(u) {
            self.field[link.node()] -= twice * i64::from(link.j());
        }
        s[u] = -s[u];
        -twice * self.field[u]
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

    /// Draw the bonds of every tile and pair, then list the components.
    fn partition(&mut self, graph: &OtsukaGraph, s: &[i8], p: &Breakup, rng: &mut SmallRng) {
        for (u, parent) in self.parent.iter_mut().enumerate() {
            *parent = u as u32;
        }
        for tile in &graph.tiles {
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
                    self.join(up[(x & 1) as usize], down[((x >> 1) & 1) as usize]);
                } else {
                    let c = (x & 1) as usize;
                    self.join(up[0], down[c]);
                    self.join(up[1], down[1 - c]);
                }
            } else if r >= p.zero_defect {
                // The source took the partner from the low bits modulo three,
                // which favours the first. A separate range draw is uniform.
                self.join(up[rng.gen_range(0..ups)], down[rng.gen_range(0..downs)]);
            }
        }
        for &(u, v) in &graph.pairs {
            if s[u as usize] == s[v as usize] && rng.gen::<f64>() < p.pair_link {
                self.join(u, v);
            }
        }
        // Number the components, count their nodes, then list the nodes.
        // A join hooks the larger root under the smaller, and path halving
        // only moves a node closer to its root, so `parent[u] <= u` holds
        // and every root is the smallest node of its component. One pass in
        // node order then numbers the components by their smallest node and
        // finds each label at the parent, which the pass has already seen.
        self.start.clear();
        self.start.push(0);
        for u in 0..self.parent.len() {
            let up = self.parent[u] as usize;
            let c = if up == u {
                self.start.push(0);
                (self.start.len() - 2) as u32
            } else {
                self.label[up]
            };
            self.label[u] = c;
            self.start[c as usize + 1] += 1;
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

    /// One cluster sweep. Returns the model energy change in milli.
    fn cluster_sweep(
        &mut self,
        graph: &OtsukaGraph,
        s: &mut [i8],
        beta: f64,
        rng: &mut SmallRng,
    ) -> i64 {
        let p = Breakup::new(beta, graph.weights);
        self.partition(graph, s, &p, rng);
        // delta[c] = -2 * sum over the weak edges that leave c of J s_u s_v.
        self.delta.fill(0);
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
        let mut change = 0;
        // Heat bath in the order of the components' smallest nodes. A flip
        // changes the sign of every edge that leaves the component, so the
        // deltas of its neighbours are patched before the next one draws.
        for c in 0..self.delta.len() {
            let d = self.delta[c];
            if !heat_bath(beta * d as f64 / 1000.0, rng) {
                continue;
            }
            for at in self.start[c] as usize..self.start[c + 1] as usize {
                let u = self.members[at] as usize;
                let su = i64::from(s[u]);
                for &link in graph.weak.row(u) {
                    let v = link.node();
                    let lv = self.label[v] as usize;
                    if lv != c {
                        self.delta[lv] += 4 * i64::from(link.j()) * su * i64::from(s[v]);
                    }
                }
                change += self.flip(graph, s, u);
            }
            self.delta[c] = -d;
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

    /// One Metropolis sweep over every node in index order at `beta`.
    /// Returns the model energy change in milli.
    pub fn single_sweep(&mut self, beta: f64, rng: &mut SmallRng) -> i64 {
        let (graph, s, work) = (self.graph, &mut self.spins, &mut self.work);
        let mut change = 0;
        work.boltzmann.clear();
        for u in 0..graph.n {
            let delta = -2 * i64::from(s[u]) * work.field[u];
            if delta <= 0 || rng.gen::<f64>() < work.boltzmann.get(delta, beta) {
                change += work.flip(graph, s, u);
            }
        }
        change
    }
}
