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

const NONE: usize = usize::MAX;

/// Union-find and component lists, reused across sweeps.
#[derive(Debug)]
struct Workspace {
    parent: Vec<usize>,
    size: Vec<usize>,
    label: Vec<usize>,
    head: Vec<usize>,
    next: Vec<usize>,
    /// The weak-energy change of flipping each component.
    delta: Vec<i64>,
    roots: Vec<usize>,
}

impl Workspace {
    fn new(n: usize) -> Self {
        Self {
            parent: vec![0; n],
            size: vec![1; n],
            label: vec![0; n],
            head: vec![NONE; n],
            next: vec![NONE; n],
            delta: vec![0; n],
            roots: Vec::with_capacity(n),
        }
    }

    fn find(&mut self, mut u: usize) -> usize {
        while self.parent[u] != u {
            let grand = self.parent[self.parent[u]];
            self.parent[u] = grand;
            u = grand;
        }
        u
    }

    fn join(&mut self, a: usize, b: usize) {
        let mut a = self.find(a);
        let mut b = self.find(b);
        if a == b {
            return;
        }
        if self.size[a] < self.size[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a;
        self.size[a] += self.size[b];
    }

    /// Draw the bonds of every tile and pair, then list the components.
    fn partition(&mut self, graph: &OtsukaGraph, s: &[i8], p: &Breakup, rng: &mut SmallRng) {
        for u in 0..graph.n {
            self.parent[u] = u;
            self.size[u] = 1;
        }
        for tile in &graph.tiles {
            let mut up = [0; 4];
            let mut down = [0; 4];
            let (mut ups, mut downs) = (0, 0);
            for k in 0..4 {
                if s[tile.nodes[k]] * tile.gauge[k] > 0 {
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
            if s[u] == s[v] && rng.gen::<f64>() < p.pair_link {
                self.join(u, v);
            }
        }
        self.head.fill(NONE);
        self.roots.clear();
        for u in 0..graph.n {
            let root = self.find(u);
            self.label[u] = root;
            if self.head[root] == NONE {
                self.roots.push(root);
            }
            self.next[u] = self.head[root];
            self.head[root] = u;
        }
    }

    /// One cluster sweep. Returns the weak-energy change in milli.
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
        for &root in &self.roots {
            self.delta[root] = 0;
        }
        for &(u, v, j) in &graph.weak_edges {
            let (lu, lv) = (self.label[u], self.label[v]);
            if lu != lv {
                let t = -2 * i64::from(j) * i64::from(s[u] * s[v]);
                self.delta[lu] += t;
                self.delta[lv] += t;
            }
        }
        let mut change = 0;
        // Heat bath in the order the components were found. A flip changes
        // the sign of every edge that leaves the component, so the deltas of
        // its neighbours are patched before the next component draws.
        for at in 0..self.roots.len() {
            let root = self.roots[at];
            let d = self.delta[root];
            if !heat_bath(beta * d as f64 / 1000.0, rng) {
                continue;
            }
            change += d;
            let mut u = self.head[root];
            while u != NONE {
                let su = i64::from(s[u]);
                for (v, j) in graph.weak.row(u) {
                    let lv = self.label[v];
                    if lv != root {
                        self.delta[lv] += 4 * i64::from(j) * su * i64::from(s[v]);
                    }
                }
                s[u] = -s[u];
                u = self.next[u];
            }
            self.delta[root] = -d;
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
        Ok(Self {
            graph,
            spins,
            work: Workspace::new(graph.n),
        })
    }

    /// A chain on `graph` at a uniformly random state drawn from `rng`.
    pub(crate) fn random(graph: &'g OtsukaGraph, rng: &mut SmallRng) -> Self {
        let spins = (0..graph.n)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();
        Self {
            graph,
            spins,
            work: Workspace::new(graph.n),
        }
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
        Ok(())
    }

    /// The current state.
    #[must_use]
    pub fn spins(&self) -> &[i8] {
        &self.spins
    }

    /// One cluster sweep at `beta`, in inverse units of 1000 milli. `beta`
    /// must be finite and positive. Returns the weak-energy change in milli.
    pub fn cluster_sweep(&mut self, beta: f64, rng: &mut SmallRng) -> i64 {
        self.work
            .cluster_sweep(self.graph, &mut self.spins, beta, rng)
    }

    /// One Metropolis sweep over every node in index order at `beta`.
    /// Returns the model energy change in milli.
    pub fn single_sweep(&mut self, beta: f64, rng: &mut SmallRng) -> i64 {
        let s = &mut self.spins;
        let mut change = 0;
        for u in 0..self.graph.n {
            let field: i64 = self
                .graph
                .all
                .row(u)
                .map(|(v, j)| i64::from(j) * i64::from(s[v]))
                .sum();
            let delta = -2 * i64::from(s[u]) * field;
            if delta <= 0 || rng.gen::<f64>() < (-beta * delta as f64 / 1000.0).exp() {
                s[u] = -s[u];
                change += delta;
            }
        }
        change
    }
}
