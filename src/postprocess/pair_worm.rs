//! Pair-aware worm descent for twisted ice with planted pairs, and the
//! `pw` postprocess schedule of the selected twisted-ice controls.
//!
//! Port of `pairworm.c` (`pair_worm_descent`, `defect_counts`, `crossover`),
//! the tile preparation `ice_tiles` of `descent.py`, the pair checks of
//! `pairs.py`, and `schedule_pw` of `run.py` with its default options, from
//! the research directory `new-pow/postprocess/optimizers/ice-pairs`.
//!
//! In the gauge frame a tile has charge `q = sum g s` over its four corners,
//! and the ice manifold is `q = 0` on every tile. A planted pair `(u, v)` is
//! a ferromagnetic coupler, satisfied when `s_u = s_v`. A worm flips both
//! ends of a pair as one unit, which charges the tiles at the partner, so a
//! worm is a branching walk: a stack of charged tiles, each resolved by one
//! more flip, until no tile the walk touched carries charge. The walk is
//! undone unless its exact energy change is negative, or zero within a
//! per-read flip budget, or wins a Metropolis draw at a positive
//! temperature.
//!
//! Seeding and tie order follow the source, so a fixed-work run replays the
//! research states. The schedule under [`Work::Budget`] reads the clock to
//! decide how many reads and anneal cycles run, so its result depends on the
//! host and the load. Only [`Work::Fixed`] runs are reproducible.

use std::fmt;
use std::time::Instant;

use quip_solver_core::CancelToken;

use super::greedy::{DescentError, Greedy, Ising};

/// Tile and pair couplers are at or above this magnitude. The weak glass is
/// below it, and crossover blocks follow only the strong couplers.
pub const STRONG_MILLI: i64 = 500;

/// The postprocess policy: the research `pw` schedule with default options.
/// Versioned apart from any family revision.
pub const PW_POLICY: &str = "ice-pairs-pw/1";

/// The CPU budget the target manifests declare for the selected cells, in
/// seconds.
pub const PW_BUDGET_SECONDS: f64 = 0.76;

/// The research `dfs` gives up on a branch that would save more stack
/// entries than this.
const STACK_SAVE: usize = 64;

/// The xorshift64 state for a zero seed.
const ZERO_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

/// Why pair or tile metadata cannot drive the worm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairError {
    /// A tile or pair names a node outside the model.
    Node {
        /// The node.
        node: u32,
    },
    /// A pair is not an edge of the model.
    NotCoupler {
        /// The pair position.
        index: usize,
    },
    /// A pair sits on a positive coupling.
    Antiferromagnetic {
        /// The pair position.
        index: usize,
    },
    /// Two pairs share a node, so the node would have two partners.
    Repeat {
        /// The node.
        node: u32,
    },
}

impl fmt::Display for PairError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Node { node } => write!(f, "node {node} is outside the model"),
            Self::NotCoupler { index } => write!(f, "pair {index} is not an edge of the model"),
            Self::Antiferromagnetic { index } => {
                write!(f, "pair {index} is not a ferromagnetic coupler")
            }
            Self::Repeat { node } => write!(f, "node {node} is in two pairs"),
        }
    }
}

impl std::error::Error for PairError {}

/// What the tile preparation kept, as the research `counts`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileCounts {
    /// Tiles whose six couplings fit a gauged antiferromagnetic K4.
    pub tiles: usize,
    /// Tiles dropped because an edge is missing or no gauge fits the signs.
    pub dropped: usize,
    /// Tile memberships dropped because the node already has two tiles.
    pub spins_over_two: usize,
    /// Planted pairs.
    pub pairs: usize,
}

/// The research `PairWorm.descend` keywords. The default is the research
/// default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WormOptions {
    /// Zero-change flips each read may accept.
    pub zero_budget: i64,
    /// Sweeps before the descent stops. A sweep without a strict improvement
    /// also stops it at zero temperature.
    pub max_sweeps: u32,
    /// Flips one walk may make before it is abandoned.
    pub max_len: usize,
    /// Run a pass from every tiled node, in shuffled order, each sweep.
    pub random_pass: bool,
    /// Units the exhaustive walk may branch over in the defect pass. Zero
    /// uses the greedy walk.
    pub defect_depth: u32,
    /// As `defect_depth`, for the random pass.
    pub random_depth: u32,
    /// The exhaustive walk cuts a branch whose running change exceeds this.
    pub prune: i64,
    /// Metropolis temperature in milli. Zero keeps only downhill and budgeted
    /// zero-change walks.
    pub temp: f64,
}

impl Default for WormOptions {
    fn default() -> Self {
        Self {
            zero_budget: 0,
            max_sweeps: 50,
            max_len: 32,
            random_pass: true,
            defect_depth: 0,
            random_depth: 0,
            prune: 6000,
            temp: 0.0,
        }
    }
}

/// The research `PairWorm.crossover` keywords. The default is the research
/// default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrossoverOptions {
    /// Sign-align each pool state to the reference first.
    pub align: bool,
    /// The smallest block that may flip.
    pub min_size: usize,
    /// The largest block that may flip.
    pub max_size: usize,
}

impl Default for CrossoverOptions {
    fn default() -> Self {
        Self {
            align: true,
            min_size: 1,
            max_size: 1 << 30,
        }
    }
}

/// Gauged tiles, pair partners, and the strong graph of one model.
#[derive(Clone, Debug)]
pub struct PairWorm {
    tiles: Vec<[u32; 4]>,
    gauge: Vec<[i8; 4]>,
    /// Up to two tiles per node, `-1` when absent.
    mtile: Vec<[i32; 2]>,
    /// The pair partner of each node, `-1` when unpaired.
    partner: Vec<i32>,
    /// The nodes with at least one tile, ascending.
    tiled: Vec<u32>,
    pairs: Vec<(u32, u32)>,
    strong_ptr: Vec<usize>,
    strong: Vec<u32>,
    counts: TileCounts,
}

/// The sign of `x`: `-1`, `0` or `1`.
fn sign(x: i64) -> i8 {
    match x.signum() {
        1 => 1,
        -1 => -1,
        _ => 0,
    }
}

impl PairWorm {
    /// Prepare the worm for `ising`.
    ///
    /// A tile is a `tetrahedra` row whose six couplers exist and whose signs
    /// equal `g_a g_b` for the gauge `g_0 = 1`, `g_k = sign(J_0k)`. Each node
    /// keeps its first two tiles. Every pair must be a coupler at or below
    /// zero, and no node may be in two pairs.
    ///
    /// # Errors
    ///
    /// [`PairError`] for a node outside the model or invalid pair metadata.
    pub fn new(
        ising: &Ising,
        tetrahedra: &[[u32; 4]],
        pairs: &[(u32, u32)],
    ) -> Result<Self, PairError> {
        let n = ising.num_nodes();
        let check = |node: u32| {
            if (node as usize) < n {
                Ok(())
            } else {
                Err(PairError::Node { node })
            }
        };
        for &node in tetrahedra.iter().flatten() {
            check(node)?;
        }
        let mut partner = vec![-1_i32; n];
        for (index, &(u, v)) in pairs.iter().enumerate() {
            check(u)?;
            check(v)?;
            match ising.coupling(u, v) {
                None => return Err(PairError::NotCoupler { index }),
                Some(j) if j > 0 => return Err(PairError::Antiferromagnetic { index }),
                Some(_) => {}
            }
            for (a, b) in [(u, v), (v, u)] {
                let slot = &mut partner[a as usize];
                if *slot >= 0 {
                    return Err(PairError::Repeat { node: a });
                }
                *slot = i32::try_from(b).map_err(|_| PairError::Node { node: b })?;
            }
        }

        const SIDES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let mut tiles = Vec::new();
        let mut gauge = Vec::new();
        let mut dropped = 0;
        for tet in tetrahedra {
            let js = SIDES.map(|(a, b)| ising.coupling(tet[a], tet[b]));
            let fits = js.iter().all(Option::is_some) && {
                let js = js.map(Option::unwrap_or_default);
                let g = [1, sign(js[0]), sign(js[1]), sign(js[2])];
                let ok = SIDES
                    .iter()
                    .zip(js)
                    .all(|(&(a, b), j)| sign(j) == g[a] * g[b]);
                if ok {
                    gauge.push(g);
                }
                ok
            };
            if fits {
                tiles.push(*tet);
            } else {
                dropped += 1;
            }
        }

        let mut mtile = vec![[-1_i32; 2]; n];
        let mut fill = vec![0_usize; n];
        let mut spins_over_two = 0;
        for (t, tile) in tiles.iter().enumerate() {
            let t = i32::try_from(t).map_err(|_| PairError::Node { node: u32::MAX })?;
            for &u in tile {
                let u = u as usize;
                if fill[u] < 2 {
                    mtile[u][fill[u]] = t;
                    fill[u] += 1;
                } else {
                    spins_over_two += 1;
                }
            }
        }
        let tiled = (0..n)
            .filter(|&u| mtile[u][0] >= 0)
            .map(|u| u32::try_from(u).map_err(|_| PairError::Node { node: u32::MAX }))
            .collect::<Result<_, _>>()?;

        let mut strong_ptr = vec![0];
        let mut strong = Vec::new();
        for u in 0..n {
            strong.extend(
                ising
                    .row(u)
                    .iter()
                    .filter(|&&(_, j)| j.abs() >= STRONG_MILLI)
                    .map(|&(v, _)| v),
            );
            strong_ptr.push(strong.len());
        }

        let counts = TileCounts {
            tiles: tiles.len(),
            dropped,
            spins_over_two,
            pairs: pairs.len(),
        };
        Ok(Self {
            tiles,
            gauge,
            mtile,
            partner,
            tiled,
            pairs: pairs.to_vec(),
            strong_ptr,
            strong,
            counts,
        })
    }

    /// The kept tiles, in input order.
    #[must_use]
    pub fn tiles(&self) -> &[[u32; 4]] {
        &self.tiles
    }

    /// The gauge of each kept tile.
    #[must_use]
    pub fn gauges(&self) -> &[[i8; 4]] {
        &self.gauge
    }

    /// Up to two kept tiles per node, `-1` when absent.
    #[must_use]
    pub fn memberships(&self) -> &[[i32; 2]] {
        &self.mtile
    }

    /// What the tile preparation kept.
    #[must_use]
    pub fn counts(&self) -> TileCounts {
        self.counts
    }

    fn charge(&self, s: &[i8], t: usize) -> i32 {
        (0..4)
            .map(|k| i32::from(self.gauge[t][k]) * i32::from(s[self.tiles[t][k] as usize]))
            .sum()
    }

    /// The charged tiles and the broken pairs of `s`. `s` must have one spin
    /// per node.
    #[must_use]
    pub fn defects(&self, s: &[i8]) -> (usize, usize) {
        let tiles = (0..self.tiles.len())
            .filter(|&t| self.charge(s, t) != 0)
            .count();
        let pairs = self
            .pairs
            .iter()
            .filter(|&&(u, v)| s[u as usize] != s[v as usize])
            .count();
        (tiles, pairs)
    }

    /// Pair-aware worm descent of `s` in place, seeded with `seed`. Returns
    /// the number of accepted worms.
    ///
    /// Each sweep runs a defect pass, one worm attempt from each corner of
    /// every charged tile (cheapest unit change first) and from both ends of
    /// every broken pair, and then, with `random_pass`, a pass from every
    /// tiled node in shuffled order.
    ///
    /// # Errors
    ///
    /// [`DescentError::Length`] or [`DescentError::Value`] when `s` does not
    /// fit the model.
    pub fn descend(
        &self,
        ising: &Ising,
        s: &mut [i8],
        opts: &WormOptions,
        seed: u64,
    ) -> Result<u64, DescentError> {
        check(ising, s)?;
        let mut w = Walk::new(self, ising, s, opts, seed);
        let mut accepted = 0;
        let mut budget = opts.zero_budget;
        let mut order = self.tiled.clone();
        for _ in 0..opts.max_sweeps {
            let mut improved = false;
            for t in 0..self.tiles.len() {
                let q = self.charge(w.s, t);
                if q == 0 {
                    continue;
                }
                let sg = q.signum();
                let mut cand: Vec<(u32, i64)> = (0..4)
                    .filter(|&k| {
                        i32::from(self.gauge[t][k]) * i32::from(w.s[self.tiles[t][k] as usize])
                            == sg
                    })
                    .map(|k| {
                        let y = self.tiles[t][k];
                        (y, w.unit_delta(y))
                    })
                    .collect();
                // A stable sort, as the source's insertion sort.
                cand.sort_by_key(|&(_, d)| d);
                for &(y, _) in &cand {
                    if w.try_move(y, opts.defect_depth, opts.prune, &mut budget, &mut improved) {
                        accepted += 1;
                        break;
                    }
                }
            }
            for &(u, v) in &self.pairs {
                if w.s[u as usize] == w.s[v as usize] {
                    continue;
                }
                if w.try_move(u, opts.defect_depth, opts.prune, &mut budget, &mut improved)
                    || w.try_move(v, opts.defect_depth, opts.prune, &mut budget, &mut improved)
                {
                    accepted += 1;
                }
            }
            if opts.random_pass {
                for i in (1..order.len()).rev() {
                    // `i + 1` fits u64, and the remainder is at most `i`.
                    let k = (w.draw() % (i as u64 + 1)) as usize;
                    order.swap(i, k);
                }
                for &u in &order {
                    if w.try_move(u, opts.random_depth, opts.prune, &mut budget, &mut improved) {
                        accepted += 1;
                    }
                }
            }
            if !improved && opts.temp <= 0.0 {
                break;
            }
        }
        Ok(accepted)
    }

    /// Flip into `s` every block on which a pool state differs from it, when
    /// the exact energy change of the block flip is negative. Returns the
    /// number of blocks flipped.
    ///
    /// Blocks are the connected components of the difference set over the
    /// strong couplers, so a block carries whole tiles and pairs. With
    /// `align`, each pool state is first negated when that brings it closer
    /// to `s`.
    ///
    /// # Errors
    ///
    /// [`DescentError::Length`] or [`DescentError::Value`] when `s` or a pool
    /// state does not fit the model.
    pub fn crossover(
        &self,
        ising: &Ising,
        s: &mut [i8],
        pool: &[Vec<i8>],
        opts: &CrossoverOptions,
    ) -> Result<u64, DescentError> {
        check(ising, s)?;
        for b in pool {
            check(ising, b)?;
        }
        let n = ising.num_nodes();
        let mut f: Vec<i64> = (0..n).map(|i| ising.field(s, i)).collect();
        let mut diff = vec![false; n];
        // 0 unseen, 1 in the current block, 2 finished.
        let mut mark = vec![0_u8; n];
        let mut queue: Vec<usize> = Vec::with_capacity(n);
        let mut flipped = 0;
        for b in pool {
            let sign: i8 = if opts.align {
                let dot: i64 = s.iter().zip(b).map(|(&x, &y)| i64::from(x * y)).sum();
                if dot < 0 {
                    -1
                } else {
                    1
                }
            } else {
                1
            };
            for i in 0..n {
                diff[i] = s[i] != sign * b[i];
                mark[i] = 0;
            }
            for root in 0..n {
                if !diff[root] || mark[root] != 0 {
                    continue;
                }
                queue.clear();
                queue.push(root);
                mark[root] = 1;
                let mut head = 0;
                while head < queue.len() {
                    let u = queue[head];
                    head += 1;
                    for &v in &self.strong[self.strong_ptr[u]..self.strong_ptr[u + 1]] {
                        let v = v as usize;
                        if diff[v] && mark[v] == 0 {
                            mark[v] = 1;
                            queue.push(v);
                        }
                    }
                }
                if queue.len() >= opts.min_size && queue.len() <= opts.max_size {
                    let mut d = 0;
                    for &u in &queue {
                        d += -2 * i64::from(s[u]) * f[u];
                        for &(v, j) in ising.row(u) {
                            if v as usize > u && mark[v as usize] == 1 {
                                d += 4 * j * i64::from(s[u]) * i64::from(s[v as usize]);
                            }
                        }
                    }
                    if d < 0 {
                        for &u in &queue {
                            flip(ising, s, &mut f, u);
                        }
                        flipped += 1;
                    }
                }
                for &u in &queue {
                    mark[u] = 2;
                }
            }
        }
        Ok(flipped)
    }
}

/// Check that `s` has one `±1` spin per node of `ising`.
fn check(ising: &Ising, s: &[i8]) -> Result<(), DescentError> {
    if s.len() != ising.num_nodes() {
        return Err(DescentError::Length);
    }
    match s.iter().position(|&x| x != 1 && x != -1) {
        Some(node) => Err(DescentError::Value { node }),
        None => Ok(()),
    }
}

/// Flip node `u` and update the local fields of its neighbours.
fn flip(ising: &Ising, s: &mut [i8], f: &mut [i64], u: usize) {
    let old = i64::from(s[u]);
    s[u] = -s[u];
    for &(v, j) in ising.row(u) {
        f[v as usize] -= 2 * j * old;
    }
}

/// The state of one worm descent: the source's `Walk`.
struct Walk<'a> {
    m: &'a PairWorm,
    ising: &'a Ising,
    s: &'a mut [i8],
    /// Local fields `h_i + sum_j J_ij s_j`.
    f: Vec<i64>,
    /// Nodes the current walk flipped.
    inpath: Vec<bool>,
    /// The flipped nodes in order.
    path: Vec<u32>,
    /// Tiles to check. Entries at and above `sp` are stale, as in the
    /// source, where a push overwrites them.
    stack: Vec<u32>,
    sp: usize,
    /// The exact energy change of the walk so far.
    cum: i64,
    max_len: usize,
    temp: f64,
    rng: u64,
}

impl<'a> Walk<'a> {
    fn new(
        m: &'a PairWorm,
        ising: &'a Ising,
        s: &'a mut [i8],
        opts: &WormOptions,
        seed: u64,
    ) -> Self {
        let n = ising.num_nodes();
        let f = (0..n).map(|i| ising.field(s, i)).collect();
        Self {
            m,
            ising,
            s,
            f,
            inpath: vec![false; n],
            path: Vec::with_capacity(opts.max_len + 2),
            stack: Vec::with_capacity(6 * (opts.max_len + 2) + 8),
            sp: 0,
            cum: 0,
            max_len: opts.max_len,
            temp: opts.temp,
            rng: if seed == 0 { ZERO_SEED } else { seed },
        }
    }

    /// The source's xorshift64.
    fn draw(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    fn charge(&self, t: u32) -> i32 {
        self.m.charge(self.s, t as usize)
    }

    fn push(&mut self, t: u32) {
        if self.sp < self.stack.len() {
            self.stack[self.sp] = t;
        } else {
            self.stack.push(t);
        }
        self.sp += 1;
    }

    fn push_tiles(&mut self, u: u32) {
        for t in self.m.mtile[u as usize] {
            if let Ok(t) = u32::try_from(t) {
                self.push(t);
            }
        }
    }

    fn flip_one(&mut self, u: u32) {
        let i = u as usize;
        self.cum += -2 * i64::from(self.s[i]) * self.f[i];
        flip(self.ising, self.s, &mut self.f, i);
        self.inpath[i] = true;
        self.path.push(u);
        self.push_tiles(u);
    }

    /// Flip `u` as part of the walk, then its partner when the pair would
    /// otherwise break.
    fn walk_flip(&mut self, u: u32) {
        self.flip_one(u);
        if let Ok(p) = u32::try_from(self.m.partner[u as usize]) {
            if !self.inpath[p as usize] && self.s[u as usize] != self.s[p as usize] {
                self.flip_one(p);
            }
        }
    }

    /// The energy change of flipping `y` and, when the pair would break, its
    /// partner.
    fn unit_delta(&self, y: u32) -> i64 {
        let (yi, s) = (y as usize, &self.s);
        let mut d = -2 * i64::from(s[yi]) * self.f[yi];
        if let Ok(p) = u32::try_from(self.m.partner[yi]) {
            let pi = p as usize;
            if !self.inpath[pi] && s[yi] == s[pi] {
                let jyp = self.ising.coupling(y, p).unwrap_or_default();
                d += -2 * i64::from(s[pi]) * self.f[pi] + 4 * jyp * i64::from(s[yi] * s[pi]);
            }
        }
        d
    }

    /// Unflip the walk back to `len` flips.
    fn rewind(&mut self, len: usize) {
        while self.path.len() > len {
            if let Some(u) = self.path.pop() {
                flip(self.ising, self.s, &mut self.f, u as usize);
                self.inpath[u as usize] = false;
            }
        }
    }

    fn undo(&mut self) {
        self.rewind(0);
        self.sp = 0;
        self.cum = 0;
    }

    fn commit(&mut self) {
        for &u in &self.path {
            self.inpath[u as usize] = false;
        }
        self.path.clear();
        self.sp = 0;
        self.cum = 0;
    }

    /// The candidates of charged tile `t` with charge sign `sg`: corners not
    /// in the walk whose gauged spin has that sign, with their unit changes.
    fn candidates(&self, t: u32, sg: i32) -> Vec<(u32, i64)> {
        let t = t as usize;
        (0..4)
            .filter_map(|k| {
                let y = self.m.tiles[t][k];
                let gs = i32::from(self.m.gauge[t][k]) * i32::from(self.s[y as usize]);
                (!self.inpath[y as usize] && gs == sg).then(|| (y, self.unit_delta(y)))
            })
            .collect()
    }

    /// The greedy walk from `u0` until every touched tile is neutral. `true`
    /// when the walk closed, `false` when it failed and was undone.
    fn worm(&mut self, u0: u32) -> bool {
        self.walk_flip(u0);
        while self.sp > 0 {
            self.sp -= 1;
            let t = self.stack[self.sp];
            let q = self.charge(t);
            if q == 0 {
                continue;
            }
            if self.path.len() >= self.max_len {
                self.undo();
                return false;
            }
            // The cheapest candidate. Ties go to a later corner with
            // probability one over the tie count, as in the source.
            let mut pick: Option<(u32, i64)> = None;
            let mut ties = 0_u64;
            for (y, d) in self.candidates(t, q.signum()) {
                match pick {
                    Some((_, best)) if d > best => {}
                    Some((_, best)) if d == best => {
                        ties += 1;
                        if self.draw().is_multiple_of(ties) {
                            pick = Some((y, d));
                        }
                    }
                    _ => {
                        pick = Some((y, d));
                        ties = 1;
                    }
                }
            }
            let Some((y, _)) = pick else {
                self.undo();
                return false;
            };
            // Re-check the tile: a charge of four needs two flips.
            self.push(t);
            self.walk_flip(y);
        }
        true
    }

    /// The Metropolis draw for an uphill walk.
    fn accept_uphill(&mut self) -> bool {
        if self.temp <= 0.0 || self.cum <= 0 {
            return false;
        }
        // A 53-bit uniform draw in [0, 1).
        let u = (self.draw() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0);
        u < (-(self.cum as f64) / self.temp).exp()
    }

    /// Whether a closed walk is kept: downhill, zero within the budget, or a
    /// won Metropolis draw.
    fn keep(&mut self, budget: &mut i64, improved: &mut bool) -> bool {
        if self.cum < 0 {
            *improved = true;
            return true;
        }
        let len = self.path.len() as i64;
        if (self.cum == 0 && *budget >= len) || self.accept_uphill() {
            if self.cum == 0 {
                *budget -= len;
            }
            return true;
        }
        false
    }

    fn try_worm(&mut self, u0: u32, budget: &mut i64, improved: &mut bool) -> bool {
        if !self.worm(u0) {
            return false;
        }
        if self.keep(budget, improved) {
            self.commit();
            return true;
        }
        self.undo();
        false
    }

    /// The exhaustive depth-limited walk: at each charged tile, branch over
    /// the candidates, cheapest first, and backtrack. The stack entries above
    /// the first charged tile are saved and restored around the branch,
    /// because pushes overwrite them.
    fn dfs(&mut self, depth_left: u32, prune: i64, budget: &mut i64, improved: &mut bool) -> bool {
        let sp0 = self.sp;
        let mut charged = None;
        while self.sp > 0 {
            let t = self.stack[self.sp - 1];
            if self.charge(t) != 0 {
                charged = Some(t);
                break;
            }
            self.sp -= 1;
        }
        let Some(t) = charged else {
            if self.keep(budget, improved) {
                return true;
            }
            self.sp = sp0;
            return false;
        };
        if depth_left == 0 || self.path.len() >= self.max_len || self.cum > prune {
            self.sp = sp0;
            return false;
        }
        let nsave = sp0 - self.sp;
        if nsave > STACK_SAVE {
            self.sp = sp0;
            return false;
        }
        let save = self.stack[self.sp..sp0].to_vec();
        let mut cand = self.candidates(t, self.charge(t).signum());
        cand.sort_by_key(|&(_, d)| d);
        for (y, _) in cand {
            let (len0, cum0, sp1) = (self.path.len(), self.cum, self.sp);
            self.walk_flip(y);
            if self.dfs(depth_left - 1, prune, budget, improved) {
                return true;
            }
            self.rewind(len0);
            self.cum = cum0;
            self.sp = sp1;
        }
        let at = self.sp;
        self.stack[at..at + nsave].copy_from_slice(&save);
        self.sp = sp0;
        false
    }

    /// A worm from `u0`: the exhaustive walk when `depth > 0`, else the
    /// greedy walk.
    fn try_move(
        &mut self,
        u0: u32,
        depth: u32,
        prune: i64,
        budget: &mut i64,
        improved: &mut bool,
    ) -> bool {
        if depth == 0 {
            return self.try_worm(u0, budget, improved);
        }
        self.walk_flip(u0);
        if self.dfs(depth, prune, budget, improved) {
            self.commit();
            return true;
        }
        self.undo();
        false
    }
}

/// How much work the schedule does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Work {
    /// Run against [`PW_BUDGET_SECONDS`] of elapsed time on the calling
    /// thread, as the research ran against process CPU time on one core. The
    /// clock decides the read and anneal counts, so the result depends on the
    /// host and its load.
    Budget,
    /// Visit at most `reads` reads (always at least one), then run exactly
    /// `anneal_cycles` anneal-quench cycles. The result depends only on the
    /// inputs and the seed.
    Fixed {
        /// The most reads to visit.
        reads: usize,
        /// The anneal-quench cycles to run.
        anneal_cycles: u64,
    },
}

/// What a schedule did, for the job log.
#[derive(Clone, Debug, PartialEq)]
pub struct PwReport {
    /// [`PW_POLICY`].
    pub policy: &'static str,
    /// The declared budget, [`PW_BUDGET_SECONDS`], whatever the work mode.
    pub budget_seconds: f64,
    /// The work mode of this run.
    pub work: Work,
    /// The reads visited.
    pub reads_used: usize,
    /// The input positions of the visited reads, in visit order.
    pub visited: Vec<usize>,
    /// The measured time of each visited read, in seconds.
    pub read_seconds: Vec<f64>,
    /// The crossover blocks flipped, over every round.
    pub crossover_blocks: u64,
    /// The crossover rounds that flipped a block.
    pub crossover_rounds: u64,
    /// The best pool energy before crossover.
    pub pre_crossover: i64,
    /// The energy after crossover, before the anneal tail.
    pub pre_anneal: i64,
    /// The anneal-quench cycles run.
    pub anneal_cycles: u64,
    /// The final energy, rescored from the model.
    pub energy: i64,
    /// The measured time of the whole schedule, in seconds.
    pub elapsed_seconds: f64,
}

/// The result of a schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct PwOutcome {
    /// The final state.
    pub state: Vec<i8>,
    /// The descended state of each visited read, in visit order.
    pub pool: Vec<Vec<i8>>,
    /// What the schedule did.
    pub report: PwReport,
}

/// The source's `max_len` and `defect_depth` defaults for the schedule.
const PW_MAX_LEN: usize = 32;
const PW_DEFECT_DEPTH: u32 = 8;
/// The anneal temperature in milli, and the sweeps of one anneal cycle.
const PW_TEMP: f64 = 250.0;
const PW_CYCLE_SWEEPS: u32 = 1;
const PW_CROSSOVER_ROUNDS: u64 = 3;
/// Rounds of worm and natural-move descent in `alternate`.
const ALTERNATE_ROUNDS: usize = 10;

/// The research `pw` schedule with default options.
///
/// The reads are visited from the lowest energy up, each through natural-move
/// descent alternated with the strict pair worm. Under [`Work::Budget`] the
/// reads take the first 0.05 s plus 45 percent of the rest of the budget.
/// The best visited state then takes improving blocks from the others
/// (crossover, up to three rounds), and anneal-quench cycles of the worm
/// keep the best quenched state for the rest of the budget.
///
/// # Errors
///
/// [`DescentError::Empty`] for no reads, [`DescentError::Length`] or
/// [`DescentError::Value`] for a read that does not fit the model, and
/// [`DescentError::Cancelled`] when the token abandons the watermark.
pub fn schedule_pw(
    greedy: &Greedy,
    pw: &PairWorm,
    reads: &[Vec<i8>],
    seed: u64,
    work: Work,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<PwOutcome, DescentError> {
    let ising = greedy.ising();
    if reads.is_empty() {
        return Err(DescentError::Empty);
    }
    for read in reads {
        check(ising, read)?;
    }
    let started = Instant::now();
    let used = || started.elapsed().as_secs_f64();
    let cancelled = || cancel.is_some_and(|(token, mark)| token.is_cancelled(mark));
    let limit = PW_BUDGET_SECONDS;
    let reserve = f64::min(0.02, 0.1 * limit);
    let read_limit = f64::min(limit, 0.05 + 0.45 * limit);

    let strict = |s: &mut [i8], k: u64| {
        let opts = WormOptions {
            max_len: PW_MAX_LEN,
            defect_depth: PW_DEFECT_DEPTH,
            ..WormOptions::default()
        };
        pw.descend(ising, s, &opts, seed.wrapping_add(k))
            .map(|_| ())
    };
    let alternate = |s: &mut [i8], k: u64| -> Result<(), DescentError> {
        greedy.descend(s, cancel)?;
        let mut e = ising.energy(s);
        for _ in 0..ALTERNATE_ROUNDS {
            strict(s, k)?;
            greedy.descend(s, cancel)?;
            let next = ising.energy(s);
            if next == e {
                break;
            }
            e = next;
        }
        Ok(())
    };

    let energies: Vec<i64> = reads.iter().map(|r| ising.energy(r)).collect();
    let mut order: Vec<usize> = (0..reads.len()).collect();
    order.sort_by_key(|&r| energies[r]);
    let mut pool: Vec<Vec<i8>> = Vec::new();
    let mut visited = Vec::new();
    let mut read_seconds: Vec<f64> = Vec::new();
    for (k, &r) in order.iter().enumerate() {
        if k > 0 {
            let stop = match work {
                Work::Fixed { reads, .. } => k >= reads,
                Work::Budget => {
                    let est = read_seconds.iter().copied().fold(0.0, f64::max);
                    used() + est + reserve > read_limit
                }
            };
            if stop {
                break;
            }
        }
        if cancelled() {
            return Err(DescentError::Cancelled);
        }
        let t0 = used();
        let mut s = reads[r].clone();
        alternate(&mut s, k as u64)?;
        pool.push(s);
        visited.push(r);
        read_seconds.push(used() - t0);
    }

    let pool_energies: Vec<i64> = pool.iter().map(|s| ising.energy(s)).collect();
    let best = (0..pool.len())
        .min_by_key(|&i| pool_energies[i])
        .unwrap_or_default();
    let pre_crossover = pool_energies[best];
    let mut s = pool[best].clone();
    let mut crossover_blocks = 0;
    let mut crossover_rounds = 0;
    if pool.len() > 1 {
        let others: Vec<Vec<i8>> = pool
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != best)
            .map(|(_, x)| x.clone())
            .collect();
        for round in 0..PW_CROSSOVER_ROUNDS {
            let mut next = s.clone();
            let blocks = pw.crossover(ising, &mut next, &others, &CrossoverOptions::default())?;
            if blocks == 0 {
                break;
            }
            crossover_blocks += blocks;
            crossover_rounds = round + 1;
            alternate(&mut next, 1000 + round)?;
            s = next;
        }
    }

    let pre_anneal = ising.energy(&s);
    let mut best_energy = pre_anneal;
    let anneal = WormOptions {
        max_sweeps: PW_CYCLE_SWEEPS,
        max_len: PW_MAX_LEN,
        temp: PW_TEMP,
        ..WormOptions::default()
    };
    let mut cycles = 0_u64;
    let mut cycle_cost = 0.0_f64;
    loop {
        let more = match work {
            Work::Fixed { anneal_cycles, .. } => cycles < anneal_cycles,
            Work::Budget => limit - used() > cycle_cost + reserve,
        };
        if !more {
            break;
        }
        if cancelled() {
            return Err(DescentError::Cancelled);
        }
        let t0 = used();
        let mut a = s.clone();
        pw.descend(ising, &mut a, &anneal, seed.wrapping_add(3000 + cycles))?;
        alternate(&mut a, 4000 + cycles)?;
        let e = ising.energy(&a);
        if e < best_energy {
            best_energy = e;
            s = a;
        }
        cycles += 1;
        cycle_cost = cycle_cost.max(used() - t0);
    }

    let report = PwReport {
        policy: PW_POLICY,
        budget_seconds: PW_BUDGET_SECONDS,
        work,
        reads_used: pool.len(),
        visited,
        read_seconds,
        crossover_blocks,
        crossover_rounds,
        pre_crossover,
        pre_anneal,
        anneal_cycles: cycles,
        energy: ising.energy(&s),
        elapsed_seconds: used(),
    };
    Ok(PwOutcome {
        state: s,
        pool,
        report,
    })
}
