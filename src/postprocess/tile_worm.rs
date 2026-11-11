//! Tile worm anneal for charged ice and field-biased ice, and the two
//! selected postprocess policies built on it.
//!
//! Port of `worm_run` of `icetiles.c`, the tile tables of `icetiles.py`, and
//! `parse` and `run` of `method.py`, from the research directory
//! `new-pow/postprocess/optimizers/ice-tiles`.
//!
//! A tile is a four-spin set with a 16-entry energy table over its local
//! state. A tile is excited when its table value is above its minimum. A worm
//! flips a seed set, then repairs every excited tile that a flip touched, one
//! corner flip at a time, until no touched tile is excited or a tile has no
//! fix. The worm is kept when its exact energy change is negative, or zero
//! within the zero-move flip budget at zero temperature, or when a Metropolis
//! draw accepts it. Every `fe` worms the free spins get one sweep.
//!
//! The kernel checks the clock once every 64 worms, at the start of each
//! block, and sets the inverse temperature from the elapsed fraction of the
//! run budget. [`Pace`] stands in for that clock. Under [`Work::Budget`] the
//! pace is the wall clock of the calling thread, so the result depends on the
//! host and the load. Under [`Work::Fixed`] the pace is a count of 64-worm
//! blocks, and the result depends only on the inputs and the seed. Seeding,
//! draw order, and tie order follow the source, so a fixed-work run replays
//! the research code run with the same count of blocks.

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use quip_solver_core::CancelToken;

use super::greedy::{DescentError, Greedy, Ising};

/// The CPU budget the target manifests declare for the selected cells, in
/// seconds.
pub const TILE_BUDGET_SECONDS: f64 = 0.25;

/// The pinned option string of the charged-ice cell.
pub const CHARGE_SPEC: &str = "anneal:sa=1,b0=2,len=64,fe=16,nat48=1";

/// The pinned option string of the field-ice cells.
pub const FIELD_SPEC: &str = "anneal:b0=2,nat48=1";

/// A selected postprocess policy: an id, the research option string, and the
/// declared budget. Versioned apart from any family revision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Policy {
    /// The policy id.
    pub id: &'static str,
    /// The research option string.
    pub spec: &'static str,
    /// The declared budget in seconds.
    pub budget_seconds: f64,
}

/// The charged-ice policy.
pub const CHARGE_POLICY: Policy = Policy {
    id: "ice-tiles-charge/1",
    spec: CHARGE_SPEC,
    budget_seconds: TILE_BUDGET_SECONDS,
};

/// The field-ice policy.
pub const FIELD_POLICY: Policy = Policy {
    id: "ice-tiles-field/1",
    spec: FIELD_SPEC,
    budget_seconds: TILE_BUDGET_SECONDS,
};

/// A coupler outside every tile at or above this magnitude is a pair seed.
pub const STRONG_PAIR_MIN: i64 = 400;

/// Worms between two clock reads.
const BLOCK: u64 = 64;

/// The source skips a tail shorter than this, in seconds.
const TAIL_MIN_SECONDS: f64 = 0.002;

/// The xorshift64 state for a zero seed.
const ZERO_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

/// The most worm legs a single flip list may hold. The source sizes its
/// buffers from `len`, and a walk needs room for a seed and a repair.
const MIN_LEN: usize = 8;

/// Why an option string cannot be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpecError {
    /// The method is not `anneal` or `desc`.
    Method(String),
    /// The key is not an option.
    Key(String),
    /// The value is not a finite number in range for its key.
    Value {
        /// The option key.
        key: String,
        /// The text after `=`.
        value: String,
    },
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Method(name) => write!(f, "unknown method {name}"),
            Self::Key(key) => write!(f, "unknown key {key}"),
            Self::Value { key, value } => write!(f, "bad value {value:?} for key {key}"),
        }
    }
}

impl std::error::Error for SpecError {}

/// What the worm does with a read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Metropolis worm anneal on a geometric schedule, then zero-temperature
    /// worm descent for the rest of the budget.
    Anneal,
    /// Zero-temperature worm descent only.
    Descent,
}

/// The parsed option string of `method.py`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileSpec {
    /// `anneal` or `desc`.
    pub method: Method,
    /// Scale of the schedule start, `ln 2 / (2 w)`.
    pub b0: f64,
    /// Scale of the schedule end, `ln 1000 / (2 w)`.
    pub b1: f64,
    /// The anneal share of a read's budget.
    pub frac: f64,
    /// Add whole tiles as seeds.
    pub tiles: bool,
    /// Add strong pairs as seeds.
    pub pairs: bool,
    /// Copies of each tile seed.
    pub tile_weight: usize,
    /// Probability of crossing a rigid tile.
    pub cross: f64,
    /// Scale of the leg-choice inverse temperature.
    pub guide: f64,
    /// Reads, best first, that share the budget evenly.
    pub reads: usize,
    /// Worms between free-spin sweeps. Zero disables the sweep.
    pub free_every: u64,
    /// The worm cap, in flips.
    pub len: usize,
    /// Flip every spin together with its rigid partner.
    pub pair_walk: bool,
    /// The periodic sweep visits every spin instead of the untiled spins.
    pub sweep_all: bool,
    /// Descend every read first and start from the lowest.
    pub nat48: bool,
}

impl TileSpec {
    /// Parse `name:key=value,...` as `method.py` does.
    ///
    /// # Errors
    ///
    /// [`SpecError`] for an unknown method or key, or a value that is not a
    /// finite non-negative number. A `len` below 8 is also a value error.
    pub fn parse(spec: &str) -> Result<Self, SpecError> {
        let (name, rest) = spec.split_once(':').unwrap_or((spec, ""));
        let mut out = Self {
            method: Method::Anneal,
            b0: 1.0,
            b1: 1.0,
            frac: 0.8,
            tiles: false,
            pairs: false,
            tile_weight: 1,
            cross: 0.0,
            guide: 1.0,
            reads: 1,
            free_every: 16,
            len: 4096,
            pair_walk: false,
            sweep_all: false,
            nat48: false,
        };
        for item in rest.split(',').filter(|item| !item.is_empty()) {
            let (key, text) = item.split_once('=').unwrap_or((item, ""));
            let bad = || SpecError::Value {
                key: key.to_owned(),
                value: text.to_owned(),
            };
            let x: f64 = text.parse().map_err(|_| bad())?;
            if !x.is_finite() || x < 0.0 {
                return Err(bad());
            }
            // The source reads an integer option with `int`, which truncates.
            let whole = || {
                let t = x.trunc();
                (t <= f64::from(u32::MAX))
                    .then_some(t as u64)
                    .ok_or_else(bad)
            };
            match key {
                "b0" => out.b0 = x,
                "b1" => out.b1 = x,
                "frac" => out.frac = x,
                "tiles" => out.tiles = x != 0.0,
                "pairs" => out.pairs = x != 0.0,
                "tw" => out.tile_weight = usize::try_from(whole()?).map_err(|_| bad())?,
                "cross" => out.cross = x,
                "guide" => out.guide = x,
                "reads" => out.reads = usize::try_from(whole()?).map_err(|_| bad())?,
                "fe" => out.free_every = whole()?,
                "len" => {
                    out.len = usize::try_from(whole()?).map_err(|_| bad())?;
                    if out.len < MIN_LEN {
                        return Err(bad());
                    }
                }
                "pw" => out.pair_walk = x != 0.0,
                "sa" => out.sweep_all = x != 0.0,
                "nat48" => out.nat48 = x != 0.0,
                _ => return Err(SpecError::Key(key.to_owned())),
            }
        }
        out.method = match name {
            "anneal" => Method::Anneal,
            "desc" => Method::Descent,
            other => return Err(SpecError::Method(other.to_owned())),
        };
        Ok(out)
    }
}

/// Why tile metadata cannot drive the worm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileError {
    /// A tile names a node outside the model.
    Node {
        /// The node.
        node: u32,
    },
    /// A tile names a node twice.
    Repeat {
        /// The tile position.
        tile: usize,
    },
    /// The model has no tile, so the worm has no seed.
    NoTiles,
    /// Every coefficient is zero, so the schedule has no scale.
    NoScale,
}

impl fmt::Display for TileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Node { node } => write!(f, "node {node} is outside the model"),
            Self::Repeat { tile } => write!(f, "tile {tile} names a node twice"),
            Self::NoTiles => write!(f, "the job has no tile"),
            Self::NoScale => write!(f, "every coefficient is zero"),
        }
    }
}

impl std::error::Error for TileError {}

/// What the tile tables hold, as the research `counts`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileCounts {
    /// Tiles.
    pub tiles: usize,
    /// Tiles with exactly two ground states.
    pub rigid_tiles: usize,
    /// Couplers that more than one tile holds.
    pub shared_tile_edges: usize,
    /// Spins in at least one tile.
    pub tiled_spins: usize,
    /// Spins in no tile.
    pub free_spins: usize,
    /// Strong couplers outside every tile.
    pub strong_pairs: usize,
    /// Spins with a rigid partner.
    pub partnered: usize,
    /// The most tiles that one spin is in.
    pub max_tiles_per_spin: usize,
}

/// The six corner pairs of a tile, in the order of the energy table.
const SIDES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Tiles, per-tile energy tables, spin-to-tile lists, free spins, and the
/// strong pairs of a model: the research `TileTables`.
///
/// Every tile row counts. Its 16-entry table comes from the model couplers
/// inside the tile, so a reversed-bond tile, a tile with a missing coupler,
/// and an overlapping tile each get their own ground set. Bit `k` of a table
/// index is set when corner `k` is `+1`.
#[derive(Clone, Debug)]
pub struct TileTables {
    tiles: Vec<[u32; 4]>,
    /// The table of each tile, minus its minimum.
    etab: Vec<[i64; 16]>,
    st_ptr: Vec<usize>,
    st_idx: Vec<u32>,
    tiled: Vec<u32>,
    free: Vec<u32>,
    all: Vec<u32>,
    strong_pairs: Vec<(u32, u32)>,
    /// The rigid partner of each spin, `-1` when none.
    partner: Vec<i32>,
    partner_j: Vec<i64>,
    counts: TileCounts,
}

impl TileTables {
    /// Build the tables of `tetrahedra` over `ising`.
    ///
    /// # Errors
    ///
    /// [`TileError`] for a node outside the model or a tile that names a node
    /// twice.
    pub fn new(ising: &Ising, tetrahedra: &[[u32; 4]]) -> Result<Self, TileError> {
        let n = ising.num_nodes();
        for (at, tile) in tetrahedra.iter().enumerate() {
            if let Some(&node) = tile.iter().find(|&&node| node as usize >= n) {
                return Err(TileError::Node { node });
            }
            if (1..4).any(|k| tile[..k].contains(&tile[k])) {
                return Err(TileError::Repeat { tile: at });
            }
        }
        let mut share: HashMap<(u32, u32), usize> = HashMap::new();
        let mut etab = Vec::with_capacity(tetrahedra.len());
        for tile in tetrahedra {
            let mut jt = [0_i64; 6];
            for (slot, &(a, b)) in jt.iter_mut().zip(&SIDES) {
                let (u, v) = (tile[a], tile[b]);
                if let Some(j) = ising.coupling(u, v) {
                    *slot = j;
                    *share.entry((u.min(v), u.max(v))).or_default() += 1;
                }
            }
            let mut row = [0_i64; 16];
            for (idx, e) in row.iter_mut().enumerate() {
                let corner = |k: usize| if idx >> k & 1 == 1 { 1_i64 } else { -1 };
                *e = jt
                    .iter()
                    .zip(&SIDES)
                    .map(|(&j, &(a, b))| j * corner(a) * corner(b))
                    .sum();
            }
            let min = row.iter().copied().min().unwrap_or_default();
            for e in &mut row {
                *e -= min;
            }
            etab.push(row);
        }

        let mut per_spin: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (t, tile) in tetrahedra.iter().enumerate() {
            let t = u32::try_from(t).map_err(|_| TileError::Node { node: u32::MAX })?;
            for &u in tile {
                per_spin[u as usize].push(t);
            }
        }
        let mut st_ptr = vec![0];
        let mut st_idx = Vec::new();
        for list in &per_spin {
            st_idx.extend_from_slice(list);
            st_ptr.push(st_idx.len());
        }
        let node = |u: usize| u32::try_from(u).map_err(|_| TileError::Node { node: u32::MAX });
        let mut tiled = Vec::new();
        let mut free = Vec::new();
        let mut all = Vec::with_capacity(n);
        for (u, list) in per_spin.iter().enumerate() {
            let u = node(u)?;
            all.push(u);
            if list.is_empty() {
                free.push(u);
            } else {
                tiled.push(u);
            }
        }

        let strong_pairs: Vec<(u32, u32)> = ising
            .edges()
            .iter()
            .copied()
            .filter(|&(u, v)| {
                !share.contains_key(&(u.min(v), u.max(v)))
                    && ising
                        .coupling(u, v)
                        .is_some_and(|j| j.abs() >= STRONG_PAIR_MIN)
            })
            .collect();
        let mut degree = vec![0_usize; n];
        for &(u, v) in &strong_pairs {
            degree[u as usize] += 1;
            degree[v as usize] += 1;
        }
        let mut partner = vec![-1_i32; n];
        let mut partner_j = vec![0_i64; n];
        for &(u, v) in &strong_pairs {
            if degree[u as usize] == 1 && degree[v as usize] == 1 {
                let j = ising.coupling(u, v).unwrap_or_default();
                partner[u as usize] = i32::try_from(v).map_err(|_| TileError::Node { node: v })?;
                partner[v as usize] = i32::try_from(u).map_err(|_| TileError::Node { node: u })?;
                partner_j[u as usize] = j;
                partner_j[v as usize] = j;
            }
        }
        let counts = TileCounts {
            tiles: tetrahedra.len(),
            rigid_tiles: etab
                .iter()
                .filter(|row| row.iter().filter(|&&e| e == 0).count() == 2)
                .count(),
            shared_tile_edges: share.values().filter(|&&c| c > 1).count(),
            tiled_spins: tiled.len(),
            free_spins: free.len(),
            strong_pairs: strong_pairs.len(),
            partnered: partner.iter().filter(|&&p| p >= 0).count(),
            max_tiles_per_spin: per_spin.iter().map(Vec::len).max().unwrap_or_default(),
        };
        Ok(Self {
            tiles: tetrahedra.to_vec(),
            etab,
            st_ptr,
            st_idx,
            tiled,
            free,
            all,
            strong_pairs,
            partner,
            partner_j,
            counts,
        })
    }

    /// What the tables hold.
    #[must_use]
    pub fn counts(&self) -> TileCounts {
        self.counts
    }

    /// The tiles, in input order.
    #[must_use]
    pub fn tiles(&self) -> &[[u32; 4]] {
        &self.tiles
    }

    /// The table of each tile: energy minus the tile minimum.
    #[must_use]
    pub fn tables(&self) -> &[[i64; 16]] {
        &self.etab
    }

    /// Spins in at least one tile, ascending.
    #[must_use]
    pub fn tiled(&self) -> &[u32] {
        &self.tiled
    }

    /// Spins in no tile, ascending.
    #[must_use]
    pub fn free(&self) -> &[u32] {
        &self.free
    }

    /// The strong couplers outside every tile, in model edge order.
    #[must_use]
    pub fn strong_pairs(&self) -> &[(u32, u32)] {
        &self.strong_pairs
    }

    /// The rigid partner of each spin, `-1` when none.
    #[must_use]
    pub fn partners(&self) -> &[i32] {
        &self.partner
    }

    /// The seed sets of the worm: tiled single spins, whole tiles repeated
    /// `tile_weight` times, and strong pairs, in that order.
    #[must_use]
    pub fn seeds(&self, tiles: bool, pairs: bool, tile_weight: usize) -> Vec<Vec<u32>> {
        let mut seeds: Vec<Vec<u32>> = self.tiled.iter().map(|&u| vec![u]).collect();
        if tiles {
            for _ in 0..tile_weight {
                seeds.extend(self.tiles.iter().map(|t| t.to_vec()));
            }
        }
        if pairs {
            seeds.extend(self.strong_pairs.iter().map(|&(u, v)| vec![u, v]));
        }
        seeds
    }
}

/// The clock of one worm run, read once per block of 64 worms.
#[derive(Clone, Copy, Debug)]
enum Pace {
    /// The elapsed fraction of `budget` seconds since `started`.
    Wall { started: Instant, budget: f64 },
    /// Block `next` of `blocks`, in the source's `frac = i / blocks`.
    Ticks { blocks: u64, next: u64 },
}

impl Pace {
    fn wall(budget: f64) -> Self {
        Self::Wall {
            started: Instant::now(),
            budget,
        }
    }

    fn ticks(blocks: u64) -> Self {
        Self::Ticks { blocks, next: 0 }
    }

    /// The elapsed fraction of the run. The run stops at 1 and above.
    fn frac(&mut self) -> f64 {
        match self {
            Self::Wall { started, budget } => {
                if *budget > 0.0 {
                    started.elapsed().as_secs_f64() / *budget
                } else {
                    f64::INFINITY
                }
            }
            Self::Ticks { blocks, next } => {
                let frac = if *blocks == 0 {
                    f64::INFINITY
                } else {
                    *next as f64 / *blocks as f64
                };
                *next += 1;
                frac
            }
        }
    }
}

/// What one worm run did, as the source's `stats`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WormStats {
    /// Worms tried.
    pub worms: u64,
    /// Worms kept.
    pub accepted: u64,
    /// Worms kept with a strictly negative change.
    pub improved: u64,
    /// Worms that never closed.
    pub failed: u64,
    /// Rigid tiles crossed.
    pub crossed: u64,
    /// Free-spin flips.
    pub free_flips: u64,
}

/// The `worm_run` arguments that the schedule varies.
#[derive(Clone, Copy, Debug)]
struct RunSpec {
    beta0: f64,
    beta1: f64,
    guide: f64,
    p_cross: f64,
    max_len: usize,
    zero_budget: i64,
    free_every: u64,
    pair_walk: bool,
    sweep_all: bool,
}

/// The state a worm run starts from, with its energy.
#[derive(Clone, Copy)]
struct Start<'a> {
    state: &'a [i8],
    energy: i64,
}

/// The result of one worm run.
struct RunOut {
    last: Vec<i8>,
    last_energy: i64,
    best: Vec<i8>,
    best_energy: i64,
    stats: WormStats,
}

/// The source's `ctx_t`: the state of one worm run.
struct Ctx<'a> {
    ising: &'a Ising,
    tt: &'a TileTables,
    pair_walk: bool,
    s: Vec<i8>,
    /// Local fields `h_i + sum_j J_ij s_j`.
    f: Vec<i64>,
    path: Vec<u32>,
    /// Pairs of (tile, entering spin).
    stack: Vec<u32>,
    stack_cap: usize,
    max_len: usize,
    rng: u64,
}

impl Ctx<'_> {
    /// The source's xorshift64.
    fn xs64(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    /// A 53-bit uniform draw in [0, 1).
    fn unif(&mut self) -> f64 {
        (self.xs64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    fn flip_spin(&mut self, u: u32) {
        let old = i64::from(self.s[u as usize]);
        self.s[u as usize] = -self.s[u as usize];
        for &(v, j) in self.ising.row(u as usize) {
            self.f[v as usize] -= 2 * j * old;
        }
    }

    fn tile_state(&self, t: usize) -> usize {
        self.tt.tiles[t]
            .iter()
            .enumerate()
            .filter(|&(_, &u)| self.s[u as usize] > 0)
            .fold(0, |idx, (k, _)| idx | 1 << k)
    }

    /// The rigid partner of `y` in the walk, if the walk flips pairs.
    fn partner(&self, y: u32) -> Option<u32> {
        if self.pair_walk {
            u32::try_from(self.tt.partner[y as usize]).ok()
        } else {
            None
        }
    }

    /// Push the other tiles of spin `y` (not `from`). `false` when the stack
    /// is full.
    fn push_tiles(&mut self, y: u32, from: Option<u32>) -> bool {
        let (lo, hi) = (self.tt.st_ptr[y as usize], self.tt.st_ptr[y as usize + 1]);
        for p in lo..hi {
            let t = self.tt.st_idx[p];
            if Some(t) == from {
                continue;
            }
            if self.stack.len() + 2 > self.stack_cap {
                return false;
            }
            self.stack.push(t);
            self.stack.push(y);
        }
        true
    }

    /// The flip delta of `y` together with its rigid partner, when it has one.
    fn unit_delta(&self, y: u32) -> i64 {
        let yi = y as usize;
        let mut d = -2 * i64::from(self.s[yi]) * self.f[yi];
        if let Some(p) = self.partner(y) {
            let pi = p as usize;
            d += -2 * i64::from(self.s[pi]) * self.f[pi]
                + 4 * self.tt.partner_j[yi] * i64::from(self.s[yi]) * i64::from(self.s[pi]);
        }
        d
    }

    /// Flip `y` and its partner, record them on the path, and push their
    /// other tiles. `false` when the stack filled.
    fn flip_unit(&mut self, y: u32, from: Option<u32>, cum: &mut i64) -> bool {
        *cum += -2 * i64::from(self.s[y as usize]) * self.f[y as usize];
        self.flip_spin(y);
        self.path.push(y);
        let mut ok = self.push_tiles(y, from);
        if let Some(p) = self.partner(y) {
            *cum += -2 * i64::from(self.s[p as usize]) * self.f[p as usize];
            self.flip_spin(p);
            self.path.push(p);
            ok = ok && self.push_tiles(p, None);
        }
        ok
    }

    /// One worm from `seed`. Returns the exact energy change and whether the
    /// worm closed. The path holds the flips, for the caller to undo.
    fn worm(
        &mut self,
        seed: &[u32],
        beta_leg: f64,
        p_cross: f64,
        crossed: &mut u64,
    ) -> (i64, bool) {
        let mut cum = 0_i64;
        self.path.clear();
        self.stack.clear();
        for &u in seed {
            if !self.flip_unit(u, None, &mut cum) {
                return (cum, false);
            }
        }
        let mut closed = true;
        while let Some(enter) = self.stack.pop() {
            let Some(t) = self.stack.pop() else { break };
            let t = t as usize;
            let idx = self.tile_state(t);
            if self.tt.etab[t][idx] == 0 {
                continue;
            }
            if self.path.len() + 8 >= self.max_len {
                closed = false;
                break;
            }
            let corner = self.tt.tiles[t];
            let row = self.tt.etab[t];
            let mut pick: Option<usize> = None;
            let mut dpick = 0_i64;
            let mut ties = 0_u64;
            let mut dk = [0_i64; 4];
            let mut cand = [false; 4];
            let mut ncand = 0;
            for k in 0..4 {
                let y = corner[k];
                if y == enter || row[idx ^ (1 << k)] >= row[idx] {
                    continue;
                }
                ncand += 1;
                cand[k] = true;
                let d = self.unit_delta(y);
                dk[k] = d;
                if pick.is_none() || d < dpick {
                    pick = Some(k);
                    dpick = d;
                    ties = 1;
                } else if d == dpick {
                    ties += 1;
                    if self.xs64().is_multiple_of(ties) {
                        pick = Some(k);
                    }
                }
            }
            let Some(mut pick) = pick else {
                // No corner lowers the tile: the leg ends here and the
                // excitation stays, or a rigid tile is crossed by flipping
                // its three other corners with probability `p_cross`.
                let ek = corner.iter().position(|&u| u == enter);
                let reach = ek.map(|ek| idx ^ (15 & !(1 << ek)));
                let Some((ek, reach)) = ek.zip(reach) else {
                    continue;
                };
                if p_cross <= 0.0 || row[reach] != 0 || self.unif() >= p_cross {
                    continue;
                }
                *crossed += 1;
                let mut ok = true;
                for (k, &u) in corner.iter().enumerate() {
                    if k == ek {
                        continue;
                    }
                    ok = self.flip_unit(u, Some(t as u32), &mut cum);
                    if !ok {
                        break;
                    }
                }
                if !ok {
                    closed = false;
                    break;
                }
                continue;
            };
            if beta_leg > 0.0 && ncand > 1 {
                let mut w = [0.0_f64; 4];
                let mut tot = 0.0;
                for k in 0..4 {
                    if cand[k] {
                        w[k] = (-beta_leg * (dk[k] - dpick) as f64).exp();
                        tot += w[k];
                    }
                }
                let mut r = self.unif() * tot;
                for k in 0..4 {
                    if cand[k] {
                        r -= w[k];
                        if r <= 0.0 {
                            pick = k;
                            break;
                        }
                    }
                }
            }
            if !self.flip_unit(corner[pick], Some(t as u32), &mut cum) {
                closed = false;
                break;
            }
        }
        (cum, closed)
    }
}

/// The prepared worm of one model: tile tables, seeds, and the schedule
/// scale of the research `Prepared`.
#[derive(Clone, Debug)]
pub struct TileWorm {
    spec: TileSpec,
    tt: TileTables,
    seeds: Vec<Vec<u32>>,
    b_lo: f64,
    b_hi: f64,
}

impl TileWorm {
    /// Prepare the worm for `ising` with the tiles `tetrahedra`.
    ///
    /// # Errors
    ///
    /// [`TileError`] for invalid tile metadata, no tile, or a model with no
    /// nonzero coefficient.
    pub fn new(ising: &Ising, tetrahedra: &[[u32; 4]], spec: &TileSpec) -> Result<Self, TileError> {
        let tt = TileTables::new(ising, tetrahedra)?;
        let seeds = tt.seeds(spec.tiles, spec.pairs, spec.tile_weight);
        if seeds.is_empty() {
            return Err(TileError::NoTiles);
        }
        let w = ising.weakest().ok_or(TileError::NoScale)? as f64;
        Ok(Self {
            spec: *spec,
            tt,
            seeds,
            b_lo: spec.b0 * 2.0_f64.ln() / (2.0 * w),
            b_hi: spec.b1 * 1000.0_f64.ln() / (2.0 * w),
        })
    }

    /// The option set.
    #[must_use]
    pub fn spec(&self) -> &TileSpec {
        &self.spec
    }

    /// The tile tables.
    #[must_use]
    pub fn tables(&self) -> &TileTables {
        &self.tt
    }

    /// The schedule ends `(b_lo, b_hi)`.
    #[must_use]
    pub fn betas(&self) -> (f64, f64) {
        (self.b_lo, self.b_hi)
    }

    /// One worm run from `s0`, the source's `worm_run`. It stops when `pace`
    /// reaches 1.
    fn run(
        &self,
        ising: &Ising,
        start: Start<'_>,
        pace: &mut Pace,
        seed: u64,
        spec: &RunSpec,
        cancel: Option<(&CancelToken, Option<u64>)>,
    ) -> Result<RunOut, DescentError> {
        let Start {
            state: s0,
            energy: e0,
        } = start;
        let n = ising.num_nodes();
        let mut c = Ctx {
            ising,
            tt: &self.tt,
            pair_walk: spec.pair_walk,
            s: s0.to_vec(),
            f: (0..n).map(|i| ising.field(s0, i)).collect(),
            path: Vec::with_capacity(spec.max_len + 2),
            stack: Vec::with_capacity(8 * spec.max_len + 64),
            stack_cap: 8 * spec.max_len + 64,
            max_len: spec.max_len,
            rng: if seed == 0 { ZERO_SEED } else { seed },
        };
        let sweep = if spec.sweep_all {
            &self.tt.all
        } else {
            &self.tt.free
        };
        let mut e = e0;
        let mut e_best = e0;
        let mut best = s0.to_vec();
        let zero_t = spec.beta0 <= 0.0;
        let mut beta = if zero_t { 0.0 } else { spec.beta0 };
        let lratio = if zero_t {
            0.0
        } else {
            (spec.beta1 / spec.beta0).ln()
        };
        let mut budget = spec.zero_budget;
        let mut crossed = 0_u64;
        let mut stats = WormStats::default();
        let mut w = 0_u64;
        loop {
            if w.is_multiple_of(BLOCK) {
                if cancel.is_some_and(|(token, mark)| token.is_cancelled(mark)) {
                    return Err(DescentError::Cancelled);
                }
                let frac = pace.frac();
                if frac >= 1.0 {
                    break;
                }
                if !zero_t {
                    beta = spec.beta0 * (lratio * frac).exp();
                }
            }
            let k = (c.xs64() % self.seeds.len() as u64) as usize;
            let beta_leg = if zero_t { 0.0 } else { spec.guide * beta };
            let (cum, closed) = c.worm(&self.seeds[k], beta_leg, spec.p_cross, &mut crossed);
            let keep = closed
                && if cum < 0 {
                    true
                } else if cum == 0 {
                    let len = c.path.len() as i64;
                    let keep = !zero_t || budget >= len;
                    if keep && zero_t {
                        budget -= len;
                    }
                    keep
                } else if !zero_t {
                    c.unif() < (-beta * cum as f64).exp()
                } else {
                    false
                };
            stats.worms += 1;
            if keep {
                stats.accepted += 1;
                e += cum;
                if cum < 0 {
                    stats.improved += 1;
                }
                if e < e_best {
                    e_best = e;
                    best.copy_from_slice(&c.s);
                }
            } else {
                if !closed {
                    stats.failed += 1;
                }
                for i in (0..c.path.len()).rev() {
                    c.flip_spin(c.path[i]);
                }
            }
            if spec.free_every > 0 && !sweep.is_empty() && w.is_multiple_of(spec.free_every) {
                for &u in sweep {
                    let d = -2 * i64::from(c.s[u as usize]) * c.f[u as usize];
                    if d < 0 || (d > 0 && !zero_t && c.unif() < (-beta * d as f64).exp()) {
                        c.flip_spin(u);
                        e += d;
                        stats.free_flips += 1;
                    }
                }
                if e < e_best {
                    e_best = e;
                    best.copy_from_slice(&c.s);
                }
            }
            w += 1;
        }
        stats.crossed += crossed;
        Ok(RunOut {
            last: c.s,
            last_energy: e,
            best,
            best_energy: e_best,
            stats,
        })
    }
}

/// How much work the schedule does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Work {
    /// Run against `seconds` of elapsed time on the calling thread, as the
    /// research ran against process CPU time on one core. The clock decides
    /// how many worms run and where the schedule stands, so the result
    /// depends on the host and its load.
    Budget {
        /// The declared budget in seconds.
        seconds: f64,
    },
    /// Run each visited read with a main run of `run_blocks` blocks of 64
    /// worms, then a zero-temperature tail of `tail_blocks` blocks (none
    /// when zero). The result depends only on the inputs and the seed.
    Fixed {
        /// Blocks of the main run.
        run_blocks: u64,
        /// Blocks of the tail. The `desc` method has no tail.
        tail_blocks: u64,
    },
}

/// What a schedule did, for the job log.
#[derive(Clone, Debug, PartialEq)]
pub struct TileReport {
    /// The work mode of this run.
    pub work: Work,
    /// The reads visited.
    pub reads_used: usize,
    /// The input positions of the visited reads, in visit order.
    pub visited: Vec<usize>,
    /// The lowest energy of the supplied reads, after the first descent when
    /// `nat48` is set.
    pub start_energy: i64,
    /// The worm run of the returned read.
    pub stats: WormStats,
    /// The improving worms of the tail of the returned read, when it had one.
    pub tail_improved: Option<u64>,
    /// The energy the final descent took off the worm's best state.
    pub polish: i64,
    /// The final energy, rescored from the model.
    pub energy: i64,
    /// The measured time of the whole schedule, in seconds.
    pub elapsed_seconds: f64,
}

/// The result of a schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct TileOutcome {
    /// The final state.
    pub state: Vec<i8>,
    /// The descended states: every supplied read in input order with `nat48`,
    /// else the visited reads in visit order.
    pub pool: Vec<Vec<i8>>,
    /// What the schedule did.
    pub report: TileReport,
}

/// The lowest descended read so far.
struct BestRead {
    energy: i64,
    state: Vec<i8>,
    stats: WormStats,
    tail_improved: Option<u64>,
    polish: i64,
}

/// The research `run` of `method.py`.
///
/// With `nat48` every read is descended first. The lowest reads (one by
/// default) then share the budget: each gets natural-move descent, a worm run
/// (an anneal and a zero-temperature tail, or a descent), and a final
/// descent. The lowest final state is the result. Under [`Work::Budget`]
/// the descent time counts against the budget, as in the source.
///
/// # Errors
///
/// [`DescentError::Empty`] for no reads, [`DescentError::Length`] or
/// [`DescentError::Value`] for a read that does not fit the model, and
/// [`DescentError::Cancelled`] when the token abandons the watermark.
pub fn schedule_tiles(
    greedy: &Greedy,
    worm: &TileWorm,
    reads: &[Vec<i8>],
    seed: u64,
    work: Work,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<TileOutcome, DescentError> {
    let ising = greedy.ising();
    if reads.is_empty() {
        return Err(DescentError::Empty);
    }
    for read in reads {
        greedy.check(read)?;
    }
    let spec = &worm.spec;
    let started = Instant::now();
    let used = || started.elapsed().as_secs_f64();
    let cancelled = || cancel.is_some_and(|(token, mark)| token.is_cancelled(mark));
    let n = i64::try_from(ising.num_nodes()).unwrap_or(i64::MAX);

    let mut states = reads.to_vec();
    if spec.nat48 {
        for s in &mut states {
            greedy.descend(s, cancel)?;
        }
    }
    let energies: Vec<i64> = states.iter().map(|s| ising.energy(s)).collect();
    let mut order: Vec<usize> = (0..states.len()).collect();
    order.sort_by_key(|&r| energies[r]);
    let n_reads = spec.reads.clamp(1, states.len());
    let per = match work {
        Work::Budget { seconds } => seconds / n_reads as f64,
        Work::Fixed { .. } => 0.0,
    };
    let common = RunSpec {
        beta0: 0.0,
        beta1: 0.0,
        guide: 1.0,
        p_cross: spec.cross,
        max_len: spec.len,
        zero_budget: 0,
        free_every: spec.free_every,
        pair_walk: spec.pair_walk,
        sweep_all: spec.sweep_all,
    };
    let zero = RunSpec {
        zero_budget: 4 * n,
        ..common
    };

    let mut best: Option<BestRead> = None;
    let mut visited = Vec::new();
    let mut descended = Vec::new();
    for (r, &i) in order.iter().take(n_reads).enumerate() {
        let mut s1 = states[i].clone();
        let e1 = if spec.nat48 {
            energies[i]
        } else {
            greedy.descend(&mut s1, cancel)?;
            ising.energy(&s1)
        };
        let left = per * (r + 1) as f64 - used();
        if matches!(work, Work::Budget { .. }) && r > 0 && left <= 0.0 {
            break;
        }
        if cancelled() {
            return Err(DescentError::Cancelled);
        }
        visited.push(i);
        let read_seed = seed.wrapping_add(r as u64);
        let tail_seed = seed.wrapping_add(1000 + r as u64);
        let (main_pace, main_spec) = match (spec.method, work) {
            (Method::Anneal, Work::Budget { .. }) => (
                Pace::wall(spec.frac * left.max(0.0)),
                RunSpec {
                    beta0: worm.b_lo,
                    beta1: worm.b_hi,
                    guide: spec.guide,
                    ..common
                },
            ),
            (Method::Anneal, Work::Fixed { run_blocks, .. }) => (
                Pace::ticks(run_blocks),
                RunSpec {
                    beta0: worm.b_lo,
                    beta1: worm.b_hi,
                    guide: spec.guide,
                    ..common
                },
            ),
            (Method::Descent, Work::Budget { .. }) => (Pace::wall(left.max(0.0)), zero),
            (Method::Descent, Work::Fixed { run_blocks, .. }) => (Pace::ticks(run_blocks), zero),
        };
        let mut main_pace = main_pace;
        let run = worm.run(
            ising,
            Start {
                state: &s1,
                energy: e1,
            },
            &mut main_pace,
            read_seed,
            &main_spec,
            cancel,
        )?;
        let (mut e_r, mut sb_r) = (run.best_energy, run.best.clone());
        let mut tail_improved = None;
        if spec.method == Method::Anneal {
            let tail_pace = match work {
                Work::Budget { .. } => {
                    let tail = per * (r + 1) as f64 - used();
                    (tail > TAIL_MIN_SECONDS).then(|| Pace::wall(tail))
                }
                Work::Fixed { tail_blocks, .. } => {
                    (tail_blocks > 0).then(|| Pace::ticks(tail_blocks))
                }
            };
            if let Some(mut pace) = tail_pace {
                let tail = worm.run(
                    ising,
                    Start {
                        state: &run.last,
                        energy: run.last_energy,
                    },
                    &mut pace,
                    tail_seed,
                    &zero,
                    cancel,
                )?;
                tail_improved = Some(tail.stats.improved);
                if tail.best_energy < e_r {
                    e_r = tail.best_energy;
                    sb_r = tail.best;
                }
            }
        }
        greedy.descend(&mut sb_r, cancel)?;
        let e_p = ising.energy(&sb_r);
        if !spec.nat48 {
            descended.push(s1);
        }
        if best.as_ref().is_none_or(|b| e_p < b.energy) {
            best = Some(BestRead {
                energy: e_p,
                state: sb_r,
                stats: run.stats,
                tail_improved,
                polish: e_r - e_p,
            });
        }
    }
    let BestRead {
        state,
        stats,
        tail_improved,
        polish,
        ..
    } = best.ok_or(DescentError::Empty)?;
    let report = TileReport {
        work,
        reads_used: visited.len(),
        visited,
        start_energy: order.first().map_or(0, |&r| energies[r]),
        stats,
        tail_improved,
        polish,
        energy: ising.energy(&state),
        elapsed_seconds: used(),
    };
    Ok(TileOutcome {
        state,
        pool: if spec.nat48 { states } else { descended },
        report,
    })
}
