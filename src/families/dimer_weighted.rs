//! The bounded weighted-dimer kernel: exact milliweight single-spin updates
//! with per-lane cached fields, exact tetrahedron block updates, and the
//! alternating ice loops of the research `ice-loop64-weighted` variant.
//!
//! This is a literal port of the research `dimer_weighted.rs` and its `ice.rs`,
//! trimmed to the variants that run on gauged-ice tiles: the local-block
//! update, the ice loop, and the ice loop with defect paths. The path-block,
//! contraction, polish, tempering, and seeded paths of the research fork are
//! not ported. The wide bond counter and the cluster moves of the fork's
//! `sa_msc.rs` are not needed: this kernel keeps its own integer fields.
//!
//! One [`State`] holds 64 reads, one per bit lane. The kernel keeps the
//! research random streams: a per-read stream for the start states, and three
//! salted streams for single flips, block flips, and ice moves, shared by all
//! chunks of 64 reads.

use quip_solver_core::{CancelToken, IsingGraph, SampleParams};
use rand::distributions::Open01;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::fmt;

use crate::coloring::Coloring;
use crate::sa_msc::{MscState, LANES};
use crate::sa_msc_f64::MAX_READS;
use crate::sa_sampler::read_rng;
use crate::sampler_core::{build_beta_schedule, CpuGraph};

mod ice;

const SCALE: f64 = 1000.0;
/// Most total absolute milliweight the integer fields may hold.
const SUM_LIMIT: i64 = i32::MAX as i64 / 16;
const SINGLE_SALT: u64 = 0x5745_4947_4854_3031;
const GROUP_SALT: u64 = 0x4752_4f55_5053_3031;
const ICE_SALT: u64 = 0x4943_454c_4f4f_5031;
/// Most spins in one block. Tiles have four.
const MAX_BLOCK_SPINS: usize = 8;
/// The milliweight of a tile bond and of a planted pair.
const STRONG: i32 = 1000;

/// Why the kernel did not return samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DimerError {
    /// The job, its metadata, or its parameters violate a documented bound.
    Invalid(String),
    /// The cancel token abandoned this job's watermark.
    Cancelled,
}

impl fmt::Display for DimerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::Cancelled => f.write_str("the anneal was cancelled"),
        }
    }
}

impl std::error::Error for DimerError {}

fn invalid<T>(message: &str) -> Result<T, DimerError> {
    Err(DimerError::Invalid(message.to_owned()))
}

/// Counts across all reads and sweeps of the ice moves.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct IceLoopStats {
    /// Random starts, including paths that end at defects or unmatched boundaries.
    pub attempts: u64,
    /// Valid closed alternating paths before the energy test.
    pub formed_loops: u64,
    /// Valid boundary-to-boundary alternating paths before the energy test.
    pub formed_strings: u64,
    /// Proposals accepted by the full-model energy test.
    pub accepted: u64,
    /// Total spin flips in accepted paths.
    pub flipped_spins: u64,
    /// Uniform tetrahedron starts reserved for defect paths.
    pub defect_attempts: u64,
    /// Valid paths with at least two spins starting at a defect.
    pub formed_defects: u64,
    /// Defect paths accepted by the full-model energy test.
    pub accepted_defects: u64,
}

/// The ice moves of one sweep: how many attempts each read gets, and whether
/// the odd attempts are defect paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Method {
    /// Ice attempts per read per sweep.
    pub ice_attempts: usize,
    /// Whether odd attempts start at a charged tetrahedron.
    pub defects: bool,
}

impl Method {
    /// Research `ice-loop-weighted`: one loop attempt per read per sweep.
    pub const LOOP: Self = Self {
        ice_attempts: 1,
        defects: false,
    };
    /// Research `ice-loop64-weighted`: 64 loop attempts per read per sweep.
    /// This is the method `quip-cpu-dimer` runs.
    pub const LOOP64: Self = Self {
        ice_attempts: 64,
        defects: false,
    };
    /// Research `ice-defect64-weighted`: 32 loop and 32 defect-path attempts.
    pub const DEFECT64: Self = Self {
        ice_attempts: 64,
        defects: true,
    };
}

struct WeightedGraph {
    neighbors: Vec<Vec<(usize, i32)>>,
}

impl WeightedGraph {
    /// The exact milliweight graph. Fields, inexact or oversize couplings,
    /// out-of-range or self-loop edges, and sums that could overflow the
    /// per-lane field are refused.
    fn new(graph: &IsingGraph) -> Result<Self, DimerError> {
        if graph.h.iter().any(|&h| h != 0.0) {
            return invalid("weighted dimer kernels require zero fields");
        }
        if graph.j.len() != graph.edges.len() {
            return invalid("the couplings and edges differ in length");
        }
        let mut neighbors = vec![Vec::new(); graph.h.len()];
        let mut total = 0i64;
        for (&(u, v), &j) in graph.edges.iter().zip(&graph.j) {
            if u == v || u >= neighbors.len() || v >= neighbors.len() {
                return invalid("an edge has a repeated or out-of-range node");
            }
            let milli = j * SCALE;
            if !milli.is_finite()
                || (milli - milli.round()).abs() > 1e-8
                || milli.abs() > SUM_LIMIT as f64
            {
                return invalid("couplings must be bounded exact milliweights");
            }
            let weight = milli.round() as i32;
            total += i64::from(weight).abs();
            if total > SUM_LIMIT {
                return invalid("sum of absolute milliweights exceeds the i32 safety bound");
            }
            neighbors[u].push((v, weight));
            neighbors[v].push((u, weight));
        }
        Ok(Self { neighbors })
    }
}

struct Group {
    nodes: Vec<usize>,
    internal: Vec<(usize, usize, i32)>,
    energies: Vec<i32>,
}

impl Group {
    /// Local blocks: each of two through eight spins, with no repeated node,
    /// connected by its nonzero bonds. Blocks may share nodes.
    fn build(graph: &WeightedGraph, groups: &[Vec<usize>]) -> Result<Vec<Self>, DimerError> {
        for nodes in groups {
            if !(2..=MAX_BLOCK_SPINS).contains(&nodes.len()) {
                return invalid("weighted blocks must have two through eight spins");
            }
            for (position, &node) in nodes.iter().enumerate() {
                if node >= graph.neighbors.len() || nodes[..position].contains(&node) {
                    return invalid("invalid or repeated node in a block definition");
                }
            }
        }
        let mut result = Vec::with_capacity(groups.len());
        for nodes in groups {
            let mut internal = Vec::new();
            for (i, &u) in nodes.iter().enumerate() {
                for &(v, j) in &graph.neighbors[u] {
                    if let Some(k) = nodes.iter().position(|&node| node == v) {
                        if i < k {
                            internal.push((i, k, j));
                        }
                    }
                }
            }
            let mut reached = vec![false; nodes.len()];
            reached[0] = true;
            for _ in 1..nodes.len() {
                for &(u, v, j) in &internal {
                    if j != 0 && (reached[u] || reached[v]) {
                        reached[u] = true;
                        reached[v] = true;
                    }
                }
            }
            if reached.iter().any(|&value| !value) {
                return invalid("each block must be connected by its bonds");
            }
            let energies = (0..1usize << nodes.len())
                .map(|bits| {
                    internal
                        .iter()
                        .map(|&(u, v, j)| {
                            if ((bits >> u) ^ (bits >> v)) & 1 == 0 {
                                j
                            } else {
                                -j
                            }
                        })
                        .sum()
                })
                .collect();
            result.push(Self {
                nodes: nodes.clone(),
                internal,
                energies,
            });
        }
        Ok(result)
    }

    fn boundary(&self, state: &State) -> [[i32; LANES]; MAX_BLOCK_SPINS] {
        let mut boundary = [[0; LANES]; MAX_BLOCK_SPINS];
        for (fields, &node) in boundary.iter_mut().zip(&self.nodes) {
            *fields = state.fields[node];
        }
        for &(u, v, j) in &self.internal {
            let (left, right) = boundary.split_at_mut(v);
            for (lane, (field_u, field_v)) in left[u].iter_mut().zip(&mut right[0]).enumerate() {
                *field_u -= j * spin(state.spins[self.nodes[v]], lane);
                *field_v -= j * spin(state.spins[self.nodes[u]], lane);
            }
        }
        boundary
    }

    fn flip_mask(
        &self,
        state: &State,
        boundary: &[[i32; LANES]; MAX_BLOCK_SPINS],
        threshold: i32,
    ) -> u64 {
        let mut delta = [0; LANES];
        for (&node, fields) in self.nodes.iter().zip(boundary) {
            for (lane, value) in delta.iter_mut().enumerate() {
                *value -= spin(state.spins[node], lane) * fields[lane];
            }
        }
        let mut mask = 0;
        for (lane, &value) in delta.iter().enumerate() {
            if value <= threshold {
                mask |= 1 << lane;
            }
        }
        mask
    }

    fn minimize(
        &self,
        graph: &WeightedGraph,
        state: &mut State,
        boundary: &[[i32; LANES]; MAX_BLOCK_SPINS],
    ) {
        let mut best = [0; LANES];
        let mut current = [self.energies[0]; LANES];
        let mut best_words = [0; MAX_BLOCK_SPINS];
        for (i, &node) in self.nodes.iter().enumerate() {
            best_words[i] = state.spins[node];
        }
        for lane in 0..LANES {
            let mut bits = 0usize;
            for (i, &word) in best_words[..self.nodes.len()].iter().enumerate() {
                bits |= (((word >> lane) & 1) as usize) << i;
                best[lane] += spin(word, lane) * boundary[i][lane];
                current[lane] += boundary[i][lane];
            }
            best[lane] += self.energies[bits];
        }
        let mut previous = 0;
        for index in 0usize..self.energies.len() {
            let bits = index ^ (index >> 1);
            if index > 0 {
                let changed = index.trailing_zeros() as usize;
                let factor = if bits & (1 << changed) == 0 { 2 } else { -2 };
                let internal_delta = self.energies[bits] - self.energies[previous];
                for (value, &field) in current.iter_mut().zip(&boundary[changed]) {
                    *value += factor * field + internal_delta;
                }
            }
            let mut mask = 0;
            for (lane, (incumbent, &candidate)) in best.iter_mut().zip(&current).enumerate() {
                if candidate < *incumbent {
                    *incumbent = candidate;
                    mask |= 1 << lane;
                }
            }
            for (i, word) in best_words[..self.nodes.len()].iter_mut().enumerate() {
                if bits & (1 << i) == 0 {
                    *word &= !mask;
                } else {
                    *word |= mask;
                }
            }
            previous = bits;
        }
        for (&node, &word) in self.nodes.iter().zip(&best_words) {
            state.apply(graph, node, state.spins[node] ^ word);
        }
    }
}

#[inline]
fn spin(word: u64, lane: usize) -> i32 {
    1 - 2 * ((word >> lane) & 1) as i32
}

struct State {
    spins: Vec<u64>,
    fields: Vec<[i32; LANES]>,
}

impl State {
    fn new(graph: &WeightedGraph, packed: MscState) -> Self {
        let spins = packed.spins().to_vec();
        let mut fields = vec![[0; LANES]; spins.len()];
        for (u, neighbors) in graph.neighbors.iter().enumerate() {
            for &(v, j) in neighbors {
                for (lane, field) in fields[u].iter_mut().enumerate() {
                    *field += j * spin(spins[v], lane);
                }
            }
        }
        Self { spins, fields }
    }

    fn apply(&mut self, graph: &WeightedGraph, node: usize, mask: u64) {
        if mask == 0 {
            return;
        }
        let old = self.spins[node];
        self.spins[node] ^= mask;
        if mask.count_ones() < 16 {
            for &(neighbor, j) in &graph.neighbors[node] {
                let mut remaining = mask;
                while remaining != 0 {
                    let lane = remaining.trailing_zeros() as usize;
                    self.fields[neighbor][lane] -= 2 * j * spin(old, lane);
                    remaining &= remaining - 1;
                }
            }
        } else {
            let mut changes = [0; LANES];
            for (lane, change) in changes.iter_mut().enumerate() {
                *change = if mask & (1 << lane) == 0 {
                    0
                } else {
                    -2 * spin(old, lane)
                };
            }
            for &(neighbor, j) in &graph.neighbors[node] {
                for (field, &change) in self.fields[neighbor].iter_mut().zip(&changes) {
                    *field += j * change;
                }
            }
        }
    }

    fn single(&mut self, graph: &WeightedGraph, node: usize, threshold: i32) {
        let mut mask = 0;
        for (lane, &field) in self.fields[node].iter().enumerate() {
            if -spin(self.spins[node], lane) * field <= threshold {
                mask |= 1 << lane;
            }
        }
        self.apply(graph, node, mask);
    }
}

fn threshold(rng: &mut SmallRng, beta: f64) -> i32 {
    let u: f64 = rng.sample(Open01);
    (-u.ln() * SCALE / (2.0 * beta))
        .floor()
        .min(i32::MAX as f64) as i32
}

/// A graph ready to sample: the exact milliweight model, the tile blocks,
/// and the ice topology. Building it checks every restriction of the kernel.
pub struct DimerGraph<'a> {
    graph: &'a IsingGraph,
    weighted: WeightedGraph,
    groups: Vec<Group>,
    topology: ice::IceTopology,
}

impl<'a> DimerGraph<'a> {
    /// Check the job and prepare it.
    ///
    /// The job must have zero fields and exact bounded milliweight
    /// couplings. Each tile must be a four-spin block with six distinct
    /// 1000-milli bonds that form a local gauge of antiferromagnetic ice, and
    /// a spin may sit in at most two tiles. Each pair must be a distinct
    /// edge of coupling -1000 between two spins that no other pair uses.
    ///
    /// # Errors
    ///
    /// [`DimerError::Invalid`] naming the first violated restriction.
    pub fn new(
        graph: &'a IsingGraph,
        tiles: &[[u32; 4]],
        pairs: &[(u32, u32)],
    ) -> Result<Self, DimerError> {
        let weighted = WeightedGraph::new(graph)?;
        let nodes = graph.h.len();
        let mut paired = vec![false; nodes];
        for &(u, v) in pairs {
            let (u, v) = (u as usize, v as usize);
            if u == v || u >= nodes || v >= nodes || paired[u] || paired[v] {
                return invalid("a pair repeats a node or names a node outside the graph");
            }
            paired[u] = true;
            paired[v] = true;
            let planted = weighted.neighbors[u]
                .iter()
                .any(|&(other, j)| other == v && j == -STRONG);
            if !planted {
                return invalid("a pair is not an edge of coupling -1000");
            }
        }
        let blocks: Vec<Vec<usize>> = tiles
            .iter()
            .map(|tile| tile.iter().map(|&node| node as usize).collect())
            .collect();
        let groups = Group::build(&weighted, &blocks)?;
        let topology = ice::IceTopology::new(&weighted, &groups).map_err(DimerError::Invalid)?;
        Ok(Self {
            graph,
            weighted,
            groups,
            topology,
        })
    }

    /// Anneal `params.num_reads` random starts. Read `r` starts from the
    /// stream `read_rng(seed, 64 * (r / 64))`, as in the research kernel, and
    /// the sweeps run once per rung of the job's beta ladder, repeated
    /// `sweeps_per_beta` times. The cancel token is checked before each sweep.
    ///
    /// # Errors
    ///
    /// [`DimerError::Invalid`] for a read count, sweep count, or ladder that
    /// the kernel cannot run, and [`DimerError::Cancelled`] when the token
    /// abandons the watermark.
    pub fn sample(
        &self,
        params: &SampleParams,
        method: Method,
        cancel: Option<(&CancelToken, Option<u64>)>,
    ) -> Result<(Vec<Vec<i8>>, IceLoopStats), DimerError> {
        if !(1..=MAX_READS).contains(&params.num_reads) {
            return invalid("num_reads is outside the supported range");
        }
        let per_beta = params.sweeps_per_beta.max(1);
        if params.num_sweeps == 0 || !params.num_sweeps.is_multiple_of(per_beta) {
            return invalid("num_sweeps must be a positive whole number of beta blocks");
        }
        if let Some((hot, cold)) = params.beta_range {
            if !(hot.is_finite() && cold.is_finite() && hot > 0.0 && cold > 0.0) {
                return invalid("the beta range must be positive and finite");
            }
        }
        anneal(
            self.graph,
            &self.weighted,
            &self.groups,
            &self.topology,
            params,
            method,
            cancel,
        )
    }
}

fn anneal(
    graph: &IsingGraph,
    weighted: &WeightedGraph,
    groups: &[Group],
    topology: &ice::IceTopology,
    params: &SampleParams,
    method: Method,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<(Vec<Vec<i8>>, IceLoopStats), DimerError> {
    let coloring = Coloring::new(&CpuGraph::from_base(graph));
    let mut states = Vec::new();
    for read in (0..params.num_reads).step_by(LANES) {
        let mut rng = read_rng(params.seed, read);
        let packed = MscState::random(graph.h.len(), &mut rng);
        states.push(State::new(weighted, packed));
    }
    let betas = build_beta_schedule(graph, params);
    let mut single_rng = SmallRng::seed_from_u64(params.seed ^ SINGLE_SALT);
    let mut group_rng = SmallRng::seed_from_u64(params.seed ^ GROUP_SALT);
    let mut ice_rng = SmallRng::seed_from_u64(params.seed ^ ICE_SALT);
    let mut workspace = ice::IceWorkspace::new(topology);
    for beta in betas {
        for _ in 0..params.sweeps_per_beta.max(1) {
            if cancel.is_some_and(|(token, watermark)| token.is_cancelled(watermark)) {
                return Err(DimerError::Cancelled);
            }
            for class in coloring.classes() {
                for &node in class {
                    let cut = threshold(&mut single_rng, beta);
                    for state in &mut states {
                        state.single(weighted, node as usize, cut);
                    }
                }
            }
            for group in groups {
                let cut = threshold(&mut group_rng, beta);
                for state in &mut states {
                    let boundary = group.boundary(state);
                    let mask = group.flip_mask(state, &boundary, cut);
                    for &node in &group.nodes {
                        state.apply(weighted, node, mask);
                    }
                    // Flipping a block leaves its boundary spins unchanged.
                    group.minimize(weighted, state, &boundary);
                }
            }
            for (chunk, state) in states.iter_mut().enumerate() {
                for lane in 0..LANES.min(params.num_reads - chunk * LANES) {
                    for attempt in 0..method.ice_attempts {
                        if method.defects && attempt % 2 == 1 {
                            workspace.defect_step(
                                topology,
                                weighted,
                                state,
                                lane,
                                beta,
                                &mut ice_rng,
                            );
                        } else {
                            workspace.step(topology, weighted, state, lane, beta, &mut ice_rng);
                        }
                    }
                }
            }
        }
    }
    let spins = (0..params.num_reads)
        .map(|read| {
            states[read / LANES]
                .spins
                .iter()
                .map(|&word| spin(word, read % LANES) as i8)
                .collect()
        })
        .collect();
    Ok((spins, workspace.stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `k`-spin block with alternating signs among its members and weak
    /// bonds to `outside` spins, as one group.
    fn fixture(k: usize, outside: usize) -> (IsingGraph, Vec<Vec<usize>>) {
        let mut edges = Vec::new();
        let mut couplings = Vec::new();
        for u in 0..k {
            for v in u + 1..k {
                edges.push((u, v));
                couplings.push(if (u + v) % 3 == 0 { 1.0 } else { -1.0 });
            }
            for v in k..k + outside {
                edges.push((u, v));
                let magnitude =
                    f64::from(u32::try_from((17 * u + 31 * v) % 125 + 1).unwrap_or(1)) / 1000.0;
                couplings.push(if v % 2 == 0 { magnitude } else { -magnitude });
            }
        }
        (
            IsingGraph::new(vec![0.0; k + outside], couplings, edges),
            vec![(0..k).collect()],
        )
    }

    fn packed_fixture(graph: &WeightedGraph) -> State {
        let mut rng = SmallRng::seed_from_u64(891);
        State::new(graph, MscState::random(graph.neighbors.len(), &mut rng))
    }

    fn lane_spins(state: &State, index: usize) -> Vec<i8> {
        state
            .spins
            .iter()
            .map(|&word| spin(word, index) as i8)
            .collect()
    }

    /// The exact energy in milli of `spins` on `graph`, from the job
    /// coefficients.
    fn energy(graph: &IsingGraph, spins: &[i8]) -> i64 {
        graph
            .edges
            .iter()
            .zip(&graph.j)
            .map(|(&(u, v), &j)| {
                (j * 1000.0).round() as i64 * i64::from(spins[u]) * i64::from(spins[v])
            })
            .sum()
    }

    fn assert_cache(graph: &WeightedGraph, state: &State) {
        for (node, neighbors) in graph.neighbors.iter().enumerate() {
            for lane in 0..LANES {
                let direct: i32 = neighbors
                    .iter()
                    .map(|&(other, j)| j * spin(state.spins[other], lane))
                    .sum();
                assert_eq!(state.fields[node][lane], direct, "node {node}, lane {lane}");
            }
        }
    }

    #[test]
    fn block_flip_deltas_exclude_all_internal_bonds_at_large_degree() {
        for k in [2, 4, 6] {
            let (base, groups) = fixture(k, 140);
            let graph = WeightedGraph::new(&base).expect("milliweights");
            let group = Group::build(&graph, &groups).expect("blocks").remove(0);
            let mut state = packed_fixture(&graph);
            let boundary = group.boundary(&state);
            for threshold in [0, 1, 37, 125, 1000, 8000, i32::MAX] {
                let mask = group.flip_mask(&state, &boundary, threshold);
                for index in 0..LANES {
                    let before = lane_spins(&state, index);
                    let mut after = before.clone();
                    for &node in &group.nodes {
                        after[node] *= -1;
                    }
                    let delta = energy(&base, &after) - energy(&base, &before);
                    assert_eq!((mask >> index) & 1 == 1, delta <= 2 * i64::from(threshold));
                }
            }
            for &node in &group.nodes {
                state.apply(&graph, node, 0xf0f0_0000_ffff_0123);
            }
            assert_cache(&graph, &state);
        }
    }

    #[test]
    fn exact_blocks_reach_the_boundary_conditioned_minimum_in_every_lane() {
        for k in [2, 4, 6] {
            let (base, groups) = fixture(k, 3);
            let graph = WeightedGraph::new(&base).expect("milliweights");
            let group = Group::build(&graph, &groups).expect("blocks").remove(0);
            let mut state = packed_fixture(&graph);
            let before: Vec<_> = (0..LANES).map(|index| lane_spins(&state, index)).collect();
            let boundary = group.boundary(&state);
            group.minimize(&graph, &mut state, &boundary);
            for (index, initial) in before.iter().enumerate() {
                let optimum = (0usize..1 << k)
                    .map(|bits| {
                        let mut candidate = initial.clone();
                        for (node, s) in candidate.iter_mut().take(k).enumerate() {
                            *s = if bits & (1 << node) == 0 { 1 } else { -1 };
                        }
                        energy(&base, &candidate)
                    })
                    .min()
                    .expect("nonempty enumeration");
                let actual = lane_spins(&state, index);
                assert_eq!(energy(&base, &actual), optimum, "k {k}, lane {index}");
                assert_eq!(actual[k..], initial[k..], "boundary spins stay fixed");
            }
            assert_cache(&graph, &state);
        }
    }

    #[test]
    fn blocks_outside_two_through_eight_spins_are_refused() {
        let (base, _) = fixture(9, 1);
        let graph = WeightedGraph::new(&base).expect("milliweights");
        for nodes in [vec![0], (0..9).collect::<Vec<_>>(), vec![]] {
            assert!(Group::build(&graph, &[nodes]).is_err());
        }
        assert!(Group::build(&graph, &[(0..8).collect()]).is_ok());
    }

    #[test]
    fn bad_block_memberships_are_refused() {
        let (base, _) = fixture(4, 2);
        let graph = WeightedGraph::new(&base).expect("milliweights");
        for nodes in [vec![0, 1, 1, 2], vec![0, 1, 2, 99]] {
            assert!(Group::build(&graph, &[nodes]).is_err(), "repeat or range");
        }
        // Spin 5 has only weak bonds to spins 0 through 3, so {4, 5} has no
        // bond between its members.
        assert!(Group::build(&graph, &[vec![4, 5]]).is_err(), "unconnected");
        // Blocks may share a spin, as tetrahedra of the family do.
        assert!(Group::build(&graph, &[vec![0, 1, 2, 3], vec![3, 4]]).is_ok());
    }

    #[test]
    fn fields_inexact_couplings_and_overflowing_sums_are_refused() {
        let edges = vec![(0, 1), (1, 2)];
        let with =
            |h: Vec<f64>, j: Vec<f64>| WeightedGraph::new(&IsingGraph::new(h, j, edges.clone()));
        assert!(with(vec![0.0; 3], vec![1.0, -0.062]).is_ok());
        assert!(
            with(vec![0.0, 0.001, 0.0], vec![1.0, 1.0]).is_err(),
            "field"
        );
        assert!(with(vec![0.0; 3], vec![1.0, 0.0005]).is_err(), "inexact");
        assert!(
            with(vec![0.0; 3], vec![1.0, f64::NAN]).is_err(),
            "not finite"
        );
        assert!(with(vec![0.0; 3], vec![1.0, 1.0e9]).is_err(), "oversize");
        let many = (0..300_000usize).map(|i| (i, i + 1)).collect::<Vec<_>>();
        let heavy = IsingGraph::new(vec![0.0; 300_001], vec![500.0; many.len()], many);
        assert!(WeightedGraph::new(&heavy).is_err(), "sum overflow");
        let looped = IsingGraph::new(vec![0.0; 2], vec![1.0], vec![(0, 0)]);
        assert!(WeightedGraph::new(&looped).is_err(), "self loop");
    }
}
