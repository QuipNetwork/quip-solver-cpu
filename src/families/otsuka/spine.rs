//! Exact conditional minimization on two-lane Zephyr spines.

mod batch;

use std::{cmp::Reverse, collections::BinaryHeap, ops::Range, sync::OnceLock};

use quip_solver_core::ising_generators::ice::zephyr_coordinates;
use quip_solver_core::SamplerResult;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

use super::{graph::Link, OtsukaGraph};
use batch::{Batch, LANES};

const MAX_WIDTH: usize = 9;
const MAX_ASSIGNMENTS: usize = 1 << MAX_WIDTH;
const OUTSIDE: usize = usize::MAX;
const PLAN_CACHE_BYTES: usize = 64 << 20;

/// A collection of conditional neighborhoods on a validated Zephyr instance.
///
/// Each neighborhood frees parallel spines in one Zephyr row and all their
/// perpendicular neighbors. Other spins supply fixed boundary fields.
pub struct ZephyrSpinePolish<'a> {
    graph: &'a OtsukaGraph,
    spines: Vec<Vec<usize>>,
    adjacent_spines: Vec<Vec<usize>>,
    lanes: usize,
}

/// Budget and move policy for conditional Zephyr search.
#[derive(Clone, Copy, Debug)]
pub struct SpinePolishOptions {
    /// Number of block minimizations per input candidate.
    pub blocks: usize,
    /// Number of center spines freed in one row, from one to four.
    pub span: usize,
    /// Evaluate separate reads in groups of eight, with strictly downhill moves.
    pub batch_reads: bool,
    /// Retain up to 64 MiB of plan buffers for repeated neighborhoods.
    pub cache_plans: bool,
}

/// Neighborhoods visited by a bounded polish.
#[derive(Clone, Debug, Default)]
pub struct SpinePolishStats {
    /// Number of exact move plans used.
    pub blocks: usize,
    /// Number of plans built during this pass.
    pub plans_built: usize,
    /// Allocated bytes retained in the plans' buffers for reuse.
    pub cached_plan_bytes: usize,
    /// Minimum number of free spins among the plans used.
    pub min_free: usize,
    /// Maximum number of free spins among the plans used.
    pub max_free: usize,
    /// Maximum number of surviving neighbors eliminated together with a spin.
    pub max_width: usize,
    /// Strictly downhill moves applied across the candidates.
    pub accepted: usize,
    /// Total canonical energy decrease between input and output candidates.
    pub gain_milli: i64,
    /// Dense Zephyr spine indices used to center the first move.
    pub first_spines: Vec<usize>,
}

impl<'a> ZephyrSpinePolish<'a> {
    /// Index the spines of a graph with canonical node IDs for shape `z(m,t)`.
    /// Returns `None` for incompatible IDs or edges between parallel spines.
    pub fn new(graph: &'a OtsukaGraph, ids: &[u32], m: u16, t: u8) -> Option<Self> {
        if !(1..=32).contains(&m)
            || !(1..=8).contains(&t)
            || ids.len() != graph.n
            || !ids.windows(2).all(|pair| pair[0] < pair[1])
        {
            return None;
        }
        let per_axis = (2 * usize::from(m) + 1) * usize::from(t);
        let mut spines = vec![Vec::new(); 2 * per_axis];
        let mut spine_of = Vec::with_capacity(ids.len());
        for (node, &id) in ids.iter().enumerate() {
            let [axis, row, lane, _, _] = zephyr_coordinates(m, t, id)?;
            let spine = axis as usize * per_axis + row as usize * usize::from(t) + lane as usize;
            spines[spine].push(node);
            spine_of.push(spine);
        }
        let mut adjacent_spines = vec![Vec::new(); spines.len()];
        for (u, &from) in spine_of.iter().enumerate() {
            for &edge in graph.all.row(u) {
                let to = spine_of[edge.node()];
                if from == to || edge.j() == 0 {
                    continue;
                }
                if from / per_axis == to / per_axis {
                    return None;
                }
                adjacent_spines[from].push(to);
            }
        }
        for adjacent in &mut adjacent_spines {
            adjacent.sort_unstable();
            adjacent.dedup();
        }
        Some(Self {
            graph,
            spines,
            adjacent_spines,
            lanes: usize::from(t),
        })
    }

    fn centers(&self, span: usize) -> Vec<Vec<usize>> {
        let mut centers = Vec::new();
        for row in (0..self.spines.len()).step_by(self.lanes) {
            let active: Vec<_> = (row..row + self.lanes)
                .filter(|&i| !self.spines[i].is_empty())
                .collect();
            if active.is_empty() {
                continue;
            }
            if span >= active.len() {
                centers.push(active);
                continue;
            }
            for start in 0..active.len() {
                let mut group: Vec<_> = active
                    .iter()
                    .cycle()
                    .skip(start)
                    .take(span)
                    .copied()
                    .collect();
                group.sort_unstable();
                centers.push(group);
            }
        }
        centers
    }

    fn plan_for(&self, group: &[usize], stopped: &impl Fn() -> bool) -> Option<Plan> {
        let mut nodes: Vec<_> = group
            .iter()
            .flat_map(|&i| &self.spines[i])
            .copied()
            .collect();
        let central = nodes.len();
        let mut neighbors: Vec<_> = group
            .iter()
            .flat_map(|&i| &self.adjacent_spines[i])
            .copied()
            .collect();
        neighbors.sort_unstable();
        neighbors.dedup();
        for spine in neighbors {
            nodes.extend_from_slice(&self.spines[spine]);
        }
        Plan::new(self.graph, &nodes, central, stopped)
    }

    /// Try a bounded sequence of exact conditional moves on every supplied
    /// read. The span is the number of center spines in one row, up to four.
    /// Applies only strict improvements to the original energy.
    /// Returns `None` if stopped or a read or order violates the assumptions.
    pub fn improve(
        &self,
        reads: &mut [SamplerResult],
        options: SpinePolishOptions,
        seed: u64,
        stopped: impl Fn() -> bool,
    ) -> Option<SpinePolishStats> {
        let SpinePolishOptions {
            blocks,
            span,
            batch_reads,
            cache_plans,
        } = options;
        if !(1..=4).contains(&span)
            || (batch_reads && !Batch::fits(self.graph))
            || reads.iter().any(|r| {
                r.spins.len() != self.graph.n
                    || r.spins.iter().any(|&s| s != 1 && s != -1)
                    || self.graph.energy(&r.spins) != r.energy_milli
            })
        {
            return None;
        }
        let centers = self.centers(span);
        let mut order: Vec<_> = (0..centers.len()).collect();
        let mut cache = Vec::<Option<Plan>>::new();
        if cache_plans && blocks > centers.len() {
            cache.resize_with(centers.len(), || None);
        }
        let mut rng = SmallRng::seed_from_u64(seed ^ 0x5a45_5048_5952_4450);
        let mut batches = batch_reads.then(|| {
            reads
                .chunks(LANES)
                .map(|chunk| Batch::new(chunk, self.graph.n))
                .collect::<Vec<_>>()
        });
        let first_energy: i64 = reads.iter().map(|read| read.energy_milli).sum();
        let mut scratch = Scratch::default();
        let mut stats = SpinePolishStats::default();
        let mut done = 0;
        while done < blocks {
            order.shuffle(&mut rng);
            for &index in order.iter().take(blocks - done) {
                if stopped() {
                    return None;
                }
                let group = &centers[index];
                let slot = cache.get_mut(index);
                let mut temporary = None;
                if slot.as_ref().is_none_or(|plan| plan.is_none()) {
                    let plan = self.plan_for(group, &stopped)?;
                    stats.plans_built += 1;
                    temporary = Some(plan);
                }
                let uncached;
                let plan = match (slot, temporary) {
                    (Some(slot), Some(new))
                        if new.heap_bytes() <= PLAN_CACHE_BYTES - stats.cached_plan_bytes =>
                    {
                        stats.cached_plan_bytes += new.heap_bytes();
                        slot.insert(new)
                    }
                    (Some(slot), None) => slot.as_ref()?,
                    (_, Some(plan)) => {
                        uncached = plan;
                        &uncached
                    }
                    (None, None) => return None,
                };
                let free = plan.steps.len();
                if stats.blocks == 0 {
                    stats.min_free = free;
                    stats.first_spines.clone_from(group);
                }
                stats.blocks += 1;
                stats.min_free = stats.min_free.min(free);
                stats.max_free = stats.max_free.max(free);
                stats.max_width = stats.max_width.max(plan.width);
                if let Some(batches) = &mut batches {
                    for (batch, chunk) in batches.iter_mut().zip(reads.chunks_mut(LANES)) {
                        let gains = batch.descend(plan, &stopped)?;
                        for (read, gain) in chunk.iter_mut().zip(gains) {
                            read.energy_milli -= gain;
                            stats.accepted += usize::from(gain > 0);
                        }
                    }
                } else {
                    for read in reads.iter_mut() {
                        let gain = plan.descend(&mut read.spins, &mut scratch, &stopped)?;
                        read.energy_milli -= gain;
                        stats.accepted += usize::from(gain > 0);
                    }
                }
                done += 1;
            }
        }
        if let Some(batches) = batches {
            for (batch, chunk) in batches.iter().zip(reads.chunks_mut(LANES)) {
                batch.store(chunk);
            }
        }
        stats.gain_milli = first_energy - reads.iter().map(|read| read.energy_milli).sum::<i64>();
        reads
            .iter()
            .all(|read| self.graph.energy(&read.spins) == read.energy_milli)
            .then_some(stats)
    }
}

struct Step {
    node: usize,
    later: [usize; MAX_WIDTH],
    width: usize,
    table: Range<usize>,
    boundary: Range<usize>,
    incoming: Range<usize>,
}

struct Plan {
    steps: Vec<Step>,
    links: Vec<Link>,
    messages: Vec<Message>,
    direct: Vec<i16>,
    width: usize,
}

struct Message {
    base: u32,
    projection: u16,
}

impl Message {
    fn index(&self, projections: &[[u8; MAX_ASSIGNMENTS]], assignment: usize) -> usize {
        self.base as usize + 2 * usize::from(projections[self.projection as usize][assignment])
    }
}

/// Pack the selected bits of each assignment in ascending order. An incoming
/// message always uses one bit for the spin being eliminated, so at most
/// eight bits are projected from the surviving assignment. The topology and
/// the coupling signs do not affect these tables.
fn projection_tables() -> &'static [[u8; MAX_ASSIGNMENTS]] {
    static TABLES: OnceLock<Box<[[u8; MAX_ASSIGNMENTS]]>> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut rows = vec![[0; MAX_ASSIGNMENTS]; MAX_ASSIGNMENTS - 1];
        for mask in 1..rows.len() {
            let rest = rows[mask & (mask - 1)];
            let lowest = mask.trailing_zeros();
            for (assignment, value) in rows[mask].iter_mut().enumerate() {
                *value = (rest[assignment] << 1) | ((assignment >> lowest) & 1) as u8;
            }
        }
        rows.into_boxed_slice()
    })
}

/// Eliminate the perpendicular spines first. Keep the original edges and all
/// fill edges in the adjacency used to pick the next minimum-degree vertex.
fn elimination_order(
    graph: &OtsukaGraph,
    nodes: &[usize],
    local: &[usize],
    central: usize,
    stopped: &impl Fn() -> bool,
) -> Option<Vec<Vec<usize>>> {
    let mut adjacent: Vec<Vec<usize>> = nodes
        .iter()
        .map(|&u| {
            graph
                .all
                .row(u)
                .iter()
                .filter_map(|link| {
                    let v = local[link.node()];
                    (v != OUTSIDE && link.j() != 0).then_some(v)
                })
                .collect()
        })
        .collect();
    let key = |v: usize, degree: usize| Reverse((v < central, degree, nodes[v], v));
    let mut queue: BinaryHeap<_> = adjacent
        .iter()
        .enumerate()
        .map(|(v, row)| key(v, row.len()))
        .collect();
    let mut removed = vec![false; nodes.len()];
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(Reverse((_, degree, _, u))) = queue.pop() {
        if removed[u] || degree != adjacent[u].len() {
            continue;
        }
        if degree > MAX_WIDTH || (order.len().is_multiple_of(256) && stopped()) {
            return None;
        }
        let row = std::mem::take(&mut adjacent[u]);
        removed[u] = true;
        for &v in &row {
            let delete = adjacent[v].iter().position(|&w| w == u)?;
            adjacent[v].swap_remove(delete);
            for &w in &row {
                if w != v && !adjacent[v].contains(&w) {
                    adjacent[v].push(w);
                }
            }
            queue.push(key(v, adjacent[v].len()));
        }
        let mut scope = Vec::with_capacity(row.len() + 1);
        scope.push(u);
        scope.extend(row);
        order.push(scope);
    }
    Some(order)
}

impl Plan {
    fn new(
        graph: &OtsukaGraph,
        nodes: &[usize],
        central: usize,
        stopped: &impl Fn() -> bool,
    ) -> Option<Self> {
        let mut local = vec![OUTSIDE; graph.n];
        for (i, &u) in nodes.iter().enumerate() {
            local[u] = i;
        }
        let order = elimination_order(graph, nodes, &local, central, stopped)?;
        let mut rank = vec![0; nodes.len()];
        for (i, scope) in order.iter().enumerate() {
            rank[scope[0]] = i;
        }

        let mut plan = Self {
            steps: Vec::with_capacity(nodes.len()),
            links: Vec::new(),
            messages: Vec::new(),
            direct: Vec::new(),
            width: 0,
        };
        let mut buckets = vec![Vec::<usize>::new(); nodes.len()];
        for mut scope in order {
            let u = nodes[scope[0]];
            scope[1..].sort_unstable_by_key(|&v| rank[v]);
            let width = scope.len() - 1;
            let here = plan.steps.len();
            let boundary_start = plan.links.len();
            // At most nine original couplings, each bounded by 1,000 milli,
            // contribute to this table. Boundary fields and incoming
            // factors are accumulated separately in the wider score type.
            let mut coeff = [0_i16; MAX_WIDTH];
            for &link in graph.all.row(u) {
                let v = local[link.node()];
                if v == OUTSIDE {
                    plan.links.push(link);
                } else if let Some(p) = scope[1..].iter().position(|&p| p == v) {
                    coeff[p] = link.j() as i16;
                }
            }
            let mut later = [0; MAX_WIDTH];
            let table = plan.direct.len()..plan.direct.len() + (1 << width);
            plan.direct.resize(table.end, 0);
            let direct = &mut plan.direct[table.clone()];
            direct[0] = -coeff[..width].iter().sum::<i16>();
            for (i, &v) in scope[1..].iter().enumerate() {
                later[i] = nodes[v];
                let bit = 1 << i;
                for mask in 0..bit {
                    direct[mask + bit] = direct[mask] + 2 * coeff[i];
                }
            }

            let incoming_start = plan.messages.len();
            for &from in &buckets[here] {
                let source = &plan.steps[from];
                let mut projection = 0;
                // Both scopes are sorted by elimination rank. The first
                // source bit is this step's spin, with its two costs adjacent.
                for &v in &source.later[1..source.width] {
                    let to = later[..width].iter().position(|&p| p == v)?;
                    projection |= 1 << to;
                }
                plan.messages.push(Message {
                    base: u32::try_from(source.table.start).ok()?,
                    projection,
                });
            }
            if let Some(&first) = scope.get(1) {
                buckets[rank[first]].push(here);
            }
            plan.steps.push(Step {
                node: u,
                later,
                width,
                table,
                boundary: boundary_start..plan.links.len(),
                incoming: incoming_start..plan.messages.len(),
            });
            plan.width = plan.width.max(width);
        }
        Some(plan)
    }

    fn heap_bytes(&self) -> usize {
        self.steps.capacity() * size_of::<Step>()
            + self.links.capacity() * size_of::<Link>()
            + self.messages.capacity() * size_of::<Message>()
            + self.direct.capacity() * size_of::<i16>()
    }

    fn descend(
        &self,
        spins: &mut [i8],
        scratch: &mut Scratch,
        stopped: &impl Fn() -> bool,
    ) -> Option<i64> {
        let projections = projection_tables();
        scratch.cost.resize(self.direct.len(), 0);
        scratch.choice.resize(self.direct.len(), false);
        let mut old = 0;
        let mut optimum = 0;
        for (i, step) in self.steps.iter().enumerate() {
            if i.is_multiple_of(256) && stopped() {
                return None;
            }
            let boundary = self.links[step.boundary.clone()]
                .iter()
                .map(|edge| i64::from(edge.j()) * i64::from(spins[edge.node()]))
                .sum::<i64>();
            let prev = spins[step.node];
            let direct = &self.direct[step.table.clone()];
            old += i64::from(prev) * (boundary + i64::from(direct[step.mask(spins)]));
            for (k, &field) in direct.iter().enumerate() {
                let h = boundary + i64::from(field);
                let (mut lo, mut hi) = (-h, h);
                for message in &self.messages[step.incoming.clone()] {
                    let index = message.index(projections, k);
                    lo += scratch.cost[index];
                    hi += scratch.cost[index + 1];
                }
                let at = step.table.start + k;
                scratch.cost[at] = lo.min(hi);
                scratch.choice[at] = hi < lo || (hi == lo && prev == 1);
            }
            if step.width == 0 {
                optimum += scratch.cost[step.table.start];
            }
        }
        let gain = old - optimum;
        if gain < 0 {
            return None;
        }
        if gain > 0 {
            if stopped() {
                return None;
            }
            for step in self.steps.iter().rev() {
                spins[step.node] = if scratch.choice[step.table.start + step.mask(spins)] {
                    1
                } else {
                    -1
                };
            }
        }
        Some(gain)
    }
}

impl Step {
    fn mask(&self, spins: &[i8]) -> usize {
        self.later[..self.width]
            .iter()
            .enumerate()
            .fold(0, |mask, (i, &v)| mask | (usize::from(spins[v] == 1) << i))
    }
}

#[derive(Default)]
struct Scratch {
    cost: Vec<i64>,
    choice: Vec<bool>,
}

#[cfg(test)]
mod tests {
    use super::{projection_tables, Batch, Plan, Scratch, MAX_WIDTH};
    use crate::families::otsuka::{Model, Motifs, OtsukaGraph, Weights};
    use quip_solver_core::SamplerResult;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    #[test]
    fn projection_tables_match_all_explicit_bit_gathers() {
        for (mask, row) in projection_tables().iter().enumerate() {
            for (assignment, &packed) in row.iter().enumerate() {
                let reference = (0..MAX_WIDTH)
                    .filter(|bit| mask & (1 << bit) != 0)
                    .enumerate()
                    .map(|(target, source)| ((assignment >> source) & 1) << target)
                    .sum::<usize>();
                assert_eq!(usize::from(packed), reference);
            }
        }
    }

    #[test]
    fn elimination_matches_every_assignment_with_boundary_spins_fixed() {
        let edges = [
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
            (4, 5),
            (0, 6),
            (2, 7),
            (5, 7),
        ];
        let free = [0, 1, 2, 3, 4, 5];
        let mut rng = SmallRng::seed_from_u64(224);
        for fixture in 0..64 {
            let j = if fixture == 0 {
                [0; 11]
            } else {
                edges.map(|_| rng.gen_range(-5..=5))
            };
            let graph = OtsukaGraph::new(
                &Model {
                    h_milli: &[0; 8],
                    edges: &edges,
                    j_milli: &j,
                },
                &Motifs {
                    tiles: &[],
                    gauges: &[],
                    pairs: &[],
                },
                Weights {
                    tile_milli: 250,
                    pair_milli: 250,
                },
            )
            .unwrap();
            let plan = Plan::new(&graph, &free, 2, &|| false).unwrap();
            let original = std::array::from_fn(|_| if rng.gen() { 1 } else { -1 });
            let mut spins: [i8; 8] = original;
            let best = (0..1 << free.len())
                .map(|mask| {
                    for &v in &free {
                        spins[v] = if mask & (1 << v) == 0 { -1 } else { 1 };
                    }
                    graph.energy(&spins)
                })
                .min()
                .unwrap();
            spins = original;
            let gain = plan
                .descend(&mut spins, &mut Scratch::default(), &|| false)
                .unwrap();
            assert_eq!(graph.energy(&spins), best);
            assert_eq!(gain, graph.energy(&original) - best);
            assert_eq!(spins[6..], original[6..]);
            if gain == 0 {
                assert_eq!(spins, original);
            }
            let mut reads = [SamplerResult {
                spins: original.to_vec(),
                energy_milli: graph.energy(&original),
            }];
            let mut batch = Batch::new(&reads, original.len());
            let batched = batch.descend(&plan, &|| false).unwrap();
            batch.store(&mut reads);
            assert_eq!(batched[0], gain);
            assert_eq!(reads[0].spins, spins);
        }
    }
}
