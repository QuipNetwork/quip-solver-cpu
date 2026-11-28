//! Downhill search over motif components and mixtures of parent assignments.

use super::OtsukaGraph;
use quip_solver_core::SamplerResult;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

const FIXED: usize = usize::MAX;
const MAX_PASSES: usize = 4;

/// Counts and checked energy change from motif descent.
#[derive(Debug, Default)]
pub struct MotifDescentStats {
    /// Sampled component partitions.
    pub partitions: usize,
    /// Largest cluster visited.
    pub max_component: usize,
    /// Strictly downhill component flips.
    pub accepted: usize,
    /// Improving recombinations accepted into the candidate batch.
    pub mixed: usize,
    /// Total decrease of the original energy across the candidates.
    pub gain_milli: i64,
}

/// Apply repeated downhill sweeps over motif-preserving components. In a
/// crossover, spins on which the parents agree are fixed and the full
/// canonical score covers any differences in parental motif energies.
/// A global spin inversion preserves energy because the model has no fields.
///
/// Returns None on cancellation or inconsistent input energies.
pub fn polish_motifs(
    graph: &OtsukaGraph,
    reads: &mut [SamplerResult],
    rounds: usize,
    mixtures: usize,
    seed: u64,
    stopped: &impl Fn() -> bool,
) -> Option<MotifDescentStats> {
    if reads.iter().any(|read| {
        read.spins.len() != graph.n
            || read.spins.iter().any(|&spin| spin != 1 && spin != -1)
            || read.energy_milli != graph.energy(&read.spins)
    }) {
        return None;
    }
    let mut stats = MotifDescentStats::default();
    let mut workspace = Workspace::new(graph.n);
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x4d4f_5449_465f_434c);
    for read in reads.iter_mut() {
        let initial = read.energy_milli;
        for _ in 0..rounds {
            if stopped() {
                return None;
            }
            workspace.partition(graph, &read.spins, None, &mut rng, &mut stats);
            read.energy_milli += workspace.descend(graph, &mut read.spins, &mut rng, &mut stats);
        }
        stats.gain_milli += initial - read.energy_milli;
    }

    if mixtures > 0 && reads.len() > 1 {
        let best = (0..reads.len()).min_by_key(|&index| reads[index].energy_milli)?;
        let mut donors: Vec<_> = (0..reads.len()).filter(|&index| index != best).collect();
        donors.sort_by_key(|&index| reads[index].energy_milli);
        let mut child = Vec::with_capacity(graph.n);
        for trial in 0..mixtures {
            for &donor in &donors {
                if stopped() {
                    return None;
                }
                let a = &reads[best];
                let b = &reads[donor];
                let disagreement = a.spins.iter().zip(&b.spins).filter(|(x, y)| x != y).count();
                let orientation = if disagreement > graph.n / 2 { -1 } else { 1 };
                let orientation = if trial % 4 >= 2 {
                    -orientation
                } else {
                    orientation
                };
                child.clear();
                let (reference, align, start_energy) = if trial % 2 == 0 {
                    child.extend_from_slice(&a.spins);
                    (&b.spins[..], orientation, a.energy_milli)
                } else {
                    child.extend(b.spins.iter().map(|&spin| spin * orientation));
                    (&a.spins[..], 1, b.energy_milli)
                };
                workspace.partition(
                    graph,
                    &child,
                    Some((reference, align)),
                    &mut rng,
                    &mut stats,
                );
                let energy =
                    start_energy + workspace.descend(graph, &mut child, &mut rng, &mut stats);
                if energy < reads[best].energy_milli {
                    stats.gain_milli += reads[best].energy_milli - energy;
                    stats.mixed += 1;
                    reads[best].energy_milli = energy;
                    std::mem::swap(&mut reads[best].spins, &mut child);
                }
            }
        }
    }

    reads
        .iter()
        .all(|read| read.energy_milli == graph.energy(&read.spins))
        .then_some(stats)
}

struct Workspace {
    parents: Vec<usize>,
    labels: Vec<usize>,
    members: Vec<usize>,
    offsets: Vec<usize>,
    deltas: Vec<i64>,
    order: Vec<usize>,
}

impl Workspace {
    fn new(n: usize) -> Self {
        Self {
            parents: vec![0; n],
            labels: vec![0; n],
            members: vec![0; n],
            offsets: Vec::with_capacity(n + 1),
            deltas: Vec::with_capacity(n),
            order: Vec::with_capacity(n),
        }
    }

    fn root(&mut self, mut u: usize) -> usize {
        while self.parents[u] != u {
            let parent = self.parents[u];
            self.parents[u] = self.parents[parent];
            u = parent;
        }
        u
    }

    fn join(&mut self, u: usize, v: usize) {
        let u = self.root(u);
        let v = self.root(v);
        self.parents[u.max(v)] = u.min(v);
    }

    fn partition(
        &mut self,
        graph: &OtsukaGraph,
        spins: &[i8],
        reference: Option<(&[i8], i8)>,
        rng: &mut SmallRng,
        stats: &mut MotifDescentStats,
    ) {
        for (u, parent) in self.parents.iter_mut().enumerate() {
            *parent = if reference.is_some_and(|(other, sign)| other[u] * sign == spins[u]) {
                FIXED
            } else {
                u
            };
        }
        for &(u, v) in &graph.pairs {
            let (u, v) = (u as usize, v as usize);
            if self.parents[u] != FIXED && self.parents[v] != FIXED {
                self.join(u, v);
            }
        }
        for tile in &graph.tiles {
            let mut by_sign = [[0; 4]; 2];
            let mut counts = [0; 2];
            for (&node, &gauge) in tile.nodes.iter().zip(&tile.gauge) {
                let u = node as usize;
                if self.parents[u] != FIXED {
                    let sign = usize::from(spins[u] == gauge);
                    by_sign[sign][counts[sign]] = u;
                    counts[sign] += 1;
                }
            }
            let paired = counts[0].min(counts[1]);
            if paired > 0 {
                by_sign[1][..counts[1]].shuffle(rng);
                by_sign[0][..counts[0]].shuffle(rng);
            }
            // Opposite gauged spins contribute zero charge and remain
            // neutral when flipped together. In a single-parent pass,
            // flipping all unmatched spins only reverses the tile charge.
            let [negative, positive] = &by_sign;
            for (&a, &b) in negative[..paired].iter().zip(&positive[..paired]) {
                self.join(a, b);
            }
            for (nodes, count) in by_sign.iter().zip(counts) {
                if let Some((&first, rest)) = nodes[paired..count].split_first() {
                    for &u in rest {
                        self.join(first, u);
                    }
                }
            }
        }

        self.offsets.clear();
        self.offsets.push(0);
        for (u, &parent) in self.parents.iter().enumerate() {
            let c = if parent == FIXED {
                FIXED
            } else if parent == u {
                let c = self.offsets.len() - 1;
                self.offsets.push(1);
                c
            } else {
                let c = self.labels[parent];
                self.offsets[c + 1] += 1;
                c
            };
            self.labels[u] = c;
        }
        self.deltas.clear();
        for c in 0..self.offsets.len() - 1 {
            let count = self.offsets[c + 1];
            stats.max_component = stats.max_component.max(count);
            self.offsets[c + 1] = self.offsets[c] + count;
            self.deltas.push(self.offsets[c] as i64);
        }
        for (u, &c) in self.labels.iter().enumerate() {
            if c != FIXED {
                self.members[self.deltas[c] as usize] = u;
                self.deltas[c] += 1;
            }
        }
        self.deltas.fill(0);
        for (u, &c) in self.labels.iter().enumerate() {
            if c == FIXED {
                continue;
            }
            let spin = i64::from(spins[u]);
            let mut outside = 0;
            for &edge in graph.all.row(u) {
                if self.labels[edge.node()] != c {
                    outside += i64::from(edge.j()) * i64::from(spins[edge.node()]);
                }
            }
            self.deltas[c] -= 2 * spin * outside;
        }
        self.order.clear();
        self.order.extend(0..self.deltas.len());
        stats.partitions += 1;
    }

    fn descend(
        &mut self,
        graph: &OtsukaGraph,
        spins: &mut [i8],
        rng: &mut SmallRng,
        stats: &mut MotifDescentStats,
    ) -> i64 {
        let mut change = 0;
        for _ in 0..MAX_PASSES {
            self.order.shuffle(rng);
            let before = change;
            for &c in &self.order {
                let delta = self.deltas[c];
                if delta >= 0 {
                    continue;
                }
                for &u in &self.members[self.offsets[c]..self.offsets[c + 1]] {
                    let old = i64::from(spins[u]);
                    for &edge in graph.all.row(u) {
                        let v = edge.node();
                        let other = self.labels[v];
                        if other != c && other != FIXED {
                            self.deltas[other] +=
                                4 * old * i64::from(edge.j()) * i64::from(spins[v]);
                        }
                    }
                    spins[u] = -spins[u];
                }
                self.deltas[c] = -delta;
                change += delta;
                stats.accepted += 1;
            }
            if change == before {
                break;
            }
        }
        change
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::otsuka::{Model, Motifs, Weights};

    fn graph() -> OtsukaGraph {
        let tiles = [[0, 1, 2, 3], [3, 4, 5, 6]];
        let gauges = [[1, -1, 1, -1], [-1, 1, 1, -1]];
        let pairs = [(1, 4), (2, 7)];
        let mut edges = vec![(0, 5), (2, 6), (6, 7), (0, 7)];
        let mut j = vec![23, -45, 67, 125];
        for (nodes, gauge) in tiles.iter().zip(&gauges) {
            for u in 0..4 {
                for v in (u + 1)..4 {
                    edges.push((nodes[u], nodes[v]));
                    j.push(1000 * i32::from(gauge[u] * gauge[v]));
                }
            }
        }
        for pair in pairs {
            edges.push(pair);
            j.push(-1000);
        }
        OtsukaGraph::new(
            &Model {
                h_milli: &[0; 8],
                edges: &edges,
                j_milli: &j,
            },
            &Motifs {
                tiles: &tiles,
                gauges: &gauges,
                pairs: &pairs,
            },
            Weights::FULL,
        )
        .unwrap()
    }

    fn state(mask: usize) -> [i8; 8] {
        std::array::from_fn(|at| if mask & (1 << at) == 0 { -1 } else { 1 })
    }

    #[test]
    fn motif_groups_preserve_strong_energy_and_track_exact_changes() {
        let graph = graph();
        let mut workspace = Workspace::new(graph.n);
        let mut rng = SmallRng::seed_from_u64(7302);
        let mut stats = MotifDescentStats::default();
        for mask in 0..256 {
            for reference in [None, Some(state(mask * 221 + 35))] {
                let mut spins = state(mask);
                let original = spins;
                let before = graph.energy(&spins);
                workspace.partition(
                    &graph,
                    &spins,
                    reference.as_ref().map(|other| (&other[..], 1)),
                    &mut rng,
                    &mut stats,
                );
                for (group, window) in workspace.offsets.windows(2).enumerate() {
                    let mut flipped = spins;
                    for &u in &workspace.members[window[0]..window[1]] {
                        flipped[u] = -flipped[u];
                    }
                    assert_eq!(graph.energy(&flipped) - before, workspace.deltas[group]);
                    if reference.is_none() {
                        assert_eq!(
                            graph.full_strong_energy(&spins),
                            graph.full_strong_energy(&flipped)
                        );
                    }
                }
                let change = workspace.descend(&graph, &mut spins, &mut rng, &mut stats);
                assert!(change <= 0);
                assert_eq!(graph.energy(&spins) - before, change);
                if let Some(reference) = reference {
                    for ((&initial, &other), &after) in original.iter().zip(&reference).zip(&spins)
                    {
                        if initial == other {
                            assert_eq!(initial, after);
                        }
                    }
                }
            }
        }
    }
}
