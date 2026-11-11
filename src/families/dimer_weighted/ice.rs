//! Alternating loops and boundary strings in a gauge-antiferromagnetic ice layer.

use super::{spin, threshold, Group, IceLoopStats, State, WeightedGraph};
use rand::rngs::SmallRng;
use rand::Rng;
use std::collections::BTreeSet;

const BOUNDARY: usize = usize::MAX;

pub(super) struct IceTopology {
    tetrahedra: Vec<[usize; 4]>,
    membership: Vec<[usize; 2]>,
    gauge: Vec<[i32; 2]>,
    covered: Vec<usize>,
}

impl IceTopology {
    pub(super) fn new(graph: &WeightedGraph, groups: &[Group]) -> Result<Self, String> {
        if groups.is_empty() {
            return Err("ice moves require explicit tetrahedra".to_owned());
        }
        let mut membership = vec![[BOUNDARY; 2]; graph.neighbors.len()];
        let mut gauge = vec![[0; 2]; graph.neighbors.len()];
        let mut tetrahedra = Vec::with_capacity(groups.len());
        let mut strong_edges = BTreeSet::new();
        for (index, group) in groups.iter().enumerate() {
            if group.nodes.len() != 4 || group.internal.len() != 6 {
                return Err("ice groups must be complete four-spin tetrahedra".to_owned());
            }
            let mut local_gauge = [0; 4];
            local_gauge[0] = 1;
            for &(u, v, j) in &group.internal {
                let a = group.nodes[u];
                let b = group.nodes[v];
                if j.abs() != 1000 || !strong_edges.insert((a.min(b), a.max(b))) {
                    return Err("ice tetrahedra need distinct 1000-milli edges".to_owned());
                }
                if u == 0 {
                    local_gauge[v] = j.signum();
                } else if v == 0 {
                    local_gauge[u] = j.signum();
                }
            }
            if group
                .internal
                .iter()
                .any(|&(u, v, j)| local_gauge[u] * local_gauge[v] != j.signum())
            {
                return Err(
                    "strong signs are not a local gauge of antiferromagnetic ice".to_owned(),
                );
            }
            for (position, &node) in group.nodes.iter().enumerate() {
                let Some((slot_index, slot)) = membership[node]
                    .iter_mut()
                    .enumerate()
                    .find(|(_, slot)| **slot == BOUNDARY)
                else {
                    return Err("an ice spin belongs to at most two tetrahedra".to_owned());
                };
                *slot = index;
                gauge[node][slot_index] = local_gauge[position];
            }
            tetrahedra.push([
                group.nodes[0],
                group.nodes[1],
                group.nodes[2],
                group.nodes[3],
            ]);
        }
        // Extra strong bonds, including dimers, are part of the full-model
        // Metropolis delta. Only the explicit tetrahedra define loop topology.
        let covered: Vec<_> = membership
            .iter()
            .enumerate()
            .filter_map(|(node, members)| (members[0] != BOUNDARY).then_some(node))
            .collect();
        Ok(Self {
            tetrahedra,
            membership,
            gauge,
            covered,
        })
    }

    fn logical_spin(&self, state: &State, tetrahedron: usize, node: usize, lane: usize) -> i32 {
        let slot = usize::from(self.membership[node][0] != tetrahedron);
        debug_assert_eq!(self.membership[node][slot], tetrahedron);
        self.gauge[node][slot] * spin(state.spins[node], lane)
    }
}

enum MoveKind {
    Loop,
    String,
}

pub(super) struct IceWorkspace {
    visited: Vec<usize>,
    touched: Vec<usize>,
    marked: Vec<bool>,
    path: Vec<usize>,
    pub(super) stats: IceLoopStats,
}

impl IceWorkspace {
    pub(super) fn new(topology: &IceTopology) -> Self {
        Self {
            visited: vec![BOUNDARY; topology.tetrahedra.len()],
            touched: Vec::new(),
            marked: vec![false; topology.membership.len()],
            path: Vec::new(),
            stats: IceLoopStats::default(),
        }
    }

    fn propose(
        &mut self,
        topology: &IceTopology,
        state: &State,
        lane: usize,
        rng: &mut SmallRng,
    ) -> Option<MoveKind> {
        for &tetrahedron in &self.touched {
            self.visited[tetrahedron] = BOUNDARY;
        }
        self.touched.clear();
        self.path.clear();
        let start = topology.covered[rng.gen_range(0..topology.covered.len())];
        self.path.push(start);
        let members = topology.membership[start];
        let start_boundary = members[1] == BOUNDARY;
        let mut tetrahedron = if start_boundary {
            members[0]
        } else {
            members[rng.gen_range(0..2)]
        };
        loop {
            let incoming = self.path[self.path.len() - 1];
            let previous = self.visited[tetrahedron];
            if previous != BOUNDARY {
                let offset = previous + 1;
                if self.path.len() - offset < 2
                    || topology.logical_spin(state, tetrahedron, self.path[offset], lane)
                        == topology.logical_spin(state, tetrahedron, incoming, lane)
                {
                    return None;
                }
                self.path.drain(..offset);
                return Some(MoveKind::Loop);
            }
            let nodes = topology.tetrahedra[tetrahedron];
            let sum: i32 = nodes
                .iter()
                .map(|&node| topology.logical_spin(state, tetrahedron, node, lane))
                .sum();
            if sum != 0 {
                return None;
            }
            self.visited[tetrahedron] = self.path.len() - 1;
            self.touched.push(tetrahedron);
            let sign = topology.logical_spin(state, tetrahedron, incoming, lane);
            let mut opposite = [0; 2];
            let mut count = 0;
            for node in nodes {
                if topology.logical_spin(state, tetrahedron, node, lane) != sign {
                    opposite[count] = node;
                    count += 1;
                }
            }
            let outgoing = opposite[rng.gen_range(0..2)];
            self.path.push(outgoing);
            let pair = topology.membership[outgoing];
            let other = if pair[0] == tetrahedron {
                pair[1]
            } else {
                pair[0]
            };
            if other == BOUNDARY {
                return start_boundary.then_some(MoveKind::String);
            }
            tetrahedron = other;
        }
    }

    fn propose_defect(
        &mut self,
        topology: &IceTopology,
        state: &State,
        lane: usize,
        rng: &mut SmallRng,
    ) -> bool {
        for &tetrahedron in &self.touched {
            self.visited[tetrahedron] = BOUNDARY;
        }
        self.touched.clear();
        self.path.clear();
        let start = rng.gen_range(0..topology.tetrahedra.len());
        let charge: i32 = topology.tetrahedra[start]
            .iter()
            .map(|&node| topology.logical_spin(state, start, node, lane))
            .sum();
        if charge == 0 {
            return false;
        }
        let mut majority = [0; 4];
        let mut count = 0;
        for node in topology.tetrahedra[start] {
            if topology.logical_spin(state, start, node, lane) == charge.signum() {
                majority[count] = node;
                count += 1;
            }
        }
        let outgoing = majority[rng.gen_range(0..count)];
        self.path.push(outgoing);
        self.visited[start] = 0;
        self.touched.push(start);
        let mut previous = start;
        loop {
            let incoming = self.path[self.path.len() - 1];
            let pair = topology.membership[incoming];
            let tetrahedron = if pair[0] == previous {
                pair[1]
            } else {
                pair[0]
            };
            if tetrahedron == BOUNDARY {
                return self.path.len() >= 2;
            }
            if self.visited[tetrahedron] != BOUNDARY {
                return false;
            }
            let nodes = topology.tetrahedra[tetrahedron];
            let charge: i32 = nodes
                .iter()
                .map(|&node| topology.logical_spin(state, tetrahedron, node, lane))
                .sum();
            if charge != 0 || rng.gen_ratio(1, 8) {
                return self.path.len() >= 2;
            }
            self.visited[tetrahedron] = self.path.len() - 1;
            self.touched.push(tetrahedron);
            let sign = topology.logical_spin(state, tetrahedron, incoming, lane);
            let mut opposite = [0; 2];
            let mut count = 0;
            for node in nodes {
                if topology.logical_spin(state, tetrahedron, node, lane) != sign {
                    opposite[count] = node;
                    count += 1;
                }
            }
            debug_assert_eq!(count, 2);
            let outgoing = opposite[rng.gen_range(0..2)];
            if self.path.contains(&outgoing) {
                return false;
            }
            self.path.push(outgoing);
            previous = tetrahedron;
        }
    }

    fn half_delta(&mut self, graph: &WeightedGraph, state: &State, lane: usize) -> i32 {
        for &node in &self.path {
            self.marked[node] = true;
        }
        let mut delta = 0;
        for &node in &self.path {
            let sign = spin(state.spins[node], lane);
            delta -= sign * state.fields[node][lane];
            for &(other, j) in &graph.neighbors[node] {
                if node < other && self.marked[other] {
                    delta += 2 * j * sign * spin(state.spins[other], lane);
                }
            }
        }
        for &node in &self.path {
            self.marked[node] = false;
        }
        delta
    }

    fn apply_if_below(
        &mut self,
        graph: &WeightedGraph,
        state: &mut State,
        lane: usize,
        cut: i32,
    ) -> bool {
        if self.half_delta(graph, state, lane) > cut {
            return false;
        }
        for &node in &self.path {
            state.apply(graph, node, 1 << lane);
        }
        true
    }

    pub(super) fn defect_step(
        &mut self,
        topology: &IceTopology,
        graph: &WeightedGraph,
        state: &mut State,
        lane: usize,
        beta: f64,
        rng: &mut SmallRng,
    ) {
        self.stats.attempts += 1;
        self.stats.defect_attempts += 1;
        if !self.propose_defect(topology, state, lane, rng) {
            return;
        }
        self.stats.formed_defects += 1;
        let cut = threshold(rng, beta);
        if self.apply_if_below(graph, state, lane, cut) {
            self.stats.accepted += 1;
            self.stats.accepted_defects += 1;
            self.stats.flipped_spins += self.path.len() as u64;
        }
    }

    pub(super) fn step(
        &mut self,
        topology: &IceTopology,
        graph: &WeightedGraph,
        state: &mut State,
        lane: usize,
        beta: f64,
        rng: &mut SmallRng,
    ) {
        self.stats.attempts += 1;
        let Some(kind) = self.propose(topology, state, lane, rng) else {
            return;
        };
        match kind {
            MoveKind::Loop => self.stats.formed_loops += 1,
            MoveKind::String => self.stats.formed_strings += 1,
        }
        let cut = threshold(rng, beta);
        if self.apply_if_below(graph, state, lane, cut) {
            self.stats.accepted += 1;
            self.stats.flipped_spins += self.path.len() as u64;
        }
    }
}
