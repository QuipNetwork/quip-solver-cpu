//! Exact conditional descent with one vector of candidates per table entry.

use quip_solver_core::SamplerResult;

use super::{projection_tables, OtsukaGraph, Plan, Step};

pub(super) const LANES: usize = 8;
type Lanes = [i32; LANES];

pub(super) struct Batch {
    spins: Vec<Lanes>,
    costs: Vec<Lanes>,
    choices: Vec<Lanes>,
}

impl Batch {
    /// Each physical coefficient occurs at most once in a factor sum. An
    /// i32 can hold every unary, message and conditional objective if it can
    /// hold the sum of absolute coefficients.
    pub(super) fn fits(graph: &OtsukaGraph) -> bool {
        let abs_twice: i64 = (0..graph.n)
            .flat_map(|u| graph.all.row(u))
            .map(|edge| i64::from(edge.j()).abs())
            .sum();
        abs_twice / 2 <= i64::from(i32::MAX)
    }

    pub(super) fn new(reads: &[SamplerResult], n: usize) -> Self {
        let mut spins = vec![[1; LANES]; n];
        for (lane, read) in reads.iter().enumerate() {
            for (row, &spin) in spins.iter_mut().zip(&read.spins) {
                row[lane] = i32::from(spin);
            }
        }
        Self {
            spins,
            costs: Vec::new(),
            choices: Vec::new(),
        }
    }

    pub(super) fn store(&self, reads: &mut [SamplerResult]) {
        for (lane, read) in reads.iter_mut().enumerate() {
            for (spin, row) in read.spins.iter_mut().zip(&self.spins) {
                *spin = row[lane] as i8;
            }
        }
    }

    pub(super) fn descend(
        &mut self,
        plan: &Plan,
        stopped: &impl Fn() -> bool,
    ) -> Option<[i64; LANES]> {
        let projections = projection_tables();
        self.costs.resize(plan.direct.len(), [0; LANES]);
        self.choices.resize(plan.direct.len(), [0; LANES]);
        let (mut old, mut optimum) = ([0; LANES], [0; LANES]);
        for (i, step) in plan.steps.iter().enumerate() {
            if i.is_multiple_of(256) && stopped() {
                return None;
            }
            let mut boundary = [0; LANES];
            for edge in &plan.links[step.boundary.clone()] {
                let spins = &self.spins[edge.node()];
                let j = edge.j();
                for lane in 0..LANES {
                    boundary[lane] += j * spins[lane];
                }
            }
            let prev = self.spins[step.node];
            let direct = &plan.direct[step.table.clone()];
            let masks = step.batch_masks(&self.spins);
            for lane in 0..LANES {
                old[lane] +=
                    prev[lane] * (boundary[lane] + i32::from(direct[masks[lane] as usize]));
            }
            for (k, &coupling) in direct.iter().enumerate() {
                let coupling = i32::from(coupling);
                let mut hi: Lanes = boundary.map(|h| h + coupling);
                let mut lo = hi.map(|h| -h);
                for message in &plan.messages[step.incoming.clone()] {
                    let index = message.index(projections, k);
                    for lane in 0..LANES {
                        lo[lane] += self.costs[index][lane];
                        hi[lane] += self.costs[index + 1][lane];
                    }
                }
                let at = step.table.start + k;
                for lane in 0..LANES {
                    self.costs[at][lane] = lo[lane].min(hi[lane]);
                    self.choices[at][lane] = if hi[lane] == lo[lane] {
                        prev[lane]
                    } else if hi[lane] < lo[lane] {
                        1
                    } else {
                        -1
                    };
                }
            }
            if step.width == 0 {
                for (total, cost) in optimum.iter_mut().zip(self.costs[step.table.start]) {
                    *total += cost;
                }
            }
        }
        let gains: [i64; LANES] =
            std::array::from_fn(|i| i64::from(old[i]) - i64::from(optimum[i]));
        if gains.iter().any(|&gain| gain < 0) {
            return None;
        }
        if gains.iter().any(|&gain| gain > 0) {
            if stopped() {
                return None;
            }
            for step in plan.steps.iter().rev() {
                let masks = step.batch_masks(&self.spins);
                for (lane, &gain) in gains.iter().enumerate() {
                    if gain > 0 {
                        self.spins[step.node][lane] =
                            self.choices[step.table.start + masks[lane] as usize][lane];
                    }
                }
            }
        }
        Some(gains)
    }
}

impl Step {
    fn batch_masks(&self, spins: &[Lanes]) -> Lanes {
        let mut masks = [0; LANES];
        for (i, &v) in self.later[..self.width].iter().enumerate() {
            for (mask, spin) in masks.iter_mut().zip(spins[v]) {
                *mask |= ((spin + 1) >> 1) << i;
            }
        }
        masks
    }
}
