//! Constrained loop search for the square-link families: `quip/flow-static`
//! revision 1 (F-static) and `quip/polariton-static` revision 1 (adapted P).
//!
//! The search reads the [`LinkTask`] of [`super::flow`]: one spin per link,
//! the canonical signed incidence, the vertex labels, and the link costs. Spin
//! `+1` points from the tail of a link to its head. The charge of a vertex is
//! the arrows in minus the arrows out. A labeled vertex is a pin. A free
//! vertex is a boundary vertex of adapted P, and it can take any charge.
//!
//! Reversing every link of a directed cycle keeps every charge. Reversing a
//! directed path moves 2 units of charge from its last vertex to its first.
//! So the moves are bounded directed cycles, and directed paths whose ends
//! repair a pin gap or sit on the free boundary. Before a move is kept, the
//! search checks every vertex it touches: no pin gets further from its label,
//! and the charge that the free boundary gains is the charge that the pins
//! lose (the boundary flux). After every kept move the search runs the direct
//! check of the whole orientation ([`LinkTask::check`]) and compares the
//! charges, the label gaps, and the exact objective with its own running
//! values. A difference is [`LoopError::Rescore`].
//!
//! The score is `(gap_squares, objective)` in that order: the sum over pins
//! of `(Q_v - q_v)^2`, then the cost of the links against their request. The
//! instance energy is `objective + (U / 2) gap_squares` in milli. F-static
//! has `2 U > sum c_e`, and every gap is even, so the order of the score is
//! the order of the energy. Adapted P has no objective and `U = 250`.
//!
//! The policy, `flow-loops-nat/1`, follows `nat_loop` of the research
//! `descent.py`. Each read first repairs pin gaps, then runs the steepest
//! descent over the short cycles (the plaquettes of the lattice and the pairs
//! of parallel links), one best move per step. Then rounds of worm descent,
//! repair, and steepest descent alternate until a round leaves the score
//! unchanged, at most [`ROUNDS`] rounds. A repair reverses the shortest
//! directed path, at most [`MAX_LOOP_LEN`] links, from a pin that is short by
//! 2 or more to a pin with an excess of 2 or more or to the free boundary. It
//! visits the pins in vertex order and the links in incidence order, so it
//! draws no random number. The worm descent follows `loop_descent` of the research
//! `greedy.c`: each sweep visits every link in a fresh random order, reverses
//! it, and follows the arrows out of the vertex left short, picking the
//! cheapest arrow (ties broken at random), until the gap closes or the walk
//! reaches the free boundary. A walk of more than [`MAX_LOOP_LEN`] links is
//! undone. A walk is kept when it lowers the score, or when it keeps the score
//! and the zero-move budget of the call still covers its length. Sweeps stop
//! after a sweep with no strict improvement, or after [`MAX_SWEEPS`].
//!
//! Every move keeps or lowers the gap squares, so a read that reaches a
//! feasible state keeps every label from then on. From that state on, a kept
//! move is a directed cycle or a string between free boundary vertices, and the
//! search is a constraint-preserving loop descent. The counts of short-cycle,
//! worm, and repair moves are kept apart in the [`ReadRecord`].
//!
//! [`Work::Fixed`] runs depend only on the inputs and the seed.
//! [`Work::Budget`] runs read the clock on the calling thread against a
//! declared budget, so their trajectories depend on the host and its load. The
//! budget counts from a start instant that the caller gives, so the setup of a
//! job counts against it. Both keep one result per input state, in input order.

use std::fmt;
use std::time::{Duration, Instant};

use quip_solver_core::ising_generators::square::{Axis, Boundary};

use super::flow::LinkTask;

/// The policy name in logs.
pub const FLOW_LOOPS_POLICY: &str = "flow-loops-nat/1";

/// The default host-time budget of one job, in seconds. Plan 2 gives each B
/// family the postprocess allowance of its paired A cell: 0.25 s for
/// `ice-charge` (the pair of adapted P) and for `ice-fields-h500` and
/// `ice-fields-h1000` (pairs of F-static), and 0.05 s for
/// `two-commodity-routing` (the other pair of F-static). A job does not name
/// its pair, so the operator sets the budget of the routing pair.
pub const FLOW_LOOPS_DEFAULT_BUDGET_SECONDS: f64 = 0.25;

/// The most links in one worm walk, the `len=64` of the selected
/// `ice-tiles` contract.
pub const MAX_LOOP_LEN: usize = 64;

/// The most sweeps of one worm descent, the `max_sweeps` default of the
/// research `LoopC.descend`.
pub const MAX_SWEEPS: u32 = 50;

/// The most rounds of worm and steepest descent per read, the `rounds`
/// default of the research `nat_loop`.
pub const ROUNDS: usize = 10;

/// The zero-move budget of one worm descent per link, the `4 n` of the
/// selected `ice-tiles` contract.
const ZERO_BUDGET_PER_LINK: u64 = 4;

/// The xorshift64 state for a zero seed, as in the research `greedy.c`.
const ZERO_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

/// The score of one orientation. The order is the order of the instance
/// energy: first the label gaps, then the objective.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Score {
    /// The sum over labeled vertices of `(Q_v - q_v)^2`. 0 exactly when every
    /// label holds.
    pub gap_squares: u64,
    /// The cost of the links whose spin differs from the target, 0 without
    /// costs.
    pub objective: u64,
}

impl Score {
    /// Whether every label holds.
    #[must_use]
    pub const fn feasible(self) -> bool {
        self.gap_squares == 0
    }
}

/// The direct score of `link_spins`, one spin per link in link order. `None`
/// when the length differs or a spin is not `+1` or `-1`.
#[must_use]
pub fn score(task: &LinkTask, link_spins: &[i8]) -> Option<Score> {
    let check = task.check(link_spins)?;
    Some(Score {
        gap_squares: gap_squares(task.labels(), &check.charges)?,
        objective: check.objective,
    })
}

fn gap_squares(labels: &[Option<i8>], charges: &[i8]) -> Option<u64> {
    labels
        .iter()
        .zip(charges)
        .try_fold(0_u64, |acc, (label, &q)| match label {
            Some(want) => acc.checked_add(gap(q, *want).unsigned_abs().pow(2)),
            None => Some(acc),
        })
}

fn gap(charge: i8, label: i8) -> i64 {
    i64::from(charge) - i64::from(label)
}

/// How much work a search does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Work {
    /// Run against `budget` of elapsed time on the calling thread, counted
    /// from `started`, with at most [`ROUNDS`] rounds per read. Read `k` of
    /// `n` must stop by `(k + 1) / n` of the budget, so time a read leaves
    /// unused passes to the next. A read that would start after the budget is
    /// returned unchanged. The trajectory depends on the host and its load.
    Budget {
        /// The declared budget of the job.
        budget: Duration,
        /// When the job started. Setup before the search counts against the
        /// budget.
        started: Instant,
    },
    /// Run at most `rounds` rounds per read with no clock. The result depends
    /// only on the inputs and the seed.
    Fixed {
        /// The most rounds of worm and steepest descent per read.
        rounds: usize,
    },
}

/// Why a read stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// A round left the score unchanged.
    Converged,
    /// The read ran every round it was allowed.
    Rounds,
    /// The budget ran out during the read. Every kept move stays.
    Deadline,
    /// The budget ran out before the read started. The result is the input.
    Skipped,
}

/// The kind of a kept move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveKind {
    /// A short cycle of the steepest descent.
    Cycle,
    /// A worm walk: a directed cycle, or a directed path that ends on the free
    /// boundary or at a pin it repairs.
    Worm,
    /// A repair: the shortest directed path from a pin that is short to a pin
    /// with an excess or to the free boundary. It always lowers the gap
    /// squares.
    Repair,
}

/// One kept move, as the observer of [`search`] sees it.
#[derive(Clone, Copy, Debug)]
pub struct Accepted<'a> {
    /// The input position of the read.
    pub read: usize,
    /// The kind of move.
    pub kind: MoveKind,
    /// The reversed links, in walk order.
    pub links: &'a [u32],
    /// The orientation after the move, in link order.
    pub spins: &'a [i8],
    /// The score before the move.
    pub before: Score,
    /// The score after the move, from the direct check.
    pub after: Score,
}

/// What one read did.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadRecord {
    /// The input position of the read. Result `k` always has parent `k`.
    pub parent: usize,
    /// The direct score of the input state.
    pub start: Score,
    /// The direct score of the result.
    pub end: Score,
    /// Why the read stopped.
    pub stop: Stop,
    /// The worm and steepest rounds run, not counting the first steepest
    /// descent.
    pub rounds: usize,
    /// The short-cycle moves kept.
    pub cycle_moves: u64,
    /// The worm moves kept.
    pub worm_moves: u64,
    /// The repair moves kept.
    pub repair_moves: u64,
    /// The kept worm moves that left the score unchanged.
    pub zero_moves: u64,
    /// The worm walks proposed.
    pub proposals: u64,
    /// The measured time of the read, in seconds.
    pub seconds: f64,
}

/// What a search did, for the job log.
#[derive(Clone, Debug, PartialEq)]
pub struct LoopReport {
    /// [`FLOW_LOOPS_POLICY`].
    pub policy: &'static str,
    /// The declared budget in seconds, or `None` for fixed work.
    pub budget_seconds: Option<f64>,
    /// The work mode of this run.
    pub work: Work,
    /// One record per input state, in input order.
    pub reads: Vec<ReadRecord>,
    /// The measured time on the calling thread, in seconds: from the start
    /// instant of the budget, or from the start of the search for fixed work.
    pub elapsed_seconds: f64,
}

/// The result of a search.
#[derive(Clone, Debug, PartialEq)]
pub struct LoopOutcome {
    /// The result of each input state, in input order, one spin per link.
    pub states: Vec<Vec<i8>>,
    /// What the search did.
    pub report: LoopReport,
}

/// Why a search stopped without a result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopError {
    /// No input state.
    Empty,
    /// An input state does not fit the task.
    Start {
        /// The input position.
        index: usize,
        /// What is wrong with it.
        why: &'static str,
    },
    /// The cancel check fired.
    Cancelled,
    /// A kept move failed the direct check or the boundary-flux balance. This
    /// is a defect of the search.
    Rescore(&'static str),
}

impl fmt::Display for LoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("no input state"),
            Self::Start { index, why } => write!(f, "input state {index}: {why}"),
            Self::Cancelled => f.write_str("cancelled"),
            Self::Rescore(why) => write!(f, "a kept move fails the direct check: {why}"),
        }
    }
}

impl std::error::Error for LoopError {}

/// A short directed-cycle candidate: each link with the direction the cycle
/// walks it, `+1` from tail to head and `-1` back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShortCycle {
    /// The links and their walk directions.
    pub links: Vec<(u32, i8)>,
}

/// The fixed structure of a task: link ends, incidence, and the short
/// cycles.
#[derive(Clone, Debug)]
pub struct LoopKernel<'t> {
    task: &'t LinkTask,
    /// `(tail, head)` of each link.
    ends: Vec<(u32, u32)>,
    cycles: Vec<ShortCycle>,
}

impl<'t> LoopKernel<'t> {
    /// The kernel of `task`. The short cycles are the plaquettes of the
    /// lattice and every pair of parallel links.
    #[must_use]
    pub fn new(task: &'t LinkTask) -> Self {
        let lattice = task.lattice();
        let ends: Vec<(u32, u32)> = (0..lattice.link_count())
            .filter_map(|e| lattice.link(e).map(|l| (l.tail, l.head)))
            .collect();
        let cycles = short_cycles(task, &ends);
        Self { task, ends, cycles }
    }

    /// The task.
    #[must_use]
    pub const fn task(&self) -> &LinkTask {
        self.task
    }

    /// The short cycles of the steepest descent.
    #[must_use]
    pub fn cycles(&self) -> &[ShortCycle] {
        &self.cycles
    }

    /// The change in objective from reversing link `e` at spin `spin`.
    fn reversal_cost(&self, e: usize, spin: i8) -> i64 {
        self.task.costs().map_or(0, |c| {
            let cost = i64::from(c.costs[e]);
            if spin == c.targets[e] {
                cost
            } else {
                -cost
            }
        })
    }

    /// The objective change of reversing short cycle `i` at `link_spins`, or
    /// `None` when the cycle is not directed there or `i` is out of range.
    #[must_use]
    pub fn cycle_delta(&self, link_spins: &[i8], i: usize) -> Option<i64> {
        let cycle = self.cycles.get(i)?;
        let &(first, first_walk) = cycle.links.first()?;
        let sense = *link_spins.get(first as usize)? * first_walk;
        let mut delta = 0_i64;
        for &(e, walk) in &cycle.links {
            let spin = *link_spins.get(e as usize)?;
            if spin * walk != sense {
                return None;
            }
            delta += self.reversal_cost(e as usize, spin);
        }
        Some(delta)
    }

    /// The other end of link `e` from vertex `v`.
    fn other(&self, e: usize, v: u32) -> u32 {
        let (tail, head) = self.ends[e];
        if v == tail {
            head
        } else {
            tail
        }
    }
}

/// The plaquettes of the lattice and every pair of parallel links.
fn short_cycles(task: &LinkTask, ends: &[(u32, u32)]) -> Vec<ShortCycle> {
    let lattice = task.lattice();
    let side = u32::from(lattice.side());
    let (width, cells) = match lattice.boundary() {
        Boundary::Open => (side + 1, side),
        Boundary::Periodic => (side, side),
    };
    let mut owned = vec![[None::<u32>; 2]; lattice.vertex_count() as usize];
    for e in 0..lattice.link_count() {
        if let Some(link) = lattice.link(e) {
            if let Some(slot) = owned.get_mut(link.tail as usize) {
                slot[usize::from(link.axis == Axis::Y)] = Some(e);
            }
        }
    }
    let at = |x: u32, y: u32| (y % width) * width + (x % width);
    let own = |v: u32, axis: usize| owned.get(v as usize).and_then(|slot| slot[axis]);
    let mut cycles = Vec::new();
    for y in 0..cells {
        for x in 0..cells {
            // Counterclockwise: bottom and right forward, top and left back.
            let links = [
                own(at(x, y), 0).map(|e| (e, 1)),
                own(at(x + 1, y), 1).map(|e| (e, 1)),
                own(at(x, y + 1), 0).map(|e| (e, -1)),
                own(at(x, y), 1).map(|e| (e, -1)),
            ];
            if let [Some(a), Some(b), Some(c), Some(d)] = links {
                cycles.push(ShortCycle {
                    links: vec![a, b, c, d],
                });
            }
        }
    }
    for (i, &(t1, h1)) in (0_u32..).zip(ends) {
        for (j, &(t2, h2)) in (0_u32..).zip(ends).skip(i as usize + 1) {
            if (t2, h2) == (h1, t1) {
                cycles.push(ShortCycle {
                    links: vec![(i, 1), (j, 1)],
                });
            } else if (t2, h2) == (t1, h1) {
                cycles.push(ShortCycle {
                    links: vec![(i, 1), (j, -1)],
                });
            }
        }
    }
    cycles
}

/// The search hooks: the cancel check and the optional observer.
struct Hooks<'c, 'o> {
    cancelled: &'c mut dyn FnMut() -> bool,
    observer: Option<&'o mut dyn FnMut(&Accepted<'_>)>,
}

/// The running state of one read.
struct Walker<'k, 't> {
    kernel: &'k LoopKernel<'t>,
    read: usize,
    spins: Vec<i8>,
    charges: Vec<i8>,
    score: Score,
    deadline: Option<Instant>,
    rng: u64,
    /// The walk under construction.
    path: Vec<u32>,
    /// Per link, whether it is in the walk.
    in_path: Vec<bool>,
    /// The vertices the walk touched and their charges before it.
    touched: Vec<(u32, i8)>,
    record: ReadRecord,
}

/// The outcome of one proposal.
enum Proposal {
    Kept { strict: bool },
    Undone,
}

impl Walker<'_, '_> {
    fn draw(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    fn out_of_time(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    fn label(&self, v: u32) -> Option<i8> {
        self.kernel.task.labels().get(v as usize).copied().flatten()
    }

    /// Reverse link `e` and update the charges of its ends.
    fn reverse(&mut self, e: usize) {
        let (tail, head) = self.kernel.ends[e];
        let spin = self.spins[e];
        // Spin `+1` adds 1 to the head and takes 1 from the tail.
        self.charges[head as usize] -= 2 * spin;
        self.charges[tail as usize] += 2 * spin;
        self.spins[e] = -spin;
    }

    fn touch(&mut self, v: u32) {
        if !self.touched.iter().any(|&(u, _)| u == v) {
            self.touched.push((v, self.charges[v as usize]));
        }
    }

    /// Check the walk on the touched vertices: no pin gets further from its
    /// label, and the free boundary gains what the pins lose. Returns the
    /// change in gap squares, or `None` when a pin gets worse.
    fn admit(&self) -> Result<Option<i64>, LoopError> {
        let (mut pins, mut free, mut delta) = (0_i64, 0_i64, 0_i64);
        for &(v, before) in &self.touched {
            let after = self.charges[v as usize];
            let change = i64::from(after) - i64::from(before);
            match self.label(v) {
                Some(want) => {
                    let (was, now) = (gap(before, want), gap(after, want));
                    if now.abs() > was.abs() {
                        return Ok(None);
                    }
                    delta += now * now - was * was;
                    pins += change;
                }
                None => free += change,
            }
        }
        if free + pins != 0 {
            return Err(LoopError::Rescore("the boundary flux does not balance"));
        }
        Ok(Some(delta))
    }

    /// Run the direct check on the whole orientation, compare it with the
    /// running state, and tell the observer.
    fn commit(&mut self, kind: MoveKind, hooks: &mut Hooks<'_, '_>) -> Result<(), LoopError> {
        let task = self.kernel.task;
        let check = task
            .check(&self.spins)
            .ok_or(LoopError::Rescore("the orientation does not check"))?;
        if check.charges != self.charges {
            return Err(LoopError::Rescore("a running charge differs"));
        }
        let direct = Score {
            gap_squares: gap_squares(task.labels(), &check.charges)
                .ok_or(LoopError::Rescore("the gap squares overflow"))?,
            objective: check.objective,
        };
        if direct != self.score {
            return Err(LoopError::Rescore("the running score differs"));
        }
        if let Some(observer) = hooks.observer.as_mut() {
            observer(&Accepted {
                read: self.read,
                kind,
                links: &self.path,
                spins: &self.spins,
                before: self.record.end,
                after: direct,
            });
        }
        self.record.end = direct;
        Ok(())
    }

    fn apply_delta(&mut self, gaps: i64, objective: i64) -> Result<(), LoopError> {
        let bad = LoopError::Rescore("a running score leaves its range");
        self.score.gap_squares = self
            .score
            .gap_squares
            .checked_add_signed(gaps)
            .ok_or(bad.clone())?;
        self.score.objective = self
            .score
            .objective
            .checked_add_signed(objective)
            .ok_or(bad)?;
        Ok(())
    }

    /// Steepest descent over the short cycles: reverse the directed cycle of
    /// the lowest objective change, first index on ties, while it is
    /// negative.
    fn steepest(&mut self, hooks: &mut Hooks<'_, '_>) -> Result<bool, LoopError> {
        loop {
            if (hooks.cancelled)() {
                return Err(LoopError::Cancelled);
            }
            if self.out_of_time() {
                return Ok(false);
            }
            let best = (0..self.kernel.cycles.len())
                .filter_map(|i| self.kernel.cycle_delta(&self.spins, i).map(|d| (d, i)))
                .min();
            let Some((delta, i)) = best.filter(|&(d, _)| d < 0) else {
                return Ok(true);
            };
            let kernel = self.kernel;
            self.path.clear();
            self.touched.clear();
            for &(e, _) in &kernel.cycles[i].links {
                let (tail, head) = kernel.ends[e as usize];
                self.touch(tail);
                self.touch(head);
                self.reverse(e as usize);
                self.path.push(e);
            }
            if self.admit()? != Some(0) {
                return Err(LoopError::Rescore("a short cycle moves a charge"));
            }
            self.apply_delta(0, delta)?;
            self.record.cycle_moves += 1;
            self.commit(MoveKind::Cycle, hooks)?;
        }
    }

    /// One worm walk from link `e0`.
    fn propose(
        &mut self,
        e0: usize,
        zero_budget: &mut u64,
        hooks: &mut Hooks<'_, '_>,
    ) -> Result<Proposal, LoopError> {
        self.record.proposals += 1;
        self.path.clear();
        self.touched.clear();
        let (tail, head) = self.kernel.ends[e0];
        // The arrow of `e0` runs from `from` to `to`. Reversing it leaves `to`
        // 2 short, so the walk continues along the arrows out of `to`.
        let (from, mut at) = if self.spins[e0] == 1 {
            (tail, head)
        } else {
            (head, tail)
        };
        self.touch(from);
        self.touch(at);
        let mut cum = self.kernel.reversal_cost(e0, self.spins[e0]);
        self.reverse(e0);
        self.path.push(e0 as u32);
        self.in_path[e0] = true;
        let mut closed = true;
        // The walk stops on the free boundary, or at a pin whose gap closed.
        while let Some(want) = self.label(at) {
            if gap(self.charges[at as usize], want) >= 0 {
                break;
            }
            if self.path.len() >= MAX_LOOP_LEN {
                closed = false;
                break;
            }
            let mut pick = None;
            let mut best = 0_i64;
            let mut ties = 0_u64;
            let kernel = self.kernel;
            for &(e, sign) in kernel.task.lattice().incident(at) {
                let e = e as usize;
                // An arrow leaves `at` when its contribution to the charge of
                // `at` is -1.
                if self.in_path[e] || sign * self.spins[e] != -1 {
                    continue;
                }
                let d = kernel.reversal_cost(e, self.spins[e]);
                if pick.is_none() || d < best {
                    pick = Some(e);
                    best = d;
                    ties = 1;
                } else if d == best {
                    ties += 1;
                    if self.draw().is_multiple_of(ties) {
                        pick = Some(e);
                    }
                }
            }
            let Some(e) = pick else {
                closed = false;
                break;
            };
            let next = self.kernel.other(e, at);
            self.touch(next);
            cum += best;
            self.reverse(e);
            self.path.push(e as u32);
            self.in_path[e] = true;
            at = next;
        }
        for &e in &self.path {
            self.in_path[e as usize] = false;
        }
        let gaps = if closed { self.admit()? } else { None };
        let len = self.path.len() as u64;
        let keep = gaps.and_then(|g| {
            if g < 0 || (g == 0 && cum < 0) {
                Some(true)
            } else if g == 0 && cum == 0 && *zero_budget >= len {
                Some(false)
            } else {
                None
            }
        });
        let Some(strict) = keep else {
            for k in (0..self.path.len()).rev() {
                let e = self.path[k] as usize;
                self.reverse(e);
            }
            return Ok(Proposal::Undone);
        };
        if !strict {
            *zero_budget -= len;
            self.record.zero_moves += 1;
        }
        self.apply_delta(gaps.unwrap_or(0), cum)?;
        self.record.worm_moves += 1;
        self.commit(MoveKind::Worm, hooks)?;
        Ok(Proposal::Kept { strict })
    }

    /// Worm descent: sweeps over every link in a fresh random order until a
    /// sweep has no strict improvement. Returns `false` when the deadline
    /// stopped it.
    fn worm(&mut self, seed: u64, hooks: &mut Hooks<'_, '_>) -> Result<bool, LoopError> {
        self.rng = if seed == 0 { ZERO_SEED } else { seed };
        let links = self.spins.len();
        let mut zero_budget = ZERO_BUDGET_PER_LINK.saturating_mul(links as u64);
        let mut order: Vec<usize> = (0..links).collect();
        for _ in 0..MAX_SWEEPS {
            let mut improved = false;
            for i in (1..links).rev() {
                let k = (self.draw() % (i as u64 + 1)) as usize;
                order.swap(i, k);
            }
            for &e0 in &order {
                if (hooks.cancelled)() {
                    return Err(LoopError::Cancelled);
                }
                if self.out_of_time() {
                    return Ok(false);
                }
                if let Proposal::Kept { strict: true } =
                    self.propose(e0, &mut zero_budget, hooks)?
                {
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        Ok(true)
    }

    /// The shortest directed path, at most [`MAX_LOOP_LEN`] links, from pin
    /// `v` along the arrows to a pin with an excess of 2 or more or to a free
    /// vertex, as links in walk order. `reach` holds the link that first
    /// reached each vertex and its depth.
    fn shortest_path(&self, v: u32, reach: &mut [Option<(u32, usize)>]) -> Option<Vec<u32>> {
        let kernel = self.kernel;
        reach.fill(None);
        let mut queue = std::collections::VecDeque::from([(v, 0_usize)]);
        let mut found = None;
        'search: while let Some((u, depth)) = queue.pop_front() {
            if depth >= MAX_LOOP_LEN {
                continue;
            }
            for &(e, sign) in kernel.task.lattice().incident(u) {
                // An arrow leaves `u` when its contribution to the charge of
                // `u` is -1.
                if sign * self.spins[e as usize] != -1 {
                    continue;
                }
                let w = kernel.other(e as usize, u);
                if w == v || reach[w as usize].is_some() {
                    continue;
                }
                reach[w as usize] = Some((e, depth + 1));
                let target = self
                    .label(w)
                    .is_none_or(|want| gap(self.charges[w as usize], want) >= 2);
                if target {
                    found = Some(w);
                    break 'search;
                }
                queue.push_back((w, depth + 1));
            }
        }
        let mut at = found?;
        let mut links = Vec::new();
        while at != v {
            let (e, _) = reach[at as usize]?;
            links.push(e);
            at = kernel.other(e as usize, at);
        }
        links.reverse();
        Some(links)
    }

    /// Repair pin gaps until no pin that is short by 2 or more has a path.
    /// Returns `false` when the deadline stopped it.
    fn repair(&mut self, hooks: &mut Hooks<'_, '_>) -> Result<bool, LoopError> {
        let kernel = self.kernel;
        let vertices = kernel.task.lattice().vertex_count();
        let mut reach = vec![None; vertices as usize];
        loop {
            let mut progressed = false;
            for v in 0..vertices {
                let short = self
                    .label(v)
                    .is_some_and(|want| gap(self.charges[v as usize], want) <= -2);
                if !short {
                    continue;
                }
                if (hooks.cancelled)() {
                    return Err(LoopError::Cancelled);
                }
                if self.out_of_time() {
                    return Ok(false);
                }
                let Some(links) = self.shortest_path(v, &mut reach) else {
                    continue;
                };
                self.path.clear();
                self.touched.clear();
                let mut cum = 0_i64;
                for &e in &links {
                    let (tail, head) = kernel.ends[e as usize];
                    self.touch(tail);
                    self.touch(head);
                    cum += kernel.reversal_cost(e as usize, self.spins[e as usize]);
                    self.reverse(e as usize);
                    self.path.push(e);
                }
                let gaps = self.admit()?.filter(|&g| g < 0).ok_or(LoopError::Rescore(
                    "a repair does not lower the gap squares",
                ))?;
                self.apply_delta(gaps, cum)?;
                self.record.repair_moves += 1;
                self.commit(MoveKind::Repair, hooks)?;
                progressed = true;
            }
            if !progressed {
                return Ok(true);
            }
        }
    }

    /// The `nat_loop` structure: repair and steepest descent, then rounds of
    /// worm, repair, and steepest descent until a round leaves the score
    /// unchanged.
    fn run(
        &mut self,
        rounds: usize,
        seed: u64,
        hooks: &mut Hooks<'_, '_>,
    ) -> Result<(), LoopError> {
        if !self.repair(hooks)? || !self.steepest(hooks)? {
            self.record.stop = Stop::Deadline;
            return Ok(());
        }
        let mut last = self.score;
        for round in 1..=rounds {
            self.record.rounds = round;
            let round_seed = seed.wrapping_add(round as u64);
            if !self.worm(round_seed, hooks)? || !self.repair(hooks)? || !self.steepest(hooks)? {
                self.record.stop = Stop::Deadline;
                return Ok(());
            }
            if self.score == last {
                self.record.stop = Stop::Converged;
                return Ok(());
            }
            last = self.score;
        }
        self.record.stop = Stop::Rounds;
        Ok(())
    }
}

/// The seed of read `k`: the job seed plus `k` in the high half, so the
/// round seeds `seed + round` of different reads never meet.
fn read_seed(seed: u64, k: usize) -> u64 {
    seed.wrapping_add((k as u64) << 32)
}

/// Run the `flow-loops-nat/1` policy on each state of `starts`, one spin per
/// link in link order. Result `k` is the search from input `k`. `cancelled`
/// runs before each short-cycle step and each worm proposal. `observer`, when
/// present, sees every kept move after its direct check.
///
/// # Errors
///
/// [`LoopError::Empty`] for no input, [`LoopError::Start`] for an input
/// that does not fit, [`LoopError::Cancelled`] when the cancel check fires,
/// and [`LoopError::Rescore`] when a kept move fails the direct check.
pub fn search(
    kernel: &LoopKernel<'_>,
    starts: &[Vec<i8>],
    seed: u64,
    work: Work,
    cancelled: &mut dyn FnMut() -> bool,
    observer: Option<&mut dyn FnMut(&Accepted<'_>)>,
) -> Result<LoopOutcome, LoopError> {
    if starts.is_empty() {
        return Err(LoopError::Empty);
    }
    let task = kernel.task;
    let links = task.positions().len();
    let mut scored = Vec::with_capacity(starts.len());
    for (index, start) in starts.iter().enumerate() {
        if start.len() != links {
            return Err(LoopError::Start {
                index,
                why: "not one spin per link",
            });
        }
        let check = task.check(start).ok_or(LoopError::Start {
            index,
            why: "a spin is not +1 or -1",
        })?;
        let start_score = Score {
            gap_squares: gap_squares(task.labels(), &check.charges).ok_or(LoopError::Start {
                index,
                why: "the gap squares overflow",
            })?,
            objective: check.objective,
        };
        scored.push((check.charges, start_score));
    }
    let (rounds, budget, started) = match work {
        Work::Budget { budget, started } => (ROUNDS, Some(budget), started),
        Work::Fixed { rounds } => (rounds, None, Instant::now()),
    };
    let mut hooks = Hooks {
        cancelled,
        observer,
    };
    let reads = starts.len();
    let mut states = Vec::with_capacity(reads);
    let mut records = Vec::with_capacity(reads);
    for (k, (start, (charges, start_score))) in starts.iter().zip(scored).enumerate() {
        let deadline = budget.map(|b| {
            let share = b.as_secs_f64() * (k + 1) as f64 / reads as f64;
            started + Duration::from_secs_f64(share)
        });
        let read_started = Instant::now();
        let mut walker = Walker {
            kernel,
            read: k,
            spins: start.clone(),
            charges,
            score: start_score,
            deadline,
            rng: ZERO_SEED,
            path: Vec::with_capacity(MAX_LOOP_LEN),
            in_path: vec![false; links],
            touched: Vec::with_capacity(MAX_LOOP_LEN + 1),
            record: ReadRecord {
                parent: k,
                start: start_score,
                end: start_score,
                stop: Stop::Skipped,
                rounds: 0,
                cycle_moves: 0,
                worm_moves: 0,
                repair_moves: 0,
                zero_moves: 0,
                proposals: 0,
                seconds: 0.0,
            },
        };
        if !walker.out_of_time() {
            walker.run(rounds, read_seed(seed, k), &mut hooks)?;
        }
        walker.record.seconds = read_started.elapsed().as_secs_f64();
        records.push(walker.record);
        states.push(walker.spins);
    }
    Ok(LoopOutcome {
        states,
        report: LoopReport {
            policy: FLOW_LOOPS_POLICY,
            budget_seconds: budget.map(|b| b.as_secs_f64()),
            work,
            reads: records,
            elapsed_seconds: started.elapsed().as_secs_f64(),
        },
    })
}
