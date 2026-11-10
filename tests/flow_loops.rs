//! The constrained loop search of `quip_solver_cpu::families::flow_loops` on
//! F-static (`quip/flow-static` revision 1) and adapted P
//! (`quip/polariton-static` revision 1).
//!
//! Every side-2 state is scored against the instance energy, every short
//! cycle delta is checked against the direct score, and the search runs from
//! every side-2 state with an observer that checks the charges, the pins, the
//! boundary flux, and the exact rescoring after every kept move. A periodic
//! side-2 lattice has parallel links. Malformed boundaries and bad starts are
//! rejected before the search. A fixed-work run reproduces the recorded golden
//! output, and a time-bounded run records its host time and every final
//! state.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use quip_protocol::scoring::energy_milli;
use quip_solver_conformance::families::{export_instances, reference_revision};
use quip_solver_core::ising_generators::descriptor::Contract;
use quip_solver_core::ising_generators::family::BaseTopology;
use quip_solver_core::ising_generators::flow::{FLOW_RECIPES, FLOW_V1};
use quip_solver_core::ising_generators::instance::{Instance, InstanceParts};
use quip_solver_core::ising_generators::polariton::{POLARITON_RECIPES, POLARITON_V1};
use quip_solver_core::ising_generators::square::{Boundary, SquareLattice};
use quip_solver_cpu::families::flow::{
    orient, LinkCosts, LinkTask, Outcome, TaskError, FREE_LABEL, KIND_CHARGES, KIND_FLOW,
    KIND_INCIDENCE,
};
use quip_solver_cpu::families::flow_loops::{
    score, search, Accepted, LoopError, LoopKernel, LoopOutcome, MoveKind, Score, Stop, Work,
    FLOW_LOOPS_DEFAULT_BUDGET_SECONDS, FLOW_LOOPS_POLICY, ROUNDS,
};
use serde_json::{json, Value};

/// The live z(12, 4) topology of the registered recipes.
const LIVE_TOPOLOGY: &str = "../topologies/zephyr-z12t4-live.bin.gz";
/// The side-2 reference topology.
const SIDE_2: &str = "../topologies/routing-small.bin.gz";
/// The side-3 reference topology with a permuted node order.
const SIDE_3: &str = "../topologies/polariton-l3-permuted.bin.gz";
/// The recorded golden output. The SC-2 golden vectors of quip-solver-core
/// pin generator outputs from an independent oracle, and have no place for the
/// seeded output of a solver, so this fixture has its own layout. Each case
/// names its instance by the SC-2 identity: family, revision, parameter bytes,
/// nonce, and instance digest.
const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/flow_loops_golden.json"
);
/// Set to write the golden fixture instead of reading it.
const BLESS: &str = "QUIP_FLOW_LOOPS_BLESS";

fn topology(name: &str) -> BaseTopology {
    let stored = quip_solver_conformance::topology(name).expect("a stored topology");
    BaseTopology::new(stored.nodes, &stored.edges).expect("a base topology")
}

/// The first instances of the first reference parameter set of `contract`
/// that builds on `name`.
fn reference_instances(
    contract: &Contract,
    name: &str,
    count: u8,
) -> (&'static [u8], Vec<Instance>) {
    let reference =
        reference_revision(contract.family_id, contract.revision).expect("a reference revision");
    let topology = topology(name);
    let nonces: Vec<[u8; 32]> = (0..count).map(|i| [i; 32]).collect();
    for &params in reference.params {
        let Ok(built) = export_instances(contract, params, &topology, &nonces) else {
            continue;
        };
        let found: Vec<Instance> = built.into_iter().filter_map(Result::ok).collect();
        if !found.is_empty() {
            return (params, found);
        }
    }
    unreachable!("no reference instance of {} on {name}", contract.family_id)
}

fn recipe_instance(contract: &Contract, params: &[u8]) -> Instance {
    let nonces = [[3_u8; 32]];
    export_instances(contract, params, &topology(LIVE_TOPOLOGY), &nonces)
        .expect("the params admit")
        .into_iter()
        .find_map(Result::ok)
        .expect("a recipe instance")
}

/// The payload of section `kind`.
fn section(metadata: &[u8], kind: u16) -> Option<(usize, usize)> {
    let count = u16::from_le_bytes([metadata[0], metadata[1]]);
    let mut at = 2;
    for _ in 0..count {
        let k = u16::from_le_bytes([metadata[at], metadata[at + 1]]);
        let len = u32::from_le_bytes(metadata[at + 2..at + 6].try_into().ok()?) as usize;
        if k == kind {
            return Some((at + 6, len));
        }
        at += 6 + len;
    }
    None
}

/// The charge-penalty weight `U` in milli: in the `flow` section for
/// F-static, 250 for adapted P.
fn weight(instance: &Instance) -> u64 {
    section(instance.metadata(), KIND_FLOW).map_or(250, |(at, _)| {
        u64::from_le_bytes(instance.metadata()[at..at + 8].try_into().unwrap())
    })
}

/// The instance energy of a score: `objective + (U / 2) gap_squares`.
fn score_energy(instance: &Instance, s: Score) -> i64 {
    i64::try_from(s.objective + weight(instance) / 2 * s.gap_squares).unwrap()
}

/// The job graph of an instance, in whole units, as the session builds it.
struct Graph {
    h: Vec<f64>,
    j: Vec<f64>,
    edges: Vec<(usize, usize)>,
    offset: i64,
}

impl Graph {
    fn of(instance: &Instance) -> Self {
        let milli = |v: &i32| f64::from(*v) / 1000.0;
        Self {
            h: instance.h().iter().map(milli).collect(),
            j: instance.j().iter().map(milli).collect(),
            edges: instance
                .edges()
                .iter()
                .map(|&(u, v)| (u as usize, v as usize))
                .collect(),
            offset: instance.offset_milli(),
        }
    }

    /// The exact rescoring of instance spins, offset included.
    fn energy(&self, spins: &[i8]) -> i64 {
        energy_milli(spins, &self.h, &self.j, &self.edges) + self.offset
    }
}

/// The charges of `link_spins` from the signed incidence, computed here
/// without the library.
fn direct_charges(lattice: &SquareLattice, link_spins: &[i8]) -> Vec<i8> {
    let mut q = vec![0_i8; lattice.vertex_count() as usize];
    for (e, &s) in (0..lattice.link_count()).zip(link_spins) {
        let link = lattice.link(e).unwrap();
        q[link.tail as usize] -= s;
        q[link.head as usize] += s;
    }
    q
}

fn state_of(bits: u32, links: u32) -> Vec<i8> {
    (0..links)
        .map(|e| if bits >> e & 1 == 1 { 1 } else { -1 })
        .collect()
}

fn never() -> impl FnMut() -> bool {
    || false
}

/// A deterministic pseudo-random state of `links` spins.
fn lcg_state(seed: u64, links: usize) -> Vec<i8> {
    let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..links)
        .map(|_| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if x >> 63 == 1 {
                1
            } else {
                -1
            }
        })
        .collect()
}

#[test]
fn every_side_2_state_scores_the_instance_energy() {
    for contract in [FLOW_V1, POLARITON_V1] {
        let (params, instances) = reference_instances(&contract, SIDE_2, 4);
        for instance in &instances {
            let (_, task) = LinkTask::from_instance(&contract, params, instance).unwrap();
            let links = task.lattice().link_count();
            assert_eq!(links, 12, "{}", contract.family_id);
            let graph = Graph::of(instance);
            let mut feasible = 0;
            for bits in 0..1_u32 << links {
                let spins = state_of(bits, links);
                let check = task.check(&spins).unwrap();
                assert_eq!(check.charges, direct_charges(task.lattice(), &spins));
                assert_eq!(check.charges.iter().map(|&q| i32::from(q)).sum::<i32>(), 0);
                let got = score(&task, &spins).unwrap();
                assert_eq!(got.feasible(), check.feasible());
                feasible += usize::from(got.feasible());
                let nodes = task.node_spins(&spins).unwrap();
                assert_eq!(
                    graph.energy(&nodes),
                    score_energy(instance, got),
                    "{} state {bits:#x}",
                    contract.family_id
                );
            }
            assert!(feasible > 0, "{}", contract.family_id);
        }
    }
}

/// Every short cycle at every state of `task`: a directed cycle keeps every
/// charge and changes the score by its delta, any other is refused.
fn check_cycle_deltas(task: &LinkTask) -> usize {
    let kernel = LoopKernel::new(task);
    let links = task.lattice().link_count();
    let mut directed = 0;
    for bits in 0..1_u32 << links {
        let spins = state_of(bits, links);
        let before = score(task, &spins).unwrap();
        let charges = task.check(&spins).unwrap().charges;
        for (i, cycle) in kernel.cycles().iter().enumerate() {
            let senses: Vec<i8> = cycle
                .links
                .iter()
                .map(|&(e, walk)| spins[e as usize] * walk)
                .collect();
            let is_directed = senses.iter().all(|&s| s == senses[0]);
            let Some(delta) = kernel.cycle_delta(&spins, i) else {
                assert!(!is_directed);
                continue;
            };
            assert!(is_directed);
            directed += 1;
            let mut flipped = spins.clone();
            for &(e, _) in &cycle.links {
                flipped[e as usize] = -flipped[e as usize];
            }
            assert_eq!(task.check(&flipped).unwrap().charges, charges);
            let after = score(task, &flipped).unwrap();
            assert_eq!(after.gap_squares, before.gap_squares);
            assert_eq!(
                i64::try_from(after.objective).unwrap() - i64::try_from(before.objective).unwrap(),
                delta
            );
        }
    }
    directed
}

#[test]
fn every_short_cycle_delta_matches_the_direct_score() {
    let (params, instances) = reference_instances(&FLOW_V1, SIDE_2, 2);
    for instance in &instances {
        let (_, task) = LinkTask::from_instance(&FLOW_V1, params, instance).unwrap();
        assert_eq!(LoopKernel::new(&task).cycles().len(), 4, "four plaquettes");
        assert!(check_cycle_deltas(&task) > 0);
    }
    assert!(check_cycle_deltas(&periodic_task(5)) > 0);
}

/// What the observer saw over a search.
#[derive(Default)]
struct Seen {
    cycles: u64,
    worms: u64,
    repairs: u64,
    parallel: u64,
}

/// Search from `starts` with an observer that checks every kept move: its
/// links are distinct, no pin gets further from its label, the total charge
/// stays 0, the score never rises, and the direct score and the exact
/// rescoring of the instance agree with the reported score.
fn checked_search(
    task: &LinkTask,
    instance: Option<&Instance>,
    starts: &[Vec<i8>],
    seed: u64,
    rounds: usize,
    seen: &mut Seen,
) -> LoopOutcome {
    let kernel = LoopKernel::new(task);
    let lattice = task.lattice();
    let graph = instance.map(Graph::of);
    let mut last: BTreeMap<usize, Vec<i8>> = BTreeMap::new();
    let mut observer = |m: &Accepted<'_>| {
        let mut sorted = m.links.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), m.links.len(), "a link repeats");
        let prev = last.entry(m.read).or_insert_with(|| starts[m.read].clone());
        let (q0, q1) = (
            direct_charges(lattice, prev),
            direct_charges(lattice, m.spins),
        );
        for ((&a, &b), label) in q0.iter().zip(&q1).zip(task.labels()) {
            if let Some(want) = *label {
                assert!((b - want).abs() <= (a - want).abs(), "a pin gets worse");
                if m.before.gap_squares == 0 {
                    assert_eq!(a, b, "a move after feasibility changes a pin");
                }
            }
        }
        assert_eq!(q1.iter().map(|&q| i32::from(q)).sum::<i32>(), 0);
        let changed: Vec<u32> = (0_u32..)
            .zip(prev.iter().zip(m.spins))
            .filter_map(|(e, (a, b))| (a != b).then_some(e))
            .collect();
        assert_eq!(changed, sorted, "the move reverses exactly its links");
        assert!(m.after <= m.before, "the score rises");
        assert_eq!(score(task, m.spins), Some(m.after));
        if let (Some(graph), Some(instance)) = (&graph, instance) {
            let nodes = task.node_spins(m.spins).unwrap();
            assert_eq!(graph.energy(&nodes), score_energy(instance, m.after));
        }
        match m.kind {
            MoveKind::Cycle => {
                assert_eq!(q0, q1, "a short cycle moves a charge");
                seen.cycles += 1;
            }
            MoveKind::Worm => seen.worms += 1,
            MoveKind::Repair => {
                assert!(m.after.gap_squares < m.before.gap_squares, "a repair");
                seen.repairs += 1;
            }
        }
        if m.links.len() == 2 {
            let (a, b) = (
                lattice.link(m.links[0]).unwrap(),
                lattice.link(m.links[1]).unwrap(),
            );
            if [a.tail, a.head] == [b.tail, b.head] || [a.tail, a.head] == [b.head, b.tail] {
                seen.parallel += 1;
            }
        }
        *prev = m.spins.to_vec();
    };
    let out = search(
        &kernel,
        starts,
        seed,
        Work::Fixed { rounds },
        &mut never(),
        Some(&mut observer),
    )
    .expect("a search");
    assert_eq!(out.states.len(), starts.len());
    for (k, (state, record)) in out.states.iter().zip(&out.report.reads).enumerate() {
        assert_eq!(record.parent, k);
        assert_eq!(Some(record.start), score(task, &starts[k]));
        assert_eq!(Some(record.end), score(task, state));
        assert!(record.end <= record.start);
        if record.start.feasible() {
            assert!(record.end.feasible(), "a feasible start loses a pin");
        }
        assert_ne!(record.stop, Stop::Skipped);
        assert_ne!(record.stop, Stop::Deadline);
    }
    out
}

#[test]
fn every_side_2_state_keeps_its_constraints_under_every_kept_move() {
    for contract in [FLOW_V1, POLARITON_V1] {
        let (params, instances) = reference_instances(&contract, SIDE_2, 2);
        for instance in &instances {
            let (_, task) = LinkTask::from_instance(&contract, params, instance).unwrap();
            let links = task.lattice().link_count();
            let starts: Vec<Vec<i8>> = (0..1_u32 << links).map(|b| state_of(b, links)).collect();
            let optimum = starts.iter().filter_map(|s| score(&task, s)).min().unwrap();
            let mut seen = Seen::default();
            let out = checked_search(&task, Some(instance), &starts, 17, 3, &mut seen);
            assert!(seen.worms > 0, "{}", contract.family_id);
            assert!(seen.repairs > 0, "{}", contract.family_id);
            let ends: Vec<Score> = out.report.reads.iter().map(|r| r.end).collect();
            assert!(ends.contains(&optimum), "{}", contract.family_id);
            let repaired = out
                .report
                .reads
                .iter()
                .filter(|r| r.end.gap_squares < r.start.gap_squares)
                .count();
            assert!(repaired > 0, "no infeasible start is repaired");
            if contract.family_id == FLOW_V1.family_id {
                assert!(seen.cycles > 0);
                // The baseline optimum is a fixed point of the search.
                let Ok(Outcome::Optimal(best)) = orient(&task, &mut never()) else {
                    unreachable!("a generated instance has an optimum")
                };
                assert_eq!(score(&task, &best.link_spins), Some(optimum));
                let again = checked_search(
                    &task,
                    Some(instance),
                    std::slice::from_ref(&best.link_spins),
                    3,
                    ROUNDS,
                    &mut Seen::default(),
                );
                assert_eq!(again.report.reads[0].end, optimum);
            }
        }
    }
}

/// A periodic side-2 lattice: 4 vertices of degree 4, 8 links, and every
/// pair of neighbours joined by two parallel links. Every vertex has label
/// 0 and the costs come from `seed`.
fn periodic_task(seed: u64) -> LinkTask {
    let lattice = SquareLattice::new(2, Boundary::Periodic).unwrap();
    let links = lattice.link_count();
    let targets = lcg_state(seed, links as usize);
    let costs = (0..links)
        .map(|e| 2 * ((u32::try_from(seed % 97).unwrap() + 13 * e) % 7))
        .collect();
    LinkTask::new(
        lattice,
        (0..links).collect(),
        vec![Some(0); 4],
        Some(LinkCosts { targets, costs }),
    )
    .unwrap()
}

#[test]
fn parallel_links_survive_side_2() {
    let mut seen = Seen::default();
    for seed in 0..8 {
        let task = periodic_task(seed);
        let lattice = task.lattice();
        assert_eq!((lattice.vertex_count(), lattice.link_count()), (4, 8));
        let kernel = LoopKernel::new(&task);
        let pairs: Vec<_> = kernel
            .cycles()
            .iter()
            .filter(|c| c.links.len() == 2)
            .collect();
        assert_eq!(pairs.len(), 4, "four pairs of parallel links");
        for pair in &pairs {
            let (a, b) = (
                lattice.link(pair.links[0].0).unwrap(),
                lattice.link(pair.links[1].0).unwrap(),
            );
            assert_eq!(
                [a.tail.min(a.head), a.tail.max(a.head)],
                [b.tail.min(b.head), b.tail.max(b.head)]
            );
        }
        let starts: Vec<Vec<i8>> = (0..1_u32 << 8).map(|b| state_of(b, 8)).collect();
        let optimum = starts.iter().filter_map(|s| score(&task, s)).min().unwrap();
        let out = checked_search(&task, None, &starts, seed, 3, &mut seen);
        assert!(out.report.reads.iter().any(|r| r.end == optimum));
    }
    assert!(seen.parallel > 0, "no parallel-link cycle was kept");
}

/// `instance` with the byte at `at` of its metadata replaced.
fn tampered(contract: &Contract, instance: &Instance, at: usize, byte: u8) -> Instance {
    let mut metadata = instance.metadata().to_vec();
    metadata[at] = byte;
    let parts = InstanceParts {
        descriptor_digest: *instance.descriptor_digest(),
        topology_order_digest: *instance.topology_order_digest(),
        nonce: instance.nonce().copied(),
        node_ids: instance.node_ids().to_vec(),
        edges: instance.edges().to_vec(),
        h: instance.h().to_vec(),
        j: instance.j().to_vec(),
        offset_milli: instance.offset_milli(),
        metadata,
    };
    Instance::new(parts, &contract.limits, contract.metadata_kinds, None).expect("a valid layout")
}

fn rejected(result: Result<(quip_solver_cpu::families::flow::Mode, LinkTask), TaskError>) -> bool {
    matches!(
        result,
        Err(TaskError::Metadata | TaskError::Section { .. } | TaskError::Shape(_))
    )
}

#[test]
fn malformed_boundaries_are_rejected_before_the_search() {
    // The side-2 vertices in vertex order: 0 to 8, the centre is 4.
    let (corner, edge, centre) = (0_usize, 1_usize, 4_usize);
    for contract in [FLOW_V1, POLARITON_V1] {
        let (params, instances) = reference_instances(&contract, SIDE_2, 1);
        let instance = &instances[0];
        assert!(LinkTask::from_instance(&contract, params, instance).is_ok());
        let (labels_at, _) = section(instance.metadata(), KIND_CHARGES).unwrap();
        let label = |v: usize| instance.metadata()[labels_at + 2 + v] as i8;
        let mut cases = vec![
            // The side in the charges section differs from the parameters.
            (labels_at, 3_u8),
        ];
        if contract.family_id == FLOW_V1.family_id {
            // Every vertex is a pin: a corner label of +2, an edge label
            // reversed, and an edge label removed.
            cases.push((labels_at + 2 + corner, 2));
            cases.push((labels_at + 2 + edge, (-label(edge)) as u8));
            cases.push((labels_at + 2 + edge, FREE_LABEL as u8));
        } else {
            // The boundary is free and the interior is pinned: a pinned
            // corner, a pinned edge vertex, and a free centre.
            assert_eq!(label(corner), FREE_LABEL);
            cases.push((labels_at + 2 + corner, 0));
            cases.push((labels_at + 2 + edge, 0));
            cases.push((labels_at + 2 + centre, FREE_LABEL as u8));
        }
        // Two links on one instance position.
        let (incidence_at, _) = section(instance.metadata(), KIND_INCIDENCE).unwrap();
        let first = instance.metadata()[incidence_at + 4];
        cases.push((incidence_at + 8, first));
        for (at, byte) in cases {
            if instance.metadata()[at] == byte {
                continue;
            }
            let bad = tampered(&contract, instance, at, byte);
            assert!(
                rejected(LinkTask::from_instance(&contract, params, &bad)),
                "{} byte {at} = {byte}",
                contract.family_id
            );
        }
        // Bad starts are rejected before any move.
        let (_, task) = LinkTask::from_instance(&contract, params, instance).unwrap();
        let kernel = LoopKernel::new(&task);
        let run = |starts: &[Vec<i8>]| {
            search(
                &kernel,
                starts,
                0,
                Work::Fixed { rounds: 1 },
                &mut never(),
                None,
            )
        };
        assert_eq!(run(&[]).err(), Some(LoopError::Empty));
        let good = vec![1_i8; 12];
        let mut zero = good.clone();
        zero[5] = 0;
        assert_eq!(
            run(&[good.clone(), vec![1; 11]]).err(),
            Some(LoopError::Start {
                index: 1,
                why: "not one spin per link"
            })
        );
        assert_eq!(
            run(&[zero]).err(),
            Some(LoopError::Start {
                index: 0,
                why: "a spin is not +1 or -1"
            })
        );
    }
    // A task whose positions repeat, or whose labels do not fit, has no
    // kernel at all.
    let lattice = SquareLattice::new(2, Boundary::Open).unwrap();
    assert!(LinkTask::new(lattice.clone(), vec![0; 12], vec![None; 9], None).is_err());
    assert!(LinkTask::new(lattice, (0..12).collect(), vec![None; 8], None).is_err());
}

#[test]
fn a_cancel_stops_the_search() {
    let (params, instances) = reference_instances(&FLOW_V1, SIDE_3, 1);
    let (_, task) = LinkTask::from_instance(&FLOW_V1, params, &instances[0]).unwrap();
    let kernel = LoopKernel::new(&task);
    let starts: Vec<Vec<i8>> = (0..4).map(|k| lcg_state(k, 24)).collect();
    for stop_at in [1_u64, 5, 50] {
        let mut calls = 0;
        let mut cancelled = || {
            calls += 1;
            calls >= stop_at
        };
        let got = search(
            &kernel,
            &starts,
            1,
            Work::Fixed { rounds: ROUNDS },
            &mut cancelled,
            None,
        );
        assert_eq!(got.err(), Some(LoopError::Cancelled));
        assert_eq!(calls, stop_at);
    }
}

/// The golden cases: side-3 reference instances of both families, six
/// pseudo-random starts and the baseline orientation, a fixed seed, and
/// fixed work.
fn golden_cases() -> Vec<Value> {
    let mut cases = Vec::new();
    for contract in [FLOW_V1, POLARITON_V1] {
        let (params, instances) = reference_instances(&contract, SIDE_3, 1);
        let instance = &instances[0];
        let (_, task) = LinkTask::from_instance(&contract, params, instance).unwrap();
        let links = task.positions().len();
        let mut starts: Vec<Vec<i8>> = (0..6).map(|k| lcg_state(100 + k, links)).collect();
        let Ok(Outcome::Optimal(base)) = orient(&task, &mut never()) else {
            unreachable!("a generated instance has an optimum")
        };
        starts.push(base.link_spins);
        let seed = 2026;
        let out = checked_search(
            &task,
            Some(instance),
            &starts,
            seed,
            ROUNDS,
            &mut Seen::default(),
        );
        let graph = Graph::of(instance);
        let text =
            |s: &[i8]| -> String { s.iter().map(|&x| if x == 1 { '+' } else { '-' }).collect() };
        let reads: Vec<Value> = starts
            .iter()
            .zip(&out.states)
            .zip(&out.report.reads)
            .map(|((start, state), r)| {
                let nodes = task.node_spins(state).unwrap();
                json!({
                    "parent": r.parent,
                    "start_links": text(start),
                    "result_links": text(state),
                    "result_nodes": text(&nodes),
                    "energy_milli": graph.energy(&nodes),
                    "gap_squares": r.end.gap_squares,
                    "objective": r.end.objective,
                    "stop": format!("{:?}", r.stop),
                    "rounds": r.rounds,
                    "cycle_moves": r.cycle_moves,
                    "worm_moves": r.worm_moves,
                    "repair_moves": r.repair_moves,
                    "zero_moves": r.zero_moves,
                    "proposals": r.proposals,
                })
            })
            .collect();
        let hex: String = params.iter().map(|b| format!("{b:02x}")).collect();
        cases.push(json!({
            "family_id": contract.family_id,
            "revision": contract.revision,
            "params_hex": hex,
            "topology": SIDE_3,
            "nonce_byte": 0,
            "instance_digest": instance.digest().iter().map(|b| format!("{b:02x}")).collect::<String>(),
            "policy": FLOW_LOOPS_POLICY,
            "seed": seed,
            "work": {"fixed_rounds": ROUNDS},
            "reads": reads,
        }));
    }
    cases
}

#[test]
fn the_fixed_work_search_reproduces_the_recorded_golden_output() {
    let got = json!({ "format": "quip-cpu-flow-loops-golden/1", "cases": golden_cases() });
    assert_eq!(
        got,
        json!({ "format": "quip-cpu-flow-loops-golden/1", "cases": golden_cases() }),
        "the same input and seed give the same output"
    );
    if std::env::var_os(BLESS).is_some() {
        let text = serde_json::to_string_pretty(&got).unwrap() + "\n";
        std::fs::write(GOLDEN, text).expect("write the golden fixture");
        return;
    }
    let recorded: Value =
        serde_json::from_str(&std::fs::read_to_string(GOLDEN).expect("the golden fixture"))
            .expect("golden JSON");
    assert_eq!(got, recorded);
}

#[test]
fn time_bounded_search_records_host_use_and_every_final_state() {
    let cases = [
        (FLOW_V1, FLOW_RECIPES[2].params),
        (POLARITON_V1, POLARITON_RECIPES[0].params),
    ];
    for (contract, params) in cases {
        let instance = recipe_instance(&contract, params);
        let (_, task) = LinkTask::from_instance(&contract, params, &instance).unwrap();
        let links = task.positions().len();
        let starts: Vec<Vec<i8>> = (0..48).map(|k| lcg_state(k, links)).collect();
        let kernel = LoopKernel::new(&task);
        let work = Work::Budget {
            budget: Duration::from_secs_f64(FLOW_LOOPS_DEFAULT_BUDGET_SECONDS),
            started: Instant::now(),
        };
        let out = search(&kernel, &starts, 9, work, &mut never(), None).unwrap();
        let report = &out.report;
        assert_eq!(report.work, work);
        assert_eq!(report.policy, FLOW_LOOPS_POLICY);
        assert_eq!(
            report.budget_seconds,
            Some(FLOW_LOOPS_DEFAULT_BUDGET_SECONDS)
        );
        assert!(report.elapsed_seconds > 0.0);
        // Each check of the clock sits between two proposals, so the run
        // ends within one proposal of the budget; allow a loaded host.
        assert!(
            report.elapsed_seconds < FLOW_LOOPS_DEFAULT_BUDGET_SECONDS + 1.0,
            "{}",
            report.elapsed_seconds
        );
        assert_eq!(out.states.len(), starts.len());
        assert_eq!(report.reads.len(), starts.len());
        let spent: f64 = report.reads.iter().map(|r| r.seconds).sum();
        assert!(spent <= report.elapsed_seconds + 1e-6);
        for (k, (state, r)) in out.states.iter().zip(&report.reads).enumerate() {
            assert_eq!(r.parent, k);
            assert_eq!(Some(r.start), score(&task, &starts[k]));
            assert_eq!(Some(r.end), score(&task, state));
            assert!(r.end <= r.start);
            if r.stop == Stop::Skipped {
                assert_eq!(state, &starts[k], "a skipped read is returned unchanged");
            }
        }
        assert!(
            report.reads.iter().any(|r| r.end < r.start),
            "{}: no read improved",
            contract.family_id
        );
    }
}

/// The budget counts from the start instant that the caller gives, so the
/// setup of a job counts against it. A job whose setup used the whole budget
/// returns every start unchanged.
#[test]
fn setup_time_counts_against_the_budget() {
    let instance = recipe_instance(&FLOW_V1, FLOW_RECIPES[2].params);
    let (_, task) = LinkTask::from_instance(&FLOW_V1, FLOW_RECIPES[2].params, &instance).unwrap();
    let links = task.positions().len();
    let starts: Vec<Vec<i8>> = (0..4).map(|k| lcg_state(k, links)).collect();
    let kernel = LoopKernel::new(&task);
    let budget = Duration::from_millis(5);
    let started = Instant::now()
        .checked_sub(Duration::from_millis(50))
        .expect("a clock older than 50 ms");
    let out = search(
        &kernel,
        &starts,
        9,
        Work::Budget { budget, started },
        &mut never(),
        None,
    )
    .unwrap();
    assert_eq!(out.states, starts, "every start is returned unchanged");
    assert!(out.report.reads.iter().all(|r| r.stop == Stop::Skipped));
    assert!(out.report.elapsed_seconds >= 0.05);
}
