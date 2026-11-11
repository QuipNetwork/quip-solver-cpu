//! The exact flow baselines of `quip_solver_cpu::families::flow`: the F-static
//! min-cost circulation and the adapted-P compatible-charge construction.
//!
//! Every tiny F-static optimum agrees with exhaustive enumeration of the
//! instance energy, every adapted-P construction meets the declared charges,
//! infeasible requests return a checked cut, and the solver declines other
//! families.

use quip_solver_conformance::families::{export_instances, reference_revision};
use quip_solver_core::ising_generators::defects::DEFECTS_V1;
use quip_solver_core::ising_generators::descriptor::{Contract, Params};
use quip_solver_core::ising_generators::family::BaseTopology;
use quip_solver_core::ising_generators::flow::{flow_observables, FLOW_RECIPES, FLOW_V1};
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::ising_generators::polariton::{
    polariton_observables, POLARITON_RECIPES, POLARITON_V1,
};
use quip_solver_core::ising_generators::routing::ROUTING_V1;
use quip_solver_core::ising_generators::square::{charges, Boundary, SquareLattice};
use quip_solver_cpu::families::flow::{
    orient, FlowError, Infeasibility, LinkCosts, LinkTask, Mode, Outcome, ResidualNetwork, Solved,
    TaskError,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// The live z(12, 4) topology of the registered recipes.
const LIVE_TOPOLOGY: &str = "../topologies/zephyr-z12t4-live.bin.gz";

fn topology(name: &str) -> BaseTopology {
    let stored = quip_solver_conformance::topology(name).expect("a stored topology");
    BaseTopology::new(stored.nodes, &stored.edges).expect("a base topology")
}

/// The instances of `params` on `topology` for `count` nonces.
fn instances(
    contract: &Contract,
    params: &[u8],
    topology: &BaseTopology,
    count: u8,
) -> Vec<Instance> {
    let nonces: Vec<[u8; 32]> = (0..count).map(|i| [i; 32]).collect();
    export_instances(contract, params, topology, &nonces)
        .expect("the params admit")
        .into_iter()
        .filter_map(Result::ok)
        .collect()
}

/// The small topologies of the registered vectors: side 2, side 3, and side 3
/// with a permuted node order.
const SMALL: [&str; 3] = [
    "../topologies/routing-small.bin.gz",
    "../topologies/routing-no-pairs.bin.gz",
    "../topologies/polariton-l3-permuted.bin.gz",
];

/// The reference parameter sets of `contract` with their instances on every
/// small topology that admits them.
fn reference_instances(contract: &Contract, count: u8) -> Vec<(&'static [u8], Instance)> {
    let reference =
        reference_revision(contract.family_id, contract.revision).expect("a reference revision");
    let mut found = Vec::new();
    for name in SMALL {
        let topology = topology(name);
        for &params in reference.params {
            let nonces: Vec<[u8; 32]> = (0..count).map(|i| [i; 32]).collect();
            let Ok(built) = export_instances(contract, params, &topology, &nonces) else {
                continue;
            };
            found.extend(
                built
                    .into_iter()
                    .filter_map(Result::ok)
                    .map(|i| (params, i)),
            );
        }
    }
    found
}

/// The instance energy of `spins` in milli, offset included.
fn energy(instance: &Instance, spins: &[i8]) -> i64 {
    let mut e = instance.offset_milli();
    for (&h, &s) in instance.h().iter().zip(spins) {
        e += i64::from(h) * i64::from(s);
    }
    for (&(u, v), &j) in instance.edges().iter().zip(instance.j()) {
        e += i64::from(j) * i64::from(spins[u as usize]) * i64::from(spins[v as usize]);
    }
    e
}

/// The lowest instance energy over every state, by Gray-code enumeration.
fn ground_energy(instance: &Instance) -> i64 {
    let n = instance.node_ids().len();
    assert!(n <= 26, "{n} nodes are too many to enumerate");
    let mut adjacent: Vec<Vec<(usize, i64)>> = vec![Vec::new(); n];
    for (&(u, v), &j) in instance.edges().iter().zip(instance.j()) {
        adjacent[u as usize].push((v as usize, i64::from(j)));
        adjacent[v as usize].push((u as usize, i64::from(j)));
    }
    let mut spins = vec![-1_i8; n];
    let mut e = energy(instance, &spins);
    let mut best = e;
    for step in 1_u64..(1 << n) {
        let i = step.trailing_zeros() as usize;
        let field = i64::from(instance.h()[i])
            + adjacent[i]
                .iter()
                .map(|&(k, j)| j * i64::from(spins[k]))
                .sum::<i64>();
        e -= 2 * i64::from(spins[i]) * field;
        spins[i] = -spins[i];
        best = best.min(e);
    }
    best
}

fn never() -> impl FnMut() -> bool {
    || false
}

/// The canonical feasibility hook of `contract` on `spins`.
fn hook_feasible(contract: &Contract, params: &[u8], instance: &Instance, spins: &[i8]) -> bool {
    let parsed = Params::parse(contract.schema, params).expect("canonical params");
    let hook = contract
        .construction
        .and_then(|c| c.feasible)
        .expect("a hard family");
    hook(&parsed, instance, spins)
}

/// Solve `instance` and return the instance spins of the certified optimum.
fn solve(contract: &Contract, params: &[u8], instance: &Instance) -> (Mode, u64, Vec<i8>) {
    let (mode, task) = LinkTask::from_instance(contract, params, instance).expect("a link task");
    let Ok(Outcome::Optimal(orientation)) = orient(&task, &mut never()) else {
        unreachable!("a generated instance has a certified optimum")
    };
    let spins = task
        .node_spins(&orientation.link_spins)
        .expect("one spin per node");
    assert!(hook_feasible(contract, params, instance, &spins));
    (mode, orientation.objective, spins)
}

#[test]
fn every_tiny_f_static_optimum_agrees_with_exhaustive_enumeration() {
    let cases = reference_instances(&FLOW_V1, 24);
    let mut checked = [0_usize; 2];
    for (params, instance) in &cases {
        let n = instance.node_ids().len();
        // Side 2 has 12 links and side 3 has 24. Enumerate side 3 on a few,
        // on both side-3 topologies.
        if n > 12 && checked[1] >= 4 {
            continue;
        }
        let (mode, objective, spins) = solve(&FLOW_V1, params, instance);
        assert_eq!(mode, Mode::FlowStatic);
        let observed = flow_observables(
            &Params::parse(FLOW_V1.schema, params).unwrap(),
            instance,
            &spins,
        )
        .expect("observables");
        assert_eq!(observed.violations, 0);
        assert_eq!(observed.objective, objective);
        let got = energy(instance, &spins);
        assert_eq!(
            got,
            i64::try_from(objective).unwrap(),
            "energy is the objective"
        );
        assert_eq!(
            ground_energy(instance),
            got,
            "the optimum is the ground state"
        );
        checked[usize::from(n > 12)] += 1;
    }
    assert!(checked[0] >= 20 && checked[1] == 4, "{checked:?}");
}

/// A random label for a vertex of `degree`: the parity of the degree, at most
/// the degree in size, and now and then one that breaks either rule.
fn random_label(rng: &mut SmallRng, degree: i8) -> i8 {
    if rng.gen_ratio(1, 20) {
        return degree + 1;
    }
    let steps = rng.gen_range(0..=degree);
    -degree + 2 * steps
}

/// Every orientation of a side-2 lattice: the best objective among those that
/// meet every label, or `None` when none does.
fn enumerate(task: &LinkTask) -> Option<u64> {
    let links = task.lattice().link_count();
    (0_u32..1 << links)
        .filter_map(|bits| {
            let spins: Vec<i8> = (0..links)
                .map(|e| if bits >> e & 1 == 1 { 1 } else { -1 })
                .collect();
            let check = task.check(&spins).expect("a check");
            check.feasible().then_some(check.objective)
        })
        .min()
}

fn identity_positions(lattice: &SquareLattice) -> Vec<u32> {
    (0..lattice.link_count()).collect()
}

/// Check one synthetic task against enumeration, and return whether it is
/// feasible.
fn agrees_with_enumeration(task: &LinkTask) -> bool {
    let best = enumerate(task);
    match orient(task, &mut never()).expect("no flow error") {
        Outcome::Optimal(orientation) => {
            assert_eq!(Some(orientation.objective), best, "{task:?}");
            let check = task.check(&orientation.link_spins).unwrap();
            assert!(check.feasible());
            assert_eq!(check.objective, orientation.objective);
            true
        }
        Outcome::Infeasible(why) => {
            assert_eq!(best, None, "{why} {task:?}");
            if let Infeasibility::Cut(cut) = why {
                assert!(cut.required > cut.capacity, "{cut:?}");
            }
            false
        }
    }
}

#[test]
fn synthetic_f_static_requests_agree_with_enumeration() {
    let mut rng = SmallRng::seed_from_u64(2021);
    let lattice = SquareLattice::new(2, Boundary::Open).unwrap();
    let (mut feasible, mut infeasible) = (0, 0);
    for case in 0..400 {
        let links = lattice.link_count() as usize;
        let mut labels: Vec<Option<i8>> = if case % 2 == 0 {
            // The charges of a random orientation, then now and then 2 units
            // moved between two vertices, which may or may not stay feasible.
            let spins: Vec<i8> = (0..links)
                .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
                .collect();
            let mut labels: Vec<Option<i8>> = charges(&lattice, &spins)
                .unwrap()
                .into_iter()
                .map(Some)
                .collect();
            if rng.gen_bool(0.5) {
                let (a, b) = (rng.gen_range(0..9), rng.gen_range(0..9));
                labels[a] = labels[a].map(|q| q + 2);
                labels[b] = labels[b].map(|q| q - 2);
            }
            labels
        } else {
            (0..lattice.vertex_count())
                .map(|v| {
                    let degree = i8::try_from(lattice.degree(v).unwrap()).unwrap();
                    Some(random_label(&mut rng, degree))
                })
                .collect()
        };
        // Some random cases move the total to 0 at the centre vertex, so the
        // flow decides them rather than the total.
        if case % 4 == 1 {
            let rest: i32 = labels.iter().flatten().map(|&q| i32::from(q)).sum::<i32>()
                - i32::from(labels[4].unwrap());
            labels[4] = i8::try_from(-rest).ok();
            if labels[4].is_none() {
                continue;
            }
        }
        let costs = LinkCosts {
            targets: (0..links)
                .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
                .collect(),
            costs: (0..links).map(|_| 2 * rng.gen_range(0..=6)).collect(),
        };
        let task = LinkTask::new(
            lattice.clone(),
            identity_positions(&lattice),
            labels,
            Some(costs),
        )
        .unwrap();
        if agrees_with_enumeration(&task) {
            feasible += 1;
        } else {
            infeasible += 1;
        }
    }
    assert!(
        feasible >= 50 && infeasible >= 50,
        "{feasible} {infeasible}"
    );
}

#[test]
fn synthetic_adapted_p_requests_agree_with_enumeration() {
    let mut rng = SmallRng::seed_from_u64(8);
    let lattice = SquareLattice::new(2, Boundary::Open).unwrap();
    let (mut feasible, mut infeasible) = (0, 0);
    for _ in 0..400 {
        let labels: Vec<Option<i8>> = (0..lattice.vertex_count())
            .map(|v| {
                let degree = i8::try_from(lattice.degree(v).unwrap()).unwrap();
                rng.gen_bool(0.6).then(|| random_label(&mut rng, degree))
            })
            .collect();
        let task =
            LinkTask::new(lattice.clone(), identity_positions(&lattice), labels, None).unwrap();
        if agrees_with_enumeration(&task) {
            feasible += 1;
        } else {
            infeasible += 1;
        }
    }
    assert!(
        feasible >= 50 && infeasible >= 50,
        "{feasible} {infeasible}"
    );
}

#[test]
fn the_adapted_p_construction_meets_the_declared_charges() {
    let mut cases = reference_instances(&POLARITON_V1, 8);
    let live = topology(LIVE_TOPOLOGY);
    for recipe in &POLARITON_RECIPES {
        cases.extend(
            instances(&POLARITON_V1, recipe.params, &live, 3)
                .into_iter()
                .map(|instance| (recipe.params, instance)),
        );
    }
    assert!(cases.len() >= 20, "{}", cases.len());
    for (params, instance) in &cases {
        let (mode, objective, spins) = solve(&POLARITON_V1, params, instance);
        assert_eq!(mode, Mode::AdaptedP);
        assert_eq!(objective, 0);
        let observed = polariton_observables(
            &Params::parse(POLARITON_V1.schema, params).unwrap(),
            instance,
            &spins,
        )
        .expect("observables");
        assert_eq!(observed.violations, 0);
        assert_eq!(observed.loss_milli, 0);
        assert_eq!(energy(instance, &spins), 0, "zero charge loss");
    }
}

#[test]
fn the_f_static_recipes_have_certified_optima() {
    let live = topology(LIVE_TOPOLOGY);
    let mut solved = 0;
    for recipe in &FLOW_RECIPES[..3] {
        for instance in instances(&FLOW_V1, recipe.params, &live, 3) {
            let (_, objective, spins) = solve(&FLOW_V1, recipe.params, &instance);
            assert_eq!(energy(&instance, &spins), i64::try_from(objective).unwrap());
            solved += 1;
        }
    }
    assert_eq!(solved, 9);
}

/// Every flow of a unit-capacity network, by enumeration: the least cost of a
/// flow that meets every supply, or `None`.
fn brute_force(arcs: &[(u32, u32, i64)], supply: &[i64]) -> Option<i64> {
    (0_u32..1 << arcs.len())
        .filter_map(|bits| {
            let mut net = vec![0_i64; supply.len()];
            let mut cost = 0;
            for (k, &(u, v, c)) in arcs.iter().enumerate() {
                if bits >> k & 1 == 1 {
                    net[u as usize] += 1;
                    net[v as usize] -= 1;
                    cost += c;
                }
            }
            (net == supply).then_some(cost)
        })
        .min()
}

#[test]
fn the_residual_network_matches_brute_force_with_negative_costs() {
    let mut rng = SmallRng::seed_from_u64(77);
    let (mut feasible, mut infeasible) = (0, 0);
    for _ in 0..500 {
        let nodes = rng.gen_range(2_u32..=6);
        let arcs: Vec<(u32, u32, i64)> = (0..rng.gen_range(1..=11))
            .map(|_| {
                let u = rng.gen_range(0..nodes);
                let v = (u + rng.gen_range(1..nodes)) % nodes;
                (u, v, rng.gen_range(-5..=5))
            })
            .collect();
        let mut supply: Vec<i64> = (0..nodes).map(|_| rng.gen_range(-2..=2)).collect();
        let total: i64 = supply.iter().sum();
        supply[0] -= total;
        let mut network = ResidualNetwork::new(nodes as usize).unwrap();
        let ids: Vec<_> = arcs
            .iter()
            .map(|&(u, v, c)| network.add_arc(u, v, 1, c).unwrap())
            .collect();
        for (v, &s) in (0_u32..).zip(&supply) {
            network.add_supply(v, s).unwrap();
        }
        let want = brute_force(&arcs, &supply);
        match network.solve(&mut never()).unwrap() {
            Solved::Optimal(certificate) => {
                assert_eq!(Some(certificate.cost), want, "{arcs:?} {supply:?}");
                let flows: i64 = ids
                    .iter()
                    .zip(&arcs)
                    .map(|(&id, &(_, _, c))| i64::from(network.flow(id)) * c)
                    .sum();
                assert_eq!(flows, certificate.cost);
                assert_eq!(
                    network.certify(&certificate.potentials),
                    Ok(certificate.cost)
                );
                feasible += 1;
            }
            Solved::Infeasible(cut) => {
                assert_eq!(want, None, "{arcs:?} {supply:?}");
                let inside: Vec<bool> = (0..nodes).map(|v| cut.nodes.contains(&v)).collect();
                let (required, capacity) = network.cut_balance(&inside).unwrap();
                assert_eq!((required, capacity), (cut.required, cut.capacity));
                assert!(required > capacity);
                infeasible += 1;
            }
        }
    }
    assert!(
        feasible >= 100 && infeasible >= 100,
        "{feasible} {infeasible}"
    );
}

#[test]
fn overflow_and_imbalance_are_explicit_errors() {
    let mut network = ResidualNetwork::new(3).unwrap();
    assert_eq!(network.add_arc(0, 1, 1, i64::MIN), Err(FlowError::Overflow));
    assert_eq!(network.add_arc(0, 3, 1, 0), Err(FlowError::NodeOutOfRange));
    network.add_arc(0, 1, 1, i64::MAX - 1).unwrap();
    network.add_arc(1, 2, 1, i64::MAX - 1).unwrap();
    network.add_supply(0, 1).unwrap();
    assert_eq!(network.solve(&mut never()), Err(FlowError::Unbalanced));
    network.add_supply(2, -1).unwrap();
    assert_eq!(network.solve(&mut never()), Err(FlowError::Overflow));
    assert_eq!(network.add_supply(0, i64::MAX), Err(FlowError::Overflow));
}

#[test]
fn a_cancel_stops_the_solve_between_graph_operations() {
    let live = topology(LIVE_TOPOLOGY);
    let instance = instances(&FLOW_V1, FLOW_RECIPES[2].params, &live, 1)
        .pop()
        .expect("an l8 instance");
    let (_, task) = LinkTask::from_instance(&FLOW_V1, FLOW_RECIPES[2].params, &instance).unwrap();
    let Ok(Outcome::Optimal(orientation)) = orient(&task, &mut never()) else {
        unreachable!("an optimum")
    };
    assert!(orientation.augmentations >= 2);
    for stop_at in [1_u64, 2, orientation.augmentations] {
        let mut calls = 0;
        let mut cancelled = || {
            calls += 1;
            calls >= stop_at
        };
        assert_eq!(orient(&task, &mut cancelled), Err(FlowError::Cancelled));
        assert_eq!(calls, stop_at);
    }
}

#[test]
fn other_families_and_mismatched_metadata_are_rejected() {
    for contract in [DEFECTS_V1, ROUTING_V1] {
        let (params, instance) = reference_instances(&contract, 2)
            .into_iter()
            .next()
            .expect("a reference instance");
        assert_eq!(
            LinkTask::from_instance(&contract, params, &instance).err(),
            Some(TaskError::Unsupported {
                family_id: contract.family_id,
                revision: contract.revision,
            }),
            "{}",
            contract.family_id
        );
    }
    // A polariton instance read as F-static fails the metadata check before
    // any section is parsed.
    let (_, polariton) = reference_instances(&POLARITON_V1, 1)
        .into_iter()
        .next()
        .unwrap();
    let flow_params = reference_revision(FLOW_V1.family_id, 1).unwrap().params[0];
    assert_eq!(
        LinkTask::from_instance(&FLOW_V1, flow_params, &polariton).err(),
        Some(TaskError::Metadata)
    );
}
