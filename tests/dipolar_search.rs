//! The `dipolar-strings/1` search on `quip/defects-static` revision 1.
//!
//! The oracle is the exhaustive enumeration of all 4096 states of the open
//! lattice with `L = 2`, checked against the family's own feasibility hook,
//! observables, and instance energy. The other tests cover admission, the
//! direct move check, routes, calibration, both work modes, the postprocess
//! capture, and a fixed-seed golden output.
#![cfg(feature = "experimental")]

use quip_solver_conformance::families::export_instances;
use quip_solver_core::ising_generators::defects::{
    check_route, defect_observables, RouteError, DEFECTS, DEFECTS_SCHEMA, DEFECTS_V1,
};
use quip_solver_core::ising_generators::descriptor::Params;
use quip_solver_core::ising_generators::dipolar::PhysicalStatus;
use quip_solver_core::ising_generators::family::BaseTopology;
use quip_solver_core::ising_generators::ice::zephyr_lattice;
use quip_solver_core::ising_generators::instance::{Instance, InstanceLimits, InstanceParts};
use quip_solver_core::ising_generators::square::{Boundary, SquareLattice};
use quip_solver_cpu::families::dipolar_search::{
    read_seed, DipolarError, Evidence, MoveFault, MoveKind, Problem, Request, State, Work,
    DIPOLAR_POLICY, MAX_TESTED_SIDE,
};
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// The parameters of a generated case.
#[derive(Clone, Copy, Debug)]
struct Case {
    m: u16,
    side: u16,
    pairs: u16,
    obstacles: u16,
    pins: u16,
    field_x: i32,
    field_y: i32,
}

impl Case {
    /// Canonical parameter bytes on `z(m, 4)`.
    fn bytes(self) -> Vec<u8> {
        let mut bytes = vec![0x01];
        bytes.extend_from_slice(&[1, 0, 2, 0, 0, 0]);
        bytes.extend_from_slice(&self.m.to_le_bytes());
        bytes.extend_from_slice(&[2, 0, 1, 0, 0, 0, 4]);
        for (tag, value) in [
            (3u8, self.side),
            (4, self.pairs),
            (5, self.obstacles),
            (6, self.pins),
        ] {
            bytes.extend_from_slice(&[tag, 0, 2, 0, 0, 0]);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for (tag, value) in [(7u8, self.field_x), (8, self.field_y)] {
            bytes.extend_from_slice(&[tag, 0, 4, 0, 0, 0]);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn instance(self, nonce: u8) -> Instance {
        let (nodes, edges) = zephyr_lattice(self.m, 4).expect("a lattice");
        let topology = BaseTopology::new(nodes, &edges).expect("a topology");
        export_instances(&DEFECTS_V1, &self.bytes(), &topology, &[[nonce; 32]])
            .expect("the case admits")
            .remove(0)
            .expect("the salt gives an instance")
    }

    fn problem(self, nonce: u8) -> (Instance, Problem) {
        let instance = self.instance(nonce);
        let problem =
            Problem::for_instance(&self.bytes(), &instance, Evidence::Analytic).expect("a problem");
        (instance, problem)
    }
}

/// The L = 2 cases: no constraint, two pins with mixed fields, every
/// boundary link pinned with the centre as an obstacle, and five pins.
const L2_CASES: [Case; 4] = [
    Case {
        m: 2,
        side: 2,
        pairs: 0,
        obstacles: 0,
        pins: 0,
        field_x: 0,
        field_y: 0,
    },
    Case {
        m: 2,
        side: 2,
        pairs: 0,
        obstacles: 0,
        pins: 2,
        field_x: 300,
        field_y: -200,
    },
    Case {
        m: 2,
        side: 2,
        pairs: 0,
        obstacles: 1,
        pins: 8,
        field_x: 1000,
        field_y: -1000,
    },
    Case {
        m: 2,
        side: 2,
        pairs: 0,
        obstacles: 1,
        pins: 5,
        field_x: -700,
        field_y: 450,
    },
];

/// L = 3 with one pair, one obstacle, and two pins.
const L3: Case = Case {
    m: 3,
    side: 3,
    pairs: 1,
    obstacles: 1,
    pins: 2,
    field_x: 1000,
    field_y: -1000,
};

fn state(bits: u32, n: usize) -> Vec<i8> {
    (0..n)
        .map(|i| if bits >> i & 1 == 1 { 1 } else { -1 })
        .collect()
}

/// The instance energy with the offset, from the instance coefficients.
fn instance_energy(instance: &Instance, spins: &[i8]) -> i64 {
    let mut total = instance.offset_milli();
    for (&h, &s) in instance.h().iter().zip(spins) {
        total += i64::from(h) * i64::from(s);
    }
    for (&(a, b), &j) in instance.edges().iter().zip(instance.j()) {
        total += i64::from(j) * i64::from(spins[a as usize]) * i64::from(spins[b as usize]);
    }
    total
}

fn flip(problem: &Problem, spins: &[i8], links: &[u32]) -> Vec<i8> {
    let mut after = spins.to_vec();
    for &e in links {
        let p = problem.request().positions[e as usize] as usize;
        after[p] = -after[p];
    }
    after
}

/// Whether two states agree on every link at an obstacle.
fn same_at_obstacles(problem: &Problem, a: &[i8], b: &[i8]) -> bool {
    let request = problem.request();
    request.obstacles.iter().all(|&v| {
        problem.lattice().incident(v).iter().all(|&(e, _)| {
            let p = request.positions[e as usize] as usize;
            a[p] == b[p]
        })
    })
}

/// Every one of the 4096 states of each L = 2 case agrees with the family:
/// the violations with the observables and the feasibility hook, the energy
/// with the instance, and the physical energy with the observables. Every
/// proposal from every state passes the direct check, keeps the violations,
/// and changes the score by the full recomputation.
#[test]
fn all_4096_l2_states_match_the_oracle() {
    let mut proposals = 0u64;
    for (k, case) in L2_CASES.iter().enumerate() {
        let (instance, problem) = case.problem(u8::try_from(k).unwrap());
        let bytes = case.bytes();
        let params = Params::parse(DEFECTS_SCHEMA, &bytes).unwrap();
        let hook = DEFECTS.feasible.expect("a hard family");
        assert_eq!(problem.nodes(), 12);
        assert_eq!(
            problem.physical_status(),
            PhysicalStatus::InsufficientPhysicalEvidence
        );
        let mut rng = SmallRng::seed_from_u64(31 + k as u64);
        let (mut max_feasible, mut min_infeasible) = (i64::MIN, i64::MAX);
        for bits in 0..4096u32 {
            let spins = state(bits, 12);
            let violations = problem.violations(&spins).unwrap();
            let seen = defect_observables(&params, &instance, &spins).expect("observables");
            assert_eq!(violations.labels, seen.violated_labels, "{case:?} {bits}");
            assert_eq!(violations.pins, seen.violated_pins, "{case:?} {bits}");
            assert_eq!(violations.feasible(), hook(&params, &instance, &spins));
            let energy = problem.energy(&spins).unwrap();
            assert_eq!(
                energy + problem.offset(),
                instance_energy(&instance, &spins)
            );
            assert_eq!(
                problem.physical_energy(&spins).unwrap(),
                seen.physical_energy
            );
            if violations.feasible() {
                assert_eq!(energy + problem.offset(), seen.physical_energy);
                max_feasible = max_feasible.max(energy);
            } else {
                min_infeasible = min_infeasible.min(energy);
            }
            let current = State::new(&problem, &spins).unwrap();
            assert_eq!(current.score(), energy);
            for _ in 0..3 {
                let Some(links) = problem.propose(&current, &mut rng) else {
                    continue;
                };
                proposals += 1;
                assert!(problem.check_move(&spins, &links).is_ok(), "{links:?}");
                let after = flip(&problem, &spins, &links);
                assert_eq!(problem.violations(&after).unwrap(), violations);
                assert!(same_at_obstacles(&problem, &spins, &after));
                let mut next = current.clone();
                let delta = next.apply(&problem, &links).unwrap();
                let full = problem.energy(&after).unwrap();
                assert_eq!(delta, full - energy, "{case:?} {bits} {links:?}");
                assert_eq!(next.score(), full);
                assert_eq!(next.spins(), after.as_slice());
            }
        }
        assert!(max_feasible < min_infeasible, "{case:?}: the penalty bound");
    }
    assert!(proposals > 10_000, "{proposals} proposals checked");
}

/// The lowest energy over the feasible states that keep the obstacle links
/// of `start`, by enumeration.
fn oracle_minimum(problem: &Problem, start: &[i8]) -> i64 {
    (0..4096u32)
        .map(|bits| state(bits, 12))
        .filter(|s| {
            problem.violations(s).unwrap().feasible() && same_at_obstacles(problem, s, start)
        })
        .map(|s| problem.energy(&s).unwrap())
        .min()
        .expect("a feasible state")
}

#[test]
fn the_search_reaches_the_l2_oracle_minimum() {
    for (k, case) in L2_CASES.iter().enumerate() {
        let (_, problem) = case.problem(u8::try_from(k).unwrap());
        let start = problem.start_state().unwrap();
        let want = oracle_minimum(&problem, &start);
        let run = problem
            .run(None, 4, Work::Fixed { proposals: 2000 }, 2032, None)
            .unwrap();
        assert_eq!(run.policy, DIPOLAR_POLICY);
        assert!(!run.warm);
        assert_eq!(run.reads.len(), 4);
        for read in &run.reads {
            assert!(read.feasible);
            assert!(same_at_obstacles(&problem, &read.spins, &start));
            assert_eq!(read.counts.proposed, 2000);
        }
        let best = run.reads.iter().map(|r| r.energy).min().unwrap();
        assert_eq!(best, want, "{case:?}");
    }
}

/// A boundary link joins two border vertices.
fn boundary_links(lattice: &SquareLattice) -> Vec<u32> {
    let side = u32::from(lattice.side());
    let border = |v: u32| {
        let (x, y) = (v % (side + 1), v / (side + 1));
        x == 0 || y == 0 || x == side || y == side
    };
    (0..lattice.link_count())
        .filter(|&e| {
            let link = lattice.link(e).unwrap();
            border(link.tail) && border(link.head)
        })
        .collect()
}

/// An analytic L = 2 fixture with the centre labeled `label` and two pins.
fn analytic(label: i8) -> Request {
    let lattice = SquareLattice::new(2, Boundary::Open).unwrap();
    let mut labels = vec![None; 9];
    labels[4] = Some(label);
    let pins = boundary_links(&lattice);
    Request {
        side: 2,
        labels,
        positions: (0..12).rev().collect(),
        obstacles: Vec::new(),
        pins: vec![pins[1], pins[6]],
        field_x: 250,
        field_y: 400,
    }
}

/// The analytic fixtures label the centre `+2` and `-2`, which the generator
/// cannot do at `L = 2`. Every state matches the direct charges, every
/// infeasible state costs more than every feasible state, and the search
/// reaches the oracle minimum.
#[test]
fn analytic_charged_fixtures_match_the_oracle() {
    for label in [2, -2, 0] {
        let problem = Problem::new(analytic(label), Evidence::Analytic).unwrap();
        let lattice = SquareLattice::new(2, Boundary::Open).unwrap();
        let (mut max_feasible, mut min_infeasible) = (i64::MIN, i64::MAX);
        for bits in 0..4096u32 {
            let spins = state(bits, 12);
            let links = problem.link_spins(&spins).unwrap();
            let charge: i32 = lattice
                .incident(4)
                .iter()
                .map(|&(e, sign)| i32::from(sign) * i32::from(links[e as usize]))
                .sum();
            let pins_hold = problem
                .request()
                .pins
                .iter()
                .all(|&e| links[e as usize] == 1);
            let violations = problem.violations(&spins).unwrap();
            assert_eq!(violations.labels == 0, charge == i32::from(label));
            assert_eq!(
                violations.feasible(),
                charge == i32::from(label) && pins_hold
            );
            let energy = problem.energy(&spins).unwrap();
            if violations.feasible() {
                assert_eq!(
                    energy + problem.offset(),
                    problem.physical_energy(&spins).unwrap()
                );
                max_feasible = max_feasible.max(energy);
            } else {
                min_infeasible = min_infeasible.min(energy);
            }
        }
        assert!(max_feasible < min_infeasible, "label {label}");
        let start = problem.start_state().unwrap();
        let run = problem
            .run(None, 4, Work::Fixed { proposals: 2000 }, 9, None)
            .unwrap();
        let best = run.reads.iter().map(|r| r.energy).min().unwrap();
        assert_eq!(best, oracle_minimum(&problem, &start), "label {label}");
    }
}

#[test]
fn invalid_requests_fail_admission() {
    let refused = |request: Request| {
        matches!(
            Problem::new(request, Evidence::Analytic),
            Err(DipolarError::Request(_))
        )
    };
    let base = analytic(2);
    assert!(Problem::new(base.clone(), Evidence::Analytic).is_ok());

    let mut r = base.clone();
    r.labels[0] = Some(0);
    assert!(refused(r), "a boundary label");
    let mut r = base.clone();
    r.labels[4] = Some(1);
    assert!(refused(r), "an odd label");
    let mut r = base.clone();
    r.labels[4] = Some(4);
    assert!(refused(r), "a label outside -2, 0, +2");
    let mut r = base.clone();
    r.labels.pop();
    assert!(refused(r), "too few labels");

    let mut r = base.clone();
    r.obstacles = vec![4];
    assert!(refused(r), "an obstacle with a nonzero label");
    let mut r = analytic(0);
    r.obstacles = vec![0];
    assert!(refused(r), "an obstacle on the boundary");
    let mut r = analytic(0);
    r.obstacles = vec![4, 4];
    assert!(refused(r), "a repeated obstacle");
    let mut r = analytic(0);
    r.obstacles = vec![99];
    assert!(refused(r), "an obstacle outside the lattice");

    let lattice = SquareLattice::new(2, Boundary::Open).unwrap();
    let interior = (0..12)
        .find(|e| !boundary_links(&lattice).contains(e))
        .unwrap();
    let mut r = base.clone();
    r.pins = vec![interior];
    assert!(refused(r), "a pin that is not a boundary link");
    let mut r = base.clone();
    r.pins = vec![base.pins[1], base.pins[0]];
    assert!(refused(r), "pins out of order");
    let mut r = base.clone();
    r.pins = vec![40];
    assert!(refused(r), "a pin outside the lattice");

    let mut r = base.clone();
    r.positions[0] = r.positions[1];
    assert!(refused(r), "positions that repeat a node");

    for side in [1, MAX_TESTED_SIDE + 1] {
        let mut r = base.clone();
        r.side = side;
        assert_eq!(
            Problem::new(r, Evidence::Analytic).err(),
            Some(DipolarError::UntestedSide(side))
        );
    }
}

/// A copy of `instance` with new metadata or fields. Admission rebuilds the
/// instance, so a tampered copy fails the contract check.
fn tampered(instance: &Instance, metadata: Vec<u8>, h: Vec<i32>) -> Instance {
    let parts = InstanceParts {
        descriptor_digest: *instance.descriptor_digest(),
        topology_order_digest: *instance.topology_order_digest(),
        nonce: instance.nonce().copied(),
        node_ids: instance.node_ids().to_vec(),
        edges: instance.edges().to_vec(),
        h,
        j: instance.j().to_vec(),
        offset_milli: instance.offset_milli(),
        metadata,
    };
    Instance::new(
        parts,
        &InstanceLimits::FAM_37,
        DEFECTS_V1.metadata_kinds,
        None,
    )
    .expect("a well-formed instance")
}

#[test]
fn tampered_instances_fail_admission() {
    let instance = L3.instance(3);
    let bytes = L3.bytes();
    assert!(Problem::for_instance(&bytes, &instance, Evidence::Analytic).is_ok());
    let region = instance.metadata().to_vec();
    // The last four bytes are the last pin id, and the four before are the
    // first pin id: swap them so the pins are out of order.
    let mut swapped = region.clone();
    let end = swapped.len();
    let (first, last) = (
        swapped[end - 8..end - 4].to_vec(),
        swapped[end - 4..].to_vec(),
    );
    swapped[end - 8..end - 4].copy_from_slice(&last);
    swapped[end - 4..].copy_from_slice(&first);
    let mut obstacle = region.clone();
    // The obstacle id follows the 2-byte obstacle count after the commitment.
    let at = end - 4 * 2 - 2 - 4;
    obstacle[at] ^= 1;
    let mut h = instance.h().to_vec();
    h[0] += 1;
    for (what, bad) in [
        (
            "pins out of order",
            tampered(&instance, swapped, instance.h().to_vec()),
        ),
        (
            "another obstacle",
            tampered(&instance, obstacle, instance.h().to_vec()),
        ),
        ("a changed field", tampered(&instance, region, h)),
    ] {
        assert!(
            matches!(
                Problem::for_instance(&bytes, &bad, Evidence::Analytic),
                Err(DipolarError::Metadata(_))
            ),
            "{what}"
        );
    }
    assert_eq!(
        Problem::for_instance(&L2_CASES[1].bytes(), &instance, Evidence::Analytic).err(),
        Some(DipolarError::Metadata("the contract metadata check fails")),
        "the params of another side"
    );
    assert_eq!(
        Problem::for_instance(&[1, 2, 3], &instance, Evidence::Analytic).err(),
        Some(DipolarError::Params)
    );
}

/// The vertex that `e` points out of and into under the node state `spins`.
fn ends(problem: &Problem, spins: &[i8], e: u32) -> (u32, u32) {
    let link = problem.lattice().link(e).unwrap();
    let s = spins[problem.request().positions[e as usize] as usize];
    if s == 1 {
        (link.tail, link.head)
    } else {
        (link.head, link.tail)
    }
}

/// The route of a state whose flipped links are vertex-disjoint paths from
/// each `+2` vertex: follow the flipped links out of it.
fn route_of(problem: &Problem, spins: &[i8]) -> Vec<Vec<u32>> {
    let request = problem.request();
    let links = problem.link_spins(spins).unwrap();
    let mut route = Vec::new();
    for (v, label) in (0u32..).zip(&request.labels) {
        if *label != Some(2) {
            continue;
        }
        let mut path = vec![v];
        let mut at = v;
        while request.labels[at as usize] != Some(-2) {
            let next = problem
                .lattice()
                .incident(at)
                .iter()
                .filter(|&&(e, _)| links[e as usize] == -1)
                .map(|&(e, _)| {
                    let link = problem.lattice().link(e).unwrap();
                    if link.tail == at {
                        link.head
                    } else {
                        link.tail
                    }
                })
                .find(|w| !path.contains(w))
                .expect("the path continues");
            path.push(next);
            at = next;
        }
        route.push(path);
    }
    route
}

#[test]
fn invalid_moves_and_routes_fail_direct_checking() {
    let (instance, problem) = L3.problem(5);
    let request = problem.request().clone();
    let spins = problem.start_state().unwrap();
    let fault = |links: &[u32]| match problem.check_move(&spins, links) {
        Err(DipolarError::Move(fault)) => Some(fault),
        _ => None,
    };
    assert_eq!(fault(&[]), Some(MoveFault::Empty));
    assert_eq!(fault(&[999]), Some(MoveFault::UnknownLink(999)));
    assert_eq!(
        fault(&[request.pins[0]]),
        Some(MoveFault::Pinned(request.pins[0]))
    );
    let free_link = (0..problem.lattice().link_count())
        .find(|e| !request.pins.contains(e))
        .unwrap();
    assert_eq!(
        fault(&[free_link, free_link]),
        Some(MoveFault::RepeatedLink(free_link))
    );
    let too_long: Vec<u32> = (0..=problem.lattice().link_count()).collect();
    assert_eq!(fault(&too_long), Some(MoveFault::TooLong));

    // A step along a link into the head of the first link, and a step along
    // a link that does not touch that head.
    let movable: Vec<u32> = (0..problem.lattice().link_count())
        .filter(|e| !request.pins.contains(e))
        .collect();
    let (first, against) = movable
        .iter()
        .find_map(|&a| {
            let (_, w) = ends(&problem, &spins, a);
            movable
                .iter()
                .find(|&&b| b != a && ends(&problem, &spins, b).1 == w)
                .map(|&b| (a, b))
        })
        .expect("two links into one vertex");
    assert_eq!(
        fault(&[first, against]),
        Some(MoveFault::NotDirected(against))
    );
    let (_, w) = ends(&problem, &spins, first);
    let away = movable
        .iter()
        .find(|&&e| {
            let (a, b) = ends(&problem, &spins, e);
            a != w && b != w
        })
        .copied()
        .unwrap();
    assert_eq!(fault(&[first, away]), Some(MoveFault::NotAdjacent(away)));

    // A link out of the obstacle, and a link from a labeled vertex to the
    // border.
    let obstacle = request.obstacles[0];
    let out_of_obstacle = problem
        .lattice()
        .incident(obstacle)
        .iter()
        .map(|&(e, _)| e)
        .find(|&e| ends(&problem, &spins, e).0 == obstacle)
        .unwrap();
    assert_eq!(
        fault(&[out_of_obstacle]),
        Some(MoveFault::Obstacle(obstacle))
    );
    let labeled_to_border = (0..problem.lattice().link_count()).find_map(|e| {
        let (a, b) = ends(&problem, &spins, e);
        let labeled = request.labels[a as usize].is_some() && a != obstacle;
        (labeled && request.labels[b as usize].is_none() && !request.pins.contains(&e))
            .then_some((e, a))
    });
    let (e, a) = labeled_to_border.expect("an interior link to the border");
    assert_eq!(fault(&[e]), Some(MoveFault::ChargeChanged(a)));

    // The start state of a pair is one path between the opposite labels that
    // avoids the obstacle, so the family accepts its route.
    let bytes = L3.bytes();
    let params = Params::parse(DEFECTS_SCHEMA, &bytes).unwrap();
    let route = route_of(&problem, &spins);
    assert_eq!(check_route(&params, &instance, &spins, &route), Ok(()));
    let mut reversed = route.clone();
    reversed[0].reverse();
    assert_eq!(check_route(&params, &instance, &spins, &reversed), Ok(()));
    let short = vec![vec![route[0][0]]];
    assert_eq!(
        check_route(&params, &instance, &spins, &short),
        Err(RouteError::Shape)
    );
    let mut wrong_end = route.clone();
    wrong_end[0].pop();
    let check = check_route(&params, &instance, &spins, &wrong_end);
    assert!(
        matches!(check, Err(RouteError::Endpoints | RouteError::Shape)),
        "{check:?}"
    );
    assert_eq!(
        check_route(&params, &instance, &vec![1; spins.len()], &route),
        Err(RouteError::Difference)
    );
}

/// Every proposal of a long walk on L = 3 keeps the constraints and the
/// incremental score equal to the full recomputation. Without an obstacle,
/// the interior block carries loops.
#[test]
fn incremental_scores_match_full_recomputation() {
    let open = Case {
        pairs: 0,
        obstacles: 0,
        pins: 0,
        ..L3
    };
    let (_, problem) = open.problem(1);
    let mut current = State::new(&problem, &problem.start_state().unwrap()).unwrap();
    let mut rng = SmallRng::seed_from_u64(77);
    let (mut loops, mut strings) = (0, 0);
    for _ in 0..5000 {
        let Some(links) = problem.propose(&current, &mut rng) else {
            continue;
        };
        match problem.check_move(current.spins(), &links).unwrap() {
            MoveKind::Loop => loops += 1,
            MoveKind::String => strings += 1,
        }
        current.apply(&problem, &links).unwrap();
        assert_eq!(current.score(), problem.energy(current.spins()).unwrap());
        assert!(problem.violations(current.spins()).unwrap().feasible());
    }
    assert!(loops > 0 && strings > 0, "{loops} loops, {strings} strings");
}

#[test]
fn an_uncalibrated_physical_job_is_rejected() {
    let instance = L3.instance(2);
    assert_eq!(
        Problem::for_instance(&L3.bytes(), &instance, Evidence::Physical).err(),
        Some(DipolarError::Uncalibrated)
    );
    assert_eq!(
        Problem::new(analytic(2), Evidence::Physical).err(),
        Some(DipolarError::Uncalibrated)
    );
    assert!(DipolarError::Uncalibrated
        .to_string()
        .contains("insufficient physical evidence"));
    // The analytic fixtures stay runnable.
    let problem = Problem::new(analytic(2), Evidence::Analytic).unwrap();
    assert_eq!(
        problem.physical_status(),
        PhysicalStatus::InsufficientPhysicalEvidence
    );
    let run = problem
        .run(None, 1, Work::Fixed { proposals: 50 }, 0, None)
        .unwrap();
    assert!(run.reads[0].feasible);
}

/// The postprocess mode keeps one final state per supplied state, in order,
/// repairs the infeasible ones, and leaves a feasible state unchanged when
/// the work is zero.
#[test]
fn the_postprocess_captures_every_supplied_state() {
    let (_, problem) = L3.problem(4);
    let n = problem.nodes();
    let feasible = problem.start_state().unwrap();
    let starts = vec![
        vec![1; n],
        feasible.clone(),
        vec![-1; n],
        (0..n).map(|i| if i % 3 == 0 { 1 } else { -1 }).collect(),
    ];
    let still = problem
        .run(
            Some(&starts),
            starts.len(),
            Work::Fixed { proposals: 0 },
            1,
            None,
        )
        .unwrap();
    assert!(still.warm);
    assert_eq!(still.reads.len(), starts.len());
    assert_eq!(still.reads[1].spins, feasible);
    assert!(!still.reads[1].repaired);
    let run = problem
        .run(
            Some(&starts),
            starts.len(),
            Work::Fixed { proposals: 3000 },
            1,
            None,
        )
        .unwrap();
    assert_eq!(run.reads.len(), starts.len());
    for (read, start) in run.reads.iter().zip(&starts) {
        assert!(read.feasible);
        assert_eq!(
            read.repaired,
            !problem.violations(start).unwrap().feasible()
        );
        assert_eq!(read.energy, problem.energy(&read.spins).unwrap());
        assert_eq!(read.energy + problem.offset(), read.physical_energy);
        assert_eq!(
            read.resolved,
            read.improvement.unsigned_abs() > problem.rounding_bound_milli()
        );
    }
    assert!(run.reads[1].energy <= still.reads[1].energy);
}

/// The read seeds come from the job seed alone. One seed repeats exactly,
/// and another seed walks another way on the same instance.
#[test]
fn solver_seeds_are_separate_from_the_instance() {
    let (_, problem) = L3.problem(6);
    let work = Work::Fixed { proposals: 400 };
    let a = problem.run(None, 3, work, 11, None).unwrap();
    let b = problem.run(None, 3, work, 11, None).unwrap();
    let c = problem.run(None, 3, work, 12, None).unwrap();
    let states = |run: &quip_solver_cpu::families::dipolar_search::RunOutcome| {
        run.reads
            .iter()
            .map(|r| (r.spins.clone(), r.counts))
            .collect::<Vec<_>>()
    };
    assert_eq!(states(&a), states(&b));
    assert_ne!(states(&a), states(&c));
    assert_ne!(read_seed(11, 0), read_seed(11, 1));
    assert_ne!(read_seed(11, 0), read_seed(12, 0));
    // Another nonce gives another instance, and the seeds do not move.
    let (_, other) = L3.problem(7);
    assert_ne!(other.h(), problem.h());
    assert_eq!(read_seed(11, 0), read_seed(11, 0));
}

#[test]
fn the_budget_mode_runs_against_the_declared_budget() {
    let (_, problem) = L3.problem(8);
    let run = problem
        .run(None, 1, Work::Budget { seconds: 0.05 }, 3, None)
        .unwrap();
    let read = &run.reads[0];
    assert!(read.counts.proposed > 0);
    assert!(
        read.seconds >= 0.05 && read.seconds < 2.0,
        "{}",
        read.seconds
    );
    assert!(read.feasible);
}

/// A fixed-seed run on a fixed L = 3 instance gives these final energies.
#[test]
fn the_fixed_seed_output_is_golden() {
    let (_, problem) = L3.problem(9);
    let run = problem
        .run(None, 3, Work::Fixed { proposals: 500 }, 2032, None)
        .unwrap();
    let got: Vec<(i64, String)> = run
        .reads
        .iter()
        .map(|r| {
            let bits = r
                .spins
                .iter()
                .map(|&s| if s == 1 { '+' } else { '-' })
                .collect();
            (r.energy, bits)
        })
        .collect();
    let want: Vec<(i64, String)> = GOLDEN.iter().map(|&(e, s)| (e, s.to_owned())).collect();
    assert_eq!(got, want);
}

const GOLDEN: [(i64, &str); 3] = [
    (-675_648, "--+---+++---++++++-+++++"),
    (-675_648, "--+---+++---++++++-+++++"),
    (-675_648, "--+---+++---++++++-+++++"),
];
