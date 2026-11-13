//! The original two-commodity routing attack (`families::routing`).
//!
//! `tests/fixtures/routing_vectors.json` holds synthetic street-grid models with
//! spins that no arc uses, and the final states of the research solver
//! `routesolve` for both start orders. `research/routing_parity/export.py` builds
//! it by running the research binary, and scores each state again with numpy.
//! The research code reads no clock, so the saved states are reproduced bit for
//! bit. The ignored `research_parity` test replays the saved states of the
//! `t1-flow` campaign.
#![cfg(feature = "experimental")]

use std::path::Path;

use quip_protocol::scoring::energy_milli;
use quip_solver_conformance::families::{export_instances, reference_revision};
use quip_solver_core::ising_generators::family::BaseTopology;
use quip_solver_core::ising_generators::instance::{Instance, InstanceParts};
use quip_solver_core::ising_generators::registry::CONTRACTS;
use quip_solver_core::ising_generators::routing::{routing_defects, ROUTING_V1};
use quip_solver_core::ising_generators::Code;
use quip_solver_core::{CancelToken, IsingGraph, SampleError, SampleParams, Sampler};
use quip_solver_cpu::families::routing::{Parts, Routing, RoutingError, ROUTING_SECTION};
use quip_solver_cpu::{
    solve_routing_instance, RoutingSampler, CPU_ROUTING_IDENTITY, ROUTING_SOLVER_ID,
};
use serde_json::Value;

/// A model of the fixture, or of a research dump.
struct Case {
    parts: Parts,
    spins: usize,
    spare: Vec<usize>,
    states: Vec<Vec<i8>>,
    energies: Vec<i64>,
    graph: IsingGraph,
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_i64().expect("an integer"))
        .collect()
}

fn pairs(v: &Value) -> Vec<[i64; 2]> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|p| {
            let p = ints(p);
            [p[0], p[1]]
        })
        .collect()
}

fn case(v: &Value) -> Case {
    let to_u32 = |x: i64| u32::try_from(x).expect("a position");
    let h = ints(&v["h"]);
    let edges: Vec<(u32, u32, i64)> = pairs(&v["edges"])
        .iter()
        .zip(ints(&v["j"]))
        .map(|(e, j)| (to_u32(e[0]), to_u32(e[1]), j))
        .collect();
    let milli = |x: i64| x as f64 / 1000.0;
    let graph = IsingGraph::new(
        h.iter().map(|&x| milli(x)).collect(),
        edges.iter().map(|e| milli(e.2)).collect(),
        edges.iter().map(|e| (e.0 as usize, e.1 as usize)).collect(),
    );
    Case {
        spins: h.len(),
        spare: v
            .get("spare")
            .map(|s| ints(s).iter().map(|&x| x as usize).collect())
            .unwrap_or_default(),
        states: v["states"]
            .as_array()
            .expect("states")
            .iter()
            .map(|row| ints(row).iter().map(|&x| x as i8).collect())
            .collect(),
        energies: ints(&v["energies"]),
        parts: Parts {
            h,
            edges,
            supply: pairs(&v["supply"]),
            arcs: pairs(&v["arcs"])
                .iter()
                .map(|a| (to_u32(a[0]), to_u32(a[1])))
                .collect(),
            spin1: ints(&v["spin1"]).into_iter().map(to_u32).collect(),
            chain2: pairs(&v["chain2"])
                .iter()
                .map(|c| [to_u32(c[0]), to_u32(c[1])])
                .collect(),
            cost: pairs(&v["cost"]),
        },
        graph,
    }
}

fn fixture() -> Vec<Case> {
    let text = include_str!("fixtures/routing_vectors.json");
    let root: Value = serde_json::from_str(text).expect("the fixture parses");
    root["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(case)
        .collect()
}

/// The source states and scores: both start orders of every model.
fn check_source(c: &Case, at: &str) {
    let model = Routing::new(c.parts.clone()).expect("a valid model");
    assert_eq!(model.spins(), c.spins, "{at}");
    for first in 0..2 {
        let got = model.candidate(first, None).expect("a final state");
        assert_eq!(got.spins, c.states[first], "{at}: start order {first}");
        assert_eq!(got.energy, c.energies[first], "{at}: start order {first}");
        // Independent score: the production scorer on the f64 graph.
        assert_eq!(
            energy_milli(&got.spins, &c.graph.h, &c.graph.j, &c.graph.edges),
            c.energies[first],
            "{at}: start order {first}"
        );
        assert!(got.spins.iter().all(|&s| s == 1 || s == -1), "{at}");
        assert_eq!(got.spins.len(), c.spins, "{at}");
    }
}

#[test]
fn frozen_source_states_are_reproduced_for_both_orders() {
    let cases = fixture();
    assert!(cases.len() >= 6);
    for (k, c) in cases.iter().enumerate() {
        check_source(c, &format!("model {k}"));
    }
}

#[test]
fn the_start_orders_differ_so_the_policy_certifies_no_optimum() {
    // On some model the two orders end at different energies, so at least one
    // of them is not a global optimum. The full policy is a heuristic.
    let differing = fixture()
        .iter()
        .filter(|c| c.energies[0] != c.energies[1])
        .count();
    assert!(differing >= 2, "{differing} models differ between orders");
}

#[test]
fn inactive_spins_keep_their_positions_and_the_vector_stays_whole() {
    let mut checked = 0;
    for c in fixture() {
        if c.spare.is_empty() {
            continue;
        }
        let model = Routing::new(c.parts.clone()).expect("a valid model");
        for first in 0..2 {
            let got = model.candidate(first, None).expect("a final state");
            assert_eq!(got.spins.len(), c.spins);
            for &spin in &c.spare {
                assert_eq!(got.spins[spin], 1, "spare spin {spin} is +1");
            }
            // The used spins fill the positions around the spare ones.
            assert_eq!(got.spins, c.states[first]);
        }
        checked += 1;
    }
    assert!(checked >= 4, "{checked} models with inactive spins");
}

#[test]
fn a_cancelled_run_stops_with_the_cancel_error() {
    let c = &fixture()[4];
    let model = Routing::new(c.parts.clone()).expect("a valid model");
    let token = CancelToken::default();
    token.cancel_through(5);
    assert_eq!(
        model.candidate(0, Some((&token, Some(3)))),
        Err(RoutingError::Cancelled)
    );
    // A watermark above the cancel point runs to the end.
    let done = model
        .candidate(0, Some((&token, Some(9))))
        .expect("a state");
    assert_eq!(done.energy, c.energies[0]);
}

#[test]
fn malformed_models_and_chains_are_refused() {
    let base = fixture().remove(1).parts;
    assert!(Routing::new(base.clone()).is_ok());
    let refused = |edit: &dyn Fn(&mut Parts)| {
        let mut parts = base.clone();
        edit(&mut parts);
        matches!(Routing::new(parts), Err(RoutingError::Model(_)))
    };
    // A chain whose two spins are one spin.
    assert!(refused(&|p| p.chain2[0][1] = p.chain2[0][0]));
    // A chain spin that another arc owns.
    assert!(refused(&|p| p.chain2[0][1] = p.spin1[1]));
    // A spin past the end of the vector.
    assert!(refused(
        &|p| p.chain2[0][1] = u32::try_from(p.h.len()).unwrap()
    ));
    assert!(refused(&|p| p.spin1[2] = u32::MAX));
    // An arc end that is no node, and an arc from a node to itself.
    assert!(refused(
        &|p| p.arcs[0].0 = u32::try_from(p.supply.len()).unwrap()
    ));
    assert!(refused(&|p| p.arcs[0].1 = p.arcs[0].0));
    // Array lengths that disagree.
    assert!(refused(&|p| {
        p.cost.pop();
    }));
    assert!(refused(&|p| {
        p.chain2.pop();
    }));
    // A supply that is no unit, and a field beyond the milli range.
    assert!(refused(&|p| p.supply[0][0] = 2));
    assert!(refused(&|p| p.h[0] = i64::MAX));
    // A coupling to a spin that does not exist, and a self-loop.
    assert!(refused(&|p| p.edges[0].1 = u32::MAX));
    assert!(refused(&|p| p.edges[0].1 = p.edges[0].0));
    // No arcs, and no spins.
    assert!(refused(&|p| {
        p.arcs.clear();
        p.spin1.clear();
        p.chain2.clear();
        p.cost.clear();
    }));
    assert!(refused(&|p| p.h.clear()));
}

// ---------------------------------------------------------------- family instances

/// Routing instances of the conformance reference, and the graph of each.
fn instances() -> Vec<(Instance, IsingGraph)> {
    let reference = reference_revision(ROUTING_V1.family_id, ROUTING_V1.revision)
        .expect("a reference revision");
    let stored = quip_solver_conformance::topology(reference.topology).expect("the topology");
    let topology = BaseTopology::new(stored.nodes, &stored.edges).expect("a topology");
    let nonces: Vec<[u8; 32]> = (0_u8..6).map(|i| [i; 32]).collect();
    let mut out = Vec::new();
    for params in reference.params {
        if let Ok(batch) = export_instances(reference.contract, params, &topology, &nonces) {
            out.extend(batch.into_iter().filter_map(Result::ok));
        }
    }
    assert!(!out.is_empty(), "the reference gives instances");
    out.into_iter()
        .map(|instance| {
            let graph = graph_of(&instance);
            (instance, graph)
        })
        .collect()
}

fn graph_of(instance: &Instance) -> IsingGraph {
    let milli = |v: &i32| f64::from(*v) / 1000.0;
    IsingGraph::new(
        instance.h().iter().map(milli).collect(),
        instance.j().iter().map(milli).collect(),
        instance
            .edges()
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect(),
    )
}

fn parts_of(instance: &Instance, metadata: Vec<u8>) -> InstanceParts {
    InstanceParts {
        descriptor_digest: *instance.descriptor_digest(),
        topology_order_digest: *instance.topology_order_digest(),
        nonce: instance.nonce().copied(),
        node_ids: instance.node_ids().to_vec(),
        edges: instance.edges().to_vec(),
        h: instance.h().to_vec(),
        j: instance.j().to_vec(),
        offset_milli: instance.offset_milli(),
        metadata,
    }
}

fn rebuilt(instance: &Instance, metadata: Vec<u8>) -> Result<Instance, Code> {
    Instance::new(
        parts_of(instance, metadata),
        &ROUTING_V1.limits,
        &[ROUTING_SECTION, FOREIGN_SECTION],
        None,
    )
}

/// A registered section kind that is not the routing kind: the twisted-ice pairs.
const FOREIGN_SECTION: u16 = 3;

/// A metadata region of the sections `(kind, payload)`.
fn region(sections: &[(u16, &[u8])]) -> Vec<u8> {
    let mut out = u16::try_from(sections.len())
        .unwrap()
        .to_le_bytes()
        .to_vec();
    for (kind, payload) in sections {
        out.extend(kind.to_le_bytes());
        out.extend(u32::try_from(payload.len()).unwrap().to_le_bytes());
        out.extend(*payload);
    }
    out
}

/// The payload of the one routing section of `instance`.
fn payload(instance: &Instance) -> Vec<u8> {
    instance.metadata()[6..].to_vec()
}

#[test]
fn family_instances_give_two_rescored_final_states() {
    for (instance, graph) in instances() {
        let reads = solve_routing_instance(&graph, &instance).expect("reads");
        assert_eq!(reads.len(), 2);
        assert!(reads[0].energy_milli <= reads[1].energy_milli);
        for read in &reads {
            assert_eq!(read.spins.len(), graph.h.len());
            assert_eq!(
                read.energy_milli,
                energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
            );
            // The final descent leaves a single-flip local minimum.
            for i in 0..read.spins.len() {
                let mut flipped = read.spins.clone();
                flipped[i] = -flipped[i];
                assert!(
                    energy_milli(&flipped, &graph.h, &graph.j, &graph.edges) >= read.energy_milli,
                    "spin {i} still lowers the energy"
                );
            }
            // The state carries routing bits that the family reads back.
            let params = ROUTING_V1_PARAMS.iter().find_map(|p| {
                routing_defects(&ising_params(p), &instance, &read.spins).map(|d| (p, d))
            });
            let (_, defects) = params.expect("the defect counts decode");
            assert_eq!(defects.broken_chains, 0, "the chains stay whole");
            assert_eq!(defects.capacity_conflicts, 0, "no arc is shared");
        }
    }
}

const ROUTING_V1_PARAMS: [&[u8]; 2] = [
    &quip_solver_conformance::families::routing::P_ROUTING_Z24_D4,
    &quip_solver_conformance::families::routing::P_ROUTING_Z34_D6,
];

fn ising_params(bytes: &[u8]) -> quip_solver_core::ising_generators::descriptor::Params<'_> {
    quip_solver_core::ising_generators::descriptor::Params::parse(ROUTING_V1.schema, bytes)
        .expect("canonical parameters")
}

#[test]
fn unknown_metadata_fails_admission() {
    let (instance, graph) = instances().remove(0);
    let body = payload(&instance);
    assert!(Routing::from_instance(&instance).is_ok());
    let declined = |metadata: Vec<u8>, must_build: bool| {
        // The instance layer already refuses a region with a bad layout.
        let Ok(tampered) = rebuilt(&instance, metadata) else {
            return !must_build;
        };
        assert!(matches!(
            Routing::from_instance(&tampered),
            Err(RoutingError::Model(_))
        ));
        matches!(
            solve_routing_instance(&graph, &tampered),
            Err(SampleError::Declined(_))
        )
    };
    // An unknown section kind alone.
    assert!(declined(region(&[(FOREIGN_SECTION, &body)]), true));
    // The routing section and a second, unknown one.
    assert!(declined(
        region(&[(FOREIGN_SECTION, b"x"), (ROUTING_SECTION, &body)]),
        true
    ));
    // No section.
    assert!(declined(region(&[]), false));
    // A byte after the routing payload.
    let mut long = body.clone();
    long.push(0);
    assert!(declined(region(&[(ROUTING_SECTION, &long)]), true));
    // A chain strength outside its range, and an unknown arc cost.
    let mut weak = body.clone();
    let end = weak.len();
    weak[end - 2..].copy_from_slice(&0u16.to_le_bytes());
    assert!(declined(region(&[(ROUTING_SECTION, &weak)]), true));
    let mut costly = body.clone();
    let cost_at = costly.len() - 2 - 4;
    costly[cost_at..cost_at + 2].copy_from_slice(&63u16.to_le_bytes());
    assert!(declined(region(&[(ROUTING_SECTION, &costly)]), true));
    // A truncated section.
    let short = &body[..body.len() - 3];
    assert!(declined(region(&[(ROUTING_SECTION, short)]), true));
    // A graph that is not the instance.
    let mut skewed = graph.clone();
    let mut h = skewed.h.clone();
    h[0] += 0.5;
    skewed = IsingGraph::new(h, skewed.j.clone(), skewed.edges.to_vec());
    assert!(matches!(
        solve_routing_instance(&skewed, &instance),
        Err(SampleError::Declined(_))
    ));
}

#[test]
fn non_routing_jobs_are_annealed() {
    let n = 6;
    let ring = IsingGraph::new(
        vec![0.0; n],
        vec![-1.0; n],
        (0..n).map(|i| (i, (i + 1) % n)).collect(),
    );
    let sampler = RoutingSampler::new();
    let params = SampleParams {
        num_reads: 2,
        num_sweeps: 16,
        seed: 1,
        ..Default::default()
    };
    let reads = sampler
        .sample(&ring, &params)
        .expect("the generic sampler solves a plain job");
    assert_eq!(reads.len(), 2);
}

#[test]
fn the_declaration_claims_only_routing_revision_one() {
    let table = RoutingSampler::declaration()
        .validate()
        .expect("a valid declaration");
    // Nothing is declared as generated. Every registered family is generated
    // on the solver-core default session path, in registry order.
    assert!(CONTRACTS
        .iter()
        .all(|c| table.generated().iter().any(|g| g.family_id == c.family_id)));
    assert!(!table.declares_generated(ROUTING_V1.family_id, ROUTING_V1.revision));
    assert_eq!(table.accepted().len(), 1);
    assert_eq!(table.accepted()[0].family_id, ROUTING_V1.family_id);
    assert_eq!(table.accepted()[0].revisions, vec![ROUTING_V1.revision]);
    assert_eq!(table.solver_id(), ROUTING_SOLVER_ID);
    assert_eq!(
        CPU_ROUTING_IDENTITY.algorithm,
        quip_proto::v1::Algorithm::Routing
    );
    assert_eq!(ROUTING_SOLVER_ID, "quip-cpu-routing/1");
}

/// Replays the saved `t1-flow` states of the research. Build the directory
/// with `research/routing_parity/export.py research`.
#[test]
#[ignore = "needs QUIP_ROUTING_PARITY from research/routing_parity/export.py"]
fn research_parity() {
    let dir = std::env::var("QUIP_ROUTING_PARITY").expect("QUIP_ROUTING_PARITY");
    let mut files: Vec<_> = std::fs::read_dir(Path::new(&dir))
        .expect("the parity directory")
        .map(|e| e.expect("an entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no cases in {dir}");
    for file in &files {
        let v: Value =
            serde_json::from_str(&std::fs::read_to_string(file).expect("read")).expect("json");
        let c = case(&v);
        let at = file
            .file_name()
            .expect("a name")
            .to_string_lossy()
            .into_owned();
        assert!(
            c.spins > 3 * c.parts.arcs.len(),
            "{at}: the model has inactive spins"
        );
        check_source(&c, &at);
    }
}
