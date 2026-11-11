//! The routing gadget repair (`gadget-repair/1`) against fixed-work vectors
//! from the research source, against direct constraint checks on registered
//! routing instances, and against the saved research states when the opt-in
//! parity data is present.
//!
//! `tests/fixtures/gadget_repair_vectors.json` comes from
//! `research/gadget_repair_parity/export.py vectors`. The research code ran on
//! two small synthetic routing models: `gadgets.extract`, the neighbour
//! tables of `run.py`, `repair_run` of `repair.c` with its clock replaced by a
//! count of clock reads, and the start schedule of `one_model` with recorded
//! work.
#![cfg(feature = "experimental")]

use std::path::Path;
use std::time::Instant;

use quip_solver_conformance::families::{export_instances, reference_revision};
use quip_solver_core::ising_generators::family::BaseTopology;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::ising_generators::routing::ROUTING_V1;
use quip_solver_core::{IsingGraph, SampleError, SampleParams, Sampler};
use quip_solver_cpu::postprocess::gadget_repair::{
    arcs_of, check_routing, check_spin_cover, is_feasible, prepare_starts, schedule_gadget, ArcMap,
    ChainStarts, ClockSpec, GadgetError, Repair, RepairOptions, Structure, Work,
    GADGET_BUDGET_SECONDS, GADGET_POLICY,
};
use quip_solver_cpu::postprocess::gadget_repair_sampler::routing_reads;
use quip_solver_cpu::postprocess::greedy::{natural_moves, Greedy, Ising, MoveSources};
use quip_solver_cpu::GadgetRepairSampler;
use serde_json::Value;

const VECTORS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/gadget_repair_vectors.json"
);

fn spins(text: &Value) -> Vec<i8> {
    text.as_str()
        .expect("a spin string")
        .bytes()
        .map(|c| if c == b'+' { 1 } else { -1 })
        .collect()
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_i64().expect("an integer"))
        .collect()
}

fn node(v: &Value) -> u32 {
    u32::try_from(v.as_u64().expect("a node")).expect("a u32 node")
}

fn rows(v: &Value) -> Vec<Vec<u32>> {
    v.as_array()
        .expect("rows")
        .iter()
        .map(|row| row.as_array().expect("a row").iter().map(node).collect())
        .collect()
}

fn ising(case: &Value) -> Ising {
    let edges: Vec<(u32, u32)> = rows(&case["edges"]).iter().map(|e| (e[0], e[1])).collect();
    Ising::new(&ints(&case["h"]), &edges, &ints(&case["j"])).expect("a valid model")
}

fn arcs(case: &Value) -> Vec<ArcMap> {
    let ends = rows(&case["arc_nodes"]);
    let chain = rows(&case["chain"]);
    let x1: Vec<u32> = case["x1"]
        .as_array()
        .expect("x1")
        .iter()
        .map(node)
        .collect();
    ends.iter()
        .zip(&chain)
        .zip(&x1)
        .map(|((e, c), &x)| ArcMap {
            tail: e[0],
            head: e[1],
            x1: x,
            chain: [c[0], c[1]],
        })
        .collect()
}

/// The natural-move descent of the research `gnat`: the single flips, the
/// edge pairs, and the rigid groups.
fn nat_greedy(ising: &Ising) -> Greedy {
    let (moves, _) = natural_moves(ising, &MoveSources::default()).expect("natural moves");
    Greedy::new(ising.clone(), &moves, true).expect("a move list")
}

fn options(starts: &str, depth: u64, tenure: u64, stall: u64) -> RepairOptions {
    RepairOptions {
        depth: i32::try_from(depth).unwrap(),
        tenure: i64::try_from(tenure).unwrap(),
        tenure_rand: std::cmp::max(1, tenure / 2),
        stall: i64::try_from(stall).unwrap(),
        starts: match starts {
            "all" => ChainStarts::All,
            "singles" => ChainStarts::Singles,
            other => unreachable!("the vectors name an unknown start mode {other}"),
        },
    }
}

fn vectors() -> Value {
    serde_json::from_str(&std::fs::read_to_string(VECTORS).expect("the vectors"))
        .expect("valid json")
}

#[test]
fn the_structure_matches_the_research_extraction() {
    let doc = vectors();
    for case in doc["cases"].as_array().unwrap() {
        let at = case["name"].as_str().unwrap();
        let want = &case["structure"];
        let got = Structure::extract(case["h"].as_array().unwrap().len(), &arcs(case))
            .expect("a structure");
        assert_eq!(got.units, rows(&want["units"]), "{at}: units");
        assert_eq!(got.gadgets, rows(&want["gadgets"]), "{at}: gadgets");
        assert_eq!(
            got.n_starts as u64,
            want["n_starts"].as_u64().unwrap(),
            "{at}: starts"
        );
        let ptr: Vec<usize> = ints(&want["gm_ptr"])
            .iter()
            .map(|&p| usize::try_from(p).unwrap())
            .collect();
        let idx: Vec<u32> = ints(&want["gm_idx"])
            .iter()
            .map(|&p| u32::try_from(p).unwrap())
            .collect();
        assert_eq!(got.gm_ptr, ptr, "{at}: neighbour offsets");
        assert_eq!(got.gm_idx, idx, "{at}: neighbours");
        assert_eq!(
            got.units.len() as u64,
            want["kinds"]["units"].as_u64().unwrap(),
            "{at}: unit count"
        );
        assert_eq!(
            got.gadgets.len() as u64,
            want["kinds"]["gadgets"].as_u64().unwrap(),
            "{at}: gadget count"
        );
    }
}

#[test]
fn repair_with_a_fixed_clock_matches_the_research_runs() {
    let doc = vectors();
    let mut tabu_runs = 0;
    for case in doc["cases"].as_array().unwrap() {
        let at = case["name"].as_str().unwrap();
        let ising = ising(case);
        let n = ising.num_nodes();
        let structure = Structure::extract(n, &arcs(case)).expect("a structure");
        let repair = Repair::new(ising, &structure).expect("repair tables");
        let reads: Vec<Vec<i8>> = case["reads"]
            .as_array()
            .unwrap()
            .iter()
            .map(spins)
            .collect();
        for run in case["repairs"].as_array().unwrap() {
            let note = format!("{at} read {} ticks {}", run["read"], run["ticks"]);
            let opts = options(
                run["chain_starts"].as_str().unwrap(),
                run["depth"].as_u64().unwrap(),
                run["tenure"].as_u64().unwrap(),
                run["stall"].as_u64().unwrap(),
            );
            let mut s = reads[usize::try_from(run["read"].as_u64().unwrap()).unwrap()].clone();
            let (delta, stats, ticks) = repair
                .run(
                    &mut s,
                    &opts,
                    run["seed"].as_u64().unwrap(),
                    ClockSpec::Ticks(run["ticks"].as_u64().unwrap(), None),
                )
                .expect("a repair");
            assert_eq!(delta, run["delta"].as_i64().unwrap(), "{note}: delta");
            let want = ints(&run["stats"]);
            let got = [
                stats.iterations,
                stats.chains_kept,
                stats.sweeps,
                stats.steps,
                stats.new_bests,
            ];
            assert_eq!(got.to_vec(), want, "{note}: stats");
            assert_eq!(
                i64::try_from(ticks).unwrap(),
                run["ticks_used"].as_i64().unwrap(),
                "{note}: clock reads"
            );
            assert_eq!(s, spins(&run["state"]), "{note}: state");
            if stats.steps > 0 {
                tabu_runs += 1;
            }
        }
    }
    assert!(tabu_runs > 0, "the vectors never reach the tabu phase");
}

#[test]
fn the_schedule_with_recorded_work_matches_the_research_start_loop() {
    let doc = vectors();
    for case in doc["cases"].as_array().unwrap() {
        let at = case["name"].as_str().unwrap();
        let ising = ising(case);
        let nat = nat_greedy(&ising);
        let structure = Structure::extract(ising.num_nodes(), &arcs(case)).unwrap();
        let repair = Repair::new(ising.clone(), &structure).unwrap();
        let reads: Vec<Vec<i8>> = case["reads"]
            .as_array()
            .unwrap()
            .iter()
            .map(spins)
            .collect();
        for (k, sched) in case["schedules"].as_array().unwrap().iter().enumerate() {
            let note = format!("{at} schedule {k}");
            let work = Work::Fixed {
                descended: usize::try_from(sched["descended"].as_u64().unwrap()).unwrap(),
                ticks: sched["ticks"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|t| t.as_u64().unwrap())
                    .collect(),
            };
            let seed = sched["seed"].as_u64().unwrap();
            let out = schedule_gadget(&nat, &repair, &reads, seed, &work, None).expect("a run");
            assert_eq!(out.report.policy, GADGET_POLICY);
            assert_eq!(out.report.budget_seconds, GADGET_BUDGET_SECONDS);
            let starts = sched["starts"].as_array().unwrap();
            assert_eq!(out.pool.len(), starts.len(), "{note}: repaired starts");
            for (j, (want, got)) in starts.iter().zip(&out.report.starts).enumerate() {
                assert_eq!(got.start_energy, want["e0"].as_i64().unwrap(), "{note} {j}");
                assert_eq!(got.energy, want["e"].as_i64().unwrap(), "{note} {j}");
                assert_eq!(
                    i64::try_from(got.ticks).unwrap(),
                    want["ticks_used"].as_i64().unwrap(),
                    "{note} {j}: clock reads"
                );
                assert_eq!(out.pool[j], spins(&want["state"]), "{note} {j}: state");
            }
            assert_eq!(
                out.report.energy,
                sched["energy"].as_i64().unwrap(),
                "{note}"
            );
            assert_eq!(out.report.energy, ising.energy(&out.state), "{note}");
            assert_eq!(out.state, spins(&sched["state"]), "{note}: final state");
            let want_starts = ints(&sched["start_energies"]);
            let prepared = prepare_starts(&nat, &reads, &work, Instant::now(), None).unwrap();
            let got_starts: Vec<i64> = prepared.starts.iter().map(|s| s.energy).collect();
            assert_eq!(got_starts, want_starts, "{note}: start energies");
        }
    }
}

// ---------------------------------------------------------------- malformed models

fn arc(tail: u32, head: u32, x1: u32, c0: u32, c1: u32) -> ArcMap {
    ArcMap {
        tail,
        head,
        x1,
        chain: [c0, c1],
    }
}

#[test]
fn a_malformed_chain_or_commodity_mapping_fails_with_the_reason() {
    let good = [arc(0, 1, 0, 1, 2), arc(1, 2, 3, 4, 5)];
    assert!(Structure::extract(6, &good).is_ok());
    let cases: Vec<(Vec<ArcMap>, usize, GadgetError)> = vec![
        (vec![], 0, GadgetError::NoArcs),
        (
            vec![arc(0, 1, 0, 1, 1)],
            3,
            GadgetError::ChainRepeat { arc: 0 },
        ),
        (
            vec![arc(0, 0, 0, 1, 2)],
            3,
            GadgetError::SelfLoop { arc: 0 },
        ),
        (
            vec![arc(0, 1, 0, 1, 7)],
            3,
            GadgetError::SpinOutside { arc: 0, spin: 7 },
        ),
        (
            vec![arc(0, 1, 0, 1, 2), arc(1, 2, 2, 4, 5)],
            6,
            GadgetError::SpinReused { arc: 1, spin: 2 },
        ),
        (
            vec![arc(0, 1, 0, 1, 2), arc(1, 2, 3, 3, 4)],
            6,
            GadgetError::SpinReused { arc: 1, spin: 3 },
        ),
    ];
    assert_eq!(check_spin_cover(6, &good), Ok(()));
    assert_eq!(
        check_spin_cover(7, &good),
        Err(GadgetError::SpinCount {
            mapped: 6,
            spins: 7
        })
    );
    for (arcs, n, want) in cases {
        let got = Structure::extract(n, &arcs).expect_err("a malformed mapping");
        assert_eq!(got, want);
        assert!(!got.to_string().is_empty());
    }
}

#[test]
fn a_chain_without_a_ferromagnetic_coupler_is_reported_by_arc() {
    let arcs = [arc(0, 1, 0, 1, 2)];
    let model = |j_chain: i64, with_capacity: bool| {
        let mut edges = vec![(1, 2)];
        let mut j = vec![j_chain];
        if with_capacity {
            edges.push((0, 1));
            j.push(124);
        }
        Ising::new(&[0, 0, 0], &edges, &j).unwrap()
    };
    assert_eq!(Repair::check_couplers(&model(-900, true), &arcs), Ok(()));
    assert_eq!(
        Repair::check_couplers(&model(900, true), &arcs),
        Err(GadgetError::ChainCoupler { arc: 0 })
    );
    assert_eq!(
        Repair::check_couplers(&model(-900, false), &arcs),
        Err(GadgetError::CapacityCoupler { arc: 0 })
    );
    let no_chain = Ising::new(&[0, 0, 0], &[(0, 1)], &[124]).unwrap();
    assert_eq!(
        Repair::check_couplers(&no_chain, &arcs),
        Err(GadgetError::ChainCoupler { arc: 0 })
    );
}

// ---------------------------------------------------------------- registered routing instances

/// The largest routing instance of the reference revision over its registered
/// parameter sets and a few nonces, with its parameters and its graph.
fn routing_job() -> (Instance, Vec<u8>, IsingGraph) {
    let reference = reference_revision(ROUTING_V1.family_id, ROUTING_V1.revision)
        .expect("a reference revision");
    let stored = quip_solver_conformance::topology(reference.topology).unwrap();
    let topology = BaseTopology::new(stored.nodes, &stored.edges).unwrap();
    let nonces: Vec<[u8; 32]> = (0_u8..16).map(|i| [i; 32]).collect();
    let mut found: Option<(Instance, Vec<u8>)> = None;
    for params in reference.params {
        let Ok(made) = export_instances(reference.contract, params, &topology, &nonces) else {
            continue;
        };
        for instance in made.into_iter().filter_map(Result::ok) {
            if found
                .as_ref()
                .is_none_or(|(best, _)| instance.node_ids().len() > best.node_ids().len())
            {
                found = Some((instance, params.to_vec()));
            }
        }
    }
    let (instance, params) = found.expect("a reference params set gives an instance");
    let milli = |v: &i32| f64::from(*v) / 1000.0;
    let graph = IsingGraph::new(
        instance.h().iter().map(milli).collect(),
        instance.j().iter().map(milli).collect(),
        instance
            .edges()
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect(),
    );
    (instance, params, graph)
}

fn random_states(n: usize, count: usize, seed: u64) -> Vec<Vec<i8>> {
    let mut x = seed;
    (0..count)
        .map(|_| {
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    if x & 1 == 1 {
                        1
                    } else {
                        -1
                    }
                })
                .collect()
        })
        .collect()
}

fn params(seed: u64) -> SampleParams {
    SampleParams {
        num_reads: 8,
        seed,
        ..SampleParams::default()
    }
}

/// Defects counted straight from the arcs, with no help from the family
/// checker: capacity conflicts, broken chains, and the arcs each commodity
/// uses.
fn direct_counts(arcs: &[ArcMap], s: &[i8]) -> (u32, u32, [u32; 2]) {
    let on = |i: u32| s[i as usize] == -1;
    let mut conflicts = 0;
    let mut broken = 0;
    let mut used = [0; 2];
    for a in arcs {
        let (c1, c2) = (on(a.x1), on(a.chain[0]));
        conflicts += u32::from(c1 && c2);
        broken += u32::from(on(a.chain[0]) != on(a.chain[1]));
        used[0] += u32::from(c1);
        used[1] += u32::from(c2);
    }
    (conflicts, broken, used)
}

#[test]
fn a_registered_instance_maps_to_a_valid_structure_and_agrees_with_the_direct_counts() {
    let (instance, family, graph) = routing_job();
    let arcs = arcs_of(&instance).expect("the routing arcs");
    assert!(!arcs.is_empty());
    let structure = Structure::extract(graph.h.len(), &arcs).expect("a structure");
    assert!(
        structure.n_starts < structure.units.len(),
        "pair units exist"
    );
    for s in random_states(graph.h.len(), 6, 3) {
        let got = check_routing(&family, &instance, &s).expect("defects");
        let (conflicts, broken, used) = direct_counts(&arcs, &s);
        assert_eq!(got.capacity_conflicts, conflicts);
        assert_eq!(got.broken_chains, broken);
        assert_eq!(got.used_arcs, used);
    }
    // A state of the wrong length and a value other than 1 or -1 are errors.
    assert_eq!(
        check_routing(&family, &instance, &[1, 1]),
        Err(GadgetError::Invalid)
    );
    let mut bad = vec![1; graph.h.len()];
    bad[0] = 0;
    assert_eq!(
        check_routing(&family, &instance, &bad),
        Err(GadgetError::Invalid)
    );
    assert_eq!(
        check_routing(&family[..3], &instance, &vec![1; graph.h.len()]),
        Err(GadgetError::Invalid)
    );
}

#[test]
fn the_output_reports_defects_that_remain_when_the_energy_improves() {
    let (instance, family, graph) = routing_job();
    let energy =
        |s: &[i8]| quip_protocol::scoring::energy_milli(s, &graph.h, &graph.j, &graph.edges);
    // The descent of one read, with no repair: some local minima break the
    // constraints, and the report must say so while the energy is lower.
    let descent_only = Work::Fixed {
        descended: 1,
        ticks: Vec::new(),
    };
    let mut improved_and_broken = 0;
    for (seed, start) in random_states(graph.h.len(), 400, 7).into_iter().enumerate() {
        let starts = vec![start];
        let job = routing_reads(
            &graph,
            &params(seed as u64),
            Some(&starts),
            &family,
            &instance,
            &descent_only,
        )
        .expect("a routing job");
        assert_eq!(job.reads.len(), 1, "no repair ran");
        assert_eq!(job.reads[0].energy_milli, energy(&job.reads[0].spins));
        assert_eq!(job.best_start_energy, energy(&starts[0]));
        // The defects are the direct check of the returned state.
        assert_eq!(
            job.final_defects,
            check_routing(&family, &instance, &job.reads[0].spins).unwrap()
        );
        assert_eq!(
            job.best_start_defects,
            check_routing(&family, &instance, &starts[0]).unwrap()
        );
        if job.reads[0].energy_milli < job.best_start_energy && !is_feasible(&job.final_defects) {
            improved_and_broken += 1;
        }
    }
    assert!(
        improved_and_broken > 0,
        "no descent ended in an infeasible local minimum"
    );
}

#[test]
fn the_repair_reports_the_policy_the_work_and_the_offset() {
    let (instance, family, graph) = routing_job();
    let starts = random_states(graph.h.len(), 6, 11);
    let work = Work::Fixed {
        descended: 6,
        ticks: vec![30; 5],
    };
    let job = routing_reads(&graph, &params(5), Some(&starts), &family, &instance, &work)
        .expect("a routing job");
    assert_eq!(job.reads.len(), 1 + job.report.starts.len());
    for read in &job.reads {
        assert_eq!(
            read.energy_milli,
            quip_protocol::scoring::energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges),
            "every state is rescored from the job graph"
        );
    }
    let best = job.reads[1..].iter().map(|r| r.energy_milli).min().unwrap();
    assert!(job.reads[0].energy_milli <= best, "the final state leads");
    assert!(job.reads[0].energy_milli <= job.best_start_energy);
    assert_eq!(job.report.policy, GADGET_POLICY);
    assert_eq!(job.report.work, work);
    assert_eq!(job.report.energy, job.reads[0].energy_milli);
    assert_eq!(job.offset_milli, instance.offset_milli());
}

#[test]
fn a_run_is_reproducible_and_the_budget_is_reported_apart_from_the_time() {
    let (instance, family, graph) = routing_job();
    let starts = random_states(graph.h.len(), 8, 21);
    let fixed = Work::Fixed {
        descended: 8,
        ticks: vec![40, 20, 20, 20, 20],
    };
    let run = |work: &Work| {
        routing_reads(&graph, &params(9), Some(&starts), &family, &instance, work).unwrap()
    };
    let (a, b) = (run(&fixed), run(&fixed));
    assert_eq!(a.reads, b.reads, "fixed work is reproducible");

    let before = Instant::now();
    let timed = run(&Work::Budget);
    assert!(before.elapsed().as_secs_f64() < GADGET_BUDGET_SECONDS + 1.0);
    assert_eq!(timed.report.budget_seconds, GADGET_BUDGET_SECONDS);
    assert!(timed.report.elapsed_seconds > 0.0);
    assert_eq!(timed.report.work, Work::Budget);
    for start in &timed.report.starts {
        assert!(start.ticks >= 1, "a repair reads the clock at least once");
    }
}

#[test]
fn a_routing_job_that_does_not_fit_is_declined_with_the_reason() {
    let (instance, family, graph) = routing_job();
    let short = IsingGraph::new(graph.h[1..].to_vec(), graph.j.clone(), graph.edges.to_vec());
    let starts = random_states(short.h.len(), 2, 1);
    let declined = routing_reads(
        &short,
        &params(1),
        Some(&starts),
        &family,
        &instance,
        &Work::Budget,
    )
    .expect_err("a node count mismatch");
    assert!(matches!(declined, SampleError::Declined(ref why) if why.contains("does not fit")));
    let bad_state = vec![vec![1, 0, 1]];
    let declined = routing_reads(
        &graph,
        &params(1),
        Some(&bad_state),
        &family,
        &instance,
        &Work::Budget,
    )
    .expect_err("a state that does not fit");
    assert!(matches!(declined, SampleError::Declined(_)));
}

#[test]
fn a_plain_job_runs_the_greedy_descent() {
    let n = 8;
    let graph = IsingGraph::new(
        vec![0.0; n],
        vec![-1.0; n],
        (0..n).map(|i| (i, (i + 1) % n)).collect(),
    );
    let reads = GadgetRepairSampler::new()
        .sample(&graph, &params(5))
        .unwrap();
    assert_eq!(reads.len(), 8, "one descended read per random state");
}

// ---------------------------------------------------------------- saved research runs

/// Replay the saved research runs of the routing cell: the structure counts,
/// the per-start statistics, and the final state.
///
/// The research stopped each repair by the clock, and saved no clock reads.
/// Each start is replayed with the smallest count of clock reads that gives
/// the saved energies and statistics, and the saved final state must come out
/// of the schedule that replays those counts. Run the cases from
/// `research/gadget_repair_parity/export.py research`.
#[test]
#[ignore = "needs QUIP_GADGET_PARITY from research/gadget_repair_parity/export.py"]
fn research_parity() {
    let dir = std::env::var("QUIP_GADGET_PARITY").expect("QUIP_GADGET_PARITY");
    let mut files: Vec<_> = std::fs::read_dir(Path::new(&dir))
        .expect("the parity directory")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no cases in {dir}");
    for file in files {
        let case: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        let at = case["nonce"].as_str().unwrap()[..12].to_owned();
        let ising = ising(&case);
        let nat = nat_greedy(&ising);
        let arcs = arcs(&case);
        let structure = Structure::extract(ising.num_nodes(), &arcs).expect("a structure");
        let kinds = &case["structure"];
        assert_eq!(
            structure.units.len() as u64,
            kinds["units"].as_u64().unwrap(),
            "{at}"
        );
        assert_eq!(
            structure.n_starts as u64,
            kinds["starts"].as_u64().unwrap(),
            "{at}"
        );
        assert_eq!(
            structure.gadgets.len() as u64,
            kinds["gadgets"].as_u64().unwrap(),
            "{at}"
        );
        Repair::check_couplers(&ising, &arcs).expect("the couplers");
        let repair = Repair::new(ising.clone(), &structure).unwrap();
        let reads: Vec<Vec<i8>> = case["reads"]
            .as_array()
            .unwrap()
            .iter()
            .map(spins)
            .collect();
        assert_eq!(
            reads.iter().map(|r| ising.energy(r)).collect::<Vec<_>>(),
            ints(&case["raw_energies"]),
            "{at}: archived energies"
        );
        let descended = usize::try_from(case["reads_descended"].as_u64().unwrap()).unwrap();
        let seed = case["seed"].as_u64().unwrap();
        let probe = Work::Fixed {
            descended,
            ticks: Vec::new(),
        };
        let prepared = prepare_starts(&nat, &reads, &probe, Instant::now(), None).unwrap();
        let recorded = case["read_stats"].as_array().unwrap();
        assert_eq!(prepared.starts.len(), recorded.len(), "{at}: starts");
        let mut ticks = Vec::new();
        for (k, (start, want)) in prepared.starts.iter().zip(recorded).enumerate() {
            let want = ints(want);
            assert_eq!(start.energy, want[1], "{at} start {k}: start energy");
            let found = (0..20_000_u64).find(|&limit| {
                let mut s = start.state.clone();
                let (d, stats, _) = repair
                    .run(
                        &mut s,
                        &RepairOptions::default(),
                        seed + 1000 * k as u64,
                        ClockSpec::Ticks(limit, None),
                    )
                    .unwrap();
                start.energy + d == want[2]
                    && [
                        stats.iterations,
                        stats.chains_kept,
                        stats.sweeps,
                        stats.steps,
                        stats.new_bests,
                    ] == want[3..8]
            });
            ticks.push(found.unwrap_or_else(|| unreachable!("{at} start {k}: no clock count")));
        }
        let work = Work::Fixed { descended, ticks };
        let out = schedule_gadget(&nat, &repair, &reads, seed, &work, None).unwrap();
        assert_eq!(
            out.report.energy,
            case["energy"].as_i64().unwrap(),
            "{at}: energy"
        );
        assert_eq!(out.state, spins(&case["state"]), "{at}: state");
    }
}
