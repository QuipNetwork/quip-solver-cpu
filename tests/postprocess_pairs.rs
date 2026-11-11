//! The pair-worm postprocess (`ice-pairs-pw/1`) against fixed-work vectors
//! from the research source, and against the saved research states when the
//! opt-in parity data is present.
//!
//! `tests/fixtures/pair_worm_vectors.json` comes from
//! `research/pairworm_parity/export.py vectors`. The research code ran on two
//! small synthetic models: the tile preparation, `pair_worm_descent`,
//! `crossover`, `defect_counts`, and `schedule_pw` with a fixed-work clock.
#![cfg(feature = "experimental")]

use std::path::Path;

use quip_solver_core::CancelToken;
use quip_solver_cpu::postprocess::greedy::{
    natural_moves, DescentError, Greedy, Ising, MoveSources,
};
use quip_solver_cpu::postprocess::pair_worm::{
    schedule_pw, CrossoverOptions, PairError, PairWorm, Work, WormOptions, PW_BUDGET_SECONDS,
    PW_POLICY,
};
use serde_json::Value;

const VECTORS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/pair_worm_vectors.json"
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

fn quads(v: &Value) -> Vec<[u32; 4]> {
    v.as_array()
        .expect("rows")
        .iter()
        .map(|row| {
            let row = row.as_array().expect("a row");
            [node(&row[0]), node(&row[1]), node(&row[2]), node(&row[3])]
        })
        .collect()
}

fn pairs(v: &Value) -> Vec<(u32, u32)> {
    v.as_array()
        .expect("pairs")
        .iter()
        .map(|p| (node(&p[0]), node(&p[1])))
        .collect()
}

fn ising(case: &Value) -> Ising {
    let edges = pairs(&case["edges"]);
    Ising::new(&ints(&case["h"]), &edges, &ints(&case["j"])).expect("a valid model")
}

/// The research natural moves of a model: its tetrahedra and pairs, the
/// listed node sets, and the rigid groups.
fn greedy(case: &Value, ising: Ising) -> Greedy {
    let tiles = quads(&case["tetrahedra"]);
    let pairs = pairs(&case["pairs"]);
    let listed: Vec<Vec<u32>> = case
        .get("listed")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .map(|r| r.as_array().expect("a row").iter().map(node).collect())
                .collect()
        })
        .unwrap_or_default();
    let sources = MoveSources {
        tiles: &tiles,
        pairs: &pairs,
        listed: &listed,
    };
    let (moves, _) = natural_moves(&ising, &sources).expect("natural moves");
    Greedy::new(ising, &moves, true).expect("a move list")
}

fn pair_worm(case: &Value, ising: &Ising) -> PairWorm {
    PairWorm::new(ising, &quads(&case["tetrahedra"]), &pairs(&case["pairs"]))
        .expect("valid pair metadata")
}

/// `PairWorm.descend` keywords, over the research defaults.
fn worm_options(v: &Value) -> WormOptions {
    let mut o = WormOptions::default();
    for (key, value) in v.as_object().expect("options") {
        match key.as_str() {
            "zero_budget" => o.zero_budget = value.as_i64().unwrap(),
            "max_len" => o.max_len = usize::try_from(value.as_u64().unwrap()).unwrap(),
            "max_sweeps" => o.max_sweeps = u32::try_from(value.as_u64().unwrap()).unwrap(),
            "defect_depth" => o.defect_depth = u32::try_from(value.as_u64().unwrap()).unwrap(),
            "random_depth" => o.random_depth = u32::try_from(value.as_u64().unwrap()).unwrap(),
            "random_pass" => o.random_pass = value.as_bool().unwrap(),
            "prune" => o.prune = value.as_i64().unwrap(),
            "temp" => o.temp = value.as_f64().unwrap(),
            other => unreachable!("the vectors name an unknown descend option {other}"),
        }
    }
    o
}

fn crossover_options(v: &Value) -> CrossoverOptions {
    let mut o = CrossoverOptions::default();
    for (key, value) in v.as_object().expect("options") {
        match key.as_str() {
            "align" => o.align = value.as_bool().unwrap(),
            "min_size" => o.min_size = usize::try_from(value.as_u64().unwrap()).unwrap(),
            "max_size" => o.max_size = usize::try_from(value.as_u64().unwrap()).unwrap(),
            other => unreachable!("the vectors name an unknown crossover option {other}"),
        }
    }
    o
}

fn vectors() -> Value {
    serde_json::from_str(&std::fs::read_to_string(VECTORS).expect("the vector file"))
        .expect("vector JSON")
}

#[test]
fn tiles_gauges_and_memberships_match_the_research_preparation() {
    let doc = vectors();
    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ising = ising(case);
        let pw = pair_worm(case, &ising);
        assert_eq!(pw.tiles(), quads(&case["tiles"]).as_slice(), "{name} tiles");
        let gauge: Vec<[i8; 4]> = case["gauge"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| {
                let g = ints(g);
                [0, 1, 2, 3].map(|k| i8::try_from(g[k]).unwrap())
            })
            .collect();
        assert_eq!(pw.gauges(), gauge.as_slice(), "{name} gauges");
        let mtile: Vec<[i32; 2]> = case["mtile"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                let m = ints(m);
                [i32::try_from(m[0]).unwrap(), i32::try_from(m[1]).unwrap()]
            })
            .collect();
        assert_eq!(pw.memberships(), mtile.as_slice(), "{name} memberships");
        let counts = pw.counts();
        assert_eq!(
            counts.tiles as u64,
            case["counts"]["tiles"].as_u64().unwrap()
        );
        assert_eq!(
            counts.dropped as u64,
            case["counts"]["dropped"].as_u64().unwrap()
        );
        assert_eq!(
            counts.spins_over_two as u64,
            case["counts"]["spins_over_two"].as_u64().unwrap()
        );
        assert_eq!(
            counts.pairs as u64,
            case["counts"]["pairs"].as_u64().unwrap()
        );
        assert!(
            counts.dropped > 0 && counts.spins_over_two > 0,
            "{name} exercises both drops"
        );
    }
}

#[test]
fn defect_counts_match_the_research() {
    let doc = vectors();
    for case in doc["cases"].as_array().unwrap() {
        let ising = ising(case);
        let pw = pair_worm(case, &ising);
        for (read, want) in case["reads"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["defects"].as_array().unwrap())
        {
            let (tiles, pairs) = pw.defects(&spins(read));
            assert_eq!(
                [tiles as i64, pairs as i64],
                [want[0].as_i64().unwrap(), want[1].as_i64().unwrap()]
            );
        }
    }
}

#[test]
fn every_worm_descent_matches_the_research_state_and_accept_count() {
    let doc = vectors();
    let options = &doc["descent_options"];
    let mut checked = 0;
    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ising = ising(case);
        let pw = pair_worm(case, &ising);
        let reads = case["reads"].as_array().unwrap();
        for d in case["descents"].as_array().unwrap() {
            let opt = d["options"].as_str().unwrap();
            let read = usize::try_from(d["read"].as_u64().unwrap()).unwrap();
            let seed = d["seed"].as_u64().unwrap();
            let mut s = spins(&reads[read]);
            let accepted = pw
                .descend(&ising, &mut s, &worm_options(&options[opt]), seed)
                .expect("a fitting state");
            assert_eq!(
                s,
                spins(&d["state"]),
                "{name} {opt} read {read} seed {seed}: state"
            );
            assert_eq!(
                accepted,
                d["accepted"].as_u64().unwrap(),
                "{name} {opt} read {read} seed {seed}: accepted worms"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 2 * 7 * 3);
}

#[test]
fn crossover_matches_the_research_blocks_and_state() {
    let doc = vectors();
    let options = &doc["crossover_options"];
    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ising = ising(case);
        let pw = pair_worm(case, &ising);
        let pool: Vec<Vec<i8>> = case["reads"].as_array().unwrap()[1..]
            .iter()
            .map(spins)
            .collect();
        for x in case["crossovers"].as_array().unwrap() {
            let opt = x["options"].as_str().unwrap();
            let mut s = spins(&case["crossover_base"]);
            let blocks = pw
                .crossover(&ising, &mut s, &pool, &crossover_options(&options[opt]))
                .expect("fitting states");
            assert_eq!(s, spins(&x["state"]), "{name} {opt}: state");
            assert_eq!(
                blocks,
                x["blocks"].as_u64().unwrap(),
                "{name} {opt}: blocks"
            );
        }
    }
}

#[test]
fn the_fixed_work_schedule_matches_the_research_schedule() {
    let doc = vectors();
    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let greedy = greedy(case, ising(case));
        let pw = pair_worm(case, greedy.ising());
        let reads: Vec<Vec<i8>> = case["reads"]
            .as_array()
            .unwrap()
            .iter()
            .map(spins)
            .collect();
        for sch in case["schedules"].as_array().unwrap() {
            let work = Work::Fixed {
                reads: usize::try_from(sch["reads_used"].as_u64().unwrap()).unwrap(),
                anneal_cycles: sch["anneal_cycles"].as_u64().unwrap(),
            };
            let seed = sch["seed"].as_u64().unwrap();
            let out = schedule_pw(&greedy, &pw, &reads, seed, work, None).expect("a schedule");
            let at = format!("{name} {work:?} seed {seed}");
            assert_eq!(out.state, spins(&sch["state"]), "{at}: state");
            let r = &out.report;
            assert_eq!(r.energy, sch["energy"].as_i64().unwrap(), "{at}: energy");
            assert_eq!(
                r.energy,
                greedy.ising().energy(&out.state),
                "{at}: rescored"
            );
            assert_eq!(
                r.crossover_blocks,
                sch["crossover_blocks"].as_u64().unwrap(),
                "{at}"
            );
            assert_eq!(
                r.crossover_rounds,
                sch["crossover_rounds"].as_u64().unwrap(),
                "{at}"
            );
            assert_eq!(
                r.pre_crossover,
                sch["pre_crossover"].as_i64().unwrap(),
                "{at}"
            );
            assert_eq!(r.pre_anneal, sch["pre_anneal"].as_i64().unwrap(), "{at}");
            assert_eq!(
                r.reads_used as u64,
                sch["reads_used"].as_u64().unwrap(),
                "{at}"
            );
            assert_eq!(
                r.anneal_cycles,
                sch["anneal_cycles"].as_u64().unwrap(),
                "{at}"
            );
            assert_eq!(
                out.pool.len(),
                r.reads_used,
                "{at}: one pool state per visited read"
            );
            for state in &out.pool {
                assert!(
                    greedy.ising().energy(state) >= r.pre_crossover,
                    "{at}: pool best"
                );
            }
        }
    }
}

/// One gauged tile on 0..4, two outside nodes, a pair (4, 5), and a
/// positive edge (1, 4).
fn small() -> Ising {
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for a in 0..4_u32 {
        for b in a + 1..4 {
            edges.push((a, b));
            j.push(500);
        }
    }
    edges.extend([(4, 5), (1, 4), (2, 5)]);
    j.extend([-1000, 62, -62]);
    Ising::new(&[0; 6], &edges, &j).unwrap()
}

#[test]
fn invalid_pair_metadata_fails_before_processing() {
    let ising = small();
    let tiles = [[0, 1, 2, 3]];
    assert!(PairWorm::new(&ising, &tiles, &[(4, 5)]).is_ok());
    assert_eq!(
        PairWorm::new(&ising, &tiles, &[(4, 5), (0, 5)]).err(),
        Some(PairError::NotCoupler { index: 1 }),
        "a pair that is not an edge"
    );
    assert_eq!(
        PairWorm::new(&ising, &tiles, &[(1, 4)]).err(),
        Some(PairError::Antiferromagnetic { index: 0 }),
        "a pair on a positive coupler"
    );
    assert_eq!(
        PairWorm::new(&ising, &tiles, &[(4, 5), (5, 2)]).err(),
        Some(PairError::Repeat { node: 5 }),
        "two pairs share a node"
    );
    assert_eq!(
        PairWorm::new(&ising, &tiles, &[(4, 6)]).err(),
        Some(PairError::Node { node: 6 }),
        "a pair node outside the model"
    );
    assert_eq!(
        PairWorm::new(&ising, &[[0, 1, 2, 9]], &[(4, 5)]).err(),
        Some(PairError::Node { node: 9 }),
        "a tile node outside the model"
    );
}

#[test]
fn a_state_that_does_not_fit_is_refused() {
    let ising = small();
    let pw = PairWorm::new(&ising, &[[0, 1, 2, 3]], &[(4, 5)]).unwrap();
    let opts = WormOptions::default();
    assert_eq!(
        pw.descend(&ising, &mut [1; 5], &opts, 1),
        Err(DescentError::Length)
    );
    assert_eq!(
        pw.descend(&ising, &mut [1, 1, 0, 1, 1, 1], &opts, 1),
        Err(DescentError::Value { node: 2 })
    );
    let pool = vec![vec![1_i8; 5]];
    assert_eq!(
        pw.crossover(&ising, &mut [1; 6], &pool, &CrossoverOptions::default()),
        Err(DescentError::Length)
    );
}

#[test]
fn a_strict_descent_never_raises_the_energy() {
    let doc = vectors();
    for case in doc["cases"].as_array().unwrap() {
        let ising = ising(case);
        let pw = pair_worm(case, &ising);
        let opts = WormOptions {
            defect_depth: 8,
            ..WormOptions::default()
        };
        for (k, read) in case["reads"].as_array().unwrap().iter().enumerate() {
            let mut s = spins(read);
            let before = ising.energy(&s);
            pw.descend(&ising, &mut s, &opts, k as u64).unwrap();
            assert!(ising.energy(&s) <= before);
        }
    }
}

#[test]
fn the_report_keeps_the_declared_budget_apart_from_the_measured_time() {
    let doc = vectors();
    let case = &doc["cases"][0];
    let greedy = greedy(case, ising(case));
    let pw = pair_worm(case, greedy.ising());
    let reads: Vec<Vec<i8>> = case["reads"]
        .as_array()
        .unwrap()
        .iter()
        .map(spins)
        .collect();

    let fixed = Work::Fixed {
        reads: 3,
        anneal_cycles: 2,
    };
    let out = schedule_pw(&greedy, &pw, &reads, 5, fixed, None).unwrap();
    assert_eq!(out.report.policy, PW_POLICY);
    assert_eq!(out.report.budget_seconds, PW_BUDGET_SECONDS);
    assert_eq!(
        out.report.work, fixed,
        "fixed work is reported as fixed work"
    );
    assert_eq!((out.report.reads_used, out.report.anneal_cycles), (3, 2));

    let timed = schedule_pw(&greedy, &pw, &reads, 5, Work::Budget, None).unwrap();
    let r = &timed.report;
    assert_eq!(r.work, Work::Budget);
    assert_eq!(r.budget_seconds, PW_BUDGET_SECONDS);
    assert!(r.reads_used >= 1 && r.reads_used <= reads.len());
    assert_eq!(r.read_seconds.len(), r.reads_used);
    assert!(
        r.anneal_cycles > 0,
        "the anneal tail runs inside the budget"
    );
    assert!(
        r.elapsed_seconds > 0.0 && r.elapsed_seconds < PW_BUDGET_SECONDS + 0.5,
        "measured {} s",
        r.elapsed_seconds
    );
    assert_eq!(r.energy, greedy.ising().energy(&timed.state));
}

#[test]
fn a_cancelled_schedule_stops_with_an_error() {
    let doc = vectors();
    let case = &doc["cases"][0];
    let greedy = greedy(case, ising(case));
    let pw = pair_worm(case, greedy.ising());
    let reads: Vec<Vec<i8>> = case["reads"]
        .as_array()
        .unwrap()
        .iter()
        .map(spins)
        .collect();
    let token = CancelToken::default();
    token.cancel_through(1);
    let work = Work::Fixed {
        reads: 6,
        anneal_cycles: 4,
    };
    assert_eq!(
        schedule_pw(&greedy, &pw, &reads, 1, work, Some((&token, Some(1)))).err(),
        Some(DescentError::Cancelled)
    );
}

/// Replays the saved research runs of the selected cells with their recorded
/// work. Set `QUIP_PAIRWORM_PARITY` to the output of
/// `research/pairworm_parity/export.py research`.
#[test]
#[ignore = "needs QUIP_PAIRWORM_PARITY from research/pairworm_parity/export.py"]
fn research_parity() {
    let dir = std::env::var("QUIP_PAIRWORM_PARITY").expect("QUIP_PAIRWORM_PARITY");
    let mut files: Vec<_> = std::fs::read_dir(Path::new(&dir))
        .expect("the parity directory")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no cases in {dir}");
    for file in files {
        let case: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        let at = format!(
            "{} {}",
            case["cell"],
            &case["nonce"].as_str().unwrap()[..12]
        );
        let greedy = greedy(&case, ising(&case));
        let pw = pair_worm(&case, greedy.ising());
        let counts = pw.counts();
        assert_eq!(
            counts.tiles as u64,
            case["tiles"]["tiles"].as_u64().unwrap(),
            "{at}"
        );
        assert_eq!(
            counts.pairs as u64,
            case["tiles"]["pairs"].as_u64().unwrap(),
            "{at}"
        );
        let reads: Vec<Vec<i8>> = case["reads"]
            .as_array()
            .unwrap()
            .iter()
            .map(spins)
            .collect();
        let work = Work::Fixed {
            reads: usize::try_from(case["reads_used"].as_u64().unwrap()).unwrap(),
            anneal_cycles: case["anneal_cycles"].as_u64().unwrap(),
        };
        let out = schedule_pw(
            &greedy,
            &pw,
            &reads,
            case["seed"].as_u64().unwrap(),
            work,
            None,
        )
        .expect("a schedule");
        assert_eq!(
            out.report.pre_crossover,
            case["pre_crossover"].as_i64().unwrap(),
            "{at}"
        );
        assert_eq!(
            out.report.pre_anneal,
            case["pre_anneal"].as_i64().unwrap(),
            "{at}"
        );
        assert_eq!(
            out.report.crossover_blocks,
            case["crossover_blocks"].as_u64().unwrap(),
            "{at}"
        );
        assert_eq!(
            out.report.energy,
            case["energy"].as_i64().unwrap(),
            "{at}: energy"
        );
        assert_eq!(out.state, spins(&case["state"]), "{at}: state");
    }
}
