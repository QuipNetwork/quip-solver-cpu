//! The tile worm postprocess (`ice-tiles-charge/1` and `ice-tiles-field/1`)
//! against fixed-work vectors from the research source, and against the saved
//! research states when the opt-in parity data is present.
//!
//! `tests/fixtures/tile_worm_vectors.json` comes from
//! `research/tileworm_parity/export.py vectors`. The research code ran on
//! three small synthetic models: the tile tables, the seed sets, the schedule
//! ends, and `method.run` for ten option strings. The thread clock of
//! `icetiles.c` returned the index of each 64-worm block, so each run did a
//! known number of blocks, which `Work::Fixed` replays.
#![cfg(feature = "experimental")]

use std::path::Path;

use quip_solver_core::CancelToken;
use quip_solver_cpu::postprocess::greedy::{
    natural_moves, DescentError, Greedy, Ising, MoveSources,
};
use quip_solver_cpu::postprocess::tile_worm::{
    schedule_tiles, Method, SpecError, TileError, TileSpec, TileWorm, Work, CHARGE_POLICY,
    CHARGE_SPEC, FIELD_POLICY, FIELD_SPEC, TILE_BUDGET_SECONDS,
};
use serde_json::Value;

const VECTORS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/tile_worm_vectors.json"
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

fn edges(v: &Value) -> Vec<(u32, u32)> {
    v.as_array()
        .expect("pairs")
        .iter()
        .map(|p| (node(&p[0]), node(&p[1])))
        .collect()
}

fn ising(case: &Value) -> Ising {
    Ising::new(&ints(&case["h"]), &edges(&case["edges"]), &ints(&case["j"])).expect("a valid model")
}

/// The research natural moves of a tile model: the tetrahedra and the rigid
/// groups.
fn greedy(case: &Value, ising: Ising) -> Greedy {
    let tiles = quads(&case["tetrahedra"]);
    let sources = MoveSources {
        tiles: &tiles,
        pairs: &[],
        listed: &[],
    };
    let (moves, _) = natural_moves(&ising, &sources).expect("natural moves");
    Greedy::new(ising, &moves, true).expect("a move list")
}

fn vectors() -> Value {
    serde_json::from_str(&std::fs::read_to_string(VECTORS).expect("the vector file"))
        .expect("vector JSON")
}

fn reads(case: &Value) -> Vec<Vec<i8>> {
    case["reads"]
        .as_array()
        .expect("reads")
        .iter()
        .map(spins)
        .collect()
}

fn spec(text: &str) -> TileSpec {
    TileSpec::parse(text).expect("a valid option string")
}

#[test]
fn the_selected_recipes_resolve_to_their_pinned_options() {
    assert_eq!(CHARGE_SPEC, "anneal:sa=1,b0=2,len=64,fe=16,nat48=1");
    assert_eq!(FIELD_SPEC, "anneal:b0=2,nat48=1");
    // The research defaults, with each pinned key changed.
    let defaults = spec("anneal");
    assert_eq!(
        defaults,
        TileSpec {
            method: Method::Anneal,
            b0: 1.0,
            b1: 1.0,
            frac: 0.8,
            tiles: false,
            pairs: false,
            tile_weight: 1,
            cross: 0.0,
            guide: 1.0,
            reads: 1,
            free_every: 16,
            len: 4096,
            pair_walk: false,
            sweep_all: false,
            nat48: false,
        }
    );
    assert_eq!(
        spec(CHARGE_SPEC),
        TileSpec {
            sweep_all: true,
            b0: 2.0,
            len: 64,
            free_every: 16,
            nat48: true,
            ..defaults
        }
    );
    assert_eq!(
        spec(FIELD_SPEC),
        TileSpec {
            b0: 2.0,
            nat48: true,
            ..defaults
        }
    );
    for policy in [CHARGE_POLICY, FIELD_POLICY] {
        assert_eq!(policy.budget_seconds, TILE_BUDGET_SECONDS);
        assert_eq!(TILE_BUDGET_SECONDS, 0.25);
    }
    assert_eq!(CHARGE_POLICY.spec, CHARGE_SPEC);
    assert_eq!(FIELD_POLICY.spec, FIELD_SPEC);
    assert_ne!(CHARGE_POLICY.id, FIELD_POLICY.id);
}

#[test]
fn the_option_parser_follows_the_research_grammar() {
    assert_eq!(spec("desc").method, Method::Descent);
    let all = spec(
        "anneal:b0=4,b1=2,frac=0.5,tiles=1,pairs=1,tw=3,cross=0.2,guide=0.5,reads=3,fe=0,len=32,pw=1,sa=1,nat48=1",
    );
    assert_eq!((all.b0, all.b1, all.frac), (4.0, 2.0, 0.5));
    assert!(all.tiles && all.pairs && all.pair_walk && all.sweep_all && all.nat48);
    assert_eq!(
        (all.tile_weight, all.reads, all.free_every, all.len),
        (3, 3, 0, 32)
    );
    assert_eq!((all.cross, all.guide), (0.2, 0.5));
    // An integer option truncates, as `int` does.
    assert_eq!(spec("anneal:len=64.9").len, 64);
    assert_eq!(
        TileSpec::parse("climb"),
        Err(SpecError::Method("climb".to_owned()))
    );
    assert_eq!(
        TileSpec::parse("anneal:speed=2"),
        Err(SpecError::Key("speed".to_owned()))
    );
    for bad in [
        "anneal:b0=fast",
        "anneal:b0=",
        "anneal:b0=-1",
        "anneal:b0=inf",
        "anneal:len=7",
        "anneal:reads=1e12",
    ] {
        assert!(
            matches!(TileSpec::parse(bad), Err(SpecError::Value { .. })),
            "{bad}"
        );
    }
}

#[test]
fn tile_tables_and_seeds_match_the_research_preparation() {
    let doc = vectors();
    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let model = ising(case);
        let tiles = quads(&case["tetrahedra"]);
        let worm = TileWorm::new(&model, &tiles, &spec("anneal")).expect("tile tables");
        let tt = worm.tables();
        let c = tt.counts();
        let want = &case["counts"];
        for (got, key) in [
            (c.tiles, "tiles"),
            (c.rigid_tiles, "rigid_tiles"),
            (c.shared_tile_edges, "shared_tile_edges"),
            (c.tiled_spins, "tiled_spins"),
            (c.free_spins, "free_spins"),
            (c.strong_pairs, "strong_pairs"),
            (c.partnered, "partnered"),
            (c.max_tiles_per_spin, "max_tiles_per_spin"),
        ] {
            assert_eq!(got as u64, want[key].as_u64().unwrap(), "{name} {key}");
        }
        let etab: Vec<Vec<i64>> = case["etab"].as_array().unwrap().iter().map(ints).collect();
        assert_eq!(etab.len(), tt.tables().len(), "{name}");
        for (t, (want, got)) in etab.iter().zip(tt.tables()).enumerate() {
            assert_eq!(want.as_slice(), got.as_slice(), "{name} tile {t}");
        }
        let nodes = |v: &Value| -> Vec<u32> { v.as_array().unwrap().iter().map(node).collect() };
        assert_eq!(nodes(&case["tiled"]), tt.tiled(), "{name}");
        assert_eq!(nodes(&case["free"]), tt.free(), "{name}");
        assert_eq!(edges(&case["strong_pairs"]), tt.strong_pairs(), "{name}");
        let partner: Vec<i32> = ints(&case["partner"])
            .into_iter()
            .map(|p| i32::try_from(p).unwrap())
            .collect();
        assert_eq!(partner, tt.partners(), "{name}");
        for run in case["runs"].as_array().unwrap() {
            let at = format!("{name} {}", run["spec"]);
            let worm = TileWorm::new(&model, &tiles, &spec(run["spec"].as_str().unwrap()))
                .expect("a worm");
            let seeds = worm.tables().seeds(
                worm.spec().tiles,
                worm.spec().pairs,
                worm.spec().tile_weight,
            );
            assert_eq!(seeds.len() as u64, run["seeds"].as_u64().unwrap(), "{at}");
            let (lo, hi) = worm.betas();
            assert_eq!(
                lo.to_bits(),
                run["b_lo_bits"].as_u64().unwrap(),
                "{at}: b_lo"
            );
            assert_eq!(
                hi.to_bits(),
                run["b_hi_bits"].as_u64().unwrap(),
                "{at}: b_hi"
            );
        }
    }
}

#[test]
fn fixed_work_runs_match_the_research_runs_and_keep_the_read_counts() {
    let doc = vectors();
    let mut runs = 0;
    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let greedy = greedy(case, ising(case));
        let tiles = quads(&case["tetrahedra"]);
        let reads = reads(case);
        for run in case["runs"].as_array().unwrap() {
            let at = format!("{name} {}", run["spec"]);
            let worm = TileWorm::new(greedy.ising(), &tiles, &spec(run["spec"].as_str().unwrap()))
                .expect("a worm");
            let work = Work::Fixed {
                run_blocks: run["run_blocks"].as_u64().unwrap(),
                tail_blocks: run["tail_blocks"].as_u64().unwrap(),
            };
            let out = schedule_tiles(
                &greedy,
                &worm,
                &reads,
                run["seed"].as_u64().unwrap(),
                work,
                None,
            )
            .expect("a schedule");
            let r = &out.report;
            let stats = &run["stats"];
            assert_eq!(out.state, spins(&run["state"]), "{at}: state");
            assert_eq!(r.energy, run["energy"].as_i64().unwrap(), "{at}: energy");
            assert_eq!(
                r.reads_used as u64,
                run["reads_used"].as_u64().unwrap(),
                "{at}: reads used"
            );
            assert_eq!(r.visited.len(), r.reads_used, "{at}");
            assert_eq!(r.work, work, "{at}");
            for (got, key) in [
                (r.stats.worms, "worms"),
                (r.stats.accepted, "accepted"),
                (r.stats.improved, "improved"),
                (r.stats.failed, "failed"),
                (r.stats.crossed, "crossed"),
                (r.stats.free_flips, "free_flips"),
            ] {
                assert_eq!(got, stats[key].as_u64().unwrap(), "{at}: {key}");
            }
            assert_eq!(r.tail_improved, stats["tail_improved"].as_u64(), "{at}");
            assert_eq!(r.polish, stats["polish"].as_i64().unwrap(), "{at}: polish");
            assert_eq!(
                out.pool.len(),
                if worm.spec().nat48 {
                    reads.len()
                } else {
                    r.reads_used
                },
                "{at}: pool"
            );
            runs += 1;
        }
    }
    assert_eq!(runs, 3 * 11);
}

#[test]
fn every_supplied_read_comes_back_descended_with_nat48() {
    let doc = vectors();
    let case = &doc["cases"][0];
    let greedy = greedy(case, ising(case));
    let tiles = quads(&case["tetrahedra"]);
    let reads = reads(case);
    let worm = TileWorm::new(greedy.ising(), &tiles, &spec(CHARGE_SPEC)).unwrap();
    let work = Work::Fixed {
        run_blocks: 8,
        tail_blocks: 4,
    };
    let out = schedule_tiles(&greedy, &worm, &reads, 3, work, None).unwrap();
    assert_eq!(out.pool.len(), reads.len(), "one descended state per read");
    for (read, state) in reads.iter().zip(&out.pool) {
        let mut want = read.clone();
        greedy.descend(&mut want, None).unwrap();
        assert_eq!(*state, want, "the pool holds the natural-move descent");
    }
    let best = out
        .pool
        .iter()
        .map(|s| greedy.ising().energy(s))
        .min()
        .unwrap();
    assert!(
        out.report.energy <= best,
        "the worm never loses the descent"
    );
    assert_eq!(out.report.start_energy, best);
}

#[test]
fn the_report_keeps_the_declared_budget_apart_from_the_measured_time() {
    let doc = vectors();
    let case = &doc["cases"][0];
    let greedy = greedy(case, ising(case));
    let tiles = quads(&case["tetrahedra"]);
    let reads = reads(case);
    let worm = TileWorm::new(greedy.ising(), &tiles, &spec(FIELD_SPEC)).unwrap();

    let fixed = Work::Fixed {
        run_blocks: 2,
        tail_blocks: 1,
    };
    let out = schedule_tiles(&greedy, &worm, &reads, 5, fixed, None).unwrap();
    assert_eq!(
        out.report.work, fixed,
        "fixed work is reported as fixed work"
    );
    assert_eq!(out.report.stats.worms, 128, "two blocks of 64 worms");

    let seconds = TILE_BUDGET_SECONDS;
    let timed = schedule_tiles(&greedy, &worm, &reads, 5, Work::Budget { seconds }, None).unwrap();
    let r = &timed.report;
    assert_eq!(r.work, Work::Budget { seconds });
    assert_eq!(r.reads_used, 1, "the pinned recipes visit one read");
    assert!(r.stats.worms >= 64, "the anneal ran inside the budget");
    assert!(
        r.elapsed_seconds >= 0.5 * seconds && r.elapsed_seconds < seconds + 0.5,
        "measured {} s against a declared {seconds} s",
        r.elapsed_seconds
    );
    assert_eq!(r.energy, greedy.ising().energy(&timed.state));
}

#[test]
fn invalid_tile_metadata_is_rejected_before_processing() {
    let doc = vectors();
    let case = &doc["cases"][0];
    let model = ising(case);
    let n = u32::try_from(model.num_nodes()).unwrap();
    let tiles = quads(&case["tetrahedra"]);
    let anneal = spec(FIELD_SPEC);
    let outside = [[0, 1, 2, n]];
    assert_eq!(
        TileWorm::new(&model, &outside, &anneal).err(),
        Some(TileError::Node { node: n })
    );
    let repeat = [tiles[0], [1, 2, 2, 3]];
    assert_eq!(
        TileWorm::new(&model, &repeat, &anneal).err(),
        Some(TileError::Repeat { tile: 1 })
    );
    assert_eq!(
        TileWorm::new(&model, &[], &anneal).err(),
        Some(TileError::NoTiles)
    );
    let silent = Ising::new(&[0, 0], &[(0, 1)], &[0]).unwrap();
    assert_eq!(
        TileWorm::new(&silent, &[[0, 1, 0, 1]], &anneal).err(),
        Some(TileError::Repeat { tile: 0 })
    );
    let flat = Ising::new(&[0; 4], &[(0, 1)], &[0]).unwrap();
    assert_eq!(
        TileWorm::new(&flat, &[[0, 1, 2, 3]], &anneal).err(),
        Some(TileError::NoScale)
    );
}

#[test]
fn reads_that_do_not_fit_the_model_are_rejected() {
    let doc = vectors();
    let case = &doc["cases"][0];
    let greedy = greedy(case, ising(case));
    let tiles = quads(&case["tetrahedra"]);
    let worm = TileWorm::new(greedy.ising(), &tiles, &spec(FIELD_SPEC)).unwrap();
    let work = Work::Fixed {
        run_blocks: 1,
        tail_blocks: 0,
    };
    let good = reads(case);
    let n = good[0].len();
    assert_eq!(
        schedule_tiles(&greedy, &worm, &[], 1, work, None).err(),
        Some(DescentError::Empty)
    );
    assert_eq!(
        schedule_tiles(&greedy, &worm, &[vec![1; n - 1]], 1, work, None).err(),
        Some(DescentError::Length)
    );
    let mut bad = good[0].clone();
    bad[3] = 0;
    assert_eq!(
        schedule_tiles(&greedy, &worm, &[bad], 1, work, None).err(),
        Some(DescentError::Value { node: 3 })
    );
}

#[test]
fn a_cancelled_schedule_stops_with_an_error() {
    let doc = vectors();
    let case = &doc["cases"][0];
    let greedy = greedy(case, ising(case));
    let tiles = quads(&case["tetrahedra"]);
    let worm = TileWorm::new(greedy.ising(), &tiles, &spec(CHARGE_SPEC)).unwrap();
    let token = CancelToken::default();
    token.cancel_through(1);
    let work = Work::Fixed {
        run_blocks: 8,
        tail_blocks: 4,
    };
    assert_eq!(
        schedule_tiles(
            &greedy,
            &worm,
            &reads(case),
            1,
            work,
            Some((&token, Some(1)))
        )
        .err(),
        Some(DescentError::Cancelled)
    );
}

/// Replays the saved research runs of the selected cells with their recorded
/// work. Set `QUIP_TILEWORM_PARITY` to the output of
/// `research/tileworm_parity/export.py research`.
///
/// The research clock set the inverse temperature at every 64th worm from the
/// elapsed fraction of the budget, and the saved runs do not record those
/// readings. The replay spaces the blocks evenly. The saved `worms` give the
/// blocks of the anneal exactly, and a tail that improved the state cannot be
/// replayed. A replay therefore matches a saved state only while the draws
/// stay in step, and the test asserts the parts that do not depend on the
/// clock: the tile counts, the pinned options, the work count, the read
/// count, and the rescored energy of the returned state. It writes how many
/// states match bit for bit to `replay.txt` in the parity directory.
#[test]
#[ignore = "needs QUIP_TILEWORM_PARITY from research/tileworm_parity/export.py"]
fn research_parity() {
    let dir = std::env::var("QUIP_TILEWORM_PARITY").expect("QUIP_TILEWORM_PARITY");
    let mut files: Vec<_> = std::fs::read_dir(Path::new(&dir))
        .expect("the parity directory")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no cases in {dir}");
    let (mut same_state, mut same_energy, mut total) = (0, 0, 0);
    let mut report = String::new();
    for file in files {
        let case: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        let at = format!(
            "{} {}",
            case["cell"],
            &case["nonce"].as_str().unwrap()[..12]
        );
        let method = case["method"].as_str().unwrap();
        let pinned = if case["cell"] == "ice-charge" {
            CHARGE_SPEC
        } else {
            FIELD_SPEC
        };
        assert_eq!(method, pinned, "{at}: the cell runs its pinned recipe");
        assert_eq!(
            case["budget_s"].as_f64().unwrap(),
            TILE_BUDGET_SECONDS,
            "{at}"
        );
        let greedy = greedy(&case, ising(&case));
        let tiles = quads(&case["tetrahedra"]);
        let worm = TileWorm::new(greedy.ising(), &tiles, &spec(method)).expect("a worm");
        let c = worm.tables().counts();
        for (got, key) in [
            (c.tiles, "tiles"),
            (c.tiled_spins, "tiled_spins"),
            (c.free_spins, "free_spins"),
            (c.strong_pairs, "strong_pairs"),
        ] {
            assert_eq!(
                got as u64,
                case["tiles"][key].as_u64().unwrap(),
                "{at} {key}"
            );
        }
        let worms = case["worms"].as_u64().unwrap();
        assert_eq!(worms % 64, 0, "{at}: the anneal stops at a block");
        let work = Work::Fixed {
            run_blocks: worms / 64,
            tail_blocks: 0,
        };
        let out = schedule_tiles(
            &greedy,
            &worm,
            &reads(&case),
            case["seed"].as_u64().unwrap(),
            work,
            None,
        )
        .expect("a schedule");
        assert_eq!(
            out.report.reads_used as u64,
            case["reads_used"].as_u64().unwrap(),
            "{at}: reads used"
        );
        assert_eq!(out.report.stats.worms, worms, "{at}: worms");
        assert_eq!(
            out.report.energy,
            greedy.ising().energy(&out.state),
            "{at}: rescored energy"
        );
        let saved = spins(&case["state"]);
        assert_eq!(
            greedy.ising().energy(&saved),
            case["energy"].as_i64().unwrap(),
            "{at}: the saved state rescores to the saved energy"
        );
        // The replay is a statistical match: its energy stays within a tenth
        // of a percent of the saved one, on either side.
        let saved_energy = case["energy"].as_i64().unwrap();
        assert!(
            (out.report.energy - saved_energy).abs() * 1000 <= saved_energy.abs(),
            "{at}: replay {} against saved {saved_energy}",
            out.report.energy
        );
        // The same job against the declared budget, as the binary runs it.
        let timed = schedule_tiles(
            &greedy,
            &worm,
            &reads(&case),
            case["seed"].as_u64().unwrap(),
            Work::Budget {
                seconds: TILE_BUDGET_SECONDS,
            },
            None,
        )
        .expect("a timed schedule");
        let t = &timed.report;
        assert!(
            t.elapsed_seconds < TILE_BUDGET_SECONDS + 0.1,
            "{at}: measured {} s",
            t.elapsed_seconds
        );
        assert!(t.energy <= t.start_energy, "{at}");
        assert!(
            (t.energy - saved_energy).abs() * 1000 <= saved_energy.abs(),
            "{at}: timed {} against saved {saved_energy}",
            t.energy
        );
        report.push_str(&format!(
            "{at}: timed {} in {:.3} s with {} worms, start {}\n",
            t.energy, t.elapsed_seconds, t.stats.worms, t.start_energy
        ));
        total += 1;
        same_state += i32::from(out.state == saved);
        same_energy += i32::from(out.report.energy == saved_energy);
        report.push_str(&format!(
            "{at}: replay {} saved {saved_energy} state match {}\n",
            out.report.energy,
            out.state == saved
        ));
    }
    report.push_str(&format!(
        "{same_state} of {total} states and {same_energy} of {total} energies match\n"
    ));
    std::fs::write(Path::new(&dir).join("replay.txt"), report).expect("the replay report");
}
