//! The strong-layer SAT encoding of synthetic ice and the Kissat process
//! boundary. Every test runs without an external solver: a tiny shell script
//! stands in for Kissat.
#![cfg(feature = "experimental")]

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Duration;

use quip_solver_cpu::families::ice_sat::{
    decode, encode, parse_output, satisfies, strong_layer_holds, Kissat, RunError, Strong, Verdict,
};

/// Every spin state of `n` nodes, `true` for +1.
fn states(n: usize) -> impl Iterator<Item = Vec<i8>> {
    (0..1_u32 << n).map(move |k| {
        (0..n)
            .map(|b| if k >> b & 1 == 1 { 1 } else { -1 })
            .collect()
    })
}

/// The four-literal truth values of `spins` on the encoding's variables.
fn assignment(nodes: &[u32], spins: &[i8]) -> Vec<bool> {
    nodes.iter().map(|&v| spins[v as usize] == 1).collect()
}

#[test]
fn a_tile_clause_set_admits_exactly_the_balanced_gauged_states() {
    for gauge in states(4) {
        let gauge: [i8; 4] = gauge.try_into().unwrap();
        let strong = Strong {
            tiles: &[[0, 1, 2, 3]],
            gauges: &[gauge],
            pairs: &[],
        };
        let encoding = encode(&strong);
        assert_eq!(encoding.nodes, vec![0, 1, 2, 3]);
        assert_eq!(
            encoding.clauses.len(),
            10,
            "one clause per unbalanced count"
        );
        for spins in states(4) {
            let balanced = (0..4).map(|p| i32::from(gauge[p] * spins[p])).sum::<i32>() == 0;
            assert_eq!(
                satisfies(&encoding, &assignment(&encoding.nodes, &spins)),
                balanced,
                "gauge {gauge:?} spins {spins:?}"
            );
            assert_eq!(strong_layer_holds(&strong, &spins), balanced);
        }
    }
}

#[test]
fn a_pair_clause_set_admits_exactly_the_aligned_states() {
    for (sign, aligned) in [(-1, true), (1, false)] {
        let strong = Strong {
            tiles: &[],
            gauges: &[],
            pairs: &[(2, 5, sign)],
        };
        let encoding = encode(&strong);
        assert_eq!(encoding.nodes, vec![2, 5]);
        for spins in states(6) {
            let holds = (spins[2] == spins[5]) == aligned;
            assert_eq!(
                satisfies(&encoding, &assignment(&encoding.nodes, &spins)),
                holds
            );
            assert_eq!(strong_layer_holds(&strong, &spins), holds);
        }
    }
}

#[test]
fn two_tiles_and_a_pair_agree_with_the_direct_check_on_every_state() {
    let strong = Strong {
        tiles: &[[0, 1, 2, 3], [3, 4, 5, 6]],
        gauges: &[[1, -1, 1, 1], [-1, 1, 1, -1]],
        pairs: &[(1, 7, -1)],
    };
    let encoding = encode(&strong);
    assert_eq!(encoding.nodes, (0..8).collect::<Vec<u32>>());
    assert_eq!(encoding.clauses.len(), 22);
    let mut satisfied = 0;
    for spins in states(8) {
        let holds = strong_layer_holds(&strong, &spins);
        assert_eq!(
            satisfies(&encoding, &assignment(&encoding.nodes, &spins)),
            holds
        );
        satisfied += usize::from(holds);
    }
    assert!(satisfied > 0, "the fixture has a satisfying state");
}

#[test]
fn a_decoded_witness_sets_unconstrained_spins_to_plus_one() {
    let strong = Strong {
        tiles: &[],
        gauges: &[],
        pairs: &[(1, 3, -1)],
    };
    let encoding = encode(&strong);
    assert_eq!(decode(&encoding, &[false, false], 5), vec![1, -1, 1, -1, 1]);
}

#[test]
fn solver_output_parses_into_a_verdict() {
    let sat = "c comment\ns SATISFIABLE\nv 1 -2\nv 3 0\n";
    assert_eq!(
        parse_output(sat, 3),
        Ok(Verdict::Sat(vec![true, false, true]))
    );
    assert_eq!(parse_output("s UNSATISFIABLE\n", 3), Ok(Verdict::Unsat));
    assert_eq!(parse_output("s UNKNOWN\n", 3), Ok(Verdict::Unknown));
    for bad in [
        "",
        "v 1 0\n",
        "s SATISFIABLE\nv 1 -2 0\n",
        "s SATISFIABLE\nv 1 -2 4 0\n",
        "s SATISFIABLE\nv 1 2 -2 3 0\n",
        "s SATISFIABLE\nv 1 x 3 0\n",
        "s SATISFIABLE\nv 1 -2 3\n",
    ] {
        assert!(parse_output(bad, 3).is_err(), "{bad:?}");
    }
}

/// The fake solvers, as shell scripts that ignore their arguments.
const FAKES: &[(&str, &str)] = &[
    ("sat", "echo 's SATISFIABLE'; echo 'v -1 -2 0'; exit 10"),
    (
        "err",
        "echo 'kissat: error: invalid long option' >&2; exit 1",
    ),
    ("unsat", "echo 's UNSATISFIABLE'; exit 20"),
    ("bad", "echo 's SATISFIABLE'; echo 'v 1 0'; exit 10"),
    ("slow", "exec sleep 30"),
    // A wrapper whose descendant keeps the output pipe open after it exits.
    (
        "orphan",
        "sleep 30 & echo 's SATISFIABLE'; echo 'v -1 -2 0'; exit 10",
    ),
];

/// The directory of the fake solvers, written once before any test spawns a
/// process. A script written while another test thread forks stays open in
/// the child for a moment, and executing it then fails with `ETXTBSY`.
static FAKE_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    let dir = std::env::temp_dir().join(format!("ice-sat-fakes-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, body) in FAKES {
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    dir
});

/// The fake solver `name`. Every test that spawns a process calls this first.
fn fake(name: &str) -> PathBuf {
    FAKE_DIR.join(name)
}

fn one_pair() -> (
    Strong<'static>,
    quip_solver_cpu::families::ice_sat::Encoding,
) {
    let strong = Strong {
        tiles: &[],
        gauges: &[],
        pairs: &[(0, 1, -1)],
    };
    let encoding = encode(&strong);
    (strong, encoding)
}

#[test]
fn a_satisfiable_answer_returns_the_assignment() {
    let solver = fake("sat");
    let (_, encoding) = one_pair();
    let run = Kissat::new(solver, Duration::from_secs(5))
        .run(&encoding, 7, None)
        .unwrap();
    assert_eq!(run.verdict, Verdict::Sat(vec![false, false]));
    assert_eq!(run.exit_code, Some(10));
    assert!(run.args.iter().any(|a| a == "--seed=7"), "{:?}", run.args);
}

#[test]
fn a_seed_beyond_the_solver_range_is_folded_into_31_bits() {
    let solver = fake("sat");
    let (_, encoding) = one_pair();
    let run = Kissat::new(solver, Duration::from_secs(5))
        .run(&encoding, u64::MAX, None)
        .unwrap();
    assert!(
        run.args.iter().any(|a| a == "--seed=2147483647"),
        "{:?}",
        run.args
    );
    let run = Kissat::new(fake("sat"), Duration::from_secs(5))
        .run(&encoding, (1 << 31) + 5, None)
        .unwrap();
    assert!(run.args.iter().any(|a| a == "--seed=5"), "{:?}", run.args);
}

#[test]
fn a_solver_error_names_its_exit_code_and_stderr() {
    let solver = fake("err");
    let (_, encoding) = one_pair();
    let err = Kissat::new(solver, Duration::from_secs(5))
        .run(&encoding, 0, None)
        .unwrap_err();
    let text = err.to_string();
    assert!(text.contains("exit code 1"), "{text}");
    assert!(text.contains("invalid long option"), "{text}");
}

#[test]
fn an_unsatisfiable_answer_is_a_verdict_not_an_error() {
    let solver = fake("unsat");
    let (_, encoding) = one_pair();
    let run = Kissat::new(solver, Duration::from_secs(5))
        .run(&encoding, 0, None)
        .unwrap();
    assert_eq!(run.verdict, Verdict::Unsat);
}

#[test]
fn malformed_output_is_an_error() {
    let solver = fake("bad");
    let (_, encoding) = one_pair();
    let err = Kissat::new(solver, Duration::from_secs(5))
        .run(&encoding, 0, None)
        .unwrap_err();
    assert!(matches!(err, RunError::Malformed(_)), "{err:?}");
}

#[test]
fn a_solver_that_overruns_its_timeout_is_killed() {
    let solver = fake("slow");
    let (_, encoding) = one_pair();
    let started = std::time::Instant::now();
    let err = Kissat::new(solver, Duration::from_millis(200))
        .run(&encoding, 0, None)
        .unwrap_err();
    assert!(matches!(err, RunError::TimedOut), "{err:?}");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "the child was not killed"
    );
}

#[test]
fn a_descendant_that_holds_the_pipe_does_not_hang_the_run() {
    let (_, encoding) = one_pair();
    let started = std::time::Instant::now();
    let result = Kissat::new(fake("orphan"), Duration::from_secs(1)).run(&encoding, 0, None);
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "the run waited for the orphan: {result:?}"
    );
    assert_eq!(result.unwrap().verdict, Verdict::Sat(vec![false, false]));
}

#[test]
fn a_cancelled_run_is_killed_promptly() {
    let (_, encoding) = one_pair();
    let token = quip_solver_core::CancelToken::default();
    token.cancel_through(5);
    let started = std::time::Instant::now();
    let err = Kissat::new(fake("slow"), Duration::from_secs(60))
        .run(&encoding, 0, Some((&token, Some(3))))
        .unwrap_err();
    assert!(matches!(err, RunError::Cancelled), "{err:?}");
    assert!(started.elapsed() < Duration::from_secs(10));
}

#[test]
fn a_huge_timeout_does_not_overflow() {
    let (_, encoding) = one_pair();
    let run = Kissat::new(fake("sat"), Duration::from_secs(u64::MAX))
        .run(&encoding, 0, None)
        .unwrap();
    assert_eq!(run.verdict, Verdict::Sat(vec![false, false]));
}

#[test]
fn a_missing_executable_is_an_error() {
    let (_, encoding) = one_pair();
    let _ = fake("sat");
    let err = Kissat::new(PathBuf::from("/nonexistent/kissat"), Duration::from_secs(1))
        .run(&encoding, 0, None)
        .unwrap_err();
    assert!(matches!(err, RunError::Spawn(_)), "{err:?}");
}
