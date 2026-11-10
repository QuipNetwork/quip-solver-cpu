//! The exact greedy postprocess on a hand-built fixture, checked state by
//! state against energies computed from scratch, and against the research
//! states when the opt-in parity data is present.
#![cfg(feature = "experimental")]

use quip_solver_core::CancelToken;
use quip_solver_cpu::postprocess::greedy::{
    natural_moves, DescentError, Greedy, Ising, ModelError, MoveError, MoveSources,
};

/// A gauged tile on 0..4, a pair (4, 5), a strong bond (3, 4) outside the
/// tile, a weak edge (1, 5), and fields on 0 and 5.
fn fixture() -> (Vec<i64>, Vec<(u32, u32)>, Vec<i64>) {
    let g = [1_i64, -1, 1, 1];
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for a in 0..4_u32 {
        for b in a + 1..4 {
            edges.push((a, b));
            j.push(1000 * g[a as usize] * g[b as usize]);
        }
    }
    edges.extend([(4, 5), (3, 4), (5, 1)]);
    j.extend([-1000, -600, 200]);
    let h = vec![150, 0, 0, 0, 0, -70];
    (h, edges, j)
}

fn ising() -> Ising {
    let (h, edges, j) = fixture();
    Ising::new(&h, &edges, &j).unwrap()
}

const TILES: [[u32; 4]; 1] = [[0, 1, 2, 3]];
const PAIRS: [(u32, u32); 1] = [(4, 5)];

fn listed() -> Vec<Vec<u32>> {
    vec![vec![5, 2]]
}

fn sources(listed: &[Vec<u32>]) -> MoveSources<'_> {
    MoveSources {
        tiles: &TILES,
        pairs: &PAIRS,
        listed,
    }
}

/// Every spin state of `n` nodes.
fn states(n: usize) -> impl Iterator<Item = Vec<i8>> {
    (0..1_u32 << n).map(move |k| {
        (0..n)
            .map(|b| if k >> b & 1 == 1 { 1 } else { -1 })
            .collect()
    })
}

fn direct_energy(s: &[i8]) -> i64 {
    let (h, edges, j) = fixture();
    let fields: i64 = h.iter().zip(s).map(|(&h, &s)| h * i64::from(s)).sum();
    let couplings: i64 = edges
        .iter()
        .zip(&j)
        .map(|(&(u, v), &j)| j * i64::from(s[u as usize]) * i64::from(s[v as usize]))
        .sum();
    fields + couplings
}

fn flipped(s: &[i8], members: &[u32]) -> Vec<i8> {
    let mut t = s.to_vec();
    for &i in members {
        t[i as usize] = -t[i as usize];
    }
    t
}

#[test]
fn natural_moves_follow_the_source_rules() {
    let listed = listed();
    let (moves, counts) = natural_moves(&ising(), &sources(&listed)).unwrap();
    let expected: Vec<Vec<u32>> = vec![
        vec![0, 1],
        vec![0, 1, 2],
        vec![0, 1, 2, 3],
        vec![0, 1, 3],
        vec![0, 2],
        vec![0, 2, 3],
        vec![0, 3],
        vec![1, 2],
        vec![1, 2, 3],
        vec![1, 3],
        vec![1, 5],
        vec![2, 3],
        vec![2, 5],
        vec![3, 4],
        vec![3, 4, 5],
        vec![4, 5],
    ];
    assert_eq!(moves, expected);
    assert_eq!(counts.edge_pairs, 9);
    assert_eq!(counts.tile_subsets, 5);
    assert_eq!(counts.pairs, 0, "the pair is already an edge pair");
    assert_eq!(counts.listed, 1);
    assert_eq!(
        counts.rigid_groups, 1,
        "the strong bond and the pair join 3, 4, 5"
    );
}

#[test]
fn a_move_delta_is_the_energy_difference_of_the_flip() {
    let listed = listed();
    let ising = ising();
    let (moves, _) = natural_moves(&ising, &sources(&listed)).unwrap();
    let greedy = Greedy::new(ising, &moves, true).unwrap();
    assert_eq!(greedy.num_moves(), 6 + moves.len());
    for s in states(6) {
        for m in 0..greedy.num_moves() {
            let members = greedy.members(m);
            assert_eq!(
                greedy.delta(&s, m).unwrap(),
                direct_energy(&flipped(&s, members)) - direct_energy(&s),
                "move {members:?} at {s:?}"
            );
        }
    }
}

/// First improvement with every delta recomputed from scratch: visit the
/// moves in order, apply each move that lowers the energy, and stop after a
/// sweep that applies none.
fn reference_descent(moves: &[Vec<u32>], start: &[i8]) -> (Vec<i8>, u64) {
    let mut s = start.to_vec();
    let mut applied = 0;
    loop {
        let mut changed = false;
        for members in moves {
            let next = flipped(&s, members);
            if direct_energy(&next) < direct_energy(&s) {
                s = next;
                applied += 1;
                changed = true;
            }
        }
        if !changed {
            return (s, applied);
        }
    }
}

#[test]
fn the_descent_matches_a_from_scratch_first_improvement_on_every_start() {
    let listed = listed();
    let ising = ising();
    let (natural, _) = natural_moves(&ising, &sources(&listed)).unwrap();
    for with_singles in [false, true] {
        let mut order: Vec<Vec<u32>> = Vec::new();
        if with_singles {
            order.extend((0..6).map(|i| vec![i]));
        }
        order.extend(natural.iter().cloned());
        let greedy = Greedy::new(ising.clone(), &natural, with_singles).unwrap();
        for start in states(6) {
            let mut s = start.clone();
            let applied = greedy.descend(&mut s, None).unwrap();
            let (expected, expected_applied) = reference_descent(&order, &start);
            assert_eq!(s, expected, "singles {with_singles} from {start:?}");
            assert_eq!(applied, expected_applied);
            assert!(direct_energy(&s) <= direct_energy(&start));
            for m in 0..greedy.num_moves() {
                assert!(
                    greedy.delta(&s, m).unwrap() >= 0,
                    "a move still lowers the energy"
                );
            }
        }
    }
}

#[test]
fn the_energy_matches_the_direct_sum() {
    let ising = ising();
    for s in states(6) {
        assert_eq!(ising.energy(&s), direct_energy(&s));
    }
}

#[test]
fn malformed_models_are_refused() {
    let (h, mut edges, j) = fixture();
    assert_eq!(
        Ising::new(&h, &edges, &j[1..]).unwrap_err(),
        ModelError::Lengths
    );
    edges[8] = (1, 5);
    assert!(Ising::new(&h, &edges, &j).is_ok(), "orientation is free");
    edges[8] = (0, 1);
    assert_eq!(
        Ising::new(&h, &edges, &j).unwrap_err(),
        ModelError::DuplicateEdge { index: 8 }
    );
    edges[8] = (5, 5);
    assert_eq!(
        Ising::new(&h, &edges, &j).unwrap_err(),
        ModelError::SelfLoop { index: 8 }
    );
    edges[8] = (5, 6);
    assert_eq!(
        Ising::new(&h, &edges, &j).unwrap_err(),
        ModelError::Node { index: 8 }
    );
    assert_eq!(
        Ising::new(&[i64::MAX / 4, 0], &[(0, 1)], &[0]).unwrap_err(),
        ModelError::Overflow
    );
}

#[test]
fn malformed_moves_and_states_are_refused() {
    let ising = ising();
    let bad = vec![vec![2, 6]];
    assert_eq!(
        natural_moves(&ising, &sources(&bad)).unwrap_err(),
        MoveError::Node { node: 6 }
    );
    assert_eq!(
        Greedy::new(ising.clone(), &[vec![0, 9]], true).unwrap_err(),
        MoveError::Node { node: 9 }
    );
    assert_eq!(
        Greedy::new(ising.clone(), &[vec![1, 0, 1]], false).unwrap_err(),
        MoveError::Repeat { node: 1 },
        "a repeated member flips back and never terminates"
    );
    let greedy = Greedy::new(ising, &[], true).unwrap();
    assert_eq!(
        greedy.descend(&mut [1; 5], None).unwrap_err(),
        DescentError::Length
    );
    assert_eq!(
        greedy.descend(&mut [1, 1, 0, 1, 1, 1], None).unwrap_err(),
        DescentError::Value { node: 2 }
    );
}

#[test]
fn a_cancelled_token_stops_the_descent_before_a_sweep() {
    let greedy = Greedy::new(ising(), &[], true).unwrap();
    let token = CancelToken::default();
    token.cancel_through(3);
    let mut s = vec![1; 6];
    assert_eq!(
        greedy.descend(&mut s, Some((&token, Some(3)))).unwrap_err(),
        DescentError::Cancelled
    );
    assert!(greedy.descend(&mut s, Some((&token, Some(4)))).is_ok());
}

/// One research model with its archived reads and the saved descents.
#[derive(serde::Deserialize)]
struct ParityCase {
    cell: String,
    nonce: String,
    h: Vec<i64>,
    edges: Vec<(u32, u32)>,
    j: Vec<i64>,
    tiles: Vec<Vec<u32>>,
    pairs: Vec<Vec<u32>>,
    listed: Vec<Vec<u32>>,
    reads: Vec<String>,
    g1_c: Vec<String>,
    gnat_c: Vec<String>,
    energies: std::collections::BTreeMap<String, Vec<i64>>,
    applied: std::collections::BTreeMap<String, u64>,
}

fn decode(s: &str) -> Vec<i8> {
    s.bytes().map(|b| if b == b'+' { 1 } else { -1 }).collect()
}

/// The research states, energies, and applied-move counts of `g1_c` and
/// `gnat_c` on every archived read. The cases come from
/// `research/greedy_parity/export.py`.
#[test]
#[ignore = "needs research data: run research/greedy_parity/export.py, set QUIP_GREEDY_PARITY, and run with --ignored"]
fn research_parity() {
    let dir =
        std::env::var("QUIP_GREEDY_PARITY").expect("QUIP_GREEDY_PARITY names the export directory");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no cases in {dir}");
    for path in paths {
        let case: ParityCase = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let label = format!("{} {}", case.cell, &case.nonce[..12]);
        let ising = Ising::new(&case.h, &case.edges, &case.j).unwrap();
        let tiles: Vec<[u32; 4]> = case
            .tiles
            .iter()
            .map(|t| t.as_slice().try_into().unwrap())
            .collect();
        let pairs: Vec<(u32, u32)> = case.pairs.iter().map(|p| (p[0], p[1])).collect();
        let sources = MoveSources {
            tiles: &tiles,
            pairs: &pairs,
            listed: &case.listed,
        };
        let (natural, _) = natural_moves(&ising, &sources).unwrap();
        let reads: Vec<Vec<i8>> = case.reads.iter().map(|s| decode(s)).collect();
        assert_eq!(reads.len(), 48, "{label}: the archive holds 48 reads");
        for name in ["raw", "g1_c", "gnat_c"] {
            assert_eq!(
                case.energies[name].len(),
                reads.len(),
                "{label}: {name} energies"
            );
        }
        assert_eq!(case.g1_c.len(), reads.len(), "{label}: g1_c states");
        assert_eq!(case.gnat_c.len(), reads.len(), "{label}: gnat_c states");
        let raw: Vec<i64> = reads.iter().map(|s| ising.energy(s)).collect();
        assert_eq!(raw, case.energies["raw"], "{label}: raw energies");
        for (name, moves, saved) in [
            ("g1_c", Vec::new(), &case.g1_c),
            ("gnat_c", natural, &case.gnat_c),
        ] {
            let greedy = Greedy::new(ising.clone(), &moves, true).unwrap();
            let mut applied = 0;
            for (r, (read, saved)) in reads.iter().zip(saved).enumerate() {
                let mut s = read.clone();
                applied += greedy.descend(&mut s, None).unwrap();
                assert!(
                    s == decode(saved),
                    "{label} {name}: read {r} differs from the research state"
                );
                assert_eq!(
                    ising.energy(&s),
                    case.energies[name][r],
                    "{label} {name}: read {r} energy"
                );
            }
            assert_eq!(applied, case.applied[name], "{label} {name}: applied moves");
        }
    }
}
