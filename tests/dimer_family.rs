//! The bounded weighted-dimer kernel and its declaration.
//!
//! `tests/fixtures/dimer_weighted_vectors.json` holds states that the research
//! `dimer_bench` produced for the `ice-loop`, `ice-loop64`, and `ice-defect64`
//! variants on small twisted-ice models (`research/dimer_parity/export.py`).
//! The kernel keeps the research random streams, so each case matches the
//! research states and counts exactly. The other tests check the kernel
//! against exhaustive energies and check that it rejects what it must reject
//! before it samples. The ignored `research_parity` test replays a saved
//! research run of a full model.

use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_core::{CancelToken, IsingGraph, SampleParams, Sampler};
use quip_solver_cpu::families::dimer_weighted::{DimerError, DimerGraph, IceLoopStats, Method};
use quip_solver_cpu::{DimerSampler, DIMER_METHOD};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    name: String,
    #[serde(default)]
    variant: String,
    h: Vec<f64>,
    edges: Vec<(usize, usize)>,
    j: Vec<f64>,
    tiles: Vec<[u32; 4]>,
    pairs: Vec<(u32, u32)>,
    beta: (f64, f64),
    seed: u64,
    sweeps: usize,
    sweeps_per_beta: usize,
    reads: usize,
    spins: Vec<String>,
    stats: Stats,
}

#[derive(Deserialize, Default)]
struct Stats {
    attempts: u64,
    formed_loops: u64,
    formed_strings: u64,
    accepted: u64,
    flipped_spins: u64,
    #[serde(default)]
    defect_attempts: u64,
    #[serde(default)]
    formed_defects: u64,
    #[serde(default)]
    accepted_defects: u64,
}

impl Case {
    fn graph(&self) -> IsingGraph {
        IsingGraph::new(self.h.clone(), self.j.clone(), self.edges.clone())
    }

    fn params(&self) -> SampleParams {
        SampleParams {
            num_reads: self.reads,
            num_sweeps: self.sweeps,
            sweeps_per_beta: self.sweeps_per_beta,
            beta_range: Some(self.beta),
            seed: self.seed,
            ..SampleParams::default()
        }
    }

    fn method(&self) -> Method {
        match self.variant.as_str() {
            "ice-loop-weighted" => Method::LOOP,
            "ice-loop64-weighted" => Method::LOOP64,
            "ice-defect64-weighted" => Method::DEFECT64,
            other => unreachable!("unknown variant {other}"),
        }
    }

    /// Run the kernel and compare every state and count with the research.
    fn assert_matches(&self, method: Method) {
        let graph = self.graph();
        let prepared = DimerGraph::new(&graph, &self.tiles, &self.pairs).expect("a valid model");
        let (states, stats) = prepared
            .sample(&self.params(), method, None)
            .expect("the kernel runs");
        assert_eq!(states.len(), self.reads, "{}", self.name);
        for (read, (state, saved)) in states.iter().zip(&self.spins).enumerate() {
            let text: String = state
                .iter()
                .map(|&s| if s > 0 { '+' } else { '-' })
                .collect();
            assert_eq!(&text, saved, "{}: read {read}", self.name);
        }
        let expected = IceLoopStats {
            attempts: self.stats.attempts,
            formed_loops: self.stats.formed_loops,
            formed_strings: self.stats.formed_strings,
            accepted: self.stats.accepted,
            flipped_spins: self.stats.flipped_spins,
            defect_attempts: self.stats.defect_attempts,
            formed_defects: self.stats.formed_defects,
            accepted_defects: self.stats.accepted_defects,
        };
        assert_eq!(stats, expected, "{}", self.name);
    }
}

#[derive(Deserialize)]
struct Vectors {
    cases: Vec<Case>,
}

fn vectors() -> Vectors {
    serde_json::from_str(include_str!("fixtures/dimer_weighted_vectors.json"))
        .expect("the committed vectors parse")
}

#[test]
fn every_research_variant_matches_the_research_states_and_counts() {
    let vectors = vectors();
    assert_eq!(vectors.cases.len(), 9);
    for case in &vectors.cases {
        case.assert_matches(case.method());
    }
}

#[test]
fn the_selected_method_is_the_loop64_variant() {
    assert_eq!(DIMER_METHOD, Method::LOOP64);
    assert!(vectors()
        .cases
        .iter()
        .any(|case| case.variant == "ice-loop64-weighted"));
}

/// One tetrahedron with all four spins coupled antiferromagnetically, one
/// planted pair, and three weak bonds. Six spins, so 64 states.
fn tiny() -> (IsingGraph, Vec<[u32; 4]>, Vec<(u32, u32)>) {
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for u in 0..4usize {
        for v in u + 1..4 {
            edges.push((u, v));
            j.push(1.0);
        }
    }
    for (u, v, c) in [(4, 5, -1.0), (0, 4, 0.062), (1, 5, -0.062), (2, 4, -0.062)] {
        edges.push((u, v));
        j.push(c);
    }
    (
        IsingGraph::new(vec![0.0; 6], j, edges),
        vec![[0, 1, 2, 3]],
        vec![(4, 5)],
    )
}

fn milli_energy(graph: &IsingGraph, bits: usize) -> i64 {
    let spin = |n: usize| if (bits >> n) & 1 == 0 { 1_i64 } else { -1 };
    graph
        .edges
        .iter()
        .zip(&graph.j)
        .map(|(&(u, v), &c)| (c * 1000.0).round() as i64 * spin(u) * spin(v))
        .sum()
}

#[test]
fn a_cold_anneal_of_a_tiny_model_finds_the_exhaustive_ground_energy() {
    let (graph, tiles, pairs) = tiny();
    let ground = (0..64)
        .map(|bits| milli_energy(&graph, bits))
        .min()
        .unwrap();
    let prepared = DimerGraph::new(&graph, &tiles, &pairs).expect("valid");
    let params = SampleParams {
        num_reads: 64,
        num_sweeps: 64,
        sweeps_per_beta: 4,
        beta_range: Some((0.5, 40.0)),
        seed: 3,
        ..SampleParams::default()
    };
    let (states, _) = prepared.sample(&params, DIMER_METHOD, None).unwrap();
    let energies: Vec<i64> = states
        .iter()
        .map(|state| {
            let bits = state
                .iter()
                .enumerate()
                .fold(0usize, |acc, (n, &s)| acc | (usize::from(s < 0) << n));
            milli_energy(&graph, bits)
        })
        .collect();
    assert!(
        energies.iter().all(|&e| e >= ground),
        "no read beats the ground state"
    );
    assert_eq!(energies.iter().copied().min(), Some(ground));
}

#[test]
fn the_same_seed_gives_the_same_states_and_a_new_seed_changes_them() {
    let vectors = vectors();
    let case = &vectors.cases[1];
    let graph = case.graph();
    let prepared = DimerGraph::new(&graph, &case.tiles, &case.pairs).unwrap();
    let mut params = case.params();
    let first = prepared.sample(&params, DIMER_METHOD, None).unwrap();
    assert_eq!(first, prepared.sample(&params, DIMER_METHOD, None).unwrap());
    params.seed += 1;
    assert_ne!(
        first.0,
        prepared.sample(&params, DIMER_METHOD, None).unwrap().0
    );
}

#[test]
fn a_cancelled_watermark_stops_the_anneal() {
    let vectors = vectors();
    let case = &vectors.cases[0];
    let graph = case.graph();
    let prepared = DimerGraph::new(&graph, &case.tiles, &case.pairs).unwrap();
    let token = CancelToken::default();
    token.cancel_through(5);
    let abandoned = prepared.sample(&case.params(), DIMER_METHOD, Some((&token, Some(5))));
    assert_eq!(abandoned.unwrap_err(), DimerError::Cancelled);
    let other = prepared.sample(&case.params(), DIMER_METHOD, Some((&token, Some(6))));
    assert!(other.is_ok(), "a later watermark runs");
    let none = prepared.sample(&case.params(), DIMER_METHOD, Some((&token, None)));
    assert!(none.is_ok(), "a job without a watermark is never cancelled");
}

fn invalid(result: Result<impl Sized, DimerError>, what: &str) {
    assert!(matches!(result, Err(DimerError::Invalid(_))), "{what}");
}

#[test]
fn the_model_is_checked_before_any_sampling() {
    let (graph, tiles, pairs) = tiny();
    assert!(DimerGraph::new(&graph, &tiles, &pairs).is_ok());

    let mut field = graph.clone();
    field.h[5] = 0.001;
    invalid(DimerGraph::new(&field, &tiles, &pairs), "a field");

    let mut inexact = graph.clone();
    inexact.j[5] = 0.0624;
    invalid(
        DimerGraph::new(&inexact, &tiles, &pairs),
        "an inexact coupling",
    );

    let mut overflow = graph.clone();
    overflow.j[5] = 1.0e9;
    invalid(
        DimerGraph::new(&overflow, &tiles, &pairs),
        "an oversize coupling",
    );

    invalid(
        DimerGraph::new(&graph, &[[0, 1, 2, 99]], &pairs),
        "a tile node outside",
    );
    invalid(
        DimerGraph::new(&graph, &[[0, 1, 2, 2]], &pairs),
        "a repeated tile node",
    );
    invalid(
        DimerGraph::new(&graph, &[[0, 1, 2, 4]], &pairs),
        "a tile without its six bonds",
    );
    invalid(DimerGraph::new(&graph, &[], &pairs), "no tiles");
    invalid(
        DimerGraph::new(&graph, &tiles, &[(4, 99)]),
        "a pair outside",
    );
    invalid(
        DimerGraph::new(&graph, &tiles, &[(4, 5), (5, 4)]),
        "a repeated pair",
    );
    invalid(
        DimerGraph::new(&graph, &tiles, &[(0, 4)]),
        "a pair that is not -1000",
    );
    invalid(
        DimerGraph::new(&graph, &tiles, &[(3, 5)]),
        "a pair without an edge",
    );
}

#[test]
fn tiles_must_be_a_local_gauge_of_antiferromagnetic_ice() {
    let (mut twisted, tiles, pairs) = tiny();
    // Flip one tile bond so the six signs are no longer a local gauge.
    twisted.j[0] = -1.0;
    invalid(
        DimerGraph::new(&twisted, &tiles, &pairs),
        "signs that no gauge explains",
    );
}

#[test]
fn a_spin_in_three_tiles_is_refused() {
    let mut edges = Vec::new();
    let mut j = Vec::new();
    let mut tiles = Vec::new();
    for t in 0..3u32 {
        let nodes = [0, 3 * t + 1, 3 * t + 2, 3 * t + 3];
        for a in 0..4 {
            for b in a + 1..4 {
                edges.push((nodes[a] as usize, nodes[b] as usize));
                j.push(1.0);
            }
        }
        tiles.push(nodes);
    }
    let graph = IsingGraph::new(vec![0.0; 10], j, edges);
    invalid(
        DimerGraph::new(&graph, &tiles, &[]),
        "a spin in three tiles",
    );
}

#[test]
fn the_sampler_declares_the_whole_revision_and_no_other_family() {
    let declaration = DimerSampler::declaration();
    assert_eq!(declaration.solver_id, "quip-cpu-dimer/1");
    for claims in [declaration.generated, declaration.accepted] {
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].revisions.len(), 1);
        assert_eq!(claims[0].revisions[0].family_id, TWISTED_DIMER_V1.family_id);
        assert_eq!(claims[0].revisions[0].revision, TWISTED_DIMER_V1.revision);
        assert!(
            claims[0].only_params.is_empty(),
            "the claim names the revision"
        );
    }
    // Charged and field-biased ice are other families, so a job of either
    // reaches the session as an unsupported family and never reaches the kernel.
    for other in ["quip/ice-charge", "quip/ice-fields"] {
        assert!(declaration
            .accepted
            .iter()
            .all(|claim| claim.revisions.iter().all(|r| r.family_id != other)));
    }
}

/// Replay a saved research run of `ice-loop64-weighted` on a full model.
///
///     QUIP_RESEARCH_ROOT=... python3 -I research/dimer_parity/export.py research OUT
///     QUIP_DIMER_PARITY=OUT cargo test --release --features experimental \
///         --test dimer_family -- --ignored research_parity
#[test]
#[ignore = "needs QUIP_DIMER_PARITY, a directory written by research/dimer_parity/export.py"]
fn research_parity() {
    let dir = std::env::var("QUIP_DIMER_PARITY").expect("QUIP_DIMER_PARITY names the directory");
    let mut replayed = 0;
    for entry in std::fs::read_dir(dir).expect("the directory reads") {
        let path = entry.expect("an entry").path();
        let text = std::fs::read_to_string(&path).expect("a case file");
        let case: Case = serde_json::from_str(&text).expect("a case");
        case.assert_matches(Method::LOOP64);
        replayed += 1;
    }
    assert!(replayed > 0, "no saved run to replay");
}
