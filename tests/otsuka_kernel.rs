//! The Otsuka cluster kernel on tiny gauged-ice fixtures.
//!
//! Every fixture is small enough to enumerate, so each test compares the
//! kernel with an exact answer: the direct energy, the exact weak-energy
//! change of a move, or the Boltzmann distribution itself.
#![cfg(feature = "experimental")]

use quip_solver_core::CancelToken;
use quip_solver_cpu::families::otsuka::{
    anneal, AnnealError, Chain, GraphError, Model, Motifs, OtsukaGraph, Schedule, ScheduleError,
    SpinError, Weights,
};
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// An owned model and its motifs.
#[derive(Clone)]
struct Fixture {
    h: Vec<i32>,
    edges: Vec<(u32, u32)>,
    j: Vec<i32>,
    tiles: Vec<[u32; 4]>,
    gauges: Vec<[i8; 4]>,
    pairs: Vec<(u32, u32)>,
}

impl Fixture {
    fn graph(&self, weights: Weights) -> Result<OtsukaGraph, GraphError> {
        let model = Model {
            h_milli: &self.h,
            edges: &self.edges,
            j_milli: &self.j,
        };
        let motifs = Motifs {
            tiles: &self.tiles,
            gauges: &self.gauges,
            pairs: &self.pairs,
        };
        OtsukaGraph::new(&model, &motifs, weights)
    }

    /// The energy in milli straight from the edge list.
    fn direct_energy(&self, s: &[i8]) -> i64 {
        self.edges
            .iter()
            .zip(&self.j)
            .map(|(&(u, v), &j)| i64::from(j) * i64::from(s[u as usize] * s[v as usize]))
            .sum()
    }

    fn num_nodes(&self) -> usize {
        self.h.len()
    }

    fn with_tile_edges(mut self) -> Self {
        for (tile, gauge) in self.tiles.iter().zip(&self.gauges) {
            for a in 0..4 {
                for b in a + 1..4 {
                    self.edges.push((tile[a], tile[b]));
                    self.j.push(1000 * i32::from(gauge[a] * gauge[b]));
                }
            }
        }
        self
    }
}

/// One gauged tile and nothing else.
fn one_tile() -> Fixture {
    Fixture {
        h: vec![0; 4],
        edges: Vec::new(),
        j: Vec::new(),
        tiles: vec![[0, 1, 2, 3]],
        gauges: vec![[1, -1, 1, -1]],
        pairs: Vec::new(),
    }
    .with_tile_edges()
}

/// Two tiles sharing node 3, two pairs, and four weak edges on eight nodes.
fn corner_sharing() -> Fixture {
    let mut fixture = Fixture {
        h: vec![0; 8],
        edges: vec![(0, 4), (5, 7), (1, 5), (2, 6), (1, 4), (6, 7)],
        j: vec![-1000, -1000, 300, -200, 700, 500],
        tiles: vec![[0, 1, 2, 3], [3, 4, 5, 6]],
        gauges: vec![[1, -1, 1, 1], [-1, 1, 1, -1]],
        pairs: vec![(0, 4), (5, 7)],
    };
    fixture = fixture.with_tile_edges();
    fixture
}

const SOFT_WEIGHTS: [Weights; 5] = [
    Weights::FULL,
    Weights {
        pair_milli: 600,
        tile_milli: 1000,
    },
    Weights {
        pair_milli: 0,
        tile_milli: 1000,
    },
    Weights {
        pair_milli: 1000,
        tile_milli: 400,
    },
    Weights {
        pair_milli: 250,
        tile_milli: 1,
    },
];

/// Every spin state of `n` nodes, in index order.
fn states(n: usize) -> impl Iterator<Item = Vec<i8>> {
    (0..1_u32 << n).map(move |bits| {
        (0..n)
            .map(|i| if bits >> i & 1 == 1 { -1 } else { 1 })
            .collect()
    })
}

fn index_of(s: &[i8]) -> usize {
    s.iter()
        .enumerate()
        .filter(|&(_, &x)| x == -1)
        .map(|(i, _)| 1 << i)
        .sum()
}

#[test]
fn a_single_tile_is_all_strong_energy() {
    let fixture = one_tile();
    let graph = fixture.graph(Weights::FULL).unwrap();
    let gauge = fixture.gauges[0];
    for s in states(4) {
        let q: i64 = (0..4).map(|k| i64::from(gauge[k] * s[k])).sum();
        assert_eq!(graph.energy(&s), 500 * (q * q - 4), "{s:?}");
        assert_eq!(graph.strong_energy(&s), graph.energy(&s), "{s:?}");
        assert_eq!(graph.weak_energy(&s), 0, "{s:?}");
    }
}

#[test]
fn the_energy_splits_exactly_for_every_state_and_soft_weight() {
    let fixture = corner_sharing();
    for weights in SOFT_WEIGHTS {
        let graph = fixture.graph(weights).unwrap();
        for s in states(fixture.num_nodes()) {
            let direct = fixture.direct_energy(&s);
            assert_eq!(graph.energy(&s), direct, "{weights:?} {s:?}");
            assert_eq!(
                graph.strong_energy(&s) + graph.weak_energy(&s),
                direct,
                "{weights:?} {s:?}"
            );
        }
    }
}

#[test]
fn a_cluster_sweep_reports_the_exact_energy_change() {
    let fixture = corner_sharing();
    let mut rng = SmallRng::seed_from_u64(11);
    for weights in SOFT_WEIGHTS {
        let graph = fixture.graph(weights).unwrap();
        for beta in [0.05, 0.7, 4.0] {
            for start in states(fixture.num_nodes()) {
                let mut chain = Chain::new(&graph, start.clone()).unwrap();
                let before = graph.energy(&start);
                let change = chain.cluster_sweep(beta, &mut rng);
                let s = chain.spins();
                assert!(s.iter().all(|&x| x == 1 || x == -1));
                assert_eq!(
                    graph.energy(s) - before,
                    change,
                    "{weights:?} beta {beta} from {start:?}"
                );
            }
        }
    }
}

#[test]
fn a_single_sweep_reports_the_exact_energy_change() {
    let fixture = corner_sharing();
    let graph = fixture.graph(Weights::FULL).unwrap();
    let mut rng = SmallRng::seed_from_u64(5);
    for beta in [0.05, 0.7, 4.0] {
        for start in states(fixture.num_nodes()) {
            let mut chain = Chain::new(&graph, start.clone()).unwrap();
            let before = graph.energy(&start);
            let change = chain.single_sweep(beta, &mut rng);
            assert_eq!(
                graph.energy(chain.spins()) - before,
                change,
                "beta {beta} {start:?}"
            );
        }
    }
}

/// Total variation distance between the visited states of `sweeps` cluster
/// sweeps and the exact Boltzmann distribution.
fn boltzmann_distance(fixture: &Fixture, weights: Weights, beta: f64, sweeps: usize) -> f64 {
    let graph = fixture.graph(weights).unwrap();
    let n = fixture.num_nodes();
    let exact: Vec<f64> = states(n)
        .map(|s| (-beta * graph.energy(&s) as f64 / 1000.0).exp())
        .collect();
    let total: f64 = exact.iter().sum();
    let mut counts = vec![0_u64; exact.len()];
    let mut chain = Chain::new(&graph, vec![1; n]).unwrap();
    let mut rng = SmallRng::seed_from_u64(2014);
    for _ in 0..1000 {
        chain.cluster_sweep(beta, &mut rng);
    }
    for _ in 0..sweeps {
        chain.cluster_sweep(beta, &mut rng);
        counts[index_of(chain.spins())] += 1;
    }
    exact
        .iter()
        .zip(&counts)
        .map(|(&p, &c)| (p / total - c as f64 / sweeps as f64).abs())
        .sum::<f64>()
        / 2.0
}

#[test]
fn cluster_sweeps_sample_the_boltzmann_distribution() {
    let fixture = corner_sharing();
    for (weights, beta) in [
        (Weights::FULL, 0.4),
        (
            Weights {
                pair_milli: 600,
                tile_milli: 700,
            },
            0.6,
        ),
        (
            Weights {
                pair_milli: 0,
                tile_milli: 1000,
            },
            0.3,
        ),
    ] {
        let distance = boltzmann_distance(&fixture, weights, beta, 400_000);
        assert!(
            distance < 0.03,
            "{weights:?} beta {beta}: distance {distance}"
        );
    }
}

/// A lone tile has three charge classes, `|q| = 0, 2, 4`: ice, defect, and
/// aligned. Aligned states are rare, so the visit rate of each class is a
/// sharper check of the defect bond probability than the full distribution.
#[test]
fn a_lone_tile_visits_each_charge_class_at_its_boltzmann_rate() {
    let fixture = one_tile();
    let gauge = fixture.gauges[0];
    let class = |s: &[i8]| {
        let q: i8 = (0..4).map(|k| gauge[k] * s[k]).sum();
        usize::from(q.unsigned_abs() / 2)
    };
    let sweeps = 1_000_000;
    for weights in [
        Weights::FULL,
        Weights {
            pair_milli: 1000,
            tile_milli: 500,
        },
    ] {
        let graph = fixture.graph(weights).unwrap();
        for beta in [0.2, 0.5, 1.0] {
            let mut exact = [0.0; 3];
            for s in states(4) {
                exact[class(&s)] += (-beta * graph.energy(&s) as f64 / 1000.0).exp();
            }
            let total: f64 = exact.iter().sum();
            let mut seen = [0_u64; 3];
            let mut chain = Chain::new(&graph, vec![1; 4]).unwrap();
            let mut rng = SmallRng::seed_from_u64(46);
            for _ in 0..sweeps {
                chain.cluster_sweep(beta, &mut rng);
                seen[class(chain.spins())] += 1;
            }
            for k in 0..3 {
                let p = exact[k] / total;
                let rate = seen[k] as f64 / f64::from(sweeps);
                let tolerance = 10.0 * (p * (1.0 - p) / f64::from(sweeps)).sqrt();
                assert!(
                    (rate - p).abs() < tolerance,
                    "{weights:?} beta {beta} class {k}: rate {rate}, exact {p}"
                );
            }
        }
    }
}

/// From a defect state the minority spin bonds to each majority spin with
/// probability `(1 - z^3) / 3`. A bond keeps the pair opposite, and a free pair
/// ends opposite half the time, so each pair ends opposite with probability
/// `1/2 + (1 - z^3) / 6`. A biased partner draw moves one pair off that rate.
#[test]
fn a_defect_tile_bonds_each_partner_equally() {
    let fixture = Fixture {
        h: vec![0; 4],
        edges: Vec::new(),
        j: Vec::new(),
        tiles: vec![[0, 1, 2, 3]],
        gauges: vec![[1, 1, 1, 1]],
        pairs: Vec::new(),
    }
    .with_tile_edges();
    let graph = fixture.graph(Weights::FULL).unwrap();
    let beta = 2.0;
    let start = [-1, 1, 1, 1];
    let mut chain = Chain::new(&graph, start.to_vec()).unwrap();
    let mut rng = SmallRng::seed_from_u64(3);
    let sweeps = 10_000_000_u32;
    let mut opposite = [0_u32; 3];
    for _ in 0..sweeps {
        chain.reset(&start).unwrap();
        chain.cluster_sweep(beta, &mut rng);
        let s = chain.spins();
        for k in 0..3 {
            opposite[k] += u32::from(s[0] != s[k + 1]);
        }
    }
    let z3 = (-6.0 * beta).exp();
    let exact = 0.5 + (1.0 - z3) / 6.0;
    let tolerance = 5.0 * (exact * (1.0 - exact) / f64::from(sweeps)).sqrt();
    for (k, &count) in opposite.iter().enumerate() {
        let rate = f64::from(count) / f64::from(sweeps);
        assert!(
            (rate - exact).abs() < tolerance,
            "partner {}: rate {rate}, exact {exact}",
            k + 1
        );
    }
}

fn geometric(low: f64, high: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| low * (high / low).powf(i as f64 / (count - 1) as f64))
        .collect()
}

#[test]
fn an_anneal_reaches_the_ground_state_of_the_fixture() {
    let fixture = corner_sharing();
    let graph = fixture.graph(Weights::FULL).unwrap();
    let ground = states(fixture.num_nodes())
        .map(|s| fixture.direct_energy(&s))
        .min()
        .unwrap();
    let betas = geometric(0.1, 8.0, 500);
    let schedule = Schedule::new(&betas, 4).unwrap();
    for seed in 0..8 {
        let sample = anneal(&graph, &schedule, seed, None).unwrap();
        assert_eq!(sample.energy_milli, fixture.direct_energy(&sample.spins));
        assert_eq!(sample.energy_milli, ground, "seed {seed}");
        assert_eq!(sample.sweeps, betas.len());
    }
}

#[test]
fn an_anneal_with_soft_weights_reaches_the_ground_state() {
    let fixture = corner_sharing();
    let ground = states(fixture.num_nodes())
        .map(|s| fixture.direct_energy(&s))
        .min()
        .unwrap();
    let betas = geometric(0.1, 8.0, 500);
    let schedule = Schedule::new(&betas, 0).unwrap();
    for weights in SOFT_WEIGHTS {
        let graph = fixture.graph(weights).unwrap();
        for seed in 0..4 {
            let sample = anneal(&graph, &schedule, seed, None).unwrap();
            assert_eq!(sample.energy_milli, fixture.direct_energy(&sample.spins));
            assert_eq!(sample.energy_milli, ground, "{weights:?} seed {seed}");
        }
    }
}

#[test]
fn a_chain_refuses_spins_that_do_not_fit_the_graph() {
    let fixture = corner_sharing();
    let graph = fixture.graph(Weights::FULL).unwrap();
    assert_eq!(
        Chain::new(&graph, vec![1; 7]).unwrap_err(),
        SpinError::Length
    );
    assert_eq!(
        Chain::new(&graph, vec![1, 1, 0, 1, 1, 1, 1, 1]).unwrap_err(),
        SpinError::Value { node: 2 }
    );
    let mut chain = Chain::new(&graph, vec![1; 8]).unwrap();
    assert_eq!(chain.reset(&[1; 9]).unwrap_err(), SpinError::Length);
    assert_eq!(
        chain.reset(&[1, 1, 1, 1, 1, 1, 1, 2]).unwrap_err(),
        SpinError::Value { node: 7 }
    );
    assert_eq!(chain.spins(), &[1; 8]);
}

#[test]
fn the_same_seed_gives_the_same_sample() {
    let fixture = corner_sharing();
    let graph = fixture.graph(Weights::FULL).unwrap();
    let betas = geometric(0.1, 2.0, 50);
    let schedule = Schedule::new(&betas, 0).unwrap();
    let first = anneal(&graph, &schedule, 99, None).unwrap();
    let again = anneal(&graph, &schedule, 99, None).unwrap();
    assert_eq!(first, again);
}

#[test]
fn a_cancelled_token_stops_the_anneal_before_a_sweep() {
    let fixture = corner_sharing();
    let graph = fixture.graph(Weights::FULL).unwrap();
    let betas = geometric(0.1, 2.0, 50);
    let schedule = Schedule::new(&betas, 0).unwrap();
    let token = CancelToken::default();
    token.cancel_through(3);
    assert_eq!(
        anneal(&graph, &schedule, 1, Some((&token, Some(3)))),
        Err(AnnealError::Cancelled)
    );
    assert!(anneal(&graph, &schedule, 1, Some((&token, Some(4)))).is_ok());
}

#[test]
fn unsupported_schedules_are_refused_before_an_anneal() {
    assert_eq!(Schedule::new(&[], 0).unwrap_err(), ScheduleError::Empty);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            Schedule::new(&[0.5, bad], 0).unwrap_err(),
            ScheduleError::Beta { index: 1 },
            "{bad}"
        );
    }
}

#[test]
fn unsupported_weights_and_fields_are_refused() {
    let fixture = corner_sharing();
    for weights in [
        Weights {
            pair_milli: -1,
            tile_milli: 1000,
        },
        Weights {
            pair_milli: 1001,
            tile_milli: 1000,
        },
        Weights {
            pair_milli: 1000,
            tile_milli: 0,
        },
        Weights {
            pair_milli: 1000,
            tile_milli: 1001,
        },
    ] {
        assert_eq!(
            fixture.graph(weights).unwrap_err(),
            GraphError::Weights,
            "{weights:?}"
        );
    }
    let mut fields = fixture.clone();
    fields.h[2] = 100;
    assert_eq!(fields.graph(Weights::FULL).unwrap_err(), GraphError::Fields);
    let mut empty = fixture;
    empty.h.clear();
    assert_eq!(empty.graph(Weights::FULL).unwrap_err(), GraphError::Empty);
}

#[test]
fn malformed_motifs_are_refused() {
    let base = corner_sharing;
    let cases: Vec<(&str, Fixture, GraphError)> = vec![
        (
            "edge and coupling counts differ",
            {
                let mut f = base();
                f.j.pop();
                f
            },
            GraphError::Lengths,
        ),
        (
            "edge end out of range",
            {
                let mut f = base();
                f.edges[2] = (1, 8);
                f
            },
            GraphError::Edge { index: 2 },
        ),
        (
            "self loop",
            {
                let mut f = base();
                f.edges[2] = (1, 1);
                f
            },
            GraphError::Edge { index: 2 },
        ),
        (
            "coupling above 1000 milli",
            {
                let mut f = base();
                f.j[2] = 1001;
                f
            },
            GraphError::Edge { index: 2 },
        ),
        (
            "coupling at the minimum i32",
            {
                let mut f = base();
                f.j[2] = i32::MIN;
                f
            },
            GraphError::Edge { index: 2 },
        ),
        (
            "duplicate edge in either orientation",
            {
                let mut f = base();
                f.edges.push((5, 1));
                f.j.push(300);
                f
            },
            GraphError::DuplicateEdge { index: 18 },
        ),
        (
            "gauge count differs from tile count",
            {
                let mut f = base();
                f.gauges.pop();
                f
            },
            GraphError::Lengths,
        ),
        (
            "gauge entry is not a sign",
            {
                let mut f = base();
                f.gauges[1][2] = 0;
                f
            },
            GraphError::Gauge { tile: 1 },
        ),
        (
            "tile node out of range",
            {
                let mut f = base();
                f.tiles[1][3] = 9;
                f
            },
            GraphError::Tile { tile: 1 },
        ),
        (
            "tile repeats a node",
            {
                let mut f = base();
                f.tiles[1] = [3, 4, 4, 6];
                f
            },
            GraphError::Tile { tile: 1 },
        ),
        (
            "tile coupling disagrees with its gauge",
            {
                let mut f = base();
                f.gauges[0][1] = 1;
                f
            },
            GraphError::TileCoupling { tile: 0 },
        ),
        (
            "tile edge missing",
            {
                let mut f = base();
                let at = f.edges.iter().position(|&e| e == (4, 5)).unwrap();
                f.edges.remove(at);
                f.j.remove(at);
                f
            },
            GraphError::TileCoupling { tile: 1 },
        ),
        (
            "two tiles share an edge",
            {
                let mut f = base();
                f.tiles.push([0, 1, 2, 3]);
                f.gauges.push([1, -1, 1, 1]);
                f
            },
            GraphError::TileOverlap { tile: 2 },
        ),
        (
            "pair node out of range",
            {
                let mut f = base();
                f.pairs[1] = (5, 8);
                f
            },
            GraphError::Pair { pair: 1 },
        ),
        (
            "pairs share a node",
            {
                let mut f = base();
                f.pairs[1] = (4, 7);
                f
            },
            GraphError::Pair { pair: 1 },
        ),
        (
            "pair coupling is not -1000 milli",
            {
                let mut f = base();
                f.j[1] = 1000;
                f
            },
            GraphError::Pair { pair: 1 },
        ),
        (
            "pair sits on a tile edge",
            {
                let mut f = base();
                f.pairs[1] = (5, 6);
                f
            },
            GraphError::Pair { pair: 1 },
        ),
        (
            "strong edge outside every motif",
            {
                let mut f = base();
                f.j[2] = -1000;
                f
            },
            GraphError::StrongEdge { index: 2 },
        ),
    ];
    for (name, fixture, want) in cases {
        assert_eq!(fixture.graph(Weights::FULL).unwrap_err(), want, "{name}");
    }
}
