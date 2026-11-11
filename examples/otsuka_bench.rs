//! Time the Otsuka kernel on a twisted-ice-dimer model file.
//!
//! The file holds a header `n m tiles pairs target_milli`, then `m` lines
//! `u v j_milli`, `tiles` lines of four nodes, and `pairs` lines of two nodes.
//!
//! Usage: `otsuka_bench <model> <anneals> <sweeps> <beta_min> <beta_max>
//! <pair_milli> <single_every> <seed>`
use std::collections::HashMap;
use std::time::Instant;

use quip_solver_cpu::families::otsuka::{anneal, Model, Motifs, OtsukaGraph, Schedule, Weights};

#[expect(
    clippy::print_stdout,
    reason = "otsuka_bench reports one parseable JSON line on stdout"
)]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).expect("a model file");
    let arg = |i: usize| -> f64 { args[i].parse().expect("a number") };
    let (anneals, sweeps) = (arg(2) as usize, arg(3) as usize);
    let (beta_min, beta_max) = (arg(4), arg(5));
    let pair_milli = arg(6) as i32;
    let single_every = arg(7) as usize;
    let seed = arg(8) as u64;

    let mut lines = text.lines();
    let head: Vec<i64> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();
    let (n, m, t, p, target) = (
        head[0] as usize,
        head[1] as usize,
        head[2] as usize,
        head[3] as usize,
        head[4],
    );
    let mut row = || -> Vec<i64> {
        lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect()
    };
    let mut edges = Vec::with_capacity(m);
    let mut j = Vec::with_capacity(m);
    let mut index = HashMap::new();
    for _ in 0..m {
        let r = row();
        edges.push((r[0] as u32, r[1] as u32));
        j.push(r[2] as i32);
        index.insert((r[0].min(r[1]), r[0].max(r[1])), r[2] as i32);
    }
    let mut tiles = Vec::with_capacity(t);
    let mut gauges = Vec::with_capacity(t);
    for _ in 0..t {
        let r = row();
        let mut g = [1_i8; 4];
        for k in 1..4 {
            let jk = index[&(r[0].min(r[k]), r[0].max(r[k]))];
            g[k] = if jk > 0 { 1 } else { -1 };
        }
        tiles.push([r[0] as u32, r[1] as u32, r[2] as u32, r[3] as u32]);
        gauges.push(g);
    }
    let pairs: Vec<(u32, u32)> = (0..p)
        .map(|_| {
            let r = row();
            (r[0] as u32, r[1] as u32)
        })
        .collect();
    let h = vec![0; n];
    let model = Model {
        h_milli: &h,
        edges: &edges,
        j_milli: &j,
    };
    let motifs = Motifs {
        tiles: &tiles,
        gauges: &gauges,
        pairs: &pairs,
    };
    let weights = Weights {
        pair_milli,
        tile_milli: 1000,
    };
    let graph = OtsukaGraph::new(&model, &motifs, weights).expect("a kernel graph");
    let betas: Vec<f64> = (0..sweeps)
        .map(|k| {
            let f = if sweeps > 1 {
                k as f64 / (sweeps - 1) as f64
            } else {
                1.0
            };
            beta_min * (beta_max / beta_min).powf(f)
        })
        .collect();
    let schedule = Schedule::new(&betas, single_every).expect("a schedule");
    let started = Instant::now();
    let mut energies = Vec::with_capacity(anneals);
    for a in 0..anneals {
        let sample = anneal(&graph, &schedule, seed + a as u64, None).expect("no cancel");
        energies.push(sample.energy_milli);
    }
    let secs = started.elapsed().as_secs_f64();
    energies.sort_unstable();
    let gaps: Vec<f64> = energies
        .iter()
        .map(|e| (e - target) as f64 / 1000.0)
        .collect();
    println!(
        "{{\"anneals\":{anneals},\"sweeps\":{sweeps},\"secs\":{secs:.3},\"sweeps_per_s\":{:.0},\"best_gap\":{:.3},\"median_gap\":{:.3},\"below\":{}}}",
        (anneals * sweeps) as f64 / secs,
        gaps[0],
        gaps[gaps.len() / 2],
        gaps.iter().filter(|g| **g < 0.0).count()
    );
}
