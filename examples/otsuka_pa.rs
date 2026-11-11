//! Time `anneal_population` on a twisted-ice-dimer model file, in the
//! `otsuka_bench` format: `replicas` states anneal together over one
//! geometric schedule from 1 to `beta_max`, with a Metropolis sweep after
//! every cluster sweep.
//!
//! Usage: `otsuka_pa <model> <replicas> <sweeps> <beta_max> <seed>`
#![expect(
    clippy::print_stdout,
    reason = "otsuka_pa reports one parseable JSON line on stdout"
)]
use std::collections::HashMap;
use std::time::Instant;

use quip_solver_cpu::families::otsuka::{
    anneal_population, Model, Motifs, OtsukaGraph, Schedule, Weights,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).expect("a model file");
    let arg = |i: usize| -> f64 { args[i].parse().expect("a number") };
    let (replicas, sweeps) = (arg(2) as usize, arg(3) as usize);
    let beta_max = arg(4);
    let seed = arg(5) as u64;
    let pair_milli = 250;
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
        .map(|k| beta_max.powf(k as f64 / (sweeps - 1).max(1) as f64))
        .collect();
    let schedule = Schedule::new(&betas, 1).expect("a schedule");
    let started = Instant::now();
    let samples = anneal_population(&graph, &schedule, replicas, seed, None).expect("no cancel");
    let mut finals: Vec<i64> = samples.iter().map(|s| s.energy_milli).collect();
    finals.sort_unstable();
    let best = finals[0];
    println!(
        "{{\"replicas\":{replicas},\"sweeps\":{sweeps},\"secs\":{:.1},\"best_gap\":{:.3},\"median_best_gap\":{:.3}}}",
        started.elapsed().as_secs_f64(),
        (best - target) as f64 / 1000.0,
        (finals[replicas / 2] - target) as f64 / 1000.0
    );
}
