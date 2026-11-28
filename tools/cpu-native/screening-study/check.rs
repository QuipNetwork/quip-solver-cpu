#![allow(dead_code)]

mod engine;
#[cfg(feature = "geometry")]
mod rank;
mod gpu_bm8 {
    // The imported engine also defines a GPU exporter. Only its CPU method
    // is exercised here, and this struct allows linking without CUDA.
    pub struct Bm8GpuInput {
        pub n_super: i32,
        pub n_tiles: i32,
        pub base_const: f32,
        pub weak_milli: f32,
        pub seed: u64,
        pub tiles: Vec<u16>,
        pub gauges: Vec<i8>,
        pub weak_off: Vec<i32>,
        pub weak_nb_packed: Vec<u16>,
    }
}

use ising_generators::descriptor::{admit, params_digest, AdvertisedLimits};
use ising_generators::family::{admit_topology, generate_instance, BaseTopology};
use ising_generators::ice::TWISTED_DIMER_V1;
use std::{fs, hint::black_box, time::Instant};

fn triangles(edges: &[(u32, u32)], nodes: usize) -> Vec<[usize; 3]> {
    let mut rows = vec![Vec::new(); nodes];
    for (edge, &(u, v)) in edges.iter().enumerate() {
        assert!(u < v);
        rows[u as usize].push((v, edge));
    }
    let mut found = Vec::new();
    for row in &rows {
        for &(v, uv) in row {
            for &(w, vw) in &rows[v as usize] {
                if let Ok(at) = row.binary_search_by_key(&w, |&(node, _)| node) {
                    found.push([uv, vw, row[at].1]);
                }
            }
        }
    }
    found
}

fn count(weights: &[i32], triangles: &[[usize; 3]]) -> usize {
    triangles
        .iter()
        .filter(|&&[a, b, c]| {
            let [a, b, c] = [weights[a], weights[b], weights[c]];
            [a, b, c].iter().all(|j| *j != 0 && j.unsigned_abs() < 1000) && a * b * c > 0
        })
        .count()
}

fn main() {
    let top = fs::read_to_string("topology.txt").unwrap();
    let mut fields = top.split_whitespace().map(|s| s.parse::<u32>().unwrap());
    let n = fields.next().unwrap() as usize;
    let m = fields.next().unwrap() as usize;
    let nodes: Vec<_> = fields.by_ref().take(n).collect();
    let edges: Vec<_> = (0..m)
        .map(|_| (fields.next().unwrap(), fields.next().unwrap()))
        .collect();
    assert!(fields.next().is_none());
    let params = fs::read("params.bin").unwrap();
    let topology = BaseTopology::new(nodes, &edges).unwrap();
    let setup = Instant::now();
    let admitted = admit(
        std::slice::from_ref(&TWISTED_DIMER_V1),
        TWISTED_DIMER_V1.family_id.as_bytes(),
        TWISTED_DIMER_V1.revision,
        &params,
        &params_digest(&params),
    )
    .unwrap();
    let bounds = admit_topology(
        &admitted,
        &topology,
        topology.order_digest(),
        &AdvertisedLimits::default(),
    )
    .unwrap();
    let reference_setup = setup.elapsed().as_secs_f64();
    let setup = Instant::now();
    let fast = engine::FastDimer::from_raw(
        topology.node_ids().to_vec(),
        topology.edges(),
        &params,
        *topology.order_digest(),
    )
    .unwrap();
    let mut scratch = fast.new_scratch();
    let fast_setup = setup.elapsed().as_secs_f64();
    let setup = Instant::now();
    let triangles = triangles(&fast.edges, n);
    let triangle_setup = setup.elapsed().as_secs_f64();
    let cases = fs::read_to_string("cases.txt").unwrap();
    let mut weights = vec![0; m];
    println!(
        "setup\t{reference_setup}\t{fast_setup}\t{triangle_setup}\t{}",
        triangles.len()
    );
    for (i, line) in cases.lines().enumerate() {
        let mut entries = line.split_whitespace();
        let seed = entries.next().unwrap().parse::<u64>().unwrap();
        let nonce_hex = entries.next().unwrap();
        let nonce =
            std::array::from_fn(|i| u8::from_str_radix(&nonce_hex[2 * i..2 * i + 2], 16).unwrap());
        let expected_triangles = entries.next().unwrap().parse::<usize>().unwrap();
        #[cfg(feature = "geometry")]
        let expected = entries.next().unwrap().parse::<f64>().unwrap().to_bits();
        #[cfg(not(feature = "geometry"))]
        let expected = expected_triangles;
        let generate = || generate_instance(&admitted, &topology, &bounds, nonce).unwrap();
        let canonical = generate();
        assert!(fast.generate_into_scratch(nonce, &mut scratch));
        fast.copy_probe_weights(&scratch, &mut weights);
        assert_eq!(canonical.edges(), fast.edges);
        assert_eq!(canonical.j(), weights, "seed {seed}");
        assert_eq!(count(canonical.j(), &triangles), expected_triangles);
        assert_eq!(scratch.weak_frustrations(&triangles), expected_triangles);
        #[cfg(feature = "geometry")]
        {
            assert!(canonical.h().iter().all(|&h| h == 0));
            assert_eq!(canonical.offset_milli(), 0);
            assert_eq!(
                rank::score(n, canonical.edges(), canonical.j(), seed),
                expected
            );
            assert_eq!(rank::score(n, &fast.edges, &weights, seed), expected);
        }
        if i % 8 != 0 {
            continue;
        }
        let mut timings = [0.0; 4];
        let order = if (i / 8) % 2 == 0 {
            [0, 2, 3, 1]
        } else {
            [2, 0, 1, 3]
        };
        for arm in order {
            let start = Instant::now();
            let scored = if arm < 2 {
                let instance = black_box(generate());
                #[cfg(feature = "geometry")]
                let score = rank::score(n, instance.edges(), instance.j(), seed);
                #[cfg(not(feature = "geometry"))]
                let score = count(instance.j(), &triangles);
                black_box(score)
            } else {
                assert!(fast.generate_into_scratch(nonce, &mut scratch));
                #[cfg(feature = "geometry")]
                let score = {
                    fast.copy_probe_weights(&scratch, &mut weights);
                    rank::score(n, &fast.edges, &weights, seed)
                };
                #[cfg(not(feature = "geometry"))]
                let score = scratch.weak_frustrations(&triangles);
                black_box(score)
            };
            timings[arm] = start.elapsed().as_secs_f64();
            assert_eq!(scored, expected);
        }
        println!(
            "pair\t{seed}\t{}\t{}\t{}\t{}",
            timings[0], timings[1], timings[2], timings[3]
        );
    }
    println!("verified\t{}", cases.lines().count());
}
