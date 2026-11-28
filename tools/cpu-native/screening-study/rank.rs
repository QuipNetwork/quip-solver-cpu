use rand::{rngs::SmallRng, Rng, SeedableRng};
const SWEEPS: usize = 24;
type Edge = (u32, i32);

fn adjacency(n: usize, edges: &[(u32, u32)], j: &[i32]) -> Vec<Vec<Edge>> {
    let mut rows = vec![Vec::new(); n];
    for (&(u, v), &weight) in edges.iter().zip(j) {
        rows[u as usize].push((v, weight));
        rows[v as usize].push((u, weight));
    }
    for row in &mut rows {
        row.sort_unstable_by_key(|&(v, _)| v);
    }
    rows
}

fn weak(j: i32) -> bool {
    j != 0 && j.unsigned_abs() < 1000
}

fn weight(j: i32, weak_scale: f64) -> f64 {
    f64::from(j) * if weak(j) { weak_scale } else { 1.0 }
}

fn dot<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    a.iter().zip(b).map(|(u, v)| u * v).sum()
}

fn vector_energy<const D: usize>(vectors: &[[f64; D]], rows: &[Vec<Edge>], weak_scale: f64) -> f64 {
    rows.iter()
        .enumerate()
        .map(|(u, row)| {
            row.iter()
                .filter(|&&(v, _)| v as usize > u)
                .map(|&(v, j)| weight(j, weak_scale) * dot(&vectors[u], &vectors[v as usize]))
                .sum::<f64>()
        })
        .sum()
}

fn gaussian<const D: usize>(rng: &mut SmallRng) -> [f64; D] {
    let mut values = [0.0; D];
    for chunk in values.chunks_mut(2) {
        let radius = (-2.0 * (1.0 - rng.gen::<f64>()).ln()).sqrt();
        let angle = std::f64::consts::TAU * rng.gen::<f64>();
        let (sin, cos) = angle.sin_cos();
        chunk[0] = radius * cos;
        if chunk.len() == 2 {
            chunk[1] = radius * sin;
        }
    }
    values
}

fn initialize<const D: usize>(n: usize, rng: &mut SmallRng) -> Vec<[f64; D]> {
    (0..n)
        .map(|_| {
            let mut node = gaussian::<D>(rng);
            let norm = dot(&node, &node).sqrt();
            if norm > 0.0 {
                for x in &mut node {
                    *x /= norm;
                }
            } else {
                node[0] = 1.0;
            }
            node
        })
        .collect()
}

fn sweep<const D: usize>(vectors: &mut [[f64; D]], rows: &[Vec<Edge>], weak_scale: f64) {
    for (u, row) in rows.iter().enumerate() {
        let mut field = [0.0; D];
        for &(v, j) in row {
            let j = weight(j, weak_scale);
            for (f, x) in field.iter_mut().zip(vectors[v as usize]) {
                *f += j * x;
            }
        }
        let norm = dot(&field, &field).sqrt();
        if norm > 0.0 {
            vectors[u] = field.map(|value| -value / norm);
        }
    }
}

pub fn score(n: usize, edges: &[(u32, u32)], j: &[i32], seed: u64) -> u64 {
    let rows = adjacency(n, edges, j);
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x424d_5f52_414e_4b31);
    let mut vectors = initialize::<4>(n, &mut rng);
    for _ in 0..SWEEPS {
        sweep(&mut vectors, &rows, 3.0);
    }
    vector_energy(&vectors, &rows, 1.0).to_bits()
}
