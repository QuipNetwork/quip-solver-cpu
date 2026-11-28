impl FastDimer {
    pub fn copy_probe_weights(&self, sc: &WorkerScratch, output: &mut [i32]) {
        for (i, weight) in output.iter_mut().enumerate() {
            *weight = if sc.strong_mark[i] == sc.cur_strong {
                -1000
            } else {
                i32::from(sc.weak_w[i])
            };
        }
        let gauge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        for (&tile, gauge) in sc.sel_cids.iter().zip(sc.gauges.chunks_exact(4)) {
            for (&edge, (a, b)) in self.cand_tiles[tile as usize].eids.iter().zip(gauge_pairs) {
                output[edge as usize] = 1000 * i32::from(gauge[a]) * i32::from(gauge[b]);
            }
        }
    }
}

impl WorkerScratch {
    pub fn weak_frustrations(&self, triangles: &[[usize; 3]]) -> usize {
        triangles
            .iter()
            .filter(|&&[a, b, c]| {
                [a, b, c]
                    .iter()
                    .all(|&edge| self.strong_mark[edge] != self.cur_strong)
                    && (self.weak_w[a] > 0) ^ (self.weak_w[b] > 0) ^ (self.weak_w[c] > 0)
            })
            .count()
    }
}
