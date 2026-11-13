//! Sequential single-flip grain annealing with the glasssa knee schedule.

use crate::sa_sampler::{coefficient_milli, read_rng, score};
use crate::sampler_core::{anneal_one_read, CpuGraph, SampleCancelled};
use crate::{CancelToken, IsingGraph, SampleParams, SamplerResult};
use quip_solver_core::config::config_override;

/// CLI defaults and coordinator overrides for `quip-cpu-sa` only.
#[derive(Debug, Clone, PartialEq, clap::Args)]
pub struct SaScheduleConfig {
    /// Annealing schedule: default or knee.
    #[arg(long, value_enum, default_value = "default")]
    pub schedule: SaSchedule,
    /// Hot beta before division by the milli-energy unit.
    #[arg(long, default_value_t = 0.5, allow_hyphen_values = true)]
    pub knee_b0: f64,
    /// Beta at the knee before division by the milli-energy unit.
    #[arg(long, default_value_t = 1.5, allow_hyphen_values = true)]
    pub knee_bk: f64,
    /// Cold beta before division by the milli-energy unit.
    #[arg(long, default_value_t = 6.0, allow_hyphen_values = true)]
    pub knee_b1: f64,
    /// Fraction of sweeps before the knee, strictly between zero and one.
    #[arg(long, default_value_t = 0.1, allow_hyphen_values = true)]
    pub knee_at: f64,
    /// Positive milli-energy unit that divides the beta values.
    #[arg(long, default_value_t = 1000.0, allow_hyphen_values = true)]
    pub knee_unit_milli: f64,
}

/// Annealing schedule used by `quip-cpu-sa`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SaSchedule {
    /// Use the existing simulated annealing sampler.
    #[default]
    Default,
    /// Use the glasssa grain annealer and knee schedule.
    Knee,
}

impl std::fmt::Display for SaSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Default => "default",
            Self::Knee => "knee",
        })
    }
}

impl Default for SaScheduleConfig {
    fn default() -> Self {
        Self {
            schedule: SaSchedule::Default,
            knee_b0: 0.5,
            knee_bk: 1.5,
            knee_b1: 6.0,
            knee_at: 0.1,
            knee_unit_milli: 1000.0,
        }
    }
}

#[derive(Default, serde::Deserialize)]
struct Overrides {
    schedule: Option<SaSchedule>,
    knee_b0: Option<f64>,
    knee_bk: Option<f64>,
    knee_b1: Option<f64>,
    knee_at: Option<f64>,
    knee_unit_milli: Option<f64>,
}

pub(crate) const CONFIG_KEYS: &[&str] = &[
    "schedule",
    "knee_b0",
    "knee_bk",
    "knee_b1",
    "knee_at",
    "knee_unit_milli",
];

impl SaScheduleConfig {
    /// Refuse invalid schedule names, non-finite values, and invalid ranges.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("knee_b0", self.knee_b0),
            ("knee_bk", self.knee_bk),
            ("knee_b1", self.knee_b1),
            ("knee_unit_milli", self.knee_unit_milli),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive, got {value}"));
            }
        }
        if !self.knee_at.is_finite() || self.knee_at <= 0.0 || self.knee_at >= 1.0 {
            return Err(format!(
                "knee_at must be strictly between zero and one, got {}",
                self.knee_at
            ));
        }
        Ok(())
    }

    pub(crate) fn resolve(&self, backend_toml: &str) -> Result<Self, String> {
        let cfg: Overrides = toml::from_str(backend_toml).map_err(|e| e.to_string())?;
        let resolved = Self {
            schedule: config_override("schedule", self.schedule, cfg.schedule),
            knee_b0: config_override("knee_b0", self.knee_b0, cfg.knee_b0),
            knee_bk: config_override("knee_bk", self.knee_bk, cfg.knee_bk),
            knee_b1: config_override("knee_b1", self.knee_b1, cfg.knee_b1),
            knee_at: config_override("knee_at", self.knee_at, cfg.knee_at),
            knee_unit_milli: config_override(
                "knee_unit_milli",
                self.knee_unit_milli,
                cfg.knee_unit_milli,
            ),
        };
        resolved.validate()?;
        Ok(resolved)
    }

    fn beta_at(&self, f: f64) -> f64 {
        let b = if f < self.knee_at {
            self.knee_b0 * (self.knee_bk / self.knee_b0).powf(f / self.knee_at)
        } else {
            self.knee_bk
                * (self.knee_b1 / self.knee_bk)
                    .powf((f - self.knee_at) / (1.0 - self.knee_at).max(1e-12))
        };
        b / self.knee_unit_milli
    }

    fn wire_betas(&self, sweeps: usize) -> Vec<f64> {
        (0..sweeps)
            .map(|k| self.beta_at(k as f64 / sweeps.saturating_sub(1).max(1) as f64) * 1000.0)
            .collect()
    }
}

#[expect(
    clippy::exit,
    clippy::print_stderr,
    reason = "invalid schedule settings must refuse the handshake with exit 64"
)]
pub(crate) fn reject_config(error: &str) -> ! {
    eprintln!("configuration error: {error}");
    std::process::exit(64);
}

struct Model {
    n: usize,
    start: Vec<u32>,
    nbr: Vec<u32>,
    /// Couplings and biases in units of grain / 2.
    j: Vec<i32>,
    h: Vec<i32>,
    grain: i64,
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

fn exact_milli(c: f64) -> Option<i32> {
    let milli = coefficient_milli(c)?;
    if milli as f64 / 1000.0 != c {
        return None;
    }
    i32::try_from(milli).ok()
}

impl Model {
    fn from_graph(graph: &IsingGraph) -> Option<Self> {
        let n = graph.h.len();
        u32::try_from(n).ok()?;
        let mut h: Vec<i32> = graph
            .h
            .iter()
            .map(|&c| exact_milli(c))
            .collect::<Option<_>>()?;
        let weights: Vec<i32> = graph
            .j
            .iter()
            .map(|&c| exact_milli(c))
            .collect::<Option<_>>()?;
        let mut grain = 0;
        for &c in h.iter().chain(&weights) {
            grain = gcd(grain, 2 * i64::from(c));
        }
        grain = grain.max(1);
        // An all-zero graph has grain 1 and no nonzero coefficient to divide.
        let half = (grain / 2).max(1);
        let mut start = vec![0u32; n + 1];
        for &(u, v) in graph.edges.iter() {
            if u < n && v < n && u != v {
                start[u + 1] = start[u + 1].checked_add(1)?;
                start[v + 1] = start[v + 1].checked_add(1)?;
            }
        }
        for i in 0..n {
            start[i + 1] = start[i + 1].checked_add(start[i])?;
        }
        let mut fill = start.clone();
        let mut nbr = vec![0; start[n] as usize];
        let mut j = vec![0; start[n] as usize];
        for (k, &(u, v)) in graph.edges.iter().enumerate() {
            if u >= n || v >= n || u == v {
                continue;
            }
            let w = i32::try_from(i64::from(weights.get(k).copied().unwrap_or(0)) / half).ok()?;
            for (src, dst) in [(u, v), (v, u)] {
                let slot = fill[src] as usize;
                nbr[slot] = dst as u32;
                j[slot] = w;
                fill[src] += 1;
            }
        }
        for i in 0..n {
            h[i] = i32::try_from(i64::from(h[i]) / half).ok()?;
            let mut bound = i64::from(h[i]).abs();
            for &w in &j[start[i] as usize..start[i + 1] as usize] {
                bound = bound.checked_add(i64::from(w).abs())?;
            }
            // Bounds both the local field and each signed 2*J update.
            if bound > i64::from(i32::MAX / 2) {
                return None;
            }
        }
        Some(Self {
            n,
            start,
            nbr,
            j,
            h,
            grain,
        })
    }
}

struct Rng(u64, u64, u64, u64);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self(next(), next(), next(), next())
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let r = (self.0.wrapping_add(self.3))
            .rotate_left(23)
            .wrapping_add(self.0);
        let t = self.1 << 17;
        self.2 ^= self.0;
        self.3 ^= self.1;
        self.1 ^= self.2;
        self.0 ^= self.3;
        self.2 ^= t;
        self.3 = self.3.rotate_left(45);
        r
    }
}

/// One anneal on the full model.
struct Anneal<'a> {
    m: &'a Model,
    s: Vec<i8>,
    /// Local field in units of grain/2, so the flip cost in grains is -s * g.
    g: Vec<i32>,
    rng: Rng,
    table: Vec<u32>,
    table_beta: f64,
}

impl<'a> Anneal<'a> {
    fn new(m: &'a Model, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let s: Vec<i8> = (0..m.n)
            .map(|_| if rng.next_u64() & 1 == 0 { 1 } else { -1 })
            .collect();
        let mut a = Self {
            m,
            s,
            g: vec![0; m.n],
            rng,
            table: Vec::new(),
            table_beta: -1.0,
        };
        a.refresh();
        a
    }

    fn refresh(&mut self) {
        let m = self.m;
        for i in 0..m.n {
            let mut acc = m.h[i];
            for k in m.start[i] as usize..m.start[i + 1] as usize {
                acc += m.j[k] * self.s[m.nbr[k] as usize] as i32;
            }
            self.g[i] = acc;
        }
    }

    /// Acceptance thresholds for costs of 1, 2, ... grains, out to probability 2^-32.
    fn set_beta(&mut self, beta: f64) {
        if (beta - self.table_beta).abs() <= 1e-3 * beta {
            return;
        }
        self.table_beta = beta;
        let r = (-beta * self.m.grain as f64).exp();
        let len = ((22.2 / (beta * self.m.grain as f64)).ceil() as usize).clamp(1, 1 << 20);
        self.table.clear();
        self.table.push(u32::MAX);
        let mut p = 1.0f64;
        for _ in 1..len {
            p *= r;
            self.table
                .push((p * 4_294_967_296.0).min(4_294_967_295.0) as u32);
        }
    }

    #[inline]
    fn flip(&mut self, i: usize) {
        let m = self.m;
        let si = -self.s[i];
        self.s[i] = si;
        let two_s = 2 * si as i32;
        let (lo, hi) = (m.start[i] as usize, m.start[i + 1] as usize);
        for (&d, &w) in m.nbr[lo..hi].iter().zip(&m.j[lo..hi]) {
            self.g[d as usize] += two_s * w;
        }
    }

    fn sweep(&mut self) {
        for i in 0..self.m.n {
            let l = -(self.s[i] as i32) * self.g[i];
            let ok = l <= 0
                || ((l as usize) < self.table.len()
                    && ((self.rng.next_u64() >> 32) as u32) < self.table[l as usize]);
            if ok {
                self.flip(i);
            }
        }
    }

    fn descend(
        &mut self,
        cancel: Option<(&CancelToken, Option<u64>)>,
    ) -> Result<(), SampleCancelled> {
        loop {
            checkpoint(cancel)?;
            let mut moved = false;
            for i in 0..self.m.n {
                if -(self.s[i] as i32) * self.g[i] < 0 {
                    self.flip(i);
                    moved = true;
                }
            }
            if !moved {
                return Ok(());
            }
        }
    }
}

fn checkpoint(cancel: Option<(&CancelToken, Option<u64>)>) -> Result<(), SampleCancelled> {
    if let Some((token, watermark)) = cancel {
        if token.is_cancelled(watermark) {
            return Err(SampleCancelled);
        }
    }
    Ok(())
}

pub(crate) fn sample(
    graph: &IsingGraph,
    params: &SampleParams,
    config: &SaScheduleConfig,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, SampleCancelled> {
    checkpoint(cancel)?;
    let model = Model::from_graph(graph);
    let fallback = model.is_none().then(|| {
        (
            CpuGraph::from_base(graph),
            config.wire_betas(params.num_sweeps),
        )
    });
    let run = |read: usize| {
        checkpoint(cancel)?;
        let spins = if let Some(m) = &model {
            let seed = params
                .seed
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add((read as u64).wrapping_mul(7919).wrapping_add(1));
            let mut a = Anneal::new(m, seed);
            for k in 0..params.num_sweeps {
                checkpoint(cancel)?;
                a.set_beta(
                    config.beta_at(k as f64 / params.num_sweeps.saturating_sub(1).max(1) as f64),
                );
                a.sweep();
            }
            a.descend(cancel)?;
            a.s
        } else {
            let (cpu, betas) = fallback
                .as_ref()
                .expect("fallback exists without an integer model");
            let mut rng = read_rng(params.seed, read);
            anneal_one_read(cpu, betas, 1, &mut rng, cancel)?
        };
        Ok(score(&spins, graph))
    };
    let reads = params.num_reads.max(1);
    (0..reads).map(run).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_protocol::scoring::energy_milli;

    fn graph() -> IsingGraph {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(29);
        let n = 32;
        let mut edges = Vec::new();
        let mut j = Vec::new();
        for u in 0..n {
            for v in u + 1..n {
                if rng.gen_bool(0.25) {
                    edges.push((u, v));
                    j.push(f64::from(rng.gen_range(-1000..=1000)) / 1000.0);
                }
            }
        }
        IsingGraph::new(vec![0.031; n], j, edges)
    }

    #[test]
    fn beta_matches_research_at_boundaries_and_midpoints() {
        let a = SaScheduleConfig::default();
        for f in [0.0, a.knee_at, 1.0, 0.05, 0.55] {
            let b = if f < a.knee_at {
                a.knee_b0 * (a.knee_bk / a.knee_b0).powf(f / a.knee_at)
            } else {
                a.knee_bk
                    * (a.knee_b1 / a.knee_bk).powf((f - a.knee_at) / (1.0 - a.knee_at).max(1e-12))
            };
            assert_eq!(a.beta_at(f), b / a.knee_unit_milli);
        }
    }

    #[test]
    fn thresholds_follow_grain_costs_and_rebuild_tolerance() {
        let m = Model::from_graph(&IsingGraph::new(vec![0.031], vec![], vec![])).unwrap();
        let mut a = Anneal::new(&m, 9);
        let beta = 0.1;
        a.set_beta(beta);
        assert_eq!(
            a.table.len(),
            (22.2 / (beta * m.grain as f64)).ceil() as usize
        );
        assert_eq!(a.table[0], u32::MAX);
        for (k, &value) in a.table.iter().enumerate().skip(1) {
            let expected = (-beta * m.grain as f64 * k as f64).exp() * 4_294_967_296.0;
            assert!((f64::from(value) - expected).abs() <= 1.00001);
        }
        a.set_beta(beta * 1.0005);
        assert_eq!(a.table_beta, beta);
        a.set_beta(beta * 1.002);
        assert_eq!(a.table_beta, beta * 1.002);
        a.set_beta(1e-20);
        assert_eq!(a.table.len(), 1 << 20);
        a.set_beta(100.0);
        assert_eq!(a.table.len(), 1);
    }

    #[test]
    fn grain_and_checked_storage_preserve_coefficients() {
        let g = IsingGraph::new(
            vec![0.5, 1.0, 0.031, -0.031],
            vec![0.25; 3],
            vec![(0, 1), (1, 2), (2, 3)],
        );
        assert_eq!(Model::from_graph(&g).unwrap().grain, 2);
        let wide = IsingGraph::new(vec![0.001, 0.0], vec![100.0], vec![(0, 1)]);
        assert_eq!(Model::from_graph(&wide).unwrap().j[0], 100_000);
        let mut large = IsingGraph::new(vec![0.0; 65_537], vec![100.0], vec![(0, 65_536)]);
        large.h[0] = 0.001;
        let model = Model::from_graph(&large).unwrap();
        assert_eq!(model.nbr, vec![65_536, 0]);
        assert_eq!(model.j, vec![100_000, 100_000]);
        let overflow = IsingGraph::new(vec![0.001, 0.0], vec![2_147_483.647], vec![(0, 1)]);
        assert!(Model::from_graph(&overflow).is_none());
        for h in [0.0001, f64::INFINITY, 1e30] {
            assert!(Model::from_graph(&IsingGraph::new(vec![h], vec![], vec![])).is_none());
        }
    }

    #[test]
    fn seeded_reads_are_repeatable_scored_and_locally_minimal() {
        let g = graph();
        let config = SaScheduleConfig::default();
        let p = SampleParams {
            num_reads: 8,
            num_sweeps: 32,
            seed: 41,
            ..Default::default()
        };
        let first = sample(&g, &p, &config, None).unwrap();
        let repeated = sample(&g, &p, &config, None).unwrap();
        assert_eq!(first, repeated);
        let different = sample(&g, &SampleParams { seed: 42, ..p }, &config, None).unwrap();
        assert_ne!(first, different);
        for r in first {
            assert_eq!(r.energy_milli, energy_milli(&r.spins, &g.h, &g.j, &g.edges));
            for i in 0..r.spins.len() {
                let mut flipped = r.spins.clone();
                flipped[i] = -flipped[i];
                assert!(energy_milli(&flipped, &g.h, &g.j, &g.edges) >= r.energy_milli);
            }
        }
    }

    #[test]
    fn fallback_uses_wire_betas_and_scores_results() {
        let g = IsingGraph::new(vec![0.0001, -0.0002], vec![0.33333], vec![(0, 1)]);
        assert!(Model::from_graph(&g).is_none());
        let config = SaScheduleConfig::default();
        let p = SampleParams {
            num_reads: 3,
            num_sweeps: 8,
            ..Default::default()
        };
        let got = sample(&g, &p, &config, None).unwrap();
        assert_eq!(got.len(), 3);
        for r in got {
            assert_eq!(r.energy_milli, energy_milli(&r.spins, &g.h, &g.j, &g.edges));
        }
        assert_eq!(config.wire_betas(2), vec![0.5, 6.0]);
    }

    #[test]
    fn fallback_matches_existing_f64_kernel_with_knee_ladder() {
        let mut g = graph();
        g.h[0] = 0.0311;
        let config = SaScheduleConfig {
            knee_b0: 0.001,
            knee_bk: 0.002,
            knee_b1: 0.003,
            ..Default::default()
        };
        let p = SampleParams {
            num_sweeps: 2,
            num_reads: 1,
            seed: 83,
            ..Default::default()
        };
        let cpu = CpuGraph::from_base(&g);
        let mut rng = read_rng(p.seed, 0);
        let expected = anneal_one_read(&cpu, &[0.001, 0.003], 1, &mut rng, None).unwrap();
        assert_eq!(
            sample(&g, &p, &config, None).unwrap(),
            vec![score(&expected, &g)]
        );
    }

    #[test]
    fn handshake_rejects_invalid_backend_config() {
        use crate::{Algorithm, CpuSampler, Sampler};
        const KEY: &str = "QUIP_KNEE_TEST_INVALID_CONFIG";
        if let Ok(bad) = std::env::var(KEY) {
            CpuSampler::new(Algorithm::Sa)
                .with_sa_config(SaScheduleConfig::default())
                .apply_config(&bad);
            return;
        }
        for bad in [
            "schedule='unknown'",
            "knee_b0=0",
            "knee_bk=-1",
            "knee_b1=nan",
            "knee_at=1",
            "knee_unit_milli=0",
        ] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "knee_sa::tests::handshake_rejects_invalid_backend_config",
                ])
                .env(KEY, bad)
                .output()
                .unwrap();
            assert_eq!(
                output.status.code(),
                Some(64),
                "{bad}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn zero_graph_and_cancel_are_safe() {
        let g = IsingGraph::new(vec![0.0; 4], vec![], vec![]);
        assert_eq!(Model::from_graph(&g).unwrap().grain, 1);
        let p = SampleParams {
            num_sweeps: 0,
            num_reads: 1,
            ..Default::default()
        };
        assert_eq!(
            sample(&g, &p, &SaScheduleConfig::default(), None)
                .unwrap()
                .len(),
            1
        );
        let cancel = CancelToken::default();
        cancel.cancel_through(9);
        assert_eq!(
            sample(
                &g,
                &p,
                &SaScheduleConfig::default(),
                Some((&cancel, Some(9)))
            ),
            Err(SampleCancelled)
        );
    }

    #[test]
    fn backend_config_overrides_cli_and_rejects_invalid_values() {
        let cli = SaScheduleConfig::default();
        let got = cli.resolve("schedule='knee'\nknee_b0=0.1\nknee_bk=0.4\nknee_b1=2.5\nknee_at=0.2\nknee_unit_milli=62").unwrap();
        assert_eq!(got.schedule, SaSchedule::Knee);
        assert_eq!(
            (
                got.knee_b0,
                got.knee_bk,
                got.knee_b1,
                got.knee_at,
                got.knee_unit_milli
            ),
            (0.1, 0.4, 2.5, 0.2, 62.0)
        );
        for bad in [
            "schedule='typo'",
            "knee_b0=0",
            "knee_bk=-1",
            "knee_b1=nan",
            "knee_at=0",
            "knee_at=1",
            "knee_unit_milli=0",
            "knee_unit_milli=inf",
            "knee_b0='bad'",
        ] {
            assert!(cli.resolve(bad).is_err(), "{bad}");
        }
        let invalid_cli = SaScheduleConfig {
            knee_b0: -1.0,
            ..cli
        };
        assert!(invalid_cli.resolve("").is_err());
        assert!(invalid_cli.resolve("knee_b0=0.5").is_ok());
    }

    #[tokio::test]
    async fn configured_knee_stream_matches_direct_sampling() {
        use crate::{Algorithm, CpuSampler, Sampler, StreamJob, StreamOutcome};
        let sampler = CpuSampler::new(Algorithm::Sa).with_sa_config(SaScheduleConfig::default());
        sampler.apply_config("schedule='knee'\nnum_cpus=2\nknee_b0=0.1\nknee_bk=0.4\nknee_b1=2.5\nknee_unit_milli=62");
        assert_eq!(sampler.stream_width(), crate::host_parallelism().min(2));
        let g = graph();
        let p = SampleParams {
            num_reads: 4,
            num_sweeps: 32,
            seed: 42,
            ..Default::default()
        };
        let expected = sampler.sample(&g, &p).unwrap();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let (out, mut results) = tokio::sync::mpsc::channel(1);
        tx.send(StreamJob {
            job_id: vec![1],
            graph: g,
            params: p,
            watermark: None,
        })
        .await
        .unwrap();
        drop(tx);
        let pump = tokio::task::spawn_blocking(move || {
            sampler.sample_stream(rx, out, CancelToken::default())
        });
        let result = tokio::time::timeout(std::time::Duration::from_secs(30), results.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.job_id, vec![1]);
        if let StreamOutcome::Completed(got) = result.outcome {
            assert_eq!(got.unwrap(), expected);
        } else {
            assert_eq!("cancelled", "completed");
        }
        pump.await.unwrap();
    }

    #[test]
    fn default_selection_preserves_existing_sampler() {
        use crate::{Algorithm, CpuSampler, Sampler};
        let g = graph();
        let p = SampleParams {
            num_reads: 3,
            num_sweeps: 16,
            ..Default::default()
        };
        let plain = CpuSampler::new(Algorithm::Sa);
        let configured = CpuSampler::new(Algorithm::Sa).with_sa_config(SaScheduleConfig {
            schedule: SaSchedule::Knee,
            ..Default::default()
        });
        configured.apply_config("schedule='default'");
        assert_eq!(
            plain.sample(&g, &p).unwrap(),
            configured.sample(&g, &p).unwrap()
        );
        assert_eq!(plain.stream_width(), configured.stream_width());
    }
}
