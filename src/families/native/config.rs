/// Bounded CPU budgets, in backend TOML or `QUIP_CPU_NATIVE_*` variables.
#[derive(Clone, Debug)]
pub struct NativeConfig {
    /// Override replicas for a generated family lease.
    pub final_reads: Option<usize>,
    /// Override sweeps for a generated family lease.
    pub final_sweeps: Option<usize>,
    /// Tile strength assigned to auxiliary cluster bonds, in milli.
    pub tile_milli: i32,
    /// Dimer strength assigned to auxiliary cluster bonds, in milli.
    pub pair_milli: i32,
    /// Heuristic multiplier of weak couplings during annealing only.
    /// Internal i32 milli values truncate once; proof scoring is unscaled.
    pub weak_scale: f64,
    /// Population resampling checkpoints.
    pub resamples: usize,
    /// Houdayer cadence; zero means checkpoints only.
    pub houdayer_every: usize,
    /// Cold cluster continuation sweeps on the four best reads.
    pub polish_sweeps: usize,
    /// Metropolis cadence during the population phase. Zero uses clusters alone.
    pub single_every: usize,
    /// Enable conditional tree descent on the completed candidate batch.
    #[cfg(feature = "tree-polish")]
    pub tree_polish: bool,
    /// Finish the best reads with downhill, pair-aware collective moves.
    #[cfg(feature = "pair-worm-polish")]
    pub pair_worm_polish: bool,
    /// Four-spine conditional moves per candidate. Zero disables this pass.
    #[cfg(feature = "zephyr-spine-polish")]
    pub zephyr_blocks: usize,
    /// Share temperature-local probabilities across replicas.
    pub shared_cache: bool,
    /// Positive half-delta capacity of the shared probability table.
    pub cache_slots: usize,
    /// Copy parent local fields at population resampling checkpoints.
    pub copy_resampling: bool,
    /// Inline the monotone union-find path.
    pub inline_union: bool,
    /// Fuse full-field and residual-delta adjacency updates.
    pub fused_updates: bool,
    /// Preserve original-energy record candidates outside replica ancestry.
    pub archive: bool,
    /// Use exact motif-domain energy conversion for archive evaluation.
    pub motif_score: bool,
    /// Optional final beta measured in units of the biased weak coupling.
    pub weak_beta_end: Option<f64>,
    /// Inverse temperature of original-objective cluster polishing.
    pub polish_beta: f64,
}

impl Default for NativeConfig {
    fn default() -> Self {
        Self {
            final_reads: None,
            final_sweeps: None,
            tile_milli: 250,
            pair_milli: 250,
            weak_scale: 1.5,
            resamples: 512,
            houdayer_every: 0,
            polish_sweeps: 256,
            single_every: 1,
            #[cfg(feature = "tree-polish")]
            tree_polish: false,
            #[cfg(feature = "pair-worm-polish")]
            pair_worm_polish: false,
            #[cfg(feature = "zephyr-spine-polish")]
            zephyr_blocks: 0,
            shared_cache: false,
            cache_slots: 8192,
            copy_resampling: false,
            inline_union: false,
            fused_updates: false,
            archive: false,
            motif_score: true,
            weak_beta_end: None,
            polish_beta: 20.0,
        }
    }
}

impl NativeConfig {
    fn read(table: &toml::Table, key: &str, default: usize, max: usize) -> Result<usize, String> {
        let Some(value) = table.get(key) else {
            return Ok(default);
        };
        let number = value.as_integer().and_then(|n| usize::try_from(n).ok());
        number
            .filter(|n| (1..=max).contains(n))
            .ok_or_else(|| format!("{key} must be in 1..={max}"))
    }

    pub(super) fn apply(&mut self, table: &toml::Table) -> Result<(), String> {
        for (key, value) in [
            ("native_shared_cache", &mut self.shared_cache),
            ("native_copy_resampling", &mut self.copy_resampling),
            ("native_inline_union", &mut self.inline_union),
            ("native_fused_updates", &mut self.fused_updates),
            ("native_archive", &mut self.archive),
            ("native_motif_score", &mut self.motif_score),
        ] {
            if let Some(v) = table.get(key) {
                *value = v
                    .as_bool()
                    .ok_or_else(|| format!("{key} must be boolean"))?;
            }
        }
        self.cache_slots = Self::read(table, "native_cache_slots", self.cache_slots, 65_536)?;
        if let Some(v) = table.get("native_weak_beta_end") {
            self.weak_beta_end = Some(
                v.as_float()
                    .or_else(|| v.as_integer().map(|n| n as f64))
                    .filter(|n| n.is_finite() && (0.1..=8.0).contains(n))
                    .ok_or("native_weak_beta_end must be in 0.1..=8")?,
            );
        }
        if let Some(v) = table.get("native_polish_beta") {
            self.polish_beta = v
                .as_float()
                .or_else(|| v.as_integer().map(|n| n as f64))
                .filter(|n| n.is_finite() && (1.0..=256.0).contains(n))
                .ok_or("native_polish_beta must be in 1..=256")?;
        }
        if table.contains_key("native_final_reads") {
            self.final_reads = Some(Self::read(table, "native_final_reads", 16, 128)?);
        }
        if table.contains_key("native_final_sweeps") {
            self.final_sweeps = Some(Self::read(table, "native_final_sweeps", 16_384, 262_144)?);
        }
        self.tile_milli = i32::try_from(Self::read(
            table,
            "native_tile_milli",
            self.tile_milli as usize,
            1000,
        )?)
        .map_err(|e| e.to_string())?;
        self.pair_milli = i32::try_from(Self::read(
            table,
            "native_pair_milli",
            self.pair_milli as usize,
            1000,
        )?)
        .map_err(|e| e.to_string())?;
        self.resamples = Self::read(table, "native_resamples", self.resamples, 4096)?;
        if let Some(v) = table.get("native_houdayer_every") {
            self.houdayer_every = v
                .as_integer()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n <= 262_144)
                .ok_or("invalid native_houdayer_every")?;
        }
        if let Some(v) = table.get("native_polish_sweeps") {
            self.polish_sweeps = v
                .as_integer()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n <= 8192)
                .ok_or("invalid native_polish_sweeps")?;
        }
        if let Some(v) = table.get("native_single_every") {
            self.single_every = v
                .as_integer()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n <= 262_144)
                .ok_or("invalid native_single_every")?;
        }
        #[cfg(feature = "tree-polish")]
        if let Some(v) = table.get("native_tree_polish") {
            self.tree_polish = v.as_bool().ok_or("native_tree_polish must be a boolean")?;
        }
        #[cfg(feature = "pair-worm-polish")]
        if let Some(v) = table.get("native_pair_worm_polish") {
            self.pair_worm_polish = v
                .as_bool()
                .ok_or("native_pair_worm_polish must be a boolean")?;
        }
        #[cfg(feature = "zephyr-spine-polish")]
        if let Some(v) = table.get("native_zephyr_blocks") {
            self.zephyr_blocks = v
                .as_integer()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n <= 200)
                .ok_or("native_zephyr_blocks must be in 0..=200")?;
        }
        if let Some(v) = table.get("native_weak_scale") {
            self.weak_scale = v
                .as_float()
                .or_else(|| v.as_integer().map(|n| n as f64))
                .filter(|n| n.is_finite() && (1.0..=1.5).contains(n))
                .ok_or("native_weak_scale must be in 1..=1.5")?;
        }
        Ok(())
    }

    pub(super) fn from_environment() -> Self {
        let defaults = Self {
            shared_cache: true,
            copy_resampling: true,
            inline_union: true,
            fused_updates: true,
            archive: true,
            ..Self::default()
        };
        let mut config = defaults.clone();
        let mut table = toml::Table::new();
        for key in [
            "final_reads",
            "final_sweeps",
            "tile_milli",
            "pair_milli",
            "weak_scale",
            "resamples",
            "houdayer_every",
            "polish_sweeps",
            "single_every",
            #[cfg(feature = "tree-polish")]
            "tree_polish",
            #[cfg(feature = "pair-worm-polish")]
            "pair_worm_polish",
            #[cfg(feature = "zephyr-spine-polish")]
            "zephyr_blocks",
            "shared_cache",
            "cache_slots",
            "copy_resampling",
            "inline_union",
            "fused_updates",
            "archive",
            "motif_score",
            "weak_beta_end",
            "polish_beta",
        ] {
            let Ok(raw) = std::env::var(format!("QUIP_CPU_NATIVE_{}", key.to_uppercase())) else {
                continue;
            };
            let value = match key {
                #[cfg(feature = "tree-polish")]
                "tree_polish" => raw.parse::<bool>().ok().map(toml::Value::Boolean),
                #[cfg(feature = "pair-worm-polish")]
                "pair_worm_polish" => raw.parse::<bool>().ok().map(toml::Value::Boolean),
                "shared_cache" | "copy_resampling" | "inline_union" | "fused_updates"
                | "archive" | "motif_score" => raw.parse::<bool>().ok().map(toml::Value::Boolean),
                "weak_scale" | "weak_beta_end" | "polish_beta" => {
                    raw.parse::<f64>().ok().map(toml::Value::Float)
                }
                _ => raw.parse::<i64>().ok().map(toml::Value::Integer),
            };
            if let Some(value) = value {
                table.insert(format!("native_{key}"), value);
            }
        }
        if let Err(error) = config.apply(&table) {
            tracing::error!(%error, "invalid CPU-native environment; using defaults");
            return defaults;
        }
        config
    }
}
