//! The `quip-cpu-ice-sat` sampler: an external Kissat executable solves the
//! strong layer of validated `quip/twisted-ice-dimer` revision 1 jobs, and the
//! production SA kernel serves every other job.
//!
//! A family job returns one checked witness per read, scored against the full
//! instance. A job with no checked witness is declined with `NO_SOLUTION`:
//! an UNSAT or unknown verdict, a timeout, malformed output, a witness that
//! fails the independent check, or no configured executable.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use quip_protocol::scoring::energy_milli;
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::ice::twisted_dimer::local_gauges;
use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::{
    BackendIdentity, CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult,
    StreamResult, WarmStart, WarmStreamJob,
};

use super::ice_sat::{decode, encode, strong_layer_holds, Kissat, RunError, Strong, Verdict};
use super::otsuka_sampler::{parse_pairs, section, PAIRS_SECTION};
use crate::sampler_core::{sample_ising, sample_ising_cancellable};
use crate::{check_sample_size, run_job_pump, NumCpusError, StreamKernelError};

/// The stable id of this binary (FAM-60).
pub const ICE_SAT_SOLVER_ID: &str = "quip-cpu-ice-sat/1";

/// Backend identity for `quip-cpu-ice-sat`. The limits are those of
/// `quip-cpu-sa`, because the SA kernel serves every job outside the family.
pub const CPU_ICE_SAT_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_proto::v1::Backend::Cpu,
    algorithm: quip_proto::v1::Algorithm::Sat,
    max_nodes: crate::DEFAULT_MAX_NODES,
    max_edges: crate::DEFAULT_MAX_EDGES,
    features: crate::STREAMING_LINEAR,
    adapt: crate::CPU_ADAPT,
    limits: crate::F64_LIMITS,
};

/// Every parameter set of `quip/twisted-ice-dimer` revision 1.
const TWISTED_DIMER: &[FamilyClaim] = &[FamilyClaim {
    revisions: &[TWISTED_DIMER_V1],
    only_params: &[],
}];

/// The Kissat time limit when `backend_toml` sets none, the research limit.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// The `backend_toml` keys of this binary.
#[derive(serde::Deserialize, Default)]
struct IceSatConfig {
    num_cpus: Option<i64>,
    /// An absolute path to the Kissat executable.
    kissat: Option<PathBuf>,
    /// The wall-time limit of one Kissat run, in seconds.
    kissat_timeout_s: Option<u64>,
    #[serde(flatten)]
    unknown: BTreeMap<String, toml::Value>,
}

/// The solver that `config` names, or why it names none.
fn configured_kissat(config: &IceSatConfig) -> Result<Kissat, String> {
    let path = config
        .kissat
        .clone()
        .ok_or_else(|| "backend_toml sets no `kissat` executable".to_owned())?;
    if !path.is_absolute() {
        return Err(format!(
            "`kissat` must be an absolute path, not {}",
            path.display()
        ));
    }
    let executable = std::fs::metadata(&path).is_ok_and(|meta| {
        use std::os::unix::fs::PermissionsExt;
        meta.is_file() && meta.permissions().mode() & 0o111 != 0
    });
    if !executable {
        return Err(format!(
            "`kissat` at {} is not an executable file",
            path.display()
        ));
    }
    let timeout = match config.kissat_timeout_s {
        None => DEFAULT_TIMEOUT,
        Some(0) => return Err("`kissat_timeout_s` must be positive".to_owned()),
        Some(seconds) => Duration::from_secs(seconds),
    };
    Ok(Kissat::new(path, timeout))
}

/// The CPU sampler behind `quip-cpu-ice-sat`.
#[derive(Debug, Default)]
pub struct IceSatSampler {
    /// 0 means use host parallelism.
    num_cpus: Arc<AtomicUsize>,
    /// The command-line solver settings. The `backend_toml` keys override them.
    defaults: (Option<PathBuf>, Option<u64>),
    /// The configured solver, or `None` when no usable one is named.
    kissat: Arc<RwLock<Option<Kissat>>>,
}

/// Why a family job has no witness.
enum Failure {
    /// The job is declined with this detail.
    Declined(String),
    /// The cancel token fired.
    Cancelled,
}

/// The strong layer of a validated twisted-ice-dimer instance: its tiles and
/// local gauges, and each pair with the sign of its coupling.
struct Layer {
    tiles: Vec<[u32; 4]>,
    gauges: Vec<[i8; 4]>,
    pairs: Vec<(u32, u32, i8)>,
}

impl Layer {
    fn of(instance: &Instance) -> Option<Self> {
        let (tiles, gauges) = local_gauges(instance)?;
        let pairs = parse_pairs(section(instance.metadata(), PAIRS_SECTION)?)?;
        let coupling: HashMap<(u32, u32), i32> = instance
            .edges()
            .iter()
            .zip(instance.j())
            .flat_map(|(&(u, v), &j)| [((u, v), j), ((v, u), j)])
            .collect();
        let pairs = pairs
            .into_iter()
            .map(|(u, v)| {
                let j = *coupling.get(&(u, v))?;
                Some((u, v, if j < 0 { -1 } else { 1 }))
            })
            .collect::<Option<_>>()?;
        Some(Self {
            tiles,
            gauges,
            pairs,
        })
    }

    fn strong(&self) -> Strong<'_> {
        Strong {
            tiles: &self.tiles,
            gauges: &self.gauges,
            pairs: &self.pairs,
        }
    }
}

/// Whether `family` is the revision this binary solves.
fn is_twisted_dimer(family: &FamilyContext) -> bool {
    family.family_id() == TWISTED_DIMER_V1.family_id
        && family.revision() == TWISTED_DIMER_V1.revision
}

impl IceSatSampler {
    /// A sampler whose solver defaults to `kissat` with `timeout_s`, from the
    /// command line. `backend_toml` can override both.
    #[must_use]
    pub fn new(kissat: Option<PathBuf>, timeout_s: Option<u64>) -> Self {
        let sampler = Self {
            defaults: (kissat, timeout_s),
            ..Self::default()
        };
        sampler.configure(&IceSatConfig::default());
        sampler
    }

    /// Resolve the solver from `config`, falling back to the command line.
    fn configure(&self, config: &IceSatConfig) {
        let merged = IceSatConfig {
            kissat: config.kissat.clone().or_else(|| self.defaults.0.clone()),
            kissat_timeout_s: config.kissat_timeout_s.or(self.defaults.1),
            ..IceSatConfig::default()
        };
        let kissat = match configured_kissat(&merged) {
            Ok(kissat) => {
                tracing::info!(kissat = %kissat.path().display(), "ice-sat solver configured");
                Some(kissat)
            }
            Err(why) => {
                tracing::error!(%why, "ice-sat has no usable solver; twisted-dimer jobs are declined");
                None
            }
        };
        if let Ok(mut cell) = self.kissat.write() {
            *cell = kissat;
        }
    }

    fn kissat(&self) -> Option<Kissat> {
        current(&self.kissat)
    }
}

/// The solver in `cell`, if one is configured.
fn current(cell: &RwLock<Option<Kissat>>) -> Option<Kissat> {
    cell.read().ok().and_then(|kissat| kissat.clone())
}

/// One Kissat witness per read of a twisted-ice-dimer job.
fn witnesses(
    kissat: Option<Kissat>,
    graph: &IsingGraph,
    params: &SampleParams,
    family: &FamilyContext,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, Failure> {
    let decline = |detail: String| {
        tracing::warn!(detail, "ice-sat declines the job");
        Failure::Declined(detail)
    };
    let kissat = kissat.ok_or_else(|| {
        decline("no usable Kissat executable is configured (backend_toml key `kissat`)".into())
    })?;
    let instance = family.instance();
    if instance.node_ids().len() != graph.h.len() {
        return Err(decline("the instance does not fit the job graph".into()));
    }
    let layer = Layer::of(instance).ok_or_else(|| {
        decline("the instance metadata has no readable tiles, gauges, or pairs".into())
    })?;
    reads(&kissat, &layer.strong(), graph, params, cancel).map_err(|failure| match failure {
        Failure::Declined(detail) => decline(detail),
        Failure::Cancelled => Failure::Cancelled,
    })
}

/// Why one read gave no checked witness.
fn read_failure(error: RunError) -> Failure {
    match error {
        RunError::Cancelled => Failure::Cancelled,
        other => Failure::Declined(other.to_string()),
    }
}

/// Up to `params.num_reads` checked witnesses of `strong`, one Kissat run per
/// read with seed `params.seed + r`, scored on `graph`. The first read that
/// gives no checked witness ends the loop: the reads so far are the result, and
/// the job is declined only when there are none. A cancel always wins.
fn reads(
    kissat: &Kissat,
    strong: &Strong<'_>,
    graph: &IsingGraph,
    params: &SampleParams,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, Failure> {
    let encoding = encode(strong);
    let mut found = Vec::with_capacity(params.num_reads);
    for read in 0..params.num_reads {
        let seed = params.seed.wrapping_add(read as u64);
        let outcome = kissat.run(&encoding, seed, cancel).map_err(read_failure).and_then(|run| {
            let values = match run.verdict {
                Verdict::Sat(values) => values,
                Verdict::Unsat => {
                    return Err(Failure::Declined(
                        "the SAT solver reports the strong layer unsatisfiable (not a checked proof)"
                            .into(),
                    ))
                }
                Verdict::Unknown => {
                    return Err(Failure::Declined("the SAT solver stopped without an answer".into()))
                }
            };
            let spins = decode(&encoding, &values, graph.h.len());
            let holds = strong_layer_holds(strong, &spins);
            let energy = energy_milli(&spins, &graph.h, &graph.j, &graph.edges);
            tracing::info!(
                kissat = %kissat.path().display(),
                args = ?run.args,
                exit_code = ?run.exit_code,
                seconds = run.elapsed.as_secs_f64(),
                verdict = "sat",
                strong_layer_holds = holds,
                full_energy_milli = energy,
                "ice-sat read"
            );
            if holds {
                Ok(SamplerResult {
                    spins,
                    energy_milli: energy,
                })
            } else {
                Err(Failure::Declined("the SAT witness fails the strong-layer check".into()))
            }
        });
        match outcome {
            Ok(result) => found.push(result),
            Err(Failure::Cancelled) => return Err(Failure::Cancelled),
            Err(Failure::Declined(detail)) if found.is_empty() => {
                return Err(Failure::Declined(detail))
            }
            Err(Failure::Declined(detail)) => {
                tracing::warn!(
                    detail,
                    kept = found.len(),
                    "ice-sat stops after a failed read"
                );
                break;
            }
        }
    }
    Ok(found)
}

impl Sampler for IceSatSampler {
    /// A job without family structure runs on the SA kernel.
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        check_sample_size(graph.h.len(), graph.edges.len())?;
        Ok(sample_ising(graph, params, crate::Algorithm::Sa))
    }

    fn accepts_family_metadata() -> bool {
        true
    }

    fn sample_family(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        _warm: Option<&WarmStart>,
        family: &FamilyContext,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        if !is_twisted_dimer(family) {
            return self.sample(graph, params);
        }
        witnesses(self.kissat(), graph, params, family, None).map_err(|failure| match failure {
            Failure::Declined(detail) => SampleError::Declined(detail),
            // Without a token the run never reports a cancel.
            Failure::Cancelled => SampleError::Declined("cancelled".into()),
        })
    }

    fn declaration() -> Declaration {
        Declaration {
            solver_id: ICE_SAT_SOLVER_ID,
            generated: TWISTED_DIMER,
            accepted: TWISTED_DIMER,
        }
    }

    fn stream_width(&self) -> usize {
        crate::core_budget(&self.num_cpus)
    }

    fn declared_stream_width() -> u32 {
        u32::try_from(crate::host_parallelism()).unwrap_or(u32::MAX)
    }

    fn apply_config(&self, backend_toml: &str) {
        let config: IceSatConfig = if backend_toml.trim().is_empty() {
            IceSatConfig::default()
        } else {
            match toml::from_str(backend_toml) {
                Ok(config) => config,
                Err(_) => crate::reject_num_cpus(NumCpusError::InvalidToml),
            }
        };
        crate::warn_cpu_unknown_fields(config.unknown.keys());
        match crate::check_num_cpus(config.num_cpus) {
            Ok(Some(requested)) => crate::store_num_cpus(&self.num_cpus, requested),
            Ok(None) => {}
            Err(e) => {
                tracing::error!(error = %e, "cpu num_cpus rejected");
                crate::reject_num_cpus(e);
            }
        }
        self.configure(&config);
    }

    /// Every job comes through here, because the sampler accepts family
    /// metadata. The cancel token reaches each Kissat run and each SA read.
    fn sample_stream_warm(
        &self,
        jobs: tokio::sync::mpsc::Receiver<WarmStreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        let cell = Arc::clone(&self.kissat);
        run_job_pump(
            || self.stream_width(),
            move |job: &WarmStreamJob, token: &CancelToken| {
                let (graph, params) = (&job.job.graph, &job.job.params);
                let guard = Some((token, job.job.watermark));
                if let Some(family) = job.family.as_ref().filter(|f| is_twisted_dimer(f)) {
                    return witnesses(current(&cell), graph, params, family, guard).map_err(
                        |failure| match failure {
                            Failure::Declined(detail) => {
                                StreamKernelError::Reject(SampleError::Declined(detail))
                            }
                            Failure::Cancelled => StreamKernelError::Cancelled,
                        },
                    );
                }
                check_sample_size(graph.h.len(), graph.edges.len())
                    .map_err(StreamKernelError::Reject)?;
                sample_ising_cancellable(graph, params, crate::Algorithm::Sa, guard)
                    .map_err(|_| StreamKernelError::Cancelled)
            },
            jobs,
            out,
            cancel,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{configured_kissat, reads, Failure, IceSatConfig, Layer, TWISTED_DIMER_V1};
    use crate::families::ice_sat::{Kissat, Strong};
    use quip_solver_conformance::families::{export_instances, reference_revision};
    use quip_solver_core::ising_generators::family::BaseTopology;
    use quip_solver_core::{IsingGraph, SampleParams};
    use std::collections::HashMap;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::sync::LazyLock;
    use std::time::Duration;

    /// Every tile's metadata gauge balances exactly the lowest-energy states of
    /// its six couplings, the way the research encoder reads a gauge. Every
    /// pair sign matches its coupling.
    #[test]
    fn the_metadata_gauges_agree_with_the_couplings_of_reference_instances() {
        let reference = reference_revision(TWISTED_DIMER_V1.family_id, TWISTED_DIMER_V1.revision)
            .expect("a reference revision");
        let stored = quip_solver_conformance::topology(reference.topology).unwrap();
        let topology = BaseTopology::new(stored.nodes, &stored.edges).unwrap();
        let nonces: Vec<[u8; 32]> = (0_u8..4).map(|i| [i; 32]).collect();
        let instances: Vec<_> = reference
            .params
            .iter()
            .filter_map(|params| {
                export_instances(reference.contract, params, &topology, &nonces).ok()
            })
            .flatten()
            .filter_map(Result::ok)
            .collect();
        assert!(!instances.is_empty(), "the reference gives instances");
        for instance in &instances {
            let j: HashMap<(u32, u32), i64> = instance
                .edges()
                .iter()
                .zip(instance.j())
                .flat_map(|(&(u, v), &j)| [((u, v), i64::from(j)), ((v, u), i64::from(j))])
                .collect();
            let layer = Layer::of(instance).expect("a twisted-dimer layer");
            assert!(!layer.tiles.is_empty() && !layer.pairs.is_empty());
            for (tile, gauge) in layer.tiles.iter().zip(&layer.gauges) {
                let energy = |k: u8| -> i64 {
                    let s = |p: usize| if k >> p & 1 == 1 { 1 } else { -1 };
                    let mut e = 0;
                    for a in 0..4 {
                        for b in a + 1..4 {
                            e += j[&(tile[a], tile[b])] * s(a) * s(b);
                        }
                    }
                    e
                };
                let floor = (0..16).map(energy).min().unwrap();
                for k in 0_u8..16 {
                    let balanced = (0..4)
                        .map(|p| i64::from(gauge[p]) * if k >> p & 1 == 1 { 1 } else { -1 })
                        .sum::<i64>()
                        == 0;
                    assert_eq!(energy(k) == floor, balanced, "tile {tile:?} state {k:04b}");
                }
            }
            for &(u, v, sign) in &layer.pairs {
                assert_eq!(j[&(u, v)] < 0, sign < 0, "pair ({u}, {v})");
            }
        }
    }

    fn config(kissat: Option<&str>, timeout: Option<u64>) -> IceSatConfig {
        IceSatConfig {
            kissat: kissat.map(PathBuf::from),
            kissat_timeout_s: timeout,
            ..IceSatConfig::default()
        }
    }

    /// Fake solvers, written once before any test here spawns a process.
    static FAKES: LazyLock<PathBuf> = LazyLock::new(|| {
        let dir = std::env::temp_dir().join(format!("ice-sat-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, body) in [
            // Answers on its first run, then hangs, so a later read times out.
            (
                "once",
                "f=\"$(dirname \"$0\")/once.done\"; if [ -e \"$f\" ]; then exec sleep 30; fi; \
                 touch \"$f\"; echo 's SATISFIABLE'; echo 'v -1 -2 0'; exit 10",
            ),
            ("unsat", "echo 's UNSATISFIABLE'; exit 20"),
        ] {
            let path = dir.join(name);
            std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        dir
    });

    const ONE_PAIR: Strong<'static> = Strong {
        tiles: &[],
        gauges: &[],
        pairs: &[(0, 1, -1)],
    };

    fn two_spins() -> (IsingGraph, SampleParams) {
        let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
        let params = SampleParams {
            num_reads: 3,
            ..SampleParams::default()
        };
        (graph, params)
    }

    #[test]
    fn a_later_failed_read_keeps_the_earlier_witnesses() {
        let kissat = Kissat::new(FAKES.join("once"), Duration::from_millis(300));
        let (graph, params) = two_spins();
        let found = reads(&kissat, &ONE_PAIR, &graph, &params, None)
            .unwrap_or_else(|_| unreachable!("the first read has a witness"));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].spins, vec![-1, -1]);
        assert_eq!(found[0].energy_milli, -1000);
    }

    #[test]
    fn a_job_with_no_witness_is_declined() {
        let kissat = Kissat::new(FAKES.join("unsat"), Duration::from_secs(5));
        let (graph, params) = two_spins();
        let failure = reads(&kissat, &ONE_PAIR, &graph, &params, None).err();
        assert!(
            matches!(&failure, Some(Failure::Declined(detail)) if detail.contains("unsatisfiable"))
        );
    }

    #[test]
    fn a_missing_or_unusable_kissat_is_reported() {
        assert!(configured_kissat(&config(None, None)).is_err());
        assert!(
            configured_kissat(&config(Some("kissat"), None)).is_err(),
            "relative"
        );
        assert!(configured_kissat(&config(Some("/nonexistent/kissat"), None)).is_err());
        assert!(
            configured_kissat(&config(Some("/etc/hostname"), None)).is_err(),
            "not executable"
        );
        assert!(
            configured_kissat(&config(Some("/bin/sh"), Some(0))).is_err(),
            "zero timeout"
        );
        assert!(configured_kissat(&config(Some("/bin/sh"), Some(5))).is_ok());
    }
}
