//! The `quip-cpu-dipolar` sampler: the `dipolar-strings/1` search on
//! `quip/defects-static` revision 1 jobs.
//!
//! The binary generates and accepts revision 1, the sides 2 and 3 that the
//! tests cover. A family job without states runs the search mode: each read
//! starts from the reference state with its labels fixed by paths. A job with
//! states runs the postprocess mode: each supplied state is repaired when it
//! is infeasible, then searched. Every read returns its final state, rescored
//! from the job graph and compared with the incremental score.
//!
//! The work is fixed by default: `num_sweeps` proposals per link for each
//! read. The `backend_toml` key `budget_seconds` sets a declared budget of
//! elapsed time per read instead. The key `evidence = "physical"` asks for a
//! physical claim, and every family job is then declined, because the table
//! of revision 1 has the oracle calibration.
//!
//! Every other job is solved by the generic simulated-annealing sampler that
//! `quip-cpu-sa` uses, as in `quip-cpu-flow` and `quip-cpu-flow-loops`. A plain
//! model, and every family the binary does not list, reach the sampler with no
//! D-static structure.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, RwLock};

use quip_protocol::scoring::energy_milli;
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::defects::DEFECTS_V1;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::{
    BackendIdentity, CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult,
    StreamResult, WarmStart, WarmStreamJob,
};

use super::dipolar_search::{DipolarError, Evidence, Problem, RunOutcome, Work};
use crate::postprocess::sampler::{into_result, Failure};
use crate::sampler_core::sample_generic_sa;
use crate::{check_sample_size, run_job_pump, NumCpusError, StreamKernelError};

/// The stable id of this binary (FAM-60).
pub const DIPOLAR_SOLVER_ID: &str = "quip-cpu-dipolar/1";

/// Backend identity for `quip-cpu-dipolar`. The session adds the
/// `initial-spins` feature, because the sampler accepts warm starts.
pub const CPU_DIPOLAR_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_proto::v1::Backend::Cpu,
    algorithm: quip_proto::v1::Algorithm::Dipolar,
    max_nodes: crate::DEFAULT_MAX_NODES,
    max_edges: crate::DEFAULT_MAX_EDGES,
    features: &["streaming"],
    adapt: crate::CPU_ADAPT,
    limits: crate::F64_LIMITS,
};

/// Every parameter set of `quip/defects-static` revision 1. The schema
/// bounds the side to 2 and 3, the sides the tests cover.
const DEFECTS: &[FamilyClaim] = &[FamilyClaim {
    revisions: &[DEFECTS_V1],
    only_params: &[],
}];

/// The `backend_toml` keys of this binary.
#[derive(serde::Deserialize, Default)]
struct DipolarToml {
    num_cpus: Option<i64>,
    /// `analytic` (the default) or `physical`.
    evidence: Option<String>,
    /// The declared elapsed seconds of one read.
    budget_seconds: Option<f64>,
    #[serde(flatten)]
    unknown: BTreeMap<String, toml::Value>,
}

/// The search settings of a session.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Mode {
    evidence: Evidence,
    budget_seconds: Option<f64>,
}

/// The settings of `config`, or why they are refused.
fn mode_of(config: &DipolarToml) -> Result<Mode, String> {
    let evidence = match config.evidence.as_deref() {
        None | Some("analytic") => Evidence::Analytic,
        Some("physical") => Evidence::Physical,
        Some(other) => {
            return Err(format!(
                "`evidence` must be \"analytic\" or \"physical\", not {other:?}"
            ))
        }
    };
    let budget_seconds = match config.budget_seconds {
        Some(s) if !(s.is_finite() && s > 0.0) => {
            return Err(format!("`budget_seconds` must be positive, not {s}"))
        }
        other => other,
    };
    Ok(Mode {
        evidence,
        budget_seconds,
    })
}

/// The D-static search.
#[derive(Debug)]
pub struct DipolarSampler {
    /// 0 means use host parallelism.
    num_cpus: Arc<AtomicUsize>,
    /// The settings, or why `backend_toml` set none. A refused setting
    /// declines every family job with its reason.
    mode: Arc<RwLock<Result<Mode, String>>>,
}

impl Default for DipolarSampler {
    fn default() -> Self {
        Self {
            num_cpus: Arc::default(),
            mode: Arc::new(RwLock::new(Ok(Mode::default()))),
        }
    }
}

impl DipolarSampler {
    /// A sampler that runs one job per core, with fixed work and analytic
    /// evidence.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn mode(&self) -> Result<Mode, String> {
        current(&self.mode)
    }
}

fn current(cell: &RwLock<Result<Mode, String>>) -> Result<Mode, String> {
    cell.read()
        .map_err(|_| "the settings lock is poisoned".to_owned())
        .and_then(|mode| mode.clone())
}

/// Whether `family` is the revision this binary searches.
fn is_defects(family: &FamilyContext) -> bool {
    family.family_id() == DEFECTS_V1.family_id && family.revision() == DEFECTS_V1.revision
}

/// Run one job: the dipolar search for a D-static job. Every other job is
/// annealed by the generic sampler.
fn job_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    starts: Option<&[Vec<i8>]>,
    family: Option<&FamilyContext>,
    mode: &Result<Mode, String>,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, Failure> {
    match family.filter(|f| is_defects(f)) {
        Some(f) => dipolar_reads(
            graph,
            params,
            starts,
            f.params(),
            f.instance(),
            mode,
            cancel,
        ),
        None => sample_generic_sa(graph, params, cancel).map_err(|_| Failure::Cancelled),
    }
}

/// The search on one D-static job.
fn dipolar_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    starts: Option<&[Vec<i8>]>,
    family_params: &[u8],
    instance: &Instance,
    mode: &Result<Mode, String>,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, Failure> {
    let decline = |detail: String| {
        tracing::warn!(detail, "quip-cpu-dipolar declines the job");
        Failure::Rejected(SampleError::Declined(detail))
    };
    let mode = mode.as_ref().map_err(|why| decline(why.clone()))?;
    if instance.node_ids().len() != graph.h.len() || instance.edges().len() != graph.edges.len() {
        return Err(decline(
            "the instance does not fit the job graph".to_owned(),
        ));
    }
    let problem = Problem::for_instance(family_params, instance, mode.evidence)
        .map_err(|e| decline(e.to_string()))?;
    let work = match mode.budget_seconds {
        Some(seconds) => Work::Budget { seconds },
        None => {
            let sweeps = u64::try_from(params.num_sweeps.max(1)).unwrap_or(u64::MAX);
            Work::Fixed {
                proposals: sweeps.saturating_mul(problem.nodes() as u64),
            }
        }
    };
    let outcome = problem
        .run(starts, params.num_reads, work, params.seed, cancel)
        .map_err(|e| match e {
            DipolarError::Cancelled => Failure::Cancelled,
            other => decline(other.to_string()),
        })?;
    let mut reads = Vec::with_capacity(outcome.reads.len());
    for read in &outcome.reads {
        let canonical = energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges);
        if canonical != read.energy {
            return Err(decline(format!(
                "the search score {} differs from the canonical score {canonical}",
                read.energy
            )));
        }
        reads.push(SamplerResult {
            spins: read.spins.clone(),
            energy_milli: canonical,
        });
    }
    log_job(&problem, &outcome, mode);
    Ok(reads)
}

fn log_job(problem: &Problem, outcome: &RunOutcome, mode: &Mode) {
    let energies: Vec<i64> = outcome.reads.iter().map(|r| r.energy).collect();
    let infeasible = outcome.reads.iter().filter(|r| !r.feasible).count();
    let repaired = outcome.reads.iter().filter(|r| r.repaired).count();
    let resolved = outcome.reads.iter().filter(|r| r.resolved).count();
    let (mut proposed, mut loops, mut strings, mut accepted) = (0u64, 0u64, 0u64, 0u64);
    for read in &outcome.reads {
        proposed = proposed.saturating_add(read.counts.proposed);
        loops = loops.saturating_add(read.counts.loops);
        strings = strings.saturating_add(read.counts.strings);
        accepted = accepted.saturating_add(read.counts.accepted);
    }
    tracing::info!(
        policy = outcome.policy,
        work = ?outcome.work,
        declared_budget_seconds = ?mode.budget_seconds,
        elapsed_seconds = outcome.elapsed_seconds,
        evidence = ?mode.evidence,
        physical_status = ?problem.physical_status(),
        warm = outcome.warm,
        reads = outcome.reads.len(),
        repaired,
        infeasible,
        resolved,
        proposed,
        loops,
        strings,
        accepted,
        rounding_bound_milli = problem.rounding_bound_milli(),
        dropped = problem.dropped(),
        truncation_bound = %problem.truncation_bound(),
        energy_milli = ?energies,
        "dipolar search"
    );
}

impl Sampler for DipolarSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        check_sample_size(graph.h.len(), graph.edges.len())?;
        into_result(job_reads(graph, params, None, None, &self.mode(), None))
    }

    fn accepts_warm_start() -> bool {
        true
    }

    fn sample_warm(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        warm: &WarmStart,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        check_sample_size(graph.h.len(), graph.edges.len())?;
        into_result(job_reads(
            graph,
            params,
            Some(&warm.spins),
            None,
            &self.mode(),
            None,
        ))
    }

    fn accepts_family_metadata() -> bool {
        true
    }

    fn sample_family(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        warm: Option<&WarmStart>,
        family: &FamilyContext,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        check_sample_size(graph.h.len(), graph.edges.len())?;
        let starts = warm.map(|w| w.spins.as_slice());
        into_result(job_reads(
            graph,
            params,
            starts,
            Some(family),
            &self.mode(),
            None,
        ))
    }

    fn declaration() -> Declaration {
        Declaration {
            solver_id: DIPOLAR_SOLVER_ID,
            generated: DEFECTS,
            accepted: DEFECTS,
        }
    }

    fn stream_width(&self) -> usize {
        crate::core_budget(&self.num_cpus)
    }

    fn declared_stream_width() -> u32 {
        u32::try_from(crate::host_parallelism()).unwrap_or(u32::MAX)
    }

    fn apply_config(&self, backend_toml: &str) {
        let config: DipolarToml = if backend_toml.trim().is_empty() {
            DipolarToml::default()
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
        let mode = mode_of(&config);
        match &mode {
            Ok(mode) => tracing::info!(?mode, "dipolar search configured"),
            Err(why) => {
                tracing::error!(%why, "dipolar settings refused; defects-static jobs are declined")
            }
        }
        if let Ok(mut cell) = self.mode.write() {
            *cell = mode;
        }
    }

    /// Every job comes through here, because the sampler accepts warm starts
    /// and family metadata. The cancel token reaches every proposal block of
    /// the search and each descent sweep.
    fn sample_stream_warm(
        &self,
        jobs: tokio::sync::mpsc::Receiver<WarmStreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        let cell = Arc::clone(&self.mode);
        run_job_pump(
            || self.stream_width(),
            move |job: &WarmStreamJob, token: &CancelToken| {
                let (graph, params) = (&job.job.graph, &job.job.params);
                check_sample_size(graph.h.len(), graph.edges.len())
                    .map_err(StreamKernelError::Reject)?;
                let guard = Some((token, job.job.watermark));
                let starts = job.warm_start.as_ref().map(|w| w.spins.as_slice());
                job_reads(
                    graph,
                    params,
                    starts,
                    job.family.as_ref(),
                    &current(&cell),
                    guard,
                )
                .map_err(|failure| match failure {
                    Failure::Rejected(error) => StreamKernelError::Reject(error),
                    Failure::Cancelled => StreamKernelError::Cancelled,
                })
            },
            jobs,
            out,
            cancel,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{dipolar_reads, mode_of, DipolarSampler, DipolarToml, Mode, DEFECTS_V1};
    use crate::postprocess::sampler::Failure;
    use quip_solver_conformance::families::{export_instances, reference_revision};
    use quip_solver_core::families::FamilyTable;
    use quip_solver_core::ising_generators::family::BaseTopology;
    use quip_solver_core::ising_generators::instance::Instance;
    use quip_solver_core::{IsingGraph, SampleError, SampleParams, Sampler};

    use crate::families::dipolar_search::Evidence;

    fn params(num_reads: usize, seed: u64) -> SampleParams {
        SampleParams {
            num_reads,
            num_sweeps: 8,
            seed,
            ..SampleParams::default()
        }
    }

    /// The params and an instance of the reference revision, and its graph.
    fn reference_job() -> (&'static [u8], Instance, IsingGraph) {
        let reference = reference_revision(DEFECTS_V1.family_id, DEFECTS_V1.revision)
            .expect("a reference revision");
        let stored = quip_solver_conformance::topology(reference.topology).unwrap();
        let topology = BaseTopology::new(stored.nodes, &stored.edges).unwrap();
        let nonces: Vec<[u8; 32]> = (0_u8..4).map(|i| [i; 32]).collect();
        let (params, instance) = reference
            .params
            .iter()
            .find_map(|params| {
                let built =
                    export_instances(reference.contract, params, &topology, &nonces).ok()?;
                built.into_iter().find_map(Result::ok).map(|i| (*params, i))
            })
            .expect("a reference params set gives an instance");
        let milli = |v: &i32| f64::from(*v) / 1000.0;
        let graph = IsingGraph::new(
            instance.h().iter().map(milli).collect(),
            instance.j().iter().map(milli).collect(),
            instance
                .edges()
                .iter()
                .map(|&(u, v)| (u as usize, v as usize))
                .collect(),
        );
        (params, instance, graph)
    }

    #[test]
    fn the_declaration_names_exactly_revision_1() {
        let table: FamilyTable = DipolarSampler::declaration()
            .validate()
            .expect("a valid declaration");
        // One declared generated revision. The solver-core default session
        // path adds every other registered family to `generated()`.
        assert_eq!(
            quip_solver_core::ising_generators::registry::CONTRACTS
                .iter()
                .filter(|c| table.declares_generated(c.family_id, c.revision))
                .count(),
            1
        );
        assert_eq!(table.accepted().len(), 1);
    }

    #[test]
    fn a_family_job_returns_every_read_rescored() {
        let (family_params, instance, graph) = reference_job();
        let reads = dipolar_reads(
            &graph,
            &params(3, 7),
            None,
            family_params,
            &instance,
            &Ok(Mode::default()),
            None,
        )
        .ok()
        .expect("reads");
        assert_eq!(reads.len(), 3);
        for read in &reads {
            assert_eq!(
                read.energy_milli,
                quip_protocol::scoring::energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
            );
        }
    }

    #[test]
    fn a_physical_job_without_calibration_is_declined() {
        let (family_params, instance, graph) = reference_job();
        let physical = Ok(Mode {
            evidence: Evidence::Physical,
            budget_seconds: None,
        });
        let result = dipolar_reads(
            &graph,
            &params(1, 0),
            None,
            family_params,
            &instance,
            &physical,
            None,
        );
        assert!(matches!(
            result,
            Err(Failure::Rejected(SampleError::Declined(ref detail)))
                if detail.contains("insufficient physical evidence")
        ));
    }

    #[test]
    fn refused_settings_decline_family_jobs() {
        let config = |evidence: Option<&str>, budget: Option<f64>| DipolarToml {
            evidence: evidence.map(str::to_owned),
            budget_seconds: budget,
            ..DipolarToml::default()
        };
        assert_eq!(mode_of(&config(None, None)), Ok(Mode::default()));
        assert!(mode_of(&config(Some("measured"), None)).is_err());
        assert!(mode_of(&config(None, Some(0.0))).is_err());
        assert!(mode_of(&config(None, Some(f64::NAN))).is_err());
        let (family_params, instance, graph) = reference_job();
        let refused = mode_of(&config(Some("measured"), None));
        assert!(matches!(
            dipolar_reads(
                &graph,
                &params(1, 0),
                None,
                family_params,
                &instance,
                &refused,
                None
            ),
            Err(Failure::Rejected(SampleError::Declined(_)))
        ));
    }

    #[test]
    fn a_plain_job_is_annealed() {
        let n = 8;
        let graph = IsingGraph::new(
            vec![0.0; n],
            vec![-1.0; n],
            (0..n).map(|i| (i, (i + 1) % n)).collect(),
        );
        let got = DipolarSampler::new()
            .sample(&graph, &params(3, 5))
            .expect("a plain job is solved by the generic sampler");
        assert_eq!(got.len(), 3);
    }
}
