//! The `quip-cpu-flow-loops` sampler: the constrained loop search of
//! [`super::flow_loops`] for validated `quip/flow-static` revision 1 and
//! `quip/polariton-static` revision 1 jobs.
//!
//! A job with initial states runs the search from each supplied state, as a
//! postprocess. A job without them starts every read from the exact baseline
//! orientation of [`super::flow::orient`] and runs a seeded search from it.
//! Either way the job returns one state per start, in start order, so result
//! `k` is the child of start `k`. The search runs against the declared budget:
//! the `backend_toml` key `budget_s`, or
//! [`super::flow_loops::FLOW_LOOPS_DEFAULT_BUDGET_SECONDS`]. The budget counts
//! from the start of the job, so parsing, the baseline orientation, and the
//! cycle tables count against it. Every returned state is rescored from the job
//! graph, and the job log keeps the declared budget apart from the measured
//! time.
//!
//! The binary declines every other job with `NO_SOLUTION`, as
//! `quip-cpu-flow` does: a plain model, and every family it does not list,
//! reach the sampler with no link structure.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use quip_protocol::scoring::energy_milli;
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::flow::FLOW_V1;
use quip_solver_core::ising_generators::polariton::POLARITON_V1;
use quip_solver_core::{
    BackendIdentity, CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult,
    StreamResult, WarmStart, WarmStreamJob,
};

use super::flow::{orient, FlowError, LinkTask, Mode, Outcome};
use super::flow_loops::{
    search, LoopError, LoopKernel, LoopReport, Work, FLOW_LOOPS_DEFAULT_BUDGET_SECONDS,
};
use crate::{check_sample_size, run_job_pump, StreamKernelError};

/// The stable id of this binary (FAM-60).
pub const FLOW_LOOPS_SOLVER_ID: &str = "quip-cpu-flow-loops/1";

/// Backend identity for `quip-cpu-flow-loops`. The session adds the
/// `initial-spins` feature, because the sampler accepts warm starts.
pub const CPU_FLOW_LOOPS_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_proto::v1::Backend::Cpu,
    algorithm: quip_proto::v1::Algorithm::FlowLoops,
    max_nodes: crate::DEFAULT_MAX_NODES,
    max_edges: crate::DEFAULT_MAX_EDGES,
    features: &["streaming"],
    adapt: crate::CPU_ADAPT,
    limits: crate::F64_LIMITS,
};

/// Every parameter set of the two revisions. Both build an open square-link
/// lattice on Zephyr cells, the only geometry the search reads.
const FAMILIES: &[FamilyClaim] = &[
    FamilyClaim {
        revisions: &[FLOW_V1],
        only_params: &[],
    },
    FamilyClaim {
        revisions: &[POLARITON_V1],
        only_params: &[],
    },
];

/// The detail of a declined job without a supported family.
const NOT_A_FLOW_JOB: &str = "quip-cpu-flow-loops solves only validated quip/flow-static@1 and \
     quip/polariton-static@1 jobs; other families and plain models have no link structure";

/// The largest budget `budget_s` may set, in seconds.
const MAX_BUDGET_SECONDS: f64 = 60.0;

/// The `backend_toml` keys of this binary.
#[derive(serde::Deserialize, Default)]
struct FlowLoopsConfig {
    num_cpus: Option<i64>,
    /// The declared host-time budget of one job, in seconds.
    budget_s: Option<f64>,
    #[serde(flatten)]
    unknown: BTreeMap<String, toml::Value>,
}

/// The budget that `budget_s` sets, or why it is refused.
fn configured_budget(value: Option<f64>) -> Result<Option<Duration>, String> {
    match value {
        None => Ok(None),
        Some(s) if s.is_finite() && s > 0.0 && s <= MAX_BUDGET_SECONDS => {
            Ok(Some(Duration::from_secs_f64(s)))
        }
        Some(s) => Err(format!(
            "budget_s = {s} is outside (0, {MAX_BUDGET_SECONDS}] seconds"
        )),
    }
}

/// Refuse a bad `budget_s` at handshake. A typo must not run the wrong
/// allowance, because the allowance is what makes the B arm comparable.
#[expect(
    clippy::exit,
    clippy::print_stderr,
    reason = "a bad budget_s is an operator typo; refuse to start rather than run another allowance"
)]
fn reject_config(why: &str) -> ! {
    eprintln!("configuration error: {why}");
    std::process::exit(64);
}

/// The constrained loop search.
#[derive(Debug)]
pub struct FlowLoopsSampler {
    /// 0 means use host parallelism.
    num_cpus: Arc<AtomicUsize>,
    /// The declared budget of one job, as the bits of an `f64` in seconds.
    budget_bits: Arc<AtomicU64>,
}

impl Default for FlowLoopsSampler {
    fn default() -> Self {
        Self {
            num_cpus: Arc::default(),
            budget_bits: Arc::new(AtomicU64::new(FLOW_LOOPS_DEFAULT_BUDGET_SECONDS.to_bits())),
        }
    }
}

impl FlowLoopsSampler {
    /// A sampler that runs one job per core with the default budget.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The declared budget of one job.
    fn budget(&self) -> Duration {
        budget_of(&self.budget_bits)
    }
}

/// The budget stored in `bits`.
fn budget_of(bits: &AtomicU64) -> Duration {
    Duration::from_secs_f64(f64::from_bits(bits.load(Ordering::Relaxed)))
}

/// Why a job has no result.
enum Failure {
    /// The job is declined with this detail.
    Declined(String),
    /// The cancel token fired.
    Cancelled,
}

impl Failure {
    fn declined(detail: impl Into<String>) -> Self {
        let detail = detail.into();
        tracing::warn!(detail, "flow-loops declines the job");
        Self::Declined(detail)
    }
}

/// The start states of a job in link order: the supplied states, or
/// `num_reads` copies of the baseline orientation.
fn job_starts(
    task: &LinkTask,
    params: &SampleParams,
    warm: Option<&[Vec<i8>]>,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<Vec<Vec<i8>>, Failure> {
    if let Some(states) = warm {
        return states
            .iter()
            .enumerate()
            .map(|(k, s)| {
                task.link_spins(s).ok_or_else(|| {
                    Failure::declined(format!("initial state {k} does not fit the instance"))
                })
            })
            .collect();
    }
    match orient(task, cancelled) {
        Ok(Outcome::Optimal(orientation)) => {
            Ok(vec![orientation.link_spins; params.num_reads.max(1)])
        }
        Ok(Outcome::Infeasible(why)) => Err(Failure::declined(format!("infeasible labels: {why}"))),
        Err(FlowError::Cancelled) => Err(Failure::Cancelled),
        Err(other) => Err(Failure::declined(other.to_string())),
    }
}

/// The search on one job, or why it has none.
fn job_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    warm: Option<&[Vec<i8>]>,
    family: Option<&FamilyContext>,
    cancel: Option<(&CancelToken, Option<u64>)>,
    budget: Duration,
) -> Result<Vec<SamplerResult>, Failure> {
    let started = Instant::now();
    let family = family
        .filter(|f| Mode::of(f.contract()).is_some())
        .ok_or_else(|| Failure::declined(NOT_A_FLOW_JOB))?;
    if family.instance().node_ids().len() != graph.h.len() {
        return Err(Failure::declined("the instance does not fit the job graph"));
    }
    // The metadata is checked before any state is read.
    let (mode, task) =
        LinkTask::from_family(family).map_err(|e| Failure::declined(e.to_string()))?;
    let mut cancelled = || cancel.is_some_and(|(token, mark)| token.is_cancelled(mark));
    let starts = job_starts(&task, params, warm, &mut cancelled)?;
    let kernel = LoopKernel::new(&task);
    let outcome = search(
        &kernel,
        &starts,
        params.seed,
        Work::Budget { budget, started },
        &mut cancelled,
        None,
    )
    .map_err(|e| match e {
        LoopError::Cancelled => Failure::Cancelled,
        other => Failure::declined(other.to_string()),
    })?;
    let mut reads = Vec::with_capacity(outcome.states.len());
    for link_spins in &outcome.states {
        let spins = task
            .node_spins(link_spins)
            .ok_or_else(|| Failure::declined("a result does not fit the instance"))?;
        reads.push(SamplerResult {
            energy_milli: energy_milli(&spins, &graph.h, &graph.j, &graph.edges),
            spins,
        });
    }
    log_job(mode, &outcome.report, warm.is_some(), &reads);
    Ok(reads)
}

fn log_job(mode: Mode, report: &LoopReport, warm: bool, reads: &[SamplerResult]) {
    let records = &report.reads;
    let feasible = records.iter().filter(|r| r.end.feasible()).count();
    tracing::info!(
        policy = report.policy,
        mode = mode.name(),
        budget_seconds = ?report.budget_seconds,
        work = ?report.work,
        elapsed_seconds = report.elapsed_seconds,
        warm,
        starts = records.len(),
        feasible,
        parents = ?records.iter().map(|r| r.parent).collect::<Vec<_>>(),
        start_scores = ?records.iter().map(|r| (r.start.gap_squares, r.start.objective)).collect::<Vec<_>>(),
        end_scores = ?records.iter().map(|r| (r.end.gap_squares, r.end.objective)).collect::<Vec<_>>(),
        energies_milli = ?reads.iter().map(|r| r.energy_milli).collect::<Vec<_>>(),
        stops = ?records.iter().map(|r| r.stop).collect::<Vec<_>>(),
        rounds = ?records.iter().map(|r| r.rounds).collect::<Vec<_>>(),
        cycle_moves = records.iter().map(|r| r.cycle_moves).sum::<u64>(),
        worm_moves = records.iter().map(|r| r.worm_moves).sum::<u64>(),
        repair_moves = records.iter().map(|r| r.repair_moves).sum::<u64>(),
        zero_moves = records.iter().map(|r| r.zero_moves).sum::<u64>(),
        proposals = records.iter().map(|r| r.proposals).sum::<u64>(),
        read_seconds = ?records.iter().map(|r| r.seconds).collect::<Vec<_>>(),
        "flow-loops search"
    );
}

fn into_result(
    reads: Result<Vec<SamplerResult>, Failure>,
) -> Result<Vec<SamplerResult>, SampleError> {
    reads.map_err(|failure| match failure {
        Failure::Declined(detail) => SampleError::Declined(detail),
        // Without a token the search never reports a cancel.
        Failure::Cancelled => SampleError::Declined("cancelled".to_owned()),
    })
}

impl Sampler for FlowLoopsSampler {
    /// A plain model has no link structure, so it is declined.
    fn sample(
        &self,
        graph: &IsingGraph,
        _params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        check_sample_size(graph.h.len(), graph.edges.len())?;
        into_result(Err(Failure::declined(NOT_A_FLOW_JOB)))
    }

    fn accepts_warm_start() -> bool {
        true
    }

    /// A warm plain model has no link structure either.
    fn sample_warm(
        &self,
        graph: &IsingGraph,
        _params: &SampleParams,
        _warm: &WarmStart,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        check_sample_size(graph.h.len(), graph.edges.len())?;
        into_result(Err(Failure::declined(NOT_A_FLOW_JOB)))
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
            None,
            self.budget(),
        ))
    }

    fn declaration() -> Declaration {
        Declaration {
            solver_id: FLOW_LOOPS_SOLVER_ID,
            generated: FAMILIES,
            accepted: FAMILIES,
        }
    }

    fn stream_width(&self) -> usize {
        crate::core_budget(&self.num_cpus)
    }

    fn declared_stream_width() -> u32 {
        u32::try_from(crate::host_parallelism()).unwrap_or(u32::MAX)
    }

    fn apply_config(&self, backend_toml: &str) {
        let config: FlowLoopsConfig = if backend_toml.trim().is_empty() {
            FlowLoopsConfig::default()
        } else {
            match toml::from_str(backend_toml) {
                Ok(config) => config,
                Err(_) => crate::reject_num_cpus(crate::NumCpusError::InvalidToml),
            }
        };
        quip_solver_core::config::warn_unknown_fields("cpu", config.unknown.keys());
        match crate::check_num_cpus(config.num_cpus) {
            Ok(Some(requested)) => crate::store_num_cpus(&self.num_cpus, requested),
            Ok(None) => {}
            Err(e) => {
                tracing::error!(error = %e, "cpu num_cpus rejected");
                crate::reject_num_cpus(e);
            }
        }
        match configured_budget(config.budget_s) {
            Ok(Some(budget)) => {
                self.budget_bits
                    .store(budget.as_secs_f64().to_bits(), Ordering::Relaxed);
                tracing::info!(
                    budget_s = budget.as_secs_f64(),
                    "flow-loops budget configured"
                );
            }
            Ok(None) => {}
            Err(why) => {
                tracing::error!(%why, "flow-loops budget_s rejected");
                reject_config(&why);
            }
        }
    }

    /// Every job comes through here, because the sampler accepts warm starts
    /// and family metadata. The cancel token is checked before each
    /// augmentation of the baseline, each short-cycle step, and each worm
    /// proposal.
    fn sample_stream_warm(
        &self,
        jobs: tokio::sync::mpsc::Receiver<WarmStreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        let budget_bits = Arc::clone(&self.budget_bits);
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
                    guard,
                    budget_of(&budget_bits),
                )
                .map_err(|failure| match failure {
                    Failure::Declined(detail) => {
                        StreamKernelError::Reject(SampleError::Declined(detail))
                    }
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
    use super::{configured_budget, FlowLoopsSampler, FAMILIES, MAX_BUDGET_SECONDS};
    use crate::families::flow_loops::FLOW_LOOPS_DEFAULT_BUDGET_SECONDS;
    use quip_solver_core::families::FamilyTable;
    use quip_solver_core::ising_generators::defects::DEFECTS_V1;
    use quip_solver_core::ising_generators::flow::FLOW_V1;
    use quip_solver_core::ising_generators::polariton::POLARITON_V1;
    use quip_solver_core::ising_generators::routing::ROUTING_V1;
    use quip_solver_core::{IsingGraph, SampleError, SampleParams, Sampler};
    use std::time::Duration;

    #[test]
    fn budget_s_sets_the_declared_budget_and_refuses_bad_values() {
        let sampler = FlowLoopsSampler::new();
        let default = Duration::from_secs_f64(FLOW_LOOPS_DEFAULT_BUDGET_SECONDS);
        assert_eq!(sampler.budget(), default);
        sampler.apply_config("");
        assert_eq!(
            sampler.budget(),
            default,
            "an empty config keeps the default"
        );
        sampler.apply_config("budget_s = 0.05");
        assert_eq!(sampler.budget(), Duration::from_millis(50));
        sampler.apply_config("num_cpus = 2");
        assert_eq!(
            sampler.budget(),
            Duration::from_millis(50),
            "no key keeps the value"
        );
        assert_eq!(configured_budget(None), Ok(None));
        assert_eq!(
            configured_budget(Some(MAX_BUDGET_SECONDS)),
            Ok(Some(Duration::from_secs_f64(MAX_BUDGET_SECONDS)))
        );
        for bad in [
            0.0,
            -0.25,
            MAX_BUDGET_SECONDS + 1.0,
            f64::NAN,
            f64::INFINITY,
        ] {
            assert!(configured_budget(Some(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_declaration_lists_exactly_the_p_and_f_revisions() {
        let table: FamilyTable = FlowLoopsSampler::declaration()
            .validate()
            .expect("a valid declaration");
        assert_eq!(FAMILIES.len(), 2);
        for contract in [FLOW_V1, POLARITON_V1] {
            assert!(table
                .generated_contract(contract.family_id, contract.revision)
                .is_some());
            assert!(table
                .accepted_contract(contract.family_id, contract.revision)
                .is_some());
        }
        for contract in [DEFECTS_V1, ROUTING_V1] {
            assert!(table
                .accepted_contract(contract.family_id, contract.revision)
                .is_none());
            assert!(table
                .generated_contract(contract.family_id, contract.revision)
                .is_none());
        }
    }

    #[test]
    fn a_plain_job_is_declined() {
        let graph = IsingGraph::new(
            vec![0.0; 4],
            vec![-1.0; 4],
            vec![(0, 1), (1, 2), (2, 3), (0, 3)],
        );
        let got = FlowLoopsSampler::new().sample(&graph, &SampleParams::default());
        assert!(
            matches!(got, Err(SampleError::Declined(detail)) if detail.contains("quip-cpu-flow-loops"))
        );
    }
}
