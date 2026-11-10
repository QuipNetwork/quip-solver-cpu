//! The `quip-cpu-gadget-repair` sampler: the `gadget-repair/1` postprocess
//! for routing jobs.
//!
//! A `quip/two-commodity-routing` revision 1 job runs the schedule against
//! its declared budget on the supplied states. It returns the final state
//! first, then the repaired state of each start, in start order. Every
//! returned state is rescored from the job graph. The schedule seeds from the
//! job seed. The job log reports the direct defects of the final state and of
//! the best supplied state: a lower energy never stands for feasibility.
//!
//! Every other job runs the `greedy-nat/1` descent of `quip-cpu-greedy`, so
//! the binary still serves plain and warm-start jobs. A routing job that
//! fails the structure checks is declined with the reason. A job without
//! states starts from random states and logs a warning, because the binary is
//! a postprocess.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use quip_protocol::scoring::energy_milli;
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::ising_generators::routing::{RoutingDefects, ROUTING_V1};
use quip_solver_core::{
    BackendIdentity, CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult,
    StreamResult, WarmStart, WarmStreamJob,
};

use super::gadget_repair::{
    arcs_of, check_routing, check_spin_cover, is_feasible, schedule_gadget, GadgetError,
    GadgetReport, Repair, Structure, Work,
};
use super::greedy::{natural_moves, DescentError, Greedy, MoveSources};
use super::sampler::{
    greedy_reads, into_result, job_ising, random_states, Failure, Structure as JobStructure,
};
use crate::{check_sample_size, run_job_pump, StreamKernelError};

/// The stable id of this binary (FAM-60).
pub const GADGET_REPAIR_SOLVER_ID: &str = "quip-cpu-gadget-repair/1";

/// Backend identity for `quip-cpu-gadget-repair`. The session adds the
/// `initial-spins` feature, because the sampler accepts warm starts.
pub const CPU_GADGET_REPAIR_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_proto::v1::Backend::Cpu,
    algorithm: quip_proto::v1::Algorithm::GadgetRepair,
    max_nodes: crate::DEFAULT_MAX_NODES,
    max_edges: crate::DEFAULT_MAX_EDGES,
    features: &["streaming"],
    adapt: crate::CPU_ADAPT,
    limits: crate::F64_LIMITS,
};

/// The binary accepts explicit routing jobs of revision 1 for their chain and
/// commodity mapping. It generates no family work: a lease salt carries no
/// states to repair.
const ACCEPTED: &[FamilyClaim] = &[FamilyClaim {
    revisions: &[ROUTING_V1],
    only_params: &[],
}];

/// The routing gadget repair, with the greedy descent for other jobs.
#[derive(Debug, Default)]
pub struct GadgetRepairSampler {
    /// 0 means use host parallelism.
    num_cpus: Arc<AtomicUsize>,
}

impl GadgetRepairSampler {
    /// A sampler that runs one job per core.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Whether `family` is a routing job of revision 1.
fn is_routing(family: &FamilyContext) -> bool {
    family.family_id() == ROUTING_V1.family_id && family.revision() == ROUTING_V1.revision
}

/// Run one job: the gadget repair for a routing job, the greedy descent for
/// every other job.
fn job_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    starts: Option<&[Vec<i8>]>,
    family: Option<&FamilyContext>,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, Failure> {
    match family.filter(|f| is_routing(f)) {
        Some(family) => gadget_reads(
            graph,
            params,
            starts,
            family.params(),
            family.instance(),
            &Work::Budget,
            cancel,
        )
        .map(|job| job.reads),
        None => greedy_reads(graph, params, starts, &JobStructure::default(), cancel),
    }
}

fn declined(reason: &dyn std::fmt::Display) -> Failure {
    Failure::Rejected(SampleError::Declined(reason.to_string()))
}

/// The result of the gadget repair on one routing job.
#[derive(Clone, Debug)]
pub struct RoutingJob {
    /// The final state first, then the repaired state of each start, all
    /// rescored from the job graph.
    pub reads: Vec<SamplerResult>,
    /// What the schedule did, and its declared budget apart from the measured
    /// time.
    pub report: GadgetReport,
    /// The direct constraint defects of the final state.
    pub final_defects: RoutingDefects,
    /// The energy of the lowest supplied state, rescored from the job graph.
    pub best_start_energy: i64,
    /// The direct constraint defects of that state.
    pub best_start_defects: RoutingDefects,
    /// The offset of the instance in milli. The objective is the energy plus
    /// this offset.
    pub offset_milli: i64,
}

/// Run the gadget repair on the states of one routing job, under `work`.
///
/// # Errors
///
/// [`SampleError::Declined`] with the reason when the instance fails the
/// routing validation or does not fit the graph, when the chain and commodity
/// mapping is malformed, or when a state does not fit the model.
/// [`SampleError::Capacity`] when a coefficient weight is past the bound.
pub fn routing_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    starts: Option<&[Vec<i8>]>,
    family_params: &[u8],
    instance: &Instance,
    work: &Work,
) -> Result<RoutingJob, SampleError> {
    match gadget_reads(graph, params, starts, family_params, instance, work, None) {
        Ok(job) => Ok(job),
        Err(Failure::Rejected(error)) => Err(error),
        // Without a token the schedule never reports a cancel.
        Err(Failure::Cancelled) => Err(SampleError::Declined("cancelled".to_owned())),
    }
}

fn gadget_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    starts: Option<&[Vec<i8>]>,
    family_params: &[u8],
    instance: &Instance,
    work: &Work,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<RoutingJob, Failure> {
    if instance.node_ids().len() != graph.h.len() || instance.edges().len() != graph.edges.len() {
        return Err(declined(&GadgetError::Fit));
    }
    // The instance and the mapping are checked before any state is touched.
    let spins = graph.h.len();
    check_routing(family_params, instance, &vec![1; spins]).map_err(|e| declined(&e))?;
    let arcs = arcs_of(instance).map_err(|e| declined(&e))?;
    check_spin_cover(spins, &arcs).map_err(|e| declined(&e))?;
    let structure = Structure::extract(spins, &arcs).map_err(|e| declined(&e))?;
    let ising = job_ising(graph).map_err(Failure::Rejected)?;
    Repair::check_couplers(&ising, &arcs).map_err(|e| declined(&e))?;
    let (moves, _) = natural_moves(&ising, &MoveSources::default()).map_err(|e| declined(&e))?;
    let nat = Greedy::new(ising.clone(), &moves, true).map_err(|e| declined(&e))?;
    let repair = Repair::new(ising, &structure).map_err(|e| declined(&e))?;

    let warm = starts.is_some();
    let starts = match starts {
        Some(starts) => starts.to_vec(),
        None => {
            tracing::warn!(
                "a job without initial spins starts from random states; \
                 quip-cpu-gadget-repair is a postprocess and should only receive warm jobs"
            );
            random_states(spins, params)
        }
    };
    let out =
        schedule_gadget(&nat, &repair, &starts, params.seed, work, cancel).map_err(
            |e| match e {
                DescentError::Cancelled => Failure::Cancelled,
                other => declined(&other),
            },
        )?;
    let energy = |s: &[i8]| energy_milli(s, &graph.h, &graph.j, &graph.edges);
    let reads: Vec<SamplerResult> = std::iter::once(out.state)
        .chain(out.pool)
        .map(|spins| SamplerResult {
            energy_milli: energy(&spins),
            spins,
        })
        .collect();

    // The direct constraint check of the final state and of the best supplied
    // state. The energy never stands in for it.
    let defects = |s: &[i8]| check_routing(family_params, instance, s).map_err(|e| declined(&e));
    let best = starts
        .iter()
        .min_by_key(|s| energy(s))
        .ok_or_else(|| declined(&DescentError::Empty))?;
    let job = RoutingJob {
        final_defects: defects(&reads[0].spins)?,
        best_start_defects: defects(best)?,
        best_start_energy: energy(best),
        offset_milli: instance.offset_milli(),
        reads,
        report: out.report,
    };
    log_job(&job, warm, starts.len());
    Ok(job)
}

fn log_job(job: &RoutingJob, warm: bool, supplied: usize) {
    let report = &job.report;
    let starts: Vec<String> = report
        .starts
        .iter()
        .map(|s| {
            format!(
                "{:?}: {} to {} after {} clock reads in {:.4} s, {:?}",
                s.read, s.start_energy, s.energy, s.ticks, s.seconds, s.stats
            )
        })
        .collect();
    let energy = job.reads[0].energy_milli;
    tracing::info!(
        policy = report.policy,
        budget_seconds = report.budget_seconds,
        work = ?report.work,
        elapsed_seconds = report.elapsed_seconds,
        warm,
        supplied,
        descended = report.descended,
        descent_best_milli = report.descent_best,
        repaired = ?starts,
        energy_milli = energy,
        objective_milli = energy + job.offset_milli,
        best_start_milli = job.best_start_energy,
        best_start_defects = ?job.best_start_defects,
        final_defects = ?job.final_defects,
        feasible = is_feasible(&job.final_defects),
        "gadget-repair postprocess"
    );
}

impl Sampler for GadgetRepairSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        check_sample_size(graph.h.len(), graph.edges.len())?;
        into_result(job_reads(graph, params, None, None, None))
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
        into_result(job_reads(graph, params, Some(&warm.spins), None, None))
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
        into_result(job_reads(graph, params, starts, Some(family), None))
    }

    fn declaration() -> Declaration {
        Declaration {
            solver_id: GADGET_REPAIR_SOLVER_ID,
            generated: &[],
            accepted: ACCEPTED,
        }
    }

    fn stream_width(&self) -> usize {
        crate::core_budget(&self.num_cpus)
    }

    fn declared_stream_width() -> u32 {
        u32::try_from(crate::host_parallelism()).unwrap_or(u32::MAX)
    }

    fn apply_config(&self, backend_toml: &str) {
        if let Err(e) = crate::apply_num_cpus(&self.num_cpus, backend_toml) {
            tracing::error!(error = %e, "cpu num_cpus rejected");
            crate::reject_num_cpus(e);
        }
    }

    /// Every job comes through here, because the sampler accepts warm starts
    /// and family metadata. The cancel token reaches each clock read of the
    /// repair, and each descent sweep.
    fn sample_stream_warm(
        &self,
        jobs: tokio::sync::mpsc::Receiver<WarmStreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        run_job_pump(
            || self.stream_width(),
            |job: &WarmStreamJob, token: &CancelToken| {
                let (graph, params) = (&job.job.graph, &job.job.params);
                check_sample_size(graph.h.len(), graph.edges.len())
                    .map_err(StreamKernelError::Reject)?;
                let guard = Some((token, job.job.watermark));
                let starts = job.warm_start.as_ref().map(|w| w.spins.as_slice());
                job_reads(graph, params, starts, job.family.as_ref(), guard).map_err(|failure| {
                    match failure {
                        Failure::Rejected(error) => StreamKernelError::Reject(error),
                        Failure::Cancelled => StreamKernelError::Cancelled,
                    }
                })
            },
            jobs,
            out,
            cancel,
        );
    }
}
