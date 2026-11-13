//! The `quip-cpu-flow` sampler: the exact structural baselines of
//! [`super::flow`] for validated `quip/flow-static` revision 1 and
//! `quip/polariton-static` revision 1 jobs.
//!
//! A F-static job returns the min-cost orientation that meets every label,
//! with a certified optimum. An adapted-P job returns an orientation that
//! meets every declared charge, which has the lowest charge loss, 0. Each job
//! returns one state, because the baseline is exact and deterministic. The
//! state passes the canonical feasibility hook of the contract and is
//! rescored from the job graph.
//!
//! The binary is a baseline, not an annealer. It declines every other job
//! with `NO_SOLUTION`: a plain model, and every family it does not list, such
//! as `quip/defects-static` (D-static, with its dipolar energy) and
//! `quip/two-commodity-routing` (occupancy). Those reach the sampler at the
//! generic tier with no family structure, and a flow method has nothing to
//! solve there. A job with infeasible labels is declined with its cut.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

use quip_protocol::scoring::energy_milli;
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::descriptor::Params;
use quip_solver_core::ising_generators::flow::FLOW_V1;
use quip_solver_core::ising_generators::polariton::POLARITON_V1;
use quip_solver_core::{
    BackendIdentity, CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult,
    StreamResult, WarmStart, WarmStreamJob,
};

use super::flow::{orient, FlowError, LinkTask, Mode, Outcome};
use crate::{check_sample_size, run_job_pump, StreamKernelError};

/// The stable id of this binary (FAM-60).
pub const FLOW_SOLVER_ID: &str = "quip-cpu-flow/1";

/// Backend identity for `quip-cpu-flow`. The limits are those of the f64
/// kernels: the baseline reads the integer instance, and the result is
/// rescored from the job graph.
pub const CPU_FLOW_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_proto::v1::Backend::Cpu,
    algorithm: quip_proto::v1::Algorithm::Flow,
    max_nodes: crate::DEFAULT_MAX_NODES,
    max_edges: crate::DEFAULT_MAX_EDGES,
    features: &["streaming"],
    adapt: crate::CPU_ADAPT,
    limits: crate::F64_LIMITS,
};

/// Every parameter set of the two revisions with a baseline.
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
const NOT_A_FLOW_JOB: &str = "quip-cpu-flow solves only validated quip/flow-static@1 and \
     quip/polariton-static@1 jobs; other families and plain models have no flow structure";

/// The exact flow baselines.
#[derive(Debug, Default)]
pub struct FlowSampler {
    /// 0 means use host parallelism.
    num_cpus: Arc<AtomicUsize>,
}

impl FlowSampler {
    /// A sampler that runs one job per core.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
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
        tracing::warn!(detail, "flow declines the job");
        Self::Declined(detail)
    }
}

/// The baseline state of one job, or why it has none.
fn job_reads(
    graph: &IsingGraph,
    family: Option<&FamilyContext>,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, Failure> {
    let family = family
        .filter(|f| Mode::of(f.contract()).is_some())
        .ok_or_else(|| Failure::declined(NOT_A_FLOW_JOB))?;
    let instance = family.instance();
    if instance.node_ids().len() != graph.h.len() {
        return Err(Failure::declined("the instance does not fit the job graph"));
    }
    // The metadata is checked before any graph is built.
    let (mode, task) =
        LinkTask::from_family(family).map_err(|e| Failure::declined(e.to_string()))?;
    let started = Instant::now();
    let mut cancelled = || cancel.is_some_and(|(token, mark)| token.is_cancelled(mark));
    let outcome = orient(&task, &mut cancelled).map_err(|e| match e {
        FlowError::Cancelled => Failure::Cancelled,
        other => Failure::declined(other.to_string()),
    })?;
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let orientation = match outcome {
        Outcome::Optimal(orientation) => orientation,
        Outcome::Infeasible(why) => {
            tracing::info!(
                mode = mode.name(),
                status = "infeasible",
                links = task.positions().len(),
                elapsed_seconds,
                %why,
                "flow baseline"
            );
            return Err(Failure::declined(format!("infeasible labels: {why}")));
        }
    };
    let spins = task
        .node_spins(&orientation.link_spins)
        .ok_or_else(|| Failure::declined("the orientation does not fit the instance"))?;
    let contract = family.contract();
    let hook = contract.construction.and_then(|c| c.feasible);
    let feasible = Params::parse(contract.schema, family.params())
        .ok()
        .zip(hook)
        .is_some_and(|(params, hook)| hook(&params, instance, &spins));
    if !feasible {
        return Err(Failure::declined(
            "the orientation fails the canonical feasibility check",
        ));
    }
    let energy = energy_milli(&spins, &graph.h, &graph.j, &graph.edges);
    tracing::info!(
        mode = mode.name(),
        status = "certified-optimum",
        objective = orientation.objective,
        energy_milli = energy,
        augmentations = orientation.augmentations,
        links = task.positions().len(),
        elapsed_seconds,
        "flow baseline"
    );
    Ok(vec![SamplerResult {
        spins,
        energy_milli: energy,
    }])
}

fn into_result(
    reads: Result<Vec<SamplerResult>, Failure>,
) -> Result<Vec<SamplerResult>, SampleError> {
    reads.map_err(|failure| match failure {
        Failure::Declined(detail) => SampleError::Declined(detail),
        // Without a token the solve never reports a cancel.
        Failure::Cancelled => SampleError::Declined("cancelled".to_owned()),
    })
}

impl Sampler for FlowSampler {
    /// A plain model has no flow structure, so it is declined.
    fn sample(
        &self,
        graph: &IsingGraph,
        _params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        check_sample_size(graph.h.len(), graph.edges.len())?;
        into_result(job_reads(graph, None, None))
    }

    fn accepts_family_metadata() -> bool {
        true
    }

    fn sample_family(
        &self,
        graph: &IsingGraph,
        _params: &SampleParams,
        _warm: Option<&WarmStart>,
        family: &FamilyContext,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        check_sample_size(graph.h.len(), graph.edges.len())?;
        into_result(job_reads(graph, Some(family), None))
    }

    fn declaration() -> Declaration {
        Declaration {
            solver_id: FLOW_SOLVER_ID,
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
        if let Err(e) = crate::apply_num_cpus(&self.num_cpus, backend_toml) {
            tracing::error!(error = %e, "cpu num_cpus rejected");
            crate::reject_num_cpus(e);
        }
    }

    /// Every job comes through here, because the sampler accepts family
    /// metadata. The cancel token is checked before the network is built and
    /// before each shortest-path augmentation.
    fn sample_stream_warm(
        &self,
        jobs: tokio::sync::mpsc::Receiver<WarmStreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        run_job_pump(
            || self.stream_width(),
            |job: &WarmStreamJob, token: &CancelToken| {
                let graph = &job.job.graph;
                check_sample_size(graph.h.len(), graph.edges.len())
                    .map_err(StreamKernelError::Reject)?;
                let guard = Some((token, job.job.watermark));
                job_reads(graph, job.family.as_ref(), guard).map_err(|failure| match failure {
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
    use super::{FlowSampler, FAMILIES};
    use quip_solver_core::families::FamilyTable;
    use quip_solver_core::ising_generators::defects::DEFECTS_V1;
    use quip_solver_core::ising_generators::flow::FLOW_V1;
    use quip_solver_core::ising_generators::polariton::POLARITON_V1;
    use quip_solver_core::ising_generators::routing::ROUTING_V1;
    use quip_solver_core::{IsingGraph, SampleError, SampleParams, Sampler};

    #[test]
    fn the_declaration_lists_exactly_the_two_baseline_revisions() {
        let table: FamilyTable = FlowSampler::declaration()
            .validate()
            .expect("a valid declaration");
        assert_eq!(FAMILIES.len(), 2);
        for contract in [FLOW_V1, POLARITON_V1] {
            assert!(table.declares_generated(contract.family_id, contract.revision));
            assert!(table
                .accepted_contract(contract.family_id, contract.revision)
                .is_some());
        }
        for contract in [DEFECTS_V1, ROUTING_V1] {
            assert!(table
                .accepted_contract(contract.family_id, contract.revision)
                .is_none());
            // Generated on the solver-core default session path, but not
            // declared, so its leases never reach this sampler's own path.
            assert!(!table.declares_generated(contract.family_id, contract.revision));
        }
    }

    #[test]
    fn a_plain_job_is_declined() {
        let graph = IsingGraph::new(
            vec![0.0; 4],
            vec![-1.0; 4],
            vec![(0, 1), (1, 2), (2, 3), (0, 3)],
        );
        let got = FlowSampler::new().sample(&graph, &SampleParams::default());
        assert!(
            matches!(got, Err(SampleError::Declined(detail)) if detail.contains("quip-cpu-flow"))
        );
    }
}
