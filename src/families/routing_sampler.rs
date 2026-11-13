//! The `quip-cpu-routing` sampler: the original alternating routing attack on
//! validated `quip/two-commodity-routing` revision 1 jobs.
//!
//! A job runs both start orders of [`super::routing`]: alternate exact
//! single-commodity responses from commodity 1, then from commodity 2, and
//! descend each result. The sampler returns both final states, best energy
//! first, each rescored from the job graph. The complete ordered spin vector
//! comes back, with `+1` on every spin no arc uses. `num_reads` and the seed
//! have no effect, because the policy is deterministic.
//!
//! The sampler claims no global optimum. A start order that fails is logged and
//! left out. A job with no final state is declined.
//!
//! Admission is strict for a routing revision 1 job: one whose instance does
//! not match the job graph, or whose routing section does not decode, is
//! declined. A job that is not a routing revision 1 job, whether it carries no
//! family metadata, another family, or another revision, is solved by the
//! generic simulated-annealing sampler that `quip-cpu-sa` uses.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

use quip_protocol::scoring::energy_milli;
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::ising_generators::routing::ROUTING_V1;
use quip_solver_core::{
    BackendIdentity, CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult,
    StreamResult, WarmStreamJob,
};

use super::routing::{Candidate, Guard, Routing, RoutingError, POLICY};
use crate::sa_sampler::coefficient_milli;
use crate::sampler_core::sample_generic_sa;
use crate::{check_sample_size, run_job_pump, StreamKernelError};

/// The stable id of this binary (FAM-60).
pub const ROUTING_SOLVER_ID: &str = "quip-cpu-routing/1";

/// Backend identity for `quip-cpu-routing`.
pub const CPU_ROUTING_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_proto::v1::Backend::Cpu,
    algorithm: quip_proto::v1::Algorithm::Routing,
    max_nodes: crate::DEFAULT_MAX_NODES,
    max_edges: crate::DEFAULT_MAX_EDGES,
    features: &["streaming"],
    adapt: crate::CPU_ADAPT,
    limits: crate::F64_LIMITS,
};

/// The binary accepts explicit jobs of every valid parameter set of
/// `quip/two-commodity-routing` revision 1. It generates no family work.
const ROUTING: &[FamilyClaim] = &[FamilyClaim {
    revisions: &[ROUTING_V1],
    only_params: &[],
}];

/// The original alternating routing attack.
#[derive(Debug, Default)]
pub struct RoutingSampler {
    /// 0 means use host parallelism.
    num_cpus: Arc<AtomicUsize>,
}

impl RoutingSampler {
    /// A sampler that runs one job per core.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Why a job has no reads.
enum Failure {
    Rejected(SampleError),
    Cancelled,
}

fn declined(detail: impl Into<String>) -> Failure {
    Failure::Rejected(SampleError::Declined(detail.into()))
}

/// The family of a routing revision 1 job, if the job is one.
fn routing_family(family: Option<&FamilyContext>) -> Option<&FamilyContext> {
    family.filter(|f| f.family_id() == ROUTING_V1.family_id && f.revision() == ROUTING_V1.revision)
}

/// The routing model of an instance of revision 1 that is the job graph.
fn admit_instance(graph: &IsingGraph, instance: &Instance) -> Result<Routing, Failure> {
    let fits = instance.node_ids().len() == graph.h.len()
        && instance.edges().len() == graph.edges.len()
        && instance
            .edges()
            .iter()
            .zip(graph.edges.iter())
            .all(|(&(u, v), &(a, b))| u as usize == a && v as usize == b)
        && instance
            .h()
            .iter()
            .zip(&graph.h)
            .all(|(&m, &g)| coefficient_milli(g) == Some(i64::from(m)))
        && instance
            .j()
            .iter()
            .zip(&graph.j)
            .all(|(&m, &g)| coefficient_milli(g) == Some(i64::from(m)));
    if !fits {
        return Err(declined("the instance does not match the job graph"));
    }
    Routing::from_instance(instance).map_err(|e| declined(e.to_string()))
}

/// Solve the routing instance `instance` of revision 1 on `graph`, without a
/// session: both start orders, best final state first, each rescored from
/// `graph`.
///
/// # Errors
/// [`SampleError::Declined`] when the instance is not the graph or its
/// routing section does not decode, or when no start order gives a state.
pub fn solve_instance(
    graph: &IsingGraph,
    instance: &Instance,
) -> Result<Vec<SamplerResult>, SampleError> {
    check_sample_size(graph.h.len(), graph.edges.len())?;
    into_result(admit_instance(graph, instance).and_then(|model| reads(&model, graph, None)))
}

/// Run both start orders of one admitted job and return the final states,
/// best first.
fn reads(
    model: &Routing,
    graph: &IsingGraph,
    cancel: Guard<'_>,
) -> Result<Vec<SamplerResult>, Failure> {
    let started = Instant::now();
    let mut finals: Vec<SamplerResult> = Vec::with_capacity(2);
    let mut failures: Vec<String> = Vec::new();
    for first in 0..2 {
        let outcome = model
            .candidate(first, cancel)
            .and_then(|candidate| score(graph, candidate));
        match outcome {
            Ok((candidate, energy)) => {
                tracing::info!(
                    policy = POLICY,
                    first,
                    energy_milli = energy,
                    pre_descent_milli = candidate.before_descent,
                    solves = candidate.solves,
                    settled = candidate.settled,
                    flips = candidate.flips,
                    "routing start order"
                );
                finals.push(SamplerResult {
                    spins: candidate.spins,
                    energy_milli: energy,
                });
            }
            Err(RoutingError::Cancelled) => return Err(Failure::Cancelled),
            Err(error) => {
                tracing::warn!(policy = POLICY, first, %error, "routing start order failed");
                failures.push(format!("start order {first}: {error}"));
            }
        }
    }
    // The stable sort keeps start order 0 first on a tie, like the research pick.
    finals.sort_by_key(|read| read.energy_milli);
    tracing::info!(
        policy = POLICY,
        spins = model.spins(),
        arcs = model.arc_count(),
        reads = finals.len(),
        failures = failures.len(),
        seconds = started.elapsed().as_secs_f64(),
        optimum_certificate = false,
        "routing job"
    );
    if finals.is_empty() {
        return Err(declined(format!(
            "no start order gave a final state: {}",
            failures.join("; ")
        )));
    }
    Ok(finals)
}

/// Rescore a candidate from the job graph. A final state whose exact score
/// differs from the kernel's, or whose length is wrong, is a failed start order.
fn score(graph: &IsingGraph, candidate: Candidate) -> Result<(Candidate, i64), RoutingError> {
    let energy = energy_milli(&candidate.spins, &graph.h, &graph.j, &graph.edges);
    if candidate.spins.len() != graph.h.len() || energy != candidate.energy {
        return Err(RoutingError::Model(format!(
            "the exact score {energy} differs from the kernel score {}",
            candidate.energy
        )));
    }
    Ok((candidate, energy))
}

/// The routing attack on a routing job. Any other job is annealed by the
/// generic sampler.
fn job_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    family: Option<&FamilyContext>,
    cancel: Guard<'_>,
) -> Result<Vec<SamplerResult>, Failure> {
    check_sample_size(graph.h.len(), graph.edges.len()).map_err(Failure::Rejected)?;
    let Some(family) = routing_family(family) else {
        return sample_generic_sa(graph, params, cancel).map_err(|_| Failure::Cancelled);
    };
    let model = admit_instance(graph, family.instance())?;
    reads(&model, graph, cancel)
}

fn into_result(
    reads: Result<Vec<SamplerResult>, Failure>,
) -> Result<Vec<SamplerResult>, SampleError> {
    match reads {
        Ok(reads) => Ok(reads),
        Err(Failure::Rejected(error)) => Err(error),
        // Without a token the run never reports a cancel.
        Err(Failure::Cancelled) => Err(SampleError::Declined("cancelled".to_owned())),
    }
}

impl Sampler for RoutingSampler {
    /// A job without family metadata is not a routing job, so it is annealed.
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        into_result(job_reads(graph, params, None, None))
    }

    fn accepts_family_metadata() -> bool {
        true
    }

    fn sample_family(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        _warm: Option<&quip_solver_core::WarmStart>,
        family: &FamilyContext,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        into_result(job_reads(graph, params, Some(family), None))
    }

    fn declaration() -> Declaration {
        Declaration {
            solver_id: ROUTING_SOLVER_ID,
            generated: &[],
            accepted: ROUTING,
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
    /// metadata. The cancel token reaches each shortest path and each
    /// descent sweep.
    fn sample_stream_warm(
        &self,
        jobs: tokio::sync::mpsc::Receiver<WarmStreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        run_job_pump(
            || self.stream_width(),
            |job: &WarmStreamJob, token: &CancelToken| {
                let guard = Some((token, job.job.watermark));
                job_reads(&job.job.graph, &job.job.params, job.family.as_ref(), guard).map_err(
                    |failure| match failure {
                        Failure::Rejected(error) => StreamKernelError::Reject(error),
                        Failure::Cancelled => StreamKernelError::Cancelled,
                    },
                )
            },
            jobs,
            out,
            cancel,
        );
    }
}
