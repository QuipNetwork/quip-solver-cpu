//! The `quip-cpu-dimer` sampler: the bounded weighted-dimer kernel on
//! validated `quip/twisted-ice-dimer` revision 1 jobs, and the production SA
//! kernel on every job that has no family structure.
//!
//! The session validates each family instance before the sampler sees it
//! (FAM-36). The sampler reads the tiles and pairs from that instance, whose
//! dense node positions are the job graph's nodes, and checks them again
//! against the job graph before it samples. A family job that has a field, a
//! coupling that is not an exact milliweight, an oversize or badly shared
//! tile, a bad pair, or a sum that could overflow is rejected with a
//! `Declined` error. It never falls back to another kernel, because the
//! binary would then claim an algorithm it did not run.
//!
//! Every returned state is rescored from the job graph, which is the original
//! model: the kernel contracts nothing.
//!
//! This binary is not the selected solver of any registered recipe. On the
//! registered twisted-dimer cells the research selects the Otsuka cluster
//! anneal, which `quip-cpu-otsuka` runs. `ice-loop64-weighted` is the weighted
//! loop method of the research QPU suite. The tempering variant
//! `pt-ice-band-strong-weighted` lost to the Otsuka anneal on the registered
//! cell and is not planned.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

use quip_protocol::scoring::energy_milli;
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::ice::twisted_dimer::local_gauges;
use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::{
    BackendIdentity, CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult,
    StreamResult, WarmStart, WarmStreamJob,
};

use super::dimer_weighted::{DimerError, DimerGraph, IceLoopStats, Method};
use super::otsuka_sampler::{parse_pairs, section, PAIRS_SECTION};
use crate::sampler_core::{sample_ising, sample_ising_cancellable};
use crate::{check_sample_size, run_job_pump, StreamKernelError};

/// The stable id of this binary (FAM-60).
pub const DIMER_SOLVER_ID: &str = "quip-cpu-dimer/1";

/// The research method this binary runs: `ice-loop64-weighted`.
pub const DIMER_METHOD: Method = Method::LOOP64;

/// Backend identity for `quip-cpu-dimer`. The limits are those of
/// `quip-cpu-sa`, because the SA kernel serves every job outside the family.
pub const CPU_DIMER_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_proto::v1::Backend::Cpu,
    algorithm: quip_proto::v1::Algorithm::WeightedDimer,
    max_nodes: crate::DEFAULT_MAX_NODES,
    max_edges: crate::DEFAULT_MAX_EDGES,
    features: crate::STREAMING_LINEAR,
    adapt: crate::CPU_ADAPT,
    limits: crate::F64_LIMITS,
};

/// Every parameter set of `quip/twisted-ice-dimer` revision 1, as
/// `quip-cpu-otsuka` claims. The research selects the weighted kernel for no
/// recipe, so restricting the claim to the registered recipes would add no
/// truth. The kernel runs the job's own sweep count, so it holds no tuning to
/// one recipe, and it rejects any instance it cannot run before sampling.
const TWISTED_DIMER: &[FamilyClaim] = &[FamilyClaim {
    revisions: &[TWISTED_DIMER_V1],
    only_params: &[],
}];

/// The CPU sampler behind `quip-cpu-dimer`.
#[derive(Debug, Default)]
pub struct DimerSampler {
    /// 0 means use host parallelism.
    num_cpus: Arc<AtomicUsize>,
}

impl DimerSampler {
    /// A sampler that runs one job per core.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// The tiles and the pairs of an instance.
type Structure = (Vec<[u32; 4]>, Vec<(u32, u32)>);

/// Why a family job produced no reads.
enum Failure {
    Rejected(SampleError),
    Cancelled,
}

/// The tiles and pairs of a validated instance, or `None` when the instance
/// does not fit the job graph.
fn structure(graph: &IsingGraph, instance: &Instance) -> Option<Structure> {
    if instance.node_ids().len() != graph.h.len() || instance.edges().len() != graph.edges.len() {
        return None;
    }
    let (tiles, _) = local_gauges(instance)?;
    let pairs = parse_pairs(section(instance.metadata(), PAIRS_SECTION)?)?;
    Some((tiles, pairs))
}

/// Run the weighted-dimer kernel on a family job.
fn dimer_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    family: &FamilyContext,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, Failure> {
    let Some((tiles, pairs)) = structure(graph, family.instance()) else {
        return Err(Failure::Rejected(SampleError::Declined(
            "the twisted-ice-dimer instance does not fit its job graph".to_owned(),
        )));
    };
    structured_reads(graph, params, &tiles, &pairs, cancel)
}

/// Run the kernel on a job whose tiles and pairs are known. Every restriction
/// of the kernel is checked before the first sweep.
fn structured_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    tiles: &[[u32; 4]],
    pairs: &[(u32, u32)],
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, Failure> {
    let declined = |message: String| Failure::Rejected(SampleError::Declined(message));
    let started = Instant::now();
    let prepared = DimerGraph::new(graph, tiles, pairs).map_err(|e| declined(e.to_string()))?;
    let (states, stats) = prepared
        .sample(params, DIMER_METHOD, cancel)
        .map_err(|e| match e {
            DimerError::Cancelled => Failure::Cancelled,
            DimerError::Invalid(message) => declined(message),
        })?;
    let reads: Vec<SamplerResult> = states
        .into_iter()
        .map(|spins| SamplerResult {
            energy_milli: energy_milli(&spins, &graph.h, &graph.j, &graph.edges),
            spins,
        })
        .collect();
    log_job(params, &stats, &reads, started.elapsed().as_secs_f64());
    Ok(reads)
}

fn log_job(params: &SampleParams, stats: &IceLoopStats, reads: &[SamplerResult], seconds: f64) {
    let best = reads.iter().map(|r| r.energy_milli).min();
    tracing::info!(
        method = "ice-loop64-weighted",
        declared_sweeps = params.num_sweeps,
        declared_reads = params.num_reads,
        measured_seconds = seconds,
        attempts = stats.attempts,
        formed_loops = stats.formed_loops,
        formed_strings = stats.formed_strings,
        accepted = stats.accepted,
        flipped_spins = stats.flipped_spins,
        best_energy_milli = ?best,
        "weighted-dimer anneal"
    );
}

fn is_twisted_dimer(family: &FamilyContext) -> bool {
    family.family_id() == TWISTED_DIMER_V1.family_id
        && family.revision() == TWISTED_DIMER_V1.revision
}

impl Sampler for DimerSampler {
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
        check_sample_size(graph.h.len(), graph.edges.len())?;
        match dimer_reads(graph, params, family, None) {
            Ok(reads) => Ok(reads),
            Err(Failure::Rejected(error)) => Err(error),
            // Without a token the kernel never reports a cancel.
            Err(Failure::Cancelled) => Err(SampleError::Declined("cancelled".to_owned())),
        }
    }

    fn declaration() -> Declaration {
        Declaration {
            solver_id: DIMER_SOLVER_ID,
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
        if let Err(e) = crate::apply_num_cpus(&self.num_cpus, backend_toml) {
            tracing::error!(error = %e, "cpu num_cpus rejected");
            crate::reject_num_cpus(e);
        }
    }

    /// Every job comes through here, because the sampler accepts family
    /// metadata. The cancel token reaches each kernel sweep and each SA read.
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
                if let Some(family) = job.family.as_ref().filter(|f| is_twisted_dimer(f)) {
                    return dimer_reads(graph, params, family, guard).map_err(
                        |failure| match failure {
                            Failure::Rejected(error) => StreamKernelError::Reject(error),
                            Failure::Cancelled => StreamKernelError::Cancelled,
                        },
                    );
                }
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
    use super::{structure, structured_reads, Failure, TWISTED_DIMER_V1};
    use quip_solver_conformance::families::{export_instances, reference_revision};
    use quip_solver_core::ising_generators::family::BaseTopology;
    use quip_solver_core::{IsingGraph, SampleError, SampleParams};

    /// The job graph the session builds from `instance`: dense positions, in
    /// the instance's edge order.
    fn job_graph(h: &[i32], j: &[i32], edges: &[(u32, u32)]) -> IsingGraph {
        let milli = |v: &i32| f64::from(*v) / 1000.0;
        IsingGraph::new(
            h.iter().map(milli).collect(),
            j.iter().map(milli).collect(),
            edges
                .iter()
                .map(|&(u, v)| (u as usize, v as usize))
                .collect(),
        )
    }

    fn reference_instance() -> quip_solver_core::ising_generators::instance::Instance {
        let reference = reference_revision(TWISTED_DIMER_V1.family_id, TWISTED_DIMER_V1.revision)
            .expect("a reference revision");
        let stored = quip_solver_conformance::topology(reference.topology).unwrap();
        let topology = BaseTopology::new(stored.nodes, &stored.edges).unwrap();
        let nonces: Vec<[u8; 32]> = (0_u8..8).map(|i| [i; 32]).collect();
        reference
            .params
            .iter()
            .filter_map(|params| {
                export_instances(reference.contract, params, &topology, &nonces).ok()
            })
            .flatten()
            .find_map(Result::ok)
            .expect("a reference params set gives an instance")
    }

    fn params(reads: usize) -> SampleParams {
        SampleParams {
            num_reads: reads,
            num_sweeps: 16,
            sweeps_per_beta: 4,
            beta_range: Some((0.1, 8.0)),
            seed: 77,
            ..SampleParams::default()
        }
    }

    #[test]
    fn a_reference_instance_samples_and_rescores_on_the_job_graph() {
        let instance = reference_instance();
        let graph = job_graph(instance.h(), instance.j(), instance.edges());
        let (tiles, pairs) = structure(&graph, &instance).expect("the instance fits its job graph");
        assert!(!tiles.is_empty());
        let Ok(reads) = structured_reads(&graph, &params(3), &tiles, &pairs, None) else {
            unreachable!("the reference instance fits the kernel");
        };
        assert_eq!(reads.len(), 3);
        for read in &reads {
            assert_eq!(read.spins.len(), graph.h.len());
            assert!(read.spins.iter().all(|&s| s == 1 || s == -1));
            let direct: i64 = graph
                .edges
                .iter()
                .zip(&graph.j)
                .map(|(&(u, v), &j)| {
                    (j * 1000.0).round() as i64
                        * i64::from(read.spins[u])
                        * i64::from(read.spins[v])
                })
                .sum();
            assert_eq!(read.energy_milli, direct);
        }
    }

    #[test]
    fn a_short_job_graph_does_not_fit_the_instance() {
        let instance = reference_instance();
        let short = job_graph(&instance.h()[1..], instance.j(), instance.edges());
        assert!(structure(&short, &instance).is_none());
    }

    #[test]
    fn a_family_job_with_a_field_is_declined_before_sampling() {
        let instance = reference_instance();
        let mut graph = job_graph(instance.h(), instance.j(), instance.edges());
        let (tiles, pairs) = structure(&graph, &instance).expect("fits");
        graph.h[0] = 0.5;
        assert!(matches!(
            structured_reads(&graph, &params(1), &tiles, &pairs, None),
            Err(Failure::Rejected(SampleError::Declined(_)))
        ));
    }
}
