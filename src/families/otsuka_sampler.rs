//! The `quip-cpu-otsuka` sampler: the Otsuka cluster kernel on validated
//! `quip/twisted-ice-dimer` revision 1 jobs, and the production SA kernel on
//! every other job.
//!
//! The session validates each family instance before the sampler sees it
//! (FAM-36), from a lease salt it regenerated or from an explicit job's
//! binding. The kernel reads the model, tiles, gauges, and pairs from that
//! instance, whose dense node positions are the job graph's nodes.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::ice::twisted_dimer::local_gauges;
use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::{
    BackendIdentity, CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult,
    StreamResult, WarmStart, WarmStreamJob,
};

use super::otsuka::{
    anneal_population, AnnealError, Model, Motifs, OtsukaGraph, Schedule, Weights,
};
use crate::sampler_core::{
    build_beta_schedule, effective_sweeps_per_beta, sample_ising, sample_ising_cancellable,
};
use crate::{check_sample_size, run_job_pump, StreamKernelError};

/// The stable id of this binary (FAM-60).
pub const OTSUKA_SOLVER_ID: &str = "quip-cpu-otsuka/1";

/// Backend identity for `quip-cpu-otsuka`. The limits are those of
/// `quip-cpu-sa`, because the SA kernel serves every job outside the family.
pub const CPU_OTSUKA_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_proto::v1::Backend::Cpu,
    algorithm: quip_proto::v1::Algorithm::Otsuka,
    max_nodes: crate::DEFAULT_MAX_NODES,
    max_edges: crate::DEFAULT_MAX_EDGES,
    features: crate::STREAMING_LINEAR,
    adapt: crate::CPU_ADAPT,
    limits: crate::F64_LIMITS,
};

/// Every parameter set of `quip/twisted-ice-dimer` revision 1. The kernel
/// accepts every valid instance of the revision.
const TWISTED_DIMER: &[FamilyClaim] = &[FamilyClaim {
    revisions: &[TWISTED_DIMER_V1],
    only_params: &[],
}];

/// The anneal ends, per full coupler, for a family job that names no beta
/// range. A schedule study on the `x-w62-p1200` cell (seven ranges, eight
/// seeds, two models, 65,536-sweep anneals) chose 12 as the top. Under
/// the population anneal at 10 replicas of 32,768 sweeps, a start of 1
/// instead of 0.5 won 26 of 28 runs against 12 of 16 on the `x-w62-p1350`
/// cell. The default SA range of these models, about 0.044 to 105, spends
/// most of a geometric ladder outside that band.
const FAMILY_BETA_RANGE: (f64, f64) = (1.0, 12.0);

/// The bond weights of the research solver: the cluster draw takes a
/// quarter of each pair coupling, and the heat bath scores the rest.
const FAMILY_WEIGHTS: Weights = Weights {
    pair_milli: 250,
    tile_milli: 1000,
};

/// A Metropolis sweep after every cluster sweep, as in the research solver.
const SINGLE_EVERY: usize = 1;

/// The metadata section that holds the pairs.
pub(crate) const PAIRS_SECTION: u16 = 3;

/// The CPU sampler behind `quip-cpu-otsuka`.
#[derive(Debug, Default)]
pub struct OtsukaSampler {
    /// 0 means use host parallelism.
    num_cpus: Arc<AtomicUsize>,
}

impl OtsukaSampler {
    /// A sampler that runs one model per core.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// The payload of section `kind` in a metadata region: a `u16` section count,
/// then each section as a `u16` kind, a `u32` length, and the payload, all
/// little-endian (FAM-67).
pub(crate) fn section(metadata: &[u8], kind: u16) -> Option<&[u8]> {
    let (count, mut rest) = metadata.split_first_chunk::<2>()?;
    for _ in 0..u16::from_le_bytes(*count) {
        let (head, tail) = rest.split_first_chunk::<6>()?;
        let len = usize::try_from(u32::from_le_bytes([head[2], head[3], head[4], head[5]])).ok()?;
        let (payload, next) = tail.split_at_checked(len)?;
        if u16::from_le_bytes([head[0], head[1]]) == kind {
            return Some(payload);
        }
        rest = next;
    }
    None
}

/// The pairs of a validated metadata section: a `u32` count, then each pair
/// as two little-endian `u32` positions.
pub(crate) fn parse_pairs(payload: &[u8]) -> Option<Vec<(u32, u32)>> {
    let (count, rest) = payload.split_first_chunk::<4>()?;
    let count = usize::try_from(u32::from_le_bytes(*count)).ok()?;
    let (pairs, tail) = rest.as_chunks::<8>();
    if pairs.len() != count || !tail.is_empty() {
        return None;
    }
    Some(
        pairs
            .iter()
            .map(|p| {
                (
                    u32::from_le_bytes([p[0], p[1], p[2], p[3]]),
                    u32::from_le_bytes([p[4], p[5], p[6], p[7]]),
                )
            })
            .collect(),
    )
}

/// The kernel graph of a validated twisted-ice-dimer job, or `None` when the
/// job is outside the family or its instance does not fit the job graph.
fn otsuka_graph(graph: &IsingGraph, family: &FamilyContext) -> Option<OtsukaGraph> {
    if family.family_id() != TWISTED_DIMER_V1.family_id
        || family.revision() != TWISTED_DIMER_V1.revision
    {
        return None;
    }
    prepare(graph, family.instance())
}

/// The kernel graph of a validated twisted-ice-dimer instance, or `None` when
/// the instance does not fit `graph` or the kernel.
fn prepare(graph: &IsingGraph, instance: &Instance) -> Option<OtsukaGraph> {
    if instance.node_ids().len() != graph.h.len() || instance.edges().len() != graph.edges.len() {
        return None;
    }
    let (tiles, gauges) = local_gauges(instance)?;
    let pairs = parse_pairs(section(instance.metadata(), PAIRS_SECTION)?)?;
    let model = Model {
        h_milli: instance.h(),
        edges: instance.edges(),
        j_milli: instance.j(),
    };
    let motifs = Motifs {
        tiles: &tiles,
        gauges: &gauges,
        pairs: &pairs,
    };
    match OtsukaGraph::new(&model, &motifs, FAMILY_WEIGHTS) {
        Ok(prepared) => Some(prepared),
        Err(error) => {
            tracing::warn!(%error, "a validated twisted-ice-dimer instance does not fit the kernel");
            None
        }
    }
}

/// The per-sweep betas of `params`: each rung of the SA ladder repeated for
/// its sweeps. A job without a beta range takes [`FAMILY_BETA_RANGE`].
fn sweep_betas(graph: &IsingGraph, params: &SampleParams) -> Vec<f64> {
    let per_beta = effective_sweeps_per_beta(params.num_sweeps, params.sweeps_per_beta);
    let params = SampleParams {
        beta_range: Some(params.beta_range.unwrap_or(FAMILY_BETA_RANGE)),
        ..params.clone()
    };
    build_beta_schedule(graph, &params)
        .into_iter()
        .flat_map(|beta| std::iter::repeat_n(beta, per_beta))
        .collect()
}

/// A population anneal of `params.num_reads` replicas on a family job, or
/// `None` when the job must run on the SA kernel instead.
fn otsuka_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    family: &FamilyContext,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Option<Result<Vec<SamplerResult>, AnnealError>> {
    let prepared = otsuka_graph(graph, family)?;
    let betas = sweep_betas(graph, params);
    let schedule = match Schedule::new(&betas, SINGLE_EVERY) {
        Ok(schedule) => schedule,
        Err(error) => {
            tracing::warn!(%error, "the job schedule does not fit the Otsuka kernel");
            return None;
        }
    };
    // The reads anneal as one population, so the sweeps go to the replicas
    // that are already low.
    let reads = anneal_population(&prepared, &schedule, params.num_reads, params.seed, cancel).map(
        |samples| {
            samples
                .into_iter()
                .map(|sample| SamplerResult {
                    spins: sample.spins,
                    energy_milli: sample.energy_milli,
                })
                .collect()
        },
    );
    Some(reads)
}

impl Sampler for OtsukaSampler {
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
        match otsuka_reads(graph, params, family, None) {
            Some(Ok(reads)) => Ok(reads),
            // Without a token the anneal never reports a cancel.
            Some(Err(AnnealError::Cancelled)) | None => self.sample(graph, params),
        }
    }

    fn declaration() -> Declaration {
        Declaration {
            solver_id: OTSUKA_SOLVER_ID,
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
    /// metadata. The cancel token reaches each anneal sweep and each SA read.
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
                let guard = Some((token, job.job.watermark));
                if let Some(family) = &job.family {
                    match otsuka_reads(graph, params, family, guard) {
                        Some(Ok(reads)) => return Ok(reads),
                        Some(Err(AnnealError::Cancelled)) => {
                            return Err(StreamKernelError::Cancelled)
                        }
                        None => {}
                    }
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
    use super::{parse_pairs, prepare, section, TWISTED_DIMER_V1};
    use quip_solver_conformance::families::{export_instances, reference_revision};
    use quip_solver_core::ising_generators::family::BaseTopology;
    use quip_solver_core::IsingGraph;

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

    #[test]
    fn a_reference_twisted_dimer_instance_takes_the_otsuka_path() {
        let reference = reference_revision(TWISTED_DIMER_V1.family_id, TWISTED_DIMER_V1.revision)
            .expect("a reference revision");
        let stored = quip_solver_conformance::topology(reference.topology).unwrap();
        let topology = BaseTopology::new(stored.nodes, &stored.edges).unwrap();
        let nonces: Vec<[u8; 32]> = (0_u8..8).map(|i| [i; 32]).collect();
        let instance = reference
            .params
            .iter()
            .filter_map(|params| {
                export_instances(reference.contract, params, &topology, &nonces).ok()
            })
            .flatten()
            .find_map(Result::ok)
            .expect("a reference params set gives an instance");

        let graph = job_graph(instance.h(), instance.j(), instance.edges());
        assert!(
            prepare(&graph, &instance).is_some(),
            "the kernel rejects the instance"
        );

        let short = job_graph(&instance.h()[1..], instance.j(), instance.edges());
        assert!(
            prepare(&short, &instance).is_none(),
            "a node count mismatch"
        );
    }

    fn region(sections: &[(u16, &[u8])]) -> Vec<u8> {
        let mut bytes = u16::try_from(sections.len())
            .unwrap()
            .to_le_bytes()
            .to_vec();
        for (kind, payload) in sections {
            bytes.extend(kind.to_le_bytes());
            bytes.extend(u32::try_from(payload.len()).unwrap().to_le_bytes());
            bytes.extend(*payload);
        }
        bytes
    }

    #[test]
    fn a_section_is_found_by_its_kind() {
        let bytes = region(&[(1, &[9, 9]), (2, &[]), (3, &[7])]);
        assert_eq!(section(&bytes, 1), Some(&[9_u8, 9][..]));
        assert_eq!(section(&bytes, 2), Some(&[][..]));
        assert_eq!(section(&bytes, 3), Some(&[7_u8][..]));
        assert_eq!(section(&bytes, 4), None);
        let mut long = region(&[(3, &[7])]);
        long[4] = 2;
        assert_eq!(section(&long, 3), None, "a length past the end");
        assert_eq!(section(&bytes[..5], 3), None, "a truncated header");
        assert_eq!(section(&[], 1), None);
    }

    #[test]
    fn pairs_parse_from_their_count_and_positions() {
        let mut payload = 2_u32.to_le_bytes().to_vec();
        for value in [0_u32, 4, 5, 7] {
            payload.extend(value.to_le_bytes());
        }
        assert_eq!(parse_pairs(&payload), Some(vec![(0, 4), (5, 7)]));
        payload.push(0);
        assert_eq!(parse_pairs(&payload), None);
        assert_eq!(parse_pairs(&3_u32.to_le_bytes()), None);
        assert_eq!(parse_pairs(&[0, 0]), None);
    }
}
