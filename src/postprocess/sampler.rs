//! The `quip-cpu-greedy` sampler: exact greedy descent from supplied states.
//!
//! A warm job descends each supplied state and returns one result per state,
//! in order. The moves are the single flips and the natural moves. A
//! validated `quip/twisted-ice-dimer` revision 1 job adds its tiles and
//! pairs. Every other job uses the couplings alone: edge pairs and rigid
//! groups. A job without states descends from random states and logs a
//! warning, because the binary is a postprocess.

use std::collections::BTreeMap;
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
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::greedy::{
    natural_moves, DescentError, Greedy, Ising, ModelError, MoveCounts, MoveSources,
};
use crate::families::otsuka_sampler::{parse_pairs, section, PAIRS_SECTION};
use crate::sa_sampler::coefficient_milli;
use crate::{check_sample_size, run_job_pump, StreamKernelError};

/// The stable id of this binary (FAM-60).
pub const GREEDY_SOLVER_ID: &str = "quip-cpu-greedy/1";

/// The postprocess policy: single flips plus natural moves, first
/// improvement. Versioned apart from any family revision.
pub const GREEDY_POLICY: &str = "greedy-nat/1";

/// Backend identity for `quip-cpu-greedy`. The session adds the
/// `initial-spins` feature, because the sampler accepts warm starts.
pub const CPU_GREEDY_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_proto::v1::Backend::Cpu,
    algorithm: quip_proto::v1::Algorithm::Greedy,
    max_nodes: crate::DEFAULT_MAX_NODES,
    max_edges: crate::DEFAULT_MAX_EDGES,
    features: &["streaming"],
    adapt: crate::CPU_ADAPT,
    limits: crate::F64_LIMITS,
};

/// The binary accepts explicit twisted-ice-dimer jobs for their structure.
/// It generates no family work: a lease salt carries no states to descend.
const TWISTED_DIMER: &[FamilyClaim] = &[FamilyClaim {
    revisions: &[TWISTED_DIMER_V1],
    only_params: &[],
}];

/// Exact greedy descent from supplied states.
#[derive(Debug, Default)]
pub struct GreedySampler {
    /// 0 means use host parallelism.
    num_cpus: Arc<AtomicUsize>,
}

impl GreedySampler {
    /// A sampler that descends one job per core.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// The integer model of a job graph.
pub(super) fn job_ising(graph: &IsingGraph) -> Result<Ising, SampleError> {
    let milli = |c: &f64| {
        coefficient_milli(*c)
            .ok_or_else(|| SampleError::Declined("a coefficient is not finite".to_owned()))
    };
    let h = graph.h.iter().map(milli).collect::<Result<Vec<_>, _>>()?;
    // A repeated edge adds to one coupling, and a self-loop adds a constant
    // that no move changes. The reported energy comes from the job graph.
    let mut merged: BTreeMap<(u32, u32), i64> = BTreeMap::new();
    for (&(u, v), c) in graph.edges.iter().zip(&graph.j) {
        let (Ok(u), Ok(v)) = (u32::try_from(u), u32::try_from(v)) else {
            return Err(SampleError::Declined("a node index exceeds u32".to_owned()));
        };
        if u != v {
            *merged.entry((u.min(v), u.max(v))).or_default() += milli(c)?;
        }
    }
    let (edges, j): (Vec<_>, Vec<_>) = merged.into_iter().unzip();
    Ising::new(&h, &edges, &j).map_err(|e| match e {
        ModelError::Overflow => SampleError::Capacity,
        other => SampleError::Declined(other.to_string()),
    })
}

/// The declared structure of a family job, in dense node positions.
#[derive(Default)]
pub(super) struct Structure {
    pub(super) tiles: Vec<[u32; 4]>,
    pub(super) pairs: Vec<(u32, u32)>,
}

/// The structure of a job: the tiles and pairs of a validated
/// twisted-ice-dimer job, and none for any other job.
pub(super) fn job_structure(graph: &IsingGraph, family: Option<&FamilyContext>) -> Structure {
    family
        .filter(|f| {
            f.family_id() == TWISTED_DIMER_V1.family_id && f.revision() == TWISTED_DIMER_V1.revision
        })
        .and_then(|f| instance_structure(graph, f.instance()))
        .unwrap_or_default()
}

/// The tiles and pairs of a twisted-ice-dimer instance, or `None` when the
/// instance does not fit `graph`.
pub(super) fn instance_structure(graph: &IsingGraph, instance: &Instance) -> Option<Structure> {
    if instance.node_ids().len() != graph.h.len() || instance.edges().len() != graph.edges.len() {
        tracing::warn!("a twisted-ice-dimer instance does not fit its job graph");
        return None;
    }
    let (tiles, _) = local_gauges(instance)?;
    let pairs = parse_pairs(section(instance.metadata(), PAIRS_SECTION)?)?;
    Some(Structure { tiles, pairs })
}

/// Why a job produced no reads.
pub(super) enum Failure {
    Rejected(SampleError),
    Cancelled,
}

/// Descend every start state of one job.
pub(super) fn greedy_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    starts: Option<&[Vec<i8>]>,
    structure: &Structure,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, Failure> {
    let started = Instant::now();
    let declined =
        |e: &dyn std::fmt::Display| Failure::Rejected(SampleError::Declined(e.to_string()));
    let ising = job_ising(graph).map_err(Failure::Rejected)?;
    let sources = MoveSources {
        tiles: &structure.tiles,
        pairs: &structure.pairs,
        listed: &[],
    };
    let (moves, counts) = natural_moves(&ising, &sources).map_err(|e| declined(&e))?;
    let greedy = Greedy::new(ising, &moves, true).map_err(|e| declined(&e))?;
    let warm = starts.is_some();
    let starts = match starts {
        Some(starts) => starts.to_vec(),
        None => {
            tracing::warn!(
                "a job without initial spins descends from random states; \
                 quip-cpu-greedy is a postprocess and should only receive warm jobs"
            );
            random_states(graph.h.len(), params)
        }
    };
    let energy = |s: &[i8]| energy_milli(s, &graph.h, &graph.j, &graph.edges);
    let mut reads = Vec::with_capacity(starts.len());
    let mut before = Vec::with_capacity(starts.len());
    let mut applied = 0;
    for mut spins in starts {
        greedy.check(&spins).map_err(|e| declined(&e))?;
        before.push(energy(&spins));
        applied += greedy.descend(&mut spins, cancel).map_err(|e| match e {
            DescentError::Cancelled => Failure::Cancelled,
            other => declined(&other),
        })?;
        let energy_milli = energy(&spins);
        reads.push(SamplerResult {
            spins,
            energy_milli,
        });
    }
    log_job(
        &counts,
        warm,
        &reads,
        &before,
        applied,
        started.elapsed().as_secs_f64(),
    );
    Ok(reads)
}

pub(super) fn random_states(n: usize, params: &SampleParams) -> Vec<Vec<i8>> {
    let mut rng = SmallRng::seed_from_u64(params.seed);
    (0..params.num_reads)
        .map(|_| {
            (0..n)
                .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
                .collect()
        })
        .collect()
}

fn log_job(
    counts: &MoveCounts,
    warm: bool,
    reads: &[SamplerResult],
    before: &[i64],
    applied: u64,
    seconds: f64,
) {
    let after: Vec<i64> = reads.iter().map(|r| r.energy_milli).collect();
    tracing::info!(
        policy = GREEDY_POLICY,
        warm,
        reads = reads.len(),
        edge_pairs = counts.edge_pairs,
        tile_subsets = counts.tile_subsets,
        pairs = counts.pairs,
        rigid_groups = counts.rigid_groups,
        applied,
        seconds,
        before_milli = ?before,
        after_milli = ?after,
        "greedy postprocess"
    );
}

pub(super) fn into_result(
    reads: Result<Vec<SamplerResult>, Failure>,
) -> Result<Vec<SamplerResult>, SampleError> {
    match reads {
        Ok(reads) => Ok(reads),
        Err(Failure::Rejected(error)) => Err(error),
        // Without a token the descent never reports a cancel.
        Err(Failure::Cancelled) => Err(SampleError::Declined("cancelled".to_owned())),
    }
}

impl Sampler for GreedySampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        check_sample_size(graph.h.len(), graph.edges.len())?;
        into_result(greedy_reads(
            graph,
            params,
            None,
            &Structure::default(),
            None,
        ))
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
        into_result(greedy_reads(
            graph,
            params,
            Some(&warm.spins),
            &Structure::default(),
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
        into_result(greedy_reads(
            graph,
            params,
            warm.map(|w| w.spins.as_slice()),
            &job_structure(graph, Some(family)),
            None,
        ))
    }

    fn declaration() -> Declaration {
        Declaration {
            solver_id: GREEDY_SOLVER_ID,
            generated: &[],
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

    /// Every job comes through here, because the sampler accepts warm starts
    /// and family metadata. The cancel token reaches each descent sweep.
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
                let structure = job_structure(graph, job.family.as_ref());
                let starts = job.warm_start.as_ref().map(|w| w.spins.as_slice());
                greedy_reads(graph, params, starts, &structure, guard).map_err(|failure| {
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

#[cfg(test)]
mod tests {
    use super::{greedy_reads, instance_structure, job_ising, Failure, GreedySampler, Structure};
    use crate::postprocess::greedy::{natural_moves, Greedy, MoveSources};
    use quip_solver_conformance::families::{export_instances, reference_revision};
    use quip_solver_core::ising_generators::family::BaseTopology;
    use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
    use quip_solver_core::{IsingGraph, SampleError, SampleParams, Sampler};

    fn ring(n: usize) -> IsingGraph {
        let edges = (0..n).map(|i| (i, (i + 1) % n)).collect();
        IsingGraph::new(vec![0.0; n], vec![-1.0; n], edges)
    }

    fn params(num_reads: usize, seed: u64) -> SampleParams {
        SampleParams {
            num_reads,
            seed,
            ..SampleParams::default()
        }
    }

    #[test]
    fn a_warm_job_returns_one_descended_read_per_state_in_order() {
        let graph = ring(8);
        let starts = vec![
            vec![1, -1, 1, -1, 1, -1, 1, -1],
            vec![1; 8],
            vec![-1, -1, -1, 1, 1, 1, 1, -1],
        ];
        let reads = greedy_reads(
            &graph,
            &params(8, 0),
            Some(&starts),
            &Structure::default(),
            None,
        )
        .ok()
        .expect("three reads");
        assert_eq!(reads.len(), starts.len(), "one read per supplied state");
        assert_eq!(reads[1].spins, vec![1; 8], "a ground state stays put");
        assert_eq!(reads[1].energy_milli, -8000);
        for (read, start) in reads.iter().zip(&starts) {
            let ising = job_ising(&graph).unwrap();
            assert!(read.energy_milli <= ising.energy(start));
            assert_eq!(read.energy_milli, ising.energy(&read.spins));
        }
    }

    #[test]
    fn a_cold_job_descends_num_reads_seeded_random_states() {
        let graph = ring(16);
        let sampler = GreedySampler::new();
        let first = sampler.sample(&graph, &params(5, 42)).unwrap();
        assert_eq!(first.len(), 5);
        assert_eq!(first, sampler.sample(&graph, &params(5, 42)).unwrap());
        assert_ne!(first, sampler.sample(&graph, &params(5, 43)).unwrap());
    }

    #[test]
    fn repeated_edges_merge_and_self_loops_count_only_in_the_energy() {
        // (0, 1) twice is one coupling of -2; the self-loop is a constant +0.5.
        let graph = IsingGraph::new(
            vec![0.0; 3],
            vec![-1.0, -1.0, -1.0, 0.5],
            vec![(0, 1), (1, 2), (1, 0), (2, 2)],
        );
        let starts = vec![vec![1, -1, 1]];
        let reads = greedy_reads(
            &graph,
            &params(1, 0),
            Some(&starts),
            &Structure::default(),
            None,
        )
        .ok()
        .expect("a read");
        assert_eq!(
            reads[0].spins,
            vec![-1, -1, -1],
            "node 0 flips first, then node 2"
        );
        assert_eq!(
            reads[0].energy_milli,
            quip_protocol::scoring::energy_milli(&reads[0].spins, &graph.h, &graph.j, &graph.edges)
        );
        assert_eq!(reads[0].energy_milli, -2000 - 1000 + 500);
    }

    #[test]
    fn a_coefficient_weight_past_the_bound_is_a_capacity_error() {
        let graph = IsingGraph::new(vec![1.0e16, 0.0], vec![0.0], vec![(0, 1)]);
        assert!(matches!(
            greedy_reads(&graph, &params(1, 0), None, &Structure::default(), None),
            Err(Failure::Rejected(SampleError::Capacity))
        ));
    }

    #[test]
    fn a_start_state_that_does_not_fit_is_declined_without_a_panic() {
        let graph = ring(4);
        for bad in [vec![1, 1, 1], vec![1, 0, 1, 1]] {
            let starts = vec![bad];
            assert!(matches!(
                greedy_reads(
                    &graph,
                    &params(1, 0),
                    Some(&starts),
                    &Structure::default(),
                    None
                ),
                Err(Failure::Rejected(SampleError::Declined(_)))
            ));
        }
    }

    #[test]
    fn a_long_ferromagnetic_chain_prepares_in_linear_time() {
        // One rigid group of every node: pairwise setup would take 10^10 steps.
        let n = 100_000;
        let graph = IsingGraph::new(
            vec![0.0; n],
            vec![-1.0; n - 1],
            (0..n - 1).map(|i| (i, i + 1)).collect(),
        );
        let started = std::time::Instant::now();
        let reads = greedy_reads(&graph, &params(1, 3), None, &Structure::default(), None)
            .ok()
            .expect("a read");
        assert!(started.elapsed().as_secs() < 30, "{:?}", started.elapsed());
        assert_eq!(
            reads[0].energy_milli,
            quip_protocol::scoring::energy_milli(&reads[0].spins, &graph.h, &graph.j, &graph.edges)
        );
    }

    #[test]
    fn a_twisted_dimer_job_descends_with_its_tiles_and_pairs() {
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
        let structure = instance_structure(&graph, &instance).expect("the family structure");
        assert!(!structure.tiles.is_empty() && !structure.pairs.is_empty());
        let short = IsingGraph::new(graph.h[1..].to_vec(), graph.j.clone(), graph.edges.to_vec());
        assert!(
            instance_structure(&short, &instance).is_none(),
            "a node count mismatch"
        );

        let reads = greedy_reads(&graph, &params(2, 7), None, &structure, None)
            .ok()
            .expect("two reads");
        let ising = job_ising(&graph).unwrap();
        let sources = MoveSources {
            tiles: &structure.tiles,
            pairs: &structure.pairs,
            listed: &[],
        };
        let (moves, counts) = natural_moves(&ising, &sources).unwrap();
        assert!(counts.tile_subsets > 0);
        let greedy = Greedy::new(ising, &moves, true).unwrap();
        for read in &reads {
            for m in 0..greedy.num_moves() {
                assert!(
                    greedy.delta(&read.spins, m).unwrap() >= 0,
                    "move {m} still lowers the energy"
                );
            }
        }
    }
}
