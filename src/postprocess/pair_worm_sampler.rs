//! The `quip-cpu-pairworm` sampler: the `ice-pairs-pw/1` postprocess for the
//! selected twisted-ice controls.
//!
//! A validated `quip/twisted-ice-dimer` revision 1 job with one of the three
//! selected recipes runs the pair-worm schedule against its declared budget
//! on the supplied states. It returns the final state first, then the
//! descended state of each visited read, in visit order. Every returned state
//! is rescored from the job graph. The schedule seeds from the job seed.
//!
//! Every other job runs the `greedy-nat/1` descent of `quip-cpu-greedy`, so
//! the binary still serves plain and warm-start jobs. A job without states
//! starts from random states and logs a warning, because the binary is a
//! postprocess.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use quip_protocol::scoring::energy_milli;
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::ice::twisted_dimer::RECIPES;
use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_core::{
    BackendIdentity, CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult,
    StreamResult, WarmStart, WarmStreamJob,
};

use super::greedy::{natural_moves, DescentError, Greedy, MoveSources};
use super::pair_worm::{schedule_pw, PairWorm, PwReport, TileCounts, Work};
use super::sampler::{
    greedy_reads, instance_structure, into_result, job_ising, job_structure, random_states,
    Failure, Structure,
};
use crate::{check_sample_size, run_job_pump, StreamKernelError};

/// The stable id of this binary (FAM-60).
pub const PAIR_WORM_SOLVER_ID: &str = "quip-cpu-pairworm/1";

/// Backend identity for `quip-cpu-pairworm`. The session adds the
/// `initial-spins` feature, because the sampler accepts warm starts.
pub const CPU_PAIR_WORM_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_proto::v1::Backend::Cpu,
    algorithm: quip_proto::v1::Algorithm::PairWorm,
    max_nodes: crate::DEFAULT_MAX_NODES,
    max_edges: crate::DEFAULT_MAX_EDGES,
    features: &["streaming"],
    adapt: crate::CPU_ADAPT,
    limits: crate::F64_LIMITS,
};

/// The canonical parameters of the three selected twisted-ice controls.
const SELECTED: &[&[u8]] = &[RECIPES[0].params, RECIPES[1].params, RECIPES[2].params];

/// The binary accepts explicit twisted-ice-dimer jobs of the selected
/// recipes. It generates no family work: a lease salt carries no states.
const ACCEPTED: &[FamilyClaim] = &[FamilyClaim {
    revisions: &[TWISTED_DIMER_V1],
    only_params: SELECTED,
}];

/// The pair-worm postprocess, with the greedy descent for other jobs.
#[derive(Debug, Default)]
pub struct PairWormSampler {
    /// 0 means use host parallelism.
    num_cpus: Arc<AtomicUsize>,
}

impl PairWormSampler {
    /// A sampler that runs one job per core.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Whether `family` is a twisted-ice-dimer job of a selected recipe.
fn is_selected(family: &FamilyContext) -> bool {
    family.family_id() == TWISTED_DIMER_V1.family_id
        && family.revision() == TWISTED_DIMER_V1.revision
        && SELECTED.contains(&family.params())
}

/// Run one job: the pair-worm schedule for a selected recipe, the greedy
/// descent for every other job.
fn job_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    starts: Option<&[Vec<i8>]>,
    family: Option<&FamilyContext>,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, Failure> {
    let selected = family
        .filter(|f| is_selected(f))
        .and_then(|f| instance_structure(graph, f.instance()));
    match selected {
        Some(structure) => pw_reads(graph, params, starts, &structure, cancel),
        None => greedy_reads(graph, params, starts, &job_structure(graph, family), cancel),
    }
}

/// The pair-worm schedule on the states of one job.
fn pw_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    starts: Option<&[Vec<i8>]>,
    structure: &Structure,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, Failure> {
    let declined =
        |e: &dyn std::fmt::Display| Failure::Rejected(SampleError::Declined(e.to_string()));
    let ising = job_ising(graph).map_err(Failure::Rejected)?;
    // The pair metadata is checked before any state is touched.
    let pw = PairWorm::new(&ising, &structure.tiles, &structure.pairs).map_err(|e| declined(&e))?;
    let sources = MoveSources {
        tiles: &structure.tiles,
        pairs: &structure.pairs,
        listed: &[],
    };
    let (moves, _) = natural_moves(&ising, &sources).map_err(|e| declined(&e))?;
    let greedy = Greedy::new(ising, &moves, true).map_err(|e| declined(&e))?;
    let warm = starts.is_some();
    let starts = match starts {
        Some(starts) => starts.to_vec(),
        None => {
            tracing::warn!(
                "a job without initial spins starts from random states; \
                 quip-cpu-pairworm is a postprocess and should only receive warm jobs"
            );
            random_states(graph.h.len(), params)
        }
    };
    let out = schedule_pw(&greedy, &pw, &starts, params.seed, Work::Budget, cancel).map_err(
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
    log_job(&out.report, pw.counts(), warm, starts.len());
    Ok(reads)
}

fn log_job(report: &PwReport, counts: TileCounts, warm: bool, supplied: usize) {
    tracing::info!(
        policy = report.policy,
        budget_seconds = report.budget_seconds,
        work = ?report.work,
        elapsed_seconds = report.elapsed_seconds,
        warm,
        supplied,
        reads_used = report.reads_used,
        visited = ?report.visited,
        read_seconds = ?report.read_seconds,
        crossover_blocks = report.crossover_blocks,
        crossover_rounds = report.crossover_rounds,
        pre_crossover_milli = report.pre_crossover,
        pre_anneal_milli = report.pre_anneal,
        anneal_cycles = report.anneal_cycles,
        energy_milli = report.energy,
        tiles = counts.tiles,
        dropped_tiles = counts.dropped,
        pairs = counts.pairs,
        "pair-worm postprocess"
    );
}

impl Sampler for PairWormSampler {
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
            solver_id: PAIR_WORM_SOLVER_ID,
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
    /// and family metadata. The cancel token reaches each read and anneal
    /// cycle of the schedule, and each descent sweep.
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

#[cfg(test)]
mod tests {
    use super::{pw_reads, PairWormSampler, SELECTED};
    use crate::postprocess::pair_worm::PW_BUDGET_SECONDS;
    use crate::postprocess::sampler::{instance_structure, Failure};
    use quip_solver_conformance::families::{export_instances, reference_revision};
    use quip_solver_core::families::FamilyTable;
    use quip_solver_core::ising_generators::family::BaseTopology;
    use quip_solver_core::ising_generators::ice::twisted_dimer::RECIPES;
    use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
    use quip_solver_core::ising_generators::instance::Instance;
    use quip_solver_core::{IsingGraph, SampleError, SampleParams, Sampler};

    fn params(num_reads: usize, seed: u64) -> SampleParams {
        SampleParams {
            num_reads,
            seed,
            ..SampleParams::default()
        }
    }

    /// A twisted-ice-dimer instance of the reference revision, and its graph.
    fn reference_job() -> (Instance, IsingGraph) {
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
        (instance, graph)
    }

    #[test]
    fn the_declaration_accepts_exactly_the_three_selected_recipes() {
        let table: FamilyTable = PairWormSampler::declaration()
            .validate()
            .expect("a valid declaration");
        // Declares no generated revision; registered families are generated
        // only on the solver-core default session path.
        assert_eq!(
            quip_solver_core::ising_generators::registry::CONTRACTS
                .iter()
                .filter(|c| table.declares_generated(c.family_id, c.revision))
                .count(),
            0
        );
        assert_eq!(table.accepted().len(), 1);
        assert_eq!(SELECTED.len(), RECIPES.len());
        for recipe in RECIPES {
            assert!(SELECTED.contains(&recipe.params), "{}", recipe.alias);
        }
    }

    #[test]
    fn a_selected_job_returns_the_final_state_then_the_pool_all_rescored() {
        let (instance, graph) = reference_job();
        let structure = instance_structure(&graph, &instance).expect("the family structure");
        let starts: Vec<Vec<i8>> = (0..4_u64)
            .map(|k| {
                (0..graph.h.len())
                    .map(|i| {
                        if (i as u64 * 7 + k).is_multiple_of(3) {
                            1
                        } else {
                            -1
                        }
                    })
                    .collect()
            })
            .collect();
        let started = std::time::Instant::now();
        let reads = pw_reads(&graph, &params(4, 11), Some(&starts), &structure, None)
            .ok()
            .expect("reads");
        assert!(started.elapsed().as_secs_f64() < PW_BUDGET_SECONDS + 1.0);
        assert!(reads.len() >= 2 && reads.len() <= 1 + starts.len());
        for read in &reads {
            assert_eq!(
                read.energy_milli,
                quip_protocol::scoring::energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
            );
        }
        let best = reads[1..].iter().map(|r| r.energy_milli).min().unwrap();
        assert!(reads[0].energy_milli <= best, "the final state leads");
    }

    #[test]
    fn invalid_pair_metadata_is_declined_before_processing() {
        let (instance, graph) = reference_job();
        let mut structure = instance_structure(&graph, &instance).expect("the family structure");
        let (a, _) = structure.pairs[0];
        let (_, d) = structure.pairs[1];
        // Node `a` now has two partners.
        structure.pairs.push((a, d));
        assert!(matches!(
            pw_reads(&graph, &params(1, 0), None, &structure, None),
            Err(Failure::Rejected(SampleError::Declined(_)))
        ));
    }

    #[test]
    fn a_plain_job_runs_the_greedy_descent() {
        let n = 8;
        let graph = IsingGraph::new(
            vec![0.0; n],
            vec![-1.0; n],
            (0..n).map(|i| (i, (i + 1) % n)).collect(),
        );
        let sampler = PairWormSampler::new();
        let reads = sampler.sample(&graph, &params(3, 5)).unwrap();
        assert_eq!(reads.len(), 3, "one descended read per random state");
    }
}
