//! The `quip-cpu-tileworm` sampler: the tile worm postprocess for the
//! selected charged-ice and field-ice cells.
//!
//! A validated `quip/charged-ice` revision 1 job with the selected recipe
//! runs `ice-tiles-charge/1`, and a `quip/ice-fields` revision 1 job with one
//! of the two selected recipes runs `ice-tiles-field/1`. Each runs the worm
//! schedule against the declared 0.25 s budget on the supplied states. The
//! sampler returns the final state first, then the descended state of each
//! supplied read, in input order. Every returned state is rescored from the
//! job graph. The schedule seeds from the job seed.
//!
//! The tile metadata and the job coefficients are checked before any state is
//! touched. A selected job that fails the check is declined.
//!
//! Every other job runs the `greedy-nat/1` descent of `quip-cpu-greedy`, so
//! the binary still serves plain and warm-start jobs. A job without states
//! starts from random states and logs a warning, because the binary is a
//! postprocess.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use quip_protocol::scoring::energy_milli;
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::ice::{
    CHARGED_ICE_V1, ICE_CHARGE_RECIPE, ICE_FIELDS_RECIPES, ICE_FIELDS_V1,
};
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::{
    BackendIdentity, CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult,
    StreamResult, WarmStart, WarmStreamJob,
};

use super::greedy::{natural_moves, DescentError, Greedy, MoveSources};
use super::sampler::{greedy_reads, into_result, job_ising, job_structure, random_states, Failure};
use super::tile_worm::{
    schedule_tiles, Policy, TileCounts, TileReport, TileSpec, TileWorm, Work, CHARGE_POLICY,
    FIELD_POLICY,
};
use crate::families::otsuka_sampler::section;
use crate::sa_sampler::coefficient_milli;
use crate::{check_sample_size, run_job_pump, StreamKernelError};

/// The stable id of this binary (FAM-60).
pub const TILE_WORM_SOLVER_ID: &str = "quip-cpu-tileworm/1";

/// Backend identity for `quip-cpu-tileworm`. The session adds the
/// `initial-spins` feature, because the sampler accepts warm starts.
pub const CPU_TILE_WORM_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_proto::v1::Backend::Cpu,
    algorithm: quip_proto::v1::Algorithm::TileWorm,
    max_nodes: crate::DEFAULT_MAX_NODES,
    max_edges: crate::DEFAULT_MAX_EDGES,
    features: &["streaming"],
    adapt: crate::CPU_ADAPT,
    limits: crate::F64_LIMITS,
};

/// The canonical parameters of the selected charged-ice cell.
const CHARGE_PARAMS: &[u8] = ICE_CHARGE_RECIPE.params;

/// The canonical parameters of the two selected field-ice cells.
const FIELD_PARAMS: &[&[u8]] = &[ICE_FIELDS_RECIPES[0].params, ICE_FIELDS_RECIPES[1].params];

/// The binary accepts explicit jobs of the three selected recipes. It
/// generates no family work: a lease salt carries no states.
const ACCEPTED: &[FamilyClaim] = &[
    FamilyClaim {
        revisions: &[CHARGED_ICE_V1],
        only_params: &[CHARGE_PARAMS],
    },
    FamilyClaim {
        revisions: &[ICE_FIELDS_V1],
        only_params: FIELD_PARAMS,
    },
];

/// The metadata section that lists the tiles, in every ice family.
const TILES_SECTION: u16 = 1;

/// The tile worm postprocess, with the greedy descent for other jobs.
#[derive(Debug, Default)]
pub struct TileWormSampler {
    /// 0 means use host parallelism.
    num_cpus: Arc<AtomicUsize>,
}

impl TileWormSampler {
    /// A sampler that runs one job per core.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// The selected policy of a family revision with these parameters, if the
/// recipe is selected.
fn policy_for(family_id: &str, revision: u32, params: &[u8]) -> Option<Policy> {
    if family_id == CHARGED_ICE_V1.family_id && revision == CHARGED_ICE_V1.revision {
        (params == CHARGE_PARAMS).then_some(CHARGE_POLICY)
    } else if family_id == ICE_FIELDS_V1.family_id && revision == ICE_FIELDS_V1.revision {
        FIELD_PARAMS.contains(&params).then_some(FIELD_POLICY)
    } else {
        None
    }
}

/// The tiles of a charged-ice or ice-fields instance, in dense node
/// positions, checked against the job graph.
///
/// # Errors
///
/// A description of the first check that fails: the instance does not fit
/// the graph, the coefficients differ, the tile section is malformed, or a
/// field sits on a spin that is not in two tiles.
pub(super) fn instance_tiles(
    graph: &IsingGraph,
    instance: &Instance,
    fields: bool,
) -> Result<Vec<[u32; 4]>, String> {
    let n = graph.h.len();
    if instance.node_ids().len() != n || instance.edges().len() != graph.edges.len() {
        return Err("the instance does not fit its job graph".to_owned());
    }
    let same = |c: &f64, milli: i32| coefficient_milli(*c) == Some(i64::from(milli));
    if !graph.h.iter().zip(instance.h()).all(|(c, &m)| same(c, m))
        || !graph.j.iter().zip(instance.j()).all(|(c, &m)| same(c, m))
        || !graph
            .edges
            .iter()
            .zip(instance.edges())
            .all(|(&(u, v), &(a, b))| u32::try_from(u) == Ok(a) && u32::try_from(v) == Ok(b))
    {
        return Err("the job coefficients differ from the instance".to_owned());
    }
    let payload = section(instance.metadata(), TILES_SECTION)
        .ok_or_else(|| "the instance has no tile section".to_owned())?;
    let tiles =
        parse_tiles(payload, n).ok_or_else(|| "the tile section is malformed".to_owned())?;
    if fields {
        check_fields(instance.h(), &tiles)?;
    }
    Ok(tiles)
}

/// A tile section: a `u32` count, then four `u32` positions per tile, each
/// tile ascending and the tiles ascending, all inside the model.
fn parse_tiles(payload: &[u8], n: usize) -> Option<Vec<[u32; 4]>> {
    let (count, rest) = payload.split_first_chunk::<4>()?;
    let count = usize::try_from(u32::from_le_bytes(*count)).ok()?;
    let (rows, tail) = rest.as_chunks::<16>();
    if rows.len() != count || !tail.is_empty() {
        return None;
    }
    let mut tiles: Vec<[u32; 4]> = Vec::with_capacity(count);
    for row in rows {
        let at = |k: usize| {
            u32::from_le_bytes([row[4 * k], row[4 * k + 1], row[4 * k + 2], row[4 * k + 3]])
        };
        let tile = [at(0), at(1), at(2), at(3)];
        let ascending = tile.windows(2).all(|w| w[0] < w[1]);
        if !ascending
            || usize::try_from(tile[3]).ok()? >= n
            || tiles.last().is_some_and(|prev| *prev >= tile)
        {
            return None;
        }
        tiles.push(tile);
    }
    Some(tiles)
}

/// Every nonzero field has the same magnitude and sits on a spin that is in
/// exactly two tiles.
fn check_fields(h: &[i32], tiles: &[[u32; 4]]) -> Result<(), String> {
    let mut seen = vec![0_u8; h.len()];
    for &u in tiles.iter().flatten() {
        let slot = seen
            .get_mut(u as usize)
            .ok_or("a tile is outside the model")?;
        *slot = slot.saturating_add(1);
    }
    let mut magnitude = None;
    for (&field, &count) in h.iter().zip(&seen) {
        if field == 0 {
            continue;
        }
        let m = field.unsigned_abs();
        if count != 2 || *magnitude.get_or_insert(m) != m {
            return Err("a field is not a single magnitude on a two-tile spin".to_owned());
        }
    }
    Ok(())
}

/// Run one job: the tile worm schedule for a selected recipe, the greedy
/// descent for every other job.
fn job_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    starts: Option<&[Vec<i8>]>,
    family: Option<&FamilyContext>,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, Failure> {
    let selected = family.and_then(|f| {
        policy_for(f.family_id(), f.revision(), f.params()).map(|policy| (f, policy))
    });
    match selected {
        Some((family, policy)) => {
            let fields = policy == FIELD_POLICY;
            let tiles = instance_tiles(graph, family.instance(), fields)
                .map_err(|e| Failure::Rejected(SampleError::Declined(e)))?;
            tile_reads(graph, params, starts, &tiles, policy, cancel)
        }
        None => greedy_reads(graph, params, starts, &job_structure(graph, family), cancel),
    }
}

/// The tile worm schedule of `policy` on the states of one job.
fn tile_reads(
    graph: &IsingGraph,
    params: &SampleParams,
    starts: Option<&[Vec<i8>]>,
    tiles: &[[u32; 4]],
    policy: Policy,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<SamplerResult>, Failure> {
    let declined =
        |e: &dyn std::fmt::Display| Failure::Rejected(SampleError::Declined(e.to_string()));
    let spec = TileSpec::parse(policy.spec).map_err(|e| declined(&e))?;
    let ising = job_ising(graph).map_err(Failure::Rejected)?;
    // The tile metadata is checked before any state is touched.
    let worm = TileWorm::new(&ising, tiles, &spec).map_err(|e| declined(&e))?;
    let sources = MoveSources {
        tiles,
        pairs: &[],
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
                 quip-cpu-tileworm is a postprocess and should only receive warm jobs"
            );
            random_states(graph.h.len(), params)
        }
    };
    let work = Work::Budget {
        seconds: policy.budget_seconds,
    };
    let out =
        schedule_tiles(&greedy, &worm, &starts, params.seed, work, cancel).map_err(
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
    log_job(
        &policy,
        &out.report,
        worm.tables().counts(),
        warm,
        starts.len(),
    );
    Ok(reads)
}

fn log_job(policy: &Policy, report: &TileReport, counts: TileCounts, warm: bool, supplied: usize) {
    tracing::info!(
        policy = policy.id,
        spec = policy.spec,
        budget_seconds = policy.budget_seconds,
        work = ?report.work,
        elapsed_seconds = report.elapsed_seconds,
        warm,
        supplied,
        reads_used = report.reads_used,
        visited = ?report.visited,
        start_milli = report.start_energy,
        worms = report.stats.worms,
        accepted = report.stats.accepted,
        improved = report.stats.improved,
        failed = report.stats.failed,
        free_flips = report.stats.free_flips,
        tail_improved = ?report.tail_improved,
        polish_milli = report.polish,
        energy_milli = report.energy,
        tiles = counts.tiles,
        tiled_spins = counts.tiled_spins,
        free_spins = counts.free_spins,
        "tile-worm postprocess"
    );
}

impl Sampler for TileWormSampler {
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
            solver_id: TILE_WORM_SOLVER_ID,
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
    /// and family metadata. The cancel token reaches each 64-worm block of
    /// the schedule and each descent sweep.
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
    use super::{
        check_fields, instance_tiles, parse_tiles, policy_for, tile_reads, TileWormSampler,
        CHARGE_PARAMS, FIELD_PARAMS,
    };
    use crate::postprocess::sampler::Failure;
    use crate::postprocess::tile_worm::{
        Policy, CHARGE_POLICY, CHARGE_SPEC, FIELD_POLICY, FIELD_SPEC, TILE_BUDGET_SECONDS,
    };
    use quip_solver_conformance::families::{export_instances, reference_revision};
    use quip_solver_core::families::FamilyTable;
    use quip_solver_core::ising_generators::descriptor::Contract;
    use quip_solver_core::ising_generators::family::BaseTopology;
    use quip_solver_core::ising_generators::ice::{
        CHARGED_ICE_V1, ICE_CHARGE_RECIPE, ICE_FIELDS_RECIPES, ICE_FIELDS_V1,
    };
    use quip_solver_core::ising_generators::instance::Instance;
    use quip_solver_core::{IsingGraph, SampleError, SampleParams, Sampler};

    fn params(num_reads: usize, seed: u64) -> SampleParams {
        SampleParams {
            num_reads,
            seed,
            ..SampleParams::default()
        }
    }

    /// An instance of the reference revision of `contract`, and its graph.
    fn reference_job(contract: &Contract) -> (Instance, IsingGraph) {
        let reference = reference_revision(contract.family_id, contract.revision)
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

    fn starts(n: usize, count: u64) -> Vec<Vec<i8>> {
        (0..count)
            .map(|k| {
                (0..n)
                    .map(|i| {
                        if (i as u64 * 7 + k).is_multiple_of(3) {
                            1
                        } else {
                            -1
                        }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn the_declaration_accepts_exactly_the_three_selected_recipes() {
        let table: FamilyTable = TileWormSampler::declaration()
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
        assert_eq!(table.accepted().len(), 2);
        let (id, rev) = (CHARGED_ICE_V1.family_id, CHARGED_ICE_V1.revision);
        assert_eq!(
            policy_for(id, rev, ICE_CHARGE_RECIPE.params),
            Some(CHARGE_POLICY)
        );
        let (id, rev) = (ICE_FIELDS_V1.family_id, ICE_FIELDS_V1.revision);
        assert_eq!(FIELD_PARAMS.len(), ICE_FIELDS_RECIPES.len());
        for recipe in ICE_FIELDS_RECIPES {
            assert_eq!(
                policy_for(id, rev, recipe.params),
                Some(FIELD_POLICY),
                "{}",
                recipe.alias
            );
            assert_eq!(policy_for(CHARGED_ICE_V1.family_id, 1, recipe.params), None);
        }
        assert_eq!(policy_for(id, rev, CHARGE_PARAMS), None);
        assert_eq!(policy_for(id, rev + 1, FIELD_PARAMS[0]), None);
        assert_eq!(policy_for("quip/twisted-ice-dimer", 1, CHARGE_PARAMS), None);
        assert_eq!(CHARGE_POLICY.spec, CHARGE_SPEC);
        assert_eq!(FIELD_POLICY.spec, FIELD_SPEC);
        assert_eq!(FIELD_POLICY.budget_seconds, TILE_BUDGET_SECONDS);
    }

    fn check_job(contract: &Contract, policy: Policy) {
        let (instance, graph) = reference_job(contract);
        let tiles = instance_tiles(&graph, &instance, policy == FIELD_POLICY).expect("tiles");
        assert!(!tiles.is_empty());
        let starts = starts(graph.h.len(), 4);
        let began = std::time::Instant::now();
        let reads = tile_reads(&graph, &params(4, 11), Some(&starts), &tiles, policy, None)
            .ok()
            .expect("reads");
        assert!(began.elapsed().as_secs_f64() < policy.budget_seconds + 1.0);
        assert_eq!(
            reads.len(),
            1 + starts.len(),
            "the final state, then one read each"
        );
        for read in &reads {
            assert_eq!(read.spins.len(), graph.h.len());
            assert_eq!(
                read.energy_milli,
                quip_protocol::scoring::energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
            );
        }
        let best = reads[1..].iter().map(|r| r.energy_milli).min().unwrap();
        assert!(reads[0].energy_milli <= best, "the final state leads");
    }

    #[test]
    fn a_charged_ice_job_returns_the_final_state_then_the_descended_reads() {
        check_job(&CHARGED_ICE_V1, CHARGE_POLICY);
    }

    #[test]
    fn a_field_ice_job_returns_the_final_state_then_the_descended_reads() {
        check_job(&ICE_FIELDS_V1, FIELD_POLICY);
    }

    #[test]
    fn coefficients_that_differ_from_the_instance_are_declined() {
        for (contract, fields) in [(&CHARGED_ICE_V1, false), (&ICE_FIELDS_V1, true)] {
            let (instance, graph) = reference_job(contract);
            let mut changed = graph.clone();
            changed.h[0] += 0.001;
            assert!(instance_tiles(&changed, &instance, fields).is_err());
            let mut changed = graph.clone();
            changed.j[0] += 0.001;
            assert!(instance_tiles(&changed, &instance, fields).is_err());
            let last = graph.edges.len() - 1;
            let changed = IsingGraph::new(
                graph.h.to_vec(),
                graph.j[..last].to_vec(),
                graph.edges[..last].to_vec(),
            );
            assert!(instance_tiles(&changed, &instance, fields).is_err());
        }
    }

    #[test]
    fn malformed_tile_metadata_is_rejected_before_processing() {
        let tile = |nodes: [u32; 4]| {
            nodes
                .iter()
                .flat_map(|n| n.to_le_bytes())
                .collect::<Vec<u8>>()
        };
        let section = |tiles: &[[u32; 4]]| {
            let mut bytes = u32::try_from(tiles.len()).unwrap().to_le_bytes().to_vec();
            for t in tiles {
                bytes.extend(tile(*t));
            }
            bytes
        };
        assert_eq!(
            parse_tiles(&section(&[[0, 1, 2, 3], [2, 3, 4, 5]]), 6).map(|t| t.len()),
            Some(2)
        );
        assert_eq!(parse_tiles(&section(&[]), 6), Some(Vec::new()));
        // Outside the model, unordered corners, unordered tiles, a repeated tile.
        assert_eq!(parse_tiles(&section(&[[0, 1, 2, 6]]), 6), None);
        assert_eq!(parse_tiles(&section(&[[0, 2, 1, 3]]), 6), None);
        assert_eq!(
            parse_tiles(&section(&[[2, 3, 4, 5], [0, 1, 2, 3]]), 6),
            None
        );
        assert_eq!(
            parse_tiles(&section(&[[0, 1, 2, 3], [0, 1, 2, 3]]), 6),
            None
        );
        // A wrong count, and trailing bytes.
        let mut short = section(&[[0, 1, 2, 3]]);
        short[0] = 2;
        assert_eq!(parse_tiles(&short, 6), None);
        let mut long = section(&[[0, 1, 2, 3]]);
        long.push(0);
        assert_eq!(parse_tiles(&long, 6), None);
        assert_eq!(parse_tiles(&[0, 0], 6), None);
    }

    #[test]
    fn a_field_must_sit_on_a_two_tile_spin_with_one_magnitude() {
        let tiles = [[0, 1, 2, 3], [3, 4, 5, 6]];
        // Spin 3 is in both tiles.
        assert!(check_fields(&[0, 0, 0, 500, 0, 0, 0], &tiles).is_ok());
        assert!(check_fields(&[0; 7], &tiles).is_ok());
        assert!(
            check_fields(&[500, 0, 0, 0, 0, 0, 0], &tiles).is_err(),
            "a one-tile spin"
        );
        assert!(
            check_fields(&[0, 0, 0, 0, 0, 0, 0, 500], &tiles).is_err(),
            "a spin outside the tiles"
        );
        let two = [[0, 1, 2, 3], [3, 4, 5, 6], [0, 4, 7, 8]];
        assert!(check_fields(&[500, 0, 0, 500, 500, 0, 0, 0, 0], &two).is_ok());
        assert!(
            check_fields(&[500, 0, 0, 1000, 0, 0, 0, 0, 0], &two).is_err(),
            "two magnitudes"
        );
    }

    #[test]
    fn a_job_the_worm_cannot_serve_is_declined_without_touching_a_state() {
        let (instance, graph) = reference_job(&CHARGED_ICE_V1);
        let mut tiles = instance_tiles(&graph, &instance, false).expect("tiles");
        let n = u32::try_from(graph.h.len()).unwrap();
        for bad in [vec![], vec![[0, 1, 2, n]], vec![[0, 1, 1, 2]]] {
            assert!(matches!(
                tile_reads(&graph, &params(1, 0), None, &bad, CHARGE_POLICY, None),
                Err(Failure::Rejected(SampleError::Declined(_)))
            ));
        }
        // A state of the wrong length.
        let short = vec![vec![1_i8; graph.h.len() - 1]];
        tiles.truncate(2);
        assert!(matches!(
            tile_reads(
                &graph,
                &params(1, 0),
                Some(&short),
                &tiles,
                CHARGE_POLICY,
                None
            ),
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
        let sampler = TileWormSampler::new();
        let reads = sampler.sample(&graph, &params(3, 5)).unwrap();
        assert_eq!(reads.len(), 3, "one descended read per random state");
    }
}
