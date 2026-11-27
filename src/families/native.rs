//! CPU-native population solver for the standard family lease protocol.

mod config;
mod pipeline;

pub use config::NativeConfig;
pub use pipeline::{instance_energy, prepared, tuned_reads};
#[cfg(feature = "zephyr-spine-polish")]
pub use pipeline::{polish_zephyr, zephyr_shape};

use crate::sampler_core::sample_generic_sa;
use crate::{
    apply_num_cpus, check_sample_size, core_budget, reject_num_cpus, run_job_pump,
    StreamKernelError,
};
use pipeline::graph_of;
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::{
    family::SaltFault, ice::TWISTED_DIMER_V1, instance::Instance,
};
use quip_solver_core::{
    CancelToken, IsingGraph, Lease, LeaseSink, SampleError, SampleParams, Sampler, SamplerResult,
    StreamResult, TopologyView, WarmStart, WarmStreamJob,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

const CLAIMS: &[FamilyClaim] = &[FamilyClaim {
    revisions: &[TWISTED_DIMER_V1],
    only_params: &[],
}];

#[derive(Default)]
struct Slots {
    active: Mutex<usize>,
    changed: Condvar,
}

struct Slot<'a>(&'a Slots);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        *self.0.active.lock().unwrap_or_else(PoisonError::into_inner) -= 1;
        self.0.changed.notify_one();
    }
}

/// Experimental native CPU sampler with bounded concurrent lease workers.
pub struct NativeSampler {
    num_cpus: AtomicUsize,
    config: Arc<Mutex<NativeConfig>>,
    slots: Slots,
}

impl Default for NativeSampler {
    fn default() -> Self {
        Self {
            num_cpus: AtomicUsize::new(crate::host_parallelism().min(2)),
            config: Arc::new(Mutex::new(NativeConfig::from_environment())),
            slots: Slots::default(),
        }
    }
}

impl NativeSampler {
    /// Create a solver with environment defaults and backend TOML overrides.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn acquire(&self, sink: &LeaseSink) -> Option<Slot<'_>> {
        let mut active = self
            .slots
            .active
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        loop {
            if sink.is_stopped() {
                return None;
            }
            if *active < self.stream_width() {
                *active += 1;
                return Some(Slot(&self.slots));
            }
            active = self
                .slots
                .changed
                .wait_timeout(active, Duration::from_millis(5))
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
}

fn supports(family_id: &str, revision: u32) -> bool {
    family_id == TWISTED_DIMER_V1.family_id && revision == TWISTED_DIMER_V1.revision
}

fn solve_family(
    instance: &Instance,
    descriptor: &[u8],
    params: &SampleParams,
    config: &NativeConfig,
    stopped: impl Fn() -> bool,
) -> Result<Option<Vec<SamplerResult>>, SampleError> {
    let Some(reads) = tuned_reads(instance, params, config, &stopped)? else {
        return Ok(None);
    };
    #[cfg(feature = "zephyr-spine-polish")]
    if config.zephyr_blocks > 0 {
        let mut polished = reads.clone();
        match polish_zephyr(
            instance,
            descriptor,
            config,
            &mut polished,
            super::otsuka::SpinePolishOptions {
                blocks: config.zephyr_blocks,
                span: 4,
                batch_reads: true,
                cache_plans: true,
            },
            params.seed,
            &stopped,
        ) {
            Ok(Some(_)) => return Ok(Some(polished)),
            Ok(None) => return Ok(None),
            Err(SampleError::Capacity) => {
                tracing::debug!("Zephyr polish exceeded its bounds; retaining the anneal results");
            }
            Err(error) => return Err(error),
        }
    }
    let _ = descriptor;
    Ok(Some(reads))
}

fn work_generic_lease(
    lease: &Lease,
    topology: &TopologyView,
    params: &SampleParams,
    sink: &LeaseSink,
) -> Result<(), SampleError> {
    let cancel = CancelToken::default();
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !done.load(Ordering::Relaxed) {
                if sink.is_stopped() {
                    cancel.cancel_through(1);
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        });
        let result = (|| {
            for index in 0..lease.salt_count() {
                if sink.is_stopped() {
                    break;
                }
                let nonce = lease.nonce(index);
                let instance = if let Some(family) = lease.family() {
                    match family.generate(nonce) {
                        Ok(instance) => Some(instance),
                        Err(SaltFault::Skip(_)) => {
                            if sink.skip(index).is_err() {
                                break;
                            }
                            continue;
                        }
                        Err(SaltFault::Defect(error)) => {
                            return Err(SampleError::DeviceFault(format!(
                                "canonical family generation failed: {error:?}"
                            )));
                        }
                    }
                } else {
                    None
                };
                let graph = if let Some(instance) = &instance {
                    graph_of(instance)
                } else {
                    let (h, j) = topology.draw(nonce).map_err(|_| SampleError::Capacity)?;
                    let milli = |v| f64::from(v) / 1000.0;
                    IsingGraph::new(
                        h.into_iter().map(milli).collect(),
                        j.into_iter().map(milli).collect(),
                        topology.edges.clone(),
                    )
                };
                check_sample_size(graph.h.len(), graph.edges.len())?;
                let Ok(mut reads) = sample_generic_sa(&graph, params, Some((&cancel, Some(1))))
                else {
                    break;
                };
                if let Some(instance) = &instance {
                    for read in &mut reads {
                        read.energy_milli = instance_energy(instance, &read.spins);
                    }
                }
                if sink.push(index, reads).is_err() {
                    break;
                }
            }
            Ok(())
        })();
        done.store(true, Ordering::Relaxed);
        result
    })
}

impl Sampler for NativeSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        check_sample_size(graph.h.len(), graph.edges.len())?;
        Ok(crate::sample_ising(graph, params, crate::Algorithm::Sa))
    }

    fn accepts_family_metadata() -> bool {
        true
    }

    fn generates_locally() -> bool {
        true
    }

    fn declaration() -> Declaration {
        Declaration {
            solver_id: "quip-cpu-native/1",
            generated: CLAIMS,
            accepted: CLAIMS,
        }
    }

    fn stream_width(&self) -> usize {
        core_budget(&self.num_cpus)
    }

    fn declared_stream_width() -> u32 {
        0
    }

    fn max_reads(&self) -> u32 {
        128
    }

    fn apply_config(&self, backend_toml: &str) {
        if let Err(error) = apply_num_cpus(&self.num_cpus, backend_toml) {
            reject_num_cpus(error);
        }
        let table = match backend_toml.parse::<toml::Table>() {
            Ok(table) => table,
            Err(error) => {
                tracing::error!(%error, "CPU-native TOML rejected");
                return;
            }
        };
        let mut stored = self.config.lock().unwrap_or_else(PoisonError::into_inner);
        let mut updated = stored.clone();
        match updated.apply(&table) {
            Ok(()) => *stored = updated,
            Err(error) => {
                tracing::error!(%error, "CPU-native config rejected; retaining previous settings");
            }
        }
    }

    fn sample_family(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        _warm: Option<&WarmStart>,
        family: &FamilyContext,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        if !supports(family.family_id(), family.revision()) {
            return self.sample(graph, params);
        }
        let config = self
            .config
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        Ok(
            solve_family(family.instance(), family.params(), params, &config, || {
                false
            })?
            .unwrap_or_default(),
        )
    }

    fn sample_lease(
        &self,
        lease: &Lease,
        topology: &TopologyView,
        params: &SampleParams,
        sink: &LeaseSink,
    ) -> Result<(), SampleError> {
        let Some(_slot) = self.acquire(sink) else {
            return Ok(());
        };
        let Some(family) = lease
            .family()
            .filter(|family| supports(family.family_id(), family.revision()))
        else {
            return work_generic_lease(lease, topology, params, sink);
        };
        let config = self
            .config
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        for index in 0..lease.salt_count() {
            if sink.is_stopped() {
                break;
            }
            let nonce = lease.nonce(index);
            let instance = match family.generate(nonce) {
                Ok(instance) => instance,
                Err(SaltFault::Skip(_)) => {
                    if sink.skip(index).is_err() {
                        break;
                    }
                    continue;
                }
                Err(SaltFault::Defect(error)) => {
                    return Err(SampleError::DeviceFault(format!(
                        "canonical family generation failed: {error:?}"
                    )));
                }
            };
            let mut seed = [0; 8];
            seed.copy_from_slice(&nonce[..8]);
            let params = SampleParams {
                num_reads: config.final_reads.unwrap_or(params.num_reads),
                num_sweeps: config.final_sweeps.unwrap_or(params.num_sweeps),
                seed: u64::from_le_bytes(seed) ^ 0x5354_4147_455f_3031,
                ..params.clone()
            };
            let Some(mut reads) =
                solve_family(&instance, family.params(), &params, &config, || {
                    sink.is_stopped()
                })?
            else {
                break;
            };
            for read in &mut reads {
                read.energy_milli = instance_energy(&instance, &read.spins);
            }
            if sink.push(index, reads).is_err() {
                break;
            }
        }
        Ok(())
    }

    fn sample_stream_warm(
        &self,
        jobs: tokio::sync::mpsc::Receiver<WarmStreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        let config = Arc::clone(&self.config);
        run_job_pump(
            || self.stream_width(),
            move |job: &WarmStreamJob, token: &CancelToken| {
                let (graph, params) = (&job.job.graph, &job.job.params);
                let watermark = job.job.watermark;
                if let Some(family) = job
                    .family
                    .as_ref()
                    .filter(|family| supports(family.family_id(), family.revision()))
                {
                    let settings = config
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .clone();
                    return solve_family(
                        family.instance(),
                        family.params(),
                        params,
                        &settings,
                        || token.is_cancelled(watermark),
                    )
                    .map_err(StreamKernelError::Reject)?
                    .ok_or(StreamKernelError::Cancelled);
                }
                check_sample_size(graph.h.len(), graph.edges.len())
                    .map_err(StreamKernelError::Reject)?;
                sample_generic_sa(graph, params, Some((token, watermark)))
                    .map_err(|_| StreamKernelError::Cancelled)
            },
            jobs,
            out,
            cancel,
        );
    }
}
