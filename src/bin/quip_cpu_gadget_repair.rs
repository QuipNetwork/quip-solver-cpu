//! CPU routing gadget-repair postprocess miner (`quip-cpu-gadget-repair`),
//! experimental.
//!
//! Build with `cargo build --release --features experimental`. The binary
//! runs the `gadget-repair/1` schedule on the supplied initial states of
//! two-commodity routing jobs, and the greedy descent of `quip-cpu-greedy` on
//! every other job. A job without initial states starts from random states
//! and logs a warning.

use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cpu::{GadgetRepairSampler, CPU_GADGET_REPAIR_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_GADGET_REPAIR_IDENTITY, &cli.common, || {
        Ok(GadgetRepairSampler::new())
    })
}
