//! CPU constrained loop search (`quip-cpu-flow-loops`), experimental.
//!
//! Build with `cargo build --release --features experimental`. The binary
//! generates and accepts `quip/flow-static` revision 1 and
//! `quip/polariton-static` revision 1. It runs a charge-preserving loop
//! search from the supplied initial states, or from the exact baseline
//! orientation when a job has none, and returns one state per start in start
//! order. Every other job is annealed by the generic sampler, because a plain
//! model has no link structure.

use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cpu::{FlowLoopsSampler, CPU_FLOW_LOOPS_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_FLOW_LOOPS_IDENTITY, &cli.common, || {
        Ok(FlowLoopsSampler::new())
    })
}
