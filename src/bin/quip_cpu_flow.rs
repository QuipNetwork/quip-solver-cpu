//! CPU exact flow baselines (`quip-cpu-flow`), experimental.
//!
//! Build with `cargo build --release --features experimental`. The binary
//! generates and accepts `quip/flow-static` revision 1 and
//! `quip/polariton-static` revision 1. It solves a F-static job with an exact
//! min-cost circulation and an adapted-P job with a compatible-charge
//! construction. It declines every other job, because a plain model has no
//! flow structure.

use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cpu::{FlowSampler, CPU_FLOW_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_FLOW_IDENTITY, &cli.common, || Ok(FlowSampler::new()))
}
