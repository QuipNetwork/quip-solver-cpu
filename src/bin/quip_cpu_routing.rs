//! CPU two-commodity routing miner (`quip-cpu-routing`), experimental.
//!
//! Build with `cargo build --release --features experimental`. The binary runs
//! the original alternating routing attack on explicit
//! `quip/two-commodity-routing` revision 1 jobs. Every other job is annealed by the
//! generic sampler.

use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cpu::{RoutingSampler, CPU_ROUTING_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_ROUTING_IDENTITY, &cli.common, || {
        Ok(RoutingSampler::new())
    })
}
