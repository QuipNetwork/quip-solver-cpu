//! CPU greedy postprocess miner (`quip-cpu-greedy`), experimental.
//!
//! Build with `cargo build --release --features experimental`. The binary
//! descends each supplied initial state with exact first-improvement greedy
//! moves: single flips plus natural moves. A job without initial states
//! descends from random states and logs a warning.

use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cpu::{GreedySampler, CPU_GREEDY_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(
        CPU_GREEDY_IDENTITY,
        &cli.common,
        || Ok(GreedySampler::new()),
    )
}
