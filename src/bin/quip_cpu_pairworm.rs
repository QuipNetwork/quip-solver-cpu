//! CPU pair-worm postprocess miner (`quip-cpu-pairworm`), experimental.
//!
//! Build with `cargo build --release --features experimental`. The binary
//! runs the `ice-pairs-pw/1` schedule on the supplied initial states of
//! twisted-ice-dimer jobs of the three selected recipes, and the greedy
//! descent of `quip-cpu-greedy` on every other job. A job without initial
//! states starts from random states and logs a warning.

use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cpu::{PairWormSampler, CPU_PAIR_WORM_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_PAIR_WORM_IDENTITY, &cli.common, || {
        Ok(PairWormSampler::new())
    })
}
