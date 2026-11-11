//! CPU tile worm postprocess miner (`quip-cpu-tileworm`), experimental.
//!
//! Build with `cargo build --release --features experimental`. The binary
//! runs the selected tile worm policies on the supplied initial states of
//! charged-ice and field-ice jobs of the three selected recipes, and the
//! greedy descent of `quip-cpu-greedy` on every other job. A job without
//! initial states starts from random states and logs a warning.

use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cpu::{TileWormSampler, CPU_TILE_WORM_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_TILE_WORM_IDENTITY, &cli.common, || {
        Ok(TileWormSampler::new())
    })
}
