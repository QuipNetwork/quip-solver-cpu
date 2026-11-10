//! CPU weighted-dimer miner (`quip-cpu-dimer`), experimental.
//!
//! Build with `cargo build --release --features experimental`. The binary
//! runs the bounded weighted-dimer kernel (research `ice-loop64-weighted`) on
//! `quip/twisted-ice-dimer` revision 1 jobs and rejects family jobs that
//! carry fields or bad structure. Every job without family structure runs on
//! the SA kernel.

use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cpu::{DimerSampler, CPU_DIMER_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_DIMER_IDENTITY, &cli.common, || Ok(DimerSampler::new()))
}
