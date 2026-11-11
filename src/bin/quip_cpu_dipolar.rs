//! CPU D-static string and loop search miner (`quip-cpu-dipolar`),
//! experimental.
//!
//! Build with `cargo build --release --features experimental`. The binary
//! runs the `dipolar-strings/1` search on `quip/defects-static` revision 1
//! jobs, from supplied initial states when the job has them, and the greedy
//! descent of `quip-cpu-greedy` on every other job.

use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cpu::{DipolarSampler, CPU_DIPOLAR_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_DIPOLAR_IDENTITY, &cli.common, || {
        Ok(DipolarSampler::new())
    })
}
