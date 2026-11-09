//! CPU Otsuka cluster miner (`quip-cpu-otsuka`), experimental.
//!
//! Build with `cargo build --release --features experimental`. The binary
//! generates and accepts `quip/twisted-ice-dimer` revision 1 and runs the
//! Otsuka cluster kernel on those jobs. Every other job runs on the SA kernel
//! of `quip-cpu-sa`.

use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cpu::{OtsukaSampler, CPU_OTSUKA_IDENTITY};
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
        CPU_OTSUKA_IDENTITY,
        &cli.common,
        || Ok(OtsukaSampler::new()),
    )
}
