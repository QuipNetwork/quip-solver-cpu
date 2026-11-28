//! CPU-native solver for Quip's family lease protocol.
use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cpu::families::native::NativeSampler;
use quip_solver_cpu::CPU_OTSUKA_IDENTITY;
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " cpu-native protocol 2"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(
        CPU_OTSUKA_IDENTITY,
        &cli.common,
        || Ok(NativeSampler::new()),
    )
}
