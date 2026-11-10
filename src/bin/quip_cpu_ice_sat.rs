//! CPU strong-layer SAT miner (`quip-cpu-ice-sat`), experimental.
//!
//! Build with `cargo build --release --features experimental`. The binary
//! generates and accepts `quip/twisted-ice-dimer` revision 1. On those jobs an
//! external Kissat executable, named by the backend_toml key `kissat`, solves
//! the strong layer, and a job with no checked witness is declined. The flag
//! `--kissat` sets the executable when backend_toml names none. Every other
//! job runs on the SA kernel of `quip-cpu-sa`.

use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cpu::{IceSatSampler, CPU_ICE_SAT_IDENTITY};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    /// Absolute path to the Kissat executable. The backend_toml key `kissat`
    /// overrides it.
    #[arg(long)]
    kissat: Option<PathBuf>,
    /// Wall-time limit of one Kissat run, in seconds (default 600). The
    /// backend_toml key `kissat_timeout_s` overrides it.
    #[arg(long)]
    kissat_timeout_s: Option<u64>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_ICE_SAT_IDENTITY, &cli.common, || {
        Ok(IceSatSampler::new(cli.kissat.clone(), cli.kissat_timeout_s))
    })
}
