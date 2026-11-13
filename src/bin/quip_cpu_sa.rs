//! CPU simulated-annealing miner (`quip-cpu-sa`).

use clap::Parser;
use quip_solver_core::{run, CommonArgs};
use quip_solver_cpu::{Algorithm, CpuSampler, SaScheduleConfig, CPU_SA_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
#[command(about = None, long_about = None)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    #[command(flatten)]
    schedule: SaScheduleConfig,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Err(error) = cli.schedule.validate() {
        #[expect(
            clippy::print_stderr,
            reason = "report invalid CLI settings before the session opens"
        )]
        {
            eprintln!("configuration error: {error}");
        }
        return ExitCode::from(64);
    }
    run(CPU_SA_IDENTITY, &cli.common, || {
        Ok(CpuSampler::new(Algorithm::Sa).with_sa_config(cli.schedule))
    })
}
