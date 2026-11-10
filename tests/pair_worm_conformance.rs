//! `quip-cpu-pairworm`: the pair-worm postprocess behind the solver session.
//!
//! The binary accepts warm starts, so the session conformance grades the
//! warm-start proof: seeded with the planted ground state of a ring, the
//! binary must return that ground energy. The ring is not a family job, so
//! the greedy descent serves it.
#![cfg(feature = "experimental")]

use quip_proto::v1::{Algorithm, Backend, RejectReason};
use quip_solver_conformance::driver::{drive_miner, INITIAL_SPINS_FEATURE};
use quip_solver_conformance::solve::drive_solve;
use std::process::Command;
use std::sync::LazyLock;
use tokio::sync::Mutex;

/// Serializes the session tests, as in `tests/family_conformance.rs`.
static GATE: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

const BIN: &str = env!("CARGO_BIN_EXE_quip-cpu-pairworm");
const SOLVER_ID: &str = "quip-cpu-pairworm/1";

fn socket(name: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "unix:///tmp/pairworm-{name}-{}-{nanos}.sock",
        std::process::id()
    )
}

#[test]
fn the_command_line_declares_the_pair_worm_identity() {
    let out = Command::new(BIN).arg("--capabilities").output().unwrap();
    assert!(out.status.success(), "--capabilities failed");
    let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(caps["backend"], "cpu", "{caps}");
    assert_eq!(caps["algorithm"], "pair-worm", "{caps}");
    assert_eq!(caps["solverId"], SOLVER_ID, "{caps}");
    assert!(caps.get("generatedFamilies").is_none(), "{caps}");
    let accepted = caps["acceptedFamilies"].as_array().expect("a family list");
    assert_eq!(accepted.len(), 1, "{caps}");
    assert_eq!(accepted[0]["familyId"], "quip/twisted-ice-dimer", "{caps}");
    assert_eq!(accepted[0]["revisions"], serde_json::json!([1]), "{caps}");
    let digests = accepted[0]["paramsDigests"]
        .as_array()
        .expect("the selected recipes");
    assert_eq!(digests.len(), 3, "one digest per selected recipe: {caps}");
    let features = caps["features"].as_array().expect("a feature list");
    assert!(
        features.iter().any(|f| f == INITIAL_SPINS_FEATURE),
        "{caps}"
    );
}

#[tokio::test]
async fn the_session_passes_the_conformance_and_uses_the_warm_start() {
    let _guard = GATE.lock().await;
    let report = drive_miner(BIN, &socket("session")).await;
    assert!(report.handshake_ok, "handshake failed");
    assert!(
        report.is_conformant(),
        "the session failed conformance: {:?}",
        report.result_job_ids()
    );
    let session = report
        .capabilities_received
        .as_ref()
        .expect("a GetCapabilities answer");
    assert_eq!(session.backend(), Backend::Cpu);
    assert_eq!(session.algorithm(), Algorithm::PairWorm);
    assert_eq!(session.solver_id, SOLVER_ID);
    assert!(
        report.advertises_initial_spins(),
        "no {INITIAL_SPINS_FEATURE}"
    );
    assert!(
        report.warm_start_conformant(),
        "the warm-start proof failed"
    );
    for id in [
        &b"job-1"[..],
        b"job-2",
        b"job-hash",
        b"job-sparse",
        b"job-seeded",
        b"job-warm-proof",
    ] {
        assert!(
            report.result_job_ids().iter().any(|got| got == id),
            "no result for {}: {:?}",
            String::from_utf8_lossy(id),
            report.result_job_ids()
        );
    }
    assert!(report.has_reject(b"job-bad-h", RejectReason::Malformed));
    assert!(report.has_reject(b"job-old", RejectReason::Expired));
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}

#[tokio::test]
async fn plain_jobs_pass_the_solve_conformance() {
    let _guard = GATE.lock().await;
    let report = drive_solve(BIN).await;
    assert!(
        report.is_conformant(),
        "--solve failed conformance:\n{}",
        report.summary()
    );
}
