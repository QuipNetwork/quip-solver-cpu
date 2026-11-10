//! `quip-cpu-dipolar`: the D-static string and loop search behind the solver
//! session.
//!
//! The binary generates and accepts `quip/defects-static` revision 1. The
//! family gate grades leases and explicit jobs on the revision's base
//! topology. The binary accepts warm starts, so the session conformance
//! grades the warm-start proof on a plain ring, which the greedy descent
//! serves.
#![cfg(feature = "experimental")]

use quip_proto::v1::{Algorithm, Backend, RejectReason};
use quip_solver_conformance::driver::{drive_miner, INITIAL_SPINS_FEATURE};
use quip_solver_conformance::families::drive_miner_family;
use quip_solver_conformance::solve::{drive_solve, CaseOutcome};
use std::process::Command;
use std::sync::LazyLock;
use tokio::sync::Mutex;

/// Serializes the session tests, as in `tests/family_conformance.rs`.
static GATE: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

const BIN: &str = env!("CARGO_BIN_EXE_quip-cpu-dipolar");
const SOLVER_ID: &str = "quip-cpu-dipolar/1";
const FAMILY: &str = "quip/defects-static";

fn socket(name: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "unix:///tmp/dipolar-{name}-{}-{nanos}.sock",
        std::process::id()
    )
}

#[test]
fn the_command_line_declares_the_dipolar_identity() {
    let out = Command::new(BIN).arg("--capabilities").output().unwrap();
    assert!(out.status.success(), "--capabilities failed");
    let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(caps["backend"], "cpu", "{caps}");
    assert_eq!(caps["algorithm"], "dipolar", "{caps}");
    assert_eq!(caps["solverId"], SOLVER_ID, "{caps}");
    for list in ["generatedFamilies", "acceptedFamilies"] {
        let entries = caps[list].as_array().expect("a family list");
        assert_eq!(entries.len(), 1, "{list}: {caps}");
        assert_eq!(entries[0]["familyId"], FAMILY, "{list}: {caps}");
        assert_eq!(
            entries[0]["revisions"],
            serde_json::json!([1]),
            "{list}: {caps}"
        );
        assert!(entries[0].get("paramsDigests").is_none(), "{list}: {caps}");
    }
    let features = caps["features"].as_array().expect("a feature list");
    assert!(
        features.iter().any(|f| f == INITIAL_SPINS_FEATURE),
        "{caps}"
    );
}

#[tokio::test]
async fn the_family_gate_grades_the_binary_conformant() {
    let _guard = GATE.lock().await;
    let report = drive_miner_family(BIN, &[], &socket("family")).await;
    let failed: Vec<String> = report
        .cases
        .iter()
        .filter(|case| matches!(case.outcome, CaseOutcome::Failed(_)))
        .map(|case| format!("{case:?}"))
        .collect();
    assert!(
        report.is_conformant(),
        "the family gate failed (exit {}): {failed:#?}\n{}",
        report.exit_code,
        report.stderr
    );
    let names: Vec<&str> = report.cases.iter().map(|case| case.name).collect();
    for name in [
        "supported-revision",
        "deadline-expired",
        "stale-lease",
        "credit-ledger-balanced",
        "explicit-supported",
        "explicit-digest-mismatch",
        "explicit-metadata-invalid",
        "explicit-params-out-of-bounds",
        "explicit-topology-hash-form",
    ] {
        assert!(
            names.contains(&name),
            "the gate ran no {name} case: {names:?}"
        );
    }
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
    assert_eq!(session.algorithm(), Algorithm::Dipolar);
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
