//! `quip-cpu-ice-sat`: the strong-layer SAT baseline behind the solver session.
//!
//! The binary generates and accepts `quip/twisted-ice-dimer` revision 1. A
//! Kissat executable solves those jobs, and the SA kernel serves every other
//! job. The family gate needs a real solver, so its test is ignored by
//! default: set `QUIP_KISSAT` to the executable and pass `--ignored`.
#![cfg(feature = "experimental")]

use quip_proto::v1::{Algorithm, Backend, RejectReason};
use quip_solver_conformance::driver::drive_miner;
use quip_solver_conformance::families::drive_miner_family;
use quip_solver_conformance::solve::{drive_solve, CaseOutcome};
use quip_solver_core::ising_generators::registry::CONTRACTS;
use std::process::Command;
use std::sync::LazyLock;
use tokio::sync::Mutex;

/// Serializes the session tests, as in `tests/family_conformance.rs`.
static GATE: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

const BIN: &str = env!("CARGO_BIN_EXE_quip-cpu-ice-sat");
const SOLVER_ID: &str = "quip-cpu-ice-sat/1";
const FAMILY: &str = "quip/twisted-ice-dimer";

/// The family ids a capability list must hold, in order. The binary accepts
/// only its declared family. It generates that family first, then every other
/// registered family through the solver-core default session path.
fn expected_family_ids(list: &str) -> Vec<&'static str> {
    let mut ids = vec![FAMILY];
    if list == "generatedFamilies" {
        ids.extend(
            CONTRACTS
                .iter()
                .map(|c| c.family_id)
                .filter(|id| *id != FAMILY),
        );
    }
    ids
}

fn command_line_capabilities() -> serde_json::Value {
    let out = Command::new(BIN).arg("--capabilities").output().unwrap();
    assert!(out.status.success(), "--capabilities failed");
    serde_json::from_slice(&out.stdout).unwrap()
}

fn socket(name: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "unix:///tmp/ice-sat-{name}-{}-{nanos}.sock",
        std::process::id()
    )
}

#[test]
fn the_command_line_declares_the_twisted_dimer_family() {
    let caps = command_line_capabilities();
    assert_eq!(caps["backend"], "cpu", "{caps}");
    assert_eq!(caps["algorithm"], "sat", "{caps}");
    assert_eq!(caps["solverId"], SOLVER_ID, "{caps}");
    for list in ["generatedFamilies", "acceptedFamilies"] {
        let entries = caps[list].as_array().expect("a family list");
        let ids: Vec<&str> = entries
            .iter()
            .map(|e| e["familyId"].as_str().unwrap())
            .collect();
        assert_eq!(ids, expected_family_ids(list), "{list}: {caps}");
        for entry in entries {
            assert_eq!(entry["revisions"], serde_json::json!([1]), "{list}: {caps}");
            assert!(entry.get("paramsDigests").is_none(), "{list}: {caps}");
        }
    }
}

#[tokio::test]
async fn the_session_advertises_what_the_command_line_prints() {
    let _guard = GATE.lock().await;
    let caps = command_line_capabilities();
    let report = drive_miner(BIN, &socket("caps")).await;
    assert!(report.handshake_ok, "handshake failed");
    let session = report
        .capabilities_received
        .as_ref()
        .expect("a GetCapabilities answer");
    assert_eq!(session.backend(), Backend::Cpu);
    assert_eq!(session.algorithm(), Algorithm::Sat);
    assert_eq!(session.solver_id, SOLVER_ID);
    for (list, families) in [
        ("generatedFamilies", &session.generated_families),
        ("acceptedFamilies", &session.accepted_families),
    ] {
        let ids: Vec<&str> = families.iter().map(|f| f.family_id.as_str()).collect();
        assert_eq!(ids, expected_family_ids(list), "{list}");
        assert!(families.iter().all(|f| f.revisions == vec![1]), "{list}");
    }
    assert_eq!(
        u64::from(session.max_nodes),
        caps["maxNodes"].as_u64().unwrap()
    );
}

/// The gate grades leases and explicit jobs against a real Kissat.
#[tokio::test]
#[ignore = "needs a real Kissat: set QUIP_KISSAT and run with --ignored"]
async fn the_family_gate_grades_the_binary_conformant_with_a_real_solver() {
    let kissat = std::env::var("QUIP_KISSAT").expect("QUIP_KISSAT names a Kissat executable");
    let _guard = GATE.lock().await;
    let report = drive_miner_family(BIN, &["--kissat", &kissat], &socket("family")).await;
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
        "credit-ledger-balanced",
        "explicit-supported",
    ] {
        assert!(
            names.contains(&name),
            "the gate ran no {name} case: {names:?}"
        );
    }
}

/// Plain jobs fall back to the SA kernel and keep the generic protocol.
#[tokio::test]
async fn plain_jobs_pass_the_session_conformance() {
    let _guard = GATE.lock().await;
    let report = drive_miner(BIN, &socket("plain")).await;
    assert!(report.handshake_ok, "handshake failed");
    for id in [
        &b"job-1"[..],
        b"job-2",
        b"job-hash",
        b"job-sparse",
        b"job-seeded",
    ] {
        assert!(
            report.result_job_ids().iter().any(|got| got == id),
            "no result for {}: {:?}",
            String::from_utf8_lossy(id),
            report.result_job_ids()
        );
    }
    assert!(report.has_reject(b"job-bad-h", RejectReason::Malformed));
    assert!(report.has_reject(b"job-gate", RejectReason::UnsupportedKind));
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
