//! `quip-cpu-dimer`: the weighted-dimer kernel behind the solver session.
//!
//! The binary generates and accepts `quip/twisted-ice-dimer` revision 1. It
//! runs the weighted-dimer kernel on validated family jobs and the production
//! SA kernel on every other job. The family gate grades leases and explicit
//! jobs, deadlines, stale leases, and the credit ledger. Leases and explicit
//! jobs run on the revision's own base topology.
#![cfg(feature = "experimental")]

use quip_proto::v1::{Algorithm, Backend, RejectReason};
use quip_solver_conformance::driver::drive_miner;
use quip_solver_conformance::families::drive_miner_family;
use quip_solver_conformance::solve::{drive_solve, CaseOutcome};
use std::process::Command;
use std::sync::LazyLock;
use tokio::sync::Mutex;

/// Serializes the session tests. Each spawns a miner with its own worker
/// threads, and unbounded parallelism can starve one past a driver deadline.
static GATE: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

const BIN: &str = env!("CARGO_BIN_EXE_quip-cpu-dimer");
const SOLVER_ID: &str = "quip-cpu-dimer/1";
const FAMILY: &str = "quip/twisted-ice-dimer";

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
        "unix:///tmp/dimer-{name}-{}-{nanos}.sock",
        std::process::id()
    )
}

#[test]
fn the_command_line_declares_the_twisted_dimer_family() {
    let caps = command_line_capabilities();
    assert_eq!(caps["backend"], "cpu", "{caps}");
    assert_eq!(caps["algorithm"], "weighted-dimer", "{caps}");
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
    let generators = caps["generators"].as_array().unwrap();
    assert!(
        generators
            .iter()
            .any(|g| g == "GENERATOR_ALGORITHM_FAMILY_CHACHA8_V1"),
        "{caps}"
    );
    let kinds = caps["supportedKinds"].as_array().unwrap();
    assert!(kinds.iter().any(|k| k == "ISING_SAMPLE"), "{caps}");
    assert_eq!(caps["maxDegree"], 0, "{caps}");
    assert_eq!(caps["exactResolutionMilli"], 1, "{caps}");
}

#[tokio::test]
async fn the_session_advertises_what_the_command_line_prints() {
    let _guard = GATE.lock().await;
    let caps = command_line_capabilities();
    let report = drive_miner(BIN, &socket("caps")).await;
    assert!(report.handshake_ok, "handshake failed");
    let hello = report
        .hello
        .as_ref()
        .and_then(|hello| hello.capabilities.as_ref())
        .expect("Hello carries capabilities");
    let answered = report
        .capabilities_received
        .as_ref()
        .expect("a GetCapabilities answer");
    for (source, session) in [("Hello", hello), ("GetCapabilities", answered)] {
        assert_eq!(session.backend(), Backend::Cpu, "{source}");
        assert_eq!(session.algorithm(), Algorithm::WeightedDimer, "{source}");
        assert_eq!(session.solver_id, SOLVER_ID, "{source}");
        for families in [&session.generated_families, &session.accepted_families] {
            assert_eq!(families.len(), 1, "{source}");
            assert_eq!(families[0].family_id, FAMILY, "{source}");
            assert_eq!(families[0].revisions, vec![1], "{source}");
        }
        assert_eq!(
            u64::from(session.max_nodes),
            caps["maxNodes"].as_u64().unwrap(),
            "{source}"
        );
        assert_eq!(
            u64::from(session.max_edges),
            caps["maxEdges"].as_u64().unwrap(),
            "{source}"
        );
    }
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
