//! `quip-cpu-flow-loops`: the constrained loop search behind the solver session.
//!
//! The binary generates and accepts `quip/flow-static` revision 1 and
//! `quip/polariton-static` revision 1, and it accepts initial states. The
//! family gate grades leases and explicit jobs of both on the reference
//! topology. Every plain job, seeded or not, which is how the session serves
//! a family the binary does not list, is declined with `NO_SOLUTION`.
#![cfg(feature = "experimental")]

use quip_proto::v1::{Algorithm, Backend, RejectReason};
use quip_solver_conformance::driver::{drive_miner, INITIAL_SPINS_FEATURE};
use quip_solver_conformance::families::drive_miner_family;
use quip_solver_conformance::solve::CaseOutcome;
use std::process::Command;
use std::sync::LazyLock;
use tokio::sync::Mutex;

/// Serializes the session tests, as in `tests/family_conformance.rs`.
static GATE: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

const BIN: &str = env!("CARGO_BIN_EXE_quip-cpu-flow-loops");
const SOLVER_ID: &str = "quip-cpu-flow-loops/1";
const FAMILIES: [&str; 2] = ["quip/flow-static", "quip/polariton-static"];

fn socket(name: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "unix:///tmp/flow-loops-{name}-{}-{nanos}.sock",
        std::process::id()
    )
}

#[test]
fn the_command_line_declares_the_p_and_f_families() {
    let out = Command::new(BIN).arg("--capabilities").output().unwrap();
    assert!(out.status.success(), "--capabilities failed");
    let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(caps["backend"], "cpu", "{caps}");
    assert_eq!(caps["algorithm"], "flow-loops", "{caps}");
    assert_eq!(caps["solverId"], SOLVER_ID, "{caps}");
    for list in ["generatedFamilies", "acceptedFamilies"] {
        let entries = caps[list].as_array().expect("a family list");
        let ids: Vec<&str> = entries
            .iter()
            .map(|e| e["familyId"].as_str().unwrap())
            .collect();
        assert_eq!(ids, FAMILIES, "{list}: {caps}");
        for entry in entries {
            assert_eq!(entry["revisions"], serde_json::json!([1]), "{list}: {caps}");
            assert!(entry.get("paramsDigests").is_none(), "{list}: {caps}");
        }
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
    for family in FAMILIES {
        for name in ["supported-revision", "explicit-supported"] {
            assert!(
                report.cases.iter().any(|case| case.name == name
                    && case.subject.as_deref().is_some_and(|s| s.contains(family))),
                "the gate ran no {name} case for {family}: {:#?}",
                report.cases
            );
        }
    }
}

#[tokio::test]
async fn plain_jobs_are_declined_and_the_session_survives() {
    let _guard = GATE.lock().await;
    let report = drive_miner(BIN, &socket("plain")).await;
    assert!(report.handshake_ok, "handshake failed");
    let session = report
        .capabilities_received
        .as_ref()
        .expect("a GetCapabilities answer");
    assert_eq!(session.backend(), Backend::Cpu);
    assert_eq!(session.algorithm(), Algorithm::FlowLoops);
    assert_eq!(session.solver_id, SOLVER_ID);
    for id in [&b"job-1"[..], b"job-2", b"job-sparse", b"job-seeded"] {
        assert!(
            report.has_reject(id, RejectReason::NoSolution),
            "{} is not declined",
            String::from_utf8_lossy(id)
        );
    }
    assert!(report.has_reject(b"job-bad-h", RejectReason::Malformed));
    assert!(report.has_reject(b"job-old", RejectReason::Expired));
    assert!(report.ping_acked);
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}
