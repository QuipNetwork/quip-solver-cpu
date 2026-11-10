//! `quip-cpu-routing`: the routing attack behind the solver session.
//!
//! The binary accepts `quip/two-commodity-routing` revision 1 and generates no
//! family work. It has no fallback kernel, so a job outside the family is
//! declined, and the plain-job session conformance does not apply.
#![cfg(feature = "experimental")]

use quip_proto::v1::{Algorithm, Backend};
use quip_solver_conformance::driver::drive_miner;
use quip_solver_conformance::families::drive_miner_family;
use quip_solver_conformance::solve::CaseOutcome;
use std::process::Command;
use std::sync::LazyLock;
use tokio::sync::Mutex;

/// Serializes the session tests, as in `tests/family_conformance.rs`.
static GATE: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

const BIN: &str = env!("CARGO_BIN_EXE_quip-cpu-routing");
const SOLVER_ID: &str = "quip-cpu-routing/1";
const FAMILY: &str = "quip/two-commodity-routing";

fn socket(name: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "unix:///tmp/routing-{name}-{}-{nanos}.sock",
        std::process::id()
    )
}

#[test]
fn the_command_line_declares_routing_revision_one_as_accepted() {
    let out = Command::new(BIN).arg("--capabilities").output().unwrap();
    assert!(out.status.success(), "--capabilities failed");
    let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(caps["backend"], "cpu", "{caps}");
    assert_eq!(caps["algorithm"], "routing", "{caps}");
    assert_eq!(caps["solverId"], SOLVER_ID, "{caps}");
    assert!(caps.get("generatedFamilies").is_none(), "{caps}");
    let accepted = caps["acceptedFamilies"].as_array().expect("a family list");
    assert_eq!(accepted.len(), 1, "{caps}");
    assert_eq!(accepted[0]["familyId"], FAMILY, "{caps}");
    assert_eq!(accepted[0]["revisions"], serde_json::json!([1]), "{caps}");
    assert!(accepted[0].get("paramsDigests").is_none(), "{caps}");
}

#[tokio::test]
async fn the_session_advertises_what_the_command_line_prints() {
    let _guard = GATE.lock().await;
    let report = drive_miner(BIN, &socket("caps")).await;
    assert!(report.handshake_ok, "handshake failed");
    let session = report
        .capabilities_received
        .as_ref()
        .expect("a GetCapabilities answer");
    assert_eq!(session.backend(), Backend::Cpu);
    assert_eq!(session.algorithm(), Algorithm::Routing);
    assert_eq!(session.solver_id, SOLVER_ID);
    assert!(session.generated_families.is_empty());
    assert_eq!(session.accepted_families.len(), 1);
    assert_eq!(session.accepted_families[0].family_id, FAMILY);
    assert_eq!(session.accepted_families[0].revisions, vec![1]);
}

/// The family gate sends explicit routing jobs: the binary must solve each one
/// and keep the credit ledger balanced.
#[tokio::test]
async fn the_family_gate_grades_the_binary() {
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
    assert!(names.contains(&"explicit-supported"), "{names:?}");
}
