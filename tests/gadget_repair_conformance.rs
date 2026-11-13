//! `quip-cpu-gadget-repair`: the routing gadget-repair postprocess behind the
//! solver session.
//!
//! The binary accepts warm starts, so the session conformance grades the
//! warm-start proof: seeded with the planted ground state of a ring, the
//! binary must return that ground energy. The ring is not a family job, so
//! the greedy descent serves it.
#![cfg(feature = "experimental")]

use quip_proto::v1::{Algorithm, Backend, RejectReason};
use quip_solver_conformance::driver::{drive_miner, INITIAL_SPINS_FEATURE};
use quip_solver_conformance::families::drive_miner_family;
use quip_solver_conformance::solve::{drive_solve, CaseOutcome};
use quip_solver_core::ising_generators::registry::CONTRACTS;
use std::process::Command;
use std::sync::LazyLock;
use tokio::sync::Mutex;

/// Serializes the session tests, as in `tests/family_conformance.rs`.
static GATE: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

const BIN: &str = env!("CARGO_BIN_EXE_quip-cpu-gadget-repair");
const SOLVER_ID: &str = "quip-cpu-gadget-repair/1";

/// Every registered family id, in registry order. The binary declares no
/// generated family, so the solver-core default session path generates all of
/// them.
fn registry_ids() -> Vec<&'static str> {
    CONTRACTS.iter().map(|c| c.family_id).collect()
}

fn socket(name: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "unix:///tmp/gadget-repair-{name}-{}-{nanos}.sock",
        std::process::id()
    )
}

#[test]
fn the_command_line_declares_the_gadget_repair_identity() {
    let out = Command::new(BIN).arg("--capabilities").output().unwrap();
    assert!(out.status.success(), "--capabilities failed");
    let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(caps["backend"], "cpu", "{caps}");
    assert_eq!(caps["algorithm"], "gadget-repair", "{caps}");
    assert_eq!(caps["solverId"], SOLVER_ID, "{caps}");
    let generated = caps["generatedFamilies"]
        .as_array()
        .expect("a generated family list");
    let ids: Vec<&str> = generated
        .iter()
        .map(|e| e["familyId"].as_str().unwrap())
        .collect();
    assert_eq!(ids, registry_ids(), "{caps}");
    let accepted = caps["acceptedFamilies"].as_array().expect("a family list");
    assert_eq!(accepted.len(), 1, "{caps}");
    assert_eq!(
        accepted[0]["familyId"], "quip/two-commodity-routing",
        "{caps}"
    );
    assert_eq!(accepted[0]["revisions"], serde_json::json!([1]), "{caps}");
    let digests = accepted[0]["paramsDigests"]
        .as_array()
        .expect("the r32 recipe only");
    assert_eq!(digests.len(), 1, "{caps}");
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
    assert_eq!(session.algorithm(), Algorithm::GadgetRepair);
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

/// The family gate passes every session case. Its explicit cases build jobs
/// from the reference routing params, z(2,4) and z(3,4), and the claim names
/// only `r32`, on z(12,4). Admission gives both reference sets code 10
/// (`ParamsRestricted`), so `explicit-supported` reports that no instance is
/// accepted and the other explicit cases skip for the same reason.
#[tokio::test]
async fn the_family_gate_passes_the_session_and_meets_the_r32_claim() {
    let _guard = GATE.lock().await;
    let report = drive_miner_family(BIN, &[], &socket("family")).await;
    assert_eq!(report.exit_code, 0, "{}", report.stderr);
    const RESTRICTED: &str = "no reference params set gives an explicit instance under the \
        accepted claims (P_ROUTING_Z24_D4 gives code 10, P_ROUTING_Z34_D6 gives code 10)";
    let mut explicit = 0;
    for case in &report.cases {
        if case.name.starts_with("explicit-") {
            explicit += 1;
            let reason = match &case.outcome {
                CaseOutcome::Failed(reason) if case.name == "explicit-supported" => reason,
                CaseOutcome::Skipped(reason) if case.name != "explicit-supported" => reason,
                other => unreachable!("{}: {other:?}", case.name),
            };
            assert_eq!(reason, RESTRICTED, "{}", case.name);
        } else {
            assert!(
                !matches!(case.outcome, CaseOutcome::Failed(_)),
                "{case:?}\n{}",
                report.stderr
            );
        }
    }
    assert_eq!(
        explicit, 5,
        "the gate ran no explicit cases: {:?}",
        report.cases
    );
}
