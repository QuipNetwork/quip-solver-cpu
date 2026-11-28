//! End-to-end lease accounting, cancellation, and exact family witnesses.
#![cfg(feature = "experimental")]

use quip_solver_conformance::driver::drive_miner;
use quip_solver_conformance::families::drive_miner_family;
use quip_solver_conformance::solve::CaseOutcome;

const BIN: &str = env!("CARGO_BIN_EXE_quip-cpu-native");

#[tokio::test]
async fn cpu_native_obeys_the_family_and_plain_session_contracts() {
    let socket = format!("unix:///tmp/cpu-native-family-{}.sock", std::process::id());
    let report = drive_miner_family(BIN, &[], &socket).await;
    let failed: Vec<_> = report
        .cases
        .iter()
        .filter(|c| matches!(c.outcome, CaseOutcome::Failed(_)))
        .collect();
    assert!(report.is_conformant(), "{failed:#?}\n{}", report.stderr);
    for name in [
        "supported-revision",
        "deadline-expired",
        "stale-lease",
        "credit-ledger-balanced",
        "explicit-supported",
        "explicit-digest-mismatch",
        "explicit-metadata-invalid",
    ] {
        assert!(
            report
                .cases
                .iter()
                .any(|c| c.name == name && matches!(c.outcome, CaseOutcome::Passed)),
            "no {name} test passed"
        );
    }
    let socket = format!("unix:///tmp/cpu-native-plain-{}.sock", std::process::id());
    let report = drive_miner(BIN, &socket).await;
    assert!(report.handshake_ok, "plain handshake failed");
    for id in [
        &b"job-1"[..],
        b"job-2",
        b"job-hash",
        b"job-sparse",
        b"job-seeded",
    ] {
        assert!(
            report.result_job_ids().iter().any(|got| got == id),
            "missing {id:?}"
        );
    }
    assert!(
        report.credit_ledger_balanced(),
        "credit ledger did not balance"
    );
}
