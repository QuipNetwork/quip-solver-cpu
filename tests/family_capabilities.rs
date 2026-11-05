//! Family capabilities of the existing CPU binaries.
//!
//! None of these binaries links a canonical family generator, so each one
//! declares the generic tier: no `solver_id` and no generated or accepted
//! family. The command line, the session `Hello`, and `GetCapabilities` must
//! say the same thing, and the family gate must grade the binary conformant.
#![cfg(feature = "experimental")]

use quip_solver_conformance::driver::drive_miner;
use quip_solver_conformance::families::drive_miner_family;
use std::process::Command;
use std::sync::LazyLock;
use tokio::sync::Mutex;

/// Serializes the session tests. Each spawns a miner with its own worker
/// threads, and unbounded parallelism can starve one past a driver deadline.
static GATE: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// The binaries CPU-1 covers, with the algorithm name each one reports.
fn bins() -> [(&'static str, &'static str, &'static str); 4] {
    [
        ("quip-cpu-sa", env!("CARGO_BIN_EXE_quip-cpu-sa"), "sa"),
        (
            "quip-cpu-gibbs",
            env!("CARGO_BIN_EXE_quip-cpu-gibbs"),
            "gibbs",
        ),
        ("quip-cpu-msa", env!("CARGO_BIN_EXE_quip-cpu-msa"), "msa"),
        (
            "quip-cpu-flatiron",
            env!("CARGO_BIN_EXE_quip-cpu-flatiron"),
            "flatiron",
        ),
    ]
}

fn command_line_capabilities(path: &str) -> serde_json::Value {
    let out = Command::new(path).arg("--capabilities").output().unwrap();
    assert!(out.status.success(), "{path} --capabilities failed");
    serde_json::from_slice(&out.stdout).unwrap()
}

fn socket(name: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "unix:///tmp/{name}-family-{}-{nanos}.sock",
        std::process::id()
    )
}

#[test]
fn the_command_line_declares_the_generic_tier_with_limits() {
    for (name, path, algorithm) in bins() {
        let caps = command_line_capabilities(path);
        for key in ["solverId", "generatedFamilies", "acceptedFamilies"] {
            assert!(caps.get(key).is_none(), "{name} lists {key}: {caps}");
        }
        assert_eq!(caps["backend"], "cpu", "{name}: {caps}");
        assert_eq!(caps["algorithm"], algorithm, "{name}: {caps}");
        assert!(
            caps["maxNodes"].as_u64().is_some_and(|n| n > 0),
            "{name}: {caps}"
        );
        assert!(
            caps["maxEdges"].as_u64().is_some_and(|n| n > 0),
            "{name}: {caps}"
        );
        let kinds = caps["supportedKinds"].as_array().unwrap();
        assert!(
            kinds.iter().any(|kind| kind == "ISING_SAMPLE"),
            "{name}: {caps}"
        );
    }
}

#[tokio::test]
async fn the_session_advertises_what_the_command_line_prints() {
    let _guard = GATE.lock().await;
    for (name, path, _) in bins() {
        let caps = command_line_capabilities(path);
        let report = drive_miner(path, &socket(name)).await;
        assert!(report.handshake_ok, "{name}: handshake failed");
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
            assert_eq!(session.solver_id, "", "{name} {source}");
            assert!(session.generated_families.is_empty(), "{name} {source}");
            assert!(session.accepted_families.is_empty(), "{name} {source}");
            assert_eq!(
                u64::from(session.max_nodes),
                caps["maxNodes"].as_u64().unwrap(),
                "{name} {source}"
            );
            assert_eq!(
                u64::from(session.max_edges),
                caps["maxEdges"].as_u64().unwrap(),
                "{name} {source}"
            );
        }
    }
}

#[tokio::test]
async fn the_family_gate_grades_each_binary_conformant() {
    let _guard = GATE.lock().await;
    for (name, path, _) in bins() {
        let report = drive_miner_family(path, &[], &socket(name)).await;
        let failed: Vec<String> = report
            .cases
            .iter()
            .filter(|case| {
                matches!(
                    case.outcome,
                    quip_solver_conformance::solve::CaseOutcome::Failed(_)
                )
            })
            .map(|case| format!("{case:?}"))
            .collect();
        assert!(
            report.is_conformant(),
            "{name} failed the family gate (exit {}): {failed:#?}\n{}",
            report.exit_code,
            report.stderr
        );
    }
}
