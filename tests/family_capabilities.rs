//! Family capabilities of the existing CPU binaries.
//!
//! None of these binaries links a canonical family generator, so each one
//! declares the generic tier: no `solver_id` and no generated or accepted
//! family. The command line, the session `Hello`, and `GetCapabilities` must
//! say the same thing, and the family gate must grade the binary conformant.
#![cfg(feature = "experimental")]

use quip_proto::v1::{
    Algorithm, Backend, Capabilities, CoefficientEncoding, GeneratorAlgorithm, JobKind,
};
use quip_solver_conformance::driver::drive_miner;
use quip_solver_conformance::families::drive_miner_family;
use serde_json::{json, Map, Value};
use std::process::Command;
use std::sync::LazyLock;
use tokio::sync::Mutex;

/// Serializes the session tests. Each spawns a miner with its own worker
/// threads, and unbounded parallelism can starve one past a driver deadline.
static GATE: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// One binary CPU-1 covers and the identity it must declare.
struct Bin {
    name: &'static str,
    path: &'static str,
    algorithm: Algorithm,
    max_nodes: u64,
    max_edges: u64,
}

fn bins() -> [Bin; 4] {
    [
        Bin {
            name: "quip-cpu-sa",
            path: env!("CARGO_BIN_EXE_quip-cpu-sa"),
            algorithm: Algorithm::Sa,
            max_nodes: 4_194_304,
            max_edges: 67_108_864,
        },
        Bin {
            name: "quip-cpu-gibbs",
            path: env!("CARGO_BIN_EXE_quip-cpu-gibbs"),
            algorithm: Algorithm::Gibbs,
            max_nodes: 4_194_304,
            max_edges: 67_108_864,
        },
        Bin {
            name: "quip-cpu-msa",
            path: env!("CARGO_BIN_EXE_quip-cpu-msa"),
            algorithm: Algorithm::Msa,
            max_nodes: 4_194_304,
            max_edges: 67_108_864,
        },
        Bin {
            name: "quip-cpu-flatiron",
            path: env!("CARGO_BIN_EXE_quip-cpu-flatiron"),
            algorithm: Algorithm::Flatiron,
            max_nodes: 65_536,
            max_edges: 524_288,
        },
    ]
}

fn command_line_capabilities(path: &str) -> Value {
    let out = Command::new(path).arg("--capabilities").output().unwrap();
    assert!(out.status.success(), "{path} --capabilities failed");
    serde_json::from_slice(&out.stdout).unwrap()
}

/// The protobuf JSON name of an enum value with its type prefix removed, as
/// `--capabilities` prints `backend` and `algorithm`.
fn short_name(name: &str, prefix: &str) -> String {
    name.strip_prefix(prefix).unwrap().to_lowercase()
}

/// A session `Capabilities` in the shape `--capabilities` prints: the
/// protobuf JSON mapping, with empty strings, empty lists, and absent
/// optionals left out.
fn session_json(c: &Capabilities) -> Value {
    let names = |values: &[i32], name: fn(i32) -> Option<&'static str>| -> Value {
        values.iter().map(|&v| json!(name(v).unwrap())).collect()
    };
    let mut view = Map::new();
    view.insert(
        "supportedKinds".into(),
        names(&c.supported_kinds, |k| {
            JobKind::try_from(k).ok().map(|k| k.as_str_name())
        }),
    );
    view.insert("maxNodes".into(), json!(c.max_nodes));
    view.insert("maxEdges".into(), json!(c.max_edges));
    if !c.features.is_empty() {
        view.insert("features".into(), json!(c.features));
    }
    view.insert("protocolVersion".into(), json!(c.protocol_version));
    view.insert("streamWidth".into(), json!(c.stream_width));
    view.insert(
        "encodings".into(),
        names(&c.encodings, |e| {
            CoefficientEncoding::try_from(e)
                .ok()
                .map(|e| e.as_str_name())
        }),
    );
    view.insert(
        "generators".into(),
        names(&c.generators, |g| {
            GeneratorAlgorithm::try_from(g)
                .ok()
                .map(|g| g.as_str_name())
        }),
    );
    view.insert(
        "backend".into(),
        json!(short_name(c.backend().as_str_name(), "BACKEND_")),
    );
    view.insert(
        "algorithm".into(),
        json!(short_name(c.algorithm().as_str_name(), "ALGORITHM_")),
    );
    for (key, value) in [
        ("maxDegree", c.max_degree),
        ("maxLocalFieldMilli", c.max_local_field_milli),
        ("exactResolutionMilli", c.exact_resolution_milli),
    ] {
        if let Some(value) = value {
            view.insert(key.into(), json!(value));
        }
    }
    assert!(c.native_topology_hash.is_none(), "{c:?}");
    assert!(c.solver_id.is_empty(), "{c:?}");
    assert!(c.generated_families.is_empty(), "{c:?}");
    assert!(c.accepted_families.is_empty(), "{c:?}");
    Value::Object(view)
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
    for bin in bins() {
        let name = bin.name;
        let caps = command_line_capabilities(bin.path);
        for key in ["solverId", "generatedFamilies", "acceptedFamilies"] {
            assert!(caps.get(key).is_none(), "{name} lists {key}: {caps}");
        }
        assert_eq!(caps["backend"], "cpu", "{name}: {caps}");
        assert_eq!(
            caps["algorithm"],
            short_name(bin.algorithm.as_str_name(), "ALGORITHM_"),
            "{name}: {caps}"
        );
        assert_eq!(caps["maxNodes"], bin.max_nodes, "{name}: {caps}");
        assert_eq!(caps["maxEdges"], bin.max_edges, "{name}: {caps}");
        assert_eq!(caps["maxDegree"], 0, "{name}: {caps}");
        assert_eq!(caps["maxLocalFieldMilli"], 0, "{name}: {caps}");
        assert_eq!(caps["exactResolutionMilli"], 1, "{name}: {caps}");
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
    for bin in bins() {
        let name = bin.name;
        let caps = command_line_capabilities(bin.path);
        let report = drive_miner(bin.path, &socket(name)).await;
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
            assert_eq!(session.backend(), Backend::Cpu, "{name} {source}");
            assert_eq!(session.algorithm(), bin.algorithm, "{name} {source}");
            assert_eq!(session_json(session), caps, "{name} {source}");
        }
    }
}

#[tokio::test]
async fn the_family_gate_grades_each_binary_conformant() {
    let _guard = GATE.lock().await;
    for bin in bins() {
        let report = drive_miner_family(bin.path, &[], &socket(bin.name)).await;
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
            "{} failed the family gate (exit {}): {failed:#?}\n{}",
            bin.name,
            report.exit_code,
            report.stderr
        );
    }
}
