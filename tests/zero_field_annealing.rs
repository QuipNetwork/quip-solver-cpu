//! Zero-field conformance: the annealing binaries solve a model with no linear
//! field through `--solve`.
//!
//! The model is a 16-spin antiferromagnetic ring. Its two ground states have
//! every bond satisfied, at energy `-16 * |J|`. A kernel that visits nodes in
//! one fixed order and accepts every flat move only shifts the ring's domain
//! walls in step, so its reads keep the energy of their random start.

use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Stdio};

const NODES: usize = 16;
const READS: usize = 64;

/// Reads that must reach the ground state. Below `READS` so that a rare read
/// frozen with one wall pair does not fail the case; a stalled kernel reaches
/// the ground state on none.
const MIN_GROUND_READS: usize = 60;

/// The ring as a `--solve` request, with coupling `j` on every bond and field
/// `h0` on node 0.
fn ring(j: f64, h0: f64, beta_range: Option<[f64; 2]>) -> Value {
    let mut h = vec![0.0; NODES];
    h[0] = h0;
    let edges: Vec<[usize; 2]> = (0..NODES)
        .map(|i| {
            let k = (i + 1) % NODES;
            [i.min(k), i.max(k)]
        })
        .collect();
    json!({
        "h": h,
        "j": vec![j; NODES],
        "edges": edges,
        "num_reads": READS,
        "num_sweeps": 1000,
        "sweeps_per_beta": 1,
        "beta_range": beta_range,
        "seed": 7,
    })
}

/// Run `bin --solve` on `problem` and return its reads.
fn solve(bin: &str, problem: &Value) -> Vec<Value> {
    let mut child = Command::new(bin)
        .arg("--solve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn solver");
    child
        .stdin
        .take()
        .expect("solver stdin")
        .write_all(problem.to_string().as_bytes())
        .expect("write problem");
    let out = child.wait_with_output().expect("solver exit");
    assert!(
        out.status.success(),
        "{bin} --solve exited {:?}: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice::<Vec<Value>>(&out.stdout).expect("solutions are a JSON array")
}

fn energies(reads: &[Value]) -> Vec<i64> {
    reads
        .iter()
        .map(|r| r["energy_milli"].as_i64().expect("energy_milli"))
        .collect()
}

/// Assert `bin` reaches the ground state of the zero-field ring with coupling
/// `j`, under the default ladder and under an explicit one.
fn assert_anneals_zero_field_ring(bin: &str, j: f64) {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "j is 1 or 1000 and the ring has 16 bonds; the product is far inside i64"
    )]
    let ground = -(NODES as i64) * (j * 1000.0) as i64;
    for beta_range in [None, Some([0.1, 10.0])] {
        let got = energies(&solve(bin, &ring(j, 0.0, beta_range)));
        assert_eq!(got.len(), READS, "{bin} j {j} beta {beta_range:?}");
        let at_ground = got.iter().filter(|&&e| e == ground).count();
        assert!(
            at_ground >= MIN_GROUND_READS,
            "{bin} j {j} beta {beta_range:?}: {at_ground}/{READS} reads at the ground energy \
             {ground}; energies {got:?}"
        );
    }
}

/// The ticket's ring: couplings of 1000, which sa, fsa and msa all run on the
/// scalar kernel.
#[test]
fn quip_cpu_sa_anneals_the_zero_field_ring() {
    assert_anneals_zero_field_ring(env!("CARGO_BIN_EXE_quip-cpu-sa"), 1000.0);
}

/// Unit couplings select the integer kernel.
#[test]
fn quip_cpu_sa_anneals_the_zero_field_unit_ring() {
    assert_anneals_zero_field_ring(env!("CARGO_BIN_EXE_quip-cpu-sa"), 1.0);
}

#[cfg(feature = "experimental")]
#[test]
fn quip_cpu_fsa_anneals_the_zero_field_ring() {
    assert_anneals_zero_field_ring(env!("CARGO_BIN_EXE_quip-cpu-fsa"), 1000.0);
}

/// Unit couplings select the tabulated-threshold kernel.
#[cfg(feature = "experimental")]
#[test]
fn quip_cpu_fsa_anneals_the_zero_field_unit_ring() {
    assert_anneals_zero_field_ring(env!("CARGO_BIN_EXE_quip-cpu-fsa"), 1.0);
}

#[cfg(feature = "experimental")]
#[test]
fn quip_cpu_msa_anneals_the_zero_field_ring() {
    assert_anneals_zero_field_ring(env!("CARGO_BIN_EXE_quip-cpu-msa"), 1000.0);
}

/// Unit couplings select the multi-spin coded kernel.
#[cfg(feature = "experimental")]
#[test]
fn quip_cpu_msa_anneals_the_zero_field_unit_ring() {
    assert_anneals_zero_field_ring(env!("CARGO_BIN_EXE_quip-cpu-msa"), 1.0);
}

/// One field on the ring picks one of the two orientations: the best read
/// gains the field's energy on top of the sixteen bonds.
#[test]
fn quip_cpu_sa_still_solves_the_ring_with_one_field() {
    let got = energies(&solve(
        env!("CARGO_BIN_EXE_quip-cpu-sa"),
        &ring(1000.0, 1000.0, None),
    ));
    assert_eq!(got.iter().min(), Some(&-17_000_000), "energies {got:?}");
}

/// The same seed gives the same samples on the zero-field ring.
#[test]
fn quip_cpu_sa_repeats_its_samples_on_the_zero_field_ring() {
    let problem = ring(1000.0, 0.0, None);
    let spins =
        |reads: &[Value]| -> Vec<Value> { reads.iter().map(|r| r["spins"].clone()).collect() };
    let first = solve(env!("CARGO_BIN_EXE_quip-cpu-sa"), &problem);
    let second = solve(env!("CARGO_BIN_EXE_quip-cpu-sa"), &problem);
    assert_eq!(spins(&first), spins(&second));
}

/// A one-node model has nothing to anneal around: every read takes the spin
/// its field prefers.
#[cfg(feature = "experimental")]
#[test]
fn quip_cpu_msa_solves_a_single_node_with_a_field() {
    let problem = json!({
        "h": [1.0],
        "j": [],
        "edges": [],
        "num_reads": READS,
        "num_sweeps": 100,
        "sweeps_per_beta": 1,
        "beta_range": null,
        "seed": 7,
    });
    let got = energies(&solve(env!("CARGO_BIN_EXE_quip-cpu-msa"), &problem));
    assert_eq!(got, vec![-1000; READS]);
}
