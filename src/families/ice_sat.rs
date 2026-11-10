//! The strong-layer SAT encoding of synthetic ice, and the boundary to an
//! external Kissat executable.
//!
//! A gauged tile is satisfied when exactly two of its four gauged spins are
//! +1, and a pair when its spins agree (negative coupling) or differ (positive
//! coupling). Satisfying this layer says nothing about the weak layer, so a
//! witness is scored against the full instance separately.

use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use quip_solver_core::CancelToken;

/// The strong layer of one instance, in dense node positions.
#[derive(Clone, Copy, Debug)]
pub struct Strong<'a> {
    /// Four nodes per tile.
    pub tiles: &'a [[u32; 4]],
    /// The local gauge of each tile.
    pub gauges: &'a [[i8; 4]],
    /// Each pair as two nodes and the sign of its coupling.
    pub pairs: &'a [(u32, u32, i8)],
}

/// A CNF over the constrained nodes. Variable `k + 1` is the spin of
/// `nodes[k]`, true for +1. `nodes` is ascending.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoding {
    /// The constrained nodes, ascending.
    pub nodes: Vec<u32>,
    /// Clauses of nonzero DIMACS literals.
    pub clauses: Vec<Vec<i32>>,
}

/// What the solver answered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// A satisfying assignment, one value per variable.
    Sat(Vec<bool>),
    /// The solver reports no satisfying assignment. This is not a checked proof.
    Unsat,
    /// The solver stopped without an answer, for example at its own time limit.
    Unknown,
}

/// The CNF of `strong`.
#[must_use]
pub fn encode(strong: &Strong<'_>) -> Encoding {
    let mut nodes: Vec<u32> = strong
        .tiles
        .iter()
        .flatten()
        .copied()
        .chain(strong.pairs.iter().flat_map(|&(u, v, _)| [u, v]))
        .collect();
    nodes.sort_unstable();
    nodes.dedup();
    let var = |node: u32| -> i32 {
        let k = nodes.binary_search(&node).unwrap_or_default();
        i32::try_from(k + 1).unwrap_or(i32::MAX)
    };
    let mut clauses = Vec::new();
    for (tile, gauge) in strong.tiles.iter().zip(strong.gauges) {
        let lits: [i32; 4] = std::array::from_fn(|p| i32::from(gauge[p]) * var(tile[p]));
        // Forbid every assignment of the gauged literals whose true count is not 2.
        for bits in 0_u8..16 {
            if bits.count_ones() != 2 {
                clauses.push(
                    (0..4)
                        .map(|p| {
                            if bits >> p & 1 == 1 {
                                -lits[p]
                            } else {
                                lits[p]
                            }
                        })
                        .collect(),
                );
            }
        }
    }
    for &(u, v, sign) in strong.pairs {
        let (a, b) = (var(u), var(v));
        let b = if sign < 0 { b } else { -b };
        clauses.push(vec![a, -b]);
        clauses.push(vec![-a, b]);
    }
    Encoding { nodes, clauses }
}

/// Whether `values` (one per variable) satisfies every clause.
#[must_use]
pub fn satisfies(encoding: &Encoding, values: &[bool]) -> bool {
    encoding.clauses.iter().all(|clause| {
        clause.iter().any(|&lit| {
            let value = values[lit.unsigned_abs() as usize - 1];
            value == (lit > 0)
        })
    })
}

/// The spins of a witness: each constrained node from `values`, every other
/// node +1.
#[must_use]
pub fn decode(encoding: &Encoding, values: &[bool], n: usize) -> Vec<i8> {
    let mut spins = vec![1_i8; n];
    for (&node, &value) in encoding.nodes.iter().zip(values) {
        spins[node as usize] = if value { 1 } else { -1 };
    }
    spins
}

/// The direct check, independent of the CNF: every gauged tile balanced and
/// every pair aligned with its coupling.
#[must_use]
pub fn strong_layer_holds(strong: &Strong<'_>, spins: &[i8]) -> bool {
    let tiles = strong.tiles.iter().zip(strong.gauges).all(|(tile, gauge)| {
        (0..4)
            .map(|p| i32::from(gauge[p]) * i32::from(spins[tile[p] as usize]))
            .sum::<i32>()
            == 0
    });
    let pairs = strong
        .pairs
        .iter()
        .all(|&(u, v, sign)| (spins[u as usize] == spins[v as usize]) == (sign < 0));
    tiles && pairs
}

/// The DIMACS text of `encoding`.
#[must_use]
pub fn to_dimacs(encoding: &Encoding) -> String {
    let mut text = format!(
        "p cnf {} {}\n",
        encoding.nodes.len(),
        encoding.clauses.len()
    );
    for clause in &encoding.clauses {
        for lit in clause {
            text.push_str(&lit.to_string());
            text.push(' ');
        }
        text.push_str("0\n");
    }
    text
}

/// Parse SAT-competition output for `vars` variables: one `s` status line,
/// and for a satisfiable answer `v` lines that assign every variable exactly
/// once and end with `0`.
///
/// # Errors
///
/// A description of the first defect.
pub fn parse_output(stdout: &str, vars: usize) -> Result<Verdict, String> {
    let mut status = None;
    let mut values: Vec<Option<bool>> = vec![None; vars];
    let mut ended = false;
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("s ") {
            if status.replace(rest.trim().to_owned()).is_some() {
                return Err("two status lines".to_owned());
            }
        } else if let Some(rest) = line.strip_prefix("v ") {
            for token in rest.split_whitespace() {
                if ended {
                    return Err("a value after the terminating 0".to_owned());
                }
                let lit: i64 = token
                    .parse()
                    .map_err(|_| format!("the value {token:?} is not an integer"))?;
                if lit == 0 {
                    ended = true;
                    continue;
                }
                let k = usize::try_from(lit.unsigned_abs()).unwrap_or(usize::MAX);
                let slot = values
                    .get_mut(k.wrapping_sub(1))
                    .ok_or_else(|| format!("the variable {k} is out of range"))?;
                if slot.replace(lit > 0).is_some() {
                    return Err(format!("the variable {k} is assigned twice"));
                }
            }
        }
    }
    match status.as_deref() {
        Some("SATISFIABLE") => {
            if !ended {
                return Err("the values do not end with 0".to_owned());
            }
            values
                .into_iter()
                .enumerate()
                .map(|(k, value)| {
                    value.ok_or_else(|| format!("the variable {} has no value", k + 1))
                })
                .collect::<Result<_, _>>()
                .map(Verdict::Sat)
        }
        Some("UNSATISFIABLE") => Ok(Verdict::Unsat),
        Some("UNKNOWN") => Ok(Verdict::Unknown),
        Some(other) => Err(format!("the status {other:?} is unknown")),
        None => Err("no status line".to_owned()),
    }
}

/// Why a solver run gave no verdict.
#[derive(Debug)]
pub enum RunError {
    /// The executable could not start or its pipes failed.
    Spawn(std::io::Error),
    /// The run passed its timeout and was killed.
    TimedOut,
    /// The cancel token fired and the run was killed.
    Cancelled,
    /// The output does not parse.
    Malformed(String),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(e) => write!(f, "the SAT solver did not run: {e}"),
            Self::TimedOut => f.write_str("the SAT solver passed its timeout"),
            Self::Cancelled => f.write_str("the SAT solve was cancelled"),
            Self::Malformed(why) => write!(f, "the SAT solver output is malformed: {why}"),
        }
    }
}

impl std::error::Error for RunError {}

/// One finished solver run.
#[derive(Clone, Debug)]
pub struct Run {
    /// The arguments passed to the executable.
    pub args: Vec<String>,
    /// The exit code, if the process exited normally.
    pub exit_code: Option<i32>,
    /// The wall time of the run.
    pub elapsed: Duration,
    /// The parsed answer.
    pub verdict: Verdict,
}

/// An external Kissat executable with a wall-time limit.
#[derive(Clone, Debug)]
pub struct Kissat {
    path: PathBuf,
    timeout: Duration,
}

/// How often a running solve checks its deadline and cancel token.
const POLL: Duration = Duration::from_millis(10);

/// The extra time a solver gets past its own `--time` limit before it is killed.
const GRACE: Duration = Duration::from_secs(30);

/// Kissat accepts seeds from 0 to `i32::MAX`. A job seed keeps its low 31
/// bits, so consecutive read seeds stay distinct.
const SEED_MASK: u64 = 0x7FFF_FFFF;

/// Read `pipe` to the end on its own thread, so a full pipe never blocks the
/// child. The text arrives on the returned channel.
fn drain(mut pipe: impl Read + Send + 'static) -> mpsc::Receiver<std::io::Result<String>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut text = String::new();
        // The receiver may have given up; the text is then unwanted.
        let _ = tx.send(pipe.read_to_string(&mut text).map(|_| text));
    });
    rx
}

/// Kill the process group that `child` leads, then reap the child. The group
/// holds any process a wrapper script started, so none outlives the run.
fn kill_group(child: &mut Child) {
    // Best effort: the group may already be gone.
    let _ = Command::new("kill")
        .args(["-KILL", "--", &format!("-{}", child.id())])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = child.kill();
    let _ = child.wait();
}

impl Kissat {
    /// A solver at `path` that stops after `timeout`.
    #[must_use]
    pub fn new(path: PathBuf, timeout: Duration) -> Self {
        Self { path, timeout }
    }

    /// The executable path.
    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Solve `encoding` with `seed`, folded into Kissat's 31-bit seed range.
    /// One thread, quiet output, and Kissat's own time limit. The solver runs
    /// in its own process group. The group is killed when the run passes the
    /// limit plus a grace period of at most 30 s, when `cancel` (a token and
    /// the job's watermark) fires, and after the solver exits.
    ///
    /// # Errors
    ///
    /// [`RunError`] when the process fails, overruns, is cancelled, or prints
    /// output that does not parse.
    pub fn run(
        &self,
        encoding: &Encoding,
        seed: u64,
        cancel: Option<(&CancelToken, Option<u64>)>,
    ) -> Result<Run, RunError> {
        let seconds = self.timeout.as_secs().max(1);
        let args = vec![
            "-q".to_owned(),
            format!("--seed={}", seed & SEED_MASK),
            format!("--time={seconds}"),
        ];
        let deadline = self.timeout.saturating_add(GRACE.min(self.timeout));
        let started = Instant::now();
        let mut child = Command::new(&self.path)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .spawn()
            .map_err(RunError::Spawn)?;
        let cnf = to_dimacs(encoding);
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| RunError::Spawn(std::io::Error::other("the solver has no stdin")))?;
        // Not joined: a descendant that never reads would block the write, and
        // a solver may exit before it reads all of its input. The verdict
        // decides, not the write.
        std::thread::spawn(move || stdin.write_all(cnf.as_bytes()));
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| RunError::Spawn(std::io::Error::other("the solver has no stdout")))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| RunError::Spawn(std::io::Error::other("the solver has no stderr")))?;
        let reader = drain(stdout);
        let errors = drain(stderr);
        let status = loop {
            if let Some(status) = child.try_wait().map_err(RunError::Spawn)? {
                break status;
            }
            let stop = if cancel.is_some_and(|(token, watermark)| token.is_cancelled(watermark)) {
                Some(RunError::Cancelled)
            } else if started.elapsed() > deadline {
                Some(RunError::TimedOut)
            } else {
                None
            };
            if let Some(error) = stop {
                kill_group(&mut child);
                return Err(error);
            }
            std::thread::sleep(POLL);
        };
        // Close the pipes that any leftover descendant still holds. The group id
        // outlives the reaped leader only while members remain, and Linux
        // assigns process ids in sequence, so the id is not reused this soon.
        kill_group(&mut child);
        let remaining = deadline.saturating_sub(started.elapsed()).max(POLL);
        let text = reader
            .recv_timeout(remaining)
            .map_err(|_| RunError::TimedOut)?
            .map_err(RunError::Spawn)?;
        let verdict = parse_output(&text, encoding.nodes.len()).map_err(|why| {
            let stderr = errors
                .recv_timeout(POLL * 10)
                .ok()
                .and_then(Result::ok)
                .unwrap_or_default();
            let first = stderr.lines().next().unwrap_or("no stderr");
            let code = status
                .code()
                .map_or_else(|| "a signal".to_owned(), |code| format!("exit code {code}"));
            RunError::Malformed(format!("{why} ({code}: {first})"))
        })?;
        Ok(Run {
            args,
            exit_code: status.code(),
            elapsed: started.elapsed(),
            verdict,
        })
    }
}
