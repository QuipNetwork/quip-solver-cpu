//! Bounded string and loop search for `quip/defects-static` revision 1.
//!
//! The family puts one dipolar island on each link of an open square
//! lattice. A state is feasible when every labeled vertex has its requested
//! charge and every pinned link has spin `+1`. The instance energy of a
//! feasible state is its physical energy: the kept dipolar couplings and the
//! field. The family has no research solver, so this module is a new search.
//!
//! A move flips a set of links that keeps every requested constraint:
//!
//! - a **loop** is a directed cycle in the current orientation, so every
//!   vertex on it keeps its charge;
//! - a **string** is a directed path between two free boundary vertices, so
//!   only free vertices change charge.
//!
//! A move is a list of canonical link ids of the open lattice. No move flips a
//! pinned link or enters an obstacle, so the defect strings of a state never
//! cross a prohibited region. [`Problem::check_move`] checks every move
//! directly before the search scores it.
//!
//! The search keeps the local field of every node and updates the full score
//! of the instance with checked integer arithmetic. At the end of each read it
//! recomputes the score from the coefficients and compares the two. Every
//! coefficient is an integer in milli, so acceptance compares exact integers.
//! The dipolar table rounds each kept coupling by at most half a milli, so two
//! states can trade places by at most one milli for each kept coupling. An
//! improvement within that bound is reported as not resolved.
//!
//! The table has the oracle calibration. It supports analytic fixtures and
//! code tests only. A request for a physical claim fails with
//! [`DipolarError::Uncalibrated`] until a calibrated record exists.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::time::Instant;

use quip_solver_core::ising_generators::defects::{DEFECTS, DEFECTS_SCHEMA};
use quip_solver_core::ising_generators::descriptor::{Params, Value};
use quip_solver_core::ising_generators::dipolar::{
    physical_status, quantize, Axis as DipolarAxis, Calibration, Entry, Geometry, Island,
    PhysicalStatus, Table,
};
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::ising_generators::square::{
    check_request, Axis, Boundary, Request as SquareRequest, SquareLattice,
};
use quip_solver_core::CancelToken;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::otsuka_sampler::section;

/// The search policy, versioned apart from the family revision.
pub const DIPOLAR_POLICY: &str = "dipolar-strings/1";

/// The largest lattice side that the tests cover. Revision 1 admits sides 2
/// and 3 only.
pub const MAX_TESTED_SIDE: u16 = 3;

/// The dipolar constant `D` of revision 1.
const DIPOLAR_D: i32 = 1;
/// The dipolar scale `S` of revision 1: a table entry is milli of `D`.
const DIPOLAR_SCALE: u32 = 1000;
/// The squared truncation radius `R2` of revision 1.
const DIPOLAR_R2: u32 = 4;
/// The metadata value of a free vertex.
const FREE_LABEL: i8 = 127;
/// Metadata kind of the charge labels.
const KIND_CHARGES: u16 = 4;
/// Metadata kind of the window and the link positions.
const KIND_INCIDENCE: u16 = 5;
/// Metadata kind of the weight, the table summary, the obstacles, and the pins.
const KIND_DEFECTS: u16 = 8;
/// Parameter tags of revision 1.
const TAG_SIDE: u16 = 3;
const TAG_FIELD_X: u16 = 7;
const TAG_FIELD_Y: u16 = 8;
/// The clock and the cancel token are read once per this many proposals.
const POLL_EVERY: u64 = 64;

/// What the result may claim.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Evidence {
    /// Code tests and analytic fixtures. The table may be an oracle.
    #[default]
    Analytic,
    /// A measured physical claim. It needs a calibrated table.
    Physical,
}

/// Why a request, a state, or a run is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DipolarError {
    /// The parameter bytes do not parse or are out of bounds.
    Params,
    /// The instance metadata is not valid.
    Metadata(&'static str),
    /// The side is outside the tested sides.
    UntestedSide(u16),
    /// The request is malformed or has no feasible state.
    Request(String),
    /// A physical claim was requested, and the table has no calibration.
    Uncalibrated,
    /// The table commitment of the instance is not the rebuilt oracle table.
    CalibrationMismatch,
    /// The instance coefficients differ from the coefficients of the request.
    Coefficients,
    /// A state is not one `+1` or `-1` spin per node.
    Spins,
    /// A checked operation overflowed.
    Overflow,
    /// A move fails the direct check.
    Move(MoveFault),
    /// A state violates a requested constraint.
    Infeasible(Violations),
    /// The incremental score differs from the full recomputation.
    ScoreMismatch {
        /// The incremental score.
        incremental: i64,
        /// The recomputed score.
        full: i64,
    },
    /// The cancel token fired.
    Cancelled,
}

impl fmt::Display for DipolarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Params => f.write_str("the defects-static parameters do not parse"),
            Self::Metadata(why) => write!(f, "the defects-static metadata is invalid: {why}"),
            Self::UntestedSide(side) => write!(
                f,
                "side {side} is outside the tested sides 2 to {MAX_TESTED_SIDE}"
            ),
            Self::Request(why) => write!(f, "the defects-static request is refused: {why}"),
            Self::Uncalibrated => f.write_str(
                "insufficient physical evidence: the dipolar table has the oracle calibration, \
                 and a physical job needs a calibrated record",
            ),
            Self::CalibrationMismatch => {
                f.write_str("the table commitment is not the oracle table of the geometry")
            }
            Self::Coefficients => {
                f.write_str("the instance coefficients differ from the coefficients of the request")
            }
            Self::Spins => f.write_str("a state is not one +1 or -1 spin per node"),
            Self::Overflow => f.write_str("an energy overflows"),
            Self::Move(fault) => write!(f, "a move fails the direct check: {fault:?}"),
            Self::Infeasible(v) => write!(
                f,
                "a state violates {} labels and {} pins",
                v.labels, v.pins
            ),
            Self::ScoreMismatch { incremental, full } => write!(
                f,
                "the incremental score {incremental} differs from the full score {full}"
            ),
            Self::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for DipolarError {}

/// Why a move fails the direct check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveFault {
    /// The move has no link.
    Empty,
    /// The move has more links than the lattice.
    TooLong,
    /// A link id is not a link of the lattice.
    UnknownLink(u32),
    /// A link appears twice.
    RepeatedLink(u32),
    /// A link is pinned.
    Pinned(u32),
    /// Two consecutive links share no vertex.
    NotAdjacent(u32),
    /// A link points against the walk in the current orientation.
    NotDirected(u32),
    /// The walk visits a vertex twice before it closes.
    RepeatedVertex(u32),
    /// The walk enters an obstacle.
    Obstacle(u32),
    /// The walk ends on a labeled vertex, so it changes that charge.
    ChargeChanged(u32),
}

/// The kind of a valid move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveKind {
    /// A directed cycle.
    Loop,
    /// A directed path between two free vertices.
    String,
}

/// The violated constraints of a state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Violations {
    /// Labeled vertices whose charge differs from the label.
    pub labels: u32,
    /// Pinned links whose spin is not `+1`.
    pub pins: u32,
}

impl Violations {
    /// Whether the state meets every constraint.
    #[must_use]
    pub const fn feasible(self) -> bool {
        self.labels == 0 && self.pins == 0
    }
}

/// A defects-static request on the open lattice of side `side`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    /// The lattice side `L`.
    pub side: u16,
    /// One label per vertex in vertex order. `None` is a free vertex.
    pub labels: Vec<Option<i8>>,
    /// The dense node position of each link, in link order.
    pub positions: Vec<u32>,
    /// Obstacle vertices, strictly ascending.
    pub obstacles: Vec<u32>,
    /// Pinned links, strictly ascending.
    pub pins: Vec<u32>,
    /// The field along `+x` in milli.
    pub field_x: i32,
    /// The field along `+y` in milli.
    pub field_y: i32,
}

/// The values that kind 8 stores.
struct Stored {
    weight: u64,
    dropped: u64,
    commitment: [u8; 32],
}

/// A little-endian reader over one metadata payload.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, tail) = self.0.split_at_checked(n)?;
        self.0 = tail;
        Some(head)
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn ids(&mut self) -> Option<Vec<u32>> {
        let count = self.u16()?;
        (0..count).map(|_| self.u32()).collect()
    }
}

/// A signed parameter value.
fn signed(params: &Params<'_>, tag: u16) -> Option<i32> {
    match params.get(tag) {
        Some(Value::Signed(v)) => i32::try_from(v).ok(),
        _ => None,
    }
}

/// An unsigned parameter value.
fn unsigned(params: &Params<'_>, tag: u16) -> Option<u64> {
    match params.get(tag) {
        Some(Value::Unsigned(v)) => Some(v),
        _ => None,
    }
}

impl Request {
    /// The request of a validated instance, and what kind 8 stores.
    fn from_instance(
        params: &Params<'_>,
        instance: &Instance,
    ) -> Result<(Self, Stored), DipolarError> {
        let side = unsigned(params, TAG_SIDE)
            .and_then(|s| u16::try_from(s).ok())
            .ok_or(DipolarError::Params)?;
        let field_x = signed(params, TAG_FIELD_X).ok_or(DipolarError::Params)?;
        let field_y = signed(params, TAG_FIELD_Y).ok_or(DipolarError::Params)?;
        let region = instance.metadata();
        let missing = DipolarError::Metadata("a section is missing or malformed");

        let mut charges = Reader(section(region, KIND_CHARGES).ok_or_else(|| missing.clone())?);
        if charges.u16() != Some(side) {
            return Err(DipolarError::Metadata("the charge side is not the side"));
        }
        let labels = charges
            .0
            .iter()
            .map(|&b| {
                let label = i8::from_le_bytes([b]);
                (label != FREE_LABEL).then_some(label)
            })
            .collect();

        let mut incidence = Reader(section(region, KIND_INCIDENCE).ok_or_else(|| missing.clone())?);
        incidence.take(4).ok_or_else(|| missing.clone())?;
        let (chunks, tail) = incidence.0.as_chunks::<4>();
        if !tail.is_empty() {
            return Err(missing);
        }
        let positions = chunks.iter().map(|c| u32::from_le_bytes(*c)).collect();

        let mut defects = Reader(section(region, KIND_DEFECTS).ok_or_else(|| missing.clone())?);
        let weight = defects.u64().ok_or_else(|| missing.clone())?;
        let dropped = defects.u64().ok_or_else(|| missing.clone())?;
        let commitment: [u8; 32] = defects
            .take(32)
            .and_then(|c| c.try_into().ok())
            .ok_or_else(|| missing.clone())?;
        let obstacles = defects.ids().ok_or_else(|| missing.clone())?;
        let pins = defects.ids().ok_or_else(|| missing.clone())?;
        if !defects.0.is_empty() {
            return Err(missing);
        }
        Ok((
            Self {
                side,
                labels,
                positions,
                obstacles,
                pins,
                field_x,
                field_y,
            },
            Stored {
                weight,
                dropped,
                commitment,
            },
        ))
    }
}

/// Whether vertex `v` of the open lattice of side `side` is on the border.
fn on_border(side: u16, v: u32) -> bool {
    let width = u32::from(side) + 1;
    let (x, y) = (v % width, v / width);
    x == 0 || y == 0 || x == u32::from(side) || y == u32::from(side)
}

/// Whether `list` is strictly ascending.
fn ascending(list: &[u32]) -> bool {
    list.windows(2).all(|w| matches!(w, [a, b] if a < b))
}

/// The oracle table over the link midpoints of `lattice`.
fn oracle_table(lattice: &SquareLattice) -> Option<Table> {
    let width = u32::from(lattice.side()) + 1;
    let mut islands = Vec::new();
    for e in 0..lattice.link_count() {
        let link = lattice.link(e)?;
        let x = u16::try_from(link.tail % width).ok()?.checked_mul(2)?;
        let y = u16::try_from(link.tail / width).ok()?.checked_mul(2)?;
        islands.push(match link.axis {
            Axis::X => Island {
                x: x.checked_add(1)?,
                y,
                axis: DipolarAxis::PlusX,
                moment: 1,
            },
            Axis::Y => Island {
                x,
                y: y.checked_add(1)?,
                axis: DipolarAxis::PlusY,
                moment: 1,
            },
        });
    }
    let geometry = Geometry::new(islands, DIPOLAR_D, DIPOLAR_SCALE, DIPOLAR_R2).ok()?;
    let entries = quantize(&geometry).ok()?;
    Table::new(geometry, entries, Calibration::Oracle).ok()
}

/// Whether links `i` and `j` share a vertex.
fn share_vertex(lattice: &SquareLattice, i: u32, j: u32) -> Option<bool> {
    let (a, b) = (lattice.link(i)?, lattice.link(j)?);
    Some(
        [a.tail, a.head]
            .iter()
            .any(|v| *v == b.tail || *v == b.head),
    )
}

/// A validated request with its coefficients, ready to search.
#[derive(Debug)]
pub struct Problem {
    lattice: SquareLattice,
    request: Request,
    /// Per vertex.
    obstacle: Vec<bool>,
    /// Per link.
    pinned: Vec<bool>,
    /// The vertices a walk may start from: every vertex but the obstacles.
    starts: Vec<u32>,
    h: Vec<i64>,
    edges: Vec<(u32, u32)>,
    j: Vec<i64>,
    offset: i64,
    /// `(neighbour, J)` of each node.
    neighbours: Vec<Vec<(usize, i64)>>,
    kept: Vec<Entry>,
    weight: u64,
    dropped: u64,
    commitment: [u8; 32],
    truncation_bound: u128,
    status: PhysicalStatus,
    /// The inverse temperatures of the schedule, in 1 / milli.
    beta: (f64, f64),
}

impl Problem {
    /// Validate `request` and build its coefficients from the oracle table,
    /// as step 6 of the contract does.
    ///
    /// # Errors
    ///
    /// [`DipolarError::UntestedSide`] for a side above [`MAX_TESTED_SIDE`],
    /// [`DipolarError::Request`] for a malformed or infeasible request, and
    /// [`DipolarError::Uncalibrated`] when `evidence` asks for a physical
    /// claim, because the table has the oracle calibration.
    pub fn new(request: Request, evidence: Evidence) -> Result<Self, DipolarError> {
        if !(2..=MAX_TESTED_SIDE).contains(&request.side) {
            return Err(DipolarError::UntestedSide(request.side));
        }
        let refuse = |why: &str| Err(DipolarError::Request(why.to_owned()));
        let lattice = SquareLattice::new(request.side, Boundary::Open)
            .map_err(|e| DipolarError::Request(e.to_string()))?;
        let (vertices, links) = (lattice.vertex_count(), lattice.link_count());
        let n = links as usize;
        if request.labels.len() != vertices as usize {
            return refuse("the labels are not one per vertex");
        }
        for (v, label) in (0u32..).zip(&request.labels) {
            match (on_border(request.side, v), label) {
                (true, None) | (false, Some(-2 | 0 | 2)) => {}
                (true, Some(_)) => return refuse("a boundary vertex has a label"),
                (false, _) => return refuse("an interior label is not -2, 0, or +2"),
            }
        }
        let mut link_of = vec![None; n];
        if request.positions.len() != n {
            return refuse("the positions are not one per link");
        }
        for (e, &p) in (0u32..).zip(&request.positions) {
            match link_of.get_mut(p as usize) {
                Some(slot @ None) => *slot = Some(e),
                _ => return refuse("the positions are not a permutation of the nodes"),
            }
        }
        if !ascending(&request.obstacles) {
            return refuse("the obstacles are not strictly ascending");
        }
        let mut obstacle = vec![false; vertices as usize];
        for &v in &request.obstacles {
            let zero = request.labels.get(v as usize) == Some(&Some(0));
            if v >= vertices || on_border(request.side, v) || !zero {
                return refuse("an obstacle is not an interior vertex with the label 0");
            }
            obstacle[v as usize] = true;
        }
        if !ascending(&request.pins) {
            return refuse("the pins are not strictly ascending");
        }
        let mut pinned = vec![false; n];
        for &e in &request.pins {
            let boundary = lattice.link(e).is_some_and(|link| {
                on_border(request.side, link.tail) && on_border(request.side, link.head)
            });
            if !boundary {
                return refuse("a pin is not a boundary link");
            }
            pinned[e as usize] = true;
        }
        let square = SquareRequest {
            charges: request.labels.clone(),
            fixed: request.pins.iter().map(|&e| (e, 1)).collect(),
            fields: Vec::new(),
        };
        check_request(&lattice, &square).map_err(|e| DipolarError::Request(e.to_string()))?;

        let table = oracle_table(&lattice).ok_or(DipolarError::Overflow)?;
        if evidence == Evidence::Physical && matches!(table.calibration(), Calibration::Oracle) {
            return Err(DipolarError::Uncalibrated);
        }
        let mut kept = Vec::new();
        let (mut dropped, mut kept_magnitude) = (0u64, 0u64);
        for entry in table.entries() {
            let magnitude = u64::from(entry.q.unsigned_abs());
            if share_vertex(&lattice, entry.i, entry.j).ok_or(DipolarError::Overflow)? {
                kept.push(*entry);
                kept_magnitude = kept_magnitude
                    .checked_add(magnitude)
                    .ok_or(DipolarError::Overflow)?;
            } else {
                dropped = dropped
                    .checked_add(magnitude)
                    .ok_or(DipolarError::Overflow)?;
            }
        }
        let weight =
            penalty_weight(&request, links, kept_magnitude).ok_or(DipolarError::Overflow)?;
        let built =
            Coefficients::build(&lattice, &request, &kept, weight).ok_or(DipolarError::Overflow)?;
        let mut neighbours = vec![Vec::new(); n];
        for (&(a, b), &j) in built.edges.iter().zip(&built.j) {
            neighbours[a as usize].push((b as usize, j));
            neighbours[b as usize].push((a as usize, j));
        }
        let beta = schedule(&kept, &request);
        let starts = (0..vertices).filter(|&v| !obstacle[v as usize]).collect();
        Ok(Self {
            starts,
            status: physical_status(&table),
            truncation_bound: table.truncation_bound(),
            commitment: table.commitment(),
            lattice,
            request,
            obstacle,
            pinned,
            h: built.h,
            edges: built.edges,
            j: built.j,
            offset: built.offset,
            neighbours,
            kept,
            weight,
            dropped,
            beta,
        })
    }

    /// The problem of a `quip/defects-static` revision 1 instance.
    ///
    /// The contract metadata check runs first. Then the request is read from
    /// kinds 4, 5, and 8, and validated by [`Problem::new`]. The stored
    /// weight, dropped magnitude, and table commitment must equal the rebuilt
    /// values, and the instance coefficients must equal the coefficients of
    /// the request.
    ///
    /// # Errors
    ///
    /// The first failed check.
    pub fn for_instance(
        params: &[u8],
        instance: &Instance,
        evidence: Evidence,
    ) -> Result<Self, DipolarError> {
        let parsed = Params::parse(DEFECTS_SCHEMA, params).map_err(|_| DipolarError::Params)?;
        parsed.check_bounds().map_err(|_| DipolarError::Params)?;
        if !(DEFECTS.check_metadata)(&parsed, instance) {
            return Err(DipolarError::Metadata("the contract metadata check fails"));
        }
        let (request, stored) = Request::from_instance(&parsed, instance)?;
        let problem = Self::new(request, evidence)?;
        if stored.commitment != problem.commitment {
            return Err(DipolarError::CalibrationMismatch);
        }
        if stored.weight != problem.weight || stored.dropped != problem.dropped {
            return Err(DipolarError::Metadata(
                "the weight or the dropped magnitude differs",
            ));
        }
        let same = instance
            .h()
            .iter()
            .map(|&v| i64::from(v))
            .eq(problem.h.iter().copied())
            && instance.edges() == problem.edges.as_slice()
            && instance
                .j()
                .iter()
                .map(|&v| i64::from(v))
                .eq(problem.j.iter().copied())
            && instance.offset_milli() == problem.offset;
        if !same {
            return Err(DipolarError::Coefficients);
        }
        Ok(problem)
    }

    /// The validated request.
    #[must_use]
    pub const fn request(&self) -> &Request {
        &self.request
    }

    /// The open lattice.
    #[must_use]
    pub const fn lattice(&self) -> &SquareLattice {
        &self.lattice
    }

    /// The number of nodes, one per link.
    #[must_use]
    pub fn nodes(&self) -> usize {
        self.h.len()
    }

    /// The fields in milli, in node order.
    #[must_use]
    pub fn h(&self) -> &[i64] {
        &self.h
    }

    /// The edges as dense node positions, ascending.
    #[must_use]
    pub fn edges(&self) -> &[(u32, u32)] {
        &self.edges
    }

    /// The couplings in milli, one per edge.
    #[must_use]
    pub fn j(&self) -> &[i64] {
        &self.j
    }

    /// The offset in milli.
    #[must_use]
    pub const fn offset(&self) -> i64 {
        self.offset
    }

    /// The penalty weight `U` in milli.
    #[must_use]
    pub const fn weight(&self) -> u64 {
        self.weight
    }

    /// The sum of `|q|` over the table entries that the layout drops.
    #[must_use]
    pub const fn dropped(&self) -> u64 {
        self.dropped
    }

    /// The table commitment, which names the calibration.
    #[must_use]
    pub const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    /// The table bound on the pairs beyond `R2`, in units of `q`.
    #[must_use]
    pub const fn truncation_bound(&self) -> u128 {
        self.truncation_bound
    }

    /// What the table may claim about physics.
    #[must_use]
    pub const fn physical_status(&self) -> PhysicalStatus {
        self.status
    }

    /// The bound in milli on the rounding error of an energy difference
    /// between two states: each kept coupling is rounded by at most half a
    /// milli and its term changes by at most twice its value.
    #[must_use]
    pub fn rounding_bound_milli(&self) -> u64 {
        self.kept.len() as u64
    }

    /// The link spins of a node state.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Spins`] for a state of the wrong length or a value
    /// that is not `+1` or `-1`.
    pub fn link_spins(&self, spins: &[i8]) -> Result<Vec<i8>, DipolarError> {
        if spins.len() != self.nodes() || spins.iter().any(|&s| s != 1 && s != -1) {
            return Err(DipolarError::Spins);
        }
        Ok(self
            .request
            .positions
            .iter()
            .map(|&p| spins[p as usize])
            .collect())
    }

    /// The charge of every vertex for link spins `links`.
    fn charges(&self, links: &[i8]) -> Vec<i32> {
        (0..self.lattice.vertex_count())
            .map(|v| {
                self.lattice
                    .incident(v)
                    .iter()
                    .map(|&(e, sign)| i32::from(sign) * i32::from(links[e as usize]))
                    .sum()
            })
            .collect()
    }

    /// The violated labels and pins of a node state.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Spins`] for a malformed state.
    pub fn violations(&self, spins: &[i8]) -> Result<Violations, DipolarError> {
        let links = self.link_spins(spins)?;
        Ok(self.link_violations(&links))
    }

    fn link_violations(&self, links: &[i8]) -> Violations {
        let charges = self.charges(links);
        let labels = charges
            .iter()
            .zip(&self.request.labels)
            .filter(|(&q, label)| label.is_some_and(|want| i32::from(want) != q))
            .count();
        let pins = self
            .request
            .pins
            .iter()
            .filter(|&&e| links[e as usize] != 1)
            .count();
        Violations {
            labels: u32::try_from(labels).unwrap_or(u32::MAX),
            pins: u32::try_from(pins).unwrap_or(u32::MAX),
        }
    }

    /// The instance energy of a node state, without the offset, recomputed
    /// from every coefficient with checked arithmetic.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Spins`] or [`DipolarError::Overflow`].
    pub fn energy(&self, spins: &[i8]) -> Result<i64, DipolarError> {
        self.link_spins(spins)?;
        let mut total = 0i64;
        for (&h, &s) in self.h.iter().zip(spins) {
            total = total
                .checked_add(h * i64::from(s))
                .ok_or(DipolarError::Overflow)?;
        }
        for (&(a, b), &j) in self.edges.iter().zip(&self.j) {
            let pair = i64::from(spins[a as usize]) * i64::from(spins[b as usize]);
            total = total
                .checked_add(j.checked_mul(pair).ok_or(DipolarError::Overflow)?)
                .ok_or(DipolarError::Overflow)?;
        }
        Ok(total)
    }

    /// The physical energy of a node state in milli: the kept couplings and
    /// the field, without any penalty, from the table directly.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Spins`] or [`DipolarError::Overflow`].
    pub fn physical_energy(&self, spins: &[i8]) -> Result<i64, DipolarError> {
        let links = self.link_spins(spins)?;
        let mut total = 0i64;
        for entry in &self.kept {
            let pair = i64::from(links[entry.i as usize]) * i64::from(links[entry.j as usize]);
            total = total
                .checked_add(i64::from(entry.q) * pair)
                .ok_or(DipolarError::Overflow)?;
        }
        for (e, &s) in (0u32..).zip(&links) {
            let field = self.field_of(e);
            total = total
                .checked_sub(i64::from(field) * i64::from(s))
                .ok_or(DipolarError::Overflow)?;
        }
        Ok(total)
    }

    fn field_of(&self, e: u32) -> i32 {
        match self.lattice.link(e).map(|link| link.axis) {
            Some(Axis::X) => self.request.field_x,
            _ => self.request.field_y,
        }
    }

    /// The tail and head of link `e` in the orientation of spin `s`.
    fn oriented(&self, e: u32, s: i8) -> Option<(u32, u32)> {
        let link = self.lattice.link(e)?;
        Some(if s == 1 {
            (link.tail, link.head)
        } else {
            (link.head, link.tail)
        })
    }

    fn free(&self, v: u32) -> bool {
        self.request.labels.get(v as usize) == Some(&None)
    }

    /// Check a move against a node state. A move is a list of link ids that
    /// forms a directed walk in the current orientation. It must be a simple
    /// directed cycle, or a simple directed path between two free vertices.
    /// It may not flip a pin or enter an obstacle.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Spins`] for a malformed state, and
    /// [`DipolarError::Move`] with the first fault of the move.
    pub fn check_move(&self, spins: &[i8], links: &[u32]) -> Result<MoveKind, DipolarError> {
        let state = self.link_spins(spins)?;
        self.check_links(&state, links).map_err(DipolarError::Move)
    }

    fn check_links(&self, state: &[i8], links: &[u32]) -> Result<MoveKind, MoveFault> {
        let count = self.lattice.link_count();
        let first = *links.first().ok_or(MoveFault::Empty)?;
        if links.len() > count as usize {
            return Err(MoveFault::TooLong);
        }
        let mut used = vec![false; count as usize];
        for &e in links {
            let slot = used.get_mut(e as usize).ok_or(MoveFault::UnknownLink(e))?;
            if *slot {
                return Err(MoveFault::RepeatedLink(e));
            }
            *slot = true;
            if self.pinned[e as usize] {
                return Err(MoveFault::Pinned(e));
            }
        }
        let orient = |e: u32| {
            self.oriented(e, state[e as usize])
                .ok_or(MoveFault::UnknownLink(e))
        };
        let (start, _) = orient(first)?;
        let mut visited = vec![false; self.lattice.vertex_count() as usize];
        let mut at = start;
        for &e in links {
            let (from, to) = orient(e)?;
            if from != at {
                return Err(if to == at {
                    MoveFault::NotDirected(e)
                } else {
                    MoveFault::NotAdjacent(e)
                });
            }
            if visited[at as usize] {
                return Err(MoveFault::RepeatedVertex(at));
            }
            visited[at as usize] = true;
            if self.obstacle[at as usize] {
                return Err(MoveFault::Obstacle(at));
            }
            at = to;
        }
        if at == start {
            return Ok(MoveKind::Loop);
        }
        if visited[at as usize] {
            return Err(MoveFault::RepeatedVertex(at));
        }
        if self.obstacle[at as usize] {
            return Err(MoveFault::Obstacle(at));
        }
        for v in [start, at] {
            if !self.free(v) {
                return Err(MoveFault::ChargeChanged(v));
            }
        }
        Ok(MoveKind::String)
    }

    /// The node of link `e`.
    fn node(&self, e: u32) -> usize {
        self.request.positions[e as usize] as usize
    }

    /// Propose a bounded move: a random directed walk from a random vertex
    /// that is not an obstacle. The walk never takes a pinned link or enters
    /// an obstacle. It stops at the first closed cycle, or at the first
    /// directed path between two free vertices, and it gives up after as many
    /// steps as the lattice has links or at a vertex with no way out.
    pub fn propose(&self, state: &State, rng: &mut SmallRng) -> Option<Vec<u32>> {
        let vertices = self.lattice.vertex_count() as usize;
        let v0 = *self
            .starts
            .get(rng.gen_range(0..self.starts.len().max(1)))?;
        let mut index: Vec<Option<usize>> = vec![None; vertices];
        let mut path = vec![v0];
        let mut links: Vec<u32> = Vec::new();
        index[v0 as usize] = Some(0);
        let mut first_free = self.free(v0).then_some(0usize);
        let mut out = Vec::with_capacity(4);
        while links.len() < self.lattice.link_count() as usize {
            let u = *path.last()?;
            out.clear();
            for &(e, _) in self.lattice.incident(u) {
                let s = state.spins[self.node(e)];
                let (from, to) = self.oriented(e, s)?;
                if from == u && !self.pinned[e as usize] && !self.obstacle[to as usize] {
                    out.push((e, to));
                }
            }
            if out.is_empty() {
                return None;
            }
            let (e, w) = out[rng.gen_range(0..out.len())];
            links.push(e);
            if let Some(k) = index[w as usize] {
                return Some(links.split_off(k));
            }
            index[w as usize] = Some(path.len());
            path.push(w);
            if self.free(w) {
                match first_free {
                    Some(k) => return Some(links.split_off(k)),
                    None => first_free = Some(path.len() - 1),
                }
            }
        }
        None
    }

    /// The shortest directed path of non-pinned links from `v`, forward
    /// along out-links or backward along in-links, to a vertex that `target`
    /// accepts. An obstacle is a transit vertex only when `obstacles` is set.
    fn bfs(
        &self,
        links: &[i8],
        v: u32,
        forward: bool,
        obstacles: bool,
        target: &dyn Fn(u32) -> bool,
    ) -> Option<Vec<u32>> {
        let vertices = self.lattice.vertex_count() as usize;
        let mut via: Vec<Option<(u32, u32)>> = vec![None; vertices];
        let mut seen = vec![false; vertices];
        seen[v as usize] = true;
        let mut queue = VecDeque::from([v]);
        while let Some(u) = queue.pop_front() {
            for &(e, _) in self.lattice.incident(u) {
                if self.pinned[e as usize] {
                    continue;
                }
                let (from, to) = self.oriented(e, links[e as usize])?;
                let next = match (forward, from == u) {
                    (true, true) => to,
                    (false, false) if to == u => from,
                    _ => continue,
                };
                if seen[next as usize] || (self.obstacle[next as usize] && !obstacles) {
                    continue;
                }
                seen[next as usize] = true;
                via[next as usize] = Some((u, e));
                if target(next) {
                    let mut path = Vec::new();
                    let mut at = next;
                    while let Some((prev, link)) = via[at as usize] {
                        path.push(link);
                        at = prev;
                    }
                    return Some(path);
                }
                queue.push_back(next);
            }
        }
        None
    }

    /// A feasible state near `spins`: every pin is set to `+1`, then each
    /// wrong label is fixed by flipping one shortest directed path to a
    /// vertex with the opposite error, or else to a free vertex. Paths avoid
    /// obstacles when they can. The result is checked directly.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Spins`] for a malformed state, and
    /// [`DipolarError::Infeasible`] when no path fixes a label.
    pub fn repair(&self, spins: &[i8]) -> Result<Vec<i8>, DipolarError> {
        let mut links = self.link_spins(spins)?;
        for &e in &self.request.pins {
            links[e as usize] = 1;
        }
        let rounds = links.len() * 4 + 4;
        for _ in 0..rounds {
            let charges = self.charges(&links);
            let error = |v: u32| -> i32 {
                self.request.labels[v as usize]
                    .map_or(0, |want| charges[v as usize] - i32::from(want))
            };
            let Some(v) = (0..self.lattice.vertex_count()).find(|&v| error(v) != 0) else {
                return self.finish_repair(&links);
            };
            // A deficit gains in-links along a path out of `v`, a surplus
            // loses them along a path into `v`.
            let forward = error(v) < 0;
            let opposite = |w: u32| if forward { error(w) > 0 } else { error(w) < 0 };
            let either = |w: u32| opposite(w) || self.free(w);
            let path = self
                .bfs(&links, v, forward, false, &opposite)
                .or_else(|| self.bfs(&links, v, forward, false, &either))
                .or_else(|| self.bfs(&links, v, forward, true, &either));
            let Some(path) = path else {
                return Err(DipolarError::Infeasible(self.link_violations(&links)));
            };
            for e in path {
                links[e as usize] = -links[e as usize];
            }
        }
        Err(DipolarError::Infeasible(self.link_violations(&links)))
    }

    fn finish_repair(&self, links: &[i8]) -> Result<Vec<i8>, DipolarError> {
        let violations = self.link_violations(links);
        if !violations.feasible() {
            return Err(DipolarError::Infeasible(violations));
        }
        let mut spins = vec![0i8; self.nodes()];
        for (e, &s) in links.iter().enumerate() {
            spins[self.node(e as u32)] = s;
        }
        Ok(spins)
    }

    /// The feasible start of a search read: the reference state, with every
    /// link spin `+1`, repaired by paths between opposite labels.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Infeasible`] when the repair fails.
    pub fn start_state(&self) -> Result<Vec<i8>, DipolarError> {
        self.repair(&vec![1; self.nodes()])
    }

    /// Run the search for one read from a feasible state.
    ///
    /// The proposals follow a geometric schedule of inverse temperatures
    /// over the work. A move is accepted with the Metropolis rule on its
    /// exact integer change. The read returns the lowest state it visited,
    /// after the incremental score, the full recomputation, the physical
    /// energy, and the direct constraint check agree.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Infeasible`] for an infeasible start,
    /// [`DipolarError::Cancelled`], or a failed check.
    pub fn search_read(
        &self,
        start: &[i8],
        work: Work,
        seed: u64,
        cancel: Option<(&CancelToken, Option<u64>)>,
    ) -> Result<ReadOutcome, DipolarError> {
        let violations = self.violations(start)?;
        if !violations.feasible() {
            return Err(DipolarError::Infeasible(violations));
        }
        let started = Instant::now();
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(self, start)?;
        let start_energy = state.score;
        let mut best = (state.score, state.spins.clone());
        let mut counts = MoveCounts::default();
        let (hot, cold) = self.beta;
        let mut fraction = 0.0;
        for t in 0u64.. {
            match work {
                Work::Fixed { proposals } => {
                    if t >= proposals {
                        break;
                    }
                    fraction = t as f64 / proposals as f64;
                }
                Work::Budget { seconds } => {
                    if t.is_multiple_of(POLL_EVERY) {
                        fraction = started.elapsed().as_secs_f64() / seconds;
                        if fraction >= 1.0 {
                            break;
                        }
                    }
                }
            }
            if t.is_multiple_of(POLL_EVERY)
                && cancel.is_some_and(|(token, mark)| token.is_cancelled(mark))
            {
                return Err(DipolarError::Cancelled);
            }
            counts.proposed += 1;
            let Some(links) = self.propose(&state, &mut rng) else {
                counts.no_move += 1;
                continue;
            };
            match self.check_links(&state.links(self), &links) {
                Ok(MoveKind::Loop) => counts.loops += 1,
                Ok(MoveKind::String) => counts.strings += 1,
                Err(_) => {
                    counts.refused += 1;
                    continue;
                }
            }
            let delta = state.delta(self, &links)?;
            let beta = hot * (cold / hot).powf(fraction);
            if delta <= 0 || rng.gen::<f64>() < (-beta * delta as f64).exp() {
                state.apply(self, &links)?;
                counts.accepted += 1;
                if state.score < best.0 {
                    best = (state.score, state.spins.clone());
                }
            }
        }
        let full = self.energy(&state.spins)?;
        if full != state.score {
            return Err(DipolarError::ScoreMismatch {
                incremental: state.score,
                full,
            });
        }
        let (energy, spins) = best;
        self.checked_final(&spins, energy)?;
        let improvement = start_energy
            .checked_sub(energy)
            .ok_or(DipolarError::Overflow)?;
        Ok(ReadOutcome {
            physical_energy: energy
                .checked_add(self.offset)
                .ok_or(DipolarError::Overflow)?,
            resolved: improvement.unsigned_abs() > self.rounding_bound_milli(),
            spins,
            energy,
            improvement,
            feasible: true,
            repaired: false,
            counts,
            seconds: started.elapsed().as_secs_f64(),
        })
    }

    /// The checks of a returned state: the full score, the direct
    /// constraints, and the physical energy from the table.
    fn checked_final(&self, spins: &[i8], energy: i64) -> Result<(), DipolarError> {
        let full = self.energy(spins)?;
        if full != energy {
            return Err(DipolarError::ScoreMismatch {
                incremental: energy,
                full,
            });
        }
        let violations = self.violations(spins)?;
        if !violations.feasible() {
            return Err(DipolarError::Infeasible(violations));
        }
        let physical = self.physical_energy(spins)?;
        let shifted = energy
            .checked_add(self.offset)
            .ok_or(DipolarError::Overflow)?;
        if physical != shifted {
            return Err(DipolarError::ScoreMismatch {
                incremental: shifted,
                full: physical,
            });
        }
        Ok(())
    }

    /// Run a job. With `starts`, the postprocess mode runs one read from each
    /// supplied state, repaired first when it is infeasible. A state that
    /// cannot be repaired is returned unchanged and marked infeasible. Without
    /// `starts`, the search mode runs `reads` reads from [`Problem::start_state`].
    /// Read `r` seeds its generator from `seed` and `r` only. The result keeps
    /// the final state of every read, in order.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Cancelled`] or a failed check.
    pub fn run(
        &self,
        starts: Option<&[Vec<i8>]>,
        reads: usize,
        work: Work,
        seed: u64,
        cancel: Option<(&CancelToken, Option<u64>)>,
    ) -> Result<RunOutcome, DipolarError> {
        let started = Instant::now();
        let owned;
        let (starts, warm) = match starts {
            Some(starts) => (starts, true),
            None => {
                owned = vec![self.start_state()?; reads];
                (owned.as_slice(), false)
            }
        };
        let mut outcomes = Vec::with_capacity(starts.len());
        for (r, start) in starts.iter().enumerate() {
            let read_seed = read_seed(seed, r as u64);
            let violations = self.violations(start)?;
            let repaired = !violations.feasible();
            let begin = if repaired {
                match self.repair(start) {
                    Ok(fixed) => fixed,
                    Err(DipolarError::Infeasible(_)) => {
                        outcomes.push(self.unrepaired(start)?);
                        continue;
                    }
                    Err(other) => return Err(other),
                }
            } else {
                start.clone()
            };
            let mut outcome = self.search_read(&begin, work, read_seed, cancel)?;
            outcome.repaired = repaired;
            outcomes.push(outcome);
        }
        Ok(RunOutcome {
            policy: DIPOLAR_POLICY,
            work,
            warm,
            reads: outcomes,
            elapsed_seconds: started.elapsed().as_secs_f64(),
        })
    }

    /// The capture of a supplied state that no repair could fix.
    fn unrepaired(&self, spins: &[i8]) -> Result<ReadOutcome, DipolarError> {
        Ok(ReadOutcome {
            energy: self.energy(spins)?,
            physical_energy: self.physical_energy(spins)?,
            spins: spins.to_vec(),
            improvement: 0,
            resolved: false,
            feasible: false,
            repaired: false,
            counts: MoveCounts::default(),
            seconds: 0.0,
        })
    }
}

/// The generator seed of read `r`: SplitMix64 of the job seed and the read
/// index. It never reads the instance nonce.
#[must_use]
pub const fn read_seed(seed: u64, r: u64) -> u64 {
    let mut z = seed.wrapping_add(r.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The weight `U` of the contract penalty bound.
fn penalty_weight(request: &Request, links: u32, kept_magnitude: u64) -> Option<u64> {
    let per_axis = u64::from(links / 2);
    let field = u64::from(request.field_x.unsigned_abs())
        .checked_add(u64::from(request.field_y.unsigned_abs()))?;
    let magnitude = kept_magnitude.checked_add(per_axis.checked_mul(field)?)?;
    let mut weight = magnitude.checked_mul(2)? / 2 + 1;
    if weight % 2 == 1 {
        weight = weight.checked_add(1)?;
    }
    Some(weight)
}

/// The inverse temperatures in 1 / milli: hot accepts an uphill move of the
/// largest coefficient with probability one half, cold accepts an uphill
/// move of the smallest nonzero coefficient with probability 1 / 1000.
fn schedule(kept: &[Entry], request: &Request) -> (f64, f64) {
    let magnitudes = kept
        .iter()
        .map(|e| e.q.unsigned_abs())
        .chain([
            request.field_x.unsigned_abs(),
            request.field_y.unsigned_abs(),
        ])
        .filter(|&m| m > 0);
    let (low, high) = magnitudes.fold((u32::MAX, 1), |(lo, hi), m| (lo.min(m), hi.max(m)));
    let low = if low == u32::MAX { 1 } else { low };
    (
        std::f64::consts::LN_2 / f64::from(high),
        1000f64.ln() / f64::from(low),
    )
}

/// The coefficients of step 6 of the contract.
struct Coefficients {
    h: Vec<i64>,
    edges: Vec<(u32, u32)>,
    j: Vec<i64>,
    offset: i64,
}

impl Coefficients {
    fn build(
        lattice: &SquareLattice,
        request: &Request,
        kept: &[Entry],
        weight: u64,
    ) -> Option<Self> {
        let u = i64::try_from(weight).ok()?;
        let pos = |e: u32| request.positions.get(e as usize).map(|&p| p as usize);
        let mut h = vec![0i64; request.positions.len()];
        let mut pairs: BTreeMap<(u32, u32), i64> = BTreeMap::new();
        let mut offset = 0i64;
        let mut add_pair = |a: usize, b: usize, value: i64| -> Option<()> {
            let (a, b) = (u32::try_from(a).ok()?, u32::try_from(b).ok()?);
            let slot = pairs.entry((a.min(b), a.max(b))).or_default();
            *slot = slot.checked_add(value)?;
            Some(())
        };
        for e in 0..lattice.link_count() {
            let field = match lattice.link(e)?.axis {
                Axis::X => request.field_x,
                Axis::Y => request.field_y,
            };
            let slot = h.get_mut(pos(e)?)?;
            *slot = slot.checked_sub(i64::from(field))?;
        }
        for entry in kept {
            add_pair(pos(entry.i)?, pos(entry.j)?, i64::from(entry.q))?;
        }
        for (v, label) in (0u32..).zip(&request.labels) {
            let Some(q) = label.map(i64::from) else {
                continue;
            };
            let at = lattice.incident(v);
            let degree = i64::try_from(at.len()).ok()?;
            offset = offset.checked_add((u / 2).checked_mul(degree.checked_add(q * q)?)?)?;
            for (k, &(e, ie)) in at.iter().enumerate() {
                let slot = h.get_mut(pos(e)?)?;
                *slot = slot.checked_sub(u.checked_mul(q * i64::from(ie))?)?;
                for &(f, jf) in at.get(k + 1..)? {
                    add_pair(
                        pos(e)?,
                        pos(f)?,
                        u.checked_mul(i64::from(ie) * i64::from(jf))?,
                    )?;
                }
            }
        }
        for &e in &request.pins {
            let slot = h.get_mut(pos(e)?)?;
            *slot = slot.checked_sub(u)?;
            offset = offset.checked_add(u)?;
        }
        let (edges, j) = pairs.into_iter().filter(|&(_, v)| v != 0).unzip();
        Some(Self {
            h,
            edges,
            j,
            offset,
        })
    }
}

/// A search state: the node spins, the local field of each node, and the
/// instance energy without the offset.
#[derive(Clone, Debug)]
pub struct State {
    spins: Vec<i8>,
    field: Vec<i64>,
    score: i64,
}

impl State {
    /// The state of `spins` on `problem`, with its fields and score.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Spins`] or [`DipolarError::Overflow`].
    pub fn new(problem: &Problem, spins: &[i8]) -> Result<Self, DipolarError> {
        let score = problem.energy(spins)?;
        let mut field = problem.h.clone();
        for (&(a, b), &j) in problem.edges.iter().zip(&problem.j) {
            let (a, b) = (a as usize, b as usize);
            field[a] = field[a]
                .checked_add(j * i64::from(spins[b]))
                .ok_or(DipolarError::Overflow)?;
            field[b] = field[b]
                .checked_add(j * i64::from(spins[a]))
                .ok_or(DipolarError::Overflow)?;
        }
        Ok(Self {
            spins: spins.to_vec(),
            field,
            score,
        })
    }

    /// The node spins.
    #[must_use]
    pub fn spins(&self) -> &[i8] {
        &self.spins
    }

    /// The incremental score: the instance energy without the offset.
    #[must_use]
    pub const fn score(&self) -> i64 {
        self.score
    }

    fn links(&self, problem: &Problem) -> Vec<i8> {
        problem
            .request
            .positions
            .iter()
            .map(|&p| self.spins[p as usize])
            .collect()
    }

    /// The exact change of the score when the links of `links` flip.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Overflow`], or [`DipolarError::Move`] for an unknown
    /// link.
    pub fn delta(&self, problem: &Problem, links: &[u32]) -> Result<i64, DipolarError> {
        let overflow = DipolarError::Overflow;
        let mut flipped = vec![false; self.spins.len()];
        for &e in links {
            let node = *problem
                .request
                .positions
                .get(e as usize)
                .ok_or(DipolarError::Move(MoveFault::UnknownLink(e)))?;
            flipped[node as usize] = true;
        }
        let mut delta = 0i64;
        for (i, _) in flipped.iter().enumerate().filter(|(_, &f)| f) {
            let s = i64::from(self.spins[i]);
            delta = delta
                .checked_sub(
                    self.field[i]
                        .checked_mul(2 * s)
                        .ok_or_else(|| overflow.clone())?,
                )
                .ok_or_else(|| overflow.clone())?;
            // A coupling with both ends flipped does not change.
            for &(k, j) in &problem.neighbours[i] {
                if k > i && flipped[k] {
                    let pair = j * s * i64::from(self.spins[k]);
                    delta = delta
                        .checked_add(pair.checked_mul(4).ok_or_else(|| overflow.clone())?)
                        .ok_or_else(|| overflow.clone())?;
                }
            }
        }
        Ok(delta)
    }

    /// Flip the links of `links` and update the fields and the score.
    ///
    /// # Errors
    ///
    /// [`DipolarError::Overflow`], or [`DipolarError::Move`] for an unknown
    /// link.
    pub fn apply(&mut self, problem: &Problem, links: &[u32]) -> Result<i64, DipolarError> {
        let delta = self.delta(problem, links)?;
        for &e in links {
            let i = problem.node(e);
            self.spins[i] = -self.spins[i];
            let now = i64::from(self.spins[i]);
            for &(k, j) in &problem.neighbours[i] {
                self.field[k] = self.field[k]
                    .checked_add(2 * j * now)
                    .ok_or(DipolarError::Overflow)?;
            }
        }
        self.score = self
            .score
            .checked_add(delta)
            .ok_or(DipolarError::Overflow)?;
        Ok(delta)
    }
}

/// How much work one read does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Work {
    /// Exactly this many proposals per read. The result depends only on the
    /// inputs and the seed.
    Fixed {
        /// The proposals of one read.
        proposals: u64,
    },
    /// Proposals until the declared budget of elapsed time on the calling
    /// thread is spent. The clock decides the proposal count, so the result
    /// depends on the host.
    Budget {
        /// The declared seconds of one read.
        seconds: f64,
    },
}

/// What one read did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoveCounts {
    /// Proposals drawn.
    pub proposed: u64,
    /// Walks that found no move.
    pub no_move: u64,
    /// Valid loop moves.
    pub loops: u64,
    /// Valid string moves.
    pub strings: u64,
    /// Moves that failed the direct check.
    pub refused: u64,
    /// Moves accepted.
    pub accepted: u64,
}

/// The final state of one read.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadOutcome {
    /// The node spins.
    pub spins: Vec<i8>,
    /// The instance energy without the offset, in milli.
    pub energy: i64,
    /// The physical energy in milli. For a feasible state it is the instance
    /// energy with the offset.
    pub physical_energy: i64,
    /// The start energy less the final energy, in milli.
    pub improvement: i64,
    /// Whether the improvement exceeds [`Problem::rounding_bound_milli`].
    pub resolved: bool,
    /// Whether the state meets every constraint.
    pub feasible: bool,
    /// Whether a supplied state was repaired before the search.
    pub repaired: bool,
    /// The work of the read.
    pub counts: MoveCounts,
    /// The measured time of the read, in seconds.
    pub seconds: f64,
}

/// What a job did.
#[derive(Clone, Debug, PartialEq)]
pub struct RunOutcome {
    /// [`DIPOLAR_POLICY`].
    pub policy: &'static str,
    /// The work mode of every read.
    pub work: Work,
    /// Whether the reads started from supplied states.
    pub warm: bool,
    /// The final state of every read, in order.
    pub reads: Vec<ReadOutcome>,
    /// The measured time of the whole job, in seconds.
    pub elapsed_seconds: f64,
}
