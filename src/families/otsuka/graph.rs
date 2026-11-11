//! Graph preparation: validate the motifs against the model and split every
//! coupling into the strong layer the cluster draw handles and the weak layer
//! the heat bath scores.

use std::collections::HashMap;
use std::fmt;

/// The strong tile coupling, in milli. A tile edge between positions `a` and
/// `b` carries `STRONG_MILLI * g[a] * g[b]`.
pub const STRONG_MILLI: i32 = 1000;

/// The pair coupling, in milli.
pub const PAIR_MILLI: i32 = -1000;

/// The explicit Ising model, in milli. Energy is `sum h s + sum J s s`.
#[derive(Clone, Copy, Debug)]
pub struct Model<'a> {
    /// One field per node. The kernel supports only zero fields.
    pub h_milli: &'a [i32],
    /// Each edge once, in either orientation.
    pub edges: &'a [(u32, u32)],
    /// One coupling per edge.
    pub j_milli: &'a [i32],
}

/// The motifs of a gauged-ice instance.
#[derive(Clone, Copy, Debug)]
pub struct Motifs<'a> {
    /// Four distinct nodes per tile. Tiles may share nodes but not edges.
    pub tiles: &'a [[u32; 4]],
    /// The local gauge of each tile, one sign per tile position.
    pub gauges: &'a [[i8; 4]],
    /// Node-disjoint pair edges with coupling [`PAIR_MILLI`].
    pub pairs: &'a [(u32, u32)],
}

/// How much of each strong coupling the cluster draw handles, in milli. The
/// remainder joins the weak layer, so the energy does not change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Weights {
    /// The pair bond weight, `0..=1000`.
    pub pair_milli: i32,
    /// The tile bond weight, `1..=1000`.
    pub tile_milli: i32,
}

impl Weights {
    /// Every strong coupling in the cluster draw.
    pub const FULL: Self = Self {
        pair_milli: 1000,
        tile_milli: 1000,
    };

    fn is_valid(self) -> bool {
        (0..=1000).contains(&self.pair_milli) && (1..=1000).contains(&self.tile_milli)
    }
}

/// Why a model and its motifs cannot run on the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphError {
    /// The model has no nodes.
    Empty,
    /// The model has more than [`MAX_NODES`] nodes.
    TooLarge,
    /// A weight is outside its range.
    Weights,
    /// A field is not zero.
    Fields,
    /// The edge and coupling counts, or the tile and gauge counts, differ.
    Lengths,
    /// An edge end is out of range, the edge is a loop, or `|J| > 1000`.
    Edge {
        /// The edge position.
        index: usize,
    },
    /// The edge repeats an earlier edge.
    DuplicateEdge {
        /// The position of the repeat.
        index: usize,
    },
    /// A tile node is out of range or repeated.
    Tile {
        /// The tile position.
        tile: usize,
    },
    /// A gauge entry is not `1` or `-1`.
    Gauge {
        /// The tile position.
        tile: usize,
    },
    /// A tile edge is missing or its coupling disagrees with the gauge.
    TileCoupling {
        /// The tile position.
        tile: usize,
    },
    /// A tile edge already belongs to an earlier tile.
    TileOverlap {
        /// The tile position.
        tile: usize,
    },
    /// A pair node is out of range or taken, the pair coupling is not
    /// [`PAIR_MILLI`], or the pair edge is a tile edge.
    Pair {
        /// The pair position.
        pair: usize,
    },
    /// An edge with `|J| = 1000` belongs to no motif.
    StrongEdge {
        /// The edge position.
        index: usize,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "the model has no nodes"),
            Self::TooLarge => write!(f, "the kernel takes at most {MAX_NODES} nodes"),
            Self::Weights => write!(
                f,
                "the pair weight must be in 0..=1000 and the tile weight in 1..=1000 milli"
            ),
            Self::Fields => write!(f, "the cluster kernel supports only zero fields"),
            Self::Lengths => write!(f, "edge, coupling, tile, or gauge counts differ"),
            Self::Edge { index } => write!(f, "edge {index} is out of range or above 1000 milli"),
            Self::DuplicateEdge { index } => write!(f, "edge {index} repeats an earlier edge"),
            Self::Tile { tile } => write!(f, "tile {tile} has an invalid or repeated node"),
            Self::Gauge { tile } => write!(f, "tile {tile} has a gauge entry that is not a sign"),
            Self::TileCoupling { tile } => {
                write!(f, "tile {tile} has an edge that does not match its gauge")
            }
            Self::TileOverlap { tile } => write!(f, "tile {tile} shares an edge with a tile"),
            Self::Pair { pair } => write!(f, "pair {pair} is not a disjoint -1000 milli edge"),
            Self::StrongEdge { index } => write!(f, "edge {index} is strong but in no motif"),
        }
    }
}

impl std::error::Error for GraphError {}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Tile {
    pub(crate) nodes: [u32; 4],
    pub(crate) gauge: [i8; 4],
}

/// The most nodes a graph may have: a [`Link`] holds the node in 21 bits.
pub const MAX_NODES: usize = 1 << 21;

/// A neighbour and its coupling in one word: the node in the high 21 bits
/// and the coupling plus 1024 in the low 11. Every coupling the kernel keeps
/// is within 1000 milli. The small word keeps a model's rows in the core's
/// cache.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Link(u32);

impl Link {
    const SHIFT: u32 = 11;

    fn new(node: u32, j: i32) -> Self {
        debug_assert!((node as usize) < MAX_NODES && j.abs() <= STRONG_MILLI);
        Self(node << Self::SHIFT | (j + 1024) as u32)
    }

    /// The neighbour.
    #[inline]
    pub(crate) fn node(self) -> usize {
        (self.0 >> Self::SHIFT) as usize
    }

    /// The coupling, in milli.
    #[inline]
    pub(crate) fn j(self) -> i32 {
        (self.0 & ((1 << Self::SHIFT) - 1)) as i32 - 1024
    }
}

/// Adjacency in compressed rows: the neighbours of `u` and their couplings
/// are `adj[start[u]..start[u + 1]]`.
#[derive(Debug)]
pub(crate) struct Csr {
    start: Vec<usize>,
    adj: Vec<Link>,
}

impl Csr {
    fn build(n: usize, edges: &[Edge]) -> Self {
        let mut start = vec![0; n + 1];
        for &(u, v, _) in edges {
            start[u as usize + 1] += 1;
            start[v as usize + 1] += 1;
        }
        for u in 0..n {
            start[u + 1] += start[u];
        }
        let mut fill = start.clone();
        let mut adj = vec![Link(0); 2 * edges.len()];
        for &(u, v, j) in edges {
            for (from, end) in [(u, v), (v, u)] {
                adj[fill[from as usize]] = Link::new(end, j);
                fill[from as usize] += 1;
            }
        }
        Self { start, adj }
    }

    /// The neighbours of `u` and their couplings.
    pub(crate) fn row(&self, u: usize) -> &[Link] {
        &self.adj[self.start[u]..self.start[u + 1]]
    }
}

/// A validated gauged-ice graph split into strong and weak layers.
#[derive(Debug)]
pub struct OtsukaGraph {
    pub(crate) n: usize,
    pub(crate) tiles: Vec<Tile>,
    pub(crate) pairs: Vec<(u32, u32)>,
    pub(crate) weights: Weights,
    /// Every coupling of the model.
    edges: Vec<Edge>,
    /// The weak layer: non-motif edges plus the residuals of soft weights,
    /// each as one end and the [`Link`] to the other.
    pub(crate) weak_edges: Vec<(u32, Link)>,
    pub(crate) weak: Csr,
    /// Every coupling, for single-spin moves.
    pub(crate) all: Csr,
}

/// An edge and its coupling in milli. Node positions fit `u32`, because the
/// model names them with `u32` ids.
pub(crate) type Edge = (u32, u32, i32);

/// The edge position of each unordered node pair.
type EdgeIndex = HashMap<(usize, usize), usize>;

fn key(u: usize, v: usize) -> (usize, usize) {
    (u.min(v), u.max(v))
}

fn node(id: u32, n: usize) -> Option<usize> {
    usize::try_from(id).ok().filter(|&u| u < n)
}

fn check_edges(model: &Model<'_>, n: usize) -> Result<(Vec<Edge>, EdgeIndex), GraphError> {
    let mut edges = Vec::with_capacity(model.edges.len());
    let mut index = EdgeIndex::with_capacity(model.edges.len());
    for (at, (&(u, v), &j)) in model.edges.iter().zip(model.j_milli).enumerate() {
        let (Some(u), Some(v)) = (node(u, n), node(v, n)) else {
            return Err(GraphError::Edge { index: at });
        };
        if u == v || !(-STRONG_MILLI..=STRONG_MILLI).contains(&j) {
            return Err(GraphError::Edge { index: at });
        }
        if index.insert(key(u, v), at).is_some() {
            return Err(GraphError::DuplicateEdge { index: at });
        }
        edges.push((u as u32, v as u32, j));
    }
    Ok((edges, index))
}

/// Check each tile and mark its edges in `strong`.
fn check_tiles(
    motifs: &Motifs<'_>,
    n: usize,
    edges: &[Edge],
    index: &EdgeIndex,
    strong: &mut [bool],
) -> Result<Vec<Tile>, GraphError> {
    let mut tiles = Vec::with_capacity(motifs.tiles.len());
    for (tile, (ids, &gauge)) in motifs.tiles.iter().zip(motifs.gauges).enumerate() {
        let mut nodes = [0; 4];
        for (slot, &id) in nodes.iter_mut().zip(ids) {
            *slot = node(id, n).ok_or(GraphError::Tile { tile })?;
        }
        if (0..4).any(|a| (a + 1..4).any(|b| nodes[a] == nodes[b])) {
            return Err(GraphError::Tile { tile });
        }
        if gauge.iter().any(|&g| g != 1 && g != -1) {
            return Err(GraphError::Gauge { tile });
        }
        for a in 0..4 {
            for b in a + 1..4 {
                let want = STRONG_MILLI * i32::from(gauge[a] * gauge[b]);
                let at = *index
                    .get(&key(nodes[a], nodes[b]))
                    .ok_or(GraphError::TileCoupling { tile })?;
                if edges[at].2 != want {
                    return Err(GraphError::TileCoupling { tile });
                }
                if std::mem::replace(&mut strong[at], true) {
                    return Err(GraphError::TileOverlap { tile });
                }
            }
        }
        tiles.push(Tile {
            nodes: nodes.map(|u| u as u32),
            gauge,
        });
    }
    Ok(tiles)
}

/// Check each pair and mark its edge in `strong`.
fn check_pairs(
    motifs: &Motifs<'_>,
    n: usize,
    edges: &[Edge],
    index: &EdgeIndex,
    strong: &mut [bool],
) -> Result<Vec<(u32, u32)>, GraphError> {
    let mut taken = vec![false; n];
    let mut pairs = Vec::with_capacity(motifs.pairs.len());
    for (pair, &(u, v)) in motifs.pairs.iter().enumerate() {
        let bad = GraphError::Pair { pair };
        let (Some(u), Some(v)) = (node(u, n), node(v, n)) else {
            return Err(bad);
        };
        if u == v || taken[u] || taken[v] {
            return Err(bad);
        }
        let at = *index.get(&key(u, v)).ok_or(bad)?;
        if edges[at].2 != PAIR_MILLI || std::mem::replace(&mut strong[at], true) {
            return Err(bad);
        }
        taken[u] = true;
        taken[v] = true;
        pairs.push((u as u32, v as u32));
    }
    Ok(pairs)
}

impl OtsukaGraph {
    /// Validate `model` against `motifs` and split its couplings.
    ///
    /// # Errors
    ///
    /// Returns the first [`GraphError`] in the order the variants are
    /// listed, before any sampling work.
    pub fn new(
        model: &Model<'_>,
        motifs: &Motifs<'_>,
        weights: Weights,
    ) -> Result<Self, GraphError> {
        let n = model.h_milli.len();
        if n == 0 {
            return Err(GraphError::Empty);
        }
        if n > MAX_NODES {
            return Err(GraphError::TooLarge);
        }
        if !weights.is_valid() {
            return Err(GraphError::Weights);
        }
        if model.h_milli.iter().any(|&h| h != 0) {
            return Err(GraphError::Fields);
        }
        if model.edges.len() != model.j_milli.len() || motifs.tiles.len() != motifs.gauges.len() {
            return Err(GraphError::Lengths);
        }
        let (edges, index) = check_edges(model, n)?;
        let mut strong = vec![false; edges.len()];
        let tiles = check_tiles(motifs, n, &edges, &index, &mut strong)?;
        let pairs = check_pairs(motifs, n, &edges, &index, &mut strong)?;
        let mut weak_edges = Vec::new();
        for (at, (&edge, &is_strong)) in edges.iter().zip(&strong).enumerate() {
            if is_strong {
                continue;
            }
            if edge.2.abs() >= STRONG_MILLI {
                return Err(GraphError::StrongEdge { index: at });
            }
            weak_edges.push(edge);
        }
        let pair_residual = weights.pair_milli - 1000;
        if pair_residual != 0 {
            weak_edges.extend(pairs.iter().map(|&(u, v)| (u, v, pair_residual)));
        }
        let tile_residual = 1000 - weights.tile_milli;
        if tile_residual != 0 {
            for tile in &tiles {
                for a in 0..4 {
                    for b in a + 1..4 {
                        let j = tile_residual * i32::from(tile.gauge[a] * tile.gauge[b]);
                        weak_edges.push((tile.nodes[a], tile.nodes[b], j));
                    }
                }
            }
        }
        Ok(Self {
            n,
            tiles,
            pairs,
            weights,
            weak: Csr::build(n, &weak_edges),
            all: Csr::build(n, &edges),
            weak_edges: weak_edges
                .iter()
                .map(|&(u, v, j)| (u, Link::new(v, j)))
                .collect(),
            edges,
        })
    }

    /// The number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    /// The model energy of `s`, in milli.
    #[must_use]
    pub fn energy(&self, s: &[i8]) -> i64 {
        self.edges
            .iter()
            .map(|&(u, v, j)| i64::from(j) * i64::from(s[u as usize] * s[v as usize]))
            .sum()
    }

    /// The energy of the strong layer at the graph's weights, in milli.
    #[must_use]
    pub fn strong_energy(&self, s: &[i8]) -> i64 {
        let tile_milli = i64::from(self.weights.tile_milli);
        let tiles: i64 = self
            .tiles
            .iter()
            .map(|tile| {
                let q: i64 = (0..4)
                    .map(|k| i64::from(tile.gauge[k] * s[tile.nodes[k] as usize]))
                    .sum();
                tile_milli * (q * q - 4) / 2
            })
            .sum();
        let pairs: i64 = self
            .pairs
            .iter()
            .map(|&(u, v)| {
                i64::from(self.weights.pair_milli) * i64::from(s[u as usize] * s[v as usize])
            })
            .sum();
        tiles - pairs
    }

    /// The energy of the weak layer, in milli: the model energy less the
    /// strong layer.
    #[must_use]
    pub fn weak_energy(&self, s: &[i8]) -> i64 {
        self.energy(s) - self.strong_energy(s)
    }
}
