//! Exact first-improvement greedy descent over single flips and natural
//! multi-spin moves.
//!
//! Port of the research descent `greedy_descent` (research inventory
//! `pp-greedy-c`), with the move set of `natural_moves` and `MoveSet`
//! (`pp-descent`) and the rigid groups of `extract_groups`
//! (`pp-prep-groups`). Energies are integer milli. A move flips a set of
//! spins. Its energy change is the sum of `-2 s_i f_i` over its members plus
//! `4 J s_a s_b` over each edge inside it, where `f_i = h_i + sum_j J_ij s_j`.
//! A sweep visits the moves in order and applies every move that lowers the
//! energy. The descent stops after a sweep that applies none.

use std::collections::{BTreeSet, HashSet};
use std::fmt;

use quip_solver_core::CancelToken;

/// A coupling at or below this value, outside every tile, is a rigid bond.
pub const RIGID_MILLI: i64 = -500;

/// Why a model cannot run the descent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    /// The edge and coupling lists differ in length.
    Lengths,
    /// An edge names a node outside the model.
    Node {
        /// The edge position.
        index: usize,
    },
    /// An edge joins a node to itself.
    SelfLoop {
        /// The edge position.
        index: usize,
    },
    /// An edge repeats an earlier edge, in either orientation.
    DuplicateEdge {
        /// The position of the repeat.
        index: usize,
    },
    /// The total coefficient weight could overflow an energy or a delta.
    Overflow,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lengths => write!(f, "the edge and coupling lists differ in length"),
            Self::Node { index } => write!(f, "edge {index} names a node outside the model"),
            Self::SelfLoop { index } => write!(f, "edge {index} is a self-loop"),
            Self::DuplicateEdge { index } => write!(f, "edge {index} repeats an earlier edge"),
            Self::Overflow => write!(f, "the coefficient weight could overflow i64"),
        }
    }
}

impl std::error::Error for ModelError {}

/// Why a move list cannot run the descent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveError {
    /// A move names a node outside the model.
    Node {
        /// The node.
        node: u32,
    },
    /// A move names a node twice, so applying it would flip the node back.
    Repeat {
        /// The node.
        node: u32,
    },
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Node { node } => write!(f, "a move names node {node} outside the model"),
            Self::Repeat { node } => write!(f, "a move names node {node} twice"),
        }
    }
}

impl std::error::Error for MoveError {}

/// Why a descent stopped without a final state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescentError {
    /// The state does not have one spin per node.
    Length,
    /// A spin is not `1` or `-1`.
    Value {
        /// The node.
        node: usize,
    },
    /// The cancel token abandoned this job's watermark.
    Cancelled,
}

impl fmt::Display for DescentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length => write!(f, "the state does not have one spin per node"),
            Self::Value { node } => write!(f, "spin {node} is not 1 or -1"),
            Self::Cancelled => write!(f, "the descent was cancelled"),
        }
    }
}

impl std::error::Error for DescentError {}

/// An integer Ising model in milli, with a symmetric adjacency for local
/// fields.
#[derive(Clone, Debug)]
pub struct Ising {
    h: Vec<i64>,
    edges: Vec<(u32, u32)>,
    j: Vec<i64>,
    adj_ptr: Vec<usize>,
    adj: Vec<(u32, i64)>,
}

impl Ising {
    /// Check the model and build its adjacency.
    ///
    /// # Errors
    ///
    /// [`ModelError`] for mismatched lists, a bad or repeated edge, or a total
    /// weight `sum |h| + sum |J|` above `i64::MAX / 8`. Every energy and every
    /// move delta is bounded by four times that weight.
    pub fn new(h: &[i64], edges: &[(u32, u32)], j: &[i64]) -> Result<Self, ModelError> {
        if edges.len() != j.len() {
            return Err(ModelError::Lengths);
        }
        let n = h.len();
        let mut seen = HashSet::with_capacity(edges.len());
        let mut degree = vec![0_usize; n];
        for (at, &(u, v)) in edges.iter().enumerate() {
            if u as usize >= n || v as usize >= n {
                return Err(ModelError::Node { index: at });
            }
            if u == v {
                return Err(ModelError::SelfLoop { index: at });
            }
            if !seen.insert((u.min(v), u.max(v))) {
                return Err(ModelError::DuplicateEdge { index: at });
            }
            degree[u as usize] += 1;
            degree[v as usize] += 1;
        }
        let weight: i128 = h.iter().chain(j).map(|&x| i128::from(x).abs()).sum();
        if weight > i128::from(i64::MAX / 8) {
            return Err(ModelError::Overflow);
        }
        let mut adj_ptr = Vec::with_capacity(n + 1);
        adj_ptr.push(0);
        for d in &degree {
            adj_ptr.push(adj_ptr[adj_ptr.len() - 1] + d);
        }
        // Each row lists its neighbours in ascending order, like a CSR matrix.
        let mut rows: Vec<Vec<(u32, i64)>> =
            degree.iter().map(|&d| Vec::with_capacity(d)).collect();
        for (&(u, v), &jv) in edges.iter().zip(j) {
            rows[u as usize].push((v, jv));
            rows[v as usize].push((u, jv));
        }
        let mut adj = Vec::with_capacity(2 * edges.len());
        for mut row in rows {
            row.sort_unstable_by_key(|&(v, _)| v);
            adj.extend(row);
        }
        Ok(Self {
            h: h.to_vec(),
            edges: edges.to_vec(),
            j: j.to_vec(),
            adj_ptr,
            adj,
        })
    }

    /// The number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.h.len()
    }

    /// The energy of `s` in milli. `s` must have one spin per node.
    #[must_use]
    pub fn energy(&self, s: &[i8]) -> i64 {
        let fields: i64 = self.h.iter().zip(s).map(|(&h, &x)| h * i64::from(x)).sum();
        let couplings: i64 = self
            .edges
            .iter()
            .zip(&self.j)
            .map(|(&(u, v), &j)| j * i64::from(s[u as usize]) * i64::from(s[v as usize]))
            .sum();
        fields + couplings
    }

    fn row(&self, u: usize) -> &[(u32, i64)] {
        &self.adj[self.adj_ptr[u]..self.adj_ptr[u + 1]]
    }

    fn field(&self, s: &[i8], u: usize) -> i64 {
        self.h[u]
            + self
                .row(u)
                .iter()
                .map(|&(v, j)| j * i64::from(s[v as usize]))
                .sum::<i64>()
    }
}

/// The structure a family declares, for natural moves. Every list holds
/// node indices. Padding entries are the caller's to drop.
#[derive(Clone, Copy, Debug, Default)]
pub struct MoveSources<'a> {
    /// Four-node tiles. Each gives its 3- and 4-node subsets, and its edges
    /// never count as rigid bonds.
    pub tiles: &'a [[u32; 4]],
    /// Pairs. Each is a move and joins its rigid group.
    pub pairs: &'a [(u32, u32)],
    /// Other declared node sets, such as clusters, strong blocks, and
    /// vertical bonds. Each is one move.
    pub listed: &'a [Vec<u32>],
}

/// How many distinct moves each source added, in the order they are added.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoveCounts {
    /// One per coupled pair.
    pub edge_pairs: usize,
    /// The 3- and 4-node subsets of the tiles.
    pub tile_subsets: usize,
    /// The declared pairs.
    pub pairs: usize,
    /// The other declared node sets.
    pub listed: usize,
    /// The rigid groups of more than one node.
    pub rigid_groups: usize,
}

/// The natural moves of `ising`: sorted node tuples of at least two distinct
/// nodes, without repeats, in ascending lexicographic order.
///
/// # Errors
///
/// [`MoveError::Node`] when a source names a node outside the model.
pub fn natural_moves(
    ising: &Ising,
    sources: &MoveSources<'_>,
) -> Result<(Vec<Vec<u32>>, MoveCounts), MoveError> {
    let n = ising.num_nodes();
    let check = |node: u32| {
        if (node as usize) < n {
            Ok(node)
        } else {
            Err(MoveError::Node { node })
        }
    };
    let mut moves: BTreeSet<Vec<u32>> = BTreeSet::new();
    let mut add = |sets: &mut dyn Iterator<Item = Vec<u32>>| -> Result<usize, MoveError> {
        let before = moves.len();
        for set in sets {
            let mut members = set.into_iter().map(check).collect::<Result<Vec<_>, _>>()?;
            members.sort_unstable();
            members.dedup();
            if members.len() >= 2 {
                moves.insert(members);
            }
        }
        Ok(moves.len() - before)
    };
    let mut counts = MoveCounts {
        edge_pairs: add(&mut ising.edges.iter().map(|&(u, v)| vec![u, v]))?,
        ..MoveCounts::default()
    };
    let subsets: [&[usize]; 5] = [
        &[0, 1, 2],
        &[0, 1, 3],
        &[0, 2, 3],
        &[1, 2, 3],
        &[0, 1, 2, 3],
    ];
    counts.tile_subsets = add(&mut sources.tiles.iter().flat_map(|t| {
        subsets
            .iter()
            .map(move |c| c.iter().map(|&k| t[k]).collect())
    }))?;
    counts.pairs = add(&mut sources.pairs.iter().map(|&(u, v)| vec![u, v]))?;
    counts.listed = add(&mut sources.listed.iter().cloned())?;
    let groups = rigid_groups(ising, sources)?;
    counts.rigid_groups = add(&mut groups.into_iter())?;
    Ok((moves.into_iter().collect(), counts))
}

/// The components of more than one node joined by rigid bonds and pairs.
/// A rigid bond is an edge at or below [`RIGID_MILLI`] that is not a tile
/// edge.
fn rigid_groups(ising: &Ising, sources: &MoveSources<'_>) -> Result<Vec<Vec<u32>>, MoveError> {
    let n = ising.num_nodes();
    let mut tile_edges = BTreeSet::new();
    for t in sources.tiles {
        for a in 0..4 {
            for b in a + 1..4 {
                tile_edges.insert((t[a].min(t[b]), t[a].max(t[b])));
            }
        }
    }
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let rigid = ising
        .edges
        .iter()
        .zip(&ising.j)
        .filter(|(&(u, v), &j)| j <= RIGID_MILLI && !tile_edges.contains(&(u.min(v), u.max(v))))
        .map(|(&e, _)| e);
    for (u, v) in rigid.chain(sources.pairs.iter().copied()) {
        for node in [u, v] {
            if node as usize >= n {
                return Err(MoveError::Node { node });
            }
        }
        let (ru, rv) = (find(&mut parent, u as usize), find(&mut parent, v as usize));
        if ru != rv {
            parent[ru.max(rv)] = ru.min(rv);
        }
    }
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); n];
    for i in 0..n {
        let root = find(&mut parent, i);
        members[root].push(u32::try_from(i).map_err(|_| MoveError::Node { node: u32::MAX })?);
    }
    Ok(members.into_iter().filter(|g| g.len() > 1).collect())
}

/// A model with its move list, ready to descend.
#[derive(Clone, Debug)]
pub struct Greedy {
    ising: Ising,
    mv_ptr: Vec<usize>,
    mv_idx: Vec<u32>,
    in_ptr: Vec<usize>,
    /// Each internal edge as its two members and its coupling.
    inner: Vec<(u32, u32, i64)>,
}

impl Greedy {
    /// The move list: the single flips of every node when `with_singles` is
    /// set, then `moves` in the order given.
    ///
    /// # Errors
    ///
    /// [`MoveError::Node`] when a move names a node outside the model.
    pub fn new(ising: Ising, moves: &[Vec<u32>], with_singles: bool) -> Result<Self, MoveError> {
        let n = ising.num_nodes();
        let mut mv_ptr = vec![0];
        let mut mv_idx = Vec::new();
        let mut in_ptr = vec![0];
        let mut inner = Vec::new();
        let singles = (0..n).map(|i| u32::try_from(i).map(|i| vec![i]));
        let singles: Vec<Vec<u32>> = if with_singles {
            singles
                .collect::<Result<_, _>>()
                .map_err(|_| MoveError::Node { node: u32::MAX })?
        } else {
            Vec::new()
        };
        // `mark[i]` is one past the last move that holds node `i`, so a member
        // seen twice and an edge inside a move are found in O(degree).
        let mut mark = vec![0_usize; n];
        for (m, members) in singles.iter().chain(moves).enumerate() {
            for &node in members {
                let slot = mark
                    .get_mut(node as usize)
                    .ok_or(MoveError::Node { node })?;
                if *slot == m + 1 {
                    return Err(MoveError::Repeat { node });
                }
                *slot = m + 1;
            }
            mv_idx.extend_from_slice(members);
            mv_ptr.push(mv_idx.len());
            if members.len() > 1 {
                for &a in members {
                    for &(b, j) in ising.row(a as usize) {
                        if a < b && mark[b as usize] == m + 1 {
                            inner.push((a, b, j));
                        }
                    }
                }
            }
            in_ptr.push(inner.len());
        }
        Ok(Self {
            ising,
            mv_ptr,
            mv_idx,
            in_ptr,
            inner,
        })
    }

    /// The model.
    #[must_use]
    pub fn ising(&self) -> &Ising {
        &self.ising
    }

    /// The number of moves, single flips included.
    #[must_use]
    pub fn num_moves(&self) -> usize {
        self.mv_ptr.len() - 1
    }

    /// The nodes move `m` flips.
    #[must_use]
    pub fn members(&self, m: usize) -> &[u32] {
        &self.mv_idx[self.mv_ptr[m]..self.mv_ptr[m + 1]]
    }

    /// The energy change of applying move `m` at `s`, from fields computed
    /// directly.
    ///
    /// # Errors
    ///
    /// [`DescentError::Length`] or [`DescentError::Value`] when `s` does not
    /// fit the model.
    pub fn delta(&self, s: &[i8], m: usize) -> Result<i64, DescentError> {
        self.check(s)?;
        let flips: i64 = self
            .members(m)
            .iter()
            .map(|&i| -2 * i64::from(s[i as usize]) * self.ising.field(s, i as usize))
            .sum();
        Ok(flips + self.inner_term(s, m))
    }

    fn inner_term(&self, s: &[i8], m: usize) -> i64 {
        self.inner[self.in_ptr[m]..self.in_ptr[m + 1]]
            .iter()
            .map(|&(a, b, j)| 4 * j * i64::from(s[a as usize]) * i64::from(s[b as usize]))
            .sum()
    }

    /// Check that `s` has one `±1` spin per node.
    ///
    /// # Errors
    ///
    /// [`DescentError::Length`] or [`DescentError::Value`].
    pub fn check(&self, s: &[i8]) -> Result<(), DescentError> {
        if s.len() != self.ising.num_nodes() {
            return Err(DescentError::Length);
        }
        match s.iter().position(|&x| x != 1 && x != -1) {
            Some(node) => Err(DescentError::Value { node }),
            None => Ok(()),
        }
    }

    /// Descend from `s` in place and return the number of applied moves. The
    /// cancel token is checked before each sweep.
    ///
    /// # Errors
    ///
    /// [`DescentError`] when `s` does not fit the model or the token abandons
    /// the watermark. A cancelled descent leaves `s` part way down.
    pub fn descend(
        &self,
        s: &mut [i8],
        cancel: Option<(&CancelToken, Option<u64>)>,
    ) -> Result<u64, DescentError> {
        self.check(s)?;
        let n = self.ising.num_nodes();
        let mut f: Vec<i64> = (0..n).map(|i| self.ising.field(s, i)).collect();
        let mut applied = 0;
        loop {
            if cancel.is_some_and(|(token, watermark)| token.is_cancelled(watermark)) {
                return Err(DescentError::Cancelled);
            }
            let mut changed = false;
            for m in 0..self.num_moves() {
                let members = self.members(m);
                let flips: i64 = members
                    .iter()
                    .map(|&i| -2 * i64::from(s[i as usize]) * f[i as usize])
                    .sum();
                if flips + self.inner_term(s, m) >= 0 {
                    continue;
                }
                for &i in members {
                    let i = i as usize;
                    let old = i64::from(s[i]);
                    s[i] = -s[i];
                    for &(v, j) in self.ising.row(i) {
                        f[v as usize] -= 2 * j * old;
                    }
                }
                applied += 1;
                changed = true;
            }
            if !changed {
                return Ok(applied);
            }
        }
    }
}
