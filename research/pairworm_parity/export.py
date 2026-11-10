"""Export pair-worm postprocess cases from the research source.

The research ran the `pw` schedule of `new-pow/postprocess/optimizers/ice-pairs/run.py` on the
selected twisted-ice cells under a CPU-time budget, and saved each final state with the work it
did: `reads_used` reads, then `anneal_cycles` anneal-quench cycles. The schedule reads the clock
only to choose those two counts, so a replay that is given them reproduces the saved state.

Two modes. Neither writes under `$QUIP_RESEARCH_ROOT`.

`vectors OUT.json` builds two small synthetic twisted-ice models and runs the research code on
them: the tile preparation of `descent.ice_tiles`, `pair_worm_descent`, `crossover` and
`defect_counts` of `pairworm.c` through `pairs.PairWorm`, and `run.schedule_pw` with a fixed-work
clock. It needs the research Python environment (numpy, scipy, dimod, dwave-samplers) and the
compiled `libpairworm.so` and `libgreedy.so` next to the sources. The output is the committed file
`tests/fixtures/pair_worm_vectors.json`:

    cd $QUIP_RESEARCH_ROOT/new-pow/postprocess/optimizers/ice-pairs
    PYTHONDONTWRITEBYTECODE=1 QUIP_RESEARCH_ROOT=... $VENV/bin/python -I \\
        /path/to/export.py vectors /path/to/tests/fixtures/pair_worm_vectors.json

`research OUT_DIR` writes one JSON file per saved model of the selected cells: the model, the
archived reads, the seed, the recorded work, and the saved final state. It needs numpy only. Run
the Rust test on OUT_DIR:

    QUIP_RESEARCH_ROOT=... python3 -I export.py research OUT_DIR
    QUIP_PAIRWORM_PARITY=OUT_DIR cargo test --release --features experimental \\
        --test postprocess_pairs -- --ignored research_parity
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np  # pyright: ignore[reportMissingImports]  # the research environment provides numpy

# The three selected twisted-ice controls of SC-9, and the models exported from each.
CELLS = {
    "ice-twisted-w62-dimer1200": 2,
    "ice-twisted-w62-dimer1350": 2,
    "ice-twisted-w45-dimer1200": 2,
}
LISTED = ("clusters", "strong_blocks", "vertical_bonds")
SUBTREES = ("new-pow", "qpu-f17", "qpu-suite", "round2")


def spins(s: np.ndarray) -> str:
    return "".join("+" if x > 0 else "-" for x in np.asarray(s).tolist())


def rows(model: dict, key: str) -> list[list[int]]:
    """The rows of a node-list key, without the -1 padding, as the research reads them."""
    if (
        key not in model
        or model[key].ndim != 2
        or model[key].dtype.kind not in "iu"
        or not len(model[key])
    ):
        return []
    return [[int(x) for x in row if x >= 0] for row in model[key].tolist()]


def rebase(path: str | Path, root: Path) -> Path:
    """`path` under `root`: the research manifests store absolute paths from the host that wrote them."""
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part in SUBTREES:
            return root.joinpath(*parts[i:])
    raise ValueError(f"{path} is outside the research subtrees {SUBTREES}")


# ---------------------------------------------------------------- research parity


def research(root: Path, out: Path) -> int:
    new_pow = root / "new-pow"
    greedy_100 = json.loads((new_pow / "targets-greedy-100.json").read_text())
    meas = new_pow / "postprocess" / "measurements" / "optimizers-100" / "ice-pairs"
    archive_roots = [
        new_pow / "qpu-recap" / "qpu",
        root / "qpu-f17" / "qpu",
        root / "qpu-suite" / "qpu",
    ]
    out.mkdir(parents=True, exist_ok=True)
    missing = 0
    for cell, count in CELLS.items():
        models = {r["nonce"]: r for r in greedy_100[cell]}
        written = 0
        for line in (meas / cell / "results.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            nonce = rec["nonce"]
            state = meas / cell / "states" / f"{nonce}.npy"
            arch = next(
                (
                    r / cell / "qpu-80" / f"{nonce}.npz"
                    for r in archive_roots
                    if (r / cell / "qpu-80" / f"{nonce}.npz").exists()
                ),
                None,
            )
            if (
                arch is None
                or not state.exists()
                or rec["opts"] != {}
                or rec["method"] != "pw"
            ):
                continue
            with np.load(rebase(models[nonce]["model"], root), allow_pickle=False) as z:
                model = {k: z[k] for k in z.files}
            with np.load(arch, allow_pickle=False) as z:
                reads = z["spins"].astype(np.int8)
            case = {
                "cell": cell,
                "nonce": nonce,
                "h": model["h_milli"].astype(np.int64).tolist(),
                "edges": model["edges"].astype(np.int64).tolist(),
                "j": model["j_milli"].astype(np.int64).tolist(),
                "tetrahedra": rows(model, "tetrahedra"),
                "pairs": rows(model, "pairs"),
                "listed": [r for key in LISTED for r in rows(model, key)],
                "reads": [spins(s) for s in reads],
                "seed": rec["seed"],
                "budget_s": rec["budget_s"],
                "reads_used": rec["reads_used"],
                "anneal_cycles": rec["anneal_cycles"],
                "crossover_blocks": rec["crossover_blocks"],
                "pre_crossover": rec["pre_crossover"],
                "pre_anneal": rec["pre_anneal"],
                "tiles": rec["tiles"],
                "state": spins(np.load(state)),
                "energy": rec["pp_best"],
            }
            (out / f"{cell}-{nonce[:12]}.json").write_text(json.dumps(case))
            print(cell, nonce[:12], len(reads), "reads", file=sys.stderr)
            written += 1
            if written == count:
                break
        if written < count:
            print(
                f"{cell}: only {written} of {count} models have saved states",
                file=sys.stderr,
            )
            missing += count - written
    return 1 if missing else 0


# ---------------------------------------------------------------- committed vectors


def synthetic(
    rng: np.random.Generator, n_ring: int, fields: bool
) -> dict[str, np.ndarray]:
    """A ring of gauged K4 tiles that share one corner with the next tile, planted pairs, weak glass.

    Tile t holds (a_t, b_t, c_t, a_{t+1}). One extra tile puts a_0 in a third tile, so its third
    membership is dropped, and one more tile has coupling signs that no gauge explains, so it is
    dropped whole.
    """
    a = list(range(n_ring))
    b = list(range(n_ring, 2 * n_ring))
    c = list(range(2 * n_ring, 3 * n_ring))
    tiles = [[a[t], b[t], c[t], a[(t + 1) % n_ring]] for t in range(n_ring)]
    n = 3 * n_ring
    extra = [a[0], n, n + 1, n + 2]
    bad = [n + 3, n + 4, n + 5, n + 6]
    n += 7
    tiles.append(extra)
    couplings: dict[tuple[int, int], int] = {}
    for tile in tiles:
        g = rng.choice([-1, 1], size=4)
        for i in range(4):
            for k in range(i + 1, 4):
                u, v = sorted((tile[i], tile[k]))
                couplings[(u, v)] = int(500 * g[i] * g[k])
    for i in range(4):
        for k in range(i + 1, 4):
            couplings[(bad[i], bad[k])] = 500
    couplings[
        (bad[0], bad[1])
    ] = -500  # no gauge gives J_01 < 0 with the other five positive
    tiles.append(bad)
    # Disjoint pairs between b and c spins of different tiles.
    free = rng.permutation(b + c).tolist()
    pairs = []
    while len(free) >= 2 and len(pairs) < n_ring // 2 + 1:
        u, v = free.pop(), free.pop()
        key = (min(u, v), max(u, v))
        if key in couplings:
            continue
        couplings[key] = -1000
        pairs.append([u, v])
    weak = 0
    while weak < 2 * n_ring:
        u, v = sorted(rng.choice(n, size=2, replace=False).tolist())
        if (u, v) in couplings:
            continue
        couplings[(u, v)] = int(rng.choice([-62, 62]))
        weak += 1
    edges = np.array(sorted(couplings), dtype=np.int64)
    j = np.array([couplings[tuple(e)] for e in edges.tolist()], dtype=np.int64)
    order = rng.permutation(len(edges))  # the research models do not sort their edges
    h = np.zeros(n, dtype=np.int64)
    if fields:
        h[rng.choice(n, size=n // 4, replace=False)] = rng.choice(
            [-40, 40], size=n // 4
        )
    return {
        "h_milli": h,
        "edges": edges[order],
        "j_milli": j[order],
        "tetrahedra": np.array(tiles, dtype=np.int64),
        "pairs": np.array(pairs, dtype=np.int64),
    }


# Option sets for `PairWorm.descend`, by name. Every key is a `descend` keyword.
DESCENTS = {
    "strict-depth8": {"zero_budget": 0, "max_len": 32, "defect_depth": 8},
    "greedy-walk": {"zero_budget": 0, "max_len": 32},
    "anneal-sweep": {"zero_budget": 0, "max_len": 32, "max_sweeps": 1, "temp": 250.0},
    "zero-budget": {"zero_budget": 40, "max_len": 32, "defect_depth": 8},
    "short-pruned": {"zero_budget": 0, "max_len": 4, "defect_depth": 3, "prune": 0},
    "no-random-pass": {
        "zero_budget": 0,
        "max_len": 32,
        "random_pass": False,
        "defect_depth": 2,
        "random_depth": 2,
    },
    "random-depth": {
        "zero_budget": 6,
        "max_len": 12,
        "defect_depth": 1,
        "random_depth": 3,
        "temp": 400.0,
        "max_sweeps": 3,
    },
}
CROSSOVERS = {
    "aligned": {"align": True},
    "unaligned": {"align": False},
    "sized": {"align": True, "min_size": 2, "max_size": 6},
}


class FixedWork:
    """A clock for `run.schedule_pw` that grants exactly `cycles` anneal-quench cycles.

    `used` stays at zero, so the read phase visits every read it is given. The caller passes the
    lowest reads by energy, in order, which is where the research stopped.
    """

    def __init__(self, limit: float, cycles: int):
        self.limit = limit
        self.cycles = cycles

    def used(self) -> float:
        return 0.0

    def left(self) -> float:
        if self.cycles > 0:
            self.cycles -= 1
            return float("inf")
        return float("-inf")


def vectors(out: Path) -> int:
    sys.path.insert(0, os.getcwd())
    import run  # noqa: PLC0415  # pyright: ignore[reportMissingImports]  # the research tree is only known at run time

    rng = np.random.default_rng(20261110)
    cases = []
    for name, n_ring, fields in (("ring12", 12, False), ("ring9-fields", 9, True)):
        model = synthetic(rng, n_ring, fields)
        ctx = run.Context(model)
        pw = ctx.pw
        n = ctx.ising.n
        reads = rng.choice(np.array([-1, 1], dtype=np.int8), size=(6, n))
        descents = []
        for opt_name, opts in DESCENTS.items():
            for r, seed in enumerate((0, 7, 2**63 + 5)):
                s, accepted = pw.descend(reads[r : r + 1], seed=seed, **opts)
                descents.append(
                    {
                        "options": opt_name,
                        "read": r,
                        "seed": seed,
                        "state": spins(s[0]),
                        "accepted": accepted,
                    }
                )
        base, _ = pw.descend(reads[:1], 0, 3, defect_depth=8)
        crossovers = []
        for opt_name, opts in CROSSOVERS.items():
            s, blocks = pw.crossover(base[0], reads[1:], **opts)
            crossovers.append(
                {"options": opt_name, "state": spins(s), "blocks": blocks}
            )
        energies = ctx.energies(reads)
        schedules = []
        for used, cycles, seed in ((4, 3, 12345), (1, 0, 99), (6, 2, 2**40 + 1)):
            order = np.argsort(energies, kind="stable")[:used]
            s, info = run.schedule_pw(
                ctx, reads[order], FixedWork(0.76, cycles), seed, {}
            )
            schedules.append(
                {
                    "reads_used": used,
                    "anneal_cycles": cycles,
                    "seed": seed,
                    "state": spins(s),
                    "energy": int(ctx.energies(s[None])[0]),
                    "crossover_blocks": info["crossover_blocks"],
                    "crossover_rounds": info["crossover_rounds"],
                    "pre_crossover": info["pre_crossover"],
                    "pre_anneal": info["pre_anneal"],
                }
            )
        cases.append(
            {
                "name": name,
                "h": model["h_milli"].tolist(),
                "edges": model["edges"].tolist(),
                "j": model["j_milli"].tolist(),
                "tetrahedra": model["tetrahedra"].tolist(),
                "pairs": model["pairs"].tolist(),
                "tiles": pw.tiles.tolist(),
                "gauge": pw.gauge.tolist(),
                "mtile": pw.mtile.tolist(),
                "counts": pw.counts,
                "reads": [spins(s) for s in reads],
                "energies": energies.tolist(),
                "defects": pw.defects(reads).tolist(),
                "descents": descents,
                "crossover_base": spins(base[0]),
                "crossovers": crossovers,
                "schedules": schedules,
            }
        )
    doc = {
        "source": "new-pow/postprocess/optimizers/ice-pairs: pairworm.c, pairs.py, run.py schedule_pw",
        "generator": "research/pairworm_parity/export.py vectors",
        "descent_options": DESCENTS,
        "crossover_options": CROSSOVERS,
        "cases": cases,
    }
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(out, len(cases), "cases", file=sys.stderr)
    return 0


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in ("vectors", "research"):
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(sys.argv[2])
    if sys.argv[1] == "vectors":
        return vectors(out)
    return research(Path(os.environ["QUIP_RESEARCH_ROOT"]), out)


if __name__ == "__main__":
    raise SystemExit(main())
