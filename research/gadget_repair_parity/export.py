"""Export gadget-repair postprocess cases from the research source.

The research ran `run.py` of `new-pow/postprocess/optimizers/constraints` on the two-commodity
routing cell under a CPU-time budget of 0.05 s, and saved each final state with its seed and its
per-start statistics. The repair of `repair.c` reads the thread CPU clock only at fixed places, so
a replay that grants the same number of clock reads (ticks) reproduces a run exactly.

Two modes. Neither writes under `$QUIP_RESEARCH_ROOT`.

`vectors OUT.json` builds small synthetic routing models and runs the research code on them:
`gadgets.extract`, `run.neighbour_csr`, `repair_run` of a copy of `repair.c` whose clock is a tick
counter (the only change), and the start schedule of `run.one_model` with recorded work counts.
It needs the research Python environment and gcc. Run it from the research directory:

    cd $QUIP_RESEARCH_ROOT/new-pow/postprocess/optimizers/constraints
    PYTHONDONTWRITEBYTECODE=1 $VENV/bin/python -I -B \\
        /path/to/export.py vectors /path/to/tests/fixtures/gadget_repair_vectors.json

`research OUT_DIR` writes one JSON file per saved model of the routing cell: the model, the
archived reads, the seed, the recorded statistics, and the saved final state. It needs numpy only:

    QUIP_RESEARCH_ROOT=... python3 -I export.py research OUT_DIR
    QUIP_GADGET_PARITY=OUT_DIR cargo test --release --features experimental \\
        --test postprocess_routing -- --ignored research_parity
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np  # pyright: ignore[reportMissingImports]  # the research environment provides numpy

CELL = "two-commodity-routing"
SUBTREES = ("new-pow", "qpu-f17", "qpu-suite", "round2")
SEED_BASE = 20261022
NODE_DEFAULTS = {"reads_used": 4, "depth": 8, "tenure": 12, "stall": 2000}


def spins(s: np.ndarray) -> str:
    return "".join("+" if x > 0 else "-" for x in np.asarray(s).tolist())


def rebase(path: str | Path, root: Path) -> Path:
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part in SUBTREES:
            return root.joinpath(*parts[i:])
    raise ValueError(f"{path} is outside the research subtrees {SUBTREES}")


# ---------------------------------------------------------------- research parity


def research(root: Path, out: Path, count: int = 6) -> int:
    new_pow = root / "new-pow"
    targets = json.loads((new_pow / "targets-pp.json").read_text())[CELL]
    models = {r["nonce"]: r for r in targets["rows"]}
    meas = new_pow / "postprocess" / "measurements" / "optimizers-100" / "constraints" / CELL
    archive = new_pow / "qpu-recap" / "qpu" / CELL / "qpu-80"
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for line in (meas / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        nonce = rec["nonce"]
        state = meas / "states" / f"{nonce}.npy"
        arch = archive / f"{nonce}.npz"
        if not (state.exists() and arch.exists() and nonce in models):
            continue
        if any(rec[k] != v for k, v in NODE_DEFAULTS.items() if k != "reads_used"):
            continue
        with np.load(rebase(models[nonce]["model"], root), allow_pickle=False) as z:
            model = {k: z[k] for k in z.files}
        with np.load(arch, allow_pickle=False) as z:
            reads = z["spins"].astype(np.int8)
            raw = z["energies"].astype(np.int64)
        case = {
            "nonce": nonce,
            "h": model["h_milli"].astype(np.int64).tolist(),
            "edges": model["edges"].astype(np.int64).tolist(),
            "j": model["j_milli"].astype(np.int64).tolist(),
            "arc_nodes": model["arc_nodes"].astype(np.int64).tolist(),
            "x1": model["arc_spin_commodity1"].astype(np.int64).tolist(),
            "chain": model["arc_chain_commodity2"].astype(np.int64).tolist(),
            "reads": [spins(s) for s in reads],
            "raw_energies": raw.tolist(),
            "seed": rec["seed"],
            "budget_s": rec["budget_s"],
            "reads_descended": rec["reads_descended"],
            "vote": rec["vote"],
            "structure": rec["structure"],
            "read_stats": rec["read_stats"],
            "state": spins(np.load(state)),
            "energy": rec["pp_best"],
        }
        (out / f"{CELL}-{nonce[:12]}.json").write_text(json.dumps(case))
        print(CELL, nonce[:12], len(reads), "reads", file=sys.stderr)
        written += 1
        if written == count:
            break
    return 0 if written else 1


# ---------------------------------------------------------------- committed vectors


def synthetic(rng: np.random.Generator, width: int, height: int, keep: float) -> dict[str, np.ndarray]:
    """A Manhattan-street grid of nodes, with the spins of the routing cell in a shuffled labelling.

    Arc a owns three spins: the commodity 1 spin x1, then the two chain spins. Every node and
    commodity has a conservation clique over its arc spins, every arc a capacity coupler between
    x1 and the first chain spin, and a strong ferromagnetic coupler inside each chain.
    """
    node = lambda c, r: r * width + c  # noqa: E731
    arcs = []
    for r in range(height):
        for c in range(width):
            if c + 1 < width:
                t, h = (node(c, r), node(c + 1, r)) if r % 2 == 0 else (node(c + 1, r), node(c, r))
                arcs.append((t, h))
            if r + 1 < height:
                t, h = (node(c, r), node(c, r + 1)) if c % 2 == 0 else (node(c, r + 1), node(c, r))
                arcs.append((t, h))
    arcs = [a for a in arcs if rng.random() < keep]
    n = 3 * len(arcs)
    label = rng.permutation(n)
    x1 = np.array([label[3 * a] for a in range(len(arcs))], dtype=np.int64)
    chain = np.array([[label[3 * a + 1], label[3 * a + 2]] for a in range(len(arcs))], dtype=np.int64)
    arc_nodes = np.array(arcs, dtype=np.int64)
    couplings: dict[tuple[int, int], int] = {}

    def add(u: int, v: int, value: int) -> None:
        key = (min(u, v), max(u, v))
        couplings[key] = couplings.get(key, 0) + value

    incident: dict[int, list[int]] = {}
    for a, (t, h) in enumerate(arcs):
        incident.setdefault(t, []).append(a)
        incident.setdefault(h, []).append(a)
    for v, ids in incident.items():
        for k in range(2):
            members = []
            for a in ids:
                t, h = arcs[a]
                sign = 1 if t == v else -1
                s = int(x1[a]) if k == 0 else int(chain[a][0] if v == min(t, h) else chain[a][1])
                members.append((s, sign))
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    add(members[i][0], members[j][0], 62 * members[i][1] * members[j][1])
    for a in range(len(arcs)):
        add(int(x1[a]), int(chain[a][0]), 124)
    h_milli = rng.choice([-93, -31, 0, 31, 93], size=n).astype(np.int64)
    local = np.zeros(n, dtype=np.int64)
    for (u, v), value in couplings.items():
        local[u] += abs(value)
        local[v] += abs(value)
    strength = int(max(local[s] + abs(h_milli[s]) for a in range(len(arcs)) for s in chain[a])) + 1
    for a in range(len(arcs)):
        add(int(chain[a][0]), int(chain[a][1]), -strength)
    edges = np.array(sorted(couplings), dtype=np.int64)
    j = np.array([couplings[tuple(e)] for e in edges.tolist()], dtype=np.int64)
    order = rng.permutation(len(edges))
    return {
        "h_milli": h_milli,
        "edges": edges[order],
        "j_milli": j[order],
        "arc_nodes": arc_nodes,
        "arc_spin_commodity1": x1,
        "arc_chain_commodity2": chain,
    }


def ticked_library(directory: Path) -> ctypes.CDLL:
    """`repair.c` with its CPU clock replaced by a tick counter, built under `directory`.

    The k-th clock read (k = 0 is the start time) returns 0 while k <= limit and a huge time after,
    so a run with limit L passes exactly L deadline checks. `repair_set_limit` resets the counter
    and sets L, and `repair_ticks` reports the reads so far. Nothing else changes.
    """
    source = Path.cwd() / "repair.c"
    text = source.read_text()
    clock = re.compile(r"static double cpu_now\(void\)\n\{.*?\n\}\n", re.S)
    assert clock.search(text), "repair.c has no cpu_now"
    ticked = (
        "static int64_t g_ticks, g_limit;\n"
        "void repair_set_limit(int64_t limit) { g_ticks = -1; g_limit = limit; }\n"
        "int64_t repair_ticks(void) { return g_ticks; }\n"
        "static double cpu_now(void)\n{\n    g_ticks++;\n    return g_ticks > g_limit ? 1.0e9 : 0.0;\n}\n"
    )
    patched = clock.sub(lambda _m: ticked, text, count=1)
    (directory / "repair_ticked.c").write_text(patched)
    lib = directory / "librepair_ticked.so"
    subprocess.run(
        ["gcc", "-O2", "-shared", "-fPIC", "-Wall", "-Wextra", "-o", str(lib), str(directory / "repair_ticked.c")],
        check=True,
    )
    cdll = ctypes.CDLL(str(lib))
    cdll.repair_init.restype = ctypes.c_void_p
    cdll.repair_free.argtypes = [ctypes.c_void_p]
    cdll.repair_run.restype = ctypes.c_int64
    cdll.repair_ticks.restype = ctypes.c_int64
    cdll.repair_set_limit.argtypes = [ctypes.c_int64]
    return cdll


def ticked_run(run, rep, lib, s0, ticks, n_starts, depth, tenure, tenure_rand, stall, seed):
    """`Repair.run` with `ticks` deadline checks. Returns (state, delta, stats, ticks used)."""
    lib.repair_set_limit(ticks)
    s, d, stats = rep.run(s0, 1.0, n_starts, depth, tenure, tenure_rand, stall, seed)
    return s, d, stats, int(lib.repair_ticks())


def schedule_fixed(D, rep, lib, ising, gnat, reads, descended, ticks, seed, depth, tenure, stall, chain_starts, vote):
    """The start schedule of `run.one_model` with recorded work: `descended` reads, `ticks[k]` ticks."""
    reads_used = NODE_DEFAULTS["reads_used"]
    raw_e = ising.energies(reads)
    s_nat = reads.copy()
    raw_order = np.argsort(raw_e, kind="stable")
    for r in raw_order[:descended]:
        s_nat[r : r + 1], _ = gnat.descend(reads[r : r + 1])
    e_nat = ising.energies(s_nat)
    order = np.argsort(e_nat, kind="stable")[:reads_used]
    best_e = int(e_nat.min())
    best_s = s_nat[int(e_nat.argmin())].copy()
    starts = [(int(r), s_nat[r], int(e_nat[r])) for r in order]
    if vote and descended >= 2:
        top = np.argsort(e_nat, kind="stable")[: min(8, descended)]
        v = np.sign(s_nat[top].astype(np.int64).sum(axis=0))
        v[v == 0] = s_nat[top[0]][v == 0]
        v_nat, _ = gnat.descend(v.astype(np.int8)[None, :])
        starts.insert(1, (-1, v_nat[0], int(ising.energies(v_nat)[0])))
    rows = []
    for k, (r, s0, e0) in enumerate(starts):
        if k >= len(ticks):
            break
        n_starts = rep.struct.n_starts if chain_starts == "singles" else len(rep.units)
        s, d, stats, used = ticked_run(
            None, rep, lib, s0, ticks[k], n_starts, depth, tenure, max(1, tenure // 2), stall, seed + 1000 * k
        )
        e = e0 + d
        rows.append(
            {"read": r, "e0": e0, "e": e, "stats": stats.tolist(), "ticks_used": used, "state": spins(s)}
        )
        if e < best_e:
            best_e, best_s = e, s
    return {
        "descended": descended,
        "ticks": list(ticks),
        "seed": seed,
        "chain_starts": chain_starts,
        "vote": vote,
        "start_energies": [e for _, _, e in starts],
        "nat_energies": e_nat.tolist(),
        "starts": rows,
        "state": spins(best_s),
        "energy": best_e,
    }


def vectors(out: Path) -> int:
    sys.path.insert(0, os.getcwd())
    import descent as D  # noqa: PLC0415  # pyright: ignore[reportMissingImports]  # the research tree is only known at run time
    import gadgets  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    import run  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    rng = np.random.default_rng(20261110)
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        lib = ticked_library(Path(tmp))
        run._LIB = lib  # noqa: SLF001  # `Repair` reads the library from the module global
        for name, width, height, keep in (("grid3x3", 3, 3, 0.9), ("grid4x3", 4, 3, 0.85)):
            model = synthetic(rng, width, height, keep)
            ising = D.Ising(model)
            n = ising.n
            struct = gadgets.extract(CELL, model)
            rep = run.Repair(CELL, ising)
            nat_moves, _ = D.natural_moves(ising)
            gnat = D.GreedyC(ising, D.MoveSet(ising, nat_moves))
            reads = rng.choice(np.array([-1, 1], dtype=np.int8), size=(10, n))
            gm_ptr, gm_idx = rep.gm_ptr, rep.gm_idx
            repairs = []
            plans = [
                # ticks, chain starts, depth, tenure, stall
                (0, "all", 8, 12, 2000),
                (3, "all", 8, 12, 2000),
                (12, "all", 8, 12, 2000),
                (60, "singles", 8, 12, 2000),
                (200, "all", 8, 12, 2000),
                (400, "all", 8, 12, 30),
                (300, "all", 1, 4, 25),
                (300, "singles", 3, 6, 40),
            ]
            for r in range(3):
                for ticks, starts_mode, depth, tenure, stall in plans:
                    seed = (7, 12345, 2**63 + 9)[r]
                    n_starts = struct.n_starts if starts_mode == "singles" else len(rep.units)
                    s, d, stats, used = ticked_run(
                        None, rep, lib, reads[r], ticks, n_starts, depth, tenure, max(1, tenure // 2), stall, seed
                    )
                    repairs.append(
                        {
                            "read": r,
                            "ticks": ticks,
                            "chain_starts": starts_mode,
                            "depth": depth,
                            "tenure": tenure,
                            "stall": stall,
                            "seed": seed,
                            "delta": d,
                            "stats": stats.tolist(),
                            "ticks_used": used,
                            "state": spins(s),
                        }
                    )
            schedules = []
            for descended, ticks, seed, starts_mode, vote in (
                (10, [40, 30, 20, 20, 20], 20261022, "all", True),
                (1, [25, 25], 99, "all", True),
                (6, [100, 10, 5, 5, 5], 2**40 + 1, "all", True),
            ):
                schedules.append(
                    schedule_fixed(D, rep, lib, ising, gnat, reads, descended, ticks, seed, 8, 12, 2000, starts_mode, vote)
                )
            cases.append(
                {
                    "name": name,
                    "h": model["h_milli"].tolist(),
                    "edges": model["edges"].tolist(),
                    "j": model["j_milli"].tolist(),
                    "arc_nodes": model["arc_nodes"].tolist(),
                    "x1": model["arc_spin_commodity1"].tolist(),
                    "chain": model["arc_chain_commodity2"].tolist(),
                    "structure": {
                        "units": [list(u) for u in struct.units],
                        "gadgets": [list(g) for g in struct.gadgets],
                        "n_starts": struct.n_starts,
                        "kinds": struct.kinds,
                        "gm_ptr": gm_ptr.tolist(),
                        "gm_idx": gm_idx.tolist(),
                    },
                    "reads": [spins(s) for s in reads],
                    "repairs": repairs,
                    "schedules": schedules,
                }
            )
            rep.close()
    doc = {
        "source": "new-pow/postprocess/optimizers/constraints: gadgets.py, run.py, repair.c (tick clock)",
        "generator": "research/gadget_repair_parity/export.py vectors",
        "cases": cases,
    }
    out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
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
