"""Export two-commodity routing cases from the research solver.

The research solver is `routesolve` (`new-pow/two-commodity-routing/solver/src/main.rs`). It reads
no clock to choose its result: the two start orders, at most 50 alternations each, and the final
single-flip descent are deterministic. So a saved state is reproduced bit for bit.

Two modes. Neither writes under `$QUIP_RESEARCH_ROOT`.

`vectors OUT.json` builds small synthetic routing models, with spins that no arc uses, runs the
research binary on them, and writes the model, both final states, and their energies. The energies
are scored again here with numpy, independent of the solver. Build the binary into a scratch
directory first:

    cd $QUIP_RESEARCH_ROOT/new-pow/two-commodity-routing/solver
    CARGO_TARGET_DIR=/scratch/routesolve-target cargo build --release --offline
    QUIP_ROUTESOLVE=/scratch/routesolve-target/release/routesolve \\
        $VENV/bin/python -I -B /path/to/export.py vectors tests/fixtures/routing_vectors.json

`research OUT_DIR` writes one JSON file per saved model of the `t1-flow` campaign: the model and
the two saved final states. It needs numpy only. Run the Rust test on OUT_DIR:

    QUIP_RESEARCH_ROOT=... python -I -B export.py research OUT_DIR
    QUIP_ROUTING_PARITY=OUT_DIR cargo test --release --features experimental \\
        --test routing_solver -- --ignored research_parity
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np  # pyright: ignore[reportMissingImports]  # the research environment provides numpy

CONSERVE = 248
CAPACITY = 496
COSTS = (62, 124)


def ising_energy(model: dict, spins: np.ndarray) -> int:
    """The Ising energy in milli, scored again from the arrays with numpy."""
    s = spins.astype(np.int64)
    edges = np.asarray(model["edges"], dtype=np.int64).reshape(-1, 2)
    j = np.asarray(model["j"], dtype=np.int64)
    h = np.asarray(model["h"], dtype=np.int64)
    return int((j * s[edges[:, 0]] * s[edges[:, 1]]).sum() + (h * s).sum())


def street_model(width: int, height: int, spare: int, seed: int, pairs: int) -> dict:
    """A street-grid routing model in the layout of `generate.py: build`.

    Horizontal arcs point right in even rows and left in odd rows. Vertical arcs point up in even
    columns and down in odd columns. Commodity 1 has one spin per arc. Commodity 2 has a chain of
    two spins. `spare` spins belong to no arc and carry no coupling and no field. They sit between
    the arc spins, so the used spins are not a prefix.
    """
    rng = random.Random(seed)
    cells = [(c, r) for c in range(width) for r in range(height)]
    index = {cell: i for i, cell in enumerate(cells)}
    arcs = []
    for c, r in cells:
        if c + 1 < width:
            a, b = index[(c, r)], index[(c + 1, r)]
            arcs.append((a, b) if r % 2 == 0 else (b, a))
        if r + 1 < height:
            a, b = index[(c, r)], index[(c, r + 1)]
            arcs.append((a, b) if c % 2 == 0 else (b, a))
    # Spin positions: every arc takes three, and a spare spin follows every few arcs.
    position = 0
    spare_at = {k * len(arcs) // max(spare, 1) for k in range(spare)} if spare else set()
    spin1, chain2 = [], []
    spares = []
    for k in range(len(arcs)):
        if k in spare_at:
            spares.append(position)
            position += 1
        spin1.append(position)
        chain2.append([position + 1, position + 2])
        position += 3
    n = position
    supply = [[0, 0] for _ in cells]
    order = list(range(len(cells)))
    rng.shuffle(order)
    pick = iter(order)
    for k in range(2):
        for _ in range(pairs):
            supply[next(pick)][k] = 1
            supply[next(pick)][k] = -1
    cost = [[rng.choice(COSTS), rng.choice(COSTS)] for _ in arcs]

    # The energy in x = (1 - s) / 2 form: linear terms c x and pair terms c x x, in milli.
    lin: dict[int, int] = defaultdict(int)
    pair: dict[tuple[int, int], int] = defaultdict(int)
    out_arcs = [[] for _ in cells]
    in_arcs = [[] for _ in cells]
    for a, (t, h) in enumerate(arcs):
        out_arcs[t].append(a)
        in_arcs[h].append(a)

    def spin(a: int, k: int, node: int) -> int:
        if k == 0:
            return spin1[a]
        lo = min(arcs[a])
        return chain2[a][0] if node == lo else chain2[a][1]

    for v in range(len(cells)):
        for k in range(2):
            coeffs = {}
            for a in out_arcs[v]:
                coeffs[spin(a, k, v)] = 1
            for a in in_arcs[v]:
                coeffs[spin(a, k, v)] = -1
            if not coeffs:
                continue
            b = supply[v][k]
            items = list(coeffs.items())
            for i, ci in items:
                lin[i] += CONSERVE * (ci * ci - 2 * b * ci)
            for x in range(len(items)):
                for y in range(x + 1, len(items)):
                    (i, ci), (j, cj) = items[x], items[y]
                    pair[(min(i, j), max(i, j))] += 2 * CONSERVE * ci * cj
    for a in range(len(arcs)):
        pair[(spin1[a], chain2[a][0])] += CAPACITY
        lin[spin1[a]] += cost[a][0]
        lin[chain2[a][0]] += cost[a][1]
    h = [0] * n
    j: dict[tuple[int, int], int] = defaultdict(int)
    for i, c in lin.items():
        assert c % 2 == 0, "odd linear term"
        h[i] -= c // 2
    for (i, k), c in pair.items():
        assert c % 4 == 0, "odd pair term"
        j[(i, k)] += c // 4
        h[i] -= c // 4
        h[k] -= c // 4
    local = defaultdict(int)
    for (i, k), c in j.items():
        local[i] += abs(c)
        local[k] += abs(c)
    strength = max(local[s] + abs(h[s]) for chain in chain2 for s in chain) + 1
    for chain in chain2:
        j[(chain[0], chain[1])] -= strength
    edges = sorted(j)
    return {
        "nodes": len(cells),
        "spins": n,
        "spare": spares,
        "edges": [list(e) for e in edges],
        "j": [j[e] for e in edges],
        "h": h,
        "supply": supply,
        "arcs": [list(a) for a in arcs],
        "spin1": spin1,
        "chain2": chain2,
        "cost": cost,
    }


def write_npz(path: Path, model: dict) -> None:
    np.savez(
        path,
        edges=np.asarray(model["edges"], dtype=np.int64).reshape(-1, 2),
        j_milli=np.asarray(model["j"], dtype=np.int64),
        h_milli=np.asarray(model["h"], dtype=np.int64),
        node_supply=np.asarray(model["supply"], dtype=np.int64),
        arc_nodes=np.asarray(model["arcs"], dtype=np.int64),
        arc_spin_commodity1=np.asarray(model["spin1"], dtype=np.int64),
        arc_chain_commodity2=np.asarray(model["chain2"], dtype=np.int64),
        arc_cost_milli=np.asarray(model["cost"], dtype=np.int64),
    )


def run_research(binary: Path, model: dict, scratch: Path) -> tuple[list[list[int]], list[int]]:
    """Run `routesolve` on the model and read both candidates, in start-order 0 and 1."""
    npz = scratch / "model.npz"
    write_npz(npz, model)
    best, states = scratch / "best.npy", scratch / "states.npy"
    done = subprocess.run(
        [str(binary), "--model", str(npz), "--out", str(best), "--states", str(states)],
        check=True,
        capture_output=True,
        text=True,
    )
    energies = json.loads(done.stdout)["energies"]
    rows = np.load(states)  # the solver writes a plain .npy matrix of rows
    return [[int(x) for x in row] for row in rows], [int(e) for e in energies]


def vectors(out: Path) -> int:
    binary = Path(os.environ["QUIP_ROUTESOLVE"])
    # width, height, spare spins, seed, terminal pairs per commodity
    shapes = [(3, 3, 4, 1, 1), (4, 3, 5, 2, 2), (4, 4, 0, 3, 2), (5, 4, 7, 4, 3), (6, 5, 9, 5, 4), (6, 6, 3, 6, 5)]
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        for width, height, spare, seed, pairs in shapes:
            model = street_model(width, height, spare, seed, pairs)
            states, energies = run_research(binary, model, Path(tmp))
            for row, energy in zip(states, energies):
                assert len(row) == model["spins"]
                assert ising_energy(model, np.asarray(row)) == energy, "numpy disagrees with the solver"
            model["states"] = states
            model["energies"] = energies
            model["shape"] = [width, height]
            cases.append(model)
    out.write_text(json.dumps({"source": "routesolve", "cases": cases}, separators=(",", ":")) + "\n")
    print(f"{len(cases)} cases, {out.stat().st_size} bytes")
    return 0


def research(root: Path, out: Path) -> int:
    base = root / "new-pow"
    models = base / "families" / "two-commodity-routing" / "models"
    saved = base / "two-commodity-routing" / "measurements" / "t1-flow"
    out.mkdir(parents=True, exist_ok=False)
    count = 0
    for states_file in sorted(saved.glob("*.states.npz")):
        nonce = states_file.name.split(".")[0]
        z = np.load(models / f"{nonce}.npz")
        s = np.load(states_file)
        case = {
            "nonce": nonce,
            "spins": int(len(z["h_milli"])),
            "edges": z["edges"].tolist(),
            "j": z["j_milli"].tolist(),
            "h": z["h_milli"].tolist(),
            "supply": z["node_supply"].tolist(),
            "arcs": z["arc_nodes"].tolist(),
            "spin1": z["arc_spin_commodity1"].tolist(),
            "chain2": z["arc_chain_commodity2"].tolist(),
            "cost": z["arc_cost_milli"].tolist(),
            "states": s["spins"].tolist(),
            "energies": [int(e) for e in s["energies"]],
        }
        for row, energy in zip(case["states"], case["energies"]):
            assert ising_energy(case, np.asarray(row)) == energy
        (out / f"{nonce}.json").write_text(json.dumps(case, separators=(",", ":")))
        count += 1
    print(f"{count} saved models written to {out}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "vectors":
        return vectors(Path(argv[2]))
    if len(argv) == 3 and argv[1] == "research":
        return research(Path(os.environ["QUIP_RESEARCH_ROOT"]), Path(argv[2]))
    sys.stderr.write(__doc__ or "")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
