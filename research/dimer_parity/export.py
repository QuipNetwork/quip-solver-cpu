"""Export weighted-dimer cases from the research Rust code.

The research fork `new-pow/ice-twisted-w62-dimer1200/cpu` holds the weighted-dimer kernel in
`src/dimer_weighted.rs` and `src/dimer_weighted/ice.rs`, and its `dimer_bench` example runs a
variant on a prepared job file. This script drives that example. The weighted kernel reads no
clock: its work is the sweep count, so a replay with the same sweeps reproduces the states.

Build the example into a scratch directory, never into the research tree:

    cp -r $QUIP_RESEARCH_ROOT/new-pow/ice-twisted-w62-dimer1200/cpu/{Cargo.toml,Cargo.lock,src,examples} SCRATCH
    cd SCRATCH && cargo build --release --offline --example dimer_bench --target-dir TARGET

Two modes. Neither writes under `$QUIP_RESEARCH_ROOT`.

`vectors OUT.json BENCH` builds three small synthetic twisted-ice models and runs the research
`ice-loop-weighted`, `ice-loop64-weighted`, and `ice-defect64-weighted` variants on them with BENCH,
the path of the built `dimer_bench`. It writes the committed file
`tests/fixtures/dimer_weighted_vectors.json`. Run it with plain Python 3:

    python3 -I research/dimer_parity/export.py vectors tests/fixtures/dimer_weighted_vectors.json BENCH

`research OUT_DIR` copies one saved research run of `ice-loop64-weighted` (the model, the tiles,
the seed, the sweeps, and the saved states) into OUT_DIR, for the ignored Rust `research_parity`
test:

    QUIP_RESEARCH_ROOT=... python3 -I research/dimer_parity/export.py research OUT_DIR
    QUIP_DIMER_PARITY=OUT_DIR cargo test --release --features experimental \\
        --test dimer_family -- --ignored research_parity
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

VARIANTS = ("ice-loop-weighted", "ice-loop64-weighted", "ice-defect64-weighted")
# The research example fixes four sweeps per beta rung.
SWEEPS_PER_BETA = 4


def model(seed: int, tiles: int, pairs: int, weak: float) -> dict:
    """A small twisted-ice model: a ring of tiles with local gauges, planted pairs, and weak edges."""
    rng = random.Random(seed)
    shared = list(range(tiles))
    inner = tiles
    tile_nodes = []
    for i in range(tiles):
        tile_nodes.append([shared[i], inner, inner + 1, shared[(i + 1) % tiles]])
        inner += 2
    pair_nodes = [(inner + 2 * k, inner + 2 * k + 1) for k in range(pairs)]
    nodes = inner + 2 * pairs
    edges: dict[tuple[int, int], float] = {}
    for tile in tile_nodes:
        gauge = [rng.choice((1, -1)) for _ in tile]
        for a in range(4):
            for b in range(a + 1, 4):
                u, v = tile[a], tile[b]
                edges[(min(u, v), max(u, v))] = float(gauge[a] * gauge[b])
    for u, v in pair_nodes:
        edges[(u, v)] = -1.0
    target = nodes * 2
    while len(edges) < len(tile_nodes) * 6 + pairs + target:
        u, v = rng.sample(range(nodes), 2)
        key = (min(u, v), max(u, v))
        if key not in edges:
            edges[key] = weak if rng.random() < 0.5 else -weak
    keys = sorted(edges)
    return {
        "h": [0.0] * nodes,
        "edges": [list(k) for k in keys],
        "j": [edges[k] for k in keys],
        "tiles": tile_nodes,
        "pairs": [list(p) for p in pair_nodes],
    }


def decode(hex_state: str, nodes: int) -> str:
    raw = bytes.fromhex(hex_state)
    return "".join("-" if (raw[n // 8] >> (n % 8)) & 1 else "+" for n in range(nodes))


def vectors(out: Path, bench: Path) -> None:
    cases = []
    specs = [
        ("ring6", 6, 4, 0.062, 11, 8, 70, (0.5, 8.0)),
        ("ring9", 9, 6, 0.045, 23, 12, 64, (0.1, 16.0)),
        ("ring5-short", 5, 3, 0.062, 5, 4, 3, (1.0, 4.0)),
    ]
    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        jobs = scratch_dir / "jobs.jsonl"
        for name, tiles, pairs, weak, seed, sweeps, reads, beta in specs:
            m = model(seed, tiles, pairs, weak)
            row = {
                "id": name,
                "nonce": "00",
                "model_hash": "00",
                "h": m["h"],
                "edges": m["edges"],
                "j": m["j"],
                "beta": list(beta),
                "seed": seed * 1_000_003 + 7,
                "clusters": m["tiles"],
            }
            jobs.write_text(json.dumps(row) + "\n")
            for variant in VARIANTS:
                result = scratch_dir / f"{name}-{variant}.jsonl"
                subprocess.run(
                    [
                        str(bench),
                        "--jobs", str(jobs),
                        "--output", str(result),
                        "--variant", variant,
                        "--sweeps", str(sweeps),
                        "--reads", str(reads),
                        "--workers", "1",
                    ],
                    check=True,
                    capture_output=True,
                )
                saved = json.loads(result.read_text().splitlines()[0])
                nodes = len(m["h"])
                cases.append(
                    {
                        "name": f"{name}/{variant}",
                        "variant": variant,
                        **{k: m[k] for k in ("h", "edges", "j", "tiles", "pairs")},
                        "beta": list(beta),
                        "seed": row["seed"],
                        "sweeps": sweeps,
                        "sweeps_per_beta": SWEEPS_PER_BETA,
                        "reads": reads,
                        "spins": [decode(s, nodes) for s in saved["spins_hex"]],
                        "stats": saved["ice_loop_stats"],
                    }
                )
    out.write_text(json.dumps({"cases": cases}, separators=(",", ":")) + "\n")


def research(out_dir: Path) -> None:
    root = Path(os.environ["QUIP_RESEARCH_ROOT"])
    cell = root / "qpu-suite/work/cells/ice-glass-r5"
    saved_dir = cell / "cpu/n12-ice-loop64-weighted-s8192"
    jobs = {}
    for line in (cell / "jobs/n12-cpu.jsonl").read_text().splitlines():
        row = json.loads(line)
        jobs[row["nonce"]] = row
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(saved_dir.glob("*.jsonl"))[:1]:
        saved = json.loads(path.read_text().splitlines()[0])
        job = jobs[saved["nonce"]]
        nodes = len(job["h"])
        case = {
            "name": saved["id"],
            "h": job["h"],
            "edges": job["edges"],
            "j": job["j"],
            "tiles": job["clusters"],
            "pairs": [],
            "beta": saved["beta"],
            "seed": saved["seed"],
            "sweeps": saved["sweeps"],
            "sweeps_per_beta": SWEEPS_PER_BETA,
            "reads": saved["reads"],
            "spins": [decode(s, nodes) for s in saved["spins_hex"]],
            "stats": saved["ice_loop_stats"],
        }
        (out_dir / f"{saved['id']}.json").write_text(json.dumps(case, separators=(",", ":")))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "vectors":
        vectors(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) == 3 and sys.argv[1] == "research":
        research(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)
