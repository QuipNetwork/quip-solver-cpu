"""Export tile-worm postprocess cases from the research source.

The research ran the ice-tiles postprocess of `new-pow/postprocess/optimizers/ice-tiles/run.py`
on the selected charged-ice and field-ice cells under a CPU-time budget, and saved each final
state with the work it did. `icetiles.c` reads the thread clock once every 64 worms, to stop and
to set the inverse temperature from the elapsed fraction of the budget. A replay that replaces the
clock with a count of those blocks reproduces a run.

Two modes. Neither writes under `$QUIP_RESEARCH_ROOT`.

`vectors OUT.json` builds three small synthetic models and runs the research code on them: the
tile tables of `icetiles.TileTables`, the seed sets, the schedule ends of `method.Prepared`, and
`method.run` for several option strings. The thread clock of a private copy of `icetiles.c`
(compiled under a temporary directory) returns the block index, and `method.time` returns a fixed
value, so each run does a known number of blocks. It needs the research Python environment (numpy,
scipy, dimod, dwave-samplers), `gcc`, and the compiled `libgreedy.so` next to the sources. The
output is the committed file `tests/fixtures/tile_worm_vectors.json`:

    cd $QUIP_RESEARCH_ROOT/new-pow/postprocess/optimizers/ice-tiles
    PYTHONDONTWRITEBYTECODE=1 $VENV/bin/python -I \\
        /path/to/export.py vectors /path/to/tests/fixtures/tile_worm_vectors.json

`research OUT_DIR` writes one JSON file per saved model of the selected cells: the model, the
archived reads, the seed, the recorded work, and the saved final state. It needs numpy only. Run
the Rust test on OUT_DIR:

    QUIP_RESEARCH_ROOT=... python3 -I export.py research OUT_DIR
    QUIP_TILEWORM_PARITY=OUT_DIR cargo test --release --features experimental \\
        --test postprocess_tiles -- --ignored research_parity
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np  # pyright: ignore[reportMissingImports]  # the research environment provides numpy

# The selected cells of the ticket, and the models exported from each.
CELLS = {
    "ice-charge": 4,
    "ice-fields-h500": 3,
    "ice-fields-h1000": 3,
}
SUBTREES = ("new-pow", "qpu-f17", "qpu-suite", "round2")
BLOCK = 64


def spins(s: np.ndarray) -> str:
    return "".join("+" if x > 0 else "-" for x in np.asarray(s).tolist())


def bits(x: float) -> int:
    """The IEEE bits of `x`, because a JSON reader may round a decimal float by one unit."""
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def rows(model: dict, key: str) -> list[list[int]]:
    if key not in model or model[key].ndim != 2 or not len(model[key]):
        return []
    return [[int(x) for x in row] for row in model[key].tolist()]


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
    targets = json.loads((new_pow / "targets-pp.json").read_text())
    meas = new_pow / "postprocess" / "measurements" / "optimizers-100" / "ice-tiles"
    archive_roots = [
        new_pow / "qpu-recap" / "qpu",
        root / "qpu-f17" / "qpu",
        root / "qpu-suite" / "qpu",
    ]
    out.mkdir(parents=True, exist_ok=True)
    missing = 0
    for cell, count in CELLS.items():
        models = {r["nonce"]: r for r in targets[cell]["rows"]}
        method = targets[cell]["method"]
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
                or rec["method"] != method
                or nonce not in models
            ):
                continue
            with np.load(rebase(models[nonce]["model"], root), allow_pickle=False) as z:
                model = {k: z[k] for k in z.files}
            with np.load(arch, allow_pickle=False) as z:
                reads = z["spins"].astype(np.int8)
            case = {
                "cell": cell,
                "nonce": nonce,
                "method": method,
                "h": model["h_milli"].astype(np.int64).tolist(),
                "edges": model["edges"].astype(np.int64).tolist(),
                "j": model["j_milli"].astype(np.int64).tolist(),
                "tetrahedra": rows(model, "tetrahedra"),
                "reads": [spins(s) for s in reads],
                "seed": rec["seed"],
                "budget_s": rec["budget_s"],
                "reads_used": rec["reads_used"],
                "worms": rec["stats"]["worms"],
                "tail_improved": rec["stats"].get("tail_improved"),
                "polish": rec["stats"].get("polish"),
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
    rng: np.random.Generator, n_ring: int, scale: int, fields: int, extras: bool
) -> dict[str, np.ndarray]:
    """A ring of gauged K4 tiles that share one corner with the next, weak glass, and six free spins.

    Tile t holds (a_t, b_t, c_t, a_{t+1}). With `extras`, five more tiles follow: a tile with a
    reversed bond (two ground states that differ by a full flip), a tile with two missing couplers,
    a tile that shares an edge with tile 0, and a tile whose nodes carry strong pairs. Strong
    couplers outside every tile join disjoint pairs of b and c spins. `fields` nodes get a field.
    """
    a = list(range(n_ring))
    b = list(range(n_ring, 2 * n_ring))
    c = list(range(2 * n_ring, 3 * n_ring))
    tiles = [[a[t], b[t], c[t], a[(t + 1) % n_ring]] for t in range(n_ring)]
    n = 3 * n_ring
    couplings: dict[tuple[int, int], int] = {}

    def gauged(tile: list[int]) -> None:
        g = rng.choice([-1, 1], size=4)
        for i in range(4):
            for k in range(i + 1, 4):
                u, v = sorted((tile[i], tile[k]))
                couplings[(u, v)] = int(scale * g[i] * g[k])

    for tile in tiles:
        gauged(tile)
    if extras:
        reversed_tile = [n, n + 1, n + 2, n + 3]
        missing_tile = [n + 4, n + 5, n + 6, n + 7]
        shared_tile = [a[0], b[0], n + 8, n + 9]  # shares edge (a0, b0) with tile 0
        n += 10
        for tile in (reversed_tile, missing_tile, shared_tile):
            for i in range(4):
                for k in range(i + 1, 4):
                    u, v = sorted((tile[i], tile[k]))
                    couplings.setdefault((u, v), scale)
        couplings[(reversed_tile[0], reversed_tile[1])] = -scale  # one bond reversed
        del couplings[(missing_tile[0], missing_tile[1])]
        del couplings[(missing_tile[2], missing_tile[3])]
        tiles += [reversed_tile, missing_tile, shared_tile]
    n += 6  # free spins: in no tile, coupled to others only by weak glass
    pairs = []
    if extras:
        free = rng.permutation(b + c).tolist()
        while len(free) >= 2 and len(pairs) < n_ring // 2:
            u, v = free.pop(), free.pop()
            key = (min(u, v), max(u, v))
            if key in couplings:
                continue
            couplings[key] = int(rng.choice([-1000, 1000, -450]))
            pairs.append(key)
    weak = 0
    while weak < 3 * n_ring:
        u, v = sorted(rng.choice(n, size=2, replace=False).tolist())
        if (u, v) in couplings:
            continue
        couplings[(u, v)] = int(rng.choice([-31, 31]))
        weak += 1
    edges = np.array(sorted(couplings), dtype=np.int64)
    j = np.array([couplings[tuple(e)] for e in edges.tolist()], dtype=np.int64)
    order = rng.permutation(len(edges))  # the research models do not sort their edges
    h = np.zeros(n, dtype=np.int64)
    if fields:
        h[rng.choice(a, size=fields, replace=False)] = rng.choice([-scale, scale], size=fields)
    return {
        "h_milli": h,
        "edges": edges[order],
        "j_milli": j[order],
        "tetrahedra": np.array(tiles, dtype=np.int64),
    }


class FixedTime:
    """`method.time` for a fixed-work run: `process_time` returns the planned reading of each call.

    `method.run` reads the clock once at the start, then once per read for the budget left and,
    for the anneal method, once more per read for the tail. The plan keeps the budget left of
    each read at `per`, so the anneal gets `frac * per` and the tail gets `per`.
    """

    def __init__(self, readings: list[float]):
        self.readings = readings

    def process_time(self) -> float:
        return self.readings.pop(0)


def patched_lib(src: Path, workdir: Path) -> ctypes.CDLL:
    """`icetiles.c` with a thread clock that returns the block index.

    The run starts at tick 0, and the check at worm 64 * i reads tick i, so `frac = i / budget_s`
    and a run with `budget_s = B` does exactly B blocks of 64 worms.
    """
    text = src.read_text()
    assert "static double cpu_now(void)" in text and "double t0 = cpu_now();" in text
    head, _, rest = text.partition("static double cpu_now(void)")
    _, _, tail = rest.partition("\n}\n")
    text = head + "static double tick;\nstatic double cpu_now(void)\n{\n    return tick++;\n}\n" + tail
    text = text.replace("double t0 = cpu_now();", "tick = 0.0;\n    double t0 = 0.0;")
    c_file = workdir / "icetiles_ticks.c"
    c_file.write_text(text)
    so = workdir / "libicetiles_ticks.so"
    subprocess.run(
        ["gcc", "-O2", "-shared", "-fPIC", "-o", str(so), str(c_file), "-lm"], check=True
    )
    lib = ctypes.CDLL(str(so))
    lib.worm_run.restype = ctypes.c_int
    return lib


# Option sets of `method.run`. Every key of method.parse appears in at least one.
SPECS = [
    "anneal:sa=1,b0=2,len=64,fe=16,nat48=1",  # the charged-ice recipe
    "anneal:b0=2,nat48=1",  # the field-ice recipe
    "anneal",
    "desc",
    "desc:fe=4,len=32,reads=2,nat48=1",
    "anneal:b0=2,cross=0.5,nat48=1",
    "anneal:sa=1,pw=1,len=32,b0=2,fe=4,nat48=1",
    "anneal:tiles=1,pairs=1,tw=2,b0=1,b1=2,guide=0.5,fe=3,len=48",
    "anneal:reads=3,nat48=1,b0=2,fe=1,len=16,frac=0.5",
    "anneal:b0=4,b1=2,fe=0,nat48=1",
    "anneal:b0=2,len=8,fe=2,nat48=1",  # every excited tile fails at the cap
]
PER = 40  # the budget of one read, in blocks


def vectors(out: Path) -> int:
    sys.path.insert(0, os.getcwd())
    import descent as D  # noqa: PLC0415  # pyright: ignore[reportMissingImports]  # the research tree is only known at run time
    import icetiles as IT  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    import method as M  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    rng = np.random.default_rng(20261111)
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        IT._LIB = patched_lib(Path(IT.__file__).with_name("icetiles.c"), Path(tmp))
        models = (
            ("ring12-charge", synthetic(rng, 12, 250, 0, False)),
            ("ring9-fields", synthetic(rng, 9, 125, 4, False)),
            ("ring10-extras", synthetic(rng, 10, 250, 3, True)),
        )
        for name, model in models:
            n = len(model["h_milli"])
            reads = rng.choice(np.array([-1, 1], dtype=np.int8), size=(6, n))
            runs = []
            for spec in SPECS:
                prep = M.Prepared(model, spec)
                kw = prep.kw
                n_reads = max(1, min(int(kw["reads"]), len(reads)))
                if prep.name == "anneal":
                    anneal_blocks, tail_blocks = kw["frac"] * PER, PER
                    readings = [0.0] + [PER * r for r in range(n_reads) for _ in range(2)]
                else:
                    anneal_blocks, tail_blocks = PER, 0
                    readings = [0.0] + [PER * r for r in range(n_reads)]
                assert float(anneal_blocks).is_integer(), spec
                M.time = FixedTime(readings)
                seed = int(rng.integers(1, 2**62))
                s, e, used, stats = M.run(prep, reads, PER * n_reads, seed)
                assert not M.time.readings, spec
                tt = prep.tt
                runs.append(
                    {
                        "spec": spec,
                        "seed": seed,
                        "run_blocks": int(anneal_blocks),
                        "tail_blocks": int(tail_blocks),
                        "state": spins(s),
                        "energy": int(e),
                        "reads_used": int(used),
                        "stats": {k: int(v) for k, v in stats.items()},
                        "b_lo_bits": bits(prep.b_lo),
                        "b_hi_bits": bits(prep.b_hi),
                        "seeds": len(prep.worm.seed_ptr) - 1,
                    }
                )
            tt = IT.TileTables(D.Ising(model))
            cases.append(
                {
                    "name": name,
                    "h": model["h_milli"].tolist(),
                    "edges": model["edges"].tolist(),
                    "j": model["j_milli"].tolist(),
                    "tetrahedra": model["tetrahedra"].tolist(),
                    "reads": [spins(s) for s in reads],
                    "counts": tt.counts,
                    "etab": tt.etab.tolist(),
                    "tiled": tt.tiled.tolist(),
                    "free": tt.free.tolist(),
                    "strong_pairs": tt.strong_pairs.tolist(),
                    "partner": tt.partner.tolist(),
                    "runs": runs,
                }
            )
    doc = {
        "source": "new-pow/postprocess/optimizers/ice-tiles: icetiles.c worm_run, icetiles.py, method.py",
        "generator": "research/tileworm_parity/export.py vectors",
        "block": BLOCK,
        "per_read_blocks": PER,
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
