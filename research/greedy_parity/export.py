"""Export research greedy-descent cases for the opt-in parity test.

The research saved the final states of its compiled greedy descent
(`new-pow/postprocess/phase_a_c.py`) for the 48 archived QPU reads of each
model: `g1_c` with single flips only, and `gnat_c` with the natural moves.
This script reads those states, the archived reads, and the models under
`$QUIP_RESEARCH_ROOT` without writing there, and writes one JSON file per
model to OUT. Run the Rust test on OUT:

    QUIP_RESEARCH_ROOT=/path/to/regimes python3 -I export.py OUT
    QUIP_GREEDY_PARITY=OUT cargo test --release --features experimental \
        --test postprocess_greedy -- --ignored research_parity

Needs numpy only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# One or two cells per kind of move metadata the research models carry.
CELLS = {
    "ice-twisted-w62-dimer1200": 2,  # tetrahedra, pairs
    "ice-stack-2": 1,  # tetrahedra, pairs, vertical_bonds
    "ice-fields-h500": 1,  # tetrahedra, nonzero fields
    "dcl-k44-l5": 1,  # clusters
    "frust-dense-k12": 1,  # clusters, strong_blocks
    "dimer-r5": 1,  # clusters, pairs
    "exact-cover-mixed": 1,  # no metadata, nonzero fields
}
LISTED = ("clusters", "strong_blocks", "vertical_bonds")


SUBTREES = ("new-pow", "qpu-f17", "qpu-suite", "round2")


def rebase(path: str | Path, root: Path) -> Path:
    """`path` under `root`: the research manifests store absolute paths from the host that wrote them."""
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part in SUBTREES:
            return root.joinpath(*parts[i:])
    raise ValueError(f"{path} is outside the research subtrees {SUBTREES}")


def research(root: Path):
    new_pow = root / "new-pow"
    sys.path.insert(0, str(new_pow))
    import rescore  # noqa: PLC0415  (the research tree is only known at run time)

    archive_roots = [new_pow / "qpu-recap" / "qpu", root / "qpu-f17" / "qpu", root / "qpu-suite" / "qpu"]
    base_archives = {
        "cbfm-p-f17": [root / "qpu-f17" / "qpu" / "cbfm-p"],
        "cbfm-p": [root / "qpu-suite" / "qpu" / "cbfm-p"],
        "parity-k44": [new_pow / "qpu-recap" / "qpu" / "parity-k44"],
    }

    def targets(cell: str) -> list[dict]:
        path, keys = rescore.CELLS[cell]
        data = json.loads(rebase(path, root).read_text())
        return [row for key in keys for row in data[key]]

    def archive(cell: str, nonce: str) -> Path | None:
        bases = base_archives.get(cell) or [r / cell for r in archive_roots if (r / cell).is_dir()]
        for base in bases:
            f = base / "qpu-80" / f"{nonce}.npz"
            if f.exists():
                return f
        return None

    return targets, archive, new_pow / "postprocess" / "measurements" / "phase-a-c"


def rows(model: dict, key: str) -> list[list[int]]:
    """The rows of a node-list key, without the -1 padding, as the research reads them."""
    if key not in model or model[key].ndim != 2 or model[key].dtype.kind not in "iu" or not len(model[key]):
        return []
    return [[int(x) for x in row if x >= 0] for row in model[key].tolist()]


def spins(s: np.ndarray) -> str:
    return "".join("+" if x > 0 else "-" for x in s.tolist())


def main(out: Path) -> int:
    root = Path(os.environ["QUIP_RESEARCH_ROOT"])
    targets, archive, phase = research(root)
    out.mkdir(parents=True, exist_ok=True)
    missing = 0
    for cell, count in CELLS.items():
        results = {}
        for line in (phase / cell / "results.jsonl").read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                results[rec["nonce"]] = rec
        written = 0
        for row in targets(cell):
            nonce = row["nonce"]
            rec = results.get(nonce)
            arch = archive(cell, nonce)
            states_path = phase / cell / f"{nonce}.states.npz"
            if rec is None or arch is None or not states_path.exists():
                continue
            with np.load(rebase(row["model"], root), allow_pickle=False) as z:
                model = {k: z[k] for k in z.files}
            with np.load(arch, allow_pickle=False) as z:
                reads = z["spins"].astype(np.int8)
            with np.load(states_path, allow_pickle=False) as z:
                g1, gnat = z["g1_c"], z["gnat_c"]
            pairs = rows(model, "pairs")
            case = {
                "cell": cell,
                "nonce": nonce,
                "h": model["h_milli"].astype(np.int64).tolist(),
                "edges": model["edges"].astype(np.int64).tolist(),
                "j": model["j_milli"].astype(np.int64).tolist(),
                "tiles": rows(model, "tetrahedra"),
                "pairs": pairs,
                "listed": [r for key in LISTED for r in rows(model, key)],
                "reads": [spins(s) for s in reads],
                "g1_c": [spins(s) for s in g1],
                "gnat_c": [spins(s) for s in gnat],
                "energies": rec["energies"],
                "applied": rec["applied"],
            }
            (out / f"{cell}-{nonce[:12]}.json").write_text(json.dumps(case))
            print(cell, nonce[:12], len(reads), "reads", file=sys.stderr)
            written += 1
            if written == count:
                break
        if written < count:
            print(f"{cell}: only {written} of {count} models have saved states", file=sys.stderr)
            missing += count - written
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
