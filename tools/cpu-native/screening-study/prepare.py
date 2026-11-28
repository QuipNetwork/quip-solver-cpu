#!/usr/bin/env python3
"""Assemble the standalone FastDimer replay in a scratch directory."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen


SOURCE = (
    "https://raw.githubusercontent.com/chenyiyi0618/quip-solver-cuda/"
    "e130011cd970128607e5931afd9c11560f396d57/contrib/super-sdp-feeder/src/engine.rs"
)
ENGINE_SHA256 = "f8327f242e990608f02a211c9e7c95532dbc978282659743194c8a3598381710"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--engine", type=Path, help="local copy of the pinned engine")
    args = parser.parse_args()
    if args.engine:
        engine = args.engine.read_bytes()
    else:
        with urlopen(SOURCE, timeout=30) as response:
            engine = response.read()
    if hashlib.sha256(engine).hexdigest() != ENGINE_SHA256:
        parser.error("FastDimer source differs from the recorded SHA-256")

    tool = Path(__file__).resolve().parent
    repository = tool.parents[2]
    inputs = tool.parent / "inputs"
    topology = json.loads((inputs / "current-topology.spec.json").read_text())
    nodes, edges = topology["nodes"], topology["edges"]
    cases = []
    data = repository / "docs/cpu-native/data/screening/nonce-scores.jsonl"
    for line in data.read_text().splitlines():
        row = json.loads(line)
        cases.append(
            f"{row['seed']} {row['nonce']} {row['frustrated_weak_triangles']} {row['rank4_milli']!r}"
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("Cargo.toml", "Cargo.lock", "check.rs", "rank.rs", "inspect.rs"):
        shutil.copyfile(tool / name, output / name)
    shutil.copyfile(inputs / "current-family.params", output / "params.bin")
    (output / "engine.rs").write_bytes(engine + b"\n" + (tool / "inspect.rs").read_bytes())
    rows = (f"{len(nodes)} {len(edges)}", " ".join(map(str, nodes)), *(f"{a} {b}" for a, b in edges))
    (output / "topology.txt").write_text("\n".join(rows) + "\n")
    (output / "cases.txt").write_text("\n".join(cases) + "\n")
    print(output)


if __name__ == "__main__":
    main()
