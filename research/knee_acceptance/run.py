#!/usr/bin/env python3
"""Check QUI-2310 against the archived ice-charge models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Set these before numpy loads its numerical libraries, including in the parent.
THREAD_ENV = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TOKIO_WORKER_THREADS",
    )
}
os.environ.update(THREAD_ENV)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/home/carback1/quip-data/regimes/new-pow")
ARCHIVE = DATA / "ice-charge/measurements/depth-2s-100"
KNEE_ARGS = shlex.split(
    "--schedule knee --knee-b0 0.1 --knee-bk 0.4 --knee-b1 2.5 "
    "--knee-at 0.1 --knee-unit-milli 62"
)
WALL_LIMIT = 3.793
SEED = 1
SWEEPS = 72155


def score_state(
    spins: list[int], edges: np.ndarray, j: np.ndarray, h: np.ndarray
) -> int:
    """Rescore with Python integers, without overflow or float rounding."""
    if len(spins) != len(h) or any(
        type(s) is not int or s not in (-1, 1) for s in spins
    ):
        raise ValueError("solver spins must contain exactly n integers in {-1, +1}")
    return sum(
        spins[int(u)] * spins[int(v)] * int(c)
        for (u, v), c in zip(edges, j, strict=True)
    ) + sum(s * int(c) for s, c in zip(spins, h, strict=True))


def run_model(row: dict[str, Any], command: list[str]) -> dict[str, Any]:
    start = time.perf_counter()
    with np.load(row["model"], allow_pickle=False) as model:
        edges, j, h = (model[k] for k in ("edges", "j_milli", "h_milli"))
    problem = {
        "h": (h / 1000).tolist(),
        "j": (j / 1000).tolist(),
        "edges": edges.tolist(),
        "num_reads": 1,
        "num_sweeps": SWEEPS,
        "sweeps_per_beta": 1,
        "beta_range": None,
        "seed": SEED,
    }
    process = subprocess.run(
        command,
        input=json.dumps(problem),
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, **THREAD_ENV},
    )
    solutions = json.loads(process.stdout)
    if not isinstance(solutions, list) or len(solutions) != 1:
        raise ValueError("expected exactly one returned read")
    solution = solutions[0]
    spins = solution["spins"]
    if not isinstance(spins, list):
        raise ValueError("solver spins must be a JSON array")
    seconds = time.perf_counter() - start
    best = score_state(spins, edges, j, h)
    pp = int(row["qpu80_pp"])
    gap = pp - best
    reported = solution["energy_milli"]
    return {
        "nonce": row["nonce"],
        "model": row["model"],
        "seconds": seconds,
        "threads": 1,
        "best_milli": best,
        "reported_energy_milli": reported,
        "reported_energy_matches": type(reported) is int and reported == best,
        "qpu80_pp": pp,
        "gap_milli": gap,
        "gap_units": gap / 1000,
        "reached": gap >= 1000,
        "at_or_below": gap >= 0,
        "spins": spins,
        "sampler": solution.get("sampler"),
    }


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "models": len(rows),
        "reached": sum(r["reached"] for r in rows),
        "at_or_below": sum(r["at_or_below"] for r in rows),
        "median_wall_s": statistics.median(r["seconds"] for r in rows)
        if rows
        else None,
    }


def summarize(
    rows: list[dict[str, Any]],
    requested: int,
    extra_args: list[str],
    baseline: dict[str, int],
) -> dict[str, Any]:
    first, total = counts(rows[:30]), counts(rows)
    criteria: dict[str, Any] = {}
    for label, sample, needed in (("first_30", first, 30), ("all", total, 100)):
        thresholds = (
            (("reached", 14), ("at_or_below", 24))
            if label == "first_30"
            else (
                ("reached", baseline["reached"]),
                ("at_or_below", baseline["at_or_below"]),
            )
        )
        for metric, threshold in thresholds:
            criteria[f"{label}_{metric}"] = {
                "observed": sample[metric],
                "threshold": threshold,
                "comparison": ">=",
                "pass": sample[metric] >= threshold
                if sample["models"] >= needed
                else None,
            }
    median = total["median_wall_s"]
    criteria["median_wall_s"] = {
        "observed": median,
        "threshold": WALL_LIMIT,
        "comparison": "<=",
        "pass": median <= WALL_LIMIT if median is not None else None,
    }
    criteria["reported_energy_matches"] = {
        "pass": all(r["reported_energy_matches"] for r in rows) if rows else None
    }
    criteria["archived_knee_args"] = {"pass": extra_args == KNEE_ARGS}
    complete = requested == len(rows) == 100
    passed = all(c["pass"] is True for c in criteria.values()) if complete else None
    return {
        "status": ("pass" if passed else "fail") if complete else "incomplete",
        "pass": passed,
        "models_requested": requested,
        "first_30": first,
        "all": total,
        "criteria": criteria,
        "archive_results_baseline": baseline,
        "archive_summary_snapshot": {"models": 30, "reached": 14, "at_or_below": 24},
        "archive_median_wall_s": 1.8965688087046146,
        "wall_limit_s": WALL_LIMIT,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/quip-cpu-sa"
    )
    parser.add_argument("--extra-args", default=shlex.join(KNEE_ARGS))
    parser.add_argument("--models", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.models <= 100 or args.jobs < 1:
        parser.error("--models must be 1..100 and --jobs must be positive")
    if not sys.flags.isolated:
        parser.error("run Python with -I when reading the archived npz files")
    return args


def main() -> int:
    args = parse_args()
    try:
        binary = args.binary.resolve(strict=True)
        extra = shlex.split(args.extra_args)
        # These modes and budget overrides would invalidate the comparison.
        if any(
            arg.split("=", 1)[0]
            in ("--solve", "--capabilities", "--check", "--sweeps-per-beta")
            for arg in extra
        ):
            raise ValueError(
                "--extra-args cannot override solve mode or sweeps per beta"
            )
        config = json.loads((ARCHIVE / "config.json").read_text())
        targets = json.loads((DATA / "targets-pp-100.json").read_text())
        by_nonce = {r["nonce"]: r for r in targets["ice-charge"]["rows"]}
        nonces = config["nonces"]
        if len(nonces) != 100 or len(set(nonces)) != 100:
            raise ValueError("archived config must list 100 distinct nonces")
        models = [by_nonce[n] for n in nonces[: args.models]]
        archived_results = (ARCHIVE / "results.jsonl").read_bytes()
        archived = [
            json.loads(line) for line in archived_results.splitlines() if line.strip()
        ]
        archived_nonces = [r["nonce"] for r in archived]
        if len(set(archived_nonces)) != len(archived_nonces) or not set(
            archived_nonces
        ).issubset(nonces):
            raise ValueError("archived results must have distinct configured nonces")
        # The archived driver already host-rescored best_milli from its saved states.
        gaps = [by_nonce[r["nonce"]]["qpu80_pp"] - r["best_milli"] for r in archived]
        baseline = {
            "models": len(archived),
            "reached": sum(g >= 1000 for g in gaps),
            "at_or_below": sum(g >= 0 for g in gaps),
        }
        if baseline["models"] < 30:
            raise ValueError("archived comparison requires at least 30 result rows")
        command = [str(binary), "--solve", *extra]
        metadata = {
            "command": command,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "archive": str(ARCHIVE),
            "targets": str(DATA / "targets-pp-100.json"),
            "archive_results_sha256": hashlib.sha256(archived_results).hexdigest(),
            "num_reads": 1,
            "num_sweeps": SWEEPS,
            "seed": SEED,
            "jobs": args.jobs,
            "thread_env": THREAD_ENV,
        }
        args.out.mkdir(parents=True, exist_ok=False)
        results: list[dict[str, Any]] = []
        with (
            (args.out / "results.jsonl").open("x") as output,
            ThreadPoolExecutor(max_workers=args.jobs) as pool,
        ):
            # map preserves archive order, even when solver processes finish out of order.
            for result in pool.map(lambda row: run_model(row, command), models):
                results.append(result)
                output.write(json.dumps(result) + "\n")
                output.flush()
                print(
                    f"{result['nonce'][:12]} best={result['best_milli']} "
                    f"pp={result['qpu80_pp']} gap={result['gap_milli']:+d} milli "
                    f"wall={result['seconds']:.6f}s "
                    f"energy_match={result['reported_energy_matches']}",
                    flush=True,
                )
        summary = {**summarize(results, args.models, extra, baseline), **metadata}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 1 if summary["status"] == "fail" else 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError):
            print(exc.stderr, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
