#!/usr/bin/env python3
"""Recompute screening point estimates and check the archived paired results."""

from collections import Counter
from itertools import combinations
import json
from math import isclose
from pathlib import Path
from statistics import mean, median


DATA = Path(__file__).resolve().parents[3] / "docs/cpu-native/data/screening"
CPU_TARGET = -5_052_195
GPU_TARGET = -5_090_000


def read(name):
    return json.loads((DATA / name).read_text())


def rows(name):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines()]


def auroc(group, score):
    low = [r[score] for r in group if r["best_energy_milli"] <= CPU_TARGET]
    high = [r[score] for r in group if r["best_energy_milli"] > CPU_TARGET]
    return mean((a < b) + 0.5 * (a == b) for a in low for b in high)


def shortlist_gain(pool):
    ordered = sorted(r["frustrated_weak_triangles"] for r in pool)
    cutoff = ordered[2]
    forced = [r["best_energy_milli"] for r in pool if r["frustrated_weak_triangles"] < cutoff]
    tied = [r["best_energy_milli"] for r in pool if r["frustrated_weak_triangles"] == cutoff]
    random = mean(map(min, combinations((r["best_energy_milli"] for r in pool), 3)))
    selected = mean(min([*forced, *tail]) for tail in combinations(tied, 3 - len(forced)))
    return random - selected


def check_timings(stem):
    saved = read(f"{stem}-timing-summary.json")
    timing = (DATA / f"{stem}-timing.tsv").read_text().splitlines()
    assert timing[0].startswith("setup\t") and timing[-1] == "verified\t204"
    pairs = [line.split("\t") for line in timing[1:-1]]
    assert len(pairs) == saved["timed_pairs"] == 26
    assert [int(pair[1]) for pair in pairs] == [*range(2301, 2405, 8), *range(2501, 2601, 8)]
    assert all(pair[0] == "pair" for pair in pairs)
    reference = [1000 * mean(map(float, pair[2:4])) for pair in pairs]
    candidate = [1000 * mean(map(float, pair[4:6])) for pair in pairs]
    ratios = [a / b for a, b in zip(reference, candidate, strict=True)]
    assert isclose(median(reference), saved["reference_stage_ms_median"])
    assert isclose(median(candidate), saved["fast_stage_ms_median"])
    assert isclose(median(ratios), saved["paired_speedup_median"])
    assert sum(x > 1 for x in ratios) == saved["paired_faster_count"]
    print(f"{stem} median timing ratio {median(ratios):.3f}x")


def main():
    cohort = rows("nonce-scores.jsonl")
    assert len({r["nonce"] for r in cohort}) == len(cohort) == 204
    assert Counter(r["cohort"] for r in cohort) == {"development": 24, "holdout": 80, "confirmation": 100}
    assert all(r["best_energy_milli"] == min(r["energies_milli"]) for r in cohort)
    scores = {
        "geometry": "rank4_milli", "structure": "frustrated_weak_triangles",
        "short_probe": "probe64_milli", "medium_probe": "probe256_milli",
    }
    for name, key in (("holdout", "top_decile_auroc"), ("confirmation", "cpu_threshold_auroc")):
        group = [r for r in cohort if r["cohort"] == name]
        saved = read(f"{name}.json")
        assert not any(r["best_energy_milli"] <= GPU_TARGET for r in group)
        for arm, field in scores.items():
            assert isclose(auroc(group, field), saved["scores"][arm][key])
        gain = mean(shortlist_gain(group[i:i + 10]) for i in range(0, len(group), 10))
        field = "mean_best_energy_gain_vs_random_milli" if name == "holdout" else "mean_best_gain_milli"
        assert isclose(gain, saved["scores"]["structure"][field])
        print(f"{name} ranking and shortlist estimates verified for {len(group)} nonces")
    for stem in ("triangle", "rank4"):
        check_timings(stem)
    warm = rows("warm-start-pairs.jsonl")
    changes = [r["cold"]["best_energy_milli"] - r["warm"]["best_energy_milli"] for r in warm]
    assert changes == read("warm-start-summary.json")["best_gains_milli"]
    assert all(r["cold"]["solver_seed"] == r["warm"]["solver_seed"] for r in warm)
    crosses = rows("recombination-pairs.jsonl")
    gains = []
    for row in crosses:
        baseline = row["results"]["reference"][0]
        child = row["results"]["pairs"][0]
        gains.append(baseline["best_energy_milli"] - child["best_energy_milli"])
        assert all(a >= b for a, b in zip(baseline["energies_milli"], child["energies_milli"], strict=True))
        if gains[-1] == 0:
            assert child["trajectory_digest"] == baseline["trajectory_digest"]
    assert gains == read("recombination-summary.json")["arms"]["pairs"]["gains_milli"]
    print(f"{len(warm)} warm pairs and {len(crosses)} recombination pairs verified")


if __name__ == "__main__":
    main()
