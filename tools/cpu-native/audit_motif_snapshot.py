#!/usr/bin/env python3
"""Rebuild a sampled canonical quotient and prove its optimum with HiGHS."""

import argparse
from collections import defaultdict
import hashlib
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def signed_energy(edges, spins):
    return sum(w * spins[u] * spins[v] for u, v, w in edges)


def eliminate_dominated(count, bonds, anchor):
    adjacency = [[] for _ in range(count)]
    for a, b, w in bonds:
        if w:
            adjacency[a].append((b, w))
            adjacency[b].append((a, w))
    fixed = {anchor: 0}
    while True:
        changed = False
        for u, neighbors in enumerate(adjacency):
            if u in fixed:
                continue
            field = sum(w * (1 - 2 * fixed[v]) for v, w in neighbors if v in fixed)
            radius = sum(abs(w) for v, w in neighbors if v not in fixed)
            if abs(field) >= radius:
                # Each choice is made after the previous choices took effect.
                fixed[u] = int(field > 0)
                changed = True
        if not changed:
            return fixed


def prove(count, bonds, fixed):
    free = {u: i for i, u in enumerate(u for u in range(count) if u not in fixed)}
    pairs = [(a, b, w) for a, b, w in bonds if a in free and b in free and w]
    cols = len(free) + len(pairs)
    objective = np.zeros(cols)
    constant = 0
    for a, b, w in bonds:
        if a in fixed and b in fixed:
            constant -= 2 * w * (fixed[a] != fixed[b])
        elif (a in fixed) != (b in fixed):
            locked, active = (a, b) if a in fixed else (b, a)
            if fixed[locked]:
                constant -= 2 * w
            objective[free[active]] -= 2 * w * (1 - 2 * fixed[locked])
    indices = []
    values = []
    upper = []
    for index, (a, b, w) in enumerate(pairs, start=len(free)):
        i, j = free[a], free[b]
        objective[i] -= 2 * w
        objective[j] -= 2 * w
        objective[index] = 4 * w
        for terms in (((index, 1), (i, -1)), ((index, 1), (j, -1)),
                      ((i, 1), (j, 1), (index, -1))):
            row = len(upper)
            upper.append(1.0 if len(terms) == 3 else 0.0)
            for column, value in terms:
                indices.append((row, column))
                values.append(value)
    if not free:
        return constant, constant, 0

    integrality = np.zeros(cols)
    integrality[:len(free)] = 1
    if indices:
        rr, cc = zip(*indices, strict=True)
        matrix = coo_matrix((values, (rr, cc)), shape=(len(upper), cols)).tocsc()
        constraints = LinearConstraint(matrix, -np.inf, upper)
    else:
        constraints = None
    result = milp(
        objective, integrality=integrality, bounds=Bounds(0.0, 1.0),
        constraints=constraints, options={"time_limit": 50.0, "mip_rel_gap": 0.0},
    )
    if not result.success:
        raise RuntimeError(result.message)
    binary = np.rint(result.x[:len(free)]).astype(int)
    assert np.allclose(result.x[:len(free)], binary, atol=1.0e-7, rtol=0)
    assignment = fixed | {u: binary[i] for u, i in free.items()}
    optimum = sum(-2 * w * (assignment[a] != assignment[b]) for a, b, w in bonds)
    assert abs(float(optimum) - (constant + result.fun)) < 1.0e-4
    bound = constant + result.mip_dual_bound
    assert abs(float(optimum) - bound) < 1.0e-4
    return int(optimum), float(bound), int(result.mip_node_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    payload = args.snapshot.read_bytes()
    if args.snapshot.suffix == ".gz":
        payload = gzip.decompress(payload)
    case = json.loads(payload)
    labels, spins, original = case["labels"], case["spins"], case["original"]
    flips, anchor, count = case["flips"], case["anchor"], case["groups"]
    assert len(spins) == len(labels) == 4560 and len(original) == 41205
    assert len(flips) == count and not flips[anchor]
    assert sum(group != anchor for group in labels) == case["free_spins"]
    assert signed_energy(original, spins) == case["base_energy"]
    weights = defaultdict(int)
    for u, v, w in original:
        a, b = sorted((labels[u], labels[v]))
        if a != b:
            weights[a, b] += w * spins[u] * spins[v]
    bonds = [(a, b, w) for (a, b), w in sorted(weights.items())]
    assert bonds == [tuple(edge) for edge in case["bonds"]]
    best_spins = [s * (1 - 2 * flips[labels[u]]) for u, s in enumerate(spins)]
    assert signed_energy(original, best_spins) - case["base_energy"] == case["delta"]
    fixed = eliminate_dominated(count, bonds, anchor)
    optimum, bound, nodes = prove(count, bonds, fixed)
    assert optimum == case["delta"]
    print(json.dumps({
        "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
        "free_original_spins": case["free_spins"], "unpinned_components": count - 1,
        "direct_elimination_width": case["width"], "dominance_fixed_excluding_anchor": len(fixed) - 1,
        "milp_active_components": count - len(fixed), "milp_nodes": nodes,
        "optimum_delta_milli": optimum, "milp_bound_milli": bound,
    }, indent=2))


if __name__ == "__main__":
    main()
