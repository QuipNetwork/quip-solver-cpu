#!/usr/bin/env python3
"""Independently verify an exact Zephyr block from cpu_native_bench."""

import argparse
from collections import defaultdict
from heapq import heapify, heappop, heappush
import json
from pathlib import Path
from time import perf_counter


def reduced_problem(case):
    """Recover the neighborhood and boundary fields from canonical edges."""
    spine = [node // (2 * case["m"]) for node in case["ids"]]
    centers = set(case["spines"])
    groups = centers.copy()
    for (u, v), j in zip(case["edges"], case["j"], strict=True):
        if j and (spine[u] in centers or spine[v] in centers):
            groups.update((spine[u], spine[v]))
    free = {i for i, group in enumerate(spine) if group in groups}
    central = {i for i in free if spine[i] in centers}
    before = case["before"]
    after = case["after"]
    if any(before[i] != after[i] for i in range(len(spine)) if i not in free):
        raise ValueError("the returned solution changed the fixed boundary")

    constant = case["offset_milli"]
    field = {}
    edges = []
    for u, h in enumerate(case["h"]):
        if u in free:
            field[u] = h
        else:
            constant += h * before[u]
    for (u, v), j in zip(case["edges"], case["j"], strict=True):
        if not j:
            continue
        if u in free and v in free:
            edges.append((u, v, j))
        elif u in free:
            field[u] += j * before[v]
        elif v in free:
            field[v] += j * before[u]
        else:
            constant += j * before[u] * before[v]
    return field, edges, central, constant


def exact_minimum(field, edges, central):
    """Minimize independent integer factor tables by direct enumeration."""
    tables = {}
    incident = defaultdict(set)
    adjacent = {v: set() for v in field}
    next_id = 0

    def add(nodes, scores):
        nonlocal next_id
        tables[next_id] = (nodes, scores)
        for v in nodes:
            incident[v].add(next_id)
        next_id += 1

    for v, h in field.items():
        add((v,), (-h, h))
    for u, v, j in edges:
        adjacent[u].add(v)
        adjacent[v].add(u)
        add((u, v), (j, -j, -j, j))

    queue = [(v in central, len(row), v) for v, row in adjacent.items()]
    heapify(queue)
    width = 0
    total = 0
    while queue:
        _, degree, u = heappop(queue)
        if u not in adjacent or degree != len(adjacent[u]):
            continue
        width = max(width, degree)
        if width > 9:
            raise ValueError(f"unexpected width {width}")
        neighbors = tuple(sorted(adjacent.pop(u)))
        positions = {v: i for i, v in enumerate((u, *neighbors))}
        costs = [0] * (2 << degree)
        for factor in incident.pop(u):
            nodes, scores = tables.pop(factor)
            bits = [positions[v] for v in nodes]
            for v in nodes:
                if v != u:
                    incident[v].remove(factor)
            for mask in range(len(costs)):
                index = sum(((mask >> bit) & 1) << i for i, bit in enumerate(bits))
                costs[mask] += scores[index]
        reduced = [min(lo, hi) for lo, hi in zip(costs[::2], costs[1::2])]
        if not neighbors:
            total += reduced[0]
            continue
        add(neighbors, reduced)
        for v in neighbors:
            row = adjacent[v]
            row.remove(u)
            row.update(neighbors)
            row.remove(v)
            heappush(queue, (v in central, len(row), v))
    return total, width


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    case = json.loads(args.snapshot.read_text())
    field, edges, central, constant = reduced_problem(case)
    if len(field) != case["free"]:
        raise ValueError("the free-spin count differs")
    for name in ("before", "after"):
        spins = case[name]
        score = constant + sum(h * spins[u] for u, h in field.items())
        score += sum(j * spins[u] * spins[v] for u, v, j in edges)
        if score != case[f"{name}_energy_milli"]:
            raise ValueError(f"canonical scoring differs for {name}")
    started = perf_counter()
    optimum, width = exact_minimum(field, edges, central)
    audit_seconds = perf_counter() - started
    optimum += constant
    if width != case["width"] or optimum != case["after_energy_milli"]:
        raise ValueError("the independent conditional optimum differs")
    report = {
        "seed": case["seed"],
        "spines": case["spines"],
        "free_spins": len(field),
        "internal_edges": len(edges),
        "width": width,
        "before_energy_milli": case["before_energy_milli"],
        "exact_optimum_milli": optimum,
        "rust_stage_seconds": case["seconds"],
        "audit_seconds": audit_seconds,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
