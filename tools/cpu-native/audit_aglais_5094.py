#!/usr/bin/env python3
"""Rescore the archived Aglais CPU reads and optionally check benchmark replays."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


DATA = Path(__file__).resolve().parents[2] / "docs" / "cpu-native" / "aglais-5094" / "data"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def archived_inputs(manifest):
    expected = manifest["archive_members"]
    values = {}
    with tarfile.open(DATA / "replay-inputs.tar.gz", "r:gz") as archive:
        for entry in archive:
            name = entry.name
            require(entry.isfile() and name in expected and name not in values,
                    f"Unexpected archive member {name}")
            with archive.extractfile(entry) as stream:
                payload = stream.read()
            require(sha256(payload) == expected[name], f"Archive checksum mismatch for {name}")
            values[name] = json.loads(payload)
    require(values.keys() == expected.keys(), "Archive members differ from the manifest")
    return values


def rescore(checkpoint, instance, row):
    require(checkpoint["instance_digest"] == instance["instance_digest"],
            "Checkpoint instance digest mismatch")
    require(row["nonce"] == instance["nonce"], "Benchmark nonce mismatch")
    reads = checkpoint["reads"]
    require(len(reads) == row["reads"] == 16, "Read count mismatch")
    fields = instance["h_milli"]
    edges = instance["edges"]
    couplings = instance["j_milli"]
    offset = instance["offset_milli"]
    energies = []
    for recorded, spins in reads:
        require(len(spins) == len(fields) and all(s in (-1, 1) for s in spins),
                "Invalid checkpoint spin vector")
        energy = sum(h * s for h, s in zip(fields, spins, strict=True))
        energy += sum(w * spins[u] * spins[v] for (u, v), w in zip(edges, couplings, strict=True))
        require(energy == recorded, f"Read energy mismatch for nonce {instance['nonce']}")
        energies.append(energy + offset)
    require(energies == row["energies_milli"] and min(energies) == row["best_energy_milli"],
            "Canonical benchmark energies differ from the checkpoint")
    return len(energies)


def verify_compact(worktree, files):
    executable = worktree / "target" / "release" / "examples" / "cpu_native_bench"
    for path, instance, expected in files:
        result = subprocess.check_output(
            [str(executable), "--nonce", instance["nonce"], "--witnesses", str(path)],
            cwd=worktree,
        )
        row = json.loads(result)
        require(row["instance_digest"] == bytes(instance["instance_digest"]).hex()
                and row["best_energy_milli"] == expected,
                f"Canonical witness rescore failed for {path.name}")


def verify_replay(path, expected):
    row = json.loads(path.read_bytes())
    for key in ("nonce", "energies_milli", "best_energy_milli", "trajectory_digest"):
        require(row[key] == expected[key], f"Replay mismatch in {key} for {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-worktree", type=Path, help="Also decode compact witnesses with the built cold benchmark")
    parser.add_argument("--cold-output", type=Path, help="Compare JSON from a fresh cold replay to the archive")
    parser.add_argument("--polish-output", type=Path, help="Compare JSON from a fresh polish replay to the archive")
    options = parser.parse_args()

    manifest = json.loads((DATA / "manifest.json").read_bytes())
    for name, expected in manifest["files_sha256"].items():
        require(sha256((DATA / name).read_bytes()) == expected, f"File checksum mismatch for {name}")
    values = archived_inputs(manifest)
    cases = manifest["chain"]["cases"]
    topology = manifest["chain"]["topology"]
    instances = {block: values[f"instances/{block}.json"] for block in cases}
    compact = []
    for block, instance in instances.items():
        case = cases[block]
        require(instance["block"] == int(block) and instance["nonce"] == case["nonce"],
                f"Canonical snapshot identity failed for block {block}")
        require(len(instance["h_milli"]) == topology["nodes"]
                and len(instance["edges"]) == len(instance["j_milli"]) == topology["edges"],
                f"Graph size mismatch for block {block}")
        chain_witness = DATA / f"chain-witness-{block}.json"
        require(sha256(chain_witness.read_bytes()) == case["witness_sha256"],
                f"Chain witness checksum mismatch for block {block}")
        compact.append((chain_witness, instance, case["energy_milli"]))

    cold = [json.loads(line) for line in (DATA / "cold-runs.jsonl").read_text().splitlines()]
    polish = json.loads((DATA / "polish-runs.json").read_bytes())
    count = sum(rescore(values[row["checkpoint"]], instances[str(row["block"])], row["result"]) for row in cold)
    instance = instances[str(polish["block"])]
    for arm in polish["arms"].values():
        count += rescore(values[arm["checkpoint"]], instance, arm["result"])

    cold_hits = [row for row in cold if row["result"]["best_energy_milli"] <= manifest["threshold_milli"]]
    require(len(cold) == 9 and len({row["block"] for row in cold}) == 3 and len(cold_hits) == 1,
            "Cold cohort count mismatch")
    selected, = cold_hits
    require(selected["id"] == manifest["headline"]["cold_id"]
            and selected["checkpoint"] == polish["input"]
            and selected["block"] == polish["block"], "Polish input is not the archived cold hit")
    baseline = selected["result"]
    polished = polish["arms"][manifest["headline"]["polish_arm"]]["result"]
    require(baseline["best_energy_milli"] == polish["baseline_milli"] == -5_093_665
            and polished["best_energy_milli"] == -5_094_135, "Headline energies differ from the record")
    for arm in ("reference", "exact"):
        reference = polish["arms"][arm]
        require(values[reference["checkpoint"]] == values[polish["input"]]
                and reference["result"]["trajectory_digest"] == baseline["trajectory_digest"],
                f"Unexpected change in the {arm} polish arm")

    for label, expected in (("cold", baseline), ("polished", polished)):
        witness = DATA / f"{label}-witness.json"
        stored = json.loads((DATA / f"{label}-witness-rescore.json").read_bytes())
        require(stored["encoded_witness_sha256"] == sha256(witness.read_bytes())
                and stored["nonce"] == instance["nonce"]
                and stored["instance_digest"] == bytes(instance["instance_digest"]).hex()
                and stored["best_energy_milli"] == expected["best_energy_milli"],
                f"Archived witness rescore mismatch for {label}")
        compact.append((witness, instance, expected["best_energy_milli"]))

    if options.cold_worktree:
        verify_compact(options.cold_worktree.resolve(), compact)
    if options.cold_output:
        verify_replay(options.cold_output, baseline)
    if options.polish_output:
        verify_replay(options.polish_output, polished)
    print(json.dumps({
        "canonical_read_scores_checked": count, "cold_runs": len(cold),
        "cold_nonces": len(cases), "cold_target_hits": len(cold_hits),
        "cold_energy_milli": baseline["best_energy_milli"],
        "polished_energy_milli": polished["best_energy_milli"],
        "compact_witnesses_decoded": len(compact) if options.cold_worktree else 0,
    }, indent=2))


if __name__ == "__main__":
    main()
