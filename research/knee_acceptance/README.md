# Knee schedule acceptance

Check QUI-2310 on the 100 archived ice-charge models.
Each run uses one read, 72,155 sweeps, and seed 1.

Run from the repository root with a binary that supports the knee schedule:

```sh
/home/carback1/quip-miner-dwave-cascade/.venv/bin/python -I \
  research/knee_acceptance/run.py \
  --binary target/release/quip-cpu-sa --jobs 1 \
  --out research/knee_acceptance/results-knee
```

The default extra arguments are exactly:

```text
--schedule knee --knee-b0 0.1 --knee-bk 0.4 --knee-b1 2.5 --knee-at 0.1 --knee-unit-milli 62
```

For a dry run with the released default schedule:

```sh
cargo build --release --bin quip-cpu-sa
/home/carback1/quip-miner-dwave-cascade/.venv/bin/python -I \
  research/knee_acceptance/run.py \
  --binary target/release/quip-cpu-sa --extra-args "" \
  --models 5 --jobs 5 --out research/knee_acceptance/results-default
```

`--models` defaults to 100. `--jobs` defaults to 1 and sets the number of concurrent model runs.
Each solver process samples on one thread. The solve schema has no `num_cpus` field.
SA reads run in order, without the session worker pool.

The runner sets these thread limits to 1:

```text
OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS RAYON_NUM_THREADS
NUMEXPR_NUM_THREADS TOKIO_WORKER_THREADS
```

`--out` must name a new directory. Existing paths cause an error.

The output files are `results.jsonl`, in archive order, and `summary.json`, after all runs finish.
Each result keeps the spins, host energy, reported energy, target, gap, flags, and wall time.

| Exit code | Meaning |
| --- | --- |
| 0 | Pass or incomplete dry run. Check the summary status. |
| 1 | Complete run fails acceptance. |
| 2 | Input or solver error. Completed rows remain for inspection. |

## Scores and limits

The source root is `/home/carback1/quip-data/regimes/new-pow`.
Model order comes from `ice-charge/measurements/depth-2s-100/config.json`, key `nonces`.
Targets and paths come from `targets-pp-100.json`, key `ice-charge`, then `rows`.
The models live in `/home/carback1/quip-data/regimes/qpu-f17/work/cells/ice-charge/models`.

The archived driver is `cpu-ladder-100/ladder100.py`. Its definitions are verbatim:

```python
UNIT = 1000
gaps.append(r["qpu80_pp"] - e)
"reached": sum(g >= UNIT for g in gaps), "at_or_below": sum(g >= 0 for g in gaps),
```

A reached model has energy at least 1,000 milli below its target.
At or below includes ties. A positive gap means a better classical energy.

Host scoring sums the edge products and fields from the original model, with exact Python integers.
The solve input divides coefficients by 1,000. The reported energy is only a unit check.

| Check | Limit |
| --- | --- |
| First 30 models | At least 14 reached and 24 at or below |
| All 100 models | At least the counts in the archived `results.jsonl` |
| Median wall | At most 3.793 seconds |
| Unit check | Every reported energy matches the host score |
| Full pass | All 100 models and the default knee arguments |

The archive summary covers 30 models, with counts of 14/24.
The results file now covers 100 models, with counts of 49/79. The runner reads this baseline at startup.

The archived median is 1.8965688087046146 seconds. The brief fixes the wall limit at 3.793 seconds.

The timer covers model loading, JSON encoding, the solver process, and state decoding.
Host scoring is outside the timer, as in the archive.

Short runs report incomplete count checks. Each check prints its threshold and pass value.
Use `--jobs 1` for a timing check with less CPU contention.
