# C61–C63. CPU screening and candidate experiments

[CPU-native research log](README.md)

The corpus uses canonical `quip/twisted-ice-dimer` revision 1 with
`m=12`, `t=4`, weak magnitude 45, and 1,350 dimers. The fixed topology
has 4,560 spins and 41,205 edges. All 204 offline nonces were derived
from deterministic seeds. The full labels use 16 replicas × 8,192
sweeps with worm polish, 200 Zephyr blocks with C60 reuse, and the C60
motif finish. [Nonce scores and full labels](data/screening/nonce-scores.jsonl) ·
[Initial label build and flags](data/screening/label-manifest.json) ·
[Confirmation label build and flags](data/screening/confirmation-label-manifest.json)

## C61. Faster preparation for the same screening scores

This experiment reuses the published feeder's CPU FastDimer generator.
It compares that path with canonical generation on the same prepared
topology. On every nonce, the harness checked the full ordered edge list,
every original integer coupling, and the structural score. Both
generation paths also reproduced the frozen continuous rank-4 scores
bit for bit on all 204 nonces. [FastDimer source at the tested revision](https://github.com/chenyiyi0618/quip-solver-cuda/blob/e130011cd970128607e5931afd9c11560f396d57/contrib/super-sdp-feeder/src/engine.rs)

| Prepared CPU path | Canonical | FastDimer | Median paired ratio (95% CI) |
|---|---:|---:|---:|
| Nonce generation and continuous rank-4 scoring | 21.77 ms | 8.64 ms | **2.52×** (2.50–2.53×) |
| Nonce generation and weak-triangle scoring | 15.57 ms | 2.13 ms | **7.28×** (7.22–7.31×) |

Each timing study used 26 nonces on one pinned AMD EPYC 7663 core. Each
arm ran twice per nonce in ABBA or BAAB order. Each paired ratio divides
the two-repeat reference mean by the candidate mean. The percentile
interval bootstraps those ratios by nonce. The rank-4 timer includes
per-nonce adjacency construction, 24 vector sweeps, and continuous
scoring. The triangle timer uses a shared precomputed table of graph
triangles. One-time topology preparation is recorded separately.
[Rank-4 timing pairs](data/screening/rank4-timing.tsv) · [Rank-4 summary](data/screening/rank4-timing-summary.json) ·
[Triangle timing pairs](data/screening/triangle-timing.tsv) · [Triangle summary](data/screening/triangle-timing-summary.json)

### Ranking quality

The first 24 nonces selected the rank-4 relaxation with 3× weak
couplings and the count of frustrated weak triangles. The continuous
score evaluates the relaxed vectors with the original couplings.
The lowest-energy decile of the first 80 holdout labels fixed the
CPU ranking cutoff at −5,052,195 milli. The next 100 nonces used
that same cutoff. [Frozen policy](data/screening/policy.json) ·
[Original scorer source](https://github.com/NSEVEjk0/quip-solver-cpu/tree/98297fdab672a0feef658f989e2285cdc747b172)

| Score | First holdout AUROC | Fresh cohort AUROC |
|---|---:|---:|
| Rank-4 relaxation | 0.710 | 0.657 |
| Frustrated weak triangles | 0.530 | 0.573 |
| Native cluster probe, 4 × 64 | 0.543 | 0.511 |
| Native cluster probe, 4 × 256 | 0.780 | 0.528 |

The paired rank-4 minus short-probe 95% interval was −0.090 to +0.448
on the first holdout and −0.117 to +0.426 on the fresh cohort. The
separate shortlist experiment keeps three out of ten nonces per pool
and averages ties at the cutoff. The structural screen's mean gain
over the exact uniform-shortlist expectation shrank from 3,426 to
792 milli on the fresh cohort. Its fresh interval was −1,527 to
+3,139 milli and its permutation probability was 0.227. It missed the
confirmation criterion. [First holdout](data/screening/holdout.json) ·
[Confirmation plan](data/screening/confirmation-plan.md) · [Fresh cohort](data/screening/confirmation.json)

The GPU study uses the literal energy cutoff −5,090,000 milli for
its AUC90, with 85 positives among 819 post-BM-SDP survivors. There
were zero hits at that cutoff across the two CPU holdouts. The
best CPU label was −5,062,065 milli. CPU AUROC, offline stage time,
and live target-hit throughput have different denominators.
[GPU screening experiments](https://github.com/chenyiyi0618/quip-solver-cuda/blob/e130011cd970128607e5931afd9c11560f396d57/contrib/super-sdp-feeder/EXPERIMENTS.md)

## C62. Rounded starts in the population annealer

Twelve paired 16 × 2,048 runs shared the same nonces and solver seeds.
The treatment replaced two initial replicas with distinct rank-4
hyperplane roundings after single-spin descent. It consumed the
original initialization RNG draws. Seven runs improved and five
regressed. The median best-energy gain was 530 milli, with a
95% interval from −4,405 to +3,910 milli. The largest regression
was 7,190 milli. The reduced-budget signal remains inconclusive.
[Paired reads](data/screening/warm-start-pairs.jsonl) ·
[Summary](data/screening/warm-start-summary.json) ·
[Prototype source](https://github.com/NSEVEjk0/quip-solver-cpu/tree/119a4ed24f3b5bcbfa20e385fec9d70ed706f8fa)

## C63. Exact search on disagreements after C60

This prototype aligns a donor's global spin orientation with the
best read, fixes their agreements, and runs width-capped variable
elimination over their disagreements. It accepts a new best read
only on a strict canonical energy improvement. Eight development
checkpoints selected two rounds of individual donor comparisons.

On 20 fresh full-budget checkpoints, gains were 90 and 110 milli
in two cases. Eighteen returned the same ordered output digest.
All 300 donor reads were byte-identical to the control. The planner
solved 244 of 330 attempted regions, with up to 172 free spins.
The added stage took a median 8.78 ms including graph preparation.
[Frozen policy](data/screening/recombination-policy.json) ·
[Paired reads and repeats](data/screening/recombination-pairs.jsonl) ·
[Summary](data/screening/recombination-summary.json) ·
[Prototype source](https://github.com/NSEVEjk0/quip-solver-cpu/tree/8f4774e824bcd28540067200ff5dfa2916404573)

The recombination prototype passed formatting, strict release Clippy,
its exhaustive conditional-optimum test, and 469 release library tests
on four offline cores. One existing library test was ignored.

## Reproduce the scoring paths

Run from the repository root with Rust 1.97.1. The standalone harness
downloads the pinned engine, checks its hash, and builds in a scratch
directory. Use `--engine /path/to/engine.rs` with an existing copy.
The executable checks all 204 case scores and emits repeated timings
on every eighth nonce. Choose an available idle CPU for `quip_cpu`.

```sh
quip_screen=$(mktemp -d)
quip_cpu=2
python3 tools/cpu-native/screening-study/audit.py
python3 tools/cpu-native/screening-study/prepare.py "$quip_screen"
(
  cd "$quip_screen"
  cargo +1.97.1 build --release --locked
  taskset -c "$quip_cpu" target/release/quip-cpu-screening-study > triangle-replay.tsv
  cargo +1.97.1 build --release --locked --features geometry
  taskset -c "$quip_cpu" target/release/quip-cpu-screening-study > rank4-replay.tsv
)
```

The auditor recomputes the point estimates from the archived timings,
full labels, and paired reads. To recompute individual full labels,
use the full 200-block reuse plus motif command in the C59–C60 guide.
Add `--save-reads` to recreate the C63 input. Replay that checkpoint
with the C63 prototype using `--load-reads` and
`--disagreement-rounds 2`. [Full-budget commands](C59-C60-polishing.md#reproduce)
