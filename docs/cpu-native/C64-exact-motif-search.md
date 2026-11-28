# C64. Exact motif search on saved C60 candidates

[CPU-native research log](README.md)

Exact elimination ran the added 64-round motif search in **85.87 ms** versus
**108.67 ms** for the cut-only prototype on the same 16 saved C60 checkpoints.
The median paired speed ratio was **1.285×** (95% bootstrap interval
**1.188–1.388×**). Exact search ran faster on 15 of 16 checkpoints. HiGHS also
verified that a sampled **983-spin neighborhood in 373 flippable groups**
was already optimal under its recorded group moves.

## Search and timing

Each round pairs opposite gauged spins within each tile and joins the two
ends of each dimer. The resulting connected groups can flip as units. The
search fixes the largest group to choose a global spin orientation. Each
original edge between groups contributes its coupling times the product
of the starting spins to the signed graph of group variables. The exact
prototype fixes variables whose boundary field dominates the remaining
incident couplings, then eliminates variables up to width ten. When
elimination exceeds the width cap, it uses the roof-duality cut also used
by the `cut` arm. The `exact-light` arm releases every sixteenth tile pairing
before building the groups.

Seeds 2701–2712 were used for development. The confirmation used seeds
6461–6476. Each checkpoint contains the ordered results of 16 replicas ×
8,192 sweeps, worm polish, 200 Zephyr blocks with C60 reuse, and the C60
motif finish. The added search works on the best saved read and accepts
only strict improvements in canonical energy.

The timed region includes graph and quotient preparation for the added
stage. The anneal and C60 finish had already run when that timer began.
Each arm ran twice per seed on the same pinned AMD EPYC 7663 core, with a
rotated order followed by its reverse. The paired ratio divides the
two-repeat cut mean by the exact mean for each seed. The percentile
interval resamples those 16 ratios by seed.

| Added 64-round stage | Median time | Quotients solved by elimination | Improved over saved C60 | Maximum gain |
|---|---:|---:|---:|---:|
| Cut | 108.67 ms | — | 4 / 16 | 900 milli |
| Exact | **85.87 ms** | **984 / 1,024** | 5 / 16 | 1,170 milli |
| Exact with pairing release | 123.58 ms | 718 / 1,024 | 5 / 16 | 990 milli |

All three arms had a median energy gain of zero versus saved C60. The
11 ties for the exact arm kept the full ordered output digest. On the
five improved checkpoints, the 15 other reads stayed byte-identical.
The best energy across the cohort was −5,050,075 milli in all three arms.
Each arm recorded zero hits at or below −5,090,000 milli.

[Raw pairs and checkpoint checksums](data/motif-exact/pairs.jsonl) ·
[Timing ratios and interval](data/motif-exact/timing.json) ·
[Energy and search totals](data/motif-exact/summary.json) ·
[Frozen policy](data/motif-exact/policy.json) ·
[C60 checkpoint provenance](data/motif-exact/label-manifest.json)

## Independent check of the conditional optima

The auditor reconstructs the group objective from the original integer
edge weights, the starting spin vector, and a group label for each spin.
It verifies the recorded energy change directly on the original edges.
It then applies the boundary-field fixing rule and sends the remaining
binary model to HiGHS. The integer objective and the HiGHS lower bound
both match the recorded delta of zero in each snapshot.

| Snapshot | Original spins outside the fixed group | Flippable groups | Elimination width | Independent audit |
|---|---:|---:|---:|---|
| 6464, strict pairings | 746 | 291 | 7 | [Input](data/motif-exact/certificate-6464.json.gz) · [Audit](data/motif-exact/certificate-6464-audit.json) |
| 6471, pairing release | **983** | **373** | **8** | [Input](data/motif-exact/certificate-6471.json.gz) · [Audit](data/motif-exact/certificate-6471-audit.json) |

These optima cover the recorded group flips and fixed orientation. Both
incumbents attained their conditional optimum.

## Reproduce

The PR publishes the study evidence. Build the experimental benchmark
from the pinned source revision with Rust 1.97.1. From the PR checkout,
extract the provided checkpoints, then run the benchmark from the
research worktree so it finds the canonical fixture. Set `quip_cpu` to
an idle core.

```sh
quip_study=$(mktemp -d)
git fetch https://github.com/NSEVEjk0/quip-solver-cpu.git cpu-motif-exact-study
git worktree add --detach "$quip_study/source" 8a018a482f51e77ad078b19246ce2b51d944de2f
tar -xzf docs/cpu-native/data/motif-exact/checkpoints.tar.gz -C "$quip_study"
quip_cpu=2
(
  cd "$quip_study/source"
  quip_features=experimental,masked-residual-deltas,table-tile-bonds,linked-component-lists,shallow-union-roots,packed-residual-fields,pair-worm-polish,zephyr-spine-polish,tree-polish,motif-descent
  cargo +1.97.1 build --release --locked --features "$quip_features" --example cpu_native_bench
  for quip_seed in 6464 6471; do
    for quip_mode in cut exact exact-light; do
      taskset -c "$quip_cpu" target/release/examples/cpu_native_bench \
        --mode pipeline --preset archive --seed "$quip_seed" --reads 16 --sweeps 8192 \
        --load-reads "$quip_study/$quip_seed.json" --pairing-rounds 64 --pairing-mode "$quip_mode"
    done
  done
)
```

Compare `energies_milli` and `trajectory_digest` with the corresponding
rows in the archive. For the saved C60 control, omit both pairing flags.
For a timing replay, use all 16 seeds and the recorded `order` within
each seed, and compare `metrics.pairing_seconds`.
[Measured source revision](https://github.com/NSEVEjk0/quip-solver-cpu/tree/8a018a482f51e77ad078b19246ce2b51d944de2f) ·
[Original checkpoints](data/motif-exact/checkpoints.tar.gz)

With NumPy and SciPy installed, verify the archived snapshots directly
from the PR checkout.

```sh
python3 tools/cpu-native/audit_motif_snapshot.py docs/cpu-native/data/motif-exact/certificate-6464.json.gz
python3 tools/cpu-native/audit_motif_snapshot.py docs/cpu-native/data/motif-exact/certificate-6471.json.gz
```
