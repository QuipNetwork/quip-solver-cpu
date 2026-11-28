# CPU-native research log

`quip-cpu-native` is an experimental CPU solver for `quip/twisted-ice-dimer`
revision 1. It builds on Quip’s Otsuka cluster kernel and uses the canonical
family generator and the standard solver protocol. The benchmarked Zephyr
graph has 4,560 spins and 41,205 edges.

## Solid results

The full-budget comparisons used 16 replicas × 8,192 annealing sweeps.
All timings were collected on one AMD EPYC 7663 VPS. Each measurement
below refers to the recorded research builds.

| Finding | Result | Evidence |
|---|---|---|
| C59–C60 polishing after the full anneal | Unchanged-batch reuse reached **1.983×** median Zephyr-stage speedup over C57 with exact outputs on all 16 starts. The full 200-block pass with reuse and motif polishing lowered best energy on six of 16 and tied on ten, at **1.305×** median stage speedup. | [Polishing methods and confirmation pairs](C59-C60-polishing.md) |
| C61 CPU screening preparation | On a prepared topology, the published FastDimer path reached **2.52×** for continuous rank-4 scoring and **7.28×** for triangle scoring. Both paths reproduced all 204 archived scores. | [Screening methods, timing pairs, and replay](C61-C63-screening.md#c61-faster-preparation-for-the-same-screening-scores) |
| C64 exact motif search | **1.285×** median speedup over the cut-only prototype for the added 64-round stage. HiGHS independently verified a conditional optimum over **983 spins in 373 flippable groups**. | [Paired read replay and independent audit](C64-exact-motif-search.md) |
| C41 packed fields and component membership | **1.341×** median pipeline speedup over C39 in eight same-core full-budget pairs, with identical ordered spins and energies. The paired bootstrap 95% median interval was 1.291–1.398×. | [Builds and raw pairs](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/C41-full-budget.md) |
| C53 exact Zephyr optimization | An independent integer solver verified ten fixed-boundary optima, including 2,385 free spins at elimination width nine. Adding 50 blocks per read to C41 with worm polish lowered the best canonical energy on five of eight full-budget instances and tied on three. The median gain was **90 milli units** across all eight. | [Exactness and full-budget pairs](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/EXPERIMENTS.md#exact-conditional-zephyr-neighborhoods) |
| C43 tree descent and recombination | Lower best canonical energy on 15 of 20 full-budget instances versus C41, with five ties and a median gain of **90 milli units**. | [Full-budget pairs](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/EXPERIMENTS.md#full-budget-candidate-quality) |
| C47 pair-aware worm polish | Lower best canonical energy on 14 of 20 full-budget instances versus C41, with six ties and a median gain of **145 milli units**. | [Full-budget pairs](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/EXPERIMENTS.md#canonical-energy-and-runtime-comparisons) |
| C55 batching and C57 plan reuse | The conditional stage reached **1.440×** median speedup for batching over scalar and **1.561×** for compact plan reuse over the C55 batched stage. Both studies used 16 random starts and 200 conditional blocks per read, with exactly matching outputs in each of 12 pairs per study. | [Batching](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/C51-C55-zephyr-exact.md#c55-conditional-stage-timing) · [Plan reuse](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/EXPERIMENTS.md#reusing-zephyr-neighborhood-plans) |

C41 also measured **1.516×** in a separate side-by-side comparison against the
earlier native reference. The C41 pipeline timer includes graph preparation
and archive polishing. C57 built 50 plans in place of 200, with 35.45 MiB in
retained plan buffers. Its four full-budget pairs matched exactly and the
whole-solver timing interval spanned parity. [C41 timing scope](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/C41-full-budget.md) · [C57 full-budget pilot](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/data/zephyr-followup/packed-plan-warm/results.json)

## Execution and candidate preservation

- C01–C04 fuse adjacency updates, specialize union-find, share exact
  temperature-local probabilities, and copy spins and fields across
  resampling. At full budget the combined C08 execution path reached
  **1.122×** median speedup over its scalar reference, with identical
  candidate outputs in all eight held-out pairs. [Execution study](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/EXPERIMENTS.md#full-budget-held-out-confirmation)
- C05–C06 preserve an original-energy record across resampling and
  use exact motif-domain scoring when weak couplings have uniform
  magnitude. The archive lowered best energy on two of eight held-out
  instances and tied on six. The lease path independently scores the
  returned spins using unscaled coefficients and the canonical offset.
  [Archive results](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/FINDINGS.md#c05) · [Energy conversion](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/EXPERIMENTS.md#exact-motif-domain-archive-scoring)

## Reproduce and browse

- Build the solver, replay the paired workloads, and independently verify
  a width-nine Zephyr block. [Reproduction guide](REPRODUCE.md)
- Reproduce the CPU scoring timings and inspect the held-out ranking,
  rounded-start, and exact recombination results. [C61–C63](C61-C63-screening.md)
- Replay the 16 saved C60 checkpoints and audit two exact motif snapshots.
  [C64](C64-exact-motif-search.md)
- Browse findings C01–C58, with methods and raw data for each comparison.
  C38’s wall-time result remains inconclusive. [Full index](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/FINDINGS.md#entry-index) · [C38 timing](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/C30-C38-timing.md#c38-wall-time)
