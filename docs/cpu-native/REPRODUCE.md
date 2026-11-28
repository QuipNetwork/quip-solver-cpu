# Reproduce the CPU-native results

Run these commands from the repository root. The toolchain is pinned to
Rust 1.97.1. The offline benchmark uses the topology and parameter fixture
in `tools/cpu-native/inputs` and derives a canonical instance from each
seed. Its JSON includes ordered canonical energies and a digest of those
energies and spin vectors.

## Build and check the solver

```sh
cargo build --release --locked --features experimental --bin quip-cpu-native
target/release/quip-cpu-native --capabilities
cargo test --release --locked --all-features
```

The solver uses native execution and the candidate archive when launched as
`quip-cpu-native`. The C41 kernel and the optional search passes are
selected through Cargo features. The search passes also need a top-level
`Configure.backend_toml` setting.

| Cargo feature | Backend TOML | Effect |
|---|---|---|
| `tree-polish` | `native_tree_polish = true` | Conditional tree descent and candidate recombination |
| `pair-worm-polish` | `native_pair_worm_polish = true` | Bounded pair-aware worm finish |
| `zephyr-spine-polish` | `native_zephyr_blocks = 50` | Four-spine blocks with batched reads and a per-pass plan cache |

Use zero blocks to disable the Zephyr pass. Instances outside the Zephyr
kernel's elimination bounds keep their pre-Zephyr candidates. `num_cpus`
sets the core budget.
The equivalent environment settings are `QUIP_CPU_NATIVE_TREE_POLISH`,
`QUIP_CPU_NATIVE_PAIR_WORM_POLISH`, and `QUIP_CPU_NATIVE_ZEPHYR_BLOCKS`.
TOML takes precedence over the environment.

## Compare execution and solution quality

Build both configurations and run them sequentially on
the same core for each seed. Alternate which runs first across seeds and
compare the ordered energies and `trajectory_digest` before summarizing
the wall-time ratios.

```sh
quip_c39=experimental,masked-residual-deltas,table-tile-bonds
quip_c41="$quip_c39,linked-component-lists,shallow-union-roots,packed-residual-fields"
quip_output=$(mktemp -d)
cargo build --release --locked --features "$quip_c39" --example cpu_native_bench
cp target/release/examples/cpu_native_bench "$quip_output/c39"
cargo build --release --locked --features "$quip_c41,tree-polish,pair-worm-polish,zephyr-spine-polish" --example cpu_native_bench
cp target/release/examples/cpu_native_bench "$quip_output/c41"
"$quip_output/c39" --mode pipeline --preset archive --seed 1231 --reads 16 --sweeps 8192
"$quip_output/c41" --mode pipeline --preset archive --seed 1231 --reads 16 --sweeps 8192
```

For the matched full-budget quality comparisons, use the same C41 executable
and begin with `--mode pipeline --preset archive --reads 16 --sweeps 8192`.
The benchmark checks each reported energy against the original instance.

| Finding | Seeds | Reference flags | Candidate flags |
|---|---|---|---|
| C43 | 1341–1352 and 1371–1378 | Archive preset | `--tree-polish` |
| C47 | 1521–1532 and 1581–1588 | Archive preset | `--pair-worm-polish` |
| C53 | 1821–1828 | `--pair-worm-polish` | `--pair-worm-polish --spine-blocks 50 --spine-span 4` |

For the C08 configurations, build with only `experimental` and compare
`--preset baseline` and `--preset core` at the same 16 × 8,192 budget
on seeds 1021–1028. For the recorded timing studies, use the archived
source revisions, compiler flags, affinity, and paired commands.
[Historical build and replay commands](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/EXPERIMENTS.md#reproduction-and-evaluation-scope)

## Replay and audit conditional Zephyr moves

Use the C41 executable built above. This isolates the conditional stage
on 16 shared random starts. It uses one placeholder sweep because the
standalone Zephyr backend has no anneal.

```sh
quip_spine="$quip_output/c41"
"$quip_spine" --mode pipeline --backend zephyr-spine --seed 1951 --reads 16 --sweeps 1 \
  --spine-blocks 200 --spine-span 4 --spine-batch
"$quip_spine" --mode pipeline --backend zephyr-spine --seed 1951 --reads 16 --sweeps 1 \
  --spine-blocks 200 --spine-span 4 --spine-batch --spine-cache-plans
```

The second invocation reuses the plans within the pass. Compare
`metrics.spine_seconds`, `metrics.spine_plans_built` and the ordered
output digest. Removing `--spine-batch` selects the scalar reference.
The historical C55 and C57 confirmations use separate baseline revisions
and seeds 1801–1812 and 1951–1962, respectively.
[C55 commands](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/EXPERIMENTS.md#reproduce-the-zephyr-tests) · [C57 commands](https://github.com/NSEVEjk0/quip-solver-cpu/blob/c5a3543dd40f02d9095078de4e6f086a1e81b80b/docs/cpu-native/EXPERIMENTS.md#reproduce-the-plan-study)

Use one block and one starting state to let the independent Python verifier
reconstruct the fixed-boundary objective and check its exact minimum.

```sh
"$quip_spine" --mode pipeline --preset archive --pair-worm-polish --seed 1852 --reads 1 --sweeps 8192 \
  --spine-blocks 1 --spine-span 4 --spine-snapshot "$quip_output/block.json"
python3 tools/cpu-native/audit_spine_snapshot.py "$quip_output/block.json"
```

## Replay CPU screening

The standalone FastDimer replay verifies canonical couplings and exact
score equality across all 204 archived nonces. It includes the two
paired timing workloads and an auditor for the published point estimates.
[C61–C63 replay commands](C61-C63-screening.md#reproduce-the-scoring-paths)

## Replay and audit exact motif search

Use the archived C60 checkpoints and the pinned research benchmark for
the 64-round comparison. The independent auditor also rebuilds both
conditional objectives from the archived original weights and spins.
[C64 replay and audit commands](C64-exact-motif-search.md#reproduce)
