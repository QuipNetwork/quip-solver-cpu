# C59–C60. Motif polishing and exact Zephyr reuse

[CPU-native research log](README.md)

The comparisons use `quip/twisted-ice-dimer` revision 1 on the canonical
4,560-spin, 41,205-edge graph. Every warm start comes from 16 replicas ×
8,192 sweeps with C41 execution, the candidate archive, and pair-aware
worm polish. All timing measurements use one AMD EPYC 7663 VPS.

## C59. Motif component descent and recombination

For a single candidate, each tile pairs opposite gauged spins and joins
any unmatched spins into one group. Both ends of each dimer join the
same group. Joining the groups across motifs gives components whose
flips preserve every tile's charge magnitude and every dimer spin
product. Downhill component updates use the full original coefficients.

Recombination fixes the spins on which two candidates agree and builds
components among the disagreements. Each donor is tried in both global
orientations and from both initial assignments. The full original
energy decides whether the best candidate is replaced. The benchmark
independently rescores all 16 outputs.

The selected configuration runs 50 four-spine Zephyr blocks per read,
32 rounds of motif component descent per read, and four recombination
passes through the other candidates. The reference runs 200 four-spine
blocks with the C57 plan cache. The selected configuration was fixed
using seeds 2111–2118 before confirmation.

### Matched full-budget inputs

The 16 fresh sets of full-budget candidates use seeds 2151–2166. Both
arms started from identical saved spins for each seed. Each timed arm
ran twice in ABBA or BAAB order on the same pinned CPU. The timed stage
includes graph and plan preparation. Generation, loading saved reads,
and annealing are outside this stage timer.

- The median paired polishing-stage speedup was **1.821×**, equivalent
  to **45.1% less stage time**. The paired bootstrap 95% median interval
  was 1.788–2.018×. All 16 pairs ran faster, with ratios from 1.503× to
  2.679×. Per-arm stage medians were 0.913 s and 0.499 s.
- The best canonical energy was lower on six instances and tied on
  ten. Gains ranged from 0 to 2,090 milli units. The median gain across
  all 16 instances was zero.
- The 50-block ablation tied the 200-block reference for best energy on
  all 16 inputs. The extra best-energy improvements came from the
  motif pass. Across the 256 sorted candidate ranks, 20 improved, 232
  tied, and four worsened. The largest rank-level increase was 20
  milli units.

### Separate full solves

The independent full-solve cohort used 12 fresh seeds, 2191–2202.
Each repetition started a fresh 16 × 8,192 anneal and continued through
the chosen polish. Each pair ran twice per arm on the same core in
ABBA or BAAB order. The stage was **1.846×** as fast at the median
(95% interval 1.603–2.083×). Best energy improved on four instances
and tied on eight, with a maximum gain of 830 milli units.

The complete offline solve, including the anneal, had a median ratio
of 1.028× with a 95% interval of 0.9996–1.0627×. Whole-solver timing
remains inconclusive.

## C60. Reuse an unchanged batch's conditional optimum

The exact Zephyr step leaves the current batch optimal for that
neighborhood. The strict downhill rule also leaves the spins in place
when the minimum ties the current energy. Each batch of up to eight
reads records its spin version after a block is solved. A later move
invalidates the record by advancing that version. A repeated visit on
the same version can skip the batch evaluation. When every batch
can skip, the pass also skips loading or rebuilding the plan for that
visit. The optimizer keeps its scheduled block order for every active
batch.

On the 16 saved starts from the C59 confirmation, reuse retained all
200 scheduled blocks and skipped a median of 222 out of 400 batch
evaluations. All 16 pairs matched ordered spins, canonical energies,
and accepted-move counts. The median stage speedup was **1.983×**
(95% interval 1.730–2.299×). All 16 pairs ran faster. The median stage
times were 0.977 s for the C57 reference and 0.473 s with reuse.

Running the motif finish after the full 200-block schedule with reuse
gave a **1.305×** median stage speedup over C57 (95% interval
1.112–1.493×). It lowered 20 of the 256 individual read energies and
tied the other 236. The best energy improved on six instances and
tied on ten, matching the shorter recipe's best scores on these
same inputs. Both arms ran twice per seed in a rotated, mirrored
four-arm order on the same pinned core.

[C59 fixed-input pairs](data/polish-followup/c59-fixed-inputs.jsonl) ·
[C59 full-solve pairs](data/polish-followup/c59-full-solve.jsonl) ·
[C60 pairs](data/polish-followup/c60-reuse.jsonl) ·
[Summary and executable provenance](data/polish-followup/summary.json)

The provenance file pins the source commits for the measured executables
and the retained build. The raw commands use relative names for benchmark
binaries and saved checkpoints. [Recorded source revisions](https://github.com/NSEVEjk0/quip-solver-cpu/tree/cpu-native-polish-study)

## Validation

The retained build returned byte-identical ordered candidates with and
without reuse on all 16 warm batches, four partial batches of 13 reads,
and four random-start batches. It also replayed both motif recipes on
all 16 warm inputs. All 468 enabled release library tests passed, with
one pre-existing ignored test. Formatting, strict Clippy, the default
feature check, and native protocol conformance with both default and
opt-in settings passed.

## Reproduce

Build with Rust 1.97.1. Run each arm on the same pinned CPU and alternate
the order. Each command emits canonical energies, ordered spin and
energy digests, and separate timings for the Zephyr and motif stages.

```sh
quip_features=experimental,masked-residual-deltas,table-tile-bonds,linked-component-lists,shallow-union-roots,packed-residual-fields,pair-worm-polish,zephyr-spine-polish,tree-polish,motif-descent
cargo build --release --locked --features "$quip_features" --example cpu_native_bench
quip_bench=target/release/examples/cpu_native_bench
quip_run=$(mktemp -d)
quip_reads="$quip_run/2152-reads.json"

"$quip_bench" --mode pipeline --preset archive --pair-worm-polish \
  --seed 2152 --reads 16 --sweeps 8192 --save-reads "$quip_reads"
"$quip_bench" --mode pipeline --preset archive --seed 2152 --reads 16 --sweeps 8192 \
  --load-reads "$quip_reads" \
  --spine-blocks 200 --spine-span 4 --spine-batch --spine-cache-plans
"$quip_bench" --mode pipeline --preset archive --seed 2152 --reads 16 --sweeps 8192 \
  --load-reads "$quip_reads" \
  --spine-blocks 50 --spine-span 4 --spine-batch --spine-cache-plans \
  --motif-rounds 32 --motif-mixtures 4
"$quip_bench" --mode pipeline --preset archive --seed 2152 --reads 16 --sweeps 8192 \
  --load-reads "$quip_reads" \
  --spine-blocks 200 --spine-span 4 --spine-batch --spine-cache-plans --spine-skip-unchanged
```

For separate full solves, omit `--load-reads` and pass
`--pair-worm-polish` in each arm. Add `--motif-rounds 32 --motif-mixtures 4`
to the 200-block reuse command for the combined pass. For a native
solver build, set these top-level backend TOML keys to select the
corresponding stages.

| Setting | Value for the recorded configurations |
|---|---|
| `native_pair_worm_polish` | `true` |
| `native_zephyr_blocks` | `50` for C59 or `200` for C60 |
| `native_zephyr_skip_unchanged` | `true` for exact block reuse |
| `native_motif_rounds` | `32` for motif descent |
| `native_motif_mixtures` | `4` for recombination |
