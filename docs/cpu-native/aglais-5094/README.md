# -5094 on CPU

A cold CPU solve reached **−5093.665** on the nonce from finalized Aglais
block 175503. C64 motif polishing took the same CPU-generated checkpoint
to **−5094.135**. Both compact witnesses decode to the stated energies
under the canonical instance. [Cold witness](data/cold-witness.json) ·
[Polished witness](data/polished-witness.json) · [Saved reads](data/replay-inputs.tar.gz)

## Cold run and polish

The fixture is `quip/twisted-ice-dimer` revision 1 with 4,560 spins and
41,205 edges. The benchmark regenerated the instance from the recorded
nonce and canonical family parameters. The anneal initialized its replicas
from the recorded RNG seed. The cold recipe used 16 replicas × 131,072
sweeps, pair-aware worm polish, 200 Zephyr blocks, and the C60 motif finish.
It took **386.59 CPU seconds** on one pinned AMD EPYC 7663 vCPU.
[Fixture and build manifest](data/manifest.json) · [Raw cold runs](data/cold-runs.jsonl)

The C64 comparison replays the same 16 saved reads for all three arms.
The 2,048-round pass with pairing release lowered strong-coupling energy
by 2.000 and raised weak-coupling energy by 1.530 for a net canonical
gain of **0.470**. The reference and strict-pairing arms preserved the
complete ordered spin-and-energy digest. [Raw polish rows](data/polish-runs.json) ·
[C64 method](https://github.com/QuipNetwork/quip-solver-cpu/pull/4)

| Stage | Best canonical energy | Integer milli-units |
|---|---:|---:|
| Cold CPU solve with C60 finish | −5093.665 | −5,093,665 |
| Strict C64 on the cold checkpoint | −5093.665 | −5,093,665 |
| C64 with pairing release on the cold checkpoint | **−5094.135** | **−5,094,135** |
| Archived Aglais winner on the same nonce | −5096.825 | −5,096,825 |

## Cold-run context

The follow-up made nine cold runs on three selected, previously solved
Aglais nonces. The benchmark reached the −5091 cutoff on one run. On
block 175503, the earlier 131,072-sweep seed reached −5088.915. The same
earlier seed reached −5086.455 at 262,144 sweeps. Block 186335 was selected
for having the best original C60 score in the separate 12-case replay. Its
131,072-sweep cold rerun reached −5088.285. The archived winning witnesses
serve as same-instance energy references for the offline runs.
[All nine rows](data/cold-runs.jsonl) · [Selected block provenance](data/manifest.json)

## Verify the archive

From the repository root with Python 3.10 or later, run the independent
integer scorer. It checks the archive and recomputes all 192 recorded read
scores using the stored original coefficients. The benchmark commands
below regenerate the instances from the nonce and verify the compact
witnesses through the canonical family decoder.

```sh
python3 tools/cpu-native/audit_aglais_5094.py
```

## Build the pinned benchmarks

Run from the repository root on Linux. Choose an available core for
`quip_cpu`. The measured executable hashes and source revisions are in
the manifest. [Cold source](https://github.com/NSEVEjk0/quip-solver-cpu/tree/925e8d290975ab7df54de6b2f6e91179cb6dfc27) ·
[Polish source](https://github.com/NSEVEjk0/quip-solver-cpu/tree/8bc0042812042dbdbb5bda7c0d231bdb90024c0f) ·
[Periodic observation diagnostic source](https://github.com/NSEVEjk0/quip-solver-cpu/tree/8136968ce564e31b7cbb8b40097b024ecaa66ec0)

```sh
quip_data="$PWD/docs/cpu-native/aglais-5094/data"
quip_run=$(mktemp -d)
quip_cpu=0
quip_nonce=dfbc9c59a8401206b5c1470635e8af79a814856bd5c463148da7cf7b95aa1bd8
quip_features=experimental,masked-residual-deltas,table-tile-bonds,linked-component-lists,shallow-union-roots,packed-residual-fields,pair-worm-polish,zephyr-spine-polish,tree-polish,motif-descent

git fetch https://github.com/NSEVEjk0/quip-solver-cpu.git cpu-aglais-5094-cold cpu-aglais-5094-polish
git worktree add --detach "$quip_run/cold" 925e8d290975ab7df54de6b2f6e91179cb6dfc27
git worktree add --detach "$quip_run/polish" 8bc0042812042dbdbb5bda7c0d231bdb90024c0f
for quip_tree in "$quip_run/cold" "$quip_run/polish"; do
  (
    cd "$quip_tree"
    cargo +1.97.1 build --release --locked --features "$quip_features" --example cpu_native_bench
  )
done
python3 tools/cpu-native/audit_aglais_5094.py --cold-worktree "$quip_run/cold"
```

## Replay the polish from the saved cold checkpoint

```sh
mkdir "$quip_run/inputs"
tar -xzf "$quip_data/replay-inputs.tar.gz" -C "$quip_run/inputs" cold/target-131072-reroll-175503.json
(
  cd "$quip_run/polish"
  taskset -c "$quip_cpu" target/release/examples/cpu_native_bench \
    --mode pipeline --preset archive --nonce "$quip_nonce" --seed 4299707 \
    --reads 16 --sweeps 8192 --load-reads "$quip_run/inputs/cold/target-131072-reroll-175503.json" \
    --pairing-rounds 2048 --pairing-mode exact-light --save-reads "$quip_run/polished-reads.json"
) > "$quip_run/polished.json"
python3 tools/cpu-native/audit_aglais_5094.py --polish-output "$quip_run/polished.json"
```

## Replay the cold solve

This command regenerates the randomized initial population. It took
about six and a half minutes of CPU time on the measured host.

```sh
(
  cd "$quip_run/cold"
  taskset -c "$quip_cpu" target/release/examples/cpu_native_bench \
    --mode pipeline --preset archive --nonce "$quip_nonce" --seed 90457687056835 \
    --reads 16 --sweeps 131072 --pair-worm-polish \
    --spine-blocks 200 --spine-span 4 --spine-batch --spine-cache-plans --spine-skip-unchanged \
    --motif-rounds 32 --motif-mixtures 4 --save-reads "$quip_run/cold-reads.json"
) > "$quip_run/cold.json"
python3 tools/cpu-native/audit_aglais_5094.py --cold-output "$quip_run/cold.json"
```
