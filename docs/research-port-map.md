# Research port map

This map compares one research fork of this crate with production. It lists every algorithm difference by file and function, and it names the code that this crate keeps. The CPU-1 ticket owns this map.

## Compared trees

| Tree | Location | Revision |
| --- | --- | --- |
| Research fork | `new-pow/ice-twisted-w62-dimer1200/cpu` under the research root | `quip-miner-cpu` 0.3.4-rc3. The BLAKE3 of its `Cargo.toml` is `f3fd76205388ff25`, the research inventory hash in `isingmark/problems/sources.json`. |
| Production | this repository | `af9625f` on `main`, version 0.3.5 |

The fork is not a Git checkout. For each fork file, the comparison renames the crate from `quip_miner_cpu` to `quip_solver_cpu` when the production revision is after the rename in `14a2e7e`. It then finds the closest production revision of the same path. Hashes are the first 16 hex digits of BLAKE3.

Each file has two kinds of difference from `af9625f`:

- A fork-only difference is a change that the fork made and production never made. The diff between the fork and the closest production revision shows it.
- A later production change is a commit after that revision. The fork does not have it, and production keeps it.

The binaries under `src/bin/` differ from production only in the order of two `use` lines.

## Flatiron

The fork has no Flatiron algorithm change. Production keeps its own Flatiron.

| File | Fork | Production `af9625f` | Closest production revision | Later production changes |
| --- | --- | --- | --- | --- |
| `src/flatiron_sampler.rs` | `01011eabb4601e6e` | `5dbe17659a843566` | equal to `19f3f4d` | `4ec7874` adds the `linear-schedule` feature list and the `F64_LIMITS` limits to `CPU_FLATIRON_IDENTITY`. |
| `src/flatiron/mod.rs` | `c2f4b34917938d7b` | `fb782e356510868b` | equal to `54d1282`, apart from a doc-test crate name | None |
| `src/flatiron/graph.rs` | `35fe5a74934b9927` | `35fe5a74934b9927` | Identical | None |
| `src/flatiron/sample.rs` | `a906e87047e96be7` | `a906e87047e96be7` | Identical | None |
| `src/flatiron/state.rs` | `2a8cea7da77baa9d` | `2a8cea7da77baa9d` | Identical | None |
| `src/mps/schedule.rs` | not hashed | not hashed | equal to `19f3f4d` | `8d6cc48` makes `build`, which Flatiron's `anneal` calls, follow `SampleParams::schedule`. The fork always builds a geometric ladder. Production builds a geometric ladder by default and a linear ladder on request. |

## Other inventoried files

| File | Closest production revision | Fork-only difference | Later production changes | Decision |
| --- | --- | --- | --- | --- |
| `src/sampler_core.rs` | equal to `19f3f4d` | None | `8d6cc48`: `build_seeded_beta_schedule` and `effective_sweeps_per_beta` honour a linear ladder. `d5b55d9`: `sample_sa_scalar` adds bench spans and an accept counter, with the same samples. | Keep production. |
| `src/sa_int.rs` | equal to `ff21e3e` | None | `137eda5`: `fill_threshold_row` splits into `fill_cuts` and `draw_threshold`, so each unit MSA lane draws its own threshold, and `draw_row` goes away. | Keep production. |
| `src/sa_msc.rs` | `ff21e3e`, 330 lines | Two changes. See the next table. | `137eda5`: `anneal_words` gives each lane its own acceptance threshold. The fork shares one threshold row across lanes. `c297965` picks the lane row width. | Keep production. |
| `src/sa_msc_f64.rs` | equal to `93fd67e` | None | `88762c2`: `anneal` and `flip_mask` draw acceptance per lane from `lane_stream`. The fork shares one draw per node visit across every lane (`shared-node-threshold-v1`). `03a803a` raises `WORKSPACE_CAP_BYTES` from 256 MiB to 8 GiB. `12557da` adds `check_size`. | Keep production. |
| `src/research.rs` | `93fd67e`, 2 lines | `validate_common` is `pub(crate)`, so the research dimer modules can call it. | `12557da` adds `check_research_counts`. `88762c2` and `137eda5` follow the kernel changes. | Keep production. CPU-2 changes the visibility only if its port calls the function. |
| `src/sa_sampler.rs` | `14a2e7e`, 2 lines | None. The fork writes one loop with `as_chunks::<8192>()` where `14a2e7e` has `chunks_exact(8192)`. Production made the same edit in `ff21e3e`. | `137eda5` replaces that loop. `4ec7874` adds identity features and limits. `12557da` refuses requests over the caps. | Keep production. |
| `src/mps_sampler.rs` | equal to `19f3f4d` | None | `4ec7874` adds `linear-schedule` and `F64_LIMITS` to the MPS and MFA identities. | Keep production. |
| `src/sb_sampler.rs` | equal to `19f3f4d` | None | `4ec7874` adds `SB_LIMITS`. `03a803a` raises the node and edge caps. | Keep production. |
| `src/lib.rs` | `14a2e7e`, 3 lines | Declares `dimer_fractional`, `dimer_research`, and `dimer_weighted`. | `0688a86` adds the `QUIP_CPU_STREAM_WIDTH` override. `4ec7874` adds `LINEAR_ONLY` and `F64_LIMITS`. `03a803a` and `12557da` raise and enforce the caps. | Not ported here. CPU-2 owns the weighted-dimer sampler. |

### Fork-only changes in `src/sa_msc.rs`

| Change | Functions | Effect | Decision |
| --- | --- | --- | --- |
| Pair and cluster moves | `anneal_words_with_clusters`, `ClusterMove::build`, `ClusterMove::flip_mask`, and five tests | Only `src/dimer_research.rs` calls them. | Not ported here. CPU-2 ports them only if its selected variant needs them. |
| Wider bond counter | `PLANES` from 6 to 7, so `MAX_DEGREE` from 63 to 127. The popcount and `bond_counts` return seven planes. | This change also affects ordinary sampling. A node of degree 64 to 127, counting its field, runs on the packed unit MSA kernel in the fork. Production refuses the packed kernel for that model and runs the scalar fixed-point SA (FSA) kernel. A zero-field star of 65 nodes with 64 unit couplings takes the packed path in the fork and the scalar path in production. | Not ported here. Production keeps 6 planes. CPU-2 widens the counter only if a selected variant needs a degree over 63 on the packed kernel. |

The fork-only files `src/dimer_fractional.rs`, `src/dimer_research.rs`, and `src/dimer_weighted.rs` belong to CPU-2.

## Capabilities

Production binaries declare the generic tier of `quip-solver-core` 0.0.5. They list no `solver_id` and no generated or accepted family, because none of them links a canonical family generator. Their limits come from each `BackendIdentity`.

`tests/family_capabilities.rs` covers `quip-cpu-sa`, `quip-cpu-gibbs`, `quip-cpu-msa`, and `quip-cpu-flatiron`:

- The command line must print the expected algorithm and limits.
- The session `Hello` and `GetCapabilities` must each equal the command-line output, field for field.
- The family gate must grade each binary conformant.
