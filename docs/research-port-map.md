# Research port map

This map compares one research fork of this crate with production. It lists every algorithm difference by file and names the code that this crate keeps. The CPU-1 ticket owns this map.

## Compared trees

| Tree | Location | Revision |
| --- | --- | --- |
| Research fork | `new-pow/ice-twisted-w62-dimer1200/cpu` under the research root | `quip-miner-cpu` 0.3.4-rc3. The BLAKE3 of its `Cargo.toml` is `f3fd76205388ff25`, the research inventory hash in `isingmark/problems/sources.json`. |
| Production | this repository | `af9625f` on `main`, version 0.3.5 |

The fork is not a Git checkout. For each fork file, the comparison renames the crate from `quip_miner_cpu` to `quip_solver_cpu` and then finds the closest production revision of the same path. Hashes are the first 16 hex digits of BLAKE3.

## Flatiron

The fork has no Flatiron algorithm change. Production keeps its own Flatiron.

| File | Fork | Production `af9625f` | Finding |
| --- | --- | --- | --- |
| `src/flatiron_sampler.rs` | `01011eabb4601e6e` | `5dbe17659a843566` | The fork equals production `19f3f4d`. Production later added the `linear-schedule` feature list and the `F64_LIMITS` limits to `CPU_FLATIRON_IDENTITY`. |
| `src/flatiron/mod.rs` | `c2f4b34917938d7b` | `fb782e356510868b` | The fork equals production `54d1282`. Only a doc-test crate name differs. |
| `src/flatiron/graph.rs` | `35fe5a74934b9927` | `35fe5a74934b9927` | Identical |
| `src/flatiron/sample.rs` | `a906e87047e96be7` | `a906e87047e96be7` | Identical |
| `src/flatiron/state.rs` | `2a8cea7da77baa9d` | `2a8cea7da77baa9d` | Identical |

## Other inventoried files

| File | Fork | Closest production revision | Fork-only difference | Decision |
| --- | --- | --- | --- | --- |
| `src/sampler_core.rs` | `7ed1a2059db029ef` | equal to `19f3f4d` | None | Keep production. |
| `src/sa_int.rs` | `548d8bfb41972d68` | equal to `ff21e3e` | None | Keep production. |
| `src/sa_msc_f64.rs` | `61f7179dd5924c72` | equal to `93fd67e` | None | Keep production. |
| `src/coloring.rs` | `cb981b5f8616eeb7` | equal to `af9625f` | None | Keep production. |
| `src/research.rs` | `1a7a3fef66c473ab` | `93fd67e`, 2 lines | `validate_common` is `pub(crate)`, so the research dimer modules can call it. | Keep production. CPU-2 changes the visibility only if its port calls the function. |
| `src/sa_sampler.rs` | `cbe4f31d13dce6ee` | `14a2e7e`, 2 lines | None. The fork writes one loop with `as_chunks::<8192>()` where `14a2e7e` has `chunks_exact(8192)`. Production made the same edit in `ff21e3e` and later replaced the loop in `137eda5`. | Keep production. |
| `src/lib.rs` | not hashed | `14a2e7e`, 3 lines | Declares `dimer_fractional`, `dimer_research`, and `dimer_weighted`. | Not ported here. CPU-2 owns the weighted-dimer sampler. |
| `src/sa_msc.rs` | `f7fcdde37a86cf70` | `ff21e3e`, 330 lines | Adds pair and cluster moves to the multi-spin kernel: `anneal_words_with_clusters`, `flip_mask`, a cluster builder, and four tests. Only `src/dimer_research.rs` calls them. | Not ported here. CPU-2 ports them only if its selected variant needs them. Production kept its later lane changes `137eda5` and `c297965`. |

The fork-only files `src/dimer_fractional.rs`, `src/dimer_research.rs`, and `src/dimer_weighted.rs` belong to CPU-2.

## Capabilities

Production binaries declare the generic tier of `quip-solver-core` 0.0.5. They list no `solver_id` and no generated or accepted family, because none of them links a canonical family generator. Their limits come from each `BackendIdentity`. `tests/family_capabilities.rs` checks that the command line, the session `Hello`, and `GetCapabilities` agree for `quip-cpu-sa`, `quip-cpu-gibbs`, `quip-cpu-msa`, and `quip-cpu-flatiron`. It also runs the family gate against each one.
