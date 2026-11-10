# quip-solver-cpu

CPU Ising miners for the [quip.network](https://gitlab.com/quip.network) v0.3
mining protocol: simulated annealing (`quip-cpu-sa`), single-site chromatic
Gibbs (`quip-cpu-gibbs`), and discrete Simulated Bifurcation (`quip-cpu-sb`),
shipped as separate binaries.

The sampler runs one model per core (model-level parallelism); each model's
reads are sequential and cache-local. Energies are scored with the canonical
`quip_protocol::scoring::energy_milli` so results match consensus.

## Binaries

| binary | algorithm | track |
|--------|-----------|-------|
| `quip-cpu-sa` | simulated annealing (Metropolis) | production |
| `quip-cpu-gibbs` | single-site heat-bath Gibbs | production |
| `quip-cpu-sb` | discrete Simulated Bifurcation | production |
| `quip-cpu-bench` | per-part timing of the scalar SA kernel (see [Benchmarking](#benchmarking)) | production |
| `quip-cpu-bsb` | ballistic Simulated Bifurcation | experimental |
| `quip-cpu-hdsb` | heated discrete Simulated Bifurcation | experimental |
| `quip-cpu-hbsb` | heated ballistic Simulated Bifurcation | experimental |
| `quip-cpu-gbsb` | generalized ballistic Simulated Bifurcation, edge-of-chaos control | experimental |
| `quip-cpu-gdsb` | edge-of-chaos control on the discrete coupling | experimental |
| `quip-cpu-tedsb` | tabu-enhanced discrete Simulated Bifurcation | experimental |
| `quip-cpu-sbqa` | replica-ring discrete Simulated Bifurcation | experimental |
| `quip-cpu-ggdsb` | globally guided discrete Simulated Bifurcation | experimental |
| `quip-cpu-fsa` | simulated annealing with tabulated Metropolis thresholds | experimental |
| `quip-cpu-msa` | multi-spin coded simulated annealing, 64 reads per word | experimental |
| `quip-cpu-mps` | tensor network: imaginary-time TEBD with exact sampling | experimental |
| `quip-cpu-mfa` | mean-field annealing (the same kernel at bond dimension 1) | experimental |
| `quip-cpu-flatiron` | belief-propagation tensor network on the problem graph | experimental |
| `quip-cpu-otsuka` | Otsuka cluster annealing on `quip/twisted-ice-dimer` jobs, simulated annealing on every other job | experimental |
| `quip-cpu-ice-sat` | Strong-layer SAT witnesses from an external Kissat on `quip/twisted-ice-dimer` jobs, simulated annealing on every other job | experimental |
| `quip-cpu-greedy` | exact greedy descent from supplied initial states, a postprocess | experimental |
| `quip-cpu-pairworm` | pair-aware worm schedule on supplied states of the selected `quip/twisted-ice-dimer` recipes, greedy descent on every other job, a postprocess | experimental |
| `quip-cpu-dimer` | weighted-dimer anneal with exact tile blocks and ice loops on `quip/twisted-ice-dimer` jobs, simulated annealing on every other job | experimental |
| `quip-cpu-flow` | exact min-cost circulation on `quip/flow-static` jobs and compatible-charge construction on `quip/polariton-static` jobs, every other job declined | experimental |
| `quip-cpu-tileworm` | tile worm anneal on supplied states of the selected `quip/charged-ice` and `quip/ice-fields` recipes, greedy descent on every other job, a postprocess | experimental |
| `quip-cpu-gadget-repair` | gadget repair of supplied states of `quip/two-commodity-routing` jobs, greedy descent on every other job, a postprocess | experimental |
| `quip-cpu-routing` | alternating exact flow responses on `quip/two-commodity-routing` jobs, declines every other job | experimental |
| `quip-cpu-flow-loops` | charge-preserving loop search on `quip/flow-static` and `quip/polariton-static` jobs, from supplied states or the exact baseline, every other job declined | experimental |
| `quip-cpu-dipolar` | string and loop search on `quip/defects-static` jobs, from supplied states when the job has them, greedy descent on every other job | experimental |

The MSA miner uses color-ordered sweeps with 64 replicas per word. Its adaptive
budget matches CUDA MSA: 7,392 to 29,568 sweeps and exactly 128 reads.
See [SA acceleration](docs/sa-acceleration.md) for the kernel and benchmark details.

Every sampler streams jobs through one shared pump, `run_job_pump` in
`src/lib.rs`, which `run_stream_pump` wraps for plain jobs. That pump drops a cancelled generation before a worker touches
the graph. The SA kernel also polls the cancel guard once per sweep (one
`Relaxed` load, never per spin flip). A long in-flight job of 80 to 300
seconds then stops at the next sweep. The kernel emits
`StreamOutcome::Cancelled`, the same outcome as the dequeue path. The SB
kernel and its sampler live in `src/sb_core.rs` and `src/sb_sampler.rs`,
separate from the annealing path. `docs/sb-variants.md` gives the method,
the constants, and the paper behind each Simulated Bifurcation kernel, and
names every place the code departs from its paper.

Prebuilt binaries are attached to each
[Release](https://gitlab.com/quip.network/quip-solver-cpu/-/releases) for
`linux-amd64`, `linux-arm64`, and `darwin-arm64`. Asset names carry the
operating system as well as the architecture, because an architecture alone
cannot separate a Linux aarch64 binary from a Darwin arm64 one.

Every binary in the table above is published, production and experimental
alike. Each release lists the two tracks in its notes. Asset names follow the
same `<binary>-<os>-<arch>` pattern on both tracks and carry no marker, so the
file alone does not say which track it came from. Read the track column above
before you deploy one.

## Build

```sh
cargo build --release
```

A plain build produces the four production binaries. The experimental
binaries need an opt-in feature:

```sh
cargo build --release --features experimental
```

`quip-cpu-mps` chooses its bond dimension per job from a deterministic flop
budget, a 64 MB per-model memory cap, and a ceiling of 32. On graphs as wide as
`advantage2-system1` that resolves to 1, where the algorithm is mean-field
annealing rather than a tensor network. The binary degrades instead of
rejecting, because the coordinator defaults to that preset.

Set `QUIP_MPS_INIT=random` to replace the anneal with uniform random starting
configurations. Both settings share the same sampler and the same greedy polish,
which is what makes the two arms comparable.

### H3 experiment: Annealed against random initialization

`quip-cpu-mfa` reads one environment variable, `QUIP_MPS_INIT`, that switches
its starting state before the greedy polish stage. The default value, `anneal`,
lowers a transverse field through imaginary-time evolution and samples the
annealed state. The alternative value, `random`, skips the anneal and samples a
uniform random product state instead. Both values share the same sampler and the
same polish stage, so a comparison between them isolates the effect of the
seeding strategy alone.

The commands below need an `isingmark` that accepts `--backend cpu-mfa`.
`isingmark` reads that flag against a fixed built-in list, so an older build
rejects the name before it starts a run. Build the two experimental binaries
onto `PATH` first, as the Build section describes.

Hypothesis H3 asks whether the annealed seed reaches a better result than the
random seed at equal wall-clock cost. Run the two arms with the same seed and
the same single-cell parameter grid, so `isingmark` draws the same 200 problems
for both arms and the results pair by `job_id`:

```sh
# Anneal arm (default): QUIP_MPS_INIT unset.
isingmark sweep --backend cpu-mfa --topology-preset advantage2-system1 \
  --param-grid '{"num_reads":[64],"num_sweeps":[320]}' \
  --num-jobs 200 --seed 2000 --output-dir out/h3/anneal

# Random arm: QUIP_MPS_INIT=random, same grid cell, same seed.
QUIP_MPS_INIT=random isingmark sweep --backend cpu-mfa \
  --topology-preset advantage2-system1 \
  --param-grid '{"num_reads":[64],"num_sweeps":[320]}' \
  --num-jobs 200 --seed 2000 --output-dir out/h3/random
```

Repeat both commands with `--topology-preset smoke` and an
`out/h3/<arm>-smoke` output directory. The design requires that second pair as
its stop criterion.

The gap depends on the linear biases. Mean-field cannot break a symmetry on its
own. On a zero-bias instance, every coupling gate is symmetric under the global
spin flip. The annealed state stays at the symmetric point. A sample drawn there
is a fair coin per site, which makes the two arms one measurement. Biases remove
the degeneracy. H3 reports the separation that appears once they do.

The accepted-rate and paired best-energy comparison that decide H3 run through
the campaign analysis script rather than through this repository.

The Quip solver contract (`quip-proto`, `quip-protocol`, `quip-solver-core`) is
published from crates.io.

Every binary except `quip-cpu-otsuka`, `quip-cpu-ice-sat`, `quip-cpu-greedy`, `quip-cpu-pairworm`, `quip-cpu-dimer`, `quip-cpu-flow`, `quip-cpu-tileworm`, `quip-cpu-gadget-repair`, `quip-cpu-routing`, `quip-cpu-flow-loops`, and `quip-cpu-dipolar` declares the generic family tier: no
`solver_id` and no generated or accepted problem family, because none links a
canonical family generator. `tests/family_capabilities.rs` checks that
`--capabilities` and the session agree.

`quip-cpu-otsuka` generates and accepts `quip/twisted-ice-dimer` revision 1.
`tests/family_conformance.rs` tests its declaration. [Research port map](docs/research-port-map.md) lists how the research
fork of this crate differs from production and which code this crate keeps.

`families::otsuka` is the Otsuka cluster kernel for gauged-ice instances with
tile and pair motifs, ported from the research `fast_bench.rs`. It refuses
fields and supports soft pair and tile weights. `tests/otsuka_kernel.rs`
checks it against exact energies and the exact Boltzmann distribution of small
fixtures.

`quip-cpu-otsuka` runs that kernel behind the solver session. Its `solver_id`
is `quip-cpu-otsuka/1`, and it generates and accepts `quip/twisted-ice-dimer`
revision 1. The session validates each family instance before the sampler sees
it. The kernel then reads the tiles, gauges, and pairs from the instance and
anneals with the job's beta ladder, one cluster sweep per sweep. Every other
job, and every family job that does not fit the kernel, runs on the SA kernel.
`tests/family_conformance.rs` runs the family gate and the plain-job
conformance against the binary.

`quip-cpu-ice-sat` is a baseline for the strong layer of
`quip/twisted-ice-dimer` revision 1. Its `solver_id` is `quip-cpu-ice-sat/1`.
`families::ice_sat` writes each gauged tile as 10 clauses that allow exactly
two of its four gauged spins at +1, and each pair as 2 clauses. An external
Kissat executable solves the CNF. Each read runs Kissat with a 31-bit seed
from `seed + r` and returns the decoded witness, with every unconstrained spin
at +1, scored on the full instance. The sampler checks every tile and pair
again without the CNF. A strong-layer witness ignores the weak layer, so its
energy is not a ground-state claim.

Set the executable with the `backend_toml` key `kissat` (an absolute path) or
the `--kissat` flag, and the time limit of one run with `kissat_timeout_s` or
`--kissat-timeout-s` (default 600 s). Kissat stops itself at that limit, and
the sampler kills the run at most 30 s later. The `backend_toml` keys win. A family
job with no checked witness gets a `Reject` with `NO_SOLUTION`. The detail
in the log names the cause, such as an UNSAT verdict. Every
other job runs on the SA kernel. `tests/ice_sat.rs` covers
the encoding exhaustively on small fixtures and the process boundary with fake
solvers. `tests/ice_sat_conformance.rs` runs the family gate against a real
Kissat when `QUIP_KISSAT` names one and the test runs with `--ignored`.

`quip-cpu-greedy` is a postprocess. Its `solver_id` is `quip-cpu-greedy/1`
and its policy is `greedy-nat/1`. It advertises `initial-spins` and accepts
`quip/twisted-ice-dimer` revision 1 jobs for their structure. It generates no
family work. `postprocess::greedy` ports the research greedy descent exactly.
Each supplied state becomes one read. A sweep visits every single flip, then
every natural move in sorted order, and applies each move that lowers the
energy. The descent stops after a sweep that applies none. The natural moves
are every coupled pair and every rigid group. A rigid group is a set of nodes
that ferromagnetic couplings of strength 0.5 or more join outside the tiles,
together with the pairs. A twisted-ice-dimer job also adds its pairs and the
three-node and four-node subsets of its tiles.

A job without initial states descends from random states drawn from its seed.
The binary then logs a warning, because it should receive only warm jobs.
Each job logs one line with
the policy and the move counts by source. The line also records the applied
moves and the time, and the best energy before and after the descent.

`tests/postprocess_greedy.rs` checks every move delta and every descent of a
6-node fixture against energies computed from scratch.
`tests/greedy_conformance.rs` runs the session conformance, which grades the
warm-start proof. The ignored test `research_parity` compares the final
states, energies, and applied moves with the research descent:

```sh
QUIP_RESEARCH_ROOT=/path/to/regimes python3 -I research/greedy_parity/export.py OUT
QUIP_GREEDY_PARITY=OUT cargo test --release --features experimental \
    --test postprocess_greedy -- --ignored research_parity
```

`quip-cpu-pairworm` is a postprocess for the three selected twisted-ice
controls. Its `solver_id` is `quip-cpu-pairworm/1`, its algorithm is
`pair-worm`, and its policy is `ice-pairs-pw/1`. It advertises
`initial-spins` and accepts `quip/twisted-ice-dimer` revision 1 jobs with the
recipes `x-w62-p1200`, `x-w62-p1350`, and `x-w45-p1200`. It generates no
family work. Every other job runs the `greedy-nat/1` descent.

`postprocess::pair_worm` ports the research `pairworm.c` and the `pw` schedule
of `run.py` with default options. A worm flips both ends of a planted pair as
one unit and walks until every tile it touched is neutral again. The schedule
takes the supplied states in order of energy, lowest first. Each state runs
the greedy descent alternated with the strict worm. The lowest-energy result
then takes improving blocks from the other results. Anneal cycles of the worm
at 250 milli use the rest of the budget.

The declared budget is 0.76 s per job. The binary measures elapsed time on the
thread that runs the job. The research measured process CPU time on one core.
The clock decides how many states and anneal cycles run, so two hosts may give
different results for one timed job. The result lists the final state first
and then the descended state of each visited read. The sampler rescores every
state from the job graph. The worm draws its random numbers from the job seed.
The log line of each job records the policy and the declared budget next to
the measured time and the work done.

`tests/postprocess_pairs.rs` checks the tile preparation, every worm mode,
crossover, the defect counts, and the schedule with fixed work against
vectors from the research code in `tests/fixtures/pair_worm_vectors.json`.
`research/pairworm_parity/export.py vectors` writes that file. The ignored
test `research_parity` replays the saved research runs with their recorded
work and compares the final states:

```sh
QUIP_RESEARCH_ROOT=/path/to/regimes python3 -I research/pairworm_parity/export.py research OUT
QUIP_PAIRWORM_PARITY=OUT cargo test --release --features experimental \
    --test postprocess_pairs -- --ignored research_parity
```

`quip-cpu-dimer` runs the bounded weighted-dimer kernel. Its `solver_id` is
`quip-cpu-dimer/1` and its algorithm is `weighted-dimer`. It generates and
accepts `quip/twisted-ice-dimer` revision 1. The claim names the whole
revision, because the research did not select the weighted kernel for one
recipe. Every other job, such as a plain Ising job, runs on the SA kernel.

`families::dimer_weighted` ports the research `dimer_weighted.rs` and its
`ice.rs`. The binary runs the research variant `ice-loop64-weighted`. The
kernel keeps exact integer milliweights and a cached field for each of 64 read
lanes. Each sweep does three things. It updates every spin once. It then flips
each tetrahedron as one block and picks the exact best state of the block. It
last makes 64 attempts per read to flip an alternating loop or string of the
ice layer. The kernel ports `ice-loop-weighted` and `ice-defect64-weighted` as
well, but the binary does not run them. The path-block, contraction, polish,
and tempering variants of the research are not ported. The fork-only cluster
moves and wide bond counter of `sa_msc.rs` are not needed, because this kernel
keeps its own integer fields.

The sampler checks the job before the first sweep and rejects it with
`Declined` when any check fails. It rejects a job with a field, a coupling that
is not an exact milliweight, or a total weight over the `i32` safety bound. It
rejects a tile that is not a four-spin block with six distinct 1000-milli bonds
forming a local gauge of antiferromagnetic ice. It rejects a spin in more than
two tiles. It rejects a pair that is not an edge of coupling minus 1000 milli
between two spins that no other pair uses. A block may have two through eight
spins. The session refuses charged-ice and field-biased-ice jobs before the
kernel sees them, because this binary does not accept those families.

The kernel contracts nothing. The sampler rescores every read from the job
graph, which holds the original coefficients. The random streams are the
research streams. A stream for each read chunk draws the start states. Salted
streams draw the single flips, the block flips, and the ice moves. The work is the sweep
count of the job, so the kernel reads no clock. The log line of each job
records the declared sweeps and reads next to the measured time and the ice
move counts.

`tests/dimer_family.rs` checks the states and move counts of nine research runs
against the kernel, exact small-model energies, and every rejection.
`research/dimer_parity/export.py vectors` writes
`tests/fixtures/dimer_weighted_vectors.json` from the research `dimer_bench`.
The ignored test `research_parity` replays a saved research run of a full model
with 1785 tiles:

```sh
QUIP_RESEARCH_ROOT=/path/to/regimes python3 -I research/dimer_parity/export.py research OUT
QUIP_DIMER_PARITY=OUT cargo test --release --features experimental \
    --test dimer_family -- --ignored research_parity
```

`tests/dimer_conformance.rs` runs the family gate and the plain-job
conformance against the binary.

`quip-cpu-flow` is an exact structural baseline for two square-link
families. Its `solver_id` is `quip-cpu-flow/1` and its algorithm is `flow`. It
generates and accepts `quip/flow-static` revision 1 (F-static) and
`quip/polariton-static` revision 1 (adapted P), with every parameter set.

Both families put one spin on each link of a square lattice. The charge of a
vertex is the arrows in minus the arrows out, and a labeled vertex must hold
its label. The baseline starts from one orientation. The reversal of a link
is one unit of flow on a network with one arc per link. `families::flow`
solves that flow exactly. The code adapts `Flow::add` and `Flow::solve` of the research
two-commodity routing solver. First it saturates each arc of negative cost.
Then it routes each excess on a shortest path of reduced cost. The port
bounds each capacity and checks each sum, so an overflow gives an error and
never a wrong answer.

The two families use the flow in two modes.

- F-static starts from the requested direction of every link. A reversal
  costs the cost of its link. The result is the orientation of lowest
  cost that holds every label. The final node potentials prove that no
  cheaper orientation exists. The status is `certified-optimum`.
- Adapted P starts from spin `+1` on every link and has no costs. The free
  boundary vertices connect to one hub node. The result holds every declared
  charge, so its charge loss is 0, the lowest possible. The status is also
  `certified-optimum`.

When no orientation holds the labels, the status is `infeasible`. The solver
returns the set of nodes that the excess can reach. That set must send more
flow out than its arcs can carry, which proves the labels infeasible. The
binary declines the job with `NO_SOLUTION` and puts the cut in the detail.

Each job returns one state, because the method is exact and deterministic.
The state passes the canonical feasibility hook of its contract, and the
sampler rescores it from the job graph. The solver checks the cancel token
before it builds the network and before each augmentation. Each job logs the
mode, the status, the cost, the energy, and the number of augmentations.

The binary declines every job without a listed family with `NO_SOLUTION`. A
plain model has no flow structure. A `quip/defects-static` job (D-static, with
its dipolar energy) or a `quip/two-commodity-routing` job (occupancy) reaches
the sampler as a plain model, so the binary declines it too. The binary does
not approximate those energies.

`tests/flow_baselines.rs` checks every side-2 and some side-3 F-static
instances against exhaustive enumeration of the instance energy. It checks
synthetic requests of both modes against enumeration. Some of those requests
have no solution. It checks that adapted-P constructions hold every declared charge
on the reference and recipe instances. `tests/flow_conformance.rs` runs the
family gate.


`quip-cpu-tileworm` is a postprocess for the selected charged-ice and
field-ice cells. Its `solver_id` is `quip-cpu-tileworm/1` and its algorithm is
`tile-worm`. It advertises `initial-spins` and accepts two families. A
`quip/charged-ice` revision 1 job with the recipe `ice-charge` runs the policy
`ice-tiles-charge/1`, with the research options
`anneal:sa=1,b0=2,len=64,fe=16,nat48=1`. A `quip/ice-fields` revision 1 job
with the recipe `h500` or `h1000` runs the policy `ice-tiles-field/1`, with
the options `anneal:b0=2,nat48=1`. The binary generates no family work. Every
other job runs the `greedy-nat/1` descent.

`postprocess::tile_worm` ports `worm_run` of `icetiles.c`, the tile tables of
`icetiles.py`, and `parse` and `run` of `method.py`. Each tile holds four
spins and an energy table. A worm flips a seed spin, then repairs each tile
that the flip left in an excited state, one corner at a time, until every
touched tile is back in a ground state. The sampler keeps the worm when its exact energy
change is negative, or when a Metropolis draw accepts it. Every 16 worms the spins that are in no tile get
one sweep, and `sa=1` makes that sweep visit every spin. The schedule descends
all supplied states first and starts from the lowest. It spends 80 percent of
the budget on the anneal, from `b0 ln 2 / (2 w)` to `ln 1000 / (2 w)` where `w`
is the weakest nonzero coefficient. It then runs a zero-temperature tail and
a final descent.

The declared budget is 0.25 s per job. The binary measures elapsed time on the
thread that runs the job. The research measured process CPU time on one core.
The kernel reads the clock once for every 64 worms, so the clock sets the
number of worms and the inverse temperature of each block. Two hosts may give
different results for one timed job. The result lists the final state first
and then the descended state of each supplied read, in input order. The
sampler rescores every state from the job graph. Before it touches a state, it
checks the tile metadata and the coefficients against the instance. The sampler
declines a selected job that fails a check. The worm draws its random numbers
from the job seed. The log line of each job records the policy and the declared budget next to the
measured time and the work done.

`tests/postprocess_tiles.rs` checks the option strings, the tile tables, the
seed sets, the schedule ends, and fixed-work runs of ten option strings against
vectors from the research code in `tests/fixtures/tile_worm_vectors.json`.
`research/tileworm_parity/export.py vectors` writes that file. It compiles a
private copy of `icetiles.c` whose clock returns the block index, so a run does
a known number of blocks. The ignored test `research_parity` replays the saved
research runs of the three selected cells. The saved runs do not record the
clock readings, so a replay cannot match a saved state bit for bit. The test
checks the tile counts, the work count, and that the replay energy stays within
0.1 percent of the saved energy:

```sh
QUIP_RESEARCH_ROOT=/path/to/regimes python3 -I research/tileworm_parity/export.py research OUT
QUIP_TILEWORM_PARITY=OUT cargo test --release --features experimental \
    --test postprocess_tiles -- --ignored research_parity

`quip-cpu-gadget-repair` is a postprocess for routing jobs. Its `solver_id`
is `quip-cpu-gadget-repair/1`, its algorithm is `gadget-repair`, and its
policy is `gadget-repair/1`. It advertises `initial-spins`. It accepts
`quip/two-commodity-routing` revision 1 jobs of any parameter set. It
generates no family work. Every other job runs the `greedy-nat/1` descent.
This keeps the binary useful for plain and warm jobs, as the other
postprocess binaries do. The binary declines a routing job with a malformed
structure, gives the reason, and touches no state.

`postprocess::gadget_repair` ports the gadget extraction of `gadgets.py` for
the routing cell, `repair_run` of `repair.c`, and the schedule of `run.py`
with default options. The routing cell has no auxiliary spins, so that part of
the source is not ported. Each arc owns one commodity 1 spin and a two-spin
commodity 2 chain. The moves are single spins, chains, and pairs of units that
share a gadget. A repair first flips every move that lowers the energy. It then walks
ejection chains over the gadget neighbours. A tabu search uses the time that
remains. The schedule descends the supplied states with the natural moves. It
repairs the best four states and a majority vote of the best eight. It keeps
the lowest energy.

The declared budget is 0.05 s per job. The binary measures elapsed time on the
thread that runs the job. The research measured process CPU time on one core.
The clock decides how many states the binary repairs and where each repair stops, so
two hosts may give different results for one timed job. The result lists the
final state first and then the repaired state of each start. The sampler
rescores every state from the job graph. The repair draws its random numbers
from the job seed.

A lower energy does not prove a valid routing. The log line of each job counts
the defects of the final state and of the best supplied state. A defect is a
capacity conflict, a node and commodity with the wrong flow, or a broken chain. It also gives the energy with the instance offset. The counts
come from the family routing check, never from the energy. The log line also
records the policy and the declared budget next to the measured time and the
clock reads of each repair.

`tests/postprocess_routing.rs` checks the extraction, the repair with a fixed
clock, and the schedule with recorded work against vectors from the research
code in `tests/fixtures/gadget_repair_vectors.json`.
`research/gadget_repair_parity/export.py vectors` writes that file. The
vectors replace the CPU clock of `repair.c` by a count of clock reads. The
same file holds the malformed chain and mapping cases. The ignored test
`research_parity` replays six saved research runs. It finds the clock count of
each repair that gives the saved statistics, and compares the final states:

```sh
QUIP_RESEARCH_ROOT=/path/to/regimes python3 -I research/gadget_repair_parity/export.py research OUT
QUIP_GADGET_PARITY=OUT cargo test --release --features experimental \
    --test postprocess_routing -- --ignored research_parity

`quip-cpu-routing` solves `quip/two-commodity-routing` revision 1. Its
`solver_id` is `quip-cpu-routing/1` and its algorithm is `routing`. It accepts
every valid parameter set of revision 1 and generates no family work. It has no
fallback kernel. It declines a job with no family metadata, another family,
another revision, or a routing section that does not decode.

`families::routing` ports the research solver `routesolve`. With one commodity
fixed, the energy of the other is a minimum-cost circulation. The solver
solves it exactly. The two commodities then alternate, at most 50 times, until
a response does not improve the energy. A single-flip descent on the full Ising
model ends each run. The solver runs both start orders and returns both final
states, the lower energy first. The cost of an arc includes the capacity penalty
when the other commodity uses it, and a hub node absorbs the conservation
imbalance. The energy keeps the original convention in milli. The constants
248 and 496 and the limit of 50 alternations are the research defaults.

The result is the complete ordered spin vector of the job. A spin that no arc
uses is `+1`. The sampler rescores every state from the job graph. It logs a
start order that fails and returns the other one. It declines a job where both
fail. The two start orders often end at different energies, so the policy
certifies no global optimum. The exact solver for the adapted F-static family
is a different tool. The routing policy has no clock and no random numbers, so
`num_reads` and the seed have no effect. The log line of each job gives the
elapsed time, the work, and `optimum_certificate = false`.

`tests/routing_solver.rs` compares both start orders to the research solver
on six models with inactive spins, with vectors in
`tests/fixtures/routing_vectors.json`. It also checks the independent scores,
refused models, tampered metadata, and instances of the family.
`research/routing_parity/export.py vectors` writes the file by running the
research binary. The ignored test `research_parity` compares the saved
`t1-flow` states of 12 research models bit for bit:

```sh
QUIP_RESEARCH_ROOT=/path/to/regimes python3 -I research/routing_parity/export.py research OUT
QUIP_ROUTING_PARITY=OUT cargo test --release --features experimental \
    --test routing_solver -- --ignored research_parity
```

`quip-cpu-flow-loops` is a local search for the same two families. Its
`solver_id` is `quip-cpu-flow-loops/1` and its algorithm is `flow-loops`. It
generates and accepts `quip/flow-static` revision 1 and
`quip/polariton-static` revision 1, with every parameter set. Both build an
open square-link lattice, which is the only geometry the search reads. The
binary also accepts initial states, so it can postprocess reads.

The search changes the orientation only by moves that keep the labels. The
reversal of a directed cycle keeps every charge. The reversal of a directed
path moves charge from its last vertex to its first. The search has these
moves:

- A short cycle is one square of the lattice, or two parallel links.
- A worm walk reverses one link and then follows the arrows out of the vertex
  that is now short. At each step it takes the cheapest arrow. It stops when
  the gap closes or when it reaches the free boundary of adapted P. A walk of
  more than 64 links is undone.
- A repair reverses the shortest directed path from a pin that is short to a
  pin with an excess, or to the free boundary.

Before the search keeps a move, it checks each vertex that the move touches.
No labeled vertex can move further from its label. The charge that the free
boundary gains must equal the charge that the labeled vertices lose. After
each kept move, the search checks the whole orientation directly. It compares
the label gaps and the cost with its running values. It also compares each
charge. A difference is an error.

The policy is `flow-loops-nat/1`. It has the structure of `nat_loop` in the
research `descent.py`. Each read first runs the repairs and a steepest descent
over the short cycles. Then rounds of worm descent, repair, and steepest
descent follow, until a round does not change the score. A read runs at most
10 rounds. The worm descent follows `loop_descent` in the research
`greedy.c`. It visits every link in a random order. It keeps a walk that
lowers the score. It also keeps a walk that does not change the score, while
a budget of 4 links per link lasts. The score is the sum of the squared label
gaps, then the cost. That order is the order of the instance energy.

When a job carries initial states, the search starts from each of them.
Without them, each read starts from the exact baseline of `quip-cpu-flow`, and
the job seed drives the search. Each job returns one state per start, in start
order. Result `k` comes from start `k`. The sampler rescores each result
from the job graph.

The search runs against a declared budget of 0.25 s of host time per job.
That is the postprocess allowance of the paired A cells of plan 2: 0.25 CPU
seconds per QPU call for `ice-charge`, `ice-fields-h500`, and
`ice-fields-h1000`. The job log records the declared budget beside the
measured time, with one record per read. The budget is a limit, and a read
that converges stops early. Read `k` of `n` has until `(k + 1) / n` of the
budget. If the budget runs out first, a later read keeps its start as its
result. The time limit makes the result depend on the host.
Only a fixed-work run is reproducible. The binary declines
every job without a listed family with `NO_SOLUTION`, as `quip-cpu-flow` does.

`tests/flow_loops.rs` scores every state of side-2 instances of both families
against the instance energy. It checks the change of each short cycle against
a direct score. It then searches from every side-2 state, and it checks the
charges, the labels, and the exact rescore after each kept move. A periodic
side-2 lattice tests parallel links. Other tests reject bad boundary labels,
bad link positions, and bad starts. A fixed-work run must match
`tests/fixtures/flow_loops_golden.json`. A timed run must record its host time
and a result for every start. To record the golden output again, set
`QUIP_FLOW_LOOPS_BLESS=1` and run the test. `tests/flow_loops_conformance.rs`
runs the family gate.

`quip-cpu-dipolar` searches static defect patterns. Its `solver_id` is
`quip-cpu-dipolar/1`, its algorithm is `dipolar`, and its policy is
`dipolar-strings/1`. It generates and accepts `quip/defects-static`
revision 1, which allows the sides 2 and 3. The tests cover those two sides.
The binary advertises `initial-spins`. Every other job runs the
`greedy-nat/1` descent.

`families::dipolar_search` is a new search. No research solver exists for this
family. Each move flips a list of link ids. The move is either a loop, which
is a directed cycle, or a string, which is a directed path between two free
boundary vertices. No move flips a pinned link, so every labeled charge stays
the same. A move may pass through an obstacle, because an obstacle is a vertex
with the label 0. Only a route witness of the family must avoid obstacles. The
search checks each move directly before it scores the move. It keeps the local
field of each node and updates the score with checked integer arithmetic. The
Metropolis rule decides on the exact integer change. At the end of each read,
the search computes the score again from the coefficients. It also computes
the physical energy from the dipolar table and checks every constraint. The
sampler then compares the score with the canonical score of the job graph. A
job fails when any check fails.

A job without states runs the search mode. Each read starts from the
reference state, with every link spin `+1`. Paths between opposite labels fix
the labels of that state. A job with states runs the postprocess mode. The
search repairs each infeasible supplied state, and then searches it. The
result keeps a state that no path can repair without a change. The result holds the final
state of every read, in order. Read `r` seeds its generator from the job seed
and `r`. The instance nonce never seeds the search.

By default each read does a fixed amount of work: `num_sweeps` proposals for
each link. The `backend_toml` key `budget_seconds` sets a declared budget of
elapsed time for each read. The job log records the declared budget apart
from the measured time and the work done. The log also records the rounding
bound of the table, which is one milli for each kept coupling. An improvement
inside that bound is not resolved.

The dipolar table of revision 1 has the oracle calibration. It supports no
physical claim. The key `evidence = "physical"` asks for a physical claim.
The binary then declines every family job with the reason
`insufficient physical evidence`. The default `evidence = "analytic"` runs
every job.

`tests/dipolar_search.rs` checks all 4096 states of four lattices with
`L = 2` against the feasibility hook, the observables, and the energy of the
family. It checks every proposal from each state against a full
recomputation. It also checks analytic fixtures with a charged center, the
admission of invalid requests and instances, the direct check of invalid
moves and routes, both work modes, and a golden output for a fixed seed.

## Running

**Connect to a coordinator** (production):

```sh
quip-cpu-sa --quip-coordinator unix:///run/quip/coord.sock
```

### `num_cpus`

The coordinator `[cpu]` section may set `num_cpus`. That key travels in
`Configure.backend_toml`. The miner reads it in `apply_config`.

| Value | Effect |
|-------|--------|
| Absent | Use `available_parallelism()` as the core budget. |
| Positive integer | Use that count as the core budget. |
| Larger than the host | Clamp to the host parallelism. Log the clamp at `debug`. |
| Zero or negative | Refuse the setting. Exit 64 at handshake. |

The budget bounds sampler concurrency:

- SA: stream worker count. Each worker runs one model. Reads inside a
  model are sequential. Total threads equal `num_cpus`.
- Gibbs: stream worker count is `num_cpus / gibbs.workers`. Each in-flight
  model uses `gibbs.workers` threads (default 4). The product stays at or
  below `num_cpus`, except when `num_cpus` is smaller than `gibbs.workers`.
  Then one model still uses `gibbs.workers` threads.

- `quip-cpu-fsa` and `quip-cpu-msa`: same rule as SA. One model per core,
  models sequential inside a job, total threads equal `num_cpus`.

Other CPU binaries (SB, MPS, Flatiron) do not read `num_cpus`.

**Driver / fixed-input (run in isolation).** Use the coordinator's
`drive` harness pointed at the binary — `--source random` for golden-drawn
problems, `--source list <jsonl>` for a fixed replay:

```sh
quip-coordinator drive --miner ./quip-cpu-sa \
  --source random --topology-preset advantage2-system1 \
  --count 8 --num-reads 16 --num-sweeps 1030 --report out.jsonl
```

**Introspection:**

```sh
quip-cpu-sa --capabilities   # capabilities JSON
quip-cpu-sa --check          # probe the backend is runnable
```

## Python binding

`py/` holds `quip_msa`, a PyO3 binding to the multi-spin annealing kernel. It
is its own cargo workspace, and it is not published. Build it into a virtual
environment:

    cd py
    python3 -m venv .venv
    .venv/bin/pip install "maturin>=1,<2" numpy pytest
    .venv/bin/maturin develop --release
    .venv/bin/pytest tests -q

To install it into the environment of another project, activate that
environment first and run `maturin develop --release` from `py/`.

`quip_msa.Msa().sample(h, edges, j, num_sweeps=...)` anneals one Ising problem
and returns `(spins, energy_milli)` as numpy arrays. The call releases the GIL,
and the kernel runs one job on one core, so call it from several threads to
use several cores. `initial_spins` starts reads from given states, and
`start_beta` sets where a seeded anneal enters the beta ladder.

## Benchmarking

`quip-cpu-bench` runs a model through the sampler under a `tracing` subscriber
and writes per-part timing plus a flame graph. The report assigns the
sampler's time to each annealing seam, not only to the whole-model total.

```sh
quip-cpu-bench --nodes 512 --edges 2048 --iters 5 --out-dir bench-out
# or against a coordinator corpus (nonce-refs need --topology for the topology):
quip-cpu-bench --source instances.jsonl --topology topology.spec.json --out-dir bench-out
```

`--source` is the coordinator's `instances.jsonl`, one JSON object per line,
keyed on `nonce`. The bench ignores other keys. `--topology` is the
coordinator's `topology.spec.json`
(`{nodes, edges, allowed_h_milli, allowed_j_milli}`). The bench redraws each
nonce-ref entry against it with `quip_protocol::chacha8::draw_ising_milli`.
`--limit K` benches only the first `K` corpus models.

The bench times the scalar SA kernel that `quip-cpu-sa` runs. Gibbs runs on a
worker pool outside these seams, so the bench does not cover it.

Flags: `--nodes`/`--edges` (synthetic model) or
`--source`/`--topology`/`--limit` (corpus JSONL), `--num-reads`,
`--num-sweeps`, `--sweeps-per-beta`, `--seed`, `--warmup`, `--iters`,
`--out-dir`.

Each model writes `<out_dir>/<model_id>.json` and `<out_dir>/<model_id>.folded`:

- `schema_version` — bump signals a JSON shape change.
- `parts[]` — one entry per traced span (`part`, `scope`, `total_ns`, `count`,
  `per_call_ns`, `source`). `scope` is `"top_level"` for the four
  non-overlapping seams (`cpu_graph_build`, `beta_schedule`, `anneal_read`,
  `score`) summed into `residual_ns`, or `"nested"` for `anneal_read`'s
  children (`random_spins`, `seed_heff`, `sweep_loop`), reported for the flame
  view but excluded from that sum.
- `derived` — `per_spin_ns` and `accept_rate` computed as total ÷
  frequency (`sweep_loop.total_ns` ÷ spin visits), plus `sweep_loop_ns_per_read`.
- `residual_ns`/`residual_frac` — `measured_model_ns` minus the summed
  top-level parts. A large fraction signals a missing seam.
- `<model_id>.folded` — `tracing-flame` folded stacks. Render them with
  `inferno-flamegraph < model.folded > model.svg`.

The headline JSON uses coarse, always-on seam spans (negligible overhead —
entered O(reads) times per model). `--features fine-spans` also spans every
spin decision and accepted flip inside the hot loop, for a diagnostic
cross-check. That build measurably distorts absolute timing, so headline
numbers never use it. Optional external cross-check:
`cargo flamegraph --bin quip-cpu-bench -- --nodes 512 --edges 2048`.

## Tests

```sh
cargo test --release
```

Conformance/golden and handshake tests drive the binary in isolation via
`quip-solver-conformance` and check energies against `conformance/golden_vectors.json`.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
