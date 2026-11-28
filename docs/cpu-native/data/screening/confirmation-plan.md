# Independent CPU screening confirmation

Freeze the screen and comparison before generating labels or scores for
seeds 2501–2600. Use ten consecutive pools of ten canonical nonces and
the retained full C60 configuration from the initial 104-label study.

The primary screen keeps the three lowest counts of frustrated weak
triangles in each pool. The score was chosen on seeds 2301–2324. Average
uniformly over ties at the third position. The primary outcome is the
mean improvement in each pool's best full-solve energy over the exact
expectation for three uniformly selected nonces. Compute a one-sided
randomization tail probability by enumerating the minimum-energy
distribution for a random three-of-ten shortlist in every pool and
convolving the ten distributions. Report the paired pool-bootstrap 95%
interval and individual pool effects. As a check on cutoff ties, also
permute the structure scores within each pool and compare the same
tie-averaged statistic over 100,000 draws. The confirmation criterion is
a positive lower interval bound and a tie-aware permutation probability
below 0.05.

As secondary assessments, evaluate the frozen rank-four relaxation and
both native probes under the same budget. Report recall and AUROC at the
CPU threshold fixed from the first holdout (−5,052,195 milli), plus the
number of full solves reaching the GPU study's −5,090,000 milli threshold.
Report the initial and confirmation samples separately. This offline
comparison charges the same number of full solves per pool. It measures
score cost separately and leaves live throughput unmeasured.

Save the first 20 full-solve batches for a separate exact recombination
experiment. Final energies from those batches do not select the screening
policy or the recombination parameters.
