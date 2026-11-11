# Experiment protocol: comparing scheduling policies

This document defines how policies are compared and how every reported metric
is calculated. Results produced with this protocol are in
[RESULTS.md](RESULTS.md).

## Comparison procedure

`python -m evaluation.runner` runs every policy on identical workloads:

1. For each workload seed, the workload is generated once from
   `profile_config(profile, seed, T)` and checked with `validate_workload`
   against the reference cluster (5 servers, 100 CPU units and 64 GB each).
2. Each policy runs on its own fresh copy of the jobs (`instantiate_jobs`),
   a fresh cluster and a fresh scheduler instance, so no run can affect
   another.
3. Every policy gets the same arrival horizon `T` (`--timesteps`) and the
   same maximum drain period `D` (`--drain`, default 500 ticks). A run stops
   when all jobs have completed or at tick `T + D`.
4. Random-valid's generator is seeded with the workload seed. Its stream is
   separate from the workload's own random streams.

Primary policies share the five-job feasible candidate window. Strict FCFS
sees the blocked head of the queue, which the primary policies cannot, so it
is run and reported separately as a reference.

## Unfinished jobs

When a run stops, any job still queued, still running or not yet arrived is
counted as unfinished. Unfinished jobs are reported next to every waiting-time
metric because averaging over completed jobs alone can mislead: the jobs left
unfinished are usually the ones that waited longest, so a policy that leaves
more of them can look better than it is. [RESULTS.md](RESULTS.md#unfinished-jobs-and-completed-only-averages)
contains an example in which the completed-only metric ranks two policies
the wrong way round.

The censored waiting time addresses this. For every job that arrived before
the run ended:

- a job that has started (completed or still running) uses its real wait,
  `start − arrival`;
- a job that never started uses `run end − arrival`, the time it has waited
  so far. Its true wait is at least this long (right-censored), so the
  censored average is a lower bound on the true average wait.

Jobs that had not arrived when the run ended are unfinished but are not
included in any waiting-time average.

## Metric definitions

Every column of the comparison table is computed separately for each workload
seed and then averaged over seeds, with every seed weighted equally regardless
of how many jobs it contains.

| Column | Per seed | Jobs or ticks included |
|---|---|---|
| completed % | completed jobs / all jobs in the workload | all jobs |
| unfinished | jobs queued, running or not yet arrived when the run stopped | all jobs |
| wait (done) | mean of `start − arrival` | completed jobs only |
| wait (all) | censored mean wait, as defined above (a lower bound) | every job that arrived before the run ended |
| p95 wait | 95th percentile of `start − arrival`, NumPy linear interpolation | completed jobs only |
| turnaround | mean of `completion − arrival` | completed jobs only |
| queue | mean queue length, sampled after scheduling each tick | ticks 0 to T−1 only |
| CPU %, RAM % | mean capacity-weighted cluster utilisation | ticks 0 to T−1 only |

`SimulationMetrics` also records the maximum completed-job wait, per-server
utilisation, the total simulated time and the number of rejected placements.

## Measurement windows

**Queue length and CPU/RAM utilisation are measured over the arrival horizon
only, ticks 0 to T−1. Drain ticks are excluded.** Runs end at different times
for different policies and the drain period is mostly idle, so a fixed window
keeps every policy on the same footing. If a run finishes before `T`, the
remaining ticks count as idle with an empty queue.

The deterministic demo (`python main.py`) has no separate drain period, so
its utilisation figures are averaged over the whole run.

Utilisation is the fraction of capacity actually reserved and can never
exceed 1. It is not the same as offered load, which is demand relative to
capacity and can exceed 1 (see
[MODEL_ASSUMPTIONS.md](MODEL_ASSUMPTIONS.md#bursty-profile)).

## Aggregation over seeds

Because each column is a mean of per-seed values:

- **"p95 wait" is the mean of the per-seed 95th percentiles,** not the 95th
  percentile of all jobs pooled together. Waits are whole ticks, so
  fractional values come from interpolation and averaging.
- **"wait (done)" is the mean of per-seed means,** not a mean over all jobs
  pooled.
- A seed with no completed jobs has no wait value and is left out of the
  wait, p95 and turnaround columns while still counting in "completed %".
  This has not occurred in any reported run, but it would make denominators
  differ between columns.

[RESULTS.md](RESULTS.md#aggregation-check) compares the per-seed and pooled
values for one case.

## Queue length across the burst cycle

For bursty workloads the runner also reports the mean queue length in each
10-tick part of the 100-tick cycle, with burst ticks marked `*`. The mean is
taken over complete cycles inside the arrival horizon and over seeds, so
every bucket covers the same number of ticks. Ticks after an early finish
count as an empty queue. The 0–9 bucket includes the first cycle, which has
no burst before it. This table shows average behaviour only; individual
cycles are not examined.

## Not yet part of the protocol

- Paired per-seed comparisons with bootstrap confidence intervals, as planned
  in the scope. Current tables are descriptive means.
- Starvation checks and job-size-stratified tail metrics for the
  feasible-only candidate window.
- Deadline and energy metrics.
