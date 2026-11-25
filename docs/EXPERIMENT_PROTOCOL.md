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

The p95 wait, turnaround and SLA metrics treat unfinished jobs the same way,
using each job's earliest possible completion: its completion time if it
completed; `start + duration` if it is running (jobs are never preempted);
`run end + duration` if it is queued; `arrival + duration` if it has not
arrived. A percentile never decreases when a value increases, so the
censored p95 is also a lower bound. Only "wait (done)" is computed over
completed jobs alone; it is kept as a diagnostic of exactly this bias.

Jobs that had not arrived when the run ended are unfinished but are not
included in any waiting-time or turnaround average.

## Metric definitions

Every column of the comparison table is computed separately for each workload
seed and then averaged over seeds, with every seed weighted equally regardless
of how many jobs it contains.

| Column | Per seed | Unit | Jobs or ticks included |
|---|---|---|---|
| completed % | completed jobs / all jobs in the workload | % | all jobs |
| unfinished | jobs queued, running or not yet arrived when the run stopped | jobs | all jobs |
| wait (done) | mean of `start − arrival` | ticks | completed jobs only |
| wait LB | censored mean wait, as defined above (a lower bound) | ticks | every job that arrived before the run ended |
| p95 wait LB | 95th percentile of the censored waits, NumPy linear interpolation (a lower bound) | ticks | as wait LB |
| turnaround LB | mean of `earliest completion − arrival`, i.e. censored wait + duration (a lower bound) | ticks | as wait LB |
| SLA viol % | confirmed SLA violations / all jobs in the workload (below) | % | all jobs |
| SLA unknown % | unfinished jobs that could still meet their deadline / all jobs in the workload | % | all jobs |
| lateness | sum of `max(0, earliest completion − deadline)` / all jobs | ticks per job | all jobs |
| queue | mean queue length, sampled after scheduling each tick | jobs | ticks 0 to T−1 only |
| CPU %, RAM % | mean capacity-weighted share of capacity reserved by running jobs | % | ticks 0 to T−1 only |

When a seed has nothing to average for a column (no completed jobs for
"wait (done)", no arrived jobs for the "LB" columns, an empty workload for
completed %, SLA viol % and lateness), its value is undefined (`None`), not
0. Utilisation is reserved capacity, not measured hardware usage.

`SimulationMetrics` also records the completed-only p95, maximum wait and
turnaround, per-server utilisation, the total simulated time and the number
of rejected placements.

## SLA deadlines

Each job's deadline is `arrival + k × duration`, with slowdown factor k = 3 by
default (`--sla-slowdown`; k must be finite and ≥ 1, where k = 1 allows no
waiting). k = 3 is a modelling assumption, not an established SLA threshold;
k = 2 and k = 5 are planned for sensitivity analysis. Deadlines are
evaluation-only: schedulers do not see them, and generated workloads are
unchanged.

- A job is on time if it completes at or before its deadline; completing
  exactly at the deadline is on time.
- Lateness is `max(0, completion − deadline)` in ticks, assessed once per job
  at completion. A penalty of one unit per tick past the deadline would give
  the same total.
- "SLA viol %" and "lateness" are reported separately because they can
  disagree: under heavy load SJF had far fewer violations than FCFS-backfill
  but no lower mean lateness, because it moves delay from short jobs to long
  ones.
- An unfinished job whose earliest possible completion is already after its
  deadline is a confirmed violation, and its lateness so far is counted. An
  unfinished job that could still finish on time is counted as unknown and
  adds no lateness. While any job is unfinished, SLA viol % and lateness are
  therefore lower bounds, and the true violation rate lies between SLA viol %
  and SLA viol % + SLA unknown %.
- Both percentages use the same denominator: every job in the workload,
  including jobs that were unfinished or had not arrived when the run
  stopped, so on-time jobs, confirmed violations and unknown jobs together
  make up 100%.

## Measurement windows

**Queue length and CPU/RAM utilisation are measured over the arrival horizon
only, ticks 0 to T−1. Drain ticks are excluded.** Job metrics (completed %,
waits, turnaround and SLA) are per job instead: every job arrives in ticks 0
to T−1 and is followed until it completes or the run stops at T + D, so they
include the drain. Runs end at different times
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

- **"p95 wait LB" is the mean of the per-seed 95th percentiles,** not the 95th
  percentile of all jobs pooled together. Waits are whole ticks, so
  fractional values come from interpolation and averaging.
- **"wait (done)" is the mean of per-seed means,** not a mean over all jobs
  pooled.
- A seed with an undefined value for a column is left out of that column's
  mean while still counting in the others. This has not occurred in any
  reported run, but it would make denominators differ between columns.

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

## Metrics considered but not reported

- **Throughput.** Every job arrives before T. Within the arrival horizon,
  completions by T equal arrivals minus the jobs still queued or running at
  T; over a complete run they equal the job count. Throughput would
  therefore repeat completed % and queue length.
- **Energy.** There is no energy model. With N always-on servers of equal
  capacity, a linear power model `P = P_idle + (P_max − P_idle) × u` gives an
  energy over T ticks of `N × T × (P_idle + (P_max − P_idle) × mean u)`, an
  affine function of the CPU % column. Over a complete run the reserved
  CPU-time, `sum(cpu × duration)`, is the same for every policy, so only the
  run length would change the result. An energy figure would add nothing
  until servers can be switched off or the cluster scales. Any future figure
  would be an abstract proxy, not measured hardware consumption.

## Not yet part of the protocol

- Paired per-seed comparisons with bootstrap confidence intervals, as planned
  in the scope. Current tables are descriptive means.
- Job-size-stratified tail metrics. Long waits of large-CPU jobs under
  heavy load were analysed once (RESULTS.md) but are not reported by the
  runner.
