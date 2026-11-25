# Results so far

All results below were reproduced on 11 November 2026 at commit `36776b4`
with the commands shown, except the SLA and long-wait sections, which were
produced on 25 November 2026 from a working tree based on commit `fa85584`. They are descriptive means over a small number of
seeds and have not been tested statistically. Metric definitions,
measurement windows and the aggregation over seeds are in
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). All workloads use the
provisional parameters in [MODEL_ASSUMPTIONS.md](MODEL_ASSUMPTIONS.md) and
the reference cluster of 5 servers with 100 CPU units and 64 GB each.

## Seed-to-seed variation in short workloads

```bash
python main.py --workload medium --seed 1 2 --timesteps 200
```

| Seed | Jobs | Mean CPU | Mean RAM (GB) | Mean duration | Observed CPU load | Observed RAM load |
|---|---|---|---|---|---|---|
| 1 | 491 | 13.12 | 5.65 | 9.91 | 0.608 | 0.392 |
| 2 | 503 | 13.73 | 5.72 | 10.07 | 0.758 | 0.448 |
| expected | 500 | 13.56 | 5.92 | 10.50 | 0.712 | 0.486 |

Over 200 ticks the observed CPU load was about 15% below the expected 0.712
for seed 1 and about 6% above it for seed 2. CPU × duration has a heavy right
tail, so short workloads vary noticeably, and results need to be averaged
over many seeds.

## Unfinished jobs and completed-only averages

```bash
python -m evaluation.runner --workload heavy --seeds 0 1 2 3 4 --timesteps 500 --drain 0
```

With no drain period, runs stop as soon as arrivals end and some jobs are left
unfinished (primary policies). The p95 column here is over completed jobs
only; the runner now reports a censored p95 over all arrived jobs instead.

| Policy | Completed % | Unfinished | Wait (done) | Wait LB | p95 (done) |
|---|---|---|---|---|---|
| fcfs-backfill | 97.73 | 36.2 | 1.94 | 2.03 | 8.00 |
| sjf | 97.79 | 35.2 | 1.62 | 1.80 | 6.80 |
| best-fit | 97.70 | 36.6 | 2.02 | 2.09 | 8.00 |
| round-robin | 97.73 | 36.2 | 2.23 | 2.35 | 9.20 |
| random-valid | 97.55 | 39.0 | 1.81 | 2.11 | 7.40 |

Measured over completed jobs only, Random-valid's average wait (1.81) is
lower than FCFS-backfill's (1.94). Once unfinished jobs are included, the
order reverses (2.11 versus 2.03), because Random-valid leaves more jobs
unfinished (39.0 versus 36.2). This is why the runner reports both waiting
times and gives every policy the same drain period.

## Bursty versus medium workloads

```bash
python -m evaluation.runner --workload bursty --seeds 0 1 2 3 4 5 6 7 8 9 --timesteps 1000 --drain 500
python -m evaluation.runner --workload medium --seeds 0 1 2 3 4 5 6 7 8 9 --timesteps 1000 --drain 500
```

Every job completed in every run, so "wait (done)" equals "wait LB". All
values are means over the 10 seeds of per-seed values; waits are in ticks.

| Policy | Bursty wait | Bursty p95 | Bursty queue | Medium wait | Medium p95 | Medium queue |
|---|---|---|---|---|---|---|
| fcfs-backfill | 1.13 | 6.30 | 2.77 | 0.23 | 0.79 | 0.57 |
| sjf | 0.81 | 4.30 | 1.99 | 0.23 | 0.40 | 0.57 |
| best-fit | 1.09 | 6.30 | 2.67 | 0.22 | 0.89 | 0.55 |
| round-robin | 1.15 | 6.30 | 2.82 | 0.35 | 1.00 | 0.88 |
| random-valid | 1.12 | 6.00 | 2.72 | 0.36 | 0.61 | 0.89 |
| fcfs-strict (reference) | 3.68 | 14.50 | 8.95 | 1.24 | 7.30 | 3.08 |

Turnaround was 11.31–11.66 ticks for the primary policies under bursty and
10.73–10.87 under medium. CPU utilisation over the arrival horizon was
68.73–68.85% (bursty) and 69.70–69.84% (medium).

The two profiles have the same average offered load (0.712), yet waits and
queues are several times larger under bursty. The timing of demand matters,
not only its average. Among the primary policies SJF had the shortest waits
under bursty load; at medium load the primary policies differed by at most
0.14 ticks of average wait. Strict FCFS had the longest waits in both cases,
reflecting head-of-line blocking.

## Queue behaviour across the burst cycle

From the bursty command above: mean queue length by tick within each
100-tick cycle (`*` marks burst ticks), over the 10 complete cycles in the
arrival horizon and the 10 seeds.

| Policy | 0–9 | 10–19 | 20–29 | 30–39 | 40–49 | 50–59 | 60–69 | 70–79 | 80–89* | 90–99* |
|---|---|---|---|---|---|---|---|---|---|---|
| fcfs-backfill | 8.82 | 3.54 | 1.42 | 0.51 | 0.24 | 0.10 | 0.09 | 0.09 | 2.10 | 10.80 |
| sjf | 5.70 | 2.09 | 0.88 | 0.37 | 0.23 | 0.15 | 0.12 | 0.11 | 1.86 | 8.35 |
| best-fit | 8.57 | 3.35 | 1.28 | 0.46 | 0.17 | 0.08 | 0.07 | 0.07 | 1.93 | 10.66 |
| round-robin | 8.68 | 3.62 | 1.62 | 0.65 | 0.41 | 0.25 | 0.20 | 0.21 | 2.16 | 10.45 |
| random-valid | 7.85 | 3.40 | 1.74 | 0.85 | 0.51 | 0.34 | 0.30 | 0.29 | 2.15 | 9.78 |
| fcfs-strict | 24.80 | 16.73 | 9.25 | 4.81 | 2.92 | 1.27 | 0.72 | 0.67 | 4.85 | 23.42 |

For the primary policies the mean queue was 8–11 jobs in the last 10 burst
ticks and then fell during the quiet period, before the next burst. It was
below 0.3 jobs from ticks 40–49 onwards for fcfs-backfill, SJF and best-fit,
from 50–59 for round-robin, and only by 70–79 for random-valid. Strict FCFS
still averaged about 0.7 jobs at the end of the quiet period.

These are averages over cycles and seeds. Individual cycles were not
checked, so this does not show that the queue recovers after every burst.
The 0–9 bucket includes the first cycle, which has no burst before it.

## Aggregation check

For FCFS-backfill under the bursty command above (seeds 0–9), per-seed and
pooled aggregation give different values:

| Quantity | Table value (mean of per-seed values) | Pooled over all jobs |
|---|---|---|
| Mean wait | 1.1295 | 1.1377 |
| p95 wait | 6.30 | 7.0 |

The per-seed 95th percentiles were 10, 6, 4, 10, 4, 5, 8, 8, 3 and 5 ticks.
Under medium load the corresponding values were 0.2284 and 0.2298 (mean
wait) and 0.79 and 1.0 (p95).

## Measured utilisation versus offered load

For FCFS-backfill with seeds 0–9 and T = 1,000, measured CPU utilisation over
the arrival horizon was 68.85% (bursty) and 69.82% (medium), below the
expected offered load of 71.2%. Two measured effects account for the gap:

| | Bursty | Medium |
|---|---|---|
| Realised offered CPU load (mean over seeds) | 0.705 | 0.706 |
| Share of reserved CPU-time falling in the drain | 2.3% | 1.1% |
| Realised load × (1 − drain share) | 0.689 | 0.698 |
| Measured CPU utilisation, ticks 0 to T−1 | 68.85% | 69.82% |
| Measured CPU utilisation, whole run including drain | 67.34% | 68.07% |

These seeds happened to generate slightly less work than expected, and some
work was finished after the horizon, where the fixed measurement window does
not count it. The whole-run figure is lower again because the drain ticks are
mostly idle. Run lengths were 1,036–1,063 ticks (bursty) and 1,021–1,060
ticks (medium).

The per-seed, pooled and drain-share figures in the last two sections come
from a short analysis script run outside the repository against these
results; they are not printed by the runner.

## SLA violations

```bash
python -m evaluation.runner --workload light --seeds 0 1 2 3 4 5 6 7 8 9 --timesteps 1000 --drain 500
```

The same command was run for medium, heavy and bursty, with the default SLA
deadline of arrival + 3 × duration. Every job completed in every run, so
"SLA unknown %" was 0.00 throughout and the values below are exact, not lower
bounds. "Viol %" is the percentage of all jobs that completed after their
deadline; "lateness" is mean ticks late per job, with on-time jobs counted as
0. Light-load lateness was at most 0.02 ticks for every policy.

| Policy | Light viol % | Medium viol % | Medium lateness | Heavy viol % | Heavy lateness | Bursty viol % | Bursty lateness |
|---|---|---|---|---|---|---|---|
| fcfs-backfill | 0.01 | 0.52 | 0.05 | 6.19 | 0.83 | 3.07 | 0.20 |
| sjf | 0.02 | 0.36 | 0.04 | 2.45 | 0.87 | 1.15 | 0.09 |
| best-fit | 0.02 | 0.53 | 0.04 | 5.93 | 0.86 | 2.97 | 0.19 |
| round-robin | 0.07 | 0.71 | 0.13 | 6.21 | 0.99 | 2.98 | 0.25 |
| random-valid | 0.04 | 0.77 | 0.15 | 5.25 | 1.61 | 2.82 | 0.31 |
| fcfs-strict (reference) | 0.03 | 3.49 | 0.16 | 58.58 | 20.48 | 11.62 | 0.72 |

Light load hardly separates the policies. Under heavy load SJF had far fewer
violations than the other primary policies but not the lowest lateness,
because it delays long jobs (next section). Random-valid had the highest
lateness of the primary policies under every non-light profile.

With no drain period (`--workload heavy --seeds 0 1 2 3 4 --timesteps 500
--drain 0`), 2.09–2.12% of jobs were SLA unknown for the primary policies, so
their violation rates of 3.33–6.34% are lower bounds and the true rates are
at most about 2.1 percentage points higher.

## Long waits under heavy load

Under heavy load (seeds 0–9, T = 1,000, drain 500) a few jobs waited
hundreds of ticks. A script outside the repository observed these runs
without changing the policies. At every placement it recorded whether a
younger job was placed while an older queued job fitted on some server
(bypassing). At the end of every tick it recorded why each queued job did
not fit: no single server had room although the cluster's total free CPU and
RAM would have been enough (fragmentation), or the total free capacity was
too small (contention).

| Policy | Jobs waiting ≥ 100 ticks (of 31,794) | Max wait | Their median CPU demand | Bypassed while fitting | Waiting ticks: contention / fragmentation |
|---|---|---|---|---|---|
| fcfs-backfill | 52 | 553 | 82.2 | 0% | 88% / 12% |
| sjf | 108 | 795 | 76.5 | 80% (22 times on average) | 77% / 23% |
| best-fit | 69 | 550 | 79.5 | 0% | 83% / 17% |
| round-robin | 70 | 517 | 82.6 | 0% | 81% / 19% |
| random-valid | 141 | 865 | 76.7 | 83% (4 times on average) | 82% / 18% |

The long waits fall on jobs that need a large share of one server's CPU: their
median demand was 77–83 units, compared with 10 for all jobs (99th
percentile 59.5). Under FCFS-backfill, Best-Fit and Round-Robin these jobs
were never passed over while they fitted, and the five-job window was never
the reason they were left out. They waited because no server had enough free
CPU at any decision, mostly because total free capacity was too small. This
is consistent with backfilling without reservation: all primary policies are
work-conserving, so capacity freed on a server is filled at once by smaller
jobs that do fit, and a server rarely accumulates the 80 or so free units a
large job needs. SJF and Random-valid add direct bypassing: most of their
long-wait jobs were in the window and fitted at some point, but a shorter or
randomly chosen job was taken instead. Strict FCFS, which never lets a
younger job start first, had a maximum wait of only 122 ticks, but 1,459 jobs
waited at least 100 ticks.

## Limitations

- Ten seeds at most, and no confidence intervals or paired tests yet.
- One burst shape; other periods or intensities may behave differently.
- Provisional, uncalibrated workload parameters.
- SLA results use one deadline factor (k = 3); there is no energy model.
