# Reinforcement Learning for Cloud Resource Scheduling and Autoscaling

Final-year BSc Computer Science project (CS3IP), University of Reading.

The project investigates whether reinforcement learning (RL) can learn cloud
job scheduling and autoscaling policies that outperform conventional
heuristics, and under which workload conditions RL gives a real advantage.

## Current stage

1. **Simulator.** A small cluster simulator in plain Python with no RL
   libraries. Jobs arrive, wait in a queue, are placed on servers, hold CPU
   and memory for their duration, complete, and release their resources.
2. **Stochastic workload generator.** Seeded, reproducible workloads with
   Poisson arrivals, log-normal CPU and RAM demands and exponential
   durations. See [Stochastic workloads](#stochastic-workloads).
3. **Heuristic baselines.** FCFS (with backfilling), SJF, Best-Fit,
   Round-Robin and Random-valid, all limited to the same five-job candidate
   window, plus strict FCFS as a reference. See
   [Scheduling policies](#scheduling-policies).

Not yet implemented: the stress workload profile, deadlines, energy,
the Gymnasium environment, RL agents and autoscaling.

## Architecture

```
env/
  job.py                 Job: resource demands, duration, lifecycle (queued -> running -> completed)
  server.py              Server: CPU/memory capacity, can_fit / allocate / release, utilisation
  cluster.py             Cluster: servers + queued/running/completed jobs, validates placements
  candidates.py          Shared candidate window: which waiting jobs a scheduler may see
  workload.py            JobSpec, instantiate_jobs, validate_workload, fixed 10-job workload
  workload_generator.py  WorkloadConfig, distributions, generate_workload, profiles, offered load
  simulator.py           Discrete-timestep loop, Scheduler interface, metrics
schedulers/
  fcfs.py                FCFS with backfilling, and strict FCFS (reference)
  sjf.py                 Shortest-job-first within the window
  best_fit.py            Tightest-fitting server
  round_robin.py         Rotating server choice
  random_valid.py        Random valid choice (seeded)
evaluation/
  runner.py              Runs every policy on identical generated workloads
tests/                   pytest suite
docs/MODEL_ASSUMPTIONS.md Time model and every parameter, labelled SOURCE / DERIVED / ASSUMPTION
main.py                  Deterministic demo, or a stochastic workload summary
```

Responsibilities are split so that each class does one thing:

| Component   | Decides                                 | Does not                          |
|-------------|-----------------------------------------|-----------------------------------|
| `Scheduler` | which candidate goes on which server    | change any state                  |
| candidates  | which waiting jobs the scheduler sees   | choose placements                 |
| `Cluster`   | whether a placement is valid            | choose placements                 |
| `Server`    | whether a job fits right now            | know about queues or time         |
| `Simulator` | the order of events in time             | know which scheduling policy runs |

## How the simulator works

Time moves in whole ticks. At each tick `t`:

1. Jobs with `arrival_time == t` join the back of the queue.
2. The simulator builds the **candidate window**: the five oldest waiting
   jobs that fit on at least one server right now. The scheduler picks one
   `(job, server)` from it, or `None`. If the placement is valid it is
   applied and the window is rebuilt with the new free capacity; this
   repeats until the scheduler returns `None` or no candidate fits.
3. Metrics are recorded (queue length, per-server and cluster utilisation).
4. Every running job advances by one tick. Jobs that finish are completed at
   `t + 1` and release their resources, which are free at the start of the
   next tick.

A job with duration `d` that starts at tick `s` completes at exactly `s + d`.
Resources are reserved for a job's whole duration and jobs are never
preempted. A scheduler may only choose a job from the window it was shown
and a server in the cluster; anything else is counted as a rejected
placement and ends scheduling for that tick.

A scheduler is any object with a `name`, a `candidate_rule` and a method
`choose(candidates, servers) -> (job, server) | None`.

## Scheduling policies

All primary policies see the same five-job window (scope §5.2), so none of
them can use information an RL agent with a five-job observation would not
have. "Oldest" means queue order, which is arrival time then job id.

| Policy | Job | Server | Ties |
|---|---|---|---|
| `fcfs-backfill` | oldest candidate | first server with room (first fit) | queue order, server order |
| `sjf` | shortest duration in the window | first fit | older job |
| `best-fit` | oldest candidate | least normalised capacity left after placing | earlier server |
| `round-robin` | oldest candidate | next server with room after the last one used | rotation order |
| `random-valid` | random candidate | random server with room | own seeded generator |
| `fcfs-strict` (reference) | head of the queue only | first fit | — |

**Strict FCFS versus backfilling.** The window only contains jobs that fit
somewhere, so if the oldest job is too big to fit yet, `fcfs-backfill` skips
it and lets younger jobs use the free space. `fcfs-strict` instead starts
nothing until the head of the queue fits (head-of-line blocking). Strict FCFS
needs to see the blocked head, which the primary policies cannot, so it is
reported separately as a reference rather than as an equal comparison.

**Best-Fit score:** `(free CPU − job CPU) / CPU capacity + (free RAM − job
RAM) / RAM capacity`. Dividing by capacity puts both resources on a 0–1 scale.

SJF uses each job's true duration as its runtime estimate (perfect
estimates). All primary policies start a job whenever one fits; none waits
on purpose.

## Metrics

- completed and unfinished job counts, completion rate
- average waiting time over completed jobs
- **censored** average waiting time over every arrived job: a job still
  queued at the end counts the time it has waited so far, so this is a lower
  bound on the true average
- 95th percentile and maximum wait, average turnaround (completion − arrival)
- queue length at each tick and its average
- per-server and cluster CPU and memory utilisation (capacity-weighted)
- total simulated time and number of rejected placements

**Why unfinished jobs matter.** If a run ends before every job finishes,
the jobs left over are usually the ones that waited longest. Averaging over
completed jobs only then makes a policy look better than it is. In one
heavy-load run with no drain period, Random-valid's completed-only average
wait (1.81 ticks) was lower than FCFS's (1.94), but once unfinished jobs were
counted the order reversed (2.11 versus 2.03). The comparison runner
therefore reports both, and gives every policy the same drain period after
arrivals stop.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the tests

```bash
python -m pytest
```

`test_simulator.py` checks the demo workload against a trace worked out by
hand: every job's server, start time and completion time, plus the queue
length at every step.

## Running the demo

```bash
python main.py
```

Runs the fixed 10-job workload with `fcfs-backfill` on three servers and
prints every event, then a summary. The demo has no separate drain period,
so its utilisation figures are averaged over the whole run. The summary ends:

```
=== Summary (fcfs-backfill) ===
Completed jobs:           10/10
Total simulation time:    8 timesteps
Average waiting time:     0.60 timesteps
Queue length over time:   [0, 1, 2, 2, 1, 0, 0, 0]
```

## Stochastic workloads

### Model

The time model is described in
[docs/MODEL_ASSUMPTIONS.md](docs/MODEL_ASSUMPTIONS.md#time-model). In short,
the arrival rate λ is the expected number of new jobs per tick.

**Arrivals.** At each tick `t` from `0` to `horizon - 1`, the number of
arriving jobs is

    N_t ~ Poisson(lambda_t)

so a tick can have zero, one or several arrivals. No job arrives at or after
`horizon`. Jobs are numbered 1, 2, 3, ... in arrival order. The rate is
constant for light, medium and heavy, and changes over time for bursty;
`WorkloadConfig.arrival_rates()` returns the rate for every tick.

**Job properties.** Each job independently gets:

| Property | Distribution | Bounds |
|---|---|---|
| CPU demand | log-normal, median 10 units, sigma 0.8 | 1 to 100 units |
| RAM demand | log-normal, median 4 GB, sigma 0.9 | 0.25 to 64 GB |
| Duration | `ceil(Exponential(mean 10))` | 1 to 100 ticks |

Values outside the bounds are **redrawn** (rejection sampling), not clipped,
so no artificial spike of jobs appears at the maximum. Rounding durations up
makes every duration at least one tick; the resulting mean is about 10.5.

`validate_workload(specs, servers)` checks any workload (generated, fixed or
loaded later from a trace) against the servers it will run on: unique ids,
arrival order, finite positive demands and durations, and that every job
fits on at least one server when that server is empty.

### Reproducibility

- All randomness comes from `numpy.random.default_rng(seed)`; there is no
  global random state.
- Arrivals, CPU, RAM and durations each use their own child stream spawned
  from that seed, so changing one distribution does not change the others.
- The same `WorkloadConfig` always produces exactly the same workload.
- `generate_workload` returns immutable `JobSpec`s. The simulator changes
  `Job` objects as it runs, so every run gets **fresh** jobs:

```python
from env.workload import instantiate_jobs
from env.workload_generator import generate_workload, profile_config

specs = generate_workload(profile_config("medium", seed=42, horizon=1000))
jobs_for_run_a = instantiate_jobs(specs)   # independent copies of the
jobs_for_run_b = instantiate_jobs(specs)   # same arrivals and demands
```

### Profiles

A workload is fully described by one frozen `WorkloadConfig`: `seed`,
`horizon`, `arrival_rate`, the `cpu`, `memory` and `duration`
distributions, and an optional `burst` schedule. Invalid values raise a
`ValueError` naming the parameter.

| Profile | Jobs per tick | Expected CPU load* | Expected RAM load* | Target band |
|---|---|---|---|---|
| light | 1.6 | 0.456 | 0.311 | 40–50% |
| medium | 2.5 | 0.712 | 0.486 | 65–75% |
| heavy | 3.2 | 0.912 | 0.622 | 85–95% |
| bursty | 1.875, or 5.0 in ticks 80–99 of every 100 | 0.712 on average (0.534 quiet, 1.425 burst) | 0.486 on average | medium average with temporary overload |

\* On 5 reference servers. Offered load = mean arrival rate over the
horizon × E[demand × duration] / total cluster capacity. The target bands
are development targets, not measured cloud workload levels.

**Bursty.** The scope asks for alternating normal and peak periods with a
medium average and temporary overload, but gives no numbers. The 100-tick
cycle, the 20-tick burst at the end of each cycle and the burst rate of 5.0
are assumptions. The quiet rate of 1.875 is derived so a whole cycle averages
2.5 jobs per tick, the same as medium. Bursty's average offered load only
equals medium's when the horizon is a whole number of cycles. Offered load
during a burst is above 1, but measured utilisation can never exceed 1: the
excess waits in the queue. See
[docs/MODEL_ASSUMPTIONS.md](docs/MODEL_ASSUMPTIONS.md#bursty-profile).

### Inspecting workloads

```bash
python main.py --workload medium --seed 42 --timesteps 1000
```

Summarises one workload: job count, observed means next to theoretical
values, the first 10 arrivals and the offered load. Generation only; nothing
is simulated.

```bash
python main.py --workload medium --seed 1 2 --timesteps 200
```

Compares two seeds of the same profile side by side. Over 200 ticks the
observed CPU load was 0.608 for seed 1 and 0.758 for seed 2 against an
expected 0.712, which is why results should be averaged over many seeds.

```bash
python main.py --workload bursty --seed 42 --timesteps 500
```

For bursty, the summary also shows the quiet and burst rates and the
expected load in each period.

## Comparing policies

```bash
python -m evaluation.runner --workload heavy --seeds 0 1 2 3 4 --timesteps 500 --drain 500
```

For each seed, the workload is generated once and validated, then every
policy runs on its own fresh copy of the jobs, a fresh cluster and a fresh
scheduler, with the same arrival horizon and maximum drain period.

**How the table is aggregated.** Every column is first computed separately
for each workload seed, then averaged over seeds, with every seed weighted
equally regardless of how many jobs it has. Paired per-seed comparisons with
confidence intervals are planned.

| Column | Per seed | Jobs or ticks included |
|---|---|---|
| completed % | completed jobs / all jobs in the workload | all jobs |
| unfinished | jobs still queued, running or not yet arrived when the run stopped | all jobs |
| wait (done) | mean of start − arrival | completed jobs only |
| wait (all) | mean wait: start − arrival for every started job (completed or still running); run end − arrival for jobs that never started (a lower bound) | every job that arrived before the run ended |
| p95 wait | 95th percentile of start − arrival (NumPy linear interpolation) | completed jobs only |
| turnaround | mean of completion − arrival | completed jobs only |
| queue | mean queue length after scheduling | ticks 0 to T−1 only |
| CPU %, RAM % | mean capacity-weighted cluster utilisation | ticks 0 to T−1 only |

T is the arrival horizon (`--timesteps`). **Queue length and CPU/RAM
utilisation are measured over the arrival horizon only; drain ticks are not
included**, so every policy is measured over the same window. If a run
finishes before T, the remaining ticks count as idle with an empty queue.

Two consequences:
- **"p95 wait" is the mean of the per-seed 95th percentiles,** not the 95th
  percentile of all jobs pooled. For bursty FCFS-backfill over seeds 0–9,
  the table gives 6.30, the pooled value is 7.0, and the per-seed values
  range from 3 to 10. Waits are whole ticks, so values like 6.30 come from
  interpolation and averaging.
- **"wait (done)" is the mean of per-seed means,** not a mean over all jobs
  pooled: 1.13 versus 1.14 in the same case. A seed with no completed jobs
  would have no wait value and would be left out of these columns. That did
  not happen in any run reported here.

```bash
python -m evaluation.runner --workload bursty --seeds 0 1 2 3 4 5 6 7 8 9 --timesteps 1000 --drain 500
```

For bursty workloads the runner also prints the mean queue length in each
10-tick part of the 100-tick cycle (burst ticks marked `*`), averaged over
complete cycles inside the arrival horizon and over seeds.

**Results from that command (10 seeds).** These are descriptive means, not
statistically tested differences. Every job finished in every run, so
"wait (done)" and "wait (all)" are equal. Both columns below are means over
seeds of per-seed values, in ticks; the medium columns come from the same
command with `--workload medium`.

| Policy | Bursty mean wait | Bursty mean of per-seed p95 | Medium mean wait | Medium mean of per-seed p95 |
|---|---|---|---|---|
| fcfs-backfill | 1.13 | 6.30 | 0.23 | 0.79 |
| sjf | 0.81 | 4.30 | 0.23 | 0.40 |
| best-fit | 1.09 | 6.30 | 0.22 | 0.89 |
| round-robin | 1.15 | 6.30 | 0.35 | 1.00 |
| random-valid | 1.12 | 6.00 | 0.36 | 0.61 |
| fcfs-strict (reference) | 3.68 | 14.50 | 1.24 | 7.30 |

Bursty and medium have the same average offered load, yet waits are several
times longer under bursty, so the timing of demand matters, not just its
average. For the primary policies the mean queue was 8–11 jobs over the
last 10 burst ticks (90–99) and then fell during the quiet period, before
the next burst. It was below 0.3 jobs from ticks 40–49 onwards for
fcfs-backfill, SJF and best-fit, from 50–59 for round-robin, and only by
70–79 for random-valid. Strict FCFS still averaged about 0.7 jobs at the end
of the quiet period. These are averages over cycles and seeds: individual
cycles were not checked, and the 0–9 bucket includes the first cycle, which
has no burst before it.

## Provisional and not yet implemented

- All workload parameters are development assumptions, to be calibrated
  later, including against the Alibaba cluster trace.
- The stress profile.
- Deadlines, energy, correlated CPU/RAM demands.
- Starvation checks for large jobs under the feasible-only window (scheduled
  with the Gymnasium work).
- Paired statistical analysis of policy comparisons.
