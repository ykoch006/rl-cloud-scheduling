# Design: simulator, schedulers and workload generator

This document describes how the implemented system works. Parameter values
and their justification are in [MODEL_ASSUMPTIONS.md](MODEL_ASSUMPTIONS.md);
how policies are compared is in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

## Components

Responsibilities are split so that each part does one thing:

| Component | Decides | Does not |
|---|---|---|
| `Scheduler` | which candidate goes on which server | change any state |
| candidate window (`env/candidates.py`) | which waiting jobs the scheduler sees | choose placements |
| `Cluster` | whether a placement is valid | choose placements |
| `Server` | whether a job fits right now | know about queues or time |
| `Simulator` | the order of events in time | know which scheduling policy runs |

`Server` recomputes its free capacity from its running jobs after every
allocation and release, rather than adding and subtracting each time. This
stops floating-point rounding from building up over thousands of placements
(see [MODEL_ASSUMPTIONS.md](MODEL_ASSUMPTIONS.md#modelling-choices-to-review)).

## Simulation loop

Time moves in whole ticks; the time model is defined in
[MODEL_ASSUMPTIONS.md](MODEL_ASSUMPTIONS.md#time-model). At each tick `t`:

1. Jobs with `arrival_time == t` join the back of the queue.
2. The simulator builds the candidate window from the current queue and free
   capacity. The scheduler picks one `(job, server)` from it, or `None`. A
   valid placement is applied and the window is rebuilt with the new free
   capacity. This repeats until the scheduler returns `None` or no candidate
   fits.
3. Metrics are recorded: queue length and per-server and cluster utilisation.
4. Every running job advances by one tick. Jobs that finish are completed at
   `t + 1` and release their resources, which are free at the start of the
   next tick.

A job with duration `d` that starts at tick `s` completes at exactly `s + d`.
Resources are reserved for the whole duration and jobs are never preempted.

The simulator enforces the scheduler contract. A scheduler may only choose a
job from the window it was shown (checked by identity) and a server that
belongs to the cluster. Anything else, or a placement the cluster rejects for
lack of capacity, is counted as a rejected placement and ends scheduling for
that tick, so a faulty scheduler cannot loop forever. Each successful
placement removes a job from the queue, so the loop runs at most once per
queued job.

`Simulator.run(max_timesteps)` stops when every job has completed or the
limit is reached. Jobs still queued, running or not yet arrived at that
point are recorded in `metrics.unfinished_jobs` rather than dropped.

## Scheduler interface

A scheduler is any object with:

- `name`: a label used in output;
- `candidate_rule`: which candidate window it receives (below);
- `choose(candidates, servers) -> (job, server) | None`.

Choosing one placement at a time, against a window that is refreshed after
every placement, is the same decision loop the planned RL environment will
use, so heuristics and learned policies face the same choices.

## Candidate window

The shared window follows the project scope (§5.2). The RL agents will
observe a fixed number of waiting jobs, so a heuristic that could inspect the
whole queue would have an unfair information advantage.

- **Size:** five jobs (`CANDIDATE_LIMIT`).
- **`CandidateRule.FEASIBLE`** (all primary policies): the five oldest
  waiting jobs that fit on at least one server with its current free
  capacity. Skipping jobs that cannot fit yet stops a few large blocked jobs
  at the front from hiding younger jobs that could run.
- **`CandidateRule.HEAD_OF_QUEUE`** (strict FCFS only): the five oldest jobs,
  whether or not they fit.
- **Order:** queue order. Jobs are enqueued by `(arrival_time, id)`, so ties
  between jobs that arrived in the same tick are broken by id.
- Selecting candidates never modifies the queue, so a job outside the window
  is not discarded.

## Scheduling policies

| Policy | Job | Server | Ties |
|---|---|---|---|
| `fcfs-backfill` | oldest candidate | first server with room (first fit) | queue order, then server order |
| `sjf` | shortest duration in the window | first fit | older job |
| `best-fit` | oldest candidate | least normalised capacity left after placing | earlier server |
| `round-robin` | oldest candidate | next server with room after the last one used | rotation order |
| `random-valid` | random candidate | random server with room | own seeded generator |
| `fcfs-strict` (reference) | head of the queue only | first fit | — |

**Strict FCFS versus backfilling.** Because the feasible window only holds
jobs that fit somewhere, `fcfs-backfill` skips an oldest job that is too big
to fit yet and lets younger jobs use the free space. `fcfs-strict` starts
nothing until the head of the queue fits (head-of-line blocking). Strict FCFS
needs to see the blocked head, which the primary policies cannot, so it is
reported separately as a reference rather than as an equal comparison.
Backfilling improves utilisation but can starve large jobs; this has not yet
been measured.

**Best-Fit score:** `(free CPU − job CPU) / CPU capacity + (free RAM − job
RAM) / RAM capacity`. Dividing by capacity puts both resources on a 0–1
scale, so neither unit dominates. Lower means a tighter fit.

**SJF** compares only jobs in the window (window-limited SJF, not global SJF)
and uses each job's true duration as its runtime estimate, i.e. perfect
estimates.

**Round-robin** keeps its rotation position between ticks, so each simulation
run needs a new instance.

**Random-valid** chooses uniformly among candidates and then among servers
that fit the chosen job. It uses its own `numpy.random.default_rng(seed)`
and never touches global random state. It is the "no preference" reference
that a learned policy should at least beat.

All primary policies are work-conserving: they start a job whenever one fits
and never wait on purpose.

## Workload generation

### Arrivals and job properties

At each tick `t` from `0` to `horizon − 1`, the number of arriving jobs is

    N_t ~ Poisson(lambda_t)

so a tick can have zero, one or several arrivals, and no job arrives at or
after `horizon`. Jobs are numbered 1, 2, 3, ... in arrival order.
`WorkloadConfig.arrival_rates()` returns `lambda_t` for every tick: constant
for light, medium and heavy, and following a `BurstSchedule` for bursty.

Each job then independently gets:

| Property | Distribution | Bounds |
|---|---|---|
| CPU demand | log-normal, median 10 units, sigma 0.8 | 1 to 100 units |
| RAM demand | log-normal, median 4 GB, sigma 0.9 | 0.25 to 64 GB |
| Duration | `ceil(Exponential(mean 10))` | 1 to 100 ticks |

Values outside the bounds are redrawn (rejection sampling) rather than
clipped, so no artificial spike of jobs appears at the maximum. Rounding
durations up makes every duration at least one tick, with a resulting mean of
about 10.5 ticks.

### Configuration and validation

A workload is fully described by one frozen `WorkloadConfig`: `seed`,
`horizon`, `arrival_rate`, the `cpu`, `memory` and `duration` distributions,
and an optional `burst` schedule. Invalid values raise a `ValueError` that
names the parameter, for example a negative rate, a non-integer horizon, a
burst that does not fit inside its period, or a maximum demand above one
server's capacity. `profile_config(profile, seed, horizon)` builds the
configuration for a named profile.

`validate_workload(specs, servers)` checks any workload (generated, the fixed
demo workload, or a trace in future) against the servers it will run on:
unique ids, arrival order, finite positive demands and durations, and that
every job fits on at least one server when that server is empty. A job that
could never fit would otherwise wait forever and inflate the unfinished
count.

### Reproducibility

- All randomness comes from `numpy.random.default_rng(seed)`; there is no
  global random state.
- Arrivals, CPU, RAM and durations each use their own child stream spawned
  from that seed, so changing one distribution does not change the values
  drawn for the others.
- The same `WorkloadConfig` always produces exactly the same workload.
- `generate_workload` returns immutable `JobSpec`s. The simulator changes
  `Job` objects as it runs, so every run is given fresh jobs:

```python
from env.workload import instantiate_jobs
from env.workload_generator import generate_workload, profile_config

specs = generate_workload(profile_config("medium", seed=42, horizon=1000))
jobs_for_run_a = instantiate_jobs(specs)   # independent copies of the
jobs_for_run_b = instantiate_jobs(specs)   # same arrivals and demands
```

### Deterministic demo workload

`build_deterministic_workload()` returns ten hand-chosen jobs for three
servers. They are chosen so that some jobs wait for CPU (job 6), some wait
for memory (jobs 5 and 8), and some later, smaller jobs start before earlier
larger ones (jobs 7 and 9). `tests/test_simulator.py` checks every job's
server, start and completion time against a hand-worked trace.
