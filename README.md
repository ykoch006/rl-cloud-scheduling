# Reinforcement Learning for Cloud Resource Scheduling and Autoscaling

Final-year BSc Computer Science project (CS3IP), University of Reading.

The project investigates whether reinforcement learning (RL) can learn cloud
job scheduling and autoscaling policies that outperform conventional
heuristics, and under which workload conditions RL gives a real advantage.

## Current stage

1. **Deterministic simulator.** A small cluster simulator in plain Python
   with no RL libraries. Jobs arrive, wait in a queue, are placed on
   servers, hold CPU and memory for their duration, complete, and release
   their resources. A first-fit scheduler is the first baseline policy.
2. **Stochastic workload generator.** Seeded, reproducible workloads with
   Poisson arrivals, log-normal CPU and RAM demands and exponential
   durations. See [Stochastic workloads](#stochastic-workloads).

## Architecture

```
env/
  job.py                 Job: resource demands, duration, lifecycle (queued -> running -> completed)
  server.py              Server: CPU/memory capacity, can_fit / allocate / release, utilisation
  cluster.py             Cluster: servers + queued/running/completed jobs, validates placements
  workload.py            JobSpec (immutable job description), instantiate_jobs, fixed 10-job workload
  workload_generator.py  WorkloadConfig, distributions, generate_workload, profiles, offered load
  simulator.py           Discrete-timestep loop, Scheduler interface, metrics
schedulers/
  first_fit.py           First-fit placement policy
tests/                   pytest suite for every module
docs/assumptions.md      Every workload parameter, labelled SOURCE / DERIVED / ASSUMPTION
main.py                  Deterministic demo, or a stochastic workload summary
```

Responsibilities are split so that each class does one thing:

| Component   | Decides                        | Does not                          |
|-------------|--------------------------------|-----------------------------------|
| `Scheduler` | which job goes on which server | change any state                  |
| `Cluster`   | whether a placement is valid   | choose placements                 |
| `Server`    | whether a job fits right now   | know about queues or time         |
| `Simulator` | the order of events in time    | know which scheduling policy runs |

## How the simulator works

Time moves in discrete steps. At each timestep `t`:

1. Jobs with `arrival_time == t` join the back of the queue.
2. The scheduler looks at the queue and the servers and proposes
   `(job, server)` placements.
3. The cluster applies each proposal if the server has enough free CPU
   *and* memory, and rejects it otherwise. Rejected jobs stay queued.
4. Metrics are recorded (queue length, per-server and cluster utilisation).
5. Every running job advances by one timestep. Jobs that finish are
   completed at `t + 1` and release their resources, which are then free
   for the scheduler at the start of the next timestep.

A job with duration `d` that starts at time `s` completes at exactly
`s + d`. Resources are reserved for a job's whole duration and jobs are
never preempted.

Any object with a `name` and a
`schedule(queued_jobs, servers) -> list[(job, server)]` method can be
passed to the simulator, so new policies plug in without changing the
simulator.

### Metrics

- completed job count
- average waiting time (time from arrival to start, over completed jobs)
- queue length at each timestep
- per-server CPU and memory utilisation at each timestep
- average cluster CPU and memory utilisation (weighted by capacity)
- total simulation duration (time at which the last job completes)
- number of placements the cluster rejected

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

Example output (shortened):

```
t=0 Job 1 arrived
t=0 Job 1 assigned to Server 1
...
t=1 Waiting in queue: jobs 5
...
t=3 Job 2 completed on Server 2
t=3 Job 5 assigned to Server 2
...
=== Summary (first-fit) ===
Completed jobs:           10/10
Total simulation time:    8 timesteps
Average waiting time:     0.60 timesteps
Queue length over time:   [0, 1, 2, 2, 1, 0, 0, 0]
Average cluster CPU:      69.6%
Average cluster memory:   45.4%
```

## Stochastic workloads

### Model

A workload is generated in two steps.

**Arrivals.** At each integer timestep `t` from `0` to `horizon - 1`, the
number of arriving jobs is

    N_t ~ Poisson(lambda_t)

so a timestep can have zero, one or several arrivals. No job arrives at or
after `horizon`. Jobs are numbered 1, 2, 3, ... in arrival order; jobs that
arrive in the same timestep keep their generation order. The rate is
constant for now. `WorkloadConfig.arrival_rates()` returns one rate per
timestep, so a time-varying `lambda(t)` only needs a change there.

**Job properties.** Each job independently gets:

| Property | Distribution | Bounds |
|---|---|---|
| CPU demand | log-normal, median 10 units, sigma 0.8 | 1 to 100 units |
| RAM demand | log-normal, median 4 GB, sigma 0.9 | 0.25 to 64 GB |
| Duration | `ceil(Exponential(mean 10))` | 1 to 100 timesteps |

Values outside the bounds are **redrawn** (rejection sampling), not clipped,
so no artificial spike of jobs appears at the maximum. Upper bounds equal one
server's capacity, so every generated job fits on an empty server.
Rounding durations up makes every duration at least one timestep; the
resulting mean is about 10.5. These parameters are provisional; see
[docs/assumptions.md](docs/assumptions.md).

### Reproducibility

- All randomness comes from `numpy.random.default_rng(seed)`; there is no
  global random state.
- Arrivals, CPU, RAM and durations each use their own child stream
  spawned from that seed. Changing one distribution does not change the
  values drawn for the others.
- The same `WorkloadConfig` always produces exactly the same workload.
- `generate_workload` returns immutable `JobSpec`s. The simulator changes
  `Job` objects as it runs, so every run should get **fresh** jobs:

```python
from env.workload import instantiate_jobs
from env.workload_generator import generate_workload, profile_config

specs = generate_workload(profile_config("medium", seed=42, horizon=1000))
jobs_for_run_a = instantiate_jobs(specs)   # independent copies of the
jobs_for_run_b = instantiate_jobs(specs)   # same arrivals and demands
```

### Configuration

A workload is fully described by one frozen `WorkloadConfig` dataclass:
`seed`, `horizon`, `arrival_rate`, and the `cpu`, `memory` and `duration`
distributions (each with its own bounds). Invalid values raise a
`ValueError` that names the parameter. Examples include a negative rate,
a non-integer horizon, or bounds above server capacity.

Three development profiles differ only in arrival rate:

| Profile | Jobs per timestep | Expected CPU load* | Expected RAM load* |
|---|---|---|---|
| light | 1.0 | 0.29 | 0.19 |
| medium | 2.0 | 0.57 | 0.39 |
| heavy | 3.0 | 0.86 | 0.58 |

\* On 5 reference servers. Offered load = arrival rate × E[demand ×
duration] / total cluster capacity. These are **not** calibrated workload
levels.

### Summarising a workload

```bash
python main.py --workload medium --seed 42 --timesteps 1000
```

This prints the job count, the observed mean demands and durations next to
their theoretical values, the first 10 arrivals, and the observed and
expected offered load. It only generates the workload; it does not run the
simulator.

### Provisional and not yet implemented

- All numerical parameters are development assumptions, to be calibrated
  later, including against the Alibaba cluster trace.
- Bursty / time-varying arrival rates (planned next).
- SLA deadlines, energy costs, correlated CPU/RAM demands.
- Running generated workloads through the simulator in experiments.
  `Simulator.run` stops at `max_timesteps` (default 1000) without a warning,
  so a run with a 1000-step arrival horizon must use a larger limit or it
  will leave jobs unfinished.
