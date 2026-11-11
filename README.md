# Reinforcement Learning for Cloud Resource Scheduling and Autoscaling

Final-year BSc Computer Science project (CS3IP), University of Reading.

The project investigates whether reinforcement learning (RL) can learn cloud
job scheduling and autoscaling policies that outperform conventional
heuristics, and under which workload conditions RL gives a real advantage.
It does not assume that RL will outperform the heuristics.

## Current status

Implemented:

- **Simulator.** A discrete-time cluster simulator in plain Python with no RL
  libraries. Jobs arrive, wait in a shared queue, are placed on servers, hold
  CPU and memory for their duration without preemption, complete and release
  their resources.
- **Workload generator.** Seeded, reproducible workloads with Poisson
  arrivals, bounded log-normal CPU and RAM demands and bounded exponential
  durations. Profiles: light, medium, heavy and bursty.
- **Heuristic baselines.** FCFS with backfilling, SJF, Best-Fit, Round-Robin
  and Random-valid, all limited to the same five-job candidate window, plus
  strict FCFS as a reference policy.
- **Comparison runner.** Runs every policy on identical workloads and reports
  waiting time, unfinished jobs, queue length and utilisation.

Not yet implemented: the stress workload profile, deadlines, energy, the
Gymnasium environment, RL agents (Q-learning, DQN, PPO) and autoscaling.

## Documentation

| File | Contents |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | Simulation loop, scheduler interface, candidate window, policy definitions, workload generator |
| [docs/MODEL_ASSUMPTIONS.md](docs/MODEL_ASSUMPTIONS.md) | Time model, every parameter labelled SOURCE / DERIVED / ASSUMPTION, expected loads |
| [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) | How policies are compared, metric definitions, measurement windows, aggregation over seeds |
| [docs/RESULTS.md](docs/RESULTS.md) | Results obtained so far, with the commands that reproduce them |

## Architecture

```
env/
  job.py                 Job: demands, duration, lifecycle (queued -> running -> completed)
  server.py              Server: capacity, can_fit / allocate / release, utilisation
  cluster.py             Cluster: servers and job states, validates placements
  candidates.py          Shared candidate window: which waiting jobs a scheduler may see
  workload.py            JobSpec, instantiate_jobs, validate_workload, fixed 10-job workload
  workload_generator.py  WorkloadConfig, distributions, burst schedule, profiles, offered load
  simulator.py           Discrete-time loop, Scheduler interface, metrics
schedulers/              fcfs.py, sjf.py, best_fit.py, round_robin.py, random_valid.py
evaluation/runner.py     Runs every policy on identical generated workloads
tests/                   pytest suite
docs/                    Design, assumptions, protocol and results
main.py                  Deterministic demo, or a stochastic workload summary
```

At each tick, arriving jobs join the queue. The scheduler is then shown the
five oldest waiting jobs that currently fit on some server and picks one
placement at a time, with the window refreshed after each placement. Running
jobs then advance by one tick. The simulator rejects any choice outside the
window or the cluster, so every policy works with the same information. See
[docs/DESIGN.md](docs/DESIGN.md) for details.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

The commands below assume the virtual environment is active.

**Deterministic demo.** Runs a fixed 10-job workload with `fcfs-backfill` on
three servers and prints every event and a summary:

```bash
python main.py
```

**Inspect a generated workload.** Prints the job count, observed means next
to their expected values, the first 10 arrivals and the offered load. This
only generates the workload; nothing is simulated.

```bash
python main.py --workload medium --seed 42 --timesteps 1000
```

Give two or more seeds to compare them side by side, or use the bursty
profile to see its quiet and burst rates:

```bash
python main.py --workload medium --seed 1 2 --timesteps 200
python main.py --workload bursty --seed 42 --timesteps 500
```

**Compare policies.** Profiles are `light`, `medium`, `heavy` and `bursty`.
`--timesteps` is the arrival horizon and `--drain` the extra time allowed for
queued jobs to finish (default 500):

```bash
python -m evaluation.runner --workload heavy --seeds 0 1 2 3 4 --timesteps 500 --drain 500
python -m evaluation.runner --workload bursty --seeds 0 1 2 3 4 5 6 7 8 9 --timesteps 1000 --drain 500
```

Each column is computed per seed and then averaged over seeds. Queue length
and utilisation cover the arrival horizon only, and unfinished jobs are
reported next to the waiting times. For bursty workloads the runner also
prints the mean queue length across the burst cycle. Column definitions are
in [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).

## Tests

```bash
python -m pytest
```

The suite covers jobs, servers, the cluster, the candidate window, every
scheduling policy, workload generation and validation, the simulator, the
metrics and the comparison runner. `tests/test_simulator.py` checks the demo
workload against a hand-worked trace of every job's server, start time and
completion time.

## Limitations

- All workload parameters are provisional development assumptions. They are
  not calibrated against production traces; calibration, possibly using the
  Alibaba cluster trace, is planned.
- Results are descriptive means over a few seeds. Paired statistical
  comparisons have not been carried out yet.
- Possible starvation of large jobs under the feasible-only candidate window
  has not been measured.
- CPU and RAM demands are sampled independently, and SJF assumes it knows
  each job's exact duration.
- Deadlines, energy and the stress profile are not implemented.
