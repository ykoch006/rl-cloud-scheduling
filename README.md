# Reinforcement Learning for Cloud Resource Scheduling and Autoscaling

Final-year BSc Computer Science project (CS3IP), University of Reading.

The project investigates whether reinforcement learning (RL) can learn cloud
job scheduling and autoscaling policies that outperform conventional
heuristics, and under which workload conditions RL gives a real advantage.

## Current stage: deterministic simulator

This stage builds the foundation everything else depends on: a small,
deterministic cluster simulator in plain Python with no RL libraries.
Jobs arrive, wait in a queue, are placed on servers, hold CPU and memory for
their duration, complete, and release their resources. A first-fit
scheduler is included as the first baseline policy.

## Architecture

```
env/
  job.py         Job: resource demands, duration, lifecycle (queued -> running -> completed)
  server.py      Server: CPU/memory capacity, can_fit / allocate / release, utilisation
  cluster.py     Cluster: servers + queued/running/completed jobs, validates placements
  workload.py    Fixed 10-job workload (no randomness)
  simulator.py   Discrete-timestep loop, Scheduler interface, metrics
schedulers/
  first_fit.py   First-fit placement policy
tests/           pytest suite for every module
main.py          Demo: 3 servers, 10 jobs
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
