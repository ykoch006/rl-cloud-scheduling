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

## Design decisions and supporting literature

**Simulation instead of a real cluster.** Testing scheduling policies on
real infrastructure is slow, expensive and hard to repeat. Simulation lets
experiments be repeated exactly. CloudSim [1] is the standard example of
this approach for cloud resource management research.

**Discrete timesteps, scheduling one or more jobs per step.** DeepRM [2]
uses the same model: jobs arrive online in discrete timesteps, each job's
resource demand and duration are known on arrival, resources are held for
the whole duration with no preemption, and the scheduler may start several
waiting jobs per timestep. Matching this model keeps the later RL work
comparable with published results.

**Two resources (CPU and memory) with separate servers.** A job only fits
if *both* resources are available on the *same* server. Grandl et al. [3]
show that ignoring some resource dimensions leads to over-allocation and
fragmentation in real clusters. DeepRM [2] treats the whole cluster as one
pool and lists this as a limitation; keeping separate servers here makes
fragmentation visible.

**The scheduler proposes, the cluster validates.** An RL agent will
eventually choose placements, and some of them will be invalid. The
environment must reject those safely, as DeepRM's does [2]. Splitting
"decide" from "apply" also means every policy is checked by the same
capacity rules.

**First fit as the first baseline.** First fit is a classic bin-packing
heuristic with proven worst-case bounds [4]. Tang et al. [5] analyse it
specifically for online cloud server allocation. Like the heuristic
baselines in DeepRM [2], it is *work-conserving*: it starts every waiting
job that fits. This means a small job that arrives later can start before a
large job that is still waiting (jobs 7 and 9 in the demo).

**Metrics.** Waiting time and utilisation are standard measures for job
schedulers. Feitelson and Rudolph [6] stress that conclusions about a
scheduler depend on choosing the right metrics *and* the right workloads.
That is why the plan is to evaluate across several workload scenarios
rather than just one.

## Planned next

1. More baseline schedulers: first-come-first-served (strict queue order),
   shortest-job-first, best-fit and round-robin.
2. A stochastic workload generator (Poisson arrivals with a time-varying
   rate, log-normal resource demands, exponential durations) with a fixed
   random seed so runs stay reproducible.
3. SLA deadlines and time-varying energy costs.
4. A Gymnasium environment wrapping the simulator.
5. Tabular Q-learning on a small environment, then DQN/PPO with
   Stable-Baselines3.
6. Autoscaling actions (turning servers on and off).

## References

[1] R. N. Calheiros, R. Ranjan, A. Beloglazov, C. A. F. De Rose and
R. Buyya, "CloudSim: a toolkit for modeling and simulation of cloud
computing environments and evaluation of resource provisioning
algorithms," *Software: Practice and Experience*, vol. 41, no. 1,
pp. 23–50, 2011.

[2] H. Mao, M. Alizadeh, I. Menache and S. Kandula, "Resource management
with deep reinforcement learning," in *Proc. ACM Workshop on Hot Topics in
Networks (HotNets)*, 2016, pp. 50–56. doi: 10.1145/3005745.3005750.

[3] R. Grandl, G. Ananthanarayanan, S. Kandula, S. Rao and A. Akella,
"Multi-resource packing for cluster schedulers," in *Proc. ACM SIGCOMM*,
2014, pp. 455–466.

[4] D. S. Johnson, A. Demers, J. D. Ullman, M. R. Garey and R. L. Graham,
"Worst-case performance bounds for simple one-dimensional packing
algorithms," *SIAM Journal on Computing*, vol. 3, no. 4, pp. 299–325, 1974.

[5] X. Tang, Y. Li, R. Ren and W. Cai, "On first fit bin packing for online
cloud server allocation," in *Proc. IEEE International Parallel and
Distributed Processing Symposium (IPDPS)*, 2016.

[6] D. G. Feitelson and L. Rudolph, "Metrics and benchmarking for parallel
job scheduling," in *Job Scheduling Strategies for Parallel Processing*,
LNCS 1459, Springer, 1998, pp. 1–24.
