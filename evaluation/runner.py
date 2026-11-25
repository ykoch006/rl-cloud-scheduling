"""Run several scheduling policies on identical generated workloads.

    python -m evaluation.runner --workload medium --seeds 0 1 2 3 4 --timesteps 500

For each workload seed the workload is generated once, then every policy runs
on its own fresh copy of those jobs, its own fresh cluster and its own fresh
scheduler. All policies get the same arrival horizon and the same maximum
drain period, so no policy can look better by being given more time.
"""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean

from env.cluster import Cluster
from env.server import Server
from env.validation import is_integer
from env.simulator import SLA_SLOWDOWN, Scheduler, SimulationMetrics, Simulator
from env.workload import JobSpec, instantiate_jobs, validate_workload
from env.workload_generator import (
    PROFILES,
    REFERENCE_SERVER_COUNT,
    REFERENCE_SERVER_CPU,
    REFERENCE_SERVER_MEMORY,
    BurstSchedule,
    generate_workload,
    profile_config,
)
from schedulers.best_fit import BestFitScheduler
from schedulers.fcfs import FCFSScheduler, StrictFCFSScheduler
from schedulers.random_valid import RandomValidScheduler
from schedulers.round_robin import RoundRobinScheduler
from schedulers.sjf import SJFScheduler

# Each factory takes the workload seed. Only random-valid uses it, as the seed
# of its own generator, which is separate from the workload's random streams.
PRIMARY_POLICIES: dict[str, Callable[[int], Scheduler]] = {
    "fcfs-backfill": lambda seed: FCFSScheduler(),
    "sjf": lambda seed: SJFScheduler(),
    "best-fit": lambda seed: BestFitScheduler(),
    "round-robin": lambda seed: RoundRobinScheduler(),
    "random-valid": lambda seed: RandomValidScheduler(seed),
}
# Strict FCFS sees the blocked head of the queue, which the primary policies
# cannot, so it is reported separately rather than as an equal comparison.
REFERENCE_POLICIES: dict[str, Callable[[int], Scheduler]] = {
    "fcfs-strict": lambda seed: StrictFCFSScheduler(),
}

DEFAULT_DRAIN = 500


@dataclass
class RunResult:
    policy: str
    workload_seed: int
    metrics: SimulationMetrics


def reference_servers(count: int = REFERENCE_SERVER_COUNT) -> list[Server]:
    return [
        Server(id=i, cpu_capacity=REFERENCE_SERVER_CPU, memory_capacity=REFERENCE_SERVER_MEMORY)
        for i in range(1, count + 1)
    ]


def run_policy(
    specs: Sequence[JobSpec], scheduler: Scheduler, horizon: int, drain: int,
    server_count: int = REFERENCE_SERVER_COUNT,
) -> SimulationMetrics:
    for name, value in (("horizon", horizon), ("drain", drain)):
        if not is_integer(value) or value < 0:
            raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
    cluster = Cluster(servers=reference_servers(server_count))
    simulator = Simulator(cluster, scheduler, instantiate_jobs(specs))
    return simulator.run(max_timesteps=horizon + drain)


def compare_policies(
    profile: str,
    seeds: Sequence[int],
    horizon: int,
    drain: int = DEFAULT_DRAIN,
    policies: dict[str, Callable[[int], Scheduler]] | None = None,
    server_count: int = REFERENCE_SERVER_COUNT,
) -> list[RunResult]:
    policies = PRIMARY_POLICIES if policies is None else policies
    results = []
    for seed in seeds:
        specs = generate_workload(profile_config(profile, seed, horizon))
        validate_workload(specs, reference_servers(server_count))
        for name, make_scheduler in policies.items():
            metrics = run_policy(specs, make_scheduler(seed), horizon, drain, server_count)
            results.append(RunResult(name, seed, metrics))
    return results


def _mean(values: list[float | None]) -> float:
    present = [v for v in values if v is not None]
    return fmean(present) if present else float("nan")


def summarise(
    results: list[RunResult], horizon: int, sla_slowdown: float = SLA_SLOWDOWN
) -> list[dict]:
    """One row per policy: each metric is computed per seed, then averaged.

    Every seed counts equally, whatever its job count. So "p95 wait LB" is the
    mean of the per-seed 95th percentiles, not the 95th percentile of all jobs
    pooled together, and "wait (done)" is the mean of per-seed means. A seed
    with no value for a column (for example no completed jobs) is left out of
    that column. Job metrics follow every job until the run ends, including
    the drain; queue length and utilisation use ticks 0..horizon-1 only. This
    is a first descriptive summary; paired per-seed comparisons with
    confidence intervals are planned separately.
    """
    rows = []
    for policy in dict.fromkeys(r.policy for r in results):
        runs = [r.metrics for r in results if r.policy == policy]
        utilisation = [m.average_utilisation_over(horizon) for m in runs]
        sla = [m.sla(sla_slowdown) for m in runs]
        rows.append({
            "policy": policy,
            "completed %": 100 * _mean([m.completion_rate for m in runs]),
            "unfinished": _mean([m.unfinished_count for m in runs]),
            "wait (done)": _mean([m.average_waiting_time for m in runs]),
            "wait LB": _mean([m.censored_average_waiting_time for m in runs]),
            "p95 wait LB": _mean([m.censored_p95_waiting_time for m in runs]),
            "turnaround LB": _mean([m.censored_average_turnaround_time for m in runs]),
            "SLA viol %": 100 * _mean([s.violation_rate for s in sla]),
            "SLA unknown %": 100 * _mean([s.unknown_rate for s in sla]),
            "lateness": _mean([s.mean_lateness for s in sla]),
            "queue": _mean([m.average_queue_length_over(horizon) for m in runs]),
            "CPU %": 100 * _mean([cpu for cpu, _ in utilisation]),
            "RAM %": 100 * _mean([memory for _, memory in utilisation]),
        })
    return rows


def queue_by_cycle_phase(
    results: list[RunResult], schedule: BurstSchedule, horizon: int, bucket: int = 10
) -> list[dict]:
    """Mean queue length at each position in the burst cycle, one row per policy.

    Only complete cycles inside the arrival horizon are used, so every bucket
    averages the same number of ticks. If a run finished before the horizon,
    the missing ticks had an empty queue and count as 0.
    """
    full_cycles = horizon // schedule.period
    if full_cycles == 0:
        raise ValueError("the arrival horizon is shorter than one burst cycle")
    rows = []
    for policy in dict.fromkeys(r.policy for r in results):
        queues = [r.metrics.queue_length for r in results if r.policy == policy]
        row = {"policy": policy}
        for start in range(0, schedule.period, bucket):
            phases = range(start, min(start + bucket, schedule.period))
            values = [
                queue[tick] if tick < len(queue) else 0
                for queue in queues
                for cycle in range(full_cycles)
                for tick in (cycle * schedule.period + phase for phase in phases)
            ]
            marker = "*" if any(schedule.in_burst(p) for p in phases) else ""
            row[f"{start}-{phases[-1]}{marker}"] = fmean(values)
        rows.append(row)
    return rows


def print_table(rows: list[dict], width: int = 13) -> None:
    columns = list(rows[0])
    print("  ".join([f"{columns[0]:>13}"] + [f"{c:>{width}}" for c in columns[1:]]))
    for row in rows:
        cells = [f"{row['policy']:>13}"] + [f"{row[c]:>{width}.2f}" for c in columns[1:]]
        print("  ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--workload", choices=PROFILES, default="medium")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--timesteps", type=int, default=500, help="arrival horizon (default 500)")
    parser.add_argument("--drain", type=int, default=DEFAULT_DRAIN,
                        help=f"extra ticks allowed after arrivals stop (default {DEFAULT_DRAIN})")
    parser.add_argument("--sla-slowdown", type=float, default=SLA_SLOWDOWN,
                        help=f"SLA deadline = arrival + k x duration (default k={SLA_SLOWDOWN})")
    args = parser.parse_args()
    if args.timesteps <= 0 or args.drain < 0:
        parser.error("--timesteps must be > 0 and --drain must be >= 0")
    if not 1 <= args.sla_slowdown < float("inf"):
        parser.error("--sla-slowdown must be a finite number >= 1")

    print(f"Profile {args.workload}, seeds {args.seeds}, arrival horizon {args.timesteps}, "
          f"max drain {args.drain}, {REFERENCE_SERVER_COUNT} servers x "
          f"{REFERENCE_SERVER_CPU:.0f} CPU / {REFERENCE_SERVER_MEMORY:.0f} GB")
    print("Each value is computed per seed, then averaged over seeds (seeds weighted equally).")
    print("'wait (done)' uses completed jobs only. The 'LB' columns, SLA and lateness also count")
    print("unfinished jobs' times so far, so they are lower bounds while any job is unfinished.")
    print(f"SLA deadline = arrival + {args.sla_slowdown:g} x duration. 'SLA viol %' (confirmed")
    print("violations) and 'SLA unknown %' (unfinished jobs that could still be on time) are")
    print("percentages of all jobs in the workload; 'lateness' is mean ticks late per job.")
    print("'queue', 'CPU %' and 'RAM %' are averaged over ticks")
    print(f"0-{args.timesteps - 1} (the arrival horizon); drain ticks are excluded.")
    print()
    primary = compare_policies(args.workload, args.seeds, args.timesteps, args.drain)
    reference = compare_policies(args.workload, args.seeds, args.timesteps, args.drain,
                                 REFERENCE_POLICIES)
    print("Primary policies (shared five-job feasible window):")
    print_table(summarise(primary, args.timesteps, args.sla_slowdown))
    print()
    print("Reference policy (sees the blocked head of the queue):")
    print_table(summarise(reference, args.timesteps, args.sla_slowdown))

    config = profile_config(args.workload, args.seeds[0], args.timesteps)
    if config.burst is not None and args.timesteps >= config.burst.period:
        print()
        print(f"Mean queue length by tick within each {config.burst.period}-tick cycle "
              f"(* = burst ticks), over complete cycles in the arrival horizon:")
        print_table(queue_by_cycle_phase(primary + reference, config.burst, args.timesteps),
                    width=7)


if __name__ == "__main__":
    main()
