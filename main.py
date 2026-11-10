"""Command-line entry point.

    python main.py
        Run the fixed 10-job workload through a 3-server cluster.

    python main.py --workload medium --seed 42 --timesteps 1000
        Generate a stochastic workload and summarise it. This only generates
        the workload; it does not simulate it.

    python main.py --workload medium --seed 1 2 --timesteps 200
        Generate the same profile with two seeds and compare them side by side.
"""

import argparse
from statistics import fmean

from env.cluster import Cluster
from env.server import Server
from env.simulator import SimulationMetrics, Simulator
from env.workload import build_deterministic_workload, validate_workload
from env.workload_generator import (
    PROFILE_ARRIVAL_RATES,
    REFERENCE_SERVER_COUNT,
    REFERENCE_SERVER_CPU,
    REFERENCE_SERVER_MEMORY,
    expected_offered_load,
    generate_workload,
    observed_offered_load,
    profile_config,
)
from schedulers.fcfs import FCFSScheduler


def build_cluster() -> Cluster:
    """Three identical servers, each with 100 CPU units and 64 GB of memory."""
    return Cluster(
        servers=[Server(id=i, cpu_capacity=100.0, memory_capacity=64.0) for i in range(1, 4)]
    )


def print_summary(metrics: SimulationMetrics, total_jobs: int, scheduler_name: str) -> None:
    waiting = metrics.average_waiting_time
    waiting_text = "n/a" if waiting is None else f"{waiting:.2f} timesteps"
    print()
    print(f"=== Summary ({scheduler_name}) ===")
    print(f"Completed jobs:           {metrics.completed_count}/{total_jobs}")
    print(f"Total simulation time:    {metrics.total_duration} timesteps")
    print(f"Average waiting time:     {waiting_text}")
    print(f"Queue length over time:   {metrics.queue_length}")
    print(f"Average cluster CPU:      {metrics.average_cpu_utilisation:.1%}")
    print(f"Average cluster memory:   {metrics.average_memory_utilisation:.1%}")
    print("Per-server average utilisation:")
    for server_id in metrics.server_cpu_utilisation:
        cpu = metrics.average_server_cpu_utilisation(server_id)
        memory = metrics.average_server_memory_utilisation(server_id)
        print(f"  Server {server_id}: CPU {cpu:.1%}, memory {memory:.1%}")


def run_deterministic_demo() -> None:
    jobs = build_deterministic_workload()
    scheduler = FCFSScheduler()
    simulator = Simulator(build_cluster(), scheduler, jobs)

    metrics = simulator.run()

    for time, message in simulator.events:
        print(f"t={time} {message}")
    print_summary(metrics, len(jobs), scheduler.name)


def print_workload_summary(profile: str, seed: int, horizon: int) -> None:
    """Generate a workload and print observed statistics next to theoretical ones."""
    config = profile_config(profile, seed, horizon)
    specs = generate_workload(config)

    print(f"Workload profile: {profile} (arrival rate {config.arrival_rate} jobs/timestep)")
    print(f"Seed: {seed}")
    print(f"Arrival horizon: {horizon} timesteps")
    print(f"Jobs generated: {len(specs)}")
    print()
    print("Observed in this workload:")
    if specs:
        print(f"  Mean CPU demand: {fmean(s.cpu_required for s in specs):.2f} units")
        print(f"  Mean RAM demand: {fmean(s.memory_required for s in specs):.2f} GB")
        print(f"  Mean duration:   {fmean(s.duration for s in specs):.2f} timesteps")
    else:
        print("  (no jobs generated)")
    print("Expected from the configuration:")
    print(f"  Jobs:            {config.arrival_rate * horizon:.0f}")
    print(f"  Mean CPU demand: {config.cpu.expected_value():.2f} units")
    print(f"  Mean RAM demand: {config.memory.expected_value():.2f} GB")
    print(f"  Mean duration:   {config.duration.expected_value():.2f} timesteps")
    print()
    print("First 10 arrivals (id, arrival time, duration, CPU, RAM GB):")
    for s in specs[:10]:
        print(
            f"  {s.id:>3}  t={s.arrival_time:<4} dur={s.duration:<3} "
            f"cpu={s.cpu_required:6.2f}  ram={s.memory_required:5.2f}"
        )
    print()
    print(
        f"Offered load on {REFERENCE_SERVER_COUNT} reference servers "
        f"({REFERENCE_SERVER_CPU:.0f} CPU, {REFERENCE_SERVER_MEMORY:.0f} GB each):"
    )
    expected_cpu, expected_memory = expected_offered_load(config)
    if horizon > 0:
        observed_cpu, observed_memory = observed_offered_load(specs, horizon)
        print(f"  Observed CPU load: {observed_cpu:.3f}   Expected: {expected_cpu:.3f}")
        print(f"  Observed RAM load: {observed_memory:.3f}   Expected: {expected_memory:.3f}")
    else:
        print(f"  Expected CPU load: {expected_cpu:.3f}")
        print(f"  Expected RAM load: {expected_memory:.3f}")


def reference_servers() -> list[Server]:
    return [
        Server(id=i, cpu_capacity=REFERENCE_SERVER_CPU, memory_capacity=REFERENCE_SERVER_MEMORY)
        for i in range(1, REFERENCE_SERVER_COUNT + 1)
    ]


def print_seed_comparison(profile: str, seeds: list[int], horizon: int) -> None:
    """Same profile, different seeds: same expected values, different sampled jobs."""
    if horizon <= 0:
        raise ValueError("horizon must be > 0 to compare seeds")
    rows = []
    for seed in seeds:
        specs = generate_workload(profile_config(profile, seed, horizon))
        validate_workload(specs, reference_servers())
        cpu_load, memory_load = observed_offered_load(specs, horizon)
        rows.append((seed, specs, cpu_load, memory_load))

    print(f"Workload profile: {profile}, arrival horizon: {horizon} timesteps")
    print(f"{'seed':>6} {'jobs':>6} {'mean CPU':>9} {'mean RAM':>9} {'mean dur':>9} "
          f"{'CPU load':>9} {'RAM load':>9}")
    for seed, specs, cpu_load, memory_load in rows:
        print(f"{seed:>6} {len(specs):>6} "
              f"{fmean(s.cpu_required for s in specs):>9.2f} "
              f"{fmean(s.memory_required for s in specs):>9.2f} "
              f"{fmean(s.duration for s in specs):>9.2f} "
              f"{cpu_load:>9.3f} {memory_load:>9.3f}")

    config = profile_config(profile, seeds[0], horizon)
    expected_cpu, expected_memory = expected_offered_load(config)
    print(f"{'expect':>6} {config.arrival_rate * horizon:>6.0f} "
          f"{config.cpu.expected_value():>9.2f} {config.memory.expected_value():>9.2f} "
          f"{config.duration.expected_value():>9.2f} {expected_cpu:>9.3f} {expected_memory:>9.3f}")
    print()
    for seed, specs, _, _ in rows:
        first = ", ".join(f"(t={s.arrival_time}, dur={s.duration}, cpu={s.cpu_required:.1f})"
                          for s in specs[:3])
        print(f"seed {seed} first 3 jobs: {first}")
    print()
    print("Every job fits on at least one reference server "
          f"({REFERENCE_SERVER_COUNT} x {REFERENCE_SERVER_CPU:.0f} CPU / "
          f"{REFERENCE_SERVER_MEMORY:.0f} GB).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--workload", choices=sorted(PROFILE_ARRIVAL_RATES), help="workload profile")
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[42],
        help="random seed(s), used with --workload; give two or more to compare (default 42)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="arrival horizon in timesteps, used with --workload (default 1000)",
    )
    args = parser.parse_args()

    if args.workload is None:
        run_deterministic_demo()
        return
    try:
        if len(args.seed) == 1:
            print_workload_summary(args.workload, args.seed[0], args.timesteps)
        else:
            print_seed_comparison(args.workload, args.seed, args.timesteps)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
