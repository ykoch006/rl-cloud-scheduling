"""Run the deterministic workload through a 3-server cluster and print the results."""

from env.cluster import Cluster
from env.server import Server
from env.simulator import SimulationMetrics, Simulator
from env.workload import build_deterministic_workload
from schedulers.first_fit import FirstFitScheduler


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


def main() -> None:
    jobs = build_deterministic_workload()
    scheduler = FirstFitScheduler()
    simulator = Simulator(build_cluster(), scheduler, jobs)

    metrics = simulator.run()

    for time, message in simulator.events:
        print(f"t={time} {message}")
    print_summary(metrics, len(jobs), scheduler.name)


if __name__ == "__main__":
    main()
