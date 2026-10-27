import pytest

from env.cluster import Cluster
from env.job import Job, JobStatus
from env.server import Server
from env.simulator import Simulator
from env.workload import build_deterministic_workload
from schedulers.first_fit import FirstFitScheduler


def make_cluster(server_count: int = 3) -> Cluster:
    return Cluster(
        servers=[
            Server(id=i, cpu_capacity=100.0, memory_capacity=64.0)
            for i in range(1, server_count + 1)
        ]
    )


def run_demo():
    jobs = build_deterministic_workload()
    simulator = Simulator(make_cluster(), FirstFitScheduler(), jobs)
    metrics = simulator.run()
    return simulator, metrics, {job.id: job for job in jobs}


def test_single_job_runs_for_exactly_its_duration():
    job = Job(id=1, arrival_time=2, duration=3, cpu_required=10.0, memory_required=4.0)
    simulator = Simulator(make_cluster(1), FirstFitScheduler(), [job])

    metrics = simulator.run()

    assert job.status == JobStatus.COMPLETED
    assert job.start_time == 2
    assert job.completion_time == 5
    assert metrics.total_duration == 5
    assert simulator.cluster.servers[0].available_cpu == 100.0


def test_job_waits_in_queue_until_resources_are_released():
    first = Job(id=1, arrival_time=0, duration=3, cpu_required=80.0, memory_required=4.0)
    second = Job(id=2, arrival_time=1, duration=2, cpu_required=50.0, memory_required=4.0)
    simulator = Simulator(make_cluster(1), FirstFitScheduler(), [first, second])

    metrics = simulator.run()

    assert second.start_time == 3
    assert second.waiting_time == 2
    assert metrics.queue_length == [0, 1, 1, 0, 0]


def test_deterministic_workload_matches_hand_trace():
    simulator, metrics, jobs = run_demo()

    expected = {
        # job id: (server, start, completion)
        1: (1, 0, 4),
        2: (2, 0, 3),
        3: (3, 0, 5),
        4: (1, 1, 4),
        5: (2, 3, 5),  # waits: not enough memory on any server at t=1, 2
        6: (1, 4, 8),  # waits: not enough CPU on any server at t=2, 3
        7: (3, 2, 4),  # starts ahead of jobs 5 and 6
        8: (2, 5, 8),  # waits: not enough memory at t=3, 4
        9: (3, 4, 6),  # starts ahead of job 8
        10: (3, 5, 8),
    }
    actual = {
        job_id: (job.server_id, job.start_time, job.completion_time)
        for job_id, job in jobs.items()
    }
    assert actual == expected

    assert metrics.completed_count == 10
    assert metrics.total_duration == 8
    assert metrics.average_waiting_time == pytest.approx(0.6)
    assert metrics.queue_length == [0, 1, 2, 2, 1, 0, 0, 0]
    assert metrics.rejected_placements == 0
    assert metrics.average_cpu_utilisation == pytest.approx(1670 / 2400)


def test_deterministic_workload_gives_identical_results_every_run():
    first_sim, first_metrics, _ = run_demo()
    second_sim, second_metrics, _ = run_demo()

    assert first_sim.events == second_sim.events
    assert first_metrics.queue_length == second_metrics.queue_length
    assert first_metrics.cluster_cpu_utilisation == second_metrics.cluster_cpu_utilisation


def test_all_resources_are_released_at_the_end():
    simulator, _, _ = run_demo()

    for server in simulator.cluster.servers:
        assert server.available_cpu == server.cpu_capacity
        assert server.available_memory == server.memory_capacity
        assert server.running_jobs == []


def test_utilisation_never_exceeds_capacity():
    _, metrics, _ = run_demo()

    for samples in metrics.server_cpu_utilisation.values():
        assert all(0.0 <= value <= 1.0 for value in samples)
    for samples in metrics.server_memory_utilisation.values():
        assert all(0.0 <= value <= 1.0 for value in samples)


def test_job_too_large_for_any_server_stops_at_max_timesteps():
    huge = Job(id=1, arrival_time=0, duration=1, cpu_required=500.0, memory_required=4.0)
    simulator = Simulator(make_cluster(), FirstFitScheduler(), [huge])

    metrics = simulator.run(max_timesteps=20)

    assert metrics.total_duration == 20
    assert metrics.completed_count == 0
    assert metrics.average_waiting_time is None
    assert simulator.cluster.queued_jobs == [huge]


class OverloadingScheduler:
    """Test scheduler that always proposes every queued job on server 1."""

    name = "overloading"

    def schedule(self, queued_jobs, servers):
        return [(job, servers[0]) for job in queued_jobs]


def test_simulator_rejects_invalid_placements_from_any_scheduler():
    jobs = [
        Job(id=1, arrival_time=0, duration=2, cpu_required=70.0, memory_required=4.0),
        Job(id=2, arrival_time=0, duration=2, cpu_required=70.0, memory_required=4.0),
    ]
    simulator = Simulator(make_cluster(), OverloadingScheduler(), jobs)

    metrics = simulator.run()

    assert jobs[0].start_time == 0
    assert jobs[1].start_time == 2
    assert metrics.rejected_placements == 2
    assert all(job.server_id == 1 for job in jobs)
