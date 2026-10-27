import pytest

from env.cluster import Cluster
from env.job import Job, JobStatus
from env.server import Server


def make_server(**overrides) -> Server:
    values = dict(id=1, cpu_capacity=100.0, memory_capacity=64.0)
    values.update(overrides)
    return Server(**values)


def make_job(**overrides) -> Job:
    values = dict(id=1, arrival_time=0, duration=5, cpu_required=10.0, memory_required=4.0)
    values.update(overrides)
    return Job(**values)


def test_cluster_requires_servers_with_unique_ids():
    with pytest.raises(ValueError):
        Cluster(servers=[])
    with pytest.raises(ValueError):
        Cluster(servers=[make_server(id=1), make_server(id=1)])


def test_queue_keeps_arrival_order():
    cluster = Cluster(servers=[make_server()])
    first, second = make_job(id=1), make_job(id=2)

    cluster.enqueue(first)
    cluster.enqueue(second)

    assert cluster.queued_jobs == [first, second]


def test_place_moves_job_from_queue_to_running():
    server = make_server(id=7)
    cluster = Cluster(servers=[server])
    job = make_job(cpu_required=10.0, memory_required=4.0)
    cluster.enqueue(job)

    assert cluster.place(job, server, current_time=3)

    assert cluster.queued_jobs == []
    assert cluster.running_jobs == [job]
    assert job.status == JobStatus.RUNNING
    assert job.start_time == 3
    assert job.server_id == 7
    assert server.available_cpu == 90.0


def test_place_rejects_job_that_does_not_fit_and_leaves_it_queued():
    server = make_server(cpu_capacity=5.0)
    cluster = Cluster(servers=[server])
    job = make_job(cpu_required=10.0)
    cluster.enqueue(job)

    assert not cluster.place(job, server, current_time=3)

    assert cluster.queued_jobs == [job]
    assert cluster.running_jobs == []
    assert job.status == JobStatus.QUEUED
    assert server.available_cpu == 5.0


def test_place_rejects_job_that_is_already_running():
    server_a, server_b = make_server(id=1), make_server(id=2)
    cluster = Cluster(servers=[server_a, server_b])
    job = make_job()
    cluster.enqueue(job)
    cluster.place(job, server_a, current_time=0)

    assert not cluster.place(job, server_b, current_time=1)

    assert server_b.available_cpu == 100.0
    assert job.server_id == 1


def test_place_rejects_job_that_was_never_queued():
    server = make_server()
    cluster = Cluster(servers=[server])

    assert not cluster.place(make_job(), server, current_time=0)
    assert server.available_cpu == 100.0


def test_complete_releases_resources_on_the_right_server():
    server_a, server_b = make_server(id=1), make_server(id=2)
    cluster = Cluster(servers=[server_a, server_b])
    job = make_job(cpu_required=10.0, memory_required=4.0)
    cluster.enqueue(job)
    cluster.place(job, server_b, current_time=0)

    cluster.complete(job, current_time=5)

    assert cluster.running_jobs == []
    assert cluster.completed_jobs == [job]
    assert job.status == JobStatus.COMPLETED
    assert job.completion_time == 5
    assert server_b.available_cpu == 100.0
    assert server_b.running_jobs == []


def test_cluster_resource_accounting():
    big = make_server(id=1, cpu_capacity=100.0, memory_capacity=64.0)
    small = make_server(id=2, cpu_capacity=50.0, memory_capacity=32.0)
    cluster = Cluster(servers=[big, small])
    job = make_job(cpu_required=30.0, memory_required=24.0)
    cluster.enqueue(job)

    cluster.place(job, small, current_time=0)

    assert cluster.total_cpu_capacity() == 150.0
    assert cluster.total_memory_capacity() == 96.0
    assert cluster.total_available_cpu() == 120.0
    assert cluster.total_available_memory() == 72.0
    # Capacity-weighted: 30 of 150 CPU, 24 of 96 GB.
    assert cluster.cpu_utilisation() == pytest.approx(0.2)
    assert cluster.memory_utilisation() == pytest.approx(0.25)
