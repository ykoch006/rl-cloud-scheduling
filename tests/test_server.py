import pytest

from env.job import Job
from env.server import Server


def make_server(**overrides) -> Server:
    values = dict(id=1, cpu_capacity=100.0, memory_capacity=64.0)
    values.update(overrides)
    return Server(**values)


def make_job(**overrides) -> Job:
    values = dict(id=1, arrival_time=0, duration=5, cpu_required=10.0, memory_required=4.0)
    values.update(overrides)
    return Job(**values)


def test_new_server_has_full_capacity_available():
    server = make_server()

    assert server.available_cpu == 100.0
    assert server.available_memory == 64.0
    assert server.running_jobs == []


@pytest.mark.parametrize("overrides", [dict(cpu_capacity=0), dict(memory_capacity=-1)])
def test_invalid_capacity_is_rejected(overrides):
    with pytest.raises(ValueError):
        make_server(**overrides)


def test_can_fit_when_enough_resources():
    assert make_server().can_fit(make_job(cpu_required=10.0, memory_required=4.0))


def test_can_fit_when_job_uses_exactly_all_resources():
    assert make_server().can_fit(make_job(cpu_required=100.0, memory_required=64.0))


def test_cannot_fit_when_cpu_insufficient():
    assert not make_server().can_fit(make_job(cpu_required=101.0, memory_required=4.0))


def test_cannot_fit_when_memory_insufficient():
    assert not make_server().can_fit(make_job(cpu_required=10.0, memory_required=65.0))


def test_allocate_reserves_resources():
    server = make_server()
    job = make_job(cpu_required=10.0, memory_required=4.0)

    server.allocate(job)

    assert server.available_cpu == 90.0
    assert server.available_memory == 60.0
    assert server.running_jobs == [job]


def test_allocate_rejects_job_when_cpu_insufficient():
    server = make_server()
    job = make_job(cpu_required=101.0, memory_required=4.0)

    with pytest.raises(ValueError):
        server.allocate(job)

    assert server.available_cpu == 100.0
    assert server.available_memory == 64.0
    assert server.running_jobs == []


def test_allocate_rejects_job_when_memory_insufficient():
    server = make_server()
    job = make_job(cpu_required=10.0, memory_required=65.0)

    with pytest.raises(ValueError):
        server.allocate(job)

    assert server.available_cpu == 100.0
    assert server.available_memory == 64.0
    assert server.running_jobs == []


def test_release_returns_resources():
    server = make_server()
    job = make_job(cpu_required=10.0, memory_required=4.0)
    server.allocate(job)

    server.release(job)

    assert server.available_cpu == 100.0
    assert server.available_memory == 64.0
    assert server.running_jobs == []


def test_release_rejects_job_not_on_server():
    server = make_server()

    with pytest.raises(ValueError):
        server.release(make_job())


def test_multiple_jobs_share_one_server():
    server = make_server()
    job_a = make_job(id=1, cpu_required=10.0, memory_required=4.0)
    job_b = make_job(id=2, cpu_required=20.0, memory_required=8.0)

    server.allocate(job_a)
    server.allocate(job_b)

    assert server.available_cpu == 70.0
    assert server.available_memory == 52.0
    assert server.running_jobs == [job_a, job_b]


def test_capacity_is_never_exceeded_by_repeated_allocation():
    server = make_server()
    for job_id in range(1, 4):
        server.allocate(make_job(id=job_id, cpu_required=30.0, memory_required=10.0))

    with pytest.raises(ValueError):
        server.allocate(make_job(id=4, cpu_required=30.0, memory_required=10.0))

    assert server.available_cpu == 10.0
    assert len(server.running_jobs) == 3


def test_utilisation_reflects_reserved_resources():
    server = make_server()
    server.allocate(make_job(cpu_required=25.0, memory_required=16.0))

    assert server.cpu_utilisation() == pytest.approx(0.25)
    assert server.memory_utilisation() == pytest.approx(0.25)
