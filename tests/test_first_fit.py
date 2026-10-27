from env.job import Job
from env.server import Server
from schedulers.first_fit import FirstFitScheduler


def make_job(job_id: int, cpu: float, memory: float) -> Job:
    return Job(id=job_id, arrival_time=0, duration=1, cpu_required=cpu, memory_required=memory)


def test_job_goes_to_first_server_with_room():
    full = Server(id=1, cpu_capacity=10.0, memory_capacity=10.0)
    roomy = Server(id=2, cpu_capacity=100.0, memory_capacity=64.0)
    job = make_job(1, cpu=50.0, memory=8.0)

    placements = FirstFitScheduler().schedule([job], [full, roomy])

    assert placements == [(job, roomy)]


def test_job_that_fits_nowhere_is_not_placed():
    server = Server(id=1, cpu_capacity=100.0, memory_capacity=64.0)
    job = make_job(1, cpu=10.0, memory=100.0)

    assert FirstFitScheduler().schedule([job], [server]) == []


def test_does_not_promise_the_same_space_twice():
    server = Server(id=1, cpu_capacity=100.0, memory_capacity=64.0)
    job_a = make_job(1, cpu=60.0, memory=8.0)
    job_b = make_job(2, cpu=60.0, memory=8.0)

    placements = FirstFitScheduler().schedule([job_a, job_b], [server])

    assert placements == [(job_a, server)]


def test_later_small_job_can_skip_past_earlier_large_job():
    server = Server(id=1, cpu_capacity=100.0, memory_capacity=64.0)
    large = make_job(1, cpu=200.0, memory=8.0)
    small = make_job(2, cpu=10.0, memory=8.0)

    placements = FirstFitScheduler().schedule([large, small], [server])

    assert placements == [(small, server)]


def test_scheduler_does_not_modify_servers():
    server = Server(id=1, cpu_capacity=100.0, memory_capacity=64.0)

    FirstFitScheduler().schedule([make_job(1, cpu=50.0, memory=8.0)], [server])

    assert server.available_cpu == 100.0
    assert server.running_jobs == []
