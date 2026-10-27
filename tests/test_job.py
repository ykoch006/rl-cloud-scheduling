import pytest

from env.job import Job, JobStatus


def make_job(**overrides) -> Job:
    values = dict(id=1, arrival_time=0, duration=5, cpu_required=10.0, memory_required=4.0)
    values.update(overrides)
    return Job(**values)


def test_new_job_is_queued_with_full_remaining_time():
    job = make_job(duration=5)

    assert job.status == JobStatus.QUEUED
    assert job.remaining_time == 5
    assert job.start_time is None
    assert job.completion_time is None
    assert job.server_id is None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(duration=0),
        dict(arrival_time=-1),
        dict(cpu_required=-1.0),
        dict(memory_required=-1.0),
    ],
)
def test_invalid_job_is_rejected(overrides):
    with pytest.raises(ValueError):
        make_job(**overrides)


def test_start_records_time_server_and_status():
    job = make_job()

    job.start(current_time=3, server_id=2)

    assert job.status == JobStatus.RUNNING
    assert job.start_time == 3
    assert job.server_id == 2


def test_tick_reduces_remaining_time_until_finished():
    job = make_job(duration=2)

    assert not job.is_finished()
    job.tick()
    assert job.remaining_time == 1
    assert not job.is_finished()
    job.tick()
    assert job.is_finished()


def test_complete_records_time_and_status():
    job = make_job()

    job.complete(current_time=10)

    assert job.status == JobStatus.COMPLETED
    assert job.completion_time == 10


def test_waiting_time_is_start_minus_arrival():
    job = make_job(arrival_time=2)
    assert job.waiting_time is None

    job.start(current_time=7, server_id=1)

    assert job.waiting_time == 5
