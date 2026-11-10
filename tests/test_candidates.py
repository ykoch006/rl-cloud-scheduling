import pytest

from env.candidates import CANDIDATE_LIMIT, CandidateRule, select_candidates, servers_that_fit
from env.cluster import Cluster
from env.job import Job
from env.server import Server
from env.simulator import Simulator
from schedulers.fcfs import FCFSScheduler


def make_job(job_id: int, cpu: float = 10.0, memory: float = 4.0, arrival: int = 0) -> Job:
    return Job(id=job_id, arrival_time=arrival, duration=3, cpu_required=cpu, memory_required=memory)


def make_server(server_id: int = 1, cpu: float = 100.0, memory: float = 64.0) -> Server:
    return Server(id=server_id, cpu_capacity=cpu, memory_capacity=memory)


def test_default_window_size_is_five():
    assert CANDIDATE_LIMIT == 5


def test_window_keeps_queue_order_and_is_truncated():
    queue = [make_job(i) for i in range(1, 9)]

    candidates = select_candidates(queue, [make_server()])

    assert [job.id for job in candidates] == [1, 2, 3, 4, 5]


def test_jobs_that_fit_nowhere_now_are_skipped():
    queue = [make_job(1, cpu=90.0), make_job(2), make_job(3, memory=70.0), make_job(4)]
    server = make_server(cpu=50.0)

    assert [job.id for job in select_candidates(queue, [server])] == [2, 4]


def test_blocked_jobs_at_the_front_do_not_hide_later_jobs():
    blocked = [make_job(i, cpu=200.0) for i in range(1, 7)]
    runnable = make_job(7)

    candidates = select_candidates(blocked + [runnable], [make_server()])

    assert candidates == [runnable]


def test_feasibility_uses_current_free_capacity():
    server = make_server(cpu=100.0)
    server.allocate(make_job(99, cpu=80.0))
    queue = [make_job(1, cpu=30.0), make_job(2, cpu=15.0)]

    assert [job.id for job in select_candidates(queue, [server])] == [2]


def test_head_of_queue_rule_includes_jobs_that_cannot_fit():
    queue = [make_job(1, cpu=200.0), make_job(2)]

    candidates = select_candidates(queue, [make_server()], rule=CandidateRule.HEAD_OF_QUEUE)

    assert [job.id for job in candidates] == [1, 2]


def test_selection_does_not_modify_the_queue():
    queue = [make_job(i) for i in range(1, 9)]
    before = list(queue)

    select_candidates(queue, [make_server()])

    assert queue == before


def test_invalid_limit_is_rejected():
    with pytest.raises(ValueError):
        select_candidates([], [make_server()], limit=0)


def test_servers_that_fit_keeps_cluster_order():
    small, big_a, big_b = make_server(1, cpu=5.0), make_server(2), make_server(3)

    assert servers_that_fit(make_job(1), [small, big_a, big_b]) == [big_a, big_b]


def test_window_refreshes_after_each_placement_within_a_tick():
    # Seven jobs arrive together; the window only shows five at a time, but
    # because it is rebuilt after every placement all seven start at t=0.
    jobs = [make_job(i) for i in range(1, 8)]
    cluster = Cluster(servers=[make_server()])

    Simulator(cluster, FCFSScheduler(), jobs).run()

    assert all(job.start_time == 0 for job in jobs)


class ChoosesHiddenJobScheduler:
    """Tries to place a job that is in the queue but not in its window."""

    name = "cheater"
    candidate_rule = CandidateRule.FEASIBLE

    def __init__(self, hidden_job: Job) -> None:
        self.hidden_job = hidden_job

    def choose(self, candidates, servers):
        return self.hidden_job, servers[0]


def test_simulator_rejects_a_job_outside_the_window():
    jobs = [make_job(i) for i in range(1, 8)]
    hidden = jobs[6]
    cluster = Cluster(servers=[make_server()])
    simulator = Simulator(cluster, ChoosesHiddenJobScheduler(hidden), jobs)

    simulator.step()

    assert hidden.start_time is None
    assert simulator.metrics.rejected_placements == 1


class UsesForeignServerScheduler:
    name = "foreign"
    candidate_rule = CandidateRule.FEASIBLE

    def choose(self, candidates, servers):
        return candidates[0], make_server(42)


def test_simulator_rejects_a_server_outside_the_cluster():
    job = make_job(1)
    simulator = Simulator(Cluster(servers=[make_server()]), UsesForeignServerScheduler(), [job])

    simulator.step()

    assert job.start_time is None
    assert simulator.metrics.rejected_placements == 1
