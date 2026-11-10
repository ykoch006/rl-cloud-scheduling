import random

import numpy as np
import pytest

from env.cluster import Cluster
from env.job import Job, JobStatus
from env.server import Server
from env.simulator import Simulator
from env.workload import instantiate_jobs
from env.workload_generator import generate_workload, profile_config
from schedulers.best_fit import BestFitScheduler, leftover_score
from schedulers.fcfs import FCFSScheduler, StrictFCFSScheduler
from schedulers.random_valid import RandomValidScheduler
from schedulers.round_robin import RoundRobinScheduler
from schedulers.sjf import SJFScheduler


def make_job(job_id, cpu=10.0, memory=4.0, duration=3, arrival=0) -> Job:
    return Job(id=job_id, arrival_time=arrival, duration=duration,
               cpu_required=cpu, memory_required=memory)


def make_servers(count=3, cpu=100.0, memory=64.0) -> list[Server]:
    return [Server(id=i, cpu_capacity=cpu, memory_capacity=memory) for i in range(1, count + 1)]


def run(scheduler, jobs, servers):
    simulator = Simulator(Cluster(servers=servers), scheduler, jobs)
    return simulator, simulator.run(max_timesteps=10_000)


# --- FCFS --------------------------------------------------------------------


def test_fcfs_takes_oldest_candidate_on_first_server_with_room():
    full, roomy, spare = make_servers(3)
    full.allocate(make_job(99, cpu=95.0))
    candidates = [make_job(1, cpu=50.0), make_job(2, cpu=5.0)]

    assert FCFSScheduler().choose(candidates, [full, roomy, spare]) == (candidates[0], roomy)


def test_fcfs_returns_none_when_nothing_fits():
    servers = make_servers(1, cpu=10.0)

    assert FCFSScheduler().choose([make_job(1, cpu=50.0)], servers) is None


def test_backfilling_fcfs_lets_a_small_job_pass_a_blocked_one():
    big_first = make_job(1, cpu=80.0, duration=3, arrival=0)
    blocked = make_job(2, cpu=90.0, duration=2, arrival=1)
    small = make_job(3, cpu=10.0, duration=1, arrival=1)

    run(FCFSScheduler(), [big_first, blocked, small], make_servers(1))

    assert small.start_time == 1
    assert blocked.start_time == 3


def test_strict_fcfs_blocks_younger_jobs_behind_the_head():
    big_first = make_job(1, cpu=80.0, duration=3, arrival=0)
    blocked = make_job(2, cpu=90.0, duration=2, arrival=1)
    small = make_job(3, cpu=10.0, duration=1, arrival=1)

    run(StrictFCFSScheduler(), [big_first, blocked, small], make_servers(1))

    # Job 3 fits at t=1 but must wait until job 2 has started.
    assert blocked.start_time == 3
    assert small.start_time == 3


# --- SJF ---------------------------------------------------------------------


def test_sjf_picks_the_shortest_candidate():
    candidates = [make_job(1, duration=8), make_job(2, duration=2), make_job(3, duration=5)]

    job, _ = SJFScheduler().choose(candidates, make_servers(1))

    assert job.id == 2


def test_sjf_breaks_ties_by_age():
    candidates = [make_job(1, duration=4), make_job(2, duration=2), make_job(3, duration=2)]

    job, _ = SJFScheduler().choose(candidates, make_servers(1))

    assert job.id == 2


def test_sjf_cannot_see_a_shorter_job_outside_the_window():
    # Six jobs arrive together; the shortest is sixth, so it is not in the
    # first five-job window. Only one fits at a time, so SJF must start job 1.
    jobs = [make_job(i, cpu=60.0, duration=5) for i in range(1, 6)]
    jobs.append(make_job(6, cpu=60.0, duration=1))

    run(SJFScheduler(), jobs, make_servers(1))

    assert jobs[0].start_time == 0
    assert jobs[5].start_time > 0


# --- Best-Fit ----------------------------------------------------------------


def test_leftover_score_normalises_cpu_and_memory():
    server = make_servers(1)[0]

    score = leftover_score(make_job(1, cpu=50.0, memory=16.0), server)

    assert score == pytest.approx(0.5 + 0.75)


def test_best_fit_chooses_the_tightest_server():
    loose, tight = make_servers(2)
    tight.allocate(make_job(99, cpu=60.0))

    _, server = BestFitScheduler().choose([make_job(1, cpu=30.0)], [loose, tight])

    assert server is tight


def test_best_fit_breaks_ties_by_server_order():
    servers = make_servers(3)

    _, server = BestFitScheduler().choose([make_job(1)], servers)

    assert server is servers[0]


# --- Round-Robin -------------------------------------------------------------


def test_round_robin_rotates_through_servers():
    jobs = [make_job(i) for i in range(1, 5)]

    run(RoundRobinScheduler(), jobs, make_servers(3))

    assert [job.server_id for job in jobs] == [1, 2, 3, 1]


def test_round_robin_skips_servers_without_room():
    servers = make_servers(3)
    servers[1].allocate(make_job(99, cpu=100.0))
    scheduler = RoundRobinScheduler()

    chosen = []
    for i in range(1, 4):
        job, server = scheduler.choose([make_job(i)], servers)
        chosen.append(server.id)

    assert chosen == [1, 3, 1]


# --- Random-valid ------------------------------------------------------------


def placements(scheduler_seed: int):
    jobs = instantiate_jobs(generate_workload(profile_config("medium", seed=5, horizon=60)))
    run(RandomValidScheduler(scheduler_seed), jobs, make_servers(5))
    return [(job.id, job.server_id, job.start_time) for job in jobs]


def test_random_valid_is_reproducible_with_the_same_seed():
    assert placements(scheduler_seed=1) == placements(scheduler_seed=1)


def test_random_valid_depends_on_its_seed():
    assert placements(scheduler_seed=1) != placements(scheduler_seed=2)


def test_random_valid_only_makes_valid_choices():
    servers = make_servers(3)
    servers[0].allocate(make_job(99, cpu=95.0))
    candidates = [make_job(1, cpu=50.0), make_job(2, cpu=3.0)]
    scheduler = RandomValidScheduler(seed=0)

    for _ in range(200):
        job, server = scheduler.choose(candidates, servers)
        assert job in candidates
        assert server.can_fit(job)


def test_random_valid_does_not_touch_global_random_state():
    python_state, numpy_state = random.getstate(), np.random.get_state()

    placements(scheduler_seed=3)

    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and (after[1] == numpy_state[1]).all()


# --- Resource constraints for every policy ----------------------------------

POLICIES = [
    FCFSScheduler,
    StrictFCFSScheduler,
    SJFScheduler,
    BestFitScheduler,
    RoundRobinScheduler,
    lambda: RandomValidScheduler(seed=0),
]


@pytest.mark.parametrize("make_scheduler", POLICIES)
@pytest.mark.parametrize("profile", ["light", "heavy"])
def test_every_policy_respects_capacity_and_timing(make_scheduler, profile):
    specs = generate_workload(profile_config(profile, seed=11, horizon=150))
    jobs = instantiate_jobs(specs)
    servers = make_servers(5)
    simulator = Simulator(Cluster(servers=servers), make_scheduler(), jobs)

    # Check capacity after the scheduling phase of every tick, when the
    # most resources are reserved.
    while not simulator.is_finished() and simulator.current_time < 10_000:
        simulator.step()
        for server in servers:
            used_cpu = sum(job.cpu_required for job in server.running_jobs)
            used_memory = sum(job.memory_required for job in server.running_jobs)
            assert used_cpu <= server.cpu_capacity + 1e-9
            assert used_memory <= server.memory_capacity + 1e-9

    assert simulator.is_finished()
    assert simulator.metrics.rejected_placements == 0
    for job in jobs:
        assert job.status == JobStatus.COMPLETED
        assert job.start_time >= job.arrival_time
        assert job.completion_time == job.start_time + job.duration
    for server in servers:
        assert server.available_cpu == server.cpu_capacity
        assert server.available_memory == server.memory_capacity
        assert server.running_jobs == []
