import math

import pytest

from env.cluster import Cluster
from env.job import Job
from env.server import Server
from env.simulator import Simulator
from env.workload import DETERMINISTIC_WORKLOAD, instantiate_jobs
from evaluation.runner import (
    PRIMARY_POLICIES,
    REFERENCE_POLICIES,
    compare_policies,
    run_policy,
    summarise,
)
from schedulers.fcfs import FCFSScheduler


def one_server_cluster() -> Cluster:
    return Cluster(servers=[Server(id=1, cpu_capacity=100.0, memory_capacity=64.0)])


def blocking_workload() -> list[Job]:
    # Only one job fits at a time, so jobs 2 and 3 queue behind job 1.
    return [Job(id=i, arrival_time=0, duration=4, cpu_required=60.0, memory_required=4.0)
            for i in range(1, 4)]


# --- Unfinished jobs ---------------------------------------------------------


def test_unfinished_jobs_are_counted_when_a_run_is_cut_short():
    jobs = blocking_workload()

    metrics = Simulator(one_server_cluster(), FCFSScheduler(), jobs).run(max_timesteps=6)

    # Job 1 completes at 4; job 2 starts at 4 and is still running; job 3 is queued.
    assert metrics.completed_count == 1
    assert metrics.unfinished_count == 2
    assert metrics.completed_count + metrics.unfinished_count == metrics.total_jobs == 3
    assert metrics.completion_rate == pytest.approx(1 / 3)


def test_censored_wait_includes_jobs_that_never_started():
    metrics = Simulator(one_server_cluster(), FCFSScheduler(), blocking_workload()).run(
        max_timesteps=6
    )

    # Completed-only: job 1 waited 0. Censored: job 1 waited 0, job 2 waited 4,
    # job 3 has waited at least 6 by the end of the run.
    assert metrics.average_waiting_time == 0
    assert metrics.censored_average_waiting_time == pytest.approx((0 + 4 + 6) / 3)


def test_jobs_that_have_not_arrived_are_unfinished_but_not_counted_as_waiting():
    jobs = [Job(id=1, arrival_time=0, duration=1, cpu_required=1.0, memory_required=1.0),
            Job(id=2, arrival_time=50, duration=1, cpu_required=1.0, memory_required=1.0)]

    metrics = Simulator(one_server_cluster(), FCFSScheduler(), jobs).run(max_timesteps=10)

    assert metrics.unfinished_count == 1
    assert metrics.censored_average_waiting_time == 0


def test_censored_and_completed_waits_agree_when_everything_finishes():
    metrics = Simulator(
        Cluster(servers=[Server(id=i, cpu_capacity=100.0, memory_capacity=64.0)
                         for i in range(1, 4)]),
        FCFSScheduler(),
        instantiate_jobs(DETERMINISTIC_WORKLOAD),
    ).run()

    assert metrics.unfinished_count == 0
    assert metrics.censored_average_waiting_time == metrics.average_waiting_time == 0.6


def test_turnaround_is_wait_plus_duration():
    jobs = blocking_workload()

    metrics = Simulator(one_server_cluster(), FCFSScheduler(), jobs).run()

    # Waits 0, 4, 8 and duration 4 each.
    assert metrics.average_turnaround_time == pytest.approx((4 + 8 + 12) / 3)
    assert metrics.max_waiting_time == 8


def test_utilisation_window_ignores_ticks_after_it():
    metrics = Simulator(one_server_cluster(), FCFSScheduler(), blocking_workload()).run()

    cpu, _ = metrics.average_utilisation_over(4)

    assert cpu == pytest.approx(0.6)
    assert metrics.average_utilisation_over(1000) == (
        metrics.average_cpu_utilisation, metrics.average_memory_utilisation
    )


# --- Comparison runner -------------------------------------------------------


def test_runner_gives_every_policy_the_same_workload():
    results = compare_policies("heavy", seeds=[0, 1], horizon=60)

    assert len(results) == 2 * len(PRIMARY_POLICIES)
    for seed in (0, 1):
        totals = {r.metrics.total_jobs for r in results if r.workload_seed == seed}
        assert len(totals) == 1


def test_runner_results_are_reproducible():
    first = summarise(compare_policies("heavy", seeds=[3], horizon=80), horizon=80)
    second = summarise(compare_policies("heavy", seeds=[3], horizon=80), horizon=80)

    assert first == second


def test_every_policy_finishes_given_enough_drain_time():
    policies = PRIMARY_POLICIES | REFERENCE_POLICIES
    results = compare_policies("heavy", seeds=[2], horizon=80, drain=1000, policies=policies)

    assert all(r.metrics.unfinished_count == 0 for r in results)
    assert all(r.metrics.rejected_placements == 0 for r in results)


def test_run_policy_uses_horizon_plus_drain_as_the_time_limit():
    specs = DETERMINISTIC_WORKLOAD

    metrics = run_policy(specs, FCFSScheduler(), horizon=2, drain=1)

    assert metrics.total_duration == 3
    assert metrics.unfinished_count > 0


def test_summary_has_one_row_per_policy_with_finite_values():
    rows = summarise(compare_policies("light", seeds=[0], horizon=40), horizon=40)

    assert [row["policy"] for row in rows] == list(PRIMARY_POLICIES)
    assert all(math.isfinite(v) for row in rows for k, v in row.items() if k != "policy")
