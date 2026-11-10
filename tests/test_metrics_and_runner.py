import math
from statistics import fmean

import pytest

from env.cluster import Cluster
from env.job import Job
from env.server import Server
from env.simulator import SimulationMetrics, Simulator
from env.workload import DETERMINISTIC_WORKLOAD, instantiate_jobs
from env.workload_generator import BURSTY_SCHEDULE, BurstSchedule
from evaluation.runner import (
    PRIMARY_POLICIES,
    REFERENCE_POLICIES,
    RunResult,
    compare_policies,
    queue_by_cycle_phase,
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


def test_queue_window_matches_the_utilisation_window():
    # Queue after scheduling: 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0 over 12 ticks.
    metrics = Simulator(one_server_cluster(), FCFSScheduler(), blocking_workload()).run()

    assert metrics.queue_length == [2] * 4 + [1] * 4 + [0] * 4
    assert metrics.average_queue_length_over(4) == 2
    assert metrics.average_queue_length_over(24) == pytest.approx((8 + 4) / 24)


def test_summary_columns_are_means_of_per_seed_values():
    results = compare_policies("bursty", seeds=[0, 1, 2], horizon=200)
    fcfs = [r.metrics for r in results if r.policy == "fcfs-backfill"]

    row = summarise(results, horizon=200)[0]

    assert row["p95 wait"] == pytest.approx(fmean(m.p95_waiting_time for m in fcfs))
    assert row["wait (done)"] == pytest.approx(fmean(m.average_waiting_time for m in fcfs))
    assert row["queue"] == pytest.approx(fmean(m.average_queue_length_over(200) for m in fcfs))
    assert row["CPU %"] == pytest.approx(100 * fmean(m.average_utilisation_over(200)[0] for m in fcfs))


def test_utilisation_window_counts_ticks_after_the_run_as_idle():
    # The three jobs run back to back for 12 ticks at 60% CPU, then the
    # cluster is empty for the rest of a 24-tick window.
    metrics = Simulator(one_server_cluster(), FCFSScheduler(), blocking_workload()).run()

    cpu, _ = metrics.average_utilisation_over(24)

    assert metrics.total_duration == 12
    assert cpu == pytest.approx(0.6 * 12 / 24)


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


# --- Bursty comparisons ------------------------------------------------------


def test_bursty_runner_gives_every_policy_the_same_workload():
    results = compare_policies("bursty", seeds=[0, 1], horizon=200)

    for seed in (0, 1):
        assert len({r.metrics.total_jobs for r in results if r.workload_seed == seed}) == 1


def test_queue_by_cycle_phase_has_one_column_per_bucket_and_marks_bursts():
    results = compare_policies("bursty", seeds=[0], horizon=200)

    rows = queue_by_cycle_phase(results, BURSTY_SCHEDULE, horizon=200)

    assert [row["policy"] for row in rows] == list(PRIMARY_POLICIES)
    columns = list(rows[0])[1:]
    assert len(columns) == 10
    assert [c for c in columns if c.endswith("*")] == ["80-89*", "90-99*"]


def test_queue_by_cycle_phase_averages_the_recorded_queue_lengths():
    # Hand-made run: cycles of 4 ticks with the burst in the last 2.
    schedule = BurstSchedule(period=4, burst_start=2, burst_length=2, burst_rate=9.0)
    results = [RunResult("p", 0, SimulationMetrics(queue_length=[0, 2, 4, 6, 1, 3, 5, 7]))]

    row = queue_by_cycle_phase(results, schedule, horizon=8, bucket=2)[0]

    assert row == {"policy": "p", "0-1": fmean([0, 2, 1, 3]), "2-3*": fmean([4, 6, 5, 7])}


def test_queue_by_cycle_phase_counts_ticks_after_an_early_finish_as_empty():
    schedule = BurstSchedule(period=4, burst_start=2, burst_length=2, burst_rate=9.0)
    results = [RunResult("p", 0, SimulationMetrics(queue_length=[2, 2]))]

    row = queue_by_cycle_phase(results, schedule, horizon=4, bucket=2)[0]

    assert row["0-1"] == 2 and row["2-3*"] == 0


def test_queue_by_cycle_phase_needs_at_least_one_full_cycle():
    with pytest.raises(ValueError, match="shorter"):
        queue_by_cycle_phase([], BURSTY_SCHEDULE, horizon=50)
