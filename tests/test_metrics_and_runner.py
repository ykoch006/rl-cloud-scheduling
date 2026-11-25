import math
from statistics import fmean

import numpy as np
import pytest

from env.cluster import Cluster
from env.job import Job
from env.server import Server
from env.simulator import SLA_SLOWDOWN, SimulationMetrics, Simulator, sla_deadline
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

    assert row["wait LB"] == pytest.approx(fmean(m.censored_average_waiting_time for m in fcfs))
    assert row["p95 wait LB"] == pytest.approx(fmean(m.censored_p95_waiting_time for m in fcfs))
    assert row["turnaround LB"] == pytest.approx(
        fmean(m.censored_average_turnaround_time for m in fcfs)
    )
    assert row["wait (done)"] == pytest.approx(fmean(m.average_waiting_time for m in fcfs))
    assert row["SLA viol %"] == pytest.approx(100 * fmean(m.sla().violation_rate for m in fcfs))
    assert row["SLA unknown %"] == pytest.approx(100 * fmean(m.sla().unknown_rate for m in fcfs))
    assert row["lateness"] == pytest.approx(fmean(m.sla().mean_lateness for m in fcfs))
    assert row["queue"] == pytest.approx(fmean(m.average_queue_length_over(200) for m in fcfs))
    assert row["CPU %"] == pytest.approx(100 * fmean(m.average_utilisation_over(200)[0] for m in fcfs))


def test_utilisation_window_counts_ticks_after_the_run_as_idle():
    # The three jobs run back to back for 12 ticks at 60% CPU, then the
    # cluster is empty for the rest of a 24-tick window.
    metrics = Simulator(one_server_cluster(), FCFSScheduler(), blocking_workload()).run()

    cpu, _ = metrics.average_utilisation_over(24)

    assert metrics.total_duration == 12
    assert cpu == pytest.approx(0.6 * 12 / 24)


@pytest.mark.parametrize("ticks", [0, -1, 2.5, True, np.bool_(True)])
def test_measurement_windows_must_be_whole_positive_tick_counts(ticks):
    metrics = Simulator(one_server_cluster(), FCFSScheduler(), blocking_workload()).run()

    with pytest.raises(ValueError, match="ticks"):
        metrics.average_queue_length_over(ticks)
    with pytest.raises(ValueError, match="ticks"):
        metrics.average_utilisation_over(ticks)


def test_empty_workload_has_no_job_metrics():
    metrics = Simulator(one_server_cluster(), FCFSScheduler(), []).run()

    assert metrics.total_jobs == 0
    assert metrics.completion_rate is None
    assert metrics.average_waiting_time is None
    assert metrics.censored_average_waiting_time is None


def test_censored_p95_and_turnaround_include_unfinished_jobs():
    metrics = Simulator(one_server_cluster(), FCFSScheduler(), blocking_workload()).run(
        max_timesteps=6
    )

    # Censored waits 0, 4, 6 (as above). p95 by linear interpolation: rank
    # 0.95 x 2 = 1.9, so 4 + 0.9 x (6 - 4) = 5.8. Turnaround: job 1 completed
    # at 4; job 2 is running and completes at 4 + 4 = 8; job 3 cannot finish
    # before 6 + 4 = 10. So (4 + 8 + 10) / 3.
    assert metrics.p95_waiting_time == 0
    assert metrics.censored_p95_waiting_time == pytest.approx(5.8)
    assert metrics.censored_average_turnaround_time == pytest.approx(22 / 3)


# --- SLA -----------------------------------------------------------------------


def sla_workload() -> list[Job]:
    # One server; every job needs all of its CPU, so FCFS runs them one after
    # another in id order. Hand trace (start, completion):
    #   job 1, d=2: (0, 2)   job 2, d=1: (2, 3)   job 3, d=1: (3, 4)
    #   job 4, d=4: (4, 8)   job 5, d=1: (8, 9)
    durations = [2, 1, 1, 4, 1]
    return [Job(id=i, arrival_time=0, duration=d, cpu_required=100.0, memory_required=1.0)
            for i, d in enumerate(durations, start=1)]


def run_sla_workload(max_timesteps: int = 1000):
    return Simulator(one_server_cluster(), FCFSScheduler(), sla_workload()).run(max_timesteps)


def test_default_sla_slowdown_is_three():
    job = Job(id=1, arrival_time=5, duration=4, cpu_required=1.0, memory_required=1.0)

    assert SLA_SLOWDOWN == 3
    assert sla_deadline(job) == 17


def test_sla_on_time_boundary_and_late_completions():
    # k = 3, deadlines 0 + 3d: 6, 3, 3, 12, 3. Completions 2, 3, 4, 8, 9.
    # Job 1 on time; job 2 exactly on its deadline (on time); job 3 late by 1;
    # job 4 on time; job 5 late by 6.
    sla = run_sla_workload().sla()

    assert (sla.jobs, sla.violations, sla.unknown) == (5, 2, 0)
    assert sla.total_lateness == 7
    assert sla.violation_rate == pytest.approx(0.4)
    assert sla.mean_lateness == pytest.approx(1.4)


def test_sla_slowdown_is_configurable():
    # k = 2, deadlines 4, 2, 2, 8, 2. Completions 2, 3, 4, 8, 9.
    # Late: job 2 by 1, job 3 by 2, job 5 by 7. Job 4 finishes exactly at 8.
    sla = run_sla_workload().sla(slowdown=2)

    assert (sla.violations, sla.unknown, sla.total_lateness) == (3, 0, 10)
    # k = 5, deadlines 10, 5, 5, 20, 5: only job 5 (completes 9) is late, by 4.
    assert run_sla_workload().sla(slowdown=5).total_lateness == 4


def test_unfinished_jobs_are_confirmed_late_or_unknown():
    # Cut at 6: jobs 1-3 completed as above. Job 4 is running (started 4) and
    # cannot finish before 8, its deadline is 12: unknown. Job 5 is queued and
    # cannot finish before 6 + 1 = 7, after its deadline 3: a confirmed
    # violation, at least 4 ticks late.
    metrics = run_sla_workload(max_timesteps=6)
    sla = metrics.sla()

    assert metrics.unfinished_count == 2
    assert (sla.violations, sla.unknown) == (2, 1)
    assert sla.total_lateness == 1 + 4
    assert sla.violation_rate == pytest.approx(2 / 5)
    assert sla.unknown_rate == pytest.approx(1 / 5)
    assert sla.mean_lateness == pytest.approx(1.0)


def test_unfinished_job_that_can_still_finish_exactly_on_time_is_unknown():
    # Cut at 2: job 1 completed at 2. Job 2 (deadline 3) is queued and could
    # finish at 2 + 1 = 3, exactly on time, so its outcome is unknown.
    jobs = sla_workload()[:2]
    metrics = Simulator(one_server_cluster(), FCFSScheduler(), jobs).run(max_timesteps=2)
    sla = metrics.sla()

    assert (sla.violations, sla.unknown, sla.total_lateness) == (0, 1, 0)


def test_job_that_has_not_arrived_is_unknown():
    jobs = [Job(id=1, arrival_time=50, duration=1, cpu_required=1.0, memory_required=1.0)]

    sla = Simulator(one_server_cluster(), FCFSScheduler(), jobs).run(max_timesteps=10).sla()

    assert (sla.jobs, sla.violations, sla.unknown, sla.total_lateness) == (1, 0, 1, 0)


def test_sla_of_an_empty_workload_is_undefined():
    sla = Simulator(one_server_cluster(), FCFSScheduler(), []).run().sla()

    assert (sla.jobs, sla.violations, sla.unknown) == (0, 0, 0)
    assert sla.violation_rate is None and sla.unknown_rate is None
    assert sla.mean_lateness is None


@pytest.mark.parametrize(
    "slowdown", [0.5, 0, -1, float("nan"), float("inf"), True, np.bool_(True)]
)
def test_invalid_sla_slowdown_is_rejected(slowdown):
    with pytest.raises(ValueError, match="slowdown"):
        run_sla_workload().sla(slowdown=slowdown)


# --- Utilisation samples behind the energy argument ----------------------------


def test_per_server_utilisation_covers_idle_partial_and_full_ticks():
    # Two servers, first fit. Job 1 fills server 1 for ticks 0-1; job 2 uses
    # half of server 2 for ticks 0-2; server 1 is idle in tick 2.
    servers = [Server(id=i, cpu_capacity=100.0, memory_capacity=64.0) for i in (1, 2)]
    jobs = [Job(id=1, arrival_time=0, duration=2, cpu_required=100.0, memory_required=8.0),
            Job(id=2, arrival_time=0, duration=3, cpu_required=50.0, memory_required=8.0)]

    metrics = Simulator(Cluster(servers=servers), FCFSScheduler(), jobs).run()

    assert metrics.server_cpu_utilisation == {1: [1.0, 1.0, 0.0], 2: [0.5, 0.5, 0.5]}
    assert metrics.cluster_cpu_utilisation == [0.75, 0.75, 0.25]
    # Reserved CPU-time: 100 x 2 + 50 x 3 = 350 unit-ticks out of 200 x 3.
    assert sum(metrics.cluster_cpu_utilisation) * 200 == pytest.approx(350)


def test_reserved_cpu_time_over_a_complete_run_is_the_same_for_every_policy():
    # Why an always-on linear energy model cannot separate policies: once all
    # jobs finish, the CPU-time reserved equals sum(cpu x duration) whatever
    # the schedule (docs/EXPERIMENT_PROTOCOL.md).
    policies = PRIMARY_POLICIES | REFERENCE_POLICIES
    results = compare_policies("heavy", seeds=[0], horizon=100, policies=policies)
    jobs = results[0].metrics.completed_jobs
    work = sum(job.cpu_required * job.duration for job in jobs)

    for result in results:
        assert result.metrics.unfinished_count == 0
        assert sum(result.metrics.cluster_cpu_utilisation) * 500 == pytest.approx(work)


# --- Comparison runner -------------------------------------------------------


@pytest.mark.parametrize(
    "horizon, drain", [(-5, 10), (10, -1), (2.5, 10), (10, 2.5), (True, 10), (np.bool_(True), 10)]
)
def test_run_policy_rejects_invalid_horizon_or_drain(horizon, drain):
    with pytest.raises(ValueError):
        run_policy(DETERMINISTIC_WORKLOAD, FCFSScheduler(), horizon=horizon, drain=drain)



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


def test_summary_passes_the_sla_slowdown_through():
    results = compare_policies("heavy", seeds=[0, 1], horizon=100)

    strict = summarise(results, horizon=100, sla_slowdown=1)[0]
    loose = summarise(results, horizon=100, sla_slowdown=5)[0]

    fcfs = [r.metrics for r in results if r.policy == "fcfs-backfill"]
    assert strict["SLA viol %"] == pytest.approx(
        100 * fmean(m.sla(slowdown=1).violation_rate for m in fcfs)
    )
    assert strict["SLA viol %"] > loose["SLA viol %"]


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
