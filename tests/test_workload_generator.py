import math
from collections import Counter
from dataclasses import replace
from statistics import fmean

import pytest

from env.cluster import Cluster
from env.job import JobStatus
from env.server import Server
from env.simulator import Simulator
from env.workload import DETERMINISTIC_WORKLOAD, JobSpec, instantiate_jobs, validate_workload
from env.workload_generator import (
    DEFAULT_CPU,
    PROFILE_ARRIVAL_RATES,
    REFERENCE_SERVER_CPU,
    REFERENCE_SERVER_MEMORY,
    TruncatedExponentialDuration,
    TruncatedLogNormal,
    WorkloadConfig,
    expected_offered_load,
    generate_workload,
    observed_offered_load,
    profile_config,
)
from schedulers.fcfs import FCFSScheduler


def make_config(**overrides) -> WorkloadConfig:
    values = dict(seed=42, horizon=200, arrival_rate=2.0)
    values.update(overrides)
    return WorkloadConfig(**values)


# --- Reproducibility -------------------------------------------------------


def test_same_seed_and_config_give_identical_workloads():
    config = make_config()

    assert generate_workload(config) == generate_workload(config)


def test_different_seeds_give_different_workloads():
    workloads = [generate_workload(make_config(seed=seed)) for seed in (1, 2, 3, 42)]

    for i in range(len(workloads)):
        for j in range(i + 1, len(workloads)):
            assert workloads[i] != workloads[j]


def test_changing_cpu_distribution_does_not_change_other_properties():
    base = generate_workload(make_config())
    other_cpu = TruncatedLogNormal(median=20.0, sigma=0.5, minimum=1.0, maximum=100.0)
    changed = generate_workload(make_config(cpu=other_cpu))

    assert [s.arrival_time for s in base] == [s.arrival_time for s in changed]
    assert [s.duration for s in base] == [s.duration for s in changed]
    assert [s.memory_required for s in base] == [s.memory_required for s in changed]
    assert [s.cpu_required for s in base] != [s.cpu_required for s in changed]


# --- Arrivals --------------------------------------------------------------


def test_arrival_times_are_integers_within_the_horizon():
    config = make_config(horizon=150)
    specs = generate_workload(config)

    assert specs
    assert all(type(spec.arrival_time) is int for spec in specs)
    assert all(0 <= spec.arrival_time < config.horizon for spec in specs)


def test_arrival_times_are_sorted():
    arrival_times = [spec.arrival_time for spec in generate_workload(make_config())]

    assert arrival_times == sorted(arrival_times)


def test_job_ids_are_unique_and_follow_arrival_order():
    specs = generate_workload(make_config())

    assert [spec.id for spec in specs] == list(range(1, len(specs) + 1))


def test_zero_arrival_rate_produces_no_jobs():
    assert generate_workload(make_config(arrival_rate=0.0)) == ()


def test_zero_horizon_produces_no_jobs():
    assert generate_workload(make_config(horizon=0)) == ()


def test_timesteps_can_have_zero_or_several_arrivals():
    config = make_config(arrival_rate=1.0, horizon=200)
    arrivals_per_step = Counter(spec.arrival_time for spec in generate_workload(config))
    counts = [arrivals_per_step.get(t, 0) for t in range(config.horizon)]

    assert 0 in counts
    assert max(counts) >= 2


def test_heavy_profile_generates_more_jobs_than_light_across_seeds():
    horizon = 200
    seeds = range(30)
    light = fmean(len(generate_workload(profile_config("light", s, horizon))) for s in seeds)
    heavy = fmean(len(generate_workload(profile_config("heavy", s, horizon))) for s in seeds)

    # Expected job count is arrival_rate x horizon.
    assert light == pytest.approx(PROFILE_ARRIVAL_RATES["light"] * horizon, rel=0.05)
    assert heavy == pytest.approx(PROFILE_ARRIVAL_RATES["heavy"] * horizon, rel=0.05)
    assert heavy > light


@pytest.mark.parametrize(
    "profile, low, high", [("light", 0.40, 0.50), ("medium", 0.65, 0.75), ("heavy", 0.85, 0.95)]
)
def test_profiles_hit_their_target_cpu_load_bands(profile, low, high):
    cpu_load, _ = expected_offered_load(profile_config(profile, seed=0, horizon=1))

    assert low <= cpu_load <= high


# --- Job properties --------------------------------------------------------


def test_resource_demands_are_positive_finite_and_within_bounds():
    config = make_config()
    specs = generate_workload(config)

    for spec in specs:
        assert math.isfinite(spec.cpu_required) and math.isfinite(spec.memory_required)
        assert 0 < config.cpu.minimum <= spec.cpu_required <= config.cpu.maximum
        assert 0 < config.memory.minimum <= spec.memory_required <= config.memory.maximum
        assert spec.cpu_required <= REFERENCE_SERVER_CPU
        assert spec.memory_required <= REFERENCE_SERVER_MEMORY


def test_durations_are_positive_integers_within_bounds():
    config = make_config()

    for spec in generate_workload(config):
        assert type(spec.duration) is int
        assert 1 <= spec.duration <= config.duration.maximum


def test_bounds_are_respected_when_truncation_is_frequent():
    tight_cpu = TruncatedLogNormal(median=10.0, sigma=0.8, minimum=5.0, maximum=20.0)
    short = TruncatedExponentialDuration(mean=10.0, maximum=8)
    specs = generate_workload(make_config(cpu=tight_cpu, duration=short))

    assert all(5.0 <= spec.cpu_required <= 20.0 for spec in specs)
    assert all(1 <= spec.duration <= 8 for spec in specs)


def test_sample_means_match_theoretical_expectations():
    config = make_config(seed=7, horizon=3000)
    specs = generate_workload(config)

    assert len(specs) > 5000
    assert fmean(s.cpu_required for s in specs) == pytest.approx(
        config.cpu.expected_value(), rel=0.05
    )
    assert fmean(s.memory_required for s in specs) == pytest.approx(
        config.memory.expected_value(), rel=0.05
    )
    assert fmean(s.duration for s in specs) == pytest.approx(
        config.duration.expected_value(), rel=0.05
    )


def test_observed_offered_load_matches_expected_offered_load():
    config = make_config(seed=7, horizon=3000)
    observed_cpu, observed_memory = observed_offered_load(generate_workload(config), config.horizon)
    expected_cpu, expected_memory = expected_offered_load(config)

    assert observed_cpu == pytest.approx(expected_cpu, rel=0.1)
    assert observed_memory == pytest.approx(expected_memory, rel=0.1)


# --- Validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(arrival_rate=-1.0), "arrival_rate"),
        (dict(arrival_rate=float("nan")), "arrival_rate"),
        (dict(arrival_rate=float("inf")), "arrival_rate"),
        (dict(horizon=-1), "horizon"),
        (dict(horizon=10.5), "horizon"),
        (dict(seed=-1), "seed"),
        (dict(cpu=replace(DEFAULT_CPU, maximum=150.0)), "CPU"),
    ],
)
def test_invalid_workload_config_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_config(**overrides)


def test_memory_maximum_above_server_capacity_is_rejected():
    big_memory = TruncatedLogNormal(median=4.0, sigma=0.9, minimum=0.25, maximum=128.0)

    with pytest.raises(ValueError, match="RAM"):
        make_config(memory=big_memory)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(median=0.0, sigma=0.8, minimum=1.0, maximum=100.0), "median"),
        (dict(median=10.0, sigma=0.0, minimum=1.0, maximum=100.0), "sigma"),
        (dict(median=10.0, sigma=0.8, minimum=0.0, maximum=100.0), "bounds"),
        (dict(median=10.0, sigma=0.8, minimum=50.0, maximum=20.0), "bounds"),
        (dict(median=10.0, sigma=0.8, minimum=90.0, maximum=100.0), "exclude"),
        (dict(median=float("nan"), sigma=0.8, minimum=1.0, maximum=100.0), "finite"),
    ],
)
def test_invalid_log_normal_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TruncatedLogNormal(**kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(mean=0.0, maximum=100), "mean"),
        (dict(mean=-5.0, maximum=100), "mean"),
        (dict(mean=10.0, maximum=0), "maximum"),
        (dict(mean=10.0, maximum=12.5), "maximum"),
        (dict(mean=10.0, maximum=2), "exclude"),
    ],
)
def test_invalid_duration_distribution_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TruncatedExponentialDuration(**kwargs)


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="profile"):
        profile_config("extreme", seed=1, horizon=10)


# --- Fresh job instances ---------------------------------------------------


def test_spec_creates_unstarted_job_with_same_properties():
    spec = JobSpec(id=3, arrival_time=5, duration=4, cpu_required=12.5, memory_required=3.0)

    job = spec.to_job()

    assert (job.id, job.arrival_time, job.duration) == (3, 5, 4)
    assert (job.cpu_required, job.memory_required) == (12.5, 3.0)
    assert job.status == JobStatus.QUEUED
    assert job.remaining_time == 4


def test_workload_can_be_instantiated_multiple_times_independently():
    specs = generate_workload(make_config(horizon=50))

    first = instantiate_jobs(specs)
    second = instantiate_jobs(specs)

    assert len(first) == len(second) == len(specs)
    assert all(a is not b for a, b in zip(first, second))
    assert [(j.id, j.arrival_time, j.duration, j.cpu_required, j.memory_required) for j in first] == [
        (j.id, j.arrival_time, j.duration, j.cpu_required, j.memory_required) for j in second
    ]


def test_simulating_one_copy_does_not_affect_another_copy_or_the_specs():
    specs = generate_workload(make_config(horizon=50, arrival_rate=1.0))
    specs_before = tuple(specs)
    simulated = instantiate_jobs(specs)
    untouched = instantiate_jobs(specs)
    cluster = Cluster(
        servers=[Server(id=i, cpu_capacity=100.0, memory_capacity=64.0) for i in range(1, 6)]
    )

    metrics = Simulator(cluster, FCFSScheduler(), simulated).run(max_timesteps=10_000)

    assert metrics.completed_count == len(specs)
    assert all(job.status == JobStatus.COMPLETED for job in simulated)
    assert all(job.status == JobStatus.QUEUED for job in untouched)
    assert all(job.start_time is None and job.server_id is None for job in untouched)
    assert all(job.remaining_time == job.duration for job in untouched)
    assert specs == specs_before
    assert instantiate_jobs(specs)[0].status == JobStatus.QUEUED


# --- Workload validation against the configured cluster -------------------


def reference_servers(count: int = 5) -> list[Server]:
    return [Server(id=i, cpu_capacity=100.0, memory_capacity=64.0) for i in range(1, count + 1)]


def spec(job_id=1, arrival_time=0, duration=3, cpu=10.0, memory=4.0) -> JobSpec:
    return JobSpec(id=job_id, arrival_time=arrival_time, duration=duration,
                   cpu_required=cpu, memory_required=memory)


@pytest.mark.parametrize("profile", sorted(PROFILE_ARRIVAL_RATES))
def test_generated_profiles_pass_validation(profile):
    specs = generate_workload(profile_config(profile, seed=3, horizon=300))

    validate_workload(specs, reference_servers())


def test_deterministic_workload_passes_validation():
    validate_workload(DETERMINISTIC_WORKLOAD, reference_servers(3))


def test_empty_workload_passes_validation():
    validate_workload((), reference_servers())


@pytest.mark.parametrize(
    "specs, message",
    [
        ((spec(job_id=1), spec(job_id=1)), "unique"),
        ((spec(job_id=1, arrival_time=5), spec(job_id=2, arrival_time=2)), "arrival order"),
        ((spec(cpu=0.0),), "> 0"),
        ((spec(memory=float("nan")),), "finite"),
        ((spec(cpu=150.0),), "do not fit"),
        ((spec(memory=80.0),), "do not fit"),
    ],
)
def test_invalid_workloads_are_rejected(specs, message):
    with pytest.raises(ValueError, match=message):
        validate_workload(specs, reference_servers())


def test_fit_check_uses_the_configured_servers_not_a_fixed_reference():
    small_cluster = [Server(id=1, cpu_capacity=50.0, memory_capacity=16.0)]
    big_cluster = [Server(id=1, cpu_capacity=50.0, memory_capacity=16.0),
                   Server(id=2, cpu_capacity=100.0, memory_capacity=64.0)]
    job = (spec(cpu=80.0, memory=8.0),)

    with pytest.raises(ValueError, match=r"\[1\]"):
        validate_workload(job, small_cluster)
    validate_workload(job, big_cluster)
