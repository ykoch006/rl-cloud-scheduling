"""Every policy in a comparison must be evaluated on exactly the same workload."""

from dataclasses import fields

import pytest

import evaluation.runner as runner
from env.candidates import CANDIDATE_LIMIT, CandidateRule
from env.cluster import Cluster
from env.job import Job, JobStatus
from env.simulator import Simulator
from env.workload import JobSpec, instantiate_jobs
from env.workload_generator import PROFILES, generate_workload, profile_config
from evaluation.runner import (
    PRIMARY_POLICIES,
    REFERENCE_POLICIES,
    compare_policies,
    reference_servers,
)

HORIZON = 150
SEEDS = [0, 1]
ALL_POLICIES = PRIMARY_POLICIES | REFERENCE_POLICIES
SPEC_FIELDS = [f.name for f in fields(JobSpec)]


def outcomes(metrics) -> dict[int, tuple]:
    """Each job's (server, start, completion) at the end of a run."""
    return {
        job.id: (job.server_id, job.start_time, job.completion_time)
        for job in metrics.completed_jobs + metrics.unfinished_jobs
    }


# --- Identical specifications ------------------------------------------------


def test_every_spec_attribute_is_carried_onto_the_job():
    # If a new attribute is added to JobSpec, the job comparisons below only
    # stay complete if Job has it too.
    assert set(SPEC_FIELDS) <= {f.name for f in fields(Job)}


@pytest.mark.parametrize("profile", PROFILES)
def test_every_policy_starts_from_identical_fresh_jobs(profile, monkeypatch):
    received = []

    def recording_instantiate(specs):
        jobs = instantiate_jobs(specs)
        # Snapshot at hand-over, before this policy's run changes anything.
        snapshot = [
            tuple(getattr(job, name) for name in SPEC_FIELDS)
            + (job.remaining_time, job.status, job.start_time, job.completion_time, job.server_id)
            for job in jobs
        ]
        received.append((id(jobs), snapshot, jobs))
        return jobs

    monkeypatch.setattr(runner, "instantiate_jobs", recording_instantiate)
    compare_policies(profile, SEEDS, HORIZON, policies=ALL_POLICIES)

    assert len(received) == len(SEEDS) * len(ALL_POLICIES)
    for index, seed in enumerate(SEEDS):
        # Regenerated independently of the runner.
        specs = generate_workload(profile_config(profile, seed, HORIZON))
        expected = [
            tuple(getattr(spec, name) for name in SPEC_FIELDS)
            + (spec.duration, JobStatus.QUEUED, None, None, None)
            for spec in specs
        ]
        runs = received[index * len(ALL_POLICIES):(index + 1) * len(ALL_POLICIES)]
        assert expected
        assert all(snapshot == expected for _, snapshot, _ in runs)

    # No Job object is shared between two runs.
    job_ids = [id(job) for _, _, jobs in received for job in jobs]
    assert len(job_ids) == len(set(job_ids))


@pytest.mark.parametrize("profile", PROFILES)
def test_same_seed_and_profile_give_identical_specifications(profile):
    first = generate_workload(profile_config(profile, 5, HORIZON))
    second = generate_workload(profile_config(profile, 5, HORIZON))

    assert first == second
    assert all(type(spec) is JobSpec for spec in first)


@pytest.mark.parametrize("profile", PROFILES)
def test_different_seeds_change_every_generated_attribute(profile):
    workloads = [generate_workload(profile_config(profile, seed, HORIZON)) for seed in range(4)]

    for i in range(len(workloads)):
        for j in range(i + 1, len(workloads)):
            for name in SPEC_FIELDS[1:]:  # ids are 1..n in every workload
                assert [getattr(s, name) for s in workloads[i]] != [
                    getattr(s, name) for s in workloads[j]
                ]


# --- No contamination between runs --------------------------------------------


@pytest.mark.parametrize("profile", PROFILES)
def test_policy_results_do_not_depend_on_run_order(profile):
    forward = compare_policies(profile, [3], HORIZON, policies=ALL_POLICIES)
    reversed_policies = dict(reversed(list(ALL_POLICIES.items())))
    backward = compare_policies(profile, [3], HORIZON, policies=reversed_policies)

    by_policy = {r.policy: r.metrics for r in backward}
    for result in forward:
        other = by_policy[result.policy]
        alone = compare_policies(profile, [3], HORIZON,
                                 policies={result.policy: ALL_POLICIES[result.policy]})[0].metrics
        assert outcomes(result.metrics) == outcomes(other) == outcomes(alone)
        assert result.metrics.queue_length == other.queue_length == alone.queue_length


# --- Candidate windows --------------------------------------------------------


class WindowChecker:
    """Wraps a policy and checks each window it is shown against the queue.

    The expected window is rebuilt here from the queue and the servers' free
    capacity, without calling env.candidates.
    """

    def __init__(self, policy) -> None:
        self.policy = policy
        self.name = policy.name
        self.candidate_rule = policy.candidate_rule
        self.simulator: Simulator | None = None
        self.windows: list[list[int]] = []

    def choose(self, candidates, servers):
        queue = self.simulator.cluster.queued_jobs
        if self.candidate_rule is CandidateRule.FEASIBLE:
            eligible = [
                job for job in queue
                if any(job.cpu_required <= s.available_cpu
                       and job.memory_required <= s.available_memory for s in servers)
            ]
        else:
            eligible = list(queue)
        expected = eligible[:CANDIDATE_LIMIT]

        assert [id(job) for job in candidates] == [id(job) for job in expected]
        assert [(j.arrival_time, j.id) for j in candidates] == sorted(
            (j.arrival_time, j.id) for j in candidates
        )
        self.windows.append([job.id for job in candidates])
        return self.policy.choose(candidates, servers)


def test_primary_policies_share_one_window_rule():
    assert {make(0).candidate_rule for make in PRIMARY_POLICIES.values()} == {CandidateRule.FEASIBLE}
    assert {make(0).candidate_rule for make in REFERENCE_POLICIES.values()} == {
        CandidateRule.HEAD_OF_QUEUE
    }


@pytest.mark.parametrize("profile", PROFILES)
def test_every_window_follows_the_shared_rule(profile):
    specs = generate_workload(profile_config(profile, 1, HORIZON))
    first_windows = {}

    for name, make_policy in ALL_POLICIES.items():
        checker = WindowChecker(make_policy(1))
        simulator = Simulator(Cluster(servers=reference_servers()), checker, instantiate_jobs(specs))
        checker.simulator = simulator
        assert simulator.candidate_limit == CANDIDATE_LIMIT

        metrics = simulator.run(max_timesteps=HORIZON + runner.DEFAULT_DRAIN)

        assert checker.windows and metrics.unfinished_count == 0
        first_windows[name] = checker.windows[0]

    # The first decision is made from the same state for every policy, so the
    # first window is the same; later windows may differ because earlier
    # decisions leave different jobs queued and different capacity free.
    assert len({tuple(first_windows[name]) for name in PRIMARY_POLICIES}) == 1
