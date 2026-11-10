"""The shared candidate-job window.

Every scheduler in the primary comparison sees the same short list of waiting
jobs. The RL agents will only observe a fixed number of jobs, so a heuristic
that could inspect the whole queue would have an unfair information advantage
(scope §5.2).
"""

from collections.abc import Sequence
from enum import Enum

from env.job import Job
from env.server import Server

CANDIDATE_LIMIT = 5


class CandidateRule(Enum):
    # Oldest jobs that fit on at least one server right now. Used by every
    # primary policy. Skipping jobs that cannot fit stops a few large blocked
    # jobs at the front from hiding younger jobs that could run.
    FEASIBLE = "feasible"
    # Oldest jobs whether or not they fit. Only strict FCFS uses this, because
    # it needs to know when the head of the queue is blocked.
    HEAD_OF_QUEUE = "head-of-queue"


def servers_that_fit(job: Job, servers: Sequence[Server]) -> list[Server]:
    """Servers with enough free CPU and RAM for `job` now, in cluster order."""
    return [server for server in servers if server.can_fit(job)]


def select_candidates(
    queue: Sequence[Job],
    servers: Sequence[Server],
    limit: int = CANDIDATE_LIMIT,
    rule: CandidateRule = CandidateRule.FEASIBLE,
) -> list[Job]:
    """Return up to `limit` waiting jobs, oldest first.

    "Oldest" means queue order. The simulator enqueues jobs by
    (arrival_time, id), so ties between jobs that arrived in the same tick are
    broken by id. The queue itself is never modified, so a job left out of the
    window is not discarded.
    """
    if limit < 1:
        raise ValueError(f"candidate limit must be >= 1, got {limit}")
    if rule is CandidateRule.HEAD_OF_QUEUE:
        return list(queue[:limit])

    candidates = []
    for job in queue:
        if servers_that_fit(job, servers):
            candidates.append(job)
            if len(candidates) == limit:
                break
    return candidates
