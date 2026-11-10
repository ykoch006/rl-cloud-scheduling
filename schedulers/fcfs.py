"""First-come-first-served, in two variants.

Both put the chosen job on the first server (in cluster order) that has room:
"first fit". They differ in what happens when the oldest job cannot fit yet.
"""

from env.candidates import CandidateRule, servers_that_fit
from env.job import Job
from env.server import Server


class FCFSScheduler:
    """Oldest candidate that fits, on the first server with room.

    The shared window only contains jobs that fit somewhere, so a large job at
    the front of the queue that cannot fit yet is skipped and younger jobs can
    start instead. This is FCFS with backfilling, not strict FCFS.
    """

    name = "fcfs-backfill"
    candidate_rule = CandidateRule.FEASIBLE

    def choose(self, candidates: list[Job], servers: list[Server]) -> tuple[Job, Server] | None:
        for job in candidates:
            fitting = servers_that_fit(job, servers)
            if fitting:
                return job, fitting[0]
        return None


class StrictFCFSScheduler:
    """Only the oldest waiting job may start; if it does not fit, nothing starts.

    This is head-of-line blocking: younger jobs wait even when there is room
    for them. It sees the head of the queue even when that job cannot fit,
    which the primary policies cannot, so it is a reference policy and is
    reported separately from the primary comparison.
    """

    name = "fcfs-strict"
    candidate_rule = CandidateRule.HEAD_OF_QUEUE

    def choose(self, candidates: list[Job], servers: list[Server]) -> tuple[Job, Server] | None:
        head = candidates[0]
        fitting = servers_that_fit(head, servers)
        return (head, fitting[0]) if fitting else None
