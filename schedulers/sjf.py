"""Shortest-job-first, limited to the candidate window."""

from env.candidates import CandidateRule, servers_that_fit
from env.job import Job
from env.server import Server


class SJFScheduler:
    """Shortest candidate (by duration) first, on the first server with room.

    It only compares jobs in the shared window, so a short job further back in
    the queue is not seen. This is window-limited SJF, not global SJF.
    Durations are treated as perfect runtime estimates. Ties go to the older
    job, because min() returns the first of equal values and the window is in
    queue order.
    """

    name = "sjf"
    candidate_rule = CandidateRule.FEASIBLE

    def choose(self, candidates: list[Job], servers: list[Server]) -> tuple[Job, Server] | None:
        options = [(job, servers_that_fit(job, servers)) for job in candidates]
        options = [(job, fitting[0]) for job, fitting in options if fitting]
        if not options:
            return None
        return min(options, key=lambda option: option[0].duration)
