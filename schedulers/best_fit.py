"""Best-fit server selection."""

from env.candidates import CandidateRule, servers_that_fit
from env.job import Job
from env.server import Server


def leftover_score(job: Job, server: Server) -> float:
    """Capacity left on `server` after placing `job`; lower is a tighter fit.

    CPU and RAM are each divided by the server's capacity so both are on a
    0-1 scale and neither unit dominates. The score is their sum (0 to 2).
    """
    cpu_left = (server.available_cpu - job.cpu_required) / server.cpu_capacity
    memory_left = (server.available_memory - job.memory_required) / server.memory_capacity
    return cpu_left + memory_left


class BestFitScheduler:
    """Oldest candidate, on the server it fits most tightly.

    Packing jobs tightly leaves larger free gaps elsewhere for big jobs.
    Ties go to the earliest server in cluster order.
    """

    name = "best-fit"
    candidate_rule = CandidateRule.FEASIBLE

    def choose(self, candidates: list[Job], servers: list[Server]) -> tuple[Job, Server] | None:
        for job in candidates:
            fitting = servers_that_fit(job, servers)
            if fitting:
                return job, min(fitting, key=lambda server: leftover_score(job, server))
        return None
