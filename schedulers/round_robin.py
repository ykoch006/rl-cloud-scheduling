"""Round-robin server selection."""

from env.candidates import CandidateRule
from env.job import Job
from env.server import Server


class RoundRobinScheduler:
    """Oldest candidate, on the next server in rotation that has room.

    Servers are tried in cluster order starting just after the one that
    received the previous job, wrapping around, so work is spread across
    servers rather than packed. The rotation position carries over between
    ticks, so use a new instance for each simulation run.
    """

    name = "round-robin"
    candidate_rule = CandidateRule.FEASIBLE

    def __init__(self) -> None:
        self._last_index = -1

    def choose(self, candidates: list[Job], servers: list[Server]) -> tuple[Job, Server] | None:
        for job in candidates:
            for offset in range(1, len(servers) + 1):
                index = (self._last_index + offset) % len(servers)
                if servers[index].can_fit(job):
                    self._last_index = index
                    return job, servers[index]
        return None
