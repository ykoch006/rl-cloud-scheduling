"""First-fit scheduling policy."""

from env.job import Job
from env.server import Server


class FirstFitScheduler:
    """Place each queued job on the first server that has room for it.

    Jobs are considered in queue order and servers in cluster order. A job
    that fits nowhere stays queued, and the scheduler moves on to the next
    job, so a small job that arrived later can start before a large job
    that is still waiting.

    The scheduler only *proposes* placements; it never changes a server or
    job. To avoid proposing two jobs into the same free space, it keeps its
    own running count of the resources it has already promised this
    timestep.
    """

    name = "first-fit"

    def schedule(self, queued_jobs: list[Job], servers: list[Server]) -> list[tuple[Job, Server]]:
        """Return the (job, server) placements to make this timestep."""
        free_cpu = {server.id: server.available_cpu for server in servers}
        free_memory = {server.id: server.available_memory for server in servers}
        placements = []

        for job in queued_jobs:
            for server in servers:
                if (
                    job.cpu_required <= free_cpu[server.id]
                    and job.memory_required <= free_memory[server.id]
                ):
                    free_cpu[server.id] -= job.cpu_required
                    free_memory[server.id] -= job.memory_required
                    placements.append((job, server))
                    break

        return placements
