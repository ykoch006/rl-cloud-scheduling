"""Cluster representation for the cloud scheduling simulator."""

from dataclasses import dataclass, field

from env.job import Job, JobStatus
from env.server import Server


@dataclass
class Cluster:
    """A group of servers plus the queued, running and completed jobs.

    The cluster does not decide where jobs go; that is the scheduler's job.
    The cluster only checks whether a proposed placement is valid, applies
    it if so, and keeps track of each job's state.
    """

    servers: list[Server]
    queued_jobs: list[Job] = field(default_factory=list, init=False)
    running_jobs: list[Job] = field(default_factory=list, init=False)
    completed_jobs: list[Job] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.servers:
            raise ValueError("A cluster needs at least one server")
        server_ids = [server.id for server in self.servers]
        if len(server_ids) != len(set(server_ids)):
            raise ValueError("Server ids must be unique")

    def get_server(self, server_id: int) -> Server:
        """Return the server with the given id."""
        for server in self.servers:
            if server.id == server_id:
                return server
        raise KeyError(f"No server with id {server_id}")

    def enqueue(self, job: Job) -> None:
        """Add a newly arrived job to the back of the queue."""
        self.queued_jobs.append(job)

    def place(self, job: Job, server: Server, current_time: int) -> bool:
        """Try to start a queued `job` on `server` at `current_time`.

        Returns True if the job was placed. Returns False, changing nothing,
        if the job is not in the queue or the server cannot fit it.
        """
        if job.status != JobStatus.QUEUED or job not in self.queued_jobs:
            return False
        if not server.can_fit(job):
            return False

        server.allocate(job)
        job.start(current_time, server.id)
        self.queued_jobs.remove(job)
        self.running_jobs.append(job)
        return True

    def complete(self, job: Job, current_time: int) -> None:
        """Mark a running job as completed and release its server resources."""
        server = self.get_server(job.server_id)
        server.release(job)
        job.complete(current_time)
        self.running_jobs.remove(job)
        self.completed_jobs.append(job)

    def total_cpu_capacity(self) -> float:
        """Total CPU capacity across all servers."""
        return sum(server.cpu_capacity for server in self.servers)

    def total_memory_capacity(self) -> float:
        """Total memory capacity across all servers."""
        return sum(server.memory_capacity for server in self.servers)

    def total_available_cpu(self) -> float:
        """Total free CPU across all servers."""
        return sum(server.available_cpu for server in self.servers)

    def total_available_memory(self) -> float:
        """Total free memory across all servers."""
        return sum(server.available_memory for server in self.servers)

    def cpu_utilisation(self) -> float:
        """Fraction of the cluster's total CPU that is reserved.

        Weighted by capacity, so a busy large server counts for more than a
        busy small one.
        """
        total = self.total_cpu_capacity()
        return (total - self.total_available_cpu()) / total

    def memory_utilisation(self) -> float:
        """Fraction of the cluster's total memory that is reserved (capacity-weighted)."""
        total = self.total_memory_capacity()
        return (total - self.total_available_memory()) / total
