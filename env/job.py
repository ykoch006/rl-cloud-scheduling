"""Job representation for the cloud scheduling simulator."""

from dataclasses import dataclass, field
from enum import Enum


class JobStatus(Enum):
    """Lifecycle state of a job: queued -> running -> completed."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"


@dataclass
class Job:
    """A unit of work submitted to the cluster.

    A job arrives at `arrival_time`, waits in the queue until it is placed on
    a server, holds `cpu_required` and `memory_required` on that server for
    `duration` timesteps, and then completes.

    Only the first five fields are set by the caller. The rest describe the
    job's progress and are updated by the simulator as the job moves through
    its lifecycle.
    """

    id: int
    arrival_time: int
    duration: int
    cpu_required: float
    memory_required: float
    remaining_time: int = field(init=False)
    start_time: int | None = field(default=None, init=False)
    completion_time: int | None = field(default=None, init=False)
    server_id: int | None = field(default=None, init=False)
    status: JobStatus = field(default=JobStatus.QUEUED, init=False)

    def __post_init__(self) -> None:
        if self.arrival_time < 0:
            raise ValueError(f"Job {self.id}: arrival_time must be >= 0")
        if self.duration < 1:
            raise ValueError(f"Job {self.id}: duration must be >= 1")
        if self.cpu_required < 0 or self.memory_required < 0:
            raise ValueError(f"Job {self.id}: resource requirements must be >= 0")
        self.remaining_time = self.duration

    def start(self, current_time: int, server_id: int) -> None:
        """Mark the job as running on `server_id` from `current_time` onward."""
        self.start_time = current_time
        self.server_id = server_id
        self.status = JobStatus.RUNNING

    def tick(self) -> None:
        """Advance a running job by one timestep."""
        self.remaining_time -= 1

    def is_finished(self) -> bool:
        """Return True once the job has no remaining execution time."""
        return self.remaining_time <= 0

    def complete(self, current_time: int) -> None:
        """Mark the job as completed at `current_time`."""
        self.completion_time = current_time
        self.status = JobStatus.COMPLETED

    @property
    def waiting_time(self) -> int | None:
        """Timesteps spent queued before starting, or None if not yet started."""
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time
