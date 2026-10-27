"""Server representation for the cloud scheduling simulator."""

from dataclasses import dataclass, field

from env.job import Job


@dataclass
class Server:
    """A single machine with fixed CPU and memory capacity.

    A job reserves its CPU and memory for its whole duration: `allocate`
    subtracts from the available resources and `release` gives them back.
    `allocate` refuses any job that does not fit, so the reserved amount can
    never exceed the server's capacity.
    """

    id: int
    cpu_capacity: float
    memory_capacity: float
    available_cpu: float = field(init=False)
    available_memory: float = field(init=False)
    running_jobs: list[Job] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.cpu_capacity <= 0 or self.memory_capacity <= 0:
            raise ValueError(f"Server {self.id}: capacities must be > 0")
        self.available_cpu = self.cpu_capacity
        self.available_memory = self.memory_capacity

    def can_fit(self, job: Job) -> bool:
        """Return True if the server currently has enough free CPU and memory for `job`."""
        return (
            job.cpu_required <= self.available_cpu
            and job.memory_required <= self.available_memory
        )

    def allocate(self, job: Job) -> None:
        """Reserve resources for `job` and add it to the running jobs.

        Raises ValueError if the job does not fit.
        """
        if not self.can_fit(job):
            raise ValueError(
                f"Server {self.id} cannot fit job {job.id}: "
                f"needs cpu={job.cpu_required}, memory={job.memory_required}; "
                f"available cpu={self.available_cpu}, memory={self.available_memory}"
            )
        self.available_cpu -= job.cpu_required
        self.available_memory -= job.memory_required
        self.running_jobs.append(job)

    def release(self, job: Job) -> None:
        """Free the resources held by `job`.

        Raises ValueError if the job is not running on this server.
        """
        if job not in self.running_jobs:
            raise ValueError(f"Job {job.id} is not running on server {self.id}")
        self.running_jobs.remove(job)
        self.available_cpu += job.cpu_required
        self.available_memory += job.memory_required

    def cpu_utilisation(self) -> float:
        """Fraction of CPU capacity currently reserved, between 0 and 1."""
        return (self.cpu_capacity - self.available_cpu) / self.cpu_capacity

    def memory_utilisation(self) -> float:
        """Fraction of memory capacity currently reserved, between 0 and 1."""
        return (self.memory_capacity - self.available_memory) / self.memory_capacity
