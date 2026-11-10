"""Workload specifications and a small fixed workload for deterministic runs.

The simulator changes `Job` objects as they run (start time, remaining time,
status, ...), so a workload is stored as immutable `JobSpec`s and turned into
fresh `Job` objects each time it is simulated. Two schedulers can then be
compared on identical jobs without one run affecting the other.

The fixed workload below is hard-coded (no randomness) so that runs can be
checked by hand. The jobs are sized for three servers with 100 CPU units and
64 GB of memory each, and are chosen so that:

- some jobs have to wait because no server has enough CPU (job 6),
- some have to wait because no server has enough memory (jobs 5 and 8),
- some later, smaller jobs start before earlier, larger ones that are
  still waiting (jobs 7 and 9).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from env.job import Job
from env.server import Server


@dataclass(frozen=True)
class JobSpec:
    """The fixed description of one job, independent of any simulation run."""

    id: int
    arrival_time: int
    duration: int
    cpu_required: float
    memory_required: float

    def to_job(self) -> Job:
        """Create a new, unstarted Job with these properties."""
        return Job(
            id=self.id,
            arrival_time=self.arrival_time,
            duration=self.duration,
            cpu_required=self.cpu_required,
            memory_required=self.memory_required,
        )


def instantiate_jobs(specs: Sequence[JobSpec]) -> list[Job]:
    """Create fresh Job objects for a workload, one per spec."""
    return [spec.to_job() for spec in specs]


def validate_workload(specs: Sequence[JobSpec], servers: Sequence[Server]) -> None:
    """Raise ValueError if the workload cannot be simulated safely on `servers`.

    A job that does not fit on any server even when that server is empty
    would wait forever, so it is rejected here rather than silently
    inflating the unfinished-job count.
    """
    ids = [spec.id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Job ids must be unique")
    arrival_times = [spec.arrival_time for spec in specs]
    if arrival_times != sorted(arrival_times):
        raise ValueError("Jobs must be listed in arrival order")

    for spec in specs:
        if spec.arrival_time < 0 or spec.duration < 1:
            raise ValueError(f"Job {spec.id}: needs arrival_time >= 0 and duration >= 1")
        demands = (spec.cpu_required, spec.memory_required)
        if not all(math.isfinite(d) and d > 0 for d in demands):
            raise ValueError(f"Job {spec.id}: CPU and RAM demands must be finite and > 0")

    fits_nowhere = [
        spec.id
        for spec in specs
        if not any(
            spec.cpu_required <= server.cpu_capacity
            and spec.memory_required <= server.memory_capacity
            for server in servers
        )
    ]
    if fits_nowhere:
        raise ValueError(f"Jobs {fits_nowhere} do not fit on any server, even an empty one")


# (arrival_time, duration, cpu_required, memory_required)
_JOB_SPECS: list[tuple[int, int, float, float]] = [
    (0, 4, 60.0, 20.0),
    (0, 3, 50.0, 30.0),
    (0, 5, 70.0, 16.0),
    (1, 3, 40.0, 24.0),
    (1, 2, 30.0, 50.0),
    (2, 4, 80.0, 10.0),
    (2, 2, 20.0, 40.0),
    (3, 3, 50.0, 20.0),
    (4, 2, 30.0, 30.0),
    (5, 3, 60.0, 12.0),
]

DETERMINISTIC_WORKLOAD: tuple[JobSpec, ...] = tuple(
    JobSpec(
        id=job_id,
        arrival_time=arrival_time,
        duration=duration,
        cpu_required=cpu_required,
        memory_required=memory_required,
    )
    for job_id, (arrival_time, duration, cpu_required, memory_required) in enumerate(
        _JOB_SPECS, start=1
    )
)


def build_deterministic_workload() -> list[Job]:
    """Return a fresh list of 10 jobs numbered 1 to 10."""
    return instantiate_jobs(DETERMINISTIC_WORKLOAD)
