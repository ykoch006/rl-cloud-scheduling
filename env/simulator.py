"""Discrete-timestep simulation of jobs moving through a cluster."""

from dataclasses import dataclass, field
from statistics import fmean
from typing import Protocol

from env.cluster import Cluster
from env.job import Job
from env.server import Server


class Scheduler(Protocol):
    """Anything with a `schedule` method like this can drive the simulator.

    The scheduler sees the queue and the servers and proposes placements.
    It must not change them itself; the simulator applies each placement
    through the cluster, which rejects any that are invalid.
    """

    name: str

    def schedule(self, queued_jobs: list[Job], servers: list[Server]) -> list[tuple[Job, Server]]:
        ...


@dataclass
class SimulationMetrics:
    """Measurements collected while the simulation runs.

    Utilisation and queue length are sampled once per timestep, after
    scheduling, so they describe the state of the cluster during that
    timestep.
    """

    queue_length: list[int] = field(default_factory=list)
    server_cpu_utilisation: dict[int, list[float]] = field(default_factory=dict)
    server_memory_utilisation: dict[int, list[float]] = field(default_factory=dict)
    cluster_cpu_utilisation: list[float] = field(default_factory=list)
    cluster_memory_utilisation: list[float] = field(default_factory=list)
    completed_jobs: list[Job] = field(default_factory=list)
    rejected_placements: int = 0
    total_duration: int = 0

    def record(self, cluster: Cluster) -> None:
        """Take one sample of the cluster's current state."""
        self.queue_length.append(len(cluster.queued_jobs))
        for server in cluster.servers:
            self.server_cpu_utilisation.setdefault(server.id, []).append(server.cpu_utilisation())
            self.server_memory_utilisation.setdefault(server.id, []).append(
                server.memory_utilisation()
            )
        self.cluster_cpu_utilisation.append(cluster.cpu_utilisation())
        self.cluster_memory_utilisation.append(cluster.memory_utilisation())

    @property
    def completed_count(self) -> int:
        return len(self.completed_jobs)

    @property
    def average_waiting_time(self) -> float | None:
        """Mean timesteps completed jobs spent queued, or None if none completed."""
        if not self.completed_jobs:
            return None
        return fmean(job.waiting_time for job in self.completed_jobs)

    @property
    def average_cpu_utilisation(self) -> float:
        """Mean cluster CPU utilisation over all timesteps."""
        return fmean(self.cluster_cpu_utilisation) if self.cluster_cpu_utilisation else 0.0

    @property
    def average_memory_utilisation(self) -> float:
        """Mean cluster memory utilisation over all timesteps."""
        return fmean(self.cluster_memory_utilisation) if self.cluster_memory_utilisation else 0.0

    def average_server_cpu_utilisation(self, server_id: int) -> float:
        """Mean CPU utilisation of one server over all timesteps."""
        return fmean(self.server_cpu_utilisation[server_id])

    def average_server_memory_utilisation(self, server_id: int) -> float:
        """Mean memory utilisation of one server over all timesteps."""
        return fmean(self.server_memory_utilisation[server_id])


class Simulator:
    """Runs a workload through a cluster one timestep at a time.

    Each timestep `t` does, in order:

    1. Move jobs whose arrival_time is `t` into the queue.
    2. Ask the scheduler for placements and apply the valid ones.
    3. Record metrics.
    4. Advance every running job by one timestep. Jobs that finish are
       completed at time `t + 1` and their resources are released, so they
       are free for the scheduler at the start of the next timestep.

    A job with duration `d` that starts at time `s` therefore completes at
    exactly `s + d`.

    The simulator never decides where a job goes. That is entirely up to
    the scheduler passed in, so different policies can be compared on the
    same workload without changing this class.
    """

    def __init__(self, cluster: Cluster, scheduler: Scheduler, jobs: list[Job]) -> None:
        self.cluster = cluster
        self.scheduler = scheduler
        self.pending_jobs = sorted(jobs, key=lambda job: (job.arrival_time, job.id))
        self.current_time = 0
        self.events: list[tuple[int, str]] = []
        self.metrics = SimulationMetrics()

    def is_finished(self) -> bool:
        """True when every job has arrived and completed."""
        return not (self.pending_jobs or self.cluster.queued_jobs or self.cluster.running_jobs)

    def run(self, max_timesteps: int = 1000) -> SimulationMetrics:
        """Step until all jobs complete or `max_timesteps` is reached.

        The limit stops the loop if some job can never be placed (for
        example, one that is larger than every server).
        """
        while not self.is_finished() and self.current_time < max_timesteps:
            self.step()
        self.metrics.total_duration = self.current_time
        return self.metrics

    def step(self) -> None:
        """Simulate a single timestep."""
        self._admit_arrivals()
        self._apply_schedule()
        self.metrics.record(self.cluster)
        self._advance_running_jobs()
        self.current_time += 1

    def _log(self, time: int, message: str) -> None:
        self.events.append((time, message))

    def _admit_arrivals(self) -> None:
        while self.pending_jobs and self.pending_jobs[0].arrival_time <= self.current_time:
            job = self.pending_jobs.pop(0)
            self.cluster.enqueue(job)
            self._log(self.current_time, f"Job {job.id} arrived")

    def _apply_schedule(self) -> None:
        placements = self.scheduler.schedule(list(self.cluster.queued_jobs), self.cluster.servers)
        for job, server in placements:
            if self.cluster.place(job, server, self.current_time):
                self._log(self.current_time, f"Job {job.id} assigned to Server {server.id}")
            else:
                self.metrics.rejected_placements += 1
                self._log(self.current_time, f"Job {job.id} rejected by Server {server.id}")

        if self.cluster.queued_jobs:
            waiting = ", ".join(str(job.id) for job in self.cluster.queued_jobs)
            self._log(self.current_time, f"Waiting in queue: jobs {waiting}")

    def _advance_running_jobs(self) -> None:
        end_of_step = self.current_time + 1
        for job in list(self.cluster.running_jobs):
            job.tick()
            if job.is_finished():
                self.cluster.complete(job, end_of_step)
                self.metrics.completed_jobs.append(job)
                self._log(end_of_step, f"Job {job.id} completed on Server {job.server_id}")
