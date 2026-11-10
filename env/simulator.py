"""Discrete-timestep simulation of jobs moving through a cluster."""

from dataclasses import dataclass, field
from statistics import fmean
from typing import Protocol

import numpy as np

from env.candidates import CANDIDATE_LIMIT, CandidateRule, select_candidates
from env.cluster import Cluster
from env.job import Job
from env.server import Server


class Scheduler(Protocol):
    """Anything shaped like this can drive the simulator.

    The scheduler is shown the shared candidate window and the servers and
    picks one (job, server) placement, or None to stop placing for this
    tick. It must not change jobs or servers itself; the simulator applies
    the placement through the cluster, which rejects invalid ones.
    """

    name: str
    candidate_rule: CandidateRule

    def choose(self, candidates: list[Job], servers: list[Server]) -> tuple[Job, Server] | None:
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
    # Jobs still queued, still running, or not yet arrived when the run ended.
    unfinished_jobs: list[Job] = field(default_factory=list)
    total_jobs: int = 0
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
    def unfinished_count(self) -> int:
        return len(self.unfinished_jobs)

    @property
    def completion_rate(self) -> float:
        """Fraction of all jobs in the workload that completed."""
        return self.completed_count / self.total_jobs if self.total_jobs else 1.0

    @property
    def censored_average_waiting_time(self) -> float | None:
        """Mean wait over every job that arrived, including unfinished ones.

        Every job that has started (completed or still running) uses its real
        wait, start - arrival. A job that never started has only waited "at
        least" end - arrival (its wait is right-censored), and that value is
        used, so this is a lower bound on the true average wait. Unlike
        average_waiting_time, it cannot look better just because the slowest
        jobs never started.
        """
        waits = [
            job.waiting_time if job.start_time is not None
            else self.total_duration - job.arrival_time
            for job in self.completed_jobs + self.unfinished_jobs
            if job.arrival_time < self.total_duration
        ]
        return fmean(waits) if waits else None

    @property
    def p95_waiting_time(self) -> float | None:
        """95th percentile wait among completed jobs (NumPy's linear interpolation)."""
        waits = [job.waiting_time for job in self.completed_jobs]
        return float(np.percentile(waits, 95)) if waits else None

    @property
    def max_waiting_time(self) -> int | None:
        waits = [job.waiting_time for job in self.completed_jobs]
        return max(waits) if waits else None

    @property
    def average_turnaround_time(self) -> float | None:
        """Mean completion - arrival (wait + duration) among completed jobs."""
        if not self.completed_jobs:
            return None
        return fmean(job.completion_time - job.arrival_time for job in self.completed_jobs)

    def average_queue_length_over(self, ticks: int) -> float:
        """Mean queue length over the first `ticks` ticks.

        Uses the same window as average_utilisation_over. Ticks after the run
        ended had an empty queue and count as 0.
        """
        if ticks <= 0:
            return 0.0
        idle = [0] * max(0, ticks - len(self.queue_length))
        return fmean(self.queue_length[:ticks] + idle)

    @property
    def average_cpu_utilisation(self) -> float:
        """Mean cluster CPU utilisation over all timesteps."""
        return fmean(self.cluster_cpu_utilisation) if self.cluster_cpu_utilisation else 0.0

    @property
    def average_memory_utilisation(self) -> float:
        """Mean cluster memory utilisation over all timesteps."""
        return fmean(self.cluster_memory_utilisation) if self.cluster_memory_utilisation else 0.0

    def average_utilisation_over(self, ticks: int) -> tuple[float, float]:
        """(CPU, RAM) cluster utilisation averaged over the first `ticks` ticks.

        Runs end at different times for different policies, and the drain
        period after arrivals stop is mostly idle. Comparing policies over the
        same fixed window (normally the arrival horizon) avoids penalising a
        policy just for running longer. If the run ended before `ticks`, the
        cluster was empty for the rest of the window, so those ticks count as 0.
        """
        if ticks <= 0:
            return 0.0, 0.0
        idle = [0.0] * max(0, ticks - len(self.cluster_cpu_utilisation))
        cpu = self.cluster_cpu_utilisation[:ticks] + idle
        memory = self.cluster_memory_utilisation[:ticks] + idle
        return fmean(cpu), fmean(memory)

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
    2. Repeatedly show the scheduler the current candidate window and apply
       its chosen placement, until it returns None or the window is empty.
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

    def __init__(
        self,
        cluster: Cluster,
        scheduler: Scheduler,
        jobs: list[Job],
        candidate_limit: int = CANDIDATE_LIMIT,
    ) -> None:
        self.cluster = cluster
        self.scheduler = scheduler
        self.candidate_limit = candidate_limit
        self.pending_jobs = sorted(jobs, key=lambda job: (job.arrival_time, job.id))
        self.current_time = 0
        self.events: list[tuple[int, str]] = []
        self.metrics = SimulationMetrics(total_jobs=len(jobs))

    def is_finished(self) -> bool:
        """True when every job has arrived and completed."""
        return not (self.pending_jobs or self.cluster.queued_jobs or self.cluster.running_jobs)

    def run(self, max_timesteps: int = 1000) -> SimulationMetrics:
        """Step until all jobs complete or `max_timesteps` is reached.

        The limit stops the loop if some job can never be placed (for
        example, one that is larger than every server). Jobs that have not
        completed by then are recorded in metrics.unfinished_jobs rather than
        silently dropped.
        """
        while not self.is_finished() and self.current_time < max_timesteps:
            self.step()
        self.metrics.total_duration = self.current_time
        self.metrics.unfinished_jobs = (
            list(self.cluster.queued_jobs) + list(self.cluster.running_jobs) + list(self.pending_jobs)
        )
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
        # One placement at a time, rebuilding the window after each, so every
        # decision sees the current free capacity. Each success removes a job
        # from the queue, so the loop runs at most len(queue) times. A rejected
        # placement ends scheduling for this tick, so a faulty scheduler cannot
        # loop forever.
        while True:
            candidates = select_candidates(
                self.cluster.queued_jobs,
                self.cluster.servers,
                self.candidate_limit,
                self.scheduler.candidate_rule,
            )
            if not candidates:
                break
            decision = self.scheduler.choose(candidates, self.cluster.servers)
            if decision is None:
                break
            job, server = decision
            if self._is_allowed(job, server, candidates) and self.cluster.place(
                job, server, self.current_time
            ):
                self._log(self.current_time, f"Job {job.id} assigned to Server {server.id}")
            else:
                self.metrics.rejected_placements += 1
                self._log(self.current_time, f"Job {job.id} rejected by Server {server.id}")
                break

        if self.cluster.queued_jobs:
            waiting = ", ".join(str(job.id) for job in self.cluster.queued_jobs)
            self._log(self.current_time, f"Waiting in queue: jobs {waiting}")

    def _is_allowed(self, job: Job, server: Server, candidates: list[Job]) -> bool:
        # Identity checks, not ==: the scheduler may only pick a job it was
        # shown and a server that belongs to this cluster.
        return any(job is c for c in candidates) and any(
            server is s for s in self.cluster.servers
        )

    def _advance_running_jobs(self) -> None:
        end_of_step = self.current_time + 1
        for job in list(self.cluster.running_jobs):
            job.tick()
            if job.is_finished():
                self.cluster.complete(job, end_of_step)
                self.metrics.completed_jobs.append(job)
                self._log(end_of_step, f"Job {job.id} completed on Server {job.server_id}")
