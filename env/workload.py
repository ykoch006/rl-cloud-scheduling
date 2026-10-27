"""A small, fixed workload for deterministic simulation runs.

Everything is hard-coded (no randomness) so that runs are reproducible and
can be checked by hand. The jobs are sized for three servers with 100 CPU
units and 64 GB of memory each, and are chosen so that:

- some jobs have to wait because no server has enough CPU (job 6),
- some have to wait because no server has enough memory (jobs 5 and 8),
- some later, smaller jobs start before earlier, larger ones that are
  still waiting (jobs 7 and 9).

A random workload generator can replace this later without changing the
simulator.
"""

from env.job import Job

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


def build_deterministic_workload() -> list[Job]:
    """Return a fresh list of 10 jobs numbered 1 to 10."""
    return [
        Job(
            id=job_id,
            arrival_time=arrival_time,
            duration=duration,
            cpu_required=cpu_required,
            memory_required=memory_required,
        )
        for job_id, (arrival_time, duration, cpu_required, memory_required) in enumerate(
            _JOB_SPECS, start=1
        )
    ]
