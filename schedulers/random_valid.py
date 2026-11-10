"""Random-valid reference policy."""

import numpy as np

from env.candidates import CandidateRule, servers_that_fit
from env.job import Job
from env.server import Server


class RandomValidScheduler:
    """A random candidate on a random server that has room for it.

    Every choice is valid, but no preference is used, so this is the "no
    intelligence" reference that a learned policy must at least beat. It uses
    its own seeded generator, so runs are reproducible and global random state
    is untouched. It always places a job when one fits; it never waits on
    purpose.
    """

    name = "random-valid"
    candidate_rule = CandidateRule.FEASIBLE

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def choose(self, candidates: list[Job], servers: list[Server]) -> tuple[Job, Server] | None:
        options = [(job, servers_that_fit(job, servers)) for job in candidates]
        options = [(job, fitting) for job, fitting in options if fitting]
        if not options:
            return None
        job, fitting = options[self._rng.integers(len(options))]
        return job, fitting[self._rng.integers(len(fitting))]
