"""Seeded stochastic workload generation.

A workload is generated in two steps:

1. Arrivals: at each timestep t in 0..horizon-1, the number of new jobs is
   N_t ~ Poisson(lambda_t). Jobs arriving at the same timestep are numbered
   in generation order, so ids follow arrival order.
2. Job properties: each job independently gets a CPU demand and a RAM demand
   (truncated log-normal) and a duration (rounded-up, truncated exponential).

The result is a tuple of immutable `JobSpec`s. Use
`env.workload.instantiate_jobs` to create fresh `Job` objects for each
simulation run.

All default parameter values are provisional development assumptions, not
measured cloud statistics. See docs/assumptions.md.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from env.workload import JobSpec

# Reference server from the PID: 100 CPU units and 64 GB of RAM.
REFERENCE_SERVER_CPU = 100.0
REFERENCE_SERVER_MEMORY = 64.0

# Smallest server count in the PID's 5-10 range, used for offered-load estimates.
REFERENCE_SERVER_COUNT = 5

# If the bounds would discard more than half of a distribution, the
# parameters are almost certainly a mistake, so they are rejected.
MIN_ACCEPTANCE_PROBABILITY = 0.5


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class TruncatedLogNormal:
    """Log-normal distribution restricted to [minimum, maximum].

    ln(X) ~ Normal(ln(median), sigma). Draws outside the bounds are redrawn
    (rejection sampling), so the result keeps the log-normal shape inside
    the interval instead of piling up at the bounds as clipping would. The
    cost is that the mean is slightly different from the untruncated
    log-normal's; `expected_value` accounts for this.
    """

    median: float
    sigma: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.median, self.sigma, self.minimum, self.maximum)):
            raise ValueError("Log-normal parameters must be finite numbers")
        if self.median <= 0:
            raise ValueError(f"median must be > 0, got {self.median}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if not 0 < self.minimum < self.maximum:
            raise ValueError(
                f"bounds must satisfy 0 < minimum < maximum, got [{self.minimum}, {self.maximum}]"
            )
        if self.acceptance_probability() < MIN_ACCEPTANCE_PROBABILITY:
            raise ValueError(
                f"bounds [{self.minimum}, {self.maximum}] exclude most of the distribution "
                f"(median={self.median}, sigma={self.sigma})"
            )

    @property
    def mu(self) -> float:
        """Mean of ln(X) before truncation."""
        return math.log(self.median)

    def _standardise(self, x: float, shift: float = 0.0) -> float:
        return (math.log(x) - self.mu - shift) / self.sigma

    def acceptance_probability(self) -> float:
        """Probability that an untruncated draw falls inside the bounds."""
        return _normal_cdf(self._standardise(self.maximum)) - _normal_cdf(
            self._standardise(self.minimum)
        )

    def expected_value(self) -> float:
        """Mean of the truncated distribution (closed form)."""
        variance = self.sigma**2
        inside = _normal_cdf(self._standardise(self.maximum, variance)) - _normal_cdf(
            self._standardise(self.minimum, variance)
        )
        return math.exp(self.mu + variance / 2) * inside / self.acceptance_probability()

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw `size` values, redrawing any that fall outside the bounds."""
        values = rng.lognormal(self.mu, self.sigma, size)
        outside = (values < self.minimum) | (values > self.maximum)
        while outside.any():
            values[outside] = rng.lognormal(self.mu, self.sigma, int(outside.sum()))
            outside = (values < self.minimum) | (values > self.maximum)
        return values


@dataclass(frozen=True)
class TruncatedExponentialDuration:
    """Job duration in whole timesteps, between 1 and `maximum`.

    A value X ~ Exponential(mean) is rounded up to D = ceil(X), so every
    duration is at least one timestep. Rounding up turns the exponential
    into a geometric distribution, P(D = k) = (1 - q) q^(k - 1) with
    q = exp(-1 / mean), whose mean is about `mean + 0.5`. Durations above
    `maximum` are redrawn.
    """

    mean: float
    maximum: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean) or self.mean <= 0:
            raise ValueError(f"mean duration must be a finite number > 0, got {self.mean}")
        if not isinstance(self.maximum, int) or self.maximum < 1:
            raise ValueError(f"maximum duration must be an integer >= 1, got {self.maximum}")
        if self.acceptance_probability() < MIN_ACCEPTANCE_PROBABILITY:
            raise ValueError(
                f"maximum duration {self.maximum} excludes most of the distribution "
                f"(mean={self.mean})"
            )

    @property
    def _q(self) -> float:
        return math.exp(-1.0 / self.mean)

    def acceptance_probability(self) -> float:
        """Probability that an untruncated duration is at most `maximum`."""
        return 1.0 - self._q**self.maximum

    def expected_value(self) -> float:
        """Mean duration after rounding up and truncation (exact sum)."""
        q = self._q
        total = sum(k * (1 - q) * q ** (k - 1) for k in range(1, self.maximum + 1))
        return total / self.acceptance_probability()

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw `size` integer durations, redrawing any above `maximum`."""

        def draw(count: int) -> np.ndarray:
            # An exponential draw of exactly 0.0 is possible in floating point
            # (probability around 1e-16); it is treated as one timestep.
            return np.maximum(1, np.ceil(rng.exponential(self.mean, count))).astype(np.int64)

        values = draw(size)
        too_long = values > self.maximum
        while too_long.any():
            values[too_long] = draw(int(too_long.sum()))
            too_long = values > self.maximum
        return values


DEFAULT_CPU = TruncatedLogNormal(median=10.0, sigma=0.8, minimum=1.0, maximum=REFERENCE_SERVER_CPU)
DEFAULT_MEMORY = TruncatedLogNormal(
    median=4.0, sigma=0.9, minimum=0.25, maximum=REFERENCE_SERVER_MEMORY
)
DEFAULT_DURATION = TruncatedExponentialDuration(mean=10.0, maximum=100)


@dataclass(frozen=True)
class WorkloadConfig:
    """Everything needed to generate one workload.

    The same config (including `seed`) always produces exactly the same
    workload. Arrivals happen at integer timesteps 0..horizon-1.
    """

    seed: int
    horizon: int
    arrival_rate: float
    cpu: TruncatedLogNormal = DEFAULT_CPU
    memory: TruncatedLogNormal = DEFAULT_MEMORY
    duration: TruncatedExponentialDuration = DEFAULT_DURATION

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be an integer >= 0, got {self.seed!r}")
        if not isinstance(self.horizon, int) or self.horizon < 0:
            raise ValueError(f"horizon must be an integer >= 0, got {self.horizon!r}")
        if not math.isfinite(self.arrival_rate) or self.arrival_rate < 0:
            raise ValueError(f"arrival_rate must be a finite number >= 0, got {self.arrival_rate}")
        if self.cpu.maximum > REFERENCE_SERVER_CPU:
            raise ValueError(
                f"maximum CPU demand {self.cpu.maximum} exceeds one server's "
                f"capacity ({REFERENCE_SERVER_CPU}), so some jobs could never run"
            )
        if self.memory.maximum > REFERENCE_SERVER_MEMORY:
            raise ValueError(
                f"maximum RAM demand {self.memory.maximum} exceeds one server's "
                f"capacity ({REFERENCE_SERVER_MEMORY}), so some jobs could never run"
            )

    def arrival_rates(self) -> np.ndarray:
        """Expected number of arrivals at each timestep 0..horizon-1.

        Constant for now. A time-varying rate (for example a bursty
        profile) only needs to change this method.
        """
        return np.full(self.horizon, self.arrival_rate, dtype=float)


def generate_workload(config: WorkloadConfig) -> tuple[JobSpec, ...]:
    """Generate the job specifications described by `config`.

    Arrivals, CPU, RAM and durations each use their own random stream,
    split from the one seed. Changing one distribution therefore does not
    change the values drawn for the others, which keeps comparisons
    between configurations controlled.
    """
    arrival_rng, cpu_rng, memory_rng, duration_rng = np.random.default_rng(config.seed).spawn(4)

    arrivals_per_step = arrival_rng.poisson(config.arrival_rates())
    arrival_times = np.repeat(np.arange(config.horizon), arrivals_per_step)
    job_count = len(arrival_times)

    cpu = config.cpu.sample(cpu_rng, job_count)
    memory = config.memory.sample(memory_rng, job_count)
    durations = config.duration.sample(duration_rng, job_count)

    return tuple(
        JobSpec(
            id=index + 1,
            arrival_time=int(arrival_times[index]),
            duration=int(durations[index]),
            cpu_required=float(cpu[index]),
            memory_required=float(memory[index]),
        )
        for index in range(job_count)
    )


# Profiles currently differ only in arrival rate (jobs per timestep). These
# are development settings, not calibrated workload levels; see
# docs/assumptions.md for the offered load each one gives.
PROFILE_ARRIVAL_RATES: dict[str, float] = {
    "light": 1.0,
    "medium": 2.0,
    "heavy": 3.0,
}


def profile_config(profile: str, seed: int, horizon: int) -> WorkloadConfig:
    """Return the WorkloadConfig for a named profile."""
    if profile not in PROFILE_ARRIVAL_RATES:
        raise ValueError(
            f"Unknown workload profile {profile!r}; choose from {sorted(PROFILE_ARRIVAL_RATES)}"
        )
    return WorkloadConfig(seed=seed, horizon=horizon, arrival_rate=PROFILE_ARRIVAL_RATES[profile])


def expected_offered_load(
    config: WorkloadConfig, server_count: int = REFERENCE_SERVER_COUNT
) -> tuple[float, float]:
    """Theoretical (CPU, RAM) offered load on `server_count` reference servers.

    offered load = arrival rate x E[demand x duration] / total capacity.
    Demand and duration are sampled independently, so
    E[demand x duration] = E[demand] x E[duration].
    """
    mean_duration = config.duration.expected_value()
    cpu_load = (
        config.arrival_rate * config.cpu.expected_value() * mean_duration
        / (server_count * REFERENCE_SERVER_CPU)
    )
    memory_load = (
        config.arrival_rate * config.memory.expected_value() * mean_duration
        / (server_count * REFERENCE_SERVER_MEMORY)
    )
    return cpu_load, memory_load


def observed_offered_load(
    specs: Sequence[JobSpec], horizon: int, server_count: int = REFERENCE_SERVER_COUNT
) -> tuple[float, float]:
    """(CPU, RAM) offered load measured from one generated workload.

    Total resource-time requested, divided by the capacity-time available
    over the arrival horizon.
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0 to measure offered load")
    cpu_time = sum(spec.cpu_required * spec.duration for spec in specs)
    memory_time = sum(spec.memory_required * spec.duration for spec in specs)
    cpu_load = cpu_time / (horizon * server_count * REFERENCE_SERVER_CPU)
    memory_load = memory_time / (horizon * server_count * REFERENCE_SERVER_MEMORY)
    return cpu_load, memory_load
