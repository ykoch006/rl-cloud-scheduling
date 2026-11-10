# Model assumptions: workloads and scheduling

Status as of 10 November 2026. These are **provisional development settings**.
They have not been calibrated against Google, AWS, Alibaba or any other
production trace, and must not be presented as representative of real cloud
workloads. Calibration (including Alibaba trace-informed validation) is a
later research task.

Each value is labelled:

- **SOURCE**: taken from the submitted Project Initiation Document (PID).
- **DERIVED**: follows from a SOURCE value or from mathematics.
- **ASSUMPTION**: chosen for development; needs justification or calibration.

"Scope" below refers to my working project scope (v1.3.1), which refines the PID.

## Time model

The simulator moves in whole ticks (timesteps). A tick has no fixed real-world
length; it is the unit of both arrival time and job duration.

- The arrival rate λ is the **expected number of new jobs per tick**. At each
  tick t, the number of arrivals is drawn as N_t ~ Poisson(λ), so a tick can
  have zero, one or several arrivals. λ = 2.5 means 2.5 jobs per tick on
  average, not one job every 2.5 ticks.
- Jobs arriving at tick t join the queue at the **start** of tick t, before
  the scheduler runs, so a job can start in the tick it arrives (waiting time 0).
- A job with duration d that starts at tick s holds its CPU and RAM during
  ticks s, s+1, ..., s+d-1 and completes at s+d. Its resources are free for
  the scheduler at the start of tick s+d.
- Arrivals happen only at ticks 0..horizon-1. The simulation can carry on
  after the horizon so that queued and running jobs can finish.

## Parameters

| Parameter | Value | Label | Notes |
|---|---|---|---|
| Server CPU capacity | 100 units | SOURCE | PID §2.3 |
| Server RAM capacity | 64 GB | SOURCE | PID §2.3 |
| Reference cluster size (for offered load) | 5 servers | SOURCE | Lower end of the PID's 5–10 range |
| Arrival process | Poisson count per timestep | SOURCE | PID §2.3. Time-varying rate not implemented yet |
| CPU / RAM demand family | Log-normal | SOURCE | PID §2.3 |
| Duration family | Exponential | SOURCE | PID §2.3 |
| Maximum CPU demand | 100 units | DERIVED | One server's capacity, so every job can run somewhere |
| Maximum RAM demand | 64 GB | DERIVED | As above |
| Duration rounding | `ceil(X)` | DERIVED | Gives whole timesteps ≥ 1; makes durations geometric |
| Offered load formula | rate × E[demand × duration] / total capacity | DERIVED | Standard queueing definition |
| CPU median, sigma | 10 units, 0.8 | ASSUMPTION | |
| CPU minimum | 1 unit | ASSUMPTION | |
| RAM median, sigma | 4 GB, 0.9 | ASSUMPTION | |
| RAM minimum | 0.25 GB | ASSUMPTION | |
| Mean duration (before rounding) | 10 timesteps | ASSUMPTION | |
| Maximum duration | 100 timesteps | ASSUMPTION | |
| Target CPU load bands | light 40–50%, medium 65–75%, heavy 85–95% | ASSUMPTION | Scope §3.5; development targets, not cloud standards |
| Light / medium / heavy arrival rate | 1.6 / 2.5 / 3.2 jobs per tick | DERIVED | Chosen so the expected CPU load lands inside each target band |
| CPU and RAM independent | — | ASSUMPTION | Real traces show correlation |
| Demand independent of duration | — | ASSUMPTION | Needed for E[demand × duration] = E[demand] × E[duration] |
| Rejection guard | Bounds must keep ≥ 50% of the distribution | ASSUMPTION | Catches mistaken parameters |

## Resulting expectations

Computed by the code (`expected_value`, `expected_offered_load`), not measured.

| Quantity | Untruncated | After truncation |
|---|---|---|
| Mean CPU demand | 13.77 units | 13.56 units (99.6% of draws accepted) |
| Mean RAM demand | 6.00 GB | 5.92 GB (99.8% accepted) |
| Mean duration | — | 10.50 timesteps (>99.99% accepted) |

| Profile | Arrival rate | Expected CPU load | Expected RAM load |
|---|---|---|---|
| light | 1.6 | 0.456 | 0.311 |
| medium | 2.5 | 0.712 | 0.486 |
| heavy | 3.2 | 0.912 | 0.622 |

Loads are for 5 servers. With 10 servers every load halves. CPU is the
bottleneck resource in all three profiles. Even "heavy" is below 1.0
nominal load, although fragmentation across servers means queueing can
appear well before that.

These are expectations. A single short workload can differ noticeably:
with a 200-tick horizon, medium gave an observed CPU load of 0.608 for seed 1
and 0.758 for seed 2, because CPU × duration has a heavy right tail. Results
should therefore be averaged over many seeds.

Not yet implemented from the scope: the bursty profile (alternating normal
and peak periods with a medium average) and the stress profile.

## Scheduling assumptions

| Parameter | Value | Label | Notes |
|---|---|---|---|
| Candidate window size | 5 jobs | ASSUMPTION | Scope §5.2; matches the planned RL observation |
| Candidate rule | 5 oldest jobs that fit on at least one server now | ASSUMPTION | Scope §5.2; rebuilt after every placement |
| Runtime estimates | equal to true durations | ASSUMPTION | Perfect estimates; real schedulers are less certain |
| Best-Fit score | sum of CPU and RAM left over, each divided by capacity | ASSUMPTION | One documented scoring function, as the scope requires |
| Random-valid seed | the workload seed | ASSUMPTION | Its own generator, separate from the workload's streams |
| Drain period in comparisons | 500 ticks | ASSUMPTION | Scope §7.1 provisional value |
| Utilisation window in comparisons | the arrival horizon | DERIVED | Same window for every policy |

## Modelling choices to review

- **Truncation is by rejection, not clipping.** Out-of-range values are
  redrawn, so there is no spike of jobs at exactly the maximum. The mean
  shifts slightly (table above), and the offered-load formulas use the
  truncated means.
- **Zero demand.** `Job` accepts a demand of 0, but the generator only
  produces strictly positive demands (minimum 1 CPU unit and 0.25 GB).
  `Job` validation was left unchanged; whether zero should be allowed is
  still to be decided.
- **Separate random streams.** Arrivals, CPU, RAM and durations each use a
  child generator spawned from the single seed. Changing one distribution
  does not change the values drawn for the others.
- **Floating-point demands.** Demands are continuous. Free capacity used to
  be updated by adding and subtracting on every placement, which drifted by
  up to about 1e-13 after thousands of jobs. It is now recomputed from the
  running jobs, so an empty server always reports exactly its full capacity.
  A single placement can still exceed capacity by one rounding step
  (about 1e-14), so the capacity tests allow a 1e-9 tolerance.
- **Starvation.** The feasible-only window lets small jobs repeatedly pass a
  large one. This has not yet been measured; the scope schedules
  size-stratified tail checks with the Gymnasium work.
