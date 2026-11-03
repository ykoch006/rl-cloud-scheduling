# Research assumptions: stochastic workload generator

Status as of 3 November 2026. These are **provisional development settings**.
They have not been calibrated against Google, AWS, Alibaba or any other
production trace, and must not be presented as representative of real cloud
workloads. Calibration (including Alibaba trace-informed validation) is a
later research task.

Each value is labelled:

- **SOURCE**: taken from the submitted Project Initiation Document (PID).
- **DERIVED**: follows from a SOURCE value or from mathematics.
- **ASSUMPTION**: chosen for development; needs justification or calibration.

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
| Light / medium / heavy arrival rate | 1 / 2 / 3 jobs per timestep | ASSUMPTION | Not calibrated load levels |
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
| light | 1.0 | 0.285 | 0.194 |
| medium | 2.0 | 0.570 | 0.389 |
| heavy | 3.0 | 0.855 | 0.583 |

Loads are for 5 servers. With 10 servers every load halves. CPU is the
bottleneck resource in all three profiles. Even "heavy" is below 1.0
nominal load, although fragmentation across servers means queueing can
appear well before that.

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
- **Floating-point demands.** Demands are continuous. In a test run, server
  resource accounting drifted by up to about 1e-13 after thousands of
  allocate/release cycles. This is a simulator issue to fix later, for
  example with integer resource units or a small tolerance.
