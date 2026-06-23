# Paper plan: Ego-centric Neural CCM for quadrotor CTBR tracking

## Maintained question

Can an ego-centric Neural CCM improve bounded CTBR velocity-heading tracking
over a world-frame Neural CCM while retaining a sampled contraction certificate
and practical PX4/Pegasus compatibility?

## Current method

- compare world-frame `(v, R)` CCM with ego-centric
  `(gamma, delta_v_body, delta_psi)` on `S2 x R3 x S1`;
- use the legal six-dimensional tangent space for the Ego-CCM certificate;
- jointly learn a linear-plus-neural controller and dual metric `W`;
- closed-loop contraction, C1 and C2 losses;
- one physical command interval:
  `[1.691, -3.840, -3.840, -1.570] <= CTBR <= [15.222, 3.840, 3.840, 1.570]`;
- trajectory-derived velocity, acceleration, attitude, yaw-rate and CTBR feedforward;
- common reference paths, CTBR limits, PI rate loop, allocation, motors and plant;
- configured motor lag/noise and deterministic angular drag;
- no position feedback or disturbance observer in the maintained benchmark.

## Current evidence

`neu_ccm_linear.pt` reaches `99.96643%` held-out contraction with maximum
`eig(C)=0.94519`; `neu_ego_ccm_active.pt` reaches `99.97253%` with maximum
`eig(C)=0.82002`. Both have 100% C1 satisfaction on 32768 held-out points, but
neither is a 100% sampled certificate.

The frozen dynamic-yaw benchmark contains four trajectories, three speed scales
and three seeds. All `144/144` runs are stable. Overall means are:

| Controller | velocity RMSE | yaw RMSE | drift RMSE | torque RMS | allocation saturation |
|---|---:|---:|---:|---:|---:|
| SO3-CTBR | 0.06979 m/s | 0.45945 deg | 0.18148 m | 0.05767 N·m | 0% |
| SO3-Full | 0.06875 m/s | **0.27933 deg** | 0.18396 m | 0.13001 N·m | 0.271% |
| world-frame CCM | 0.10100 m/s | 0.84956 deg | 0.25417 m | 0.04620 N·m | 0% |
| Ego-CCM | **0.06791 m/s** | 0.91337 deg | 0.21559 m | **0.04565 N·m** | 0% |

Observed result: Ego-CCM reduces velocity RMSE by 32.8% relative to the
world-frame CCM and uses 20.8% less RMS torque than SO3-CTBR. It does not yet
improve yaw or accumulated position drift over SO3-CTBR. SO3-Full has a
different 400 Hz direct-wrench interface and is reported as a high-bandwidth
reference, not a matched CTBR controller.

## Defensible contribution direction

1. an ego-centric, symmetry-reduced CTBR model with a certificate enforced only
   on `T_gamma S2 x R3 x S1`;
2. bounded linear-plus-residual Neural CCM learning with trajectory-consistent
   feedforward;
3. sampled certificate evaluation plus matched simulation and real-flight tests.

The current evidence supports improved velocity tracking over the world-frame
CCM. It does not support a general tracking or robustness superiority claim over
SO3 control.

## Fixed experiment plan

1. **Coordinate ablation:** identical optimizer, loss, architecture capacity,
   samples and linear branch for world-frame CCM and Ego-CCM.
2. **Certificate:** training, held-out, multi-seed Monte Carlo and adversarial
   search for worst `eig(C)`, C1 and C2 residuals.
3. **Tracking:** Circle, Figure-8, helix and multi-sine at `1/1.5/2x`; report
   velocity/yaw RMSE, drift, CTBR/torque, saturation and failures.
4. **Robustness:** freeze test cases before retraining; vary mass, inertia, drag,
   motor lag, sensor noise, delay and initial error.
5. **Real flight:** the same references and bounds on Pegasus; report at least
   repeated nominal and mismatch trials plus CPU latency.
6. **MPCC integration:** treat MPCC only as the shared outer reference generator;
   compare SO3-CTBR, CCM and Ego-CCM under exactly the same MPCC output.

The reproducible baseline is `cfg/benchmark.yaml`. Each run writes
`fig/baseline/benchmark_summary.csv` and a `manifest.json` with source,
configuration and checkpoint hashes. Generated artifacts are intentionally
ignored by Git.

## Submission gap

Before an ICRA submission, the work still needs:

1. a checkpoint with 100% independent sampled contraction validation;
2. a controlled coordinate-only ablation separating representation from training;
3. a mechanism-backed yaw improvement without losing velocity performance;
4. frozen disturbance/model-mismatch experiments with confidence intervals;
5. matched real-flight experiments and runtime statistics.
