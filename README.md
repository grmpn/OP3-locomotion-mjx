# OP3 locomotion with MJX and PPO

A small Robotis OP3 walking baseline using direct MuJoCo MJX physics and Brax's
standard PPO trainer. Experiment decisions, evidence, and run lineage are recorded
in [EXPERIMENTS.md](EXPERIMENTS.md). The native MuJoCo demo loads exported policy parameters and
accepts body-frame velocity commands from the keyboard. Physics parameters and
terrain are fixed; no sensor noise or gait generator is used.
An optional experiment samples predefined standing or left/right raised-foot
reset poses; the default and selected demo still start standing.

The selected trained policy is available locally in `outputs/op3-policy/`, with
its source lineage in `training_lineage.json`. Launch it from the repository root:

```sh
uv run op3-play outputs/op3-policy
```

It survived all seven independent ten-second command cases and a continuous
50-second sequence of standing, walking, turning while walking, lateral motion,
and stopping. Settled native MuJoCo measurements (excluding the first second):

| Command | Requested | Measured |
| --- | ---: | ---: |
| forward velocity | +0.150 m/s | +0.150 m/s |
| backward velocity | −0.150 m/s | −0.144 m/s |
| left velocity | +0.090 m/s | +0.084 m/s |
| right velocity | −0.090 m/s | −0.076 m/s |
| left yaw rate | +0.480 rad/s | +0.429 rad/s |
| right yaw rate | −0.480 rad/s | −0.404 rad/s |

The gait uses quick, short steps with millimetre-scale clearance. Tracking is
imperfect: there is yaw drift, asymmetry, some contact slip, and small foot
unloading even at zero command. These are fixed-start simulation results on the
default floor, not evidence of physical-robot robustness. Detailed reports are
in `outputs/balanced-candidate-12m/`; decisions and comparisons are in
[EXPERIMENTS.md](EXPERIMENTS.md).

Eight-second tests also survived all individual-axis command limits and a
40-second reversal/stop sequence (`outputs/demo-command-limits/`). Small left
turns can undershoot substantially: +0.4 rad/s requested produced +0.208 rad/s
settled; backward motion at −0.25 m/s had approximately −0.195 rad/s unwanted
yaw. Combined command extremes have not been exhaustively validated.

The selected export inherits 15,052,800 baseline plus 12,042,240 refinement
transitions. The final refinement checkpoint was rejected because sustained
left turning regressed. Runs, policies, reports and videos are ignored by Git.
A fresh clone needs a copy of the local export (or `outputs/op3-policy.zip`,
unpacked into `outputs/`) or a new training run using the recipe below.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
uv sync --locked
uv run scripts/fetch_assets.py
uv run python scripts/check_env.py
```

The lockfile pins JAX/jaxlib 0.8.2, MuJoCo/MJX 4.0.0, and Brax 0.14.2.
Flax 0.12.2 and Orbax 0.11.33 are pinned with it: newer JAX removes a replication
API still used by this Brax release. Linux installs JAX's CUDA 13 wheels. This was selected for the development
machine's RTX 3070 Laptop GPU (8 GB, SM 8.6), driver 591.59 (CUDA driver 13.1).
The [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
requires SM 7.5+ and driver 580+ for CUDA 13. Do not copy this CUDA choice to an
older GPU/driver without checking that guide. NVIDIA support in WSL depends on
the host driver. CPU inference is supported; full training refuses CPU fallback.

MJX's official [Warp backend](https://mujoco.readthedocs.io/en/stable/mjx.html)
is the default for GPU physics, preserving mesh collision geometry without the
large compile cost of MJX-JAX. Both expose the same pure JAX reset/step interface.
The first GPU run compiles and caches Warp kernels and can take several minutes.
JAX preallocation is disabled and its default memory fraction is 0.35 to leave
space for the desktop and Warp. Environment variables can override these settings.

## Model and collision policy

`scripts/fetch_assets.py` downloads only `robotis_op3` from Menagerie revision
`98a70c569c4c1b6da467a20012f766ee1f8dc521`. It verifies Git blob checksums,
retains the upstream Apache-2.0 `LICENSE` and README, and writes SHA-256 provenance.
The source is [MuJoCo Menagerie OP3](https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/robotis_op3).
All fetched assets are reproducible and ignored by Git. Loading checks their hashes.

The default `scene.xml` and its floor are used without editing the downloaded
files. The shared model loader disables **head and arm collision participation**
(the user-authorized `locomotion` collision profile). Torso/leg self collisions
and the original four overlapping foot boxes remain. All meshes, visuals,
body masses/inertias, joint dynamics, actuator gains/force limits, friction, and
solver settings remain upstream defaults. This simplifies contact, so the robot
cannot support itself on its head or hands; falls terminate before such contact
is useful. Set `collision_profile` to `full` in a config to restore all colliders.
Collision hull reduction is optional: `--collision-maxhullvert 16` (or `8`)
uses MuJoCo's [maximum hull vertex setting](https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-mesh-maxhullvert)
on collision meshes in memory. It approximates their contact surfaces, while
preserving visual meshes, inertias, foot boxes and terrain. `-1` retains upstream
hulls and remains the default. Training and inference share the loader; exports
record the cap and reject a different cap at inference. Downloaded files stay intact.

## Train and save a policy

```sh
# A short pipeline check, not a walking policy:
uv run op3-train --output outputs/smoke --num-timesteps 640 \
  --num-envs 32 --batch-size 16 --num-minibatches 2 \
  --num-evals 2 --num-eval-envs 4

# Initial policy (GPU required; development run used 512 environments):
uv run op3-train --output outputs/baseline --num-timesteps 15000000 \
  --num-envs 512 --num-evals 16 --foot-slip-weight 0 --foot-air-time-weight 0

# Warm-start refinement with direct coverage of each keyboard command mode:
uv run op3-train --config outputs/baseline/config.json --output outputs/balanced \
  --initialize-from outputs/baseline/checkpoints/000015052800 \
  --command-sampling balanced --num-timesteps 15000000 --num-evals 16

# Slip-penalty trial from the selected policy, with reduced collision hulls:
uv run op3-train --config outputs/op3-policy/run_config.json \
  --initialize-from outputs/op3-policy --output outputs/slip-trial \
  --foot-slip-weight -5 --collision-maxhullvert 16 \
  --num-timesteps 1000000 --num-evals 3

# Add air time without prescribing a gait period, starting from the slip trial:
uv run op3-train --config outputs/slip-hull-full/config.json \
  --initialize-from outputs/slip-hull-full --output outputs/air-time-trial \
  --foot-air-time-weight 2 --num-timesteps 1000000 --num-evals 3
# Use --foot-air-time-weight 0 with another output directory for a matched control.

# Full 30M-transition run from scratch: upright +1, posture -0.05, air time +2.
uv run op3-train --config configs/upright_air_time.json \
  --output outputs/upright-air-time-full
```

The output directory must be new. Each run prints the selected GPU and records
`config.json`, `mapping.json`, `run.json`, `metrics.jsonl`, and exports under
`checkpoints/<12-digit-step>/`. A `checkpoints/latest` pointer selects the last
complete export. Training episode return is `episode/sum_reward`; evaluation return is
`eval/episode_reward`. Metrics include individual weighted reward terms, tracking
errors, episode lengths, PPO losses, and throughput. Brax's `eval/episode_*`
metrics aggregate over each episode; divide error sums by episode length to
compare to per-step evaluation errors. `finished.json` indicates successful
completion of the requested PPO run, not policy quality.

Adjust `--num-envs`, `--batch-size`, `--num-minibatches`, `--unroll-length`, and
`--num-eval-envs` for GPU memory. `batch_size * num_minibatches` must be divisible
by `num_envs`. Defaults are 256 environments and 128 samples per minibatch, with
four minibatches and 20 control steps per unroll. No physics parameters are
randomized. The run's seed separates command sampling and stochastic PPO actions
through explicit split PRNG keys.

For other settings, copy a run's `config.json`, edit the typed environment,
reward, or PPO values, and pass `--config path/to/config.json`. The actor and
critic are separate `(128, 128, 128)` MLPs using Brax's standard tanh-normal action
distribution and running observation normalization. PPO uses observation
normalization, clipping, entropy regularization, GAE, and gradient clipping.

Exports contain actor and critic parameters, observation normalization, network
settings, seed/configuration, package/model versions, joint/action mappings,
nominal pose, bounds, and control timing. **These are policy exports, not full
training-state checkpoints**: optimizer state and rollout/RNG state are not
saved. `--initialize-from outputs/baseline` starts a new run from saved weights
and normalization, resetting the optimizer, RNG and step count. It requires the
same environment/network configuration except `command_sampling`, reward settings,
`collision_maxhullvert`, and raised-foot reset settings. These exceptions support
explicit refinement trials;
physics timing, action/observation mappings and command bounds must still match.
This is explicitly not exact resume. Old exports without slip settings load with
zero slip weight; use `--foot-slip-weight -5` to enable it when refining one.
Likewise, configs predating air time load with zero air-time weight; enable it
explicitly with `--foot-air-time-weight 2`. New runs default to both terms enabled.

The experimental `outputs/air-time-2` export completed a 1M-transition trial and
survived the full command suite. It did not improve swing duration or overall
tracking against a matched zero-weight control; see D021 in `EXPERIMENTS.md`.
Run it with `uv run op3-play outputs/air-time-2`. The selected demo remains
`outputs/op3-policy`.

Two independent standing/exploration experiments can start from the air-time
checkpoint, with one shared control. Both options default to zero (disabled):

```sh
# 10x total posture penalty at zero requested velocity: -0.05 - 0.45 = -0.5.
uv run op3-train --config outputs/air-time-2/config.json \
  --initialize-from outputs/air-time-2 --output outputs/standing-posture-10x \
  --standing-posture-weight -0.45 --num-timesteps 1000000 --num-evals 3

# 75% standing, 12.5% left raised and 12.5% right raised; original posture weight.
uv run op3-train --config outputs/air-time-2/config.json \
  --initialize-from outputs/air-time-2 --output outputs/lift-reset-25 \
  --lift-reset-probability 0.25 --num-timesteps 1000000 --num-evals 3

# Matched continuation without either change.
uv run op3-train --config outputs/air-time-2/config.json \
  --initialize-from outputs/air-time-2 --output outputs/standing-lift-control \
  --num-timesteps 1000000 --num-evals 3
```

The extra posture term applies only when `[vx, vy, yaw_rate]` is exactly zero,
regardless of the policy's joint actions. It does not increase the moving penalty.
Raised resets use fixed mirrored joint poses with about 2 cm sole clearance,
zero velocities, and a separate seeded pose-choice stream. Physics parameters,
nominal action targets and policy inputs are unchanged. Initial landings earn
no air-time bonus. These starts expose single-foot support and recovery; useful
lifting must still be checked from an ordinary standing start.

The completed 1M-transition trials survived every standard case. Against the
shared control, stronger standing posture reduced posture error by 5.3% but
increased slip; raised resets increased air time but did not increase foot
clearance. Both features remain opt-in, and the selected demo is unchanged.
See D022–D023 in `EXPERIMENTS.md` for the full comparison and recovery checks.

The full upright/air-time recipe completed 30,105,600 transitions from scratch
in `outputs/upright-air-time-full`. Its final export survives the native and MJX command
suite and moves in several directions, but right turning fails and backward
motion includes hopping. It is experimental; the selected demo is unchanged.
Inspect it with `uv run op3-play outputs/upright-air-time-full`; see D024 in
`EXPERIMENTS.md` for checkpoint comparisons and gait limitations.

## Evaluate and inspect behavior

```sh
uv run op3-evaluate outputs/op3-policy
uv run op3-evaluate outputs/op3-policy --physics mjx
# Render videos (use MUJOCO_GL=egl on a compatible headless Linux host):
uv run op3-evaluate outputs/op3-policy --video --cases stand forward turn_left changes
# Compare against fixed standing targets to detect a stand-still solution:
uv run op3-evaluate outputs/op3-policy --nominal
```

A run directory, its `checkpoints` directory, or an individual export can be
passed to evaluation/play. Default evaluation runs independent standing,
forward, backward, both lateral directions, and both yaw directions, followed by
a continuous command-change sequence. Commands use 60% of trained limits, five
seconds per segment, deterministic actions and seed 123. `--seconds` and
`--cases` select a shorter check. The JSON report includes survival, linear/yaw
RMSE (including transients), settled segment velocities, displacement, foot-height
variation, minimum base height and contact counts. Gait diagnostics include each
foot's swing fraction, touchdowns, maximum sole clearance, and mean foot-body
speed while supported, plus force-weighted tangential contact-point slip speed. Support requires >1 N of floor normal force; clearance
uses the original foot boxes. Supported foot-body speed is a slip proxy and also
includes rolling, so it is not an exact slip measurement. Contact-point slip is
computed from the contact Jacobian and generalized velocity and excludes the
normal component. These diagnostics
exclude the first second (or half of a shorter rollout). Compressed traces contain
commands, velocities, joint state, actions, foot positions, clearances, and
normal forces. Settled tracking RMSE is reported for each command segment. Videos and traces
are necessary to distinguish stepping from sliding; foot-height variation alone
does not prove a walking gait.

Evaluation stops a case on a fall rather than resetting and hiding failures.

Evaluation and play default to the fixed standing start even for policies trained
with mixed resets. Use `--reset-pose left` or `--reset-pose right` to inspect
recovery from a raised foot, for example
`uv run op3-play outputs/lift-reset-25 --reset-pose left`. R repeats that selected
pose. Save forced-pose evaluations to separate `--output` directories. Reports
include the chosen start pose and mean summed squared leg-posture error in rad².
The `changes` case preserves physical state across command updates. Native
MuJoCo evaluation uses CPU policy inference; `--physics mjx` uses the training
physics backend on GPU. Reports are stored alongside the selected checkpoint,
or in `--output DIRECTORY`.

The completed selected-policy videos are `outputs/demo-video/forward.mp4`,
`turn_left.mp4`, `turn_right.mp4`, and `changes.mp4`. The last clip contains a
continuous fifteen-second stand/walk/moving-turn/lateral/stop sequence. These
short videos complement the longer headless tracking tests; their matching
traces and report are in the same directory.

## Keyboard demo

```sh
uv run op3-play outputs/op3-policy
```

The display requires GLFW/OpenGL. The renderer has its own key callback so it
does not inherit conflicting MuJoCo viewer shortcuts. Hold a direction to ramp
the requested velocity toward its trained bound; release it to decay to zero:

| Keys | Command change |
| --- | --- |
| W / S | ramp forward / backward velocity |
| A / D | ramp left / right velocity |
| Q / E | ramp left / right yaw rate |
| Space | zero all commands |
| R | reset to the selected start pose and zero command |
| Escape | exit |

Ramping from zero to a bound takes 0.5 s; release from a bound takes 0.35 s to
return to zero, measured in simulated control time. Opposite keys held together
decay that axis to zero. Rates are independent of OS key repeat and render rate.
Space, reset, and window focus loss clear held keys and immediately zero the
command; release and press again to resume movement after clearing.

The title and on-screen overlay display `[vx, vy, yaw_rate]`. Commands clamp to
the export's trained range. The camera follows the torso; a fall freezes physics
until R. Simulation and inference run at 50 Hz, with ten 2 ms MuJoCo substeps per
action, matching training. If the host cannot keep up, simulated time runs more
slowly. Control updates use an elapsed-time accumulator independent of rendered
frames, with catch-up capped at 250 ms after a slow/paused window. The viewer
disables shadows and reflections to reduce display cost; physics and the default
floor are unchanged. On this machine's llvmpipe software OpenGL display, the
final check advances 9.98 simulated seconds in the ten-second viewer loop while
rendering about 9.6 fps (`outputs/demo-viewer-final.json`). GPU training does not
imply GPU rendering. `--seconds 5` performs a finite viewer startup check.

The viewer also shows three orientation arrows beside the torso, with a grey
line connecting their shared origin to the base:

- **Green:** the robot's body +Z axis, which tilts with the torso.
- **Cyan:** up reconstructed from the policy's gravity input and rotated into
  world coordinates. It should remain vertical even when the robot leans.
- **Orange:** the actual MuJoCo gravity direction, pointing down.

The arrows follow the current displayed pose using the actor's gravity
calculation before observation normalization. The policy does not have a separate
learned up-vector estimate. The arrows are display geometry only; use
`--no-vectors` to hide them. There is no lower legend or numeric overlay, keeping
the feet visible. Command and keyboard information remains at the top.

## Environment conventions

Body axes are +x forward, +y left, +z up; positive yaw is counterclockwise from
above. Commands are `[vx, vy, yaw_rate]` in m/s, m/s, rad/s, bounded by
`[0.25, 0.15, 0.8]` in each direction. Training samples all components uniformly,
with a 20% probability of exactly zero command, every 3 seconds and at reset.
`--command-sampling balanced` instead divides non-standing commands equally
between forward/backward-only, lateral-only, yaw-only, and combined motion.
This provides direct training coverage for individual keyboard commands. With
20% standing probability, each of the five command modes occurs 20% of the time.
The default `uniform` distribution is retained for the initial baseline; the
sampling choice is saved in the resolved config and may change during a warm start.
Physics always starts from the same pose with zero velocity; the PRNG never
changes physical environment parameters.

The free-joint quaternion is MuJoCo's **wxyz**, rotating body to world. Free-joint
linear qvel is world-frame; angular qvel is body-frame. Observations rotate
translation/gravity into the body frame. The 64 values, before Brax normalization:

| Slice | Values | Scale |
| --- | --- | --- |
| 0:3 | unit world gravity `[0,0,-1]` expressed in body frame | 1 |
| 3:6 | local base linear velocity (m/s) | 2 |
| 6:9 | local base angular velocity (rad/s) | 0.25 |
| 9:29 | all 20 joint angles relative to nominal (rad) | 1 |
| 29:49 | all 20 joint velocities (rad/s) | 0.05 |
| 49:52 | velocity command | `[2,2,0.25]` |
| 52:64 | previous bounded 12D action | 1 |

Joint order follows named upstream actuators: head pan/tilt, left shoulder
pitch/roll/elbow, right shoulder pitch/roll/elbow, then left/right hip yaw,
hip roll, hip pitch, knee, ankle pitch, ankle roll. The 12 leg joints are policy
controlled in that order. Head and arm targets are held at nominal (shoulder roll
+1.25 left/-1.25 right; other head/arm joints zero). Left hip/knee/ankle pitch
nominals are `[-0.3, 0.6, 0.3]` rad; right is their negative. Initial height is
calculated from the unchanged foot boxes to leave 0.5 mm floor clearance.

Actions are bounded to `[-1,1]`, multiplied by 0.5 rad, and added to nominal
position-actuator targets. Targets are clamped to the intersection of actuator
and any model joint limits. Upstream joint limits are absent; actuator targets
are limited to approximately ±pi. Upstream force limits remain ±5 Nm.
Action repeat is applied only inside MJX `step`; Brax's action repeat is 1.
Falls are height below 0.18 m, upright cosine below 0.5, or nonfinite state.
The training wrapper separately marks 10-second time limits as truncations for
Brax's GAE bootstrapping/masking, and resets physics, command and previous action
together. Physical failure at the horizon remains termination, not truncation.
MJX Warp allocates 64 contacts per environment across the batch and 256 constraint
rows per environment. Buffer exhaustion is logged and aborts training rather
than silently accepting incomplete collision calculations.

## Rewards

Terms are weighted then multiplied by the 0.02 s control timestep **once**.
Brax reward scaling is 1. Negative penalties are never clipped to zero.

| Term | Formula before weight | Weight |
| --- | --- | --- |
| planar tracking | `exp(-||v_xy-command_xy||² / 0.15²)` | 3 |
| yaw tracking | `exp(-(yaw-command_yaw)² / 0.4²)` | 1 |
| upright | clipped cosine between projected gravity and `[0,0,-1]` | 0.3 |
| vertical velocity | squared body vertical velocity, (m/s)² | -0.5 |
| roll/pitch velocity | summed squares, (rad/s)² | -0.05 |
| effort | summed actuator torque squares, Nm² | -0.002 |
| action change | summed squared action differences | -0.02 |
| joint-limit violation | summed angle outside target bounds, rad | -1 |
| nominal posture | summed leg-angle error squares, rad² | -0.05 |
| extra standing posture | same error only at exactly zero requested velocity | 0; optional negative weight |
| foot slip | sum of per-foot load-weighted squared contact tangent speeds, (m/s)² | -5 (new-run default); 0 in selected policy |
| foot air time | completed unsupported seconds, paid on touchdown during motion commands | +2 (new-run default); 0 in selected policy |
| termination | physical failure indicator | -2 |

On a physical failure the entire step reward is replaced by `-2 * control_dt`;
no positive tracking reward is paid. Standing still under a 0.15 m/s forward
command earns only `exp(-1)` of the planar tracking term. Evaluation compares
this baseline explicitly. The slip penalty uses four fixed slots for each
original foot-box/floor pair, through MuJoCo contact sensors (force and relative
velocity in the contact frame). Each foot's squared tangential speed is averaged
with normal-force weights, then summed over feet carrying more than 1 N. It is
active at zero command too; airborne/unloaded feet receive no slip penalty.
These sensors do not change the 64 actor observations; the optional asymmetric
critic also uses their derived support information. Sensors are omitted when
all three foot-reward weights are zero and the asymmetric critic is off. Air time uses the
same force readings, sampled at the 2 ms physics timestep: a foot is supported
above 1 N, otherwise it is unsupported. After observing support during a motion
command, accumulate that foot's unsupported duration and pay it once on landing.
There is no minimum/target/maximum swing duration, phase, frequency, or foot-height
target. Equal total completed unsupported time earns equal raw reward, regardless
of how many steps it is split into. PPO discounting and other rewards still matter.

Motion requires any command component to exceed 5% of its trained bound. Standing
clears pending air time; resets clear all foot history, and the initial landing
earns nothing. A fall receives no air-time bonus. A 0.1 s swing earns
`2 × 0.1 × 0.02 = 0.004` on touchdown. Physics timestep measures duration; the
common control timestep scales the reward once. Air time means **unsupported**
time, so evaluate actual sole clearance too: unloading does not guarantee lift.
No clearance reward is enabled. The selected `outputs/op3-policy` predates these
foot rewards; trial evidence is recorded separately in `EXPERIMENTS.md`.

The optional `foot_contact_area` reward estimates the fraction of each original
sole near the floor, using area-weighted samples without double-counting the
overlapping sole boxes. It is gated by actual floor force (>1 N), then averaged
over touching feet; airborne feet earn nothing. A Gaussian distance width of
2 mm makes this a smooth geometric proxy for area contact. Penetration also
reduces the score. It uses the existing contact sensors and adds no observations.
The term is disabled by default; see `CONFIG.md` for its exact formula.

Evaluation reports mean/maximum completed unsupported duration per foot and the
fraction of samples with both feet unsupported, alongside slip and tracking.
These diagnostics use the 50 Hz saved trajectory; training detects contacts at
500 Hz, so short contact events can differ. Partial swings at the diagnostic
window edges are excluded from the duration statistics.

## Asymmetric critic experiment

The optional critic receives the original 64 observations plus left/right sole
height, unsupported time and contact state. The actor retains its 64 inputs and
128×128×128 network. Contact timers for the critic run at every physics substep,
including standing, independently of the air-time reward. See `CONFIG.md` for
units, scales and warm-start normalization.

Start from the original baseline's earlier 7,024,640-transition checkpoint, using
the latest experiment's unchanged rewards (upright +1, slip −5, air time +2):

```sh
uv run op3-train --config configs/asymmetric_critic.json \
  --initialize-from outputs/baseline/checkpoints/000007024640 \
  --output outputs/asymmetric-critic
uv run op3-play outputs/asymmetric-critic
```

This requests 30M additional transitions. The source export must exist locally.
`--asymmetric-critic` also enables this option with another configuration. The
actor weights/statistics are copied exactly; six zero rows extend the critic's
input kernel, preserving its initial predictions. Optimizer/RNG/step count
restart. Native play requires no privileged foot signals. After training, report
the trainer's final average total episode reward against previous runs and use
the launch command above for manual behavior evaluation.

The interrupted asymmetric-critic run continues from its last complete export
at 2,007,040 transitions. Its continuation uses the same settings and the remaining
budget, with optimizer/RNG/local step count restarted:

```sh
uv run op3-train --config outputs/asymmetric-critic/config.json \
  --initialize-from outputs/asymmetric-critic/checkpoints/000002007040 \
  --num-timesteps 27992960 --num-evals 15 \
  --output outputs/asymmetric-critic-continued
uv run op3-play outputs/asymmetric-critic-continued
```

The continuation is already launched locally; the training command above records
its recipe and requires a new output directory if repeated. Its log is
`outputs/asymmetric-critic-continued.log`; `finished.json` inside the run directory
marks completion. See `CONFIG.md` for step accounting.

## Upright and sole-area experiment

The full upright/sole-area experiment triples the preceding run's upright weight
from 1 to 3 and adds a 0.5 contact-area weight, retaining the asymmetric critic:

```sh
uv run op3-train --config configs/upright_contact_area.json \
  --initialize-from outputs/asymmetric-critic-continued/checkpoints/000028098560 \
  --output outputs/upright-contact-area
uv run op3-play outputs/upright-contact-area
```

This requests 30M additional transitions from the latest policy weights and
normalization; optimizer/RNG/step count restart. The output directory must be
new. The changed rewards make total episode returns incomparable without
accounting for those changes. Use the viewer for behavior assessment.

The local run was stopped at the user's request after the 24,084,480-step
checkpoint failed to improve on the previous evaluation. That is the latest
export opened by the play command above; the 30M budget was not completed.

To train fresh actor/critic parameters and observation statistics for 40M steps
with those same reward weights:

```sh
uv run op3-train --config configs/upright_contact_area_scratch_40m.json \
  --output outputs/upright-contact-area-scratch-40m
uv run op3-play outputs/upright-contact-area-scratch-40m
```

This uses seed 0 and 21 evaluations, preserving the roughly 2M-step checkpoint
interval. No checkpoint is loaded. The output directory must be new.
The local run completed 40,140,800 transitions; its final trainer mean episode
return was 71.7568, versus 71.2868 for the preceding warm-start experiment under
the same rewards. See `EXPERIMENTS.md` for the comparison and limitations.

## Two-second torque-symmetry experiment

Train from scratch for 30M transitions with a penalty on the difference between
left/right cumulative squared joint torque over a sliding two-second window:

```sh
uv run op3-train --config configs/torque_symmetry_scratch_30m.json \
  --output outputs/torque-symmetry-scratch-30m
uv run op3-play outputs/torque-symmetry-scratch-30m
```

The penalty sums absolute differences between the six mirrored leg-joint
integrals, with weight −0.1. All physics substeps contribute; episode resets
clear history. Other settings match the preceding 40M scratch run, including
upright +3, sole area +0.5 and the asymmetric critic. No checkpoint is loaded.
The output directory must be new; Brax rounds to 30,105,600 transitions.
See `CONFIG.md` for the exact formula and `EXPERIMENTS.md` for results.

The local run completed all 30,105,600 transitions. Final trainer mean episode
return was 34.2463, including −27.8922 from torque symmetry, versus 71.7568 for
the preceding 40M scratch run without this penalty. These totals use different
reward definitions. No post-training behavior suite was run; use the play
command above to assess the gait.

## Focused checks

```sh
uv run ruff check src scripts
uv run python scripts/check_env.py
uv run python scripts/check_env.py --gpu --output outputs/checks.json
uv run python scripts/check_slip_controls.py
uv run python scripts/check_air_time.py
uv run python scripts/check_standing_resets.py --gpu
uv run python scripts/check_asymmetric_critic.py --gpu
uv run python scripts/check_viewer_vectors.py
uv run python scripts/check_contact_area.py --gpu
uv run python scripts/check_torque_symmetry.py --gpu
# Check a reduced hull profile with the same native/MJX action sequence:
uv run python scripts/check_env.py --gpu --collision-maxhullvert 8
```

The GPU check covers fixed physical resets under different command seeds,
finite steps, a one-second native/MJX action-sequence comparison, observation
parity, correct reset of command/action history, and fall versus timeout flags.
A short PPO run, export reload, deterministic evaluation and viewer startup are
separate integration checks. Source responsibilities follow `AGENTS.md`; there
is no broad test framework or custom RL implementation.

Measured on the development machine with the locked dependencies: maximum native/MJX
qpos discrepancy `5.60e-6` and qvel discrepancy `5.22e-5` over the check's one-second
small-action sequence; reset/timeout checks passed (`outputs/checks.json`). The
three-second viewer startup passed. Actual X11 key events sent to the newly
launched viewer verified W/S, A/D, Q/E, Space, R, and Escape, including title
updates and clean exit (`outputs/viewer-keyboard-check.json`). Command clamping
was also checked directly. The selected packaged policy also passes a five-second
viewer startup check (`outputs/demo-viewer.log`).

The selected policy also survives ten-second MJX forward/turn tests and the
50-second command-change sequence (`outputs/demo-mjx/`). Settled MJX forward
speed is 0.149 m/s; left/right yaw is +0.422/−0.426 rad/s. Change-sequence planar
and yaw RMSE are 0.0174 m/s and 0.0914 rad/s, compared with native 0.0171 and
0.0878. CPU/GPU actions on identical observations differ by at most `3.88e-7`.
Contact-rich trajectories are not identical: one-second identical-action replay
has worst qpos/qvel component differences 0.0167/0.417 across the tested cases
(`outputs/demo-parity.json`). Thus the tighter small-action check above is not a
claim of exact full-policy trajectory parity. MJX evaluation's geometry/contact
diagnostics use native forward recomputation at the MJX state.

The original baseline requested 30M steps and was stopped after saving
16,056,320; refinement used the evaluated 15,052,800 export. The reproduction
recipe above stops the first stage at that export instead. Both stages use seed
0, 512 environments, the same rewards/physics and default network/PPO settings.
Refinement completed 15,052,800 additional steps in 826.8 s, with approximately
23k control transitions/s after compilation. Checkpoint selection remains
necessary; do not assume the latest export is the best policy.
