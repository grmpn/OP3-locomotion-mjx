# OP3 locomotion with MJX and PPO

A small Robotis OP3 walking baseline using direct MuJoCo MJX physics and Brax's
standard PPO trainer. The native MuJoCo demo loads exported policy parameters and
accepts body-frame velocity commands from the keyboard. There is no domain
randomization, terrain generation, reset noise, or gait generator.

**Current validation:** GPU reset/step, native/MJX parity, a 640-step GPU PPO
update, policy export/reload, rendered inference, and viewer startup pass. The
smoke policy in `outputs/smoke-compatible` stands, but does not walk or turn on
command. A trained walking policy has not yet been demonstrated. Run artifacts
and evaluation reports live in `outputs/` (ignored by Git).

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
No mesh decimation is currently applied. Training and inference use this same
loader, and exports include the collision profile.

## Train and save a policy

```sh
# A short pipeline check, not a walking policy:
uv run op3-train --output outputs/smoke --num-timesteps 640 \
  --num-envs 32 --batch-size 16 --num-minibatches 2 \
  --num-evals 2 --num-eval-envs 4

# Baseline training (GPU required):
uv run op3-train --output outputs/baseline --num-timesteps 30000000
```

The output directory must be new. Each run prints the selected GPU and records
`config.json`, `mapping.json`, `run.json`, `metrics.jsonl`, and exports under
`checkpoints/<12-digit-step>/`. A `checkpoints/latest` pointer selects the last
complete export. Metrics include individual weighted reward terms, tracking
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
same environment/network configuration; this is explicitly not exact resume.

## Evaluate and inspect behavior

```sh
uv run op3-evaluate outputs/baseline
uv run op3-evaluate outputs/baseline --physics mjx
# Render videos (use MUJOCO_GL=egl on a compatible headless Linux host):
uv run op3-evaluate outputs/baseline --video --cases stand forward turn_left changes
# Compare against fixed standing targets to detect a stand-still solution:
uv run op3-evaluate outputs/baseline --nominal
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
speed while supported. Support requires >1 N of floor normal force; clearance
uses the original foot boxes. Supported foot-body speed is a slip proxy and also
includes rolling, so it is not an exact slip measurement. These diagnostics
exclude the first second (or half of a shorter rollout). Compressed traces contain
commands, velocities, joint state, actions, foot positions, clearances, and
normal forces. Settled tracking RMSE is reported for each command segment. Videos and traces
are necessary to distinguish stepping from sliding; foot-height variation alone
does not prove a walking gait.

Evaluation stops a case on a fall rather than resetting and hiding failures.
The `changes` case preserves physical state across command updates. Native
MuJoCo evaluation uses CPU policy inference; `--physics mjx` uses the training
physics backend on GPU. Reports are stored alongside the selected checkpoint,
or in `--output DIRECTORY`.

## Keyboard demo

```sh
uv run op3-play outputs/baseline
```

The display requires GLFW/OpenGL. The renderer has its own key callback so it
does not inherit conflicting MuJoCo viewer shortcuts. Keys increment a persistent
command (holding a key repeats); releasing keys keeps the command:

| Keys | Command change |
| --- | --- |
| W / S | forward / backward, 0.05 m/s increments |
| A / D | left / right, 0.05 m/s increments |
| Q / E | left / right yaw, 0.2 rad/s increments |
| Space | zero all commands |
| R | fixed standing reset and zero command |
| Escape | exit |

The title and on-screen overlay display `[vx, vy, yaw_rate]`. Commands clamp to
the export's trained range. The camera follows the torso; a fall freezes physics
until R. Simulation and inference run at 50 Hz, with ten 2 ms MuJoCo substeps per
action, matching training. If the host cannot keep up, simulated time runs more
slowly. `--seconds 5` performs a finite viewer startup check.

## Environment conventions

Body axes are +x forward, +y left, +z up; positive yaw is counterclockwise from
above. Commands are `[vx, vy, yaw_rate]` in m/s, m/s, rad/s, bounded by
`[0.25, 0.15, 0.8]` in each direction. Training samples all components uniformly,
with a 20% probability of exactly zero command, every 3 seconds and at reset.
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
| termination | physical failure indicator | -2 |

On a physical failure the entire step reward is replaced by `-2 * control_dt`;
no positive tracking reward is paid. Standing still under a 0.15 m/s forward
command earns only `exp(-1)` of the planar tracking term. Evaluation compares
this baseline explicitly. No foot-slip, clearance, or air-time shaping is enabled
until rollout evidence justifies it.

## Focused checks

```sh
uv run ruff check src scripts
uv run python scripts/check_env.py
uv run python scripts/check_env.py --gpu --output outputs/checks.json
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
was also checked directly. These are integration checks; longer
closed-loop walking/turning evaluation is still required.

The initial baseline run uses 512 environments with otherwise default settings,
seed 0, and 30 million requested steps, exporting approximately every million
steps. Its seven-million-step checkpoint reaches about 0.18 m/s for a 0.15 m/s
forward command, with repeated foot lift/landing, but substantial uncommanded yaw.
Rightward motion and in-place turns remain poor. This intermediate checkpoint is
not the final walking deliverable; reports are in `outputs/baseline-eval-7m/`.
