# Current configuration

Current values describe the selected policy in `outputs/op3-policy/`
(refinement checkpoint `000012042240`). Sources: its `run_config.json` and
`mapping.json`, plus `src/op3_locomotion/{config,env,train,policy,rewards,observations}.py`.
“Default if different” lists values used by a fresh training run without overrides.
A dash means the code default matches the current value.

## PPO hyperparameters

| Setting | Current value | Default if different |
| --- | --- | --- |
| `seed` | 0 | — |
| `num_timesteps` | 15,000,000 requested control transitions | 30,000,000 |
| `num_envs` | 512 | 256 |
| `batch_size` | 128 unroll sequences per minibatch | — |
| `num_minibatches` | 4 | — |
| `unroll_length` | 20 control steps (0.4 s) | — |
| `num_updates_per_batch` | 4 passes through the minibatches | — |
| `learning_rate` | 0.0003 | — |
| `entropy_cost` | 0.01 | — |
| `discounting` | 0.99 per control step | — |
| `gae_lambda` | 0.95 | — |
| `clipping_epsilon` | 0.2 (policy probability-ratio clip) | — |
| `max_grad_norm` | 1.0 (global gradient norm) | — |
| `num_evals` | 16 | 11 |
| `num_eval_envs` | 32 | — |
| `hidden_sizes` | `[128, 128, 128]` for both actor and critic | — |

Each rollout batch contains `128 × 4 × 20 = 10,240` transitions; each minibatch
contains `128 × 20 = 2,560`. Each batch receives `4 × 4 = 16` optimizer updates.
`batch_size × num_minibatches` must be divisible by `num_envs`.

### Trainer and network settings

Explicit trainer settings and effective defaults from locked Brax 0.14.2/Optax:

| Setting | Value |
| --- | --- |
| Algorithm | Standard Brax PPO; separate actor and scalar-value critic |
| Optimizer | Adam; `b1=0.9`, `b2=0.999`, `eps=1e-8`, `eps_root=0`; Nesterov off |
| Learning-rate schedule | None; constant learning rate |
| Adaptive-schedule defaults (inactive) | `desired_kl=0.01`; min/max learning rate `1e-5` / `1e-2` |
| `normalize_advantage` | `True`; denominator epsilon `1e-8` |
| `vf_loss_coefficient` | 0.5; scalar value loss is `0.5 × 0.5 × mean(error²)` |
| `clipping_epsilon_value` | `None` (value clipping off) |
| `use_distributional_critic` | `False` |
| `bootstrap_on_timeout` | `False` (Brax's optional timeout reward-bootstrap switch); truncation flags are still supplied to GAE |
| `normalize_observations` | `True`; same saved statistics for actor, critic and inference |
| `normalize_observations_mode` | `welford` |
| `normalize_observations_std_eps` | 0.0; running standard deviation bounded to `[1e-6, 1e6]` |
| `normalize_until_count` | `None` (statistics update throughout training) |
| Normalized observation clipping | None |
| `reward_scaling` | 1.0 |
| Hidden activation | Swish / SiLU |
| Weight initialization | LeCun uniform; zero biases; no layer normalization |
| Action distribution | Tanh-normal; actor outputs 12 means and 12 scale logits |
| Gaussian standard deviation | `(softplus(scale_logits) + 0.001) × 1.0`; observation-dependent |
| Actor output activation | Linear distribution parameters, then tanh action transform |
| Critic output | One linear scalar |
| Training actions | Stochastic, seeded PPO exploration |
| `deterministic_eval` / play inference | `True` / deterministic |
| `num_resets_per_eval` | 0 additional host resets; episode auto-reset remains enabled |
| `wrap_env` / `use_pmap_on_reset` | `True` / `True` |
| `vision` / `augment_pixels` | `False` / `False` |
| `run_evals` / `log_training_metrics` | `True` / `True` |
| `training_metrics_steps` | 51,200 current; 25,600 default (`max(num_envs × 100, 10000)`) |
| `max_devices_per_host` | 1 GPU; full training refuses CPU fallback |
| Current warm start | Baseline checkpoint `000015052800`; actor, critic and observation statistics restored |
| Warm-start training state | Optimizer, rollout/RNG state and step count restarted; `restore_value_fn=True` |

## Environment and control

| Setting | Current value | Default if different |
| --- | --- | --- |
| `physics_backend` | `warp` (direct MuJoCo MJX) | — |
| `collision_profile` | `locomotion`: head/arm contacts disabled; torso/leg/foot contacts retained | — |
| `collision_maxhullvert` | −1 (upstream collision hulls) | — |
| `physics_dt` | 0.002 s (500 Hz) | — |
| `control_dt` | 0.02 s (50 Hz) | — |
| `action_repeat` (derived) | 10 physics steps per control step | — |
| Brax `action_repeat` | 1; repeat applied only inside environment step | — |
| `episode_seconds` / `episode_steps` | 10 s / 500 control steps | — |
| `command_seconds` / `command_steps` | 3 s / 150 control steps | — |
| `command_max` | `[0.25, 0.15, 0.8]`; symmetric bounds in m/s, m/s, rad/s | — |
| `standing_probability` | 0.2 | — |
| `lift_reset_probability` | 0.0 (fixed standing reset) | — |
| `lift_reset_knee_flexion` | 0.45 rad extra knee flexion in optional raised pose | — |
| `lift_reset_hip_roll` | 0.15 rad magnitude in optional raised pose | — |
| `command_sampling` | `balanced` | `uniform` |
| `action_scale` | 0.5 rad per normalized action unit | — |
| `min_height` | 0.18 m base height | — |
| `min_up_cosine` | 0.5 | — |

Commands are body-frame `[vx, vy, yaw_rate]`: +x forward, +y left, +z up;
positive yaw is counterclockwise from above. Commands are sampled at reset and
every 150 control steps. Balanced sampling assigns 20% probability each to
standing, forward/backward-only, lateral-only, yaw-only, and combined motion;
active components are uniform within their signed bounds. Uniform sampling uses
20% standing and 80% jointly sampled forward/lateral/yaw commands.

| Model, reset and runtime setting | Value |
| --- | --- |
| Model | Menagerie Robotis OP3, revision `98a70c569c4c1b6da467a20012f766ee1f8dc521` |
| Terrain | Original `scene.xml` floor |
| Physics parameters | Upstream masses, inertias, friction, solver settings, actuator gains and force limits; timestep/collision changes listed above |
| Domain randomization / reset perturbations / sensor noise / delays | Disabled |
| Initial base position | `[0, 0, approximately 0.26981732]` m |
| Initial base quaternion | `[1, 0, 0, 0]` in wxyz order |
| Initial joint pose | Nominal pose below; fixed across seeds |
| Initial velocities / previous action | All zero |
| Initial sole clearance | 0.0005 m, resolved from original foot boxes |
| Fall / invalid-state termination | Base height `<0.18`, upright cosine `<0.5`, nonfinite qpos/qvel/actuator force, or MJX buffer overflow |
| Time limit | Truncation at 500 steps; physical failure at the horizon remains termination |
| Warp graph mode | `WARP_STAGED` |
| Warp `naconmax` / `naccdmax` | Each `64 × max(num_envs, num_eval_envs)`: 32,768 current; 16,384 default |
| Warp `njmax` | 256 constraint rows per environment |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 0.35 unless overridden in the environment |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` unless overridden in the environment |

Optional raised-foot resets select nominal standing with probability `1-p`,
left raised with `p/2`, or right raised with `p/2`. The choice uses a separate
`fold_in(reset_key, 0x4C494654)` stream; command and PPO streams are unchanged.
The poses are precomputed once on the host and selected as fixed-shape MJX data
inside reset. No rollout IK or physics-parameter randomization is used.

For left raised, add −0.225 rad at left hip pitch, +0.45 at left knee, +0.225 at
left ankle pitch; add −0.15 at both hip rolls and both ankle rolls. For right
raised, mirror the side and all signs. Lower the base to put the lowest sole
0.0005 m above the original floor; the lifted sole starts about 0.020094 m high.
Base orientation, x/y position and all velocities retain their nominal values;
initial actuator targets match the chosen pose. Previous action and air-time
history are zero. Nominal observation offsets and action targets still refer to
the original standing pose. The initial raised-foot landing earns no air reward.

### Actions and nominal pose

Twelve actions are clipped to `[-1, 1]`. Controlled targets are
`nominal_angle + 0.5 × action` rad, clamped to the intersection of actuator and
joint limits. The pinned model has no joint-position limits; actuator target
limits are approximately ±3.141592 rad and actuator force limits are ±5 Nm.
The eight head/arm position actuators hold their nominal targets.

Action order (left leg, then right):

```text
l_hip_yaw, l_hip_roll, l_hip_pitch, l_knee, l_ank_pitch, l_ank_roll,
r_hip_yaw, r_hip_roll, r_hip_pitch, r_knee, r_ank_pitch, r_ank_roll
```

| Nominal joint angle | Radians |
| --- | ---: |
| `l_sho_roll` | +1.25 |
| `r_sho_roll` | −1.25 |
| `l_hip_pitch` / `r_hip_pitch` | −0.3 / +0.3 |
| `l_knee` / `r_knee` | +0.6 / −0.6 |
| `l_ank_pitch` / `r_ank_pitch` | +0.3 / −0.3 |
| All other joints | 0 |

## Reward terms

`v` and `ω` are unscaled body-frame base velocities, `c` is the velocity command,
`g` is unit world gravity `[0, 0, −1]` expressed in the body frame, `a` is the
clipped action, and `q₀` is the nominal joint pose. `q`, `lower`, and `upper`
refer to the 12 controlled leg joints; effort uses all 20 actuator torques `τ`.

| Term | Formula before weight | Weight |
| --- | --- | ---: |
| `linear_tracking` | `exp(-sum((v_xy - c_xy)²) / sigma_linear²)` | +3.0 |
| `yaw_tracking` | `exp(-(ω_z - c_yaw)² / sigma_yaw²)` | +1.0 |
| `upright` | `clip(-g_z, 0, 1)` | +0.3 |
| `vertical_velocity` | `v_z²` in (m/s)² | −0.5 |
| `roll_pitch_velocity` | `ω_x² + ω_y²` in (rad/s)² | −0.05 |
| `effort` | `sum(τ²)` in Nm² | −0.002 |
| `action_rate` | `sum((a - previous_action)²)` | −0.02 |
| `joint_limits` | `sum(max(lower - q, 0) + max(q - upper, 0))` in rad | −1.0 |
| `posture` | `sum((q - q₀)²)` in rad² | −0.05 |
| `standing_posture` | Same posture error, only when all three requested velocities equal zero | 0; optional additional penalty |
| `termination` | Physical failure indicator | −2.0 |
| `foot_slip` | Sum over supported feet of `sum(F_n × (v_t1² + v_t2²)) / sum(F_n)`, (m/s)² | 0 in selected policy; **−5.0 code default** |
| `foot_air_time` | Sum of completed unsupported durations at touchdown, seconds | 0 in selected policy; **+2.0 code default** |

| Tracking width | Value |
| --- | --- |
| `sigma_linear` | 0.15 m/s |
| `sigma_yaw` | 0.4 rad/s |
| `slip_contact_force` | 1.0 N; require summed foot normal force strictly above this |
| `air_time_contact_force` | 1.0 N; a foot is supported strictly above this summed normal force |
| `air_time_command_threshold` | 0.05; require `max(abs(command) / command_max) > 0.05` |

Per-step reward is `control_dt × sum(weight × term)`, with `control_dt=0.02`;
Brax adds no further scaling (`reward_scaling=1.0`). On physical failure, replace
the entire reward with `−2 × 0.02 = −0.04`; all other paid terms are zero.
Time-limit truncation alone does not receive this failure penalty. Reward terms
are logged separately after weighting and timestep scaling, including
`reward/foot_slip` and `reward/foot_air_time`. The selected policy has both foot
weights zero; new runs default to −5 slip and +2 air time. Old configuration files
lacking either field explicitly load that weight as zero. No clearance,
phase/cadence or body-height reward is enabled.

Slip uses MuJoCo contact sensors matching each original foot box against `floor`:
four slots per box, two boxes per foot, force and relative linear velocity in
contact coordinates (normal x, tangential yz), `maxforce` reduction. Each sensor
has 24 values, 96 total, used only for reward and never appended to observations.
The force-weighted squared speed is summed over loaded feet at the last physics
substep, then weighted and timestep-scaled once. Missing contacts are zero;
airborne/unloaded feet are excluded. Zero velocity commands do not disable this
penalty. Sensors are omitted only when both foot-reward weights are zero.

Air-time history is sampled at every 0.002 s physics substep using the same floor
force sensors. Each foot must first be supported under a motion command to arm
its timer; then unsupported samples accrue duration. The next supported sample
pays that duration and clears the timer, so holding contact does not pay again.
Touchdowns across the ten substeps are summed and weighted/scaled once at the
control step. For example, a completed 0.1 s swing pays `2 × 0.1 × 0.02 = 0.004`.
The physics timestep measures seconds, not an additional reward scaling factor.

There is no target/minimum/maximum swing duration, height target, phase clock,
frequency target, alternation target or per-step-count bonus. Equal accumulated
completed air time gives equal raw reward; discounting still favors earlier
payment. At current bounds, the motion gate opens if `abs(vx)>0.0125` m/s,
`abs(vy)>0.0075` m/s, or `abs(yaw)>0.04` rad/s. Otherwise both timers and eligibility
flags clear. Episode resets clear them too, and initial landings are ineligible.
The usual failure override removes any bonus on a fall. An unfinished swing earns
nothing at timeout. These two timers and two eligibility flags are reward-only
state and are not appended to policy observations.

`collision_maxhullvert` accepts −1 or integers ≥4. A positive value caps the
convex hull of each collision mesh using MuJoCo's compiler, including inactive
head/arm colliders. Visual meshes, explicit inertias, original foot boxes and
floor remain unchanged. It changes the mesh contact approximation and is saved
in the policy mapping. Current trial caps are 16 and 8; the default and selected
policy remain −1. Warm starts may explicitly change rewards, command sampling
or this cap; evaluation/play require the saved cap.

## Observations

Actor and critic share **64 values**, in the following order. Slices use Python's
zero-based, end-exclusive convention. Scales are applied before running-statistic
normalization; the saved normalizer is reused at inference.

| Slice | Count | Values | Scale |
| --- | ---: | --- | --- |
| `[0:3]` | 3 | Unit gravity projected into the body frame | 1 |
| `[3:6]` | 3 | Body-frame base linear velocity `[vx, vy, vz]`, m/s | 2.0 |
| `[6:9]` | 3 | Body-frame base angular velocity `[ωx, ωy, ωz]`, rad/s | 0.25 |
| `[9:29]` | 20 | Joint positions minus nominal joint positions, rad | 1 |
| `[29:49]` | 20 | Joint velocities, rad/s | 0.05 |
| `[49:52]` | 3 | Command `[vx, vy, yaw_rate]` | `[2.0, 2.0, 0.25]` |
| `[52:64]` | 12 | Previous normalized action, in action order above | 1 |

The 20 joint observations follow actuator order:

```text
head_pan, head_tilt,
l_sho_pitch, l_sho_roll, l_el, r_sho_pitch, r_sho_roll, r_el,
l_hip_yaw, l_hip_roll, l_hip_pitch, l_knee, l_ank_pitch, l_ank_roll,
r_hip_yaw, r_hip_roll, r_hip_pitch, r_knee, r_ank_pitch, r_ank_roll
```

MuJoCo's free-joint quaternion is **wxyz**, rotating body vectors to world.
World gravity and world-frame linear qvel are rotated by its inverse into the
body frame. Free-joint angular qvel is already body-frame. There is no frame
stacking, privileged critic observation, absolute position/heading, contact
observation, terrain input, or added observation noise.

## Standalone evaluation defaults

| Setting | Value |
| --- | --- |
| Physics / inference | Native MuJoCo / deterministic CPU policy; optional MJX GPU physics |
| Seed | 123 |
| Segment duration | 5 s |
| Command magnitude | 60% of trained bounds: `[0.15, 0.09, 0.48]` |
| Cases | Stand, forward, backward, left, right, left/right yaw, continuous changes |
| Settled-metric exclusion | First 1 s of each segment, or half if shorter |
| Air-time diagnostics | Mean/max completed unsupported duration per foot; both-feet-unsupported fraction |
| Diagnostic contact sampling | 50 Hz, >1 N support; exclude partial swings at window edges |
| `--reset-pose` | `standing` default; `left` / `right` force the corresponding raised foot |
| Posture diagnostic | Mean summed squared 12-leg-joint deviation from nominal, rad², over the case |

## Keyboard command settings

The viewer tracks GLFW press/release state, ignoring key-repeat events. Each
50 Hz control tick approaches the signed bound on a held axis, or zero on a
released axis. W/S control x, A/D y, Q/E yaw; opposing held keys cancel.

| Setting | Value |
| --- | --- |
| Zero-to-bound ramp time | 0.5 s of control time |
| Bound-to-zero release time | 0.35 s of control time |
| Ramp rates at current bounds | `[0.5, 0.3, 1.6]` m/s², m/s², rad/s² |
| Release rates at current bounds | `[0.714286, 0.428571, 2.285714]` m/s², m/s², rad/s² |
| Limits | Saved policy `command_max`; no overshoot |
| Immediate clear | Space, R (also resets physics), window focus loss |
| Clear behavior | Clears held-key state; release/repress to resume |
| Exit | Escape |

These viewer ramps do not alter training's sampled command schedule. Existing
catch-up cap is 0.25 s; command ramps follow simulated time if rendering is slow.

## Slip and collision trials (2026-11-07)

These are experimental exports, separate from the selected `outputs/op3-policy`.
All four trials initialize actor, critic and observation normalization from that
export. They restart the optimizer/RNG/step count and use its balanced commands.
PPO settings match the current column above except `num_timesteps=1000000` and
`num_evals=3`; actual final step is 1,003,520. Seed is 0, with 512 training and
32 evaluation environments, batch 128 × 4, and unroll length 20.

| Run under `outputs/` | Slip weight | Hull cap |
| --- | ---: | ---: |
| `slip-hull-full` | −5 | −1 |
| `slip-hull-16` | −5 | 16 |
| `slip-hull-8` | −5 | 8 |
| `slip-control-zero` | 0 | −1 |

Each uses the code-default 1 N slip support threshold. Other reward, environment,
network and normalization settings match the selected policy. Standalone native
evaluations use seed 123, 10 s per command segment, and the standard eight cases.
Results and selection decisions belong in `EXPERIMENTS.md`.

## Air-time trials (2026-11-07)

`outputs/air-time-2` and `outputs/air-time-control-0` both initialize actor, critic
and normalization from `outputs/slip-hull-full` checkpoint `000001003520`, with
optimizer/RNG/step count restarted. Both use original hulls (−1), slip weight −5,
balanced commands, seed 0, and the same 512-environment PPO settings as the slip
trials: 1,000,000 requested transitions and three evaluations. Air-time weights
are +2 and 0 respectively; support threshold is 1 N and command threshold 0.05.
All other settings match the slip source. Standalone evaluation uses native
MuJoCo, seed 123, 10 s per segment and all eight command cases. These experimental
exports are separate from the selected demo; results belong in `EXPERIMENTS.md`.

## Standing-posture and raised-reset trials (2026-11-07)

All three runs initialize from `outputs/air-time-2` checkpoint `000001003520`,
retaining slip −5, air time +2, original hulls, balanced commands, seed 0,
512 training/32 evaluation environments and all source PPO settings. Each requests
1,000,000 transitions with three evaluations; optimizer/RNG/step count restart.

| Run under `outputs/` | Additional standing posture weight | Raised-reset probability |
| --- | ---: | ---: |
| `standing-lift-control` | 0 | 0 |
| `standing-posture-10x` | −0.45 | 0 |
| `lift-reset-25` | 0 | 0.25 |

The posture trial's total coefficient is −0.5 at exactly zero requested velocity
and −0.05 otherwise. Raised poses use the 0.45 rad knee and 0.15 rad roll offsets
above, with equal left/right selection. Both features remain disabled by default.
Standalone evaluations use 10 s per segment and all eight standard cases from
standing; separate left/right raised-start checks measure recovery. The selected
demo remains `outputs/op3-policy`.
