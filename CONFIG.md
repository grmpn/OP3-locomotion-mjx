# Current configuration

Current values describe the selected policy in `outputs/op3-policy/`
(refinement checkpoint `000012042240`). Sources: its `run_config.json` and
`mapping.json`, plus `src/op3_locomotion/{config,env,train,policy,rewards,observations}.py`.
“Default if different” lists values used by a fresh training run without overrides.
A dash means the code default matches the current value.

The current training experiment is the [20M torque-symmetry warm start](#current-20m-torque-symmetry-warm-start-2026-11-08-utc)
below; its overrides are separate from the selected demo and code defaults.

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
| `normalize_observations` | `True`; selected policy shares statistics; asymmetric critic has separate observation leaves |
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
| `asymmetric_critic` | `False`; optional 70-value critic described below | — |
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
| `torque_symmetry` | `sum_j(abs(E_left,j - E_right,j))`, with `E` the sliding integral of joint torque squared, (Nm)² s | 0 default/selected policy; current experiment −0.1 |
| `action_rate` | `sum((a - previous_action)²)` | −0.02 |
| `joint_limits` | `sum(max(lower - q, 0) + max(q - upper, 0))` in rad | −1.0 |
| `posture` | `sum((q - q₀)²)` in rad² | −0.05 |
| `standing_posture` | Same posture error, only when all three requested velocities equal zero | 0; optional additional penalty |
| `termination` | Physical failure indicator | −2.0 |
| `foot_slip` | Sum over supported feet of `sum(F_n × (v_t1² + v_t2²)) / sum(F_n)`, (m/s)² | 0 in selected policy; **−5.0 code default** |
| `foot_air_time` | Sum of completed unsupported durations at touchdown, seconds | 0 in selected policy; **+2.0 code default** |
| `foot_contact_area` | Mean smooth near-floor sole-area fraction over loaded feet; zero with no loaded feet | 0; experiment below uses +0.5 |

| Tracking width | Value |
| --- | --- |
| `sigma_linear` | 0.15 m/s |
| `sigma_yaw` | 0.4 rad/s |
| `slip_contact_force` | 1.0 N; require summed foot normal force strictly above this |
| `air_time_contact_force` | 1.0 N; a foot is supported strictly above this summed normal force |
| `air_time_command_threshold` | 0.05; require `max(abs(command) / command_max) > 0.05` |
| `area_contact_force` | 1.0 N; require summed foot normal force strictly above this |
| `area_height_sigma` | 0.002 m; Gaussian width for absolute sole-to-floor distance |
| `torque_window_seconds` | 2.0 s; integer multiple of `control_dt`, at least one control step |

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
has 24 values, 96 total. Raw sensor values are never appended to observations;
the optional critic uses the derived binary support state and air-time history.
The force-weighted squared speed is summed over loaded feet at the last physics
substep, then weighted and timestep-scaled once. Missing contacts are zero;
airborne/unloaded feet are excluded. Zero velocity commands do not disable this
penalty. Sensors are omitted only when all three foot-reward weights are zero and
the asymmetric critic is disabled.

The optional contact-area term partitions the union of the two original sole
boxes into five disjoint rectangles per foot. It uses a fixed 4×4 midpoint grid
in each rectangle (80 samples per foot), with cell areas as weights. The union
area is 0.00962 m² per foot; overlapping boxes contribute area only once. For
world sample height `h_i` relative to the default floor and sample area `A_i`,
each foot's score is `sum(A_i * exp(-(h_i / area_height_sigma)²)) / sum(A_i)`.
This is a smooth area-proximity estimate, not a measured physical contact patch.
It is maximal for a flat sole at floor height and penalizes penetration too.

Only feet with summed floor force > `area_contact_force` are included. Their
scores are averaged by the number of loaded feet; a foot with its local sole
normal pointing sideways/down earns zero. No loaded feet means zero reward.
There is no command gate: standing and moving use the same rule. Sampling is
at the final physics substep, and weight/control-dt scaling is applied once.
This adds no observation or temporal state. Old exports and code defaults keep
the weight zero. The existing physical-failure override also zeros this term.

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

The actor always uses **64 values**; the default and selected-policy critic share
them. The optional asymmetric critic appends six values described below.
Slices use Python's
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
stacking, absolute position/heading, terrain input, or added observation noise.
The actor never receives privileged foot observations.

### Optional asymmetric critic

`env.asymmetric_critic=True` returns `state` (64 actor inputs) and
`privileged_state` (70 critic inputs). Brax's standard `policy_obs_key` and
`value_obs_key` select these branches. Both networks retain three 128-unit Swish
layers; actor outputs, action transforms, control timing and all rewards are
unchanged by this flag. The critic's first 64 values equal `state` exactly.

| Critic slice | Values, left then right | Scale before normalization |
| --- | --- | --- |
| `[64:66]` | `foot_height`: lowest point of each foot's two original sole boxes above the flat floor, m; signed, including penetration | 10 |
| `[66:68]` | `foot_air_time`: elapsed unsupported time since contact or reset, s | 5 |
| `[68:70]` | `foot_contact`: summed sole-floor normal load strictly above `air_time_contact_force` (1 N), float 0/1 | 1 |

Critic timers update every physics substep (2 ms), clear on support or episode
reset, and accumulate regardless of command, reward eligibility or reward weight.
They are separate from the command-gated reward timers. Height and contact use
the final substep's kinematics/sensors. Command changes retain critic timers;
auto-reset replaces both observation branches and all history together.
Initial timers are zero, initial sole clearance is 0.5 mm, and initial contacts
come from the reset state's sensors. Inference requires only the actor's 64 values.

Warm-starting a symmetric export copies the actor exactly and pads the critic's
first-layer kernel with six zero rows, preserving its initial value predictions.
Copy the original normalization mean, standard deviation, summed variance and
count into the actor branch and the first 64 critic entries. For the six added,
scaled features, initialize mean 0, standard deviation 1 and summed variance
equal to the restored sample count (a unit-variance prior). Brax's shared Welford
count then updates both branches normally; the actor's update rate is preserved.
Same-shape asymmetric exports reload unchanged; removing critic inputs during a
warm start is rejected. Optimizer, RNG and training step count still restart.

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

### Viewer orientation display

| Setting | Value |
| --- | --- |
| `--vectors` / `--no-vectors` | Enabled by default; decorative arrows only |
| Raw policy gravity | `base_motion(qpos, qvel)[0]`, identical to actor `obs[0:3]` before running normalization; current displayed pose |
| Green arrow | Body +Z rotated to world coordinates; length 0.24 m, width 0.006 m |
| Cyan arrow | Negative raw policy gravity rotated from body to world; unit direction, length 0.20 m, width 0.009 m |
| Orange arrow | Normalized `model.opt.gravity` in world coordinates; length 0.20 m, width 0.009 m |
| Shared arrow origin | Base position plus world offset `[0, -0.22, 0.12]` m; grey 1-pixel line back to base |
| Legend / numeric readouts | None; the lower view of the feet stays unobstructed |
| Geometry / update | Decorative scene geoms; each rendered frame; no physics, observation, policy or training changes |

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
1,000,000 transitions with three evaluations; actual final step is 1,003,520.
Optimizer/RNG/step count restart.

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

## Full upright / air-time run (2026-11-07)

Reproducible settings: `configs/upright_air_time.json`. Run directory:
`outputs/upright-air-time-full`. Initialize actor, critic, optimizer and
observation normalization from scratch, seed 0; no source checkpoint.

| Setting | Run value |
| --- | --- |
| `upright` | +1.0 (code default +0.3) |
| `posture` / `standing_posture` | −0.05 / 0; original posture penalty in every command mode |
| `foot_air_time` / `foot_slip` | +2 / −5 |
| `lift_reset_probability` | 0; fixed nominal standing start |
| `command_sampling` | `balanced` |
| `num_timesteps` | 30,000,000 requested control transitions |
| Completed export | 30,105,600 transitions; `checkpoints/000030105600` |
| `num_envs` / `num_eval_envs` | 512 / 32 |
| `num_evals` | 16, including the initial untrained policy |
| `batch_size` / `num_minibatches` / `unroll_length` | 128 / 4 / 20 |
| `collision_maxhullvert` | −1; original collision hulls |

Other reward weights/widths, PPO hyperparameters, 128×128×128 networks,
normalization, observations and 2 ms / 20 ms timing match the reference above.
Air time retains the touchdown duration reward with no cadence or clearance
target. Default code settings and the separately selected demo are unchanged.
Results belong in D024 of `EXPERIMENTS.md`.

## Asymmetric critic run (2026-11-07)

Reproducible settings: `configs/asymmetric_critic.json`. Run directory:
`outputs/asymmetric-critic`. Warm start from original baseline checkpoint
`outputs/baseline/checkpoints/000007024640`. Preserve actor weights, actor
normalization and the original critic function using the expansion above;
optimizer/RNG/step count restart. The asymmetric flag defaults to false.

Use the preceding full-run recipe unchanged except `asymmetric_critic=True`:
upright +1, slip −5, air time +2, posture −0.05, no extra standing penalty,
balanced commands, fixed standing resets, seed 0, 512 training/32 evaluation
environments, 16 evaluations, 30,000,000 requested additional transitions.
Both MLPs remain 128×128×128; actor input size 64, critic input size 70.
Other PPO, physics, command, reward and normalization settings are identical to
`configs/upright_air_time.json`. The selected demo remains `outputs/op3-policy`.

### Completed continuation (2026-11-08 UTC)

Run directory: `outputs/asymmetric-critic-continued`. Load the saved configuration
from `outputs/asymmetric-critic/config.json` and warm-start actor, critic and both
observation-normalization branches from
`outputs/asymmetric-critic/checkpoints/000002007040`. Optimizer, RNG and local
step count restart; this is not a full training-state resume.

Override only `num_timesteps=27_992_960` and `num_evals=15` (initial evaluation
plus 14 training epochs). This preserves the original 2,007,040-transition
evaluation/export interval after Brax rounds to complete batches. Completed
28,098,560 new transitions, or 30,105,600 including the saved
2,007,040 asymmetric-critic transitions. All other run settings above remain
unchanged, including seed 0. Continuation exports use the new local step count.

## Upright / sole-area warm-start experiment (2026-11-08 UTC)

Reproducible configuration: `configs/upright_contact_area.json`. Output:
`outputs/upright-contact-area`. Warm-start actor, critic and observation
normalization from `outputs/asymmetric-critic-continued/checkpoints/000028098560`;
optimizer, RNG and local step count restart.

| Setting | Experiment value |
| --- | --- |
| `upright` | +3.0; preceding asymmetric run +1.0; code default remains +0.3 |
| `foot_contact_area` | +0.5; new optional term, default 0 |
| `area_contact_force` / `area_height_sigma` | 1.0 N / 0.002 m |
| `foot_slip` / `foot_air_time` | −5 / +2, unchanged |
| `asymmetric_critic` | True; 64 actor / 70 critic inputs |
| `num_timesteps` / `num_evals` | 30,000,000 additional transitions / 16 evaluations |
| `num_envs` / `num_eval_envs` | 512 / 32 |
| `seed` / `command_sampling` / reset | 0 / balanced / fixed nominal standing |

All remaining reward, network, PPO, normalization, physics and control settings
match the preceding asymmetric run. The requested full-batch target was
30,105,600 new transitions; the user subsequently requested an early stop if
the next checkpoint did not improve. Latest saved/viewed export:
`outputs/upright-contact-area/checkpoints/000024084480` (24,084,480 additional
transitions). The saved run configuration retains the original 30M request;
`stopped.json` records the early stop. Individual area reward is logged as
`reward/foot_contact_area`. This experiment does not replace the separately
selected `outputs/op3-policy`.

## Preceding 40M scratch run (2026-11-08 UTC)

Configuration: `configs/upright_contact_area_scratch_40m.json`. Run directory:
`outputs/upright-contact-area-scratch-40m`. Actor, critic, observation statistics,
optimizer and RNG initialize fresh with seed 0; no `--initialize-from` source.

Reward weights and widths exactly match the preceding sole-area experiment:
upright +3, contact area +0.5, slip −5, air time +2, floor-contact threshold 1 N,
area height width 0.002 m. All other rewards, physics, fixed standing reset,
balanced commands, 64/70-input asymmetric networks, normalization settings and
PPO update settings remain identical.

The requested budget is 40,000,000 transitions with 21 evaluations (initial
plus 20 training epochs), retaining the 2,007,040-step evaluation/export cadence.
Brax rounds to 40,140,800 actual transitions, all completed. Final export:
`outputs/upright-contact-area-scratch-40m/checkpoints/000040140800`. Training used
512 environments, 32 evaluation environments, batch size 128 × 4 and unroll
length 20. The separately selected demo remains `outputs/op3-policy`.

## Preceding 30M torque-symmetry scratch run (2026-11-08 UTC)

Configuration: `configs/torque_symmetry_scratch_30m.json`. Run directory:
`outputs/torque-symmetry-scratch-30m`. Actor, critic, normalization, optimizer and
RNG initialize fresh with seed 0; no checkpoint is loaded.

Relative to the preceding 40M scratch recipe, change only `torque_symmetry` from
0 to −0.1, request 30,000,000 transitions and use 16 evaluations (initial plus
15 epochs). Completed all 30,105,600 rounded transitions at a 2,007,040-step
export interval. Final export:
`outputs/torque-symmetry-scratch-30m/checkpoints/000030105600`.
Retain 512 training/32 evaluation environments, batch 128 × 4,
unroll 20, balanced commands, fixed standing resets, upright +3, area +0.5,
slip −5 and air time +2. Other settings match the preceding recipe.

For each of the six mirrored joint pairs (hip yaw/roll/pitch, knee, ankle
pitch/roll), compute `E_side,j = sum(τ_side,j² × physics_dt)` over the trailing
two seconds. The actual unit-gear leg actuator torques are sampled at all ten
physics substeps per control step. The raw term is
`sum_j(abs(E_left,j - E_right,j))`, in (Nm)² s. Different joints cannot cancel
each other's imbalance; alternating left/right effort at the same joint can.
Head/arm actuators do not contribute. There is no normalization by total torque,
window duration, joint count, or actuator force limit, and no command gate.

The fixed-shape ring buffer contains 100 control-interval differences × six
joint pairs. It advances every 20 ms and sums its entries directly to avoid
running-sum drift. Reset zero-fills the window, so the first two seconds use
only elapsed episode history; no artificial pre-reset torque or delayed onset.
Command changes preserve history. Physical termination and time-limit
auto-reset both clear the buffer and index independently for each environment.
When weight is zero (including older configurations), history is empty and
substep accumulation is skipped.

The weighted rate is `−0.1 × raw_term`; the usual outer `control_dt=0.02`
converts it to per-step reward exactly once. The inner `physics_dt` belongs to
the torque integral, not reward scaling. A sustained raw imbalance of 1 (Nm)² s
costs 0.1 reward per simulated second. Existing failure-transition override
still applies. Log the paid contribution as `reward/torque_symmetry` alongside
all other terms. Actor/critic inputs remain 64/70; torque history is reward
state only and is not exposed to either MLP. Model, action mappings, observation
normalization, control timing and native inference are unchanged. Results and
experiment rationale belong in `EXPERIMENTS.md`.

## Current 20M torque-symmetry warm start (2026-11-08 UTC)

Configuration: `configs/torque_symmetry_warm_start_20m.json`. Run directory:
`outputs/torque-symmetry-warm-start-20m`. Initialize actor, critic and observation
normalization from the preceding 40M scratch run's final export:
`outputs/upright-contact-area-scratch-40m/checkpoints/000040140800`.
Optimizer, RNG, environment histories and local step count restart; this is a
policy warm start, not a full training-state resume.

Use the identical reward objective and settings from the 30M torque experiment:
`torque_symmetry=-0.1`, `torque_window_seconds=2.0`, upright +3, area +0.5,
slip −5, air time +2, 64/70-input asymmetric networks, balanced commands, fixed
standing resets, seed 0, 512 training/32 evaluation environments, batch 128 × 4
and unroll 20. All other PPO, model, timing and normalization settings are
unchanged. Override only `num_timesteps=20_000_000` and `num_evals=11` (initial
evaluation plus ten epochs), retaining the 2,007,040-step export cadence.
Expected full-batch total: 20,070,400 additional transitions, or 60,211,200 in
the saved source lineage. The separately selected demo remains `outputs/op3-policy`.
