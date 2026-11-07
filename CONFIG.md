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
| `physics_dt` | 0.002 s (500 Hz) | — |
| `control_dt` | 0.02 s (50 Hz) | — |
| `action_repeat` (derived) | 10 physics steps per control step | — |
| Brax `action_repeat` | 1; repeat applied only inside environment step | — |
| `episode_seconds` / `episode_steps` | 10 s / 500 control steps | — |
| `command_seconds` / `command_steps` | 3 s / 150 control steps | — |
| `command_max` | `[0.25, 0.15, 0.8]`; symmetric bounds in m/s, m/s, rad/s | — |
| `standing_probability` | 0.2 | — |
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
| `termination` | Physical failure indicator | −2.0 |

| Tracking width | Value |
| --- | --- |
| `sigma_linear` | 0.15 m/s |
| `sigma_yaw` | 0.4 rad/s |

Per-step reward is `control_dt × sum(weight × term)`, with `control_dt=0.02`;
Brax adds no further scaling (`reward_scaling=1.0`). On physical failure, replace
the entire reward with `−2 × 0.02 = −0.04`; all other paid terms are zero.
Time-limit truncation alone does not receive this failure penalty. Reward terms
are logged separately after weighting and timestep scaling. No foot-slip,
clearance, air-time or body-height reward is enabled.

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
