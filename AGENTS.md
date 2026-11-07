# Project instructions

## Goal and scope

Build a simple PPO walking policy for the Robotis OP3 using JAX, MuJoCo MJX,
and Brax. Train on GPU. The final deliverable is a runnable simulation that
loads a trained policy and accepts keyboard commands for linear and angular
velocity.

- Use the [MuJoCo Menagerie OP3 model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotis_op3).
- Use its default `scene.xml` terrain. Do not introduce obstacles, heightfields,
  terrain generation, or a terrain curriculum.
- Do not implement domain randomization: no randomized physics, masses,
  friction, actuator properties, external pushes, sensor noise, or delays.
  Default to a fixed, valid standing configuration without reset perturbations.
  The explicitly requested raised-foot reset experiment may sample standing,
  left-foot-raised and right-foot-raised poses with a separate seeded stream.
  Keep this optional; it does not authorize other physical randomization.
- Sampling velocity commands and stochastic PPO exploration are allowed. Keep
  them seeded and separate from physical environment parameters.
- Keep the baseline small and understandable. Avoid custom RL algorithms,
  elaborate gait generators, extra frameworks, and unrelated features.

## Environment and dependencies

- Use `uv` for Python environment creation, dependency management, and execution.
  Keep dependencies and tool configuration in `pyproject.toml`; commit `uv.lock`
  and declare the supported Python version. Keep `.venv/` out of Git.
- Use `uv sync` and `uv run` in documented workflows. Do not introduce Conda,
  Poetry, or a separate manually maintained requirements file.
- Choose compatible versions of JAX, `jaxlib`, `mujoco`, `mujoco-mjx`, and `brax`.
  Configure GPU dependencies in the project and lockfile for the actual GPU and
  driver. Consult the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
  when selecting CUDA support; do not assume a CUDA version.
- Use MJX directly for physics and Brax's PPO training utilities. Avoid building
  on the legacy Brax physics pipelines; [Brax maintains its training code](https://github.com/google/brax).
- Training must check that JAX can use a GPU and report the selected device.
  Fail clearly rather than silently running a full training job on CPU.
  CPU execution is acceptable for lightweight checks and interactive inference.

## Repository structure

Use a `src/` package layout with clear responsibilities. This is the target
structure; create files as they become necessary, without empty scaffolding.

```text
pyproject.toml
uv.lock
README.md
CONFIG.md                 # Current hyperparameters, reward terms, and observations
AGENTS.md
src/op3_locomotion/
    __init__.py
    config.py             # Typed environment, reward, and PPO settings
    env.py                # MJX reset/step and the Brax training interface
    observations.py       # Shared observation construction and frame conversions
    rewards.py            # Small, named reward terms
    policy.py             # Shared network and action transformation
    checkpoints.py        # Save/load policy, normalization, and configuration
    train.py              # GPU PPO entry point
    evaluate.py           # Repeatable command-tracking evaluation
    play.py               # Viewer, keyboard input, and inference loop
scripts/
    fetch_assets.py       # Reproducible download of a pinned model revision
assets/robotis_op3/        # Model XML, meshes, and upstream license
outputs/                  # Ignored training runs, checkpoints, logs, and videos
```

- Keep application logic in the package and utility scripts thin. Prefer simple
  typed configuration over a configuration framework.
- Pin the Menagerie revision and retain the model's license and provenance.
  Keep downloaded assets reproducible; avoid checking in the whole Menagerie.
- Preserve the upstream model and default floor. If MJX compatibility requires
  adjustments, make them minimal, document them, and apply them consistently in
  training and inference. Do not silently change robot dynamics.
- Keep generated artifacts and caches out of Git. Document how to locate or
  obtain the trained checkpoint needed by the demo.

## Environment and policy

- Implement pure, JIT-compatible reset and step functions with explicit JAX PRNG
  keys and fixed-shape state. Batch simulation on GPU with JAX transformations;
  keep rendering, filesystem access, and Python callbacks out of the rollout.
- Define commands as body-frame forward/lateral velocity in m/s and yaw rate in
  rad/s: `[vx, vy, yaw_rate]`. Document axes, signs, command bounds, and update
  intervals. Include standing, forward/backward motion, lateral motion, and turns
  in training commands.
- Start with proprioceptive observations: projected gravity, local base linear
  and angular velocity, joint positions relative to the nominal pose, joint
  velocities, the velocity command, and the previous action. Document ordering,
  scaling, and quaternion/frame conventions.
- Use a small MLP actor and critic with Brax's standard PPO implementation.
  Prefer existing normalization, clipping, entropy, and checkpoint utilities.
- Start with bounded joint-position targets around a nominal standing pose,
  using the model's position actuators. Resolve joints and actuators by name;
  document which joints are controlled and how remaining joints are held.
  Respect actuator limits and any model joint limits.
- Configure physics timestep, control timestep, and action repeat explicitly.
  Apply action repeat exactly once and keep control timing identical at inference.
- End episodes on falls, invalid simulation state, or time limits. Distinguish
  termination from time-limit truncation for PPO bootstrapping.

## Rewards

Use a weighted sum of standard locomotion terms. Put weights and tracking widths
in configuration, document units and signs, and log each term separately.

- Reward tracking commanded planar linear velocity, for example
  `exp(-sum((v_xy - command_xy)**2) / sigma_linear**2)`.
- Reward tracking commanded yaw rate with an equivalent exponential error term.
- Encourage upright posture by aligning normalized gravity expressed in the
  body frame with `[0, 0, -1]`, with a consistent frame convention.
- Penalize unwanted vertical velocity and roll/pitch angular velocity.
- Use modest regularization for actuator effort, action changes, and joint-limit
  violations. Add nominal-posture or body-height terms only where helpful.
- Add foot slip, clearance, or air-time shaping only if evaluation shows the
  baseline needs it. Gate gait incentives appropriately for a zero command.

Keep velocity tracking central. Check that the policy cannot earn high returns
by falling, sliding, or standing still under a nonzero command. Document whether
reward terms are scaled by the control timestep and avoid double scaling.

## Training and inference deliverables

- Provide `uv run` entry points for training, evaluation, and interactive play.
  Document the actual implemented commands in `README.md` with setup and asset
  retrieval steps; do not present planned commands as working features.
- Save the seed, resolved configuration, dependency/model versions, device,
  reward metrics, episode lengths, tracking errors, and throughput with each run.
  Make batch size and environment count configurable for available GPU memory.
- Save reloadable policy parameters, observation normalization, network settings,
  joint/action mappings, and control timing. Preserve training state when
  supporting resume; a policy-only export is not a full training checkpoint.
- Interactive play must load the saved policy and use the same observation,
  normalization, action transformation, and timing as training. Use deterministic
  policy inference by default. A native MuJoCo viewer may render the simulation;
  verify parity if stepping native MuJoCo instead of MJX.
- Provide keyboard controls for forward/backward and lateral velocity, left/right
  yaw, zero command, reset, and exit. Suggested bindings are W/S, A/D, Q/E,
  Space, R, and Escape; handle viewer shortcut conflicts and document final keys.
  Clamp commands to the trained range and display the current command.
- Demonstrate a saved policy standing at zero command, walking under nonzero
  commands, turning, and responding to command changes on the default terrain.
  Report observed behavior and limitations; a successful PPO run alone does not
  establish a working walking policy.

## Validation and commits

- Do not write unnecessary tests or add a broad test framework by default.
  Prefer focused smoke checks: model load, finite reset/step outputs, a short GPU
  PPO update, checkpoint reload, and an inference rollout.
- Add a small targeted test only for a meaningful regression risk, such as frame
  conversions or training/inference observation parity. Avoid tests that simply
  repeat implementation details. Documentation-only changes need no RL tests.
- Inspect a rollout and command-tracking metrics before claiming walking works.
  Report checks actually run and any GPU or display limitations honestly.
- Create commits for meaningful, coherent changes with descriptive messages.
  Review the diff, run checks appropriate to the change, and stage only relevant
  files. Do not commit generated runs, caches, or unrelated user changes.
- Keep the README aligned with supported setup, training, checkpoint, and
  keyboard-control workflows as implementation progresses.
- Update `CONFIG.md` in the same change whenever settings change, including
  training hyperparameters, environment/control settings, network or normalization
  settings, reward terms/weights/widths, and observation contents/order/scales.
  Include hard-coded settings and relevant dependency defaults. Keep current
  run/policy overrides distinct from code defaults; refresh the current values
  when adopting a different run configuration or selected policy. Keep this file
  a settings reference; put experiment rationale and results in `EXPERIMENTS.md`.
