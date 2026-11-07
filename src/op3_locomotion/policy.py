"""One network factory and target transform for training and inference."""

from functools import partial

import jax.numpy as jp
from brax.training.agents.ppo import networks


def network_factory(config, asymmetric_critic=False):
    return partial(
        networks.make_ppo_networks,
        policy_hidden_layer_sizes=config.hidden_sizes,
        value_hidden_layer_sizes=config.hidden_sizes,
        policy_obs_key="state",
        value_obs_key="privileged_state" if asymmetric_critic else "state",
    )


def position_targets(action, nominal_ctrl, actuator_indices, lower, upper, scale):
    action = jp.clip(action, -1.0, 1.0)
    targets = nominal_ctrl.at[actuator_indices].add(scale * action)
    return jp.clip(targets, lower, upper)


class PolicySimulation:
    """Deterministic inference with identical observations, targets, and control timing."""

    def __init__(self, env, policy, physics="native", seed=0, nominal=False, reset_pose="standing"):
        import jax
        import mujoco

        self.env, self.policy, self.physics = env, policy, physics
        self.key = jax.random.PRNGKey(seed)
        self.nominal = nominal
        self.reset_index = ("standing", "left", "right").index(reset_pose)
        self.data = mujoco.MjData(env.model)
        from op3_locomotion.observations import observe

        self._observation = jax.jit(
            lambda q, v, c, a: observe(q, v, env.nominal, env.qpos_indices, env.qvel_indices, c, a)
        )
        self._targets = jax.jit(env.targets)
        self._reset = (
            jax.jit(lambda key: env.reset(key, self.reset_index)) if physics == "mjx" else None
        )
        self._step = jax.jit(env.step) if physics == "mjx" else None
        self.reset()

    def reset(self):
        import mujoco
        import numpy as np

        self.previous_action = jp.zeros(self.env.action_size)
        self.done = False
        if self.physics == "mjx":
            self.state = self._reset(self.key)
        mujoco.mj_resetData(self.env.model, self.data)
        self.data.qpos[:] = np.asarray(self.env.reset_qpos[self.reset_index])
        self.data.ctrl[:] = np.asarray(self.env.reset_ctrl[self.reset_index])
        mujoco.mj_forward(self.env.model, self.data)

    def step(self, command):
        import mujoco
        import numpy as np

        from op3_locomotion.observations import base_motion

        command = jp.clip(
            jp.asarray(command),
            -jp.array(self.env.config.command_max),
            jp.array(self.env.config.command_max),
        )
        if self.physics == "mjx":
            self.state = self.env.set_command(self.state, command)
            obs = self.state.obs
        else:
            obs = self._observation(self.data.qpos, self.data.qvel, command, self.previous_action)
        action = jp.zeros(self.env.action_size) if self.nominal else self.policy(obs, self.key)[0]
        if self.physics == "mjx":
            self.state = self._step(self.state, action)
            data = self.state.pipeline_state
            self.data.qpos[:] = np.asarray(data.qpos)
            self.data.qvel[:] = np.asarray(data.qvel)
            self.data.ctrl[:] = np.asarray(data.ctrl)
            mujoco.mj_forward(self.env.model, self.data)
            self.done = bool(self.state.done)
        else:
            self.data.ctrl[:] = np.asarray(self._targets(action))
            for _ in range(self.env.config.action_repeat):
                mujoco.mj_step(self.env.model, self.data)
            # Recompute kinematics at the integrated configuration for rendering/diagnostics.
            mujoco.mj_forward(self.env.model, self.data)
            gravity, _, _ = base_motion(jp.asarray(self.data.qpos), jp.asarray(self.data.qvel))
            self.done = bool(
                not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qvel).all()
                or self.data.qpos[2] < self.env.config.min_height
                or -float(gravity[2]) < self.env.config.min_up_cosine
            )
        self.previous_action = action
        return self.data
