"""Direct MJX dynamics behind the small Brax Env/State training interface."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper
from mujoco import mjx

from op3_locomotion.config import LEG_JOINTS, REVISION, EnvConfig
from op3_locomotion.observations import base_motion, observe
from op3_locomotion.policy import position_targets
from op3_locomotion.rewards import reward_terms, weighted_terms


def default_assets() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "robotis_op3"


def load_model(assets: Path, config: EnvConfig):
    provenance_path = assets / "provenance.json"
    if not provenance_path.exists():
        raise FileNotFoundError("Missing OP3 assets. Run: uv run scripts/fetch_assets.py")
    provenance = json.loads(provenance_path.read_text())
    if provenance["revision"] != REVISION:
        raise ValueError("OP3 model revision does not match this implementation")
    for relative, expected in provenance["sha256"].items():
        if hashlib.sha256((assets / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Modified asset: {relative}. Re-run scripts/fetch_assets.py")
    model = mujoco.MjModel.from_xml_path(str(assets / "scene.xml"))
    if config.collision_profile == "locomotion":
        # Authorized simplification: retain inertias/visuals, disable only head/arm contact.
        upper_bodies = {
            model.body(name).id
            for name in (
                "head_pan_link",
                "head_tilt_link",
                "l_sho_pitch_link",
                "l_sho_roll_link",
                "l_el_link",
                "r_sho_pitch_link",
                "r_sho_roll_link",
                "r_el_link",
            )
        }
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] in upper_bodies:
                model.geom_contype[geom] = 0
                model.geom_conaffinity[geom] = 0
        # MuJoCo compiles body broadphase masks; update those alongside geom masks.
        for body in upper_bodies:
            model.body_contype[body] = 0
            model.body_conaffinity[body] = 0
    model.opt.timestep = config.physics_dt
    return model


class OP3Env(Env):
    def __init__(
        self,
        config: EnvConfig = EnvConfig(),
        assets: Path | None = None,
        build_mjx: bool = True,
        batch_capacity: int = 1,
    ):
        self.config = config
        self.model = load_model(assets or default_assets(), config)
        m = self.model
        self.joint_names = tuple(m.joint(int(j)).name for j in m.actuator_trnid[:, 0])
        self.actuator_names = tuple(m.actuator(i).name for i in range(m.nu))
        self.actuator_indices = jp.array([m.actuator(f"{name}_act").id for name in LEG_JOINTS])
        joint_ids = np.array([m.joint(name).id for name in self.joint_names])
        self.qpos_indices = jp.array(m.jnt_qposadr[joint_ids])
        self.qvel_indices = jp.array(m.jnt_dofadr[joint_ids])
        controlled = np.array([m.joint(name).id for name in LEG_JOINTS])
        self.controlled_qpos = jp.array(m.jnt_qposadr[controlled])
        self.lower = jp.array(
            np.maximum(
                np.where(m.actuator_ctrllimited, m.actuator_ctrlrange[:, 0], -np.inf),
                np.where(m.jnt_limited[joint_ids], m.jnt_range[joint_ids, 0], -np.inf),
            )
        )
        self.upper = jp.array(
            np.minimum(
                np.where(m.actuator_ctrllimited, m.actuator_ctrlrange[:, 1], np.inf),
                np.where(m.jnt_limited[joint_ids], m.jnt_range[joint_ids, 1], np.inf),
            )
        )
        d = mujoco.MjData(m)
        pose = {
            "l_sho_roll": 1.25,
            "r_sho_roll": -1.25,
            "l_hip_pitch": -0.3,
            "l_knee": 0.6,
            "l_ank_pitch": 0.3,
            "r_hip_pitch": 0.3,
            "r_knee": -0.6,
            "r_ank_pitch": -0.3,
        }
        for name, angle in pose.items():
            d.qpos[m.joint(name).qposadr[0]] = angle
        mujoco.mj_forward(m, d)
        self.foot_bodies = [m.body(f"{side}_ank_roll_link").id for side in ("l", "r")]
        feet = [
            i
            for i in range(m.ngeom)
            if m.geom_bodyid[i] in self.foot_bodies and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
        ]
        floor_height = min(
            d.geom_xpos[i, 2] - np.abs(d.geom_xmat[i].reshape(3, 3)[2]) @ m.geom_size[i]
            for i in feet
        )
        d.qpos[2] += 0.0005 - floor_height
        d.ctrl[:] = d.qpos[np.asarray(self.qpos_indices)]
        mujoco.mj_forward(m, d)
        self.nominal = jp.array(d.qpos)
        self.nominal_ctrl = jp.array(d.ctrl)
        self.mjx_model = None
        self.initial_data = None
        if build_mjx:
            kwargs = {}
            buffers = {}
            if config.physics_backend == "warp":
                from mujoco.mjx.warp.types import GraphMode

                kwargs["graph_mode"] = GraphMode.WARP_STAGED
                buffers = dict(
                    naconmax=64 * batch_capacity, naccdmax=64 * batch_capacity, njmax=256
                )
            self.mjx_model = mjx.put_model(m, impl=config.physics_backend, **kwargs)
            self.initial_data = mjx.put_data(m, d, impl=config.physics_backend, **buffers)

    @property
    def observation_size(self):
        return 12 + 2 * len(self.joint_names) + self.action_size

    @property
    def action_size(self):
        return len(LEG_JOINTS)

    @property
    def backend(self):
        return "mjx"

    @property
    def dt(self):
        return self.config.control_dt

    def sample_command(self, key):
        velocity_key, standing_key = jax.random.split(key)
        if self.config.command_sampling == "balanced":
            velocity_key, mode_key = jax.random.split(velocity_key)
        command = jax.random.uniform(velocity_key, (3,), minval=-1.0, maxval=1.0)
        command *= jp.array(self.config.command_max)
        if self.config.command_sampling == "balanced":
            # Equal non-standing coverage of forward/backward, lateral, yaw, and combined motion.
            masks = jp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            command *= masks[jax.random.randint(mode_key, (), 0, 4)]
        return jp.where(
            jax.random.uniform(standing_key) < self.config.standing_probability,
            jp.zeros(3),
            command,
        )

    def observation(self, data, command, previous_action):
        return observe(
            data.qpos,
            data.qvel,
            self.nominal,
            self.qpos_indices,
            self.qvel_indices,
            command,
            previous_action,
        )

    def targets(self, action):
        return position_targets(
            action,
            self.nominal_ctrl,
            self.actuator_indices,
            self.lower,
            self.upper,
            self.config.action_scale,
        )

    def set_command(self, state, command):
        command = jp.clip(
            jp.asarray(command),
            -jp.array(self.config.command_max),
            jp.array(self.config.command_max),
        )
        info = {**state.info, "command": command, "command_age": jp.array(0, dtype=jp.int32)}
        return state.replace(
            info=info, obs=self.observation(state.pipeline_state, command, info["previous_action"])
        )

    def reset(self, rng):
        rng, command_key = jax.random.split(rng)
        command = self.sample_command(command_key)
        action = jp.zeros(self.action_size)
        data = self.initial_data
        terms = reward_terms(
            data,
            command,
            action,
            action,
            self.nominal,
            self.controlled_qpos,
            self.lower[self.actuator_indices],
            self.upper[self.actuator_indices],
            jp.array(False),
            self.config.reward,
        )
        metrics = {f"reward/{name}": jp.array(0.0) for name in terms}
        metrics.update(
            {
                name: jp.array(0.0)
                for name in (
                    "linear_error",
                    "yaw_error",
                    "height",
                    "terminated",
                    "physics_overflow",
                    "reward",
                )
            }
        )
        info = {
            "rng": rng,
            "command": command,
            "command_age": jp.array(0, dtype=jp.int32),
            "previous_action": action,
            "time_out": jp.array(0.0),
        }
        return State(
            data,
            self.observation(data, command, action),
            jp.array(0.0),
            jp.array(0.0),
            metrics,
            info,
        )

    def step(self, state, action):
        action = jp.clip(action, -1.0, 1.0)
        data = state.pipeline_state.replace(ctrl=self.targets(action))
        data = jax.lax.fori_loop(
            0, self.config.action_repeat, lambda _, d: mjx.step(self.mjx_model, d), data
        )
        gravity, linear, angular = base_motion(data.qpos, data.qvel)
        finite = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
        finite &= jp.all(jp.isfinite(data.actuator_force))
        # The pinned MJX Warp backend exposes buffer exhaustion per world here.
        overflow = (
            jp.any(data._impl.overflow != 0)
            if self.config.physics_backend == "warp"
            else jp.array(False)
        )
        finite &= ~overflow
        terminated = (
            ~finite
            | (data.qpos[2] < self.config.min_height)
            | (-gravity[2] < self.config.min_up_cosine)
        )
        terms = reward_terms(
            data,
            state.info["command"],
            action,
            state.info["previous_action"],
            self.nominal,
            self.controlled_qpos,
            self.lower[self.actuator_indices],
            self.upper[self.actuator_indices],
            terminated,
            self.config.reward,
        )
        weighted = weighted_terms(terms, self.config.reward)
        # No positive reward on a fall/invalid transition. Metrics reflect the paid terms.
        weighted = {
            name: jp.where(
                terminated, self.config.reward.termination if name == "termination" else 0.0, value
            )
            for name, value in weighted.items()
        }
        reward = sum(weighted.values()) * self.dt
        metrics = {f"reward/{name}": value * self.dt for name, value in weighted.items()}
        metrics.update(
            linear_error=jp.nan_to_num(
                jp.linalg.norm(linear[:2] - state.info["command"][:2]), nan=10.0, posinf=10.0
            ),
            yaw_error=jp.nan_to_num(
                jp.abs(angular[2] - state.info["command"][2]), nan=10.0, posinf=10.0
            ),
            height=jp.nan_to_num(data.qpos[2]),
            terminated=terminated.astype(jp.float32),
            physics_overflow=overflow.astype(jp.float32),
            reward=reward,
        )
        rng, command_key = jax.random.split(state.info["rng"])
        age = state.info["command_age"] + 1
        command = jp.where(
            age >= self.config.command_steps,
            self.sample_command(command_key),
            state.info["command"],
        )
        age = jp.where(age >= self.config.command_steps, 0, age)
        info = {
            **state.info,
            "rng": rng,
            "command": command,
            "command_age": age,
            "previous_action": action,
            "time_out": jp.array(0.0),
        }
        obs = jp.nan_to_num(self.observation(data, command, action))
        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=terminated.astype(jp.float32),
            metrics=metrics,
            info=info,
        )

    def mapping(self):
        return {
            "joint_names": self.joint_names,
            "actuator_names": self.actuator_names,
            "controlled_joints": LEG_JOINTS,
            "actuator_indices": self.actuator_indices.tolist(),
            "qpos_indices": self.qpos_indices.tolist(),
            "qvel_indices": self.qvel_indices.tolist(),
            "nominal_qpos": self.nominal.tolist(),
            "nominal_ctrl": self.nominal_ctrl.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "physics_dt": self.config.physics_dt,
            "control_dt": self.dt,
            "action_repeat": self.config.action_repeat,
            "action_scale": self.config.action_scale,
            "observation_size": self.observation_size,
            "model_revision": REVISION,
            "collision_profile": self.config.collision_profile,
            "physics_backend": self.config.physics_backend,
        }


class AutoReset(Wrapper):
    """Reset command, previous action, and physics together; preserve episode metrics."""

    def reset(self, rng):
        return self.env.reset(rng)

    def step(self, state, action):
        info = {**state.info, "steps": jp.where(state.done, 0, state.info["steps"])}
        state = self.env.step(state.replace(info=info, done=jp.zeros_like(state.done)), action)
        fresh = self.env.reset(state.info["rng"])

        def select(fresh_value, value):
            mask = state.done.reshape(state.done.shape + (1,) * (value.ndim - state.done.ndim))
            return jp.where(mask, fresh_value, value)

        info = dict(state.info)
        for name in ("rng", "command", "command_age", "previous_action"):
            info[name] = select(fresh.info[name], info[name])
        info["time_out"] = info["truncation"]
        return state.replace(
            pipeline_state=jax.vmap(lambda d, done, r: d.where(done, r))(
                state.pipeline_state, state.done, fresh.pipeline_state
            ),
            obs=select(fresh.obs, state.obs),
            info=info,
        )


def wrap_for_training(env, episode_length, action_repeat=1, randomization_fn=None):
    if action_repeat != 1 or randomization_fn is not None:
        raise ValueError("Physics repeat is internal; domain randomization is unsupported")
    return AutoReset(EpisodeWrapper(VmapWrapper(env), episode_length, action_repeat=1))
