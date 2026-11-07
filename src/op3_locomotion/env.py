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
from op3_locomotion.observations import (
    base_motion,
    contact_air_time,
    critic_observation,
    foot_heights,
    observe,
)
from op3_locomotion.policy import position_targets
from op3_locomotion.rewards import foot_slip, reward_terms, update_air_time, weighted_terms


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
    spec = mujoco.MjSpec.from_file(str(assets / "scene.xml"))
    if config.collision_maxhullvert != -1:
        # Only collision assets; visual meshes and the original foot boxes stay intact.
        collision_meshes = {
            geom.meshname
            for geom in spec.geoms
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH and (geom.contype or geom.conaffinity)
        }
        for mesh in spec.meshes:
            if mesh.name in collision_meshes:
                mesh.maxhullvert = config.collision_maxhullvert
    if config.reward.foot_slip or config.reward.foot_air_time or config.asymmetric_critic:
        for side in ("l", "r"):
            body = spec.body(f"{side}_ank_roll_link")
            boxes = [geom for geom in body.geoms if geom.type == mujoco.mjtGeom.mjGEOM_BOX]
            if len(boxes) != 2:
                raise ValueError("Expected two original foot boxes per foot")
            for index, geom in enumerate(boxes):
                geom.name = f"{side}_sole_{index}"
                spec.add_sensor(
                    name=f"{side}_slip_{index}",
                    type=mujoco.mjtSensor.mjSENS_CONTACT,
                    objtype=mujoco.mjtObj.mjOBJ_GEOM,
                    objname="floor",
                    reftype=mujoco.mjtObj.mjOBJ_GEOM,
                    refname=geom.name,
                    # force + linvel, maxforce reduction, four box-plane contacts.
                    intprm=[(1 << 1) | (1 << 7), 2, 4],
                )
    model = spec.compile()
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
        self.slip_sensor_indices = None
        if config.reward.foot_slip or config.reward.foot_air_time or config.asymmetric_critic:
            self.slip_sensor_indices = jp.array(
                [
                    [
                        int(m.sensor(f"{side}_slip_{index}").adr[0]) + offset
                        for index in range(2)
                        for offset in range(24)
                    ]
                    for side in ("l", "r")
                ]
            )
        feet = [
            i
            for i in range(m.ngeom)
            if m.geom_bodyid[i] in self.foot_bodies and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
        ]
        self.foot_geom_indices = jp.array(
            [[i for i in feet if m.geom_bodyid[i] == body] for body in self.foot_bodies]
        )
        self.foot_geom_sizes = jp.array(m.geom_size)[self.foot_geom_indices]
        self.floor_height = float(d.geom_xpos[m.geom("floor").id, 2])
        floor_height = min(
            d.geom_xpos[i, 2] - np.abs(d.geom_xmat[i].reshape(3, 3)[2]) @ m.geom_size[i]
            for i in feet
        )
        d.qpos[2] += 0.0005 - floor_height
        d.ctrl[:] = d.qpos[np.asarray(self.qpos_indices)]
        mujoco.mj_forward(m, d)
        self.nominal = jp.array(d.qpos)
        self.nominal_ctrl = jp.array(d.ctrl)
        reset_data = [d]
        for side, direction in (("l", -1), ("r", 1)):
            lifted = mujoco.MjData(m)
            lifted.qpos[:] = d.qpos
            offsets = {
                f"{leg}_{joint}": direction * config.lift_reset_hip_roll
                for leg in ("l", "r")
                for joint in ("hip_roll", "ank_roll")
            }
            offsets.update(
                {
                    f"{side}_hip_pitch": direction * config.lift_reset_knee_flexion / 2,
                    f"{side}_knee": -direction * config.lift_reset_knee_flexion,
                    f"{side}_ank_pitch": -direction * config.lift_reset_knee_flexion / 2,
                }
            )
            for name, offset in offsets.items():
                lifted.qpos[m.joint(name).qposadr[0]] += offset
            mujoco.mj_forward(m, lifted)
            lowest = min(
                lifted.geom_xpos[i, 2]
                - np.abs(lifted.geom_xmat[i].reshape(3, 3)[2]) @ m.geom_size[i]
                for i in feet
            )
            lifted.qpos[2] += 0.0005 - lowest
            lifted.ctrl[:] = lifted.qpos[np.asarray(self.qpos_indices)]
            mujoco.mj_forward(m, lifted)
            reset_data.append(lifted)
        self.reset_qpos = jp.array(np.stack([value.qpos for value in reset_data]))
        self.reset_ctrl = jp.array(np.stack([value.ctrl for value in reset_data]))
        self.mjx_model = None
        self.initial_data = None
        self.reset_data = None
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
            self.reset_data = tuple(
                mjx.put_data(m, value, impl=config.physics_backend, **buffers)
                for value in reset_data
            )
            self.initial_data = self.reset_data[0]

    @property
    def observation_size(self):
        size = 12 + 2 * len(self.joint_names) + self.action_size
        return (
            {"state": size, "privileged_state": size + 6} if self.config.asymmetric_critic else size
        )

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

    def observation(self, data, command, previous_action, critic_air_time=None):
        actor_obs = observe(
            data.qpos,
            data.qvel,
            self.nominal,
            self.qpos_indices,
            self.qvel_indices,
            command,
            previous_action,
        )
        if not self.config.asymmetric_critic:
            return actor_obs
        if critic_air_time is None:
            raise ValueError("Asymmetric observations require explicit contact-time history")
        return critic_observation(
            actor_obs,
            foot_heights(data, self.foot_geom_indices, self.foot_geom_sizes, self.floor_height),
            critic_air_time,
            self.foot_contact(data),
        )

    def foot_contact(self, data):
        contacts = data.sensordata[self.slip_sensor_indices].reshape(2, 8, 6)
        load = jp.sum(jp.maximum(contacts[..., 0], 0.0), axis=-1)
        return load > self.config.reward.air_time_contact_force

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
            info=info,
            obs=self.observation(
                state.pipeline_state, command, info["previous_action"], info["critic_air_time"]
            ),
        )

    def sample_reset_index(self, rng):
        # A separate derived stream preserves the existing command/PPO RNG sequence.
        choose, side = jax.random.split(jax.random.fold_in(rng, 0x4C494654))
        return jp.where(
            jax.random.uniform(choose) < self.config.lift_reset_probability,
            jax.random.randint(side, (), 1, 3),
            0,
        )

    def reset(self, rng, reset_index=None):
        if reset_index is None:
            reset_index = self.sample_reset_index(rng) if self.config.lift_reset_probability else 0
        rng, command_key = jax.random.split(rng)
        command = self.sample_command(command_key)
        action = jp.zeros(self.action_size)
        # Data.where preserves Warp's shared contact buffers while selecting per-world state.
        data = self.initial_data
        for index, pose in enumerate(self.reset_data[1:], 1):
            data = data.where(reset_index == index, pose)
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
            "feet_air_time": jp.zeros(2),
            "feet_air_armed": jp.zeros(2, dtype=jp.bool_),
            "critic_air_time": jp.zeros(2),
            "time_out": jp.array(0.0),
        }
        return State(
            data,
            self.observation(data, command, action, info["critic_air_time"]),
            jp.array(0.0),
            jp.array(0.0),
            metrics,
            info,
        )

    def step(self, state, action):
        action = jp.clip(action, -1.0, 1.0)
        data = state.pipeline_state.replace(ctrl=self.targets(action))
        moving = jp.max(jp.abs(state.info["command"]) / jp.array(self.config.command_max))
        moving = moving > self.config.reward.air_time_command_threshold

        def substep(_, carry):
            data, air_time, armed, bonus, critic_time = carry
            data = mjx.step(self.mjx_model, data)
            if self.config.reward.foot_air_time or self.config.asymmetric_critic:
                supported = self.foot_contact(data)
            if self.config.reward.foot_air_time:
                air_time, armed, touchdown_bonus = update_air_time(
                    air_time, armed, supported, moving, self.config.physics_dt
                )
                bonus += touchdown_bonus
            if self.config.asymmetric_critic:
                critic_time = contact_air_time(critic_time, supported, self.config.physics_dt)
            return data, air_time, armed, bonus, critic_time

        data, air_time, armed, air_time_bonus, critic_time = jax.lax.fori_loop(
            0,
            self.config.action_repeat,
            substep,
            (
                data,
                state.info["feet_air_time"],
                state.info["feet_air_armed"],
                jp.array(0.0),
                state.info["critic_air_time"],
            ),
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
            slip=(
                foot_slip(
                    data.sensordata[self.slip_sensor_indices].reshape(2, 8, 6),
                    self.config.reward.slip_contact_force,
                )
                if self.config.reward.foot_slip
                else 0.0
            ),
            air_time=air_time_bonus,
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
            "feet_air_time": air_time,
            "feet_air_armed": armed,
            "critic_air_time": critic_time,
            "time_out": jp.array(0.0),
        }
        obs = jax.tree.map(jp.nan_to_num, self.observation(data, command, action, critic_time))
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
            "collision_maxhullvert": self.config.collision_maxhullvert,
            "physics_backend": self.config.physics_backend,
        }


class AutoReset(Wrapper):
    """Reset command, previous action, and physics together; preserve episode metrics."""

    def reset(self, rng):
        state = self.env.reset(rng)
        # EpisodeWrapper deliberately skips the per-step "reward" metric when
        # aggregating; its actual return is "sum_reward". Drop the unused zero.
        state.info["episode_metrics"].pop("reward", None)
        return state

    def step(self, state, action):
        info = {**state.info, "steps": jp.where(state.done, 0, state.info["steps"])}
        state = self.env.step(state.replace(info=info, done=jp.zeros_like(state.done)), action)
        fresh = self.env.reset(state.info["rng"])

        def select(fresh_value, value):
            mask = state.done.reshape(state.done.shape + (1,) * (value.ndim - state.done.ndim))
            return jp.where(mask, fresh_value, value)

        info = dict(state.info)
        for name in (
            "rng",
            "command",
            "command_age",
            "previous_action",
            "feet_air_time",
            "feet_air_armed",
            "critic_air_time",
        ):
            info[name] = select(fresh.info[name], info[name])
        info["time_out"] = info["truncation"]
        return state.replace(
            pipeline_state=jax.vmap(lambda d, done, r: d.where(done, r))(
                state.pipeline_state, state.done, fresh.pipeline_state
            ),
            obs=jax.tree.map(select, fresh.obs, state.obs),
            info=info,
        )


def wrap_for_training(env, episode_length, action_repeat=1, randomization_fn=None):
    if action_repeat != 1 or randomization_fn is not None:
        raise ValueError("Physics repeat is internal; domain randomization is unsupported")
    return AutoReset(EpisodeWrapper(VmapWrapper(env), episode_length, action_repeat=1))
