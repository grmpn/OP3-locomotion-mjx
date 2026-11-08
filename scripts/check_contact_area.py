"""Targeted sole-area geometry, contact gating, legacy config and GPU smoke checks."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from op3_locomotion.config import Config
from op3_locomotion.env import OP3Env, wrap_for_training
from op3_locomotion.rewards import foot_contact_area


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if not args.gpu:
        jax.config.update("jax_platforms", "cpu")
    config = Config.load(Path("configs/upright_contact_area.json"))
    env = OP3Env(config.env, build_mjx=args.gpu, batch_capacity=2)
    assert Config.load(Path("configs/asymmetric_critic.json")).env.reward.foot_contact_area == 0
    np.testing.assert_allclose(env.sole_areas.sum(axis=1), [0.00962, 0.00962], atol=1e-9)
    assert env.sole_points.shape == (2, 80, 3)
    reward = jax.jit(
        lambda h, f, up: foot_contact_area(
            h,
            env.sole_areas,
            f,
            up,
            config.env.reward.area_contact_force,
            config.env.reward.area_height_sigma,
        )
    )
    flat = jp.zeros((2, 80))
    contacts = jp.zeros((2, 8, 6)).at[:, 0, 0].set(10)
    up = jp.ones(2)
    np.testing.assert_allclose(reward(flat, contacts, up), 1, atol=2e-7)
    np.testing.assert_allclose(reward(flat, contacts.at[1].set(0), up), 1, atol=2e-7)
    np.testing.assert_allclose(reward(flat, contacts * 10, up), 1, atol=2e-7)
    assert float(reward(flat, jp.zeros_like(contacts), up)) == 0
    assert float(reward(flat, contacts.at[:, 0, 0].set(1), up)) == 0
    assert float(reward(flat, contacts, -up)) == 0
    np.testing.assert_allclose(reward(flat + 0.002, contacts, up), np.exp(-1), rtol=1e-6)
    np.testing.assert_allclose(reward(flat - 0.002, contacts, up), np.exp(-1), rtol=1e-6)

    # Edge contact progressively loses near-floor area as either pitch or roll grows.
    scores = {}
    for axis in (0, 1):
        values = []
        coordinate = env.sole_points[:, :, axis]
        for angle in (0, 5, 15, 30):
            heights = (coordinate - coordinate.min(axis=1, keepdims=True)) * np.sin(
                np.radians(angle)
            )
            values.append(float(reward(heights, contacts, up)))
        assert all(a > b for a, b in zip(values[:-1], values[1:], strict=True)), values
        scores[("pitch", "roll")[axis]] = values

    # Actual loaded OP3 soles: the reward is high while standing and zero in flight.
    data = mujoco.MjData(env.model)
    data.qpos[:] = env.nominal
    data.ctrl[:] = env.nominal_ctrl
    mujoco.mj_forward(env.model, data)
    for _ in range(100):
        mujoco.mj_step(env.model, data)
    mujoco.mj_forward(env.model, data)

    def area_native():
        geometry = SimpleNamespace(
            xpos=jp.array(data.xpos),
            xmat=jp.array(data.xmat.reshape(-1, 3, 3)),
            sensordata=jp.array(data.sensordata),
        )
        return float(env.contact_area_reward(geometry))

    standing_area = area_native()
    assert 0.8 < standing_area <= 1, standing_area
    data.qpos[2] += 0.1
    mujoco.mj_forward(env.model, data)
    assert area_native() == 0

    # Enabling only this reward must install contact sensors without changing dynamics.
    area_only = OP3Env(
        replace(
            config.env,
            asymmetric_critic=False,
            reward=replace(config.env.reward, foot_slip=0, foot_air_time=0),
        ),
        build_mjx=False,
    )
    assert area_only.model.nsensordata == 96
    previous = OP3Env(Config.load(Path("configs/asymmetric_critic.json")).env, build_mjx=False)
    for field in (
        "body_mass",
        "body_inertia",
        "dof_damping",
        "geom_size",
        "geom_pos",
        "geom_friction",
        "actuator_gainprm",
        "actuator_forcerange",
    ):
        np.testing.assert_array_equal(getattr(previous.model, field), getattr(env.model, field))
    assert previous.mapping() == env.mapping()

    if args.gpu:
        assert jax.default_backend() == "gpu"
        wrapped = wrap_for_training(env, episode_length=50)
        state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
        step = jax.jit(wrapped.step)
        paid_area = 0.0
        for _ in range(10):
            state = step(state, jp.zeros((2, 12)))
            assert np.isfinite(state.reward).all()
            assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(state.obs))
            assert not np.asarray(state.metrics["physics_overflow"]).any()
            paid_area += float(jp.sum(state.metrics["reward/foot_contact_area"]))
            measured = jax.jit(jax.vmap(env.contact_area_reward))(state.pipeline_state)
            np.testing.assert_allclose(
                state.metrics["reward/foot_contact_area"],
                measured * config.env.reward.foot_contact_area * env.dt,
                atol=1e-7,
            )
        assert paid_area > 0
    print(
        json.dumps(
            {
                "device": str(jax.devices()[0]),
                "standing_area_fraction": standing_area,
                "edge_contact_scores_0_5_15_30_deg": scores,
                "gpu_smoke": args.gpu,
            }
        )
    )


if __name__ == "__main__":
    main()
