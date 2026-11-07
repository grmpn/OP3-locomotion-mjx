"""Focused model/frame/reset/timeout and native-MJX parity smoke checks."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from op3_locomotion.env import OP3Env, wrap_for_training
from op3_locomotion.observations import observe, world_to_body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Also test GPU MJX physics and wrappers")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.gpu:
        jax.config.update("jax_platforms", "cpu")
    started = time.monotonic()
    quat = np.array([np.cos(0.7), 0, 0, np.sin(0.7)])
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    vector = np.array([0.2, -0.3, 0.4])
    np.testing.assert_allclose(
        world_to_body(jp.array(quat), jp.array(vector)), matrix.reshape(3, 3).T @ vector, atol=1e-7
    )
    env = OP3Env(build_mjx=args.gpu, batch_capacity=2)
    m = env.model
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(env.nominal)
    d.ctrl[:] = np.asarray(env.nominal_ctrl)
    mujoco.mj_forward(m, d)
    assert d.qpos[2] > env.config.min_height
    assert all(c.dist > -0.001 for c in d.contact)
    assert np.isfinite(d.qacc).all()
    for _ in range(1000):
        mujoco.mj_step(m, d)
    assert d.qpos[2] > env.config.min_height, "Nominal pose cannot stand for two seconds"
    assert abs(d.qpos[0]) < 0.1, "Nominal pose drifts excessively"
    report = {
        "device": str(jax.devices()[0]),
        "native_standing_height": float(d.qpos[2]),
        "initial_height": float(env.nominal[2]),
        "frame_rotation": "passed",
    }
    if args.gpu:
        assert jax.devices()[0].platform == "gpu"
        reset, step = jax.jit(env.reset), jax.jit(env.step)
        state = reset(jax.random.PRNGKey(1))
        other = reset(jax.random.PRNGKey(2))
        np.testing.assert_array_equal(state.pipeline_state.qpos, other.pipeline_state.qpos)
        np.testing.assert_array_equal(state.pipeline_state.qvel, other.pipeline_state.qvel)
        command, previous = state.info["command"], jp.zeros(env.action_size)
        shared_obs = observe(
            state.pipeline_state.qpos,
            state.pipeline_state.qvel,
            env.nominal,
            env.qpos_indices,
            env.qvel_indices,
            command,
            previous,
        )
        np.testing.assert_array_equal(state.obs, shared_obs)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = np.asarray(env.nominal)
        d.ctrl[:] = np.asarray(env.nominal_ctrl)
        mujoco.mj_forward(m, d)
        max_qpos_error = 0.0
        max_qvel_error = 0.0
        for index in range(50):
            action = jp.sin(jp.arange(env.action_size) + index * 0.05) * 0.05
            state = step(state, action)
            d.ctrl[:] = np.asarray(env.targets(action))
            for _ in range(env.config.action_repeat):
                mujoco.mj_step(m, d)
            assert np.isfinite(state.obs).all() and np.isfinite(state.reward)
            assert not bool(state.done)
            max_qpos_error = max(
                max_qpos_error, float(np.max(np.abs(d.qpos - state.pipeline_state.qpos)))
            )
            max_qvel_error = max(
                max_qvel_error, float(np.max(np.abs(d.qvel - state.pipeline_state.qvel)))
            )
        report.update(
            native_mjx_max_qpos_error=max_qpos_error, native_mjx_max_qvel_error=max_qvel_error
        )
        assert max_qpos_error < 0.01, report
        assert max_qvel_error < 0.15, report
        wrapped = wrap_for_training(env, episode_length=2)
        batch = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(10), 2))
        wrapped_step = jax.jit(wrapped.step)
        batch = wrapped_step(batch, jp.zeros((2, env.action_size)))
        assert "reward" not in batch.info["episode_metrics"]
        np.testing.assert_allclose(batch.info["episode_metrics"]["sum_reward"], batch.reward)
        assert not np.asarray(batch.done).any()
        batch = wrapped_step(batch, jp.zeros((2, env.action_size)))
        np.testing.assert_array_equal(batch.done, np.ones(2))
        np.testing.assert_array_equal(batch.info["truncation"], np.ones(2))
        np.testing.assert_array_equal(batch.info["previous_action"], np.zeros((2, env.action_size)))
        expected_obs = jax.vmap(env.observation)(
            batch.pipeline_state, batch.info["command"], batch.info["previous_action"]
        )
        np.testing.assert_array_equal(batch.obs, expected_obs)
        # A physical fall must not be labelled a time limit, even at the horizon.
        fallen_qpos = batch.pipeline_state.qpos.at[:, 2].set(1.0)
        fallen_qpos = fallen_qpos.at[:, 3:7].set(jp.array([0.0, 1.0, 0.0, 0.0]))
        fallen = batch.pipeline_state.replace(qpos=fallen_qpos)
        batch = batch.replace(
            pipeline_state=fallen, done=jp.zeros(2), info={**batch.info, "steps": jp.ones(2)}
        )
        batch = wrapped_step(batch, jp.zeros((2, env.action_size)))
        np.testing.assert_array_equal(batch.done, np.ones(2))
        np.testing.assert_array_equal(batch.info["truncation"], np.zeros(2))
        report["reset_and_timeout"] = "passed"
    report["wall_seconds"] = time.monotonic() - started
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
