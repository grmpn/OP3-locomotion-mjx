"""Focused standing-reward and seeded raised-foot reset regressions."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from op3_locomotion.config import EnvConfig
from op3_locomotion.env import OP3Env, wrap_for_training
from op3_locomotion.evaluate import foot_diagnostics
from op3_locomotion.policy import PolicySimulation
from op3_locomotion.rewards import reward_terms, weighted_terms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.gpu:
        jax.config.update("jax_platforms", "cpu")
    config = replace(EnvConfig(), lift_reset_probability=0.25)
    config = replace(config, reward=replace(config.reward, standing_posture=-0.45))
    env = OP3Env(config, build_mjx=args.gpu, batch_capacity=8)
    m = env.model
    diagnostics = []
    for index in range(3):
        d = mujoco.MjData(m)
        d.qpos[:] = env.reset_qpos[index]
        d.ctrl[:] = env.reset_ctrl[index]
        mujoco.mj_forward(m, d)
        clearance, _, _ = foot_diagnostics(m, d, env.foot_bodies)
        assert np.isfinite(d.qacc).all() and d.qpos[2] > config.min_height
        assert min((c.dist for c in d.contact), default=0) >= -1e-6
        np.testing.assert_array_equal(d.qvel, 0)
        assert np.max(np.abs(d.ctrl - env.nominal_ctrl)) <= config.action_scale
        assert np.all(d.ctrl >= env.lower) and np.all(d.ctrl <= env.upper)
        if index:
            assert 0.018 < clearance[index - 1] < 0.022
            np.testing.assert_allclose(clearance[2 - index], 0.0005, atol=1e-7)
        diagnostics.append(clearance.tolist())
        for _ in range(500):
            mujoco.mj_step(m, d)
        assert d.qpos[2] > config.min_height, "Reset pose falls while its targets are held"

    # Changing only reset selection must not alter nominal actions or sampled commands.
    keys = jax.random.split(jax.random.PRNGKey(8), 2048)
    choices = jax.jit(jax.vmap(env.sample_reset_index))(keys)
    np.testing.assert_array_equal(choices, jax.jit(jax.vmap(env.sample_reset_index))(keys))
    counts = np.bincount(choices, minlength=3)
    assert 0.70 < counts[0] / len(keys) < 0.80 and min(counts[1:]) > 180

    # Gate on the requested velocity, independently of the policy's actions.
    data = type("RewardData", (), {})()
    data.qpos = env.reset_qpos[1]
    data.qvel = jp.zeros(m.nv)
    data.actuator_force = jp.zeros(m.nu)

    def terms(command, action):
        return reward_terms(
            data,
            command,
            action,
            jp.zeros(env.action_size),
            env.nominal,
            env.controlled_qpos,
            env.lower[env.actuator_indices],
            env.upper[env.actuator_indices],
            jp.array(False),
            config.reward,
        )

    standing = terms(jp.zeros(3), jp.ones(env.action_size) * 0.3)
    paid = weighted_terms(standing, config.reward)
    np.testing.assert_allclose(
        paid["posture"] + paid["standing_posture"], -0.5 * standing["posture"]
    )
    for command in ([0.1, 0, 0], [0, -0.1, 0], [0, 0, 0.1], [1e-6, 0, 0]):
        assert float(terms(jp.array(command), jp.zeros(env.action_size))["standing_posture"]) == 0

    report = {
        "device": str(jax.devices()[0]),
        "clearance": diagnostics,
        "sample_counts": counts.tolist(),
        "standing_gate": "passed",
    }
    if args.gpu:
        assert jax.devices()[0].platform == "gpu"
        reset = jax.jit(env.reset)
        for index in range(3):
            state = reset(jax.random.PRNGKey(0), index)
            np.testing.assert_array_equal(state.pipeline_state.qpos, env.reset_qpos[index])
            np.testing.assert_array_equal(state.pipeline_state.qvel, 0)
            np.testing.assert_array_equal(state.info["feet_air_time"], 0)
            np.testing.assert_array_equal(state.info["feet_air_armed"], False)
            expected = env.sample_command(jax.random.split(jax.random.PRNGKey(0))[1])
            np.testing.assert_array_equal(state.info["command"], expected)

        # Fixed-pose native/MJX parity from both lifted starts, including their first landing.
        errors = []

        def policy(obs, key):
            return jp.zeros(env.action_size), {}

        for pose in ("left", "right"):
            native = PolicySimulation(env, policy, reset_pose=pose)
            mjx = PolicySimulation(env, policy, physics="mjx", reset_pose=pose)
            q_error, v_error = 0.0, 0.0
            for _ in range(25):
                a, b = native.step(jp.zeros(3)), mjx.step(jp.zeros(3))
                q_error = max(q_error, float(np.max(np.abs(a.qpos - b.qpos))))
                v_error = max(v_error, float(np.max(np.abs(a.qvel - b.qvel))))
                assert not native.done and not mjx.done
                assert np.isfinite(mjx.state.obs).all()
                assert float(mjx.state.metrics["reward/foot_air_time"]) == 0
            assert q_error < 0.01 and v_error < 0.15
            errors.append([q_error, v_error])

        wrapped = wrap_for_training(env, episode_length=2)
        state = jax.jit(wrapped.reset)(keys[:8])
        step = jax.jit(wrapped.step)
        state = step(state, jp.zeros((8, env.action_size)))
        state = step(state, jp.zeros((8, env.action_size)))
        np.testing.assert_array_equal(state.done, 1)
        np.testing.assert_array_equal(state.info["truncation"], 1)
        np.testing.assert_array_equal(state.info["feet_air_time"], 0)
        for qpos in np.asarray(state.pipeline_state.qpos):
            assert any(np.array_equal(qpos, pose) for pose in np.asarray(env.reset_qpos))
        np.testing.assert_array_equal(
            state.obs,
            jax.vmap(env.observation)(
                state.pipeline_state, state.info["command"], state.info["previous_action"]
            ),
        )
        report.update(native_mjx_errors=errors, batched_resets="passed")
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
