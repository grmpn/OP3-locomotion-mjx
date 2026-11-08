"""Focused rolling-window accounting, physics integration and reset regressions."""

import argparse
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from op3_locomotion.config import Config
from op3_locomotion.env import OP3Env, wrap_for_training
from op3_locomotion.rewards import update_torque_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if not args.gpu:
        jax.config.update("jax_platforms", "cpu")
    config = Config.load(Path("configs/torque_symmetry_scratch_30m.json"))
    assert config.env.torque_window_steps == 100
    old = Config.load(Path("configs/upright_contact_area_scratch_40m.json"))
    assert old.env.reward.torque_symmetry == 0
    for changes in (
        {"torque_symmetry": 0.1},
        {"torque_symmetry": float("nan")},
        {"torque_window_seconds": 0},
        {"torque_window_seconds": 2.001},
        {"torque_window_seconds": float("inf")},
    ):
        try:
            replace(config.env, reward=replace(config.env.reward, **changes))
        except ValueError:
            pass
        else:
            raise AssertionError(changes)

    update = jax.jit(update_torque_window)
    history, index = jp.zeros((100, 6)), jp.array(0, dtype=jp.int32)
    # Independent NumPy time-window oracle, including multiple buffer wraps.
    # Physics-rate signs flip: integrate torque squared, never squared mean torque.
    rng = np.random.default_rng(17)
    integrals = []
    for tick in range(305):
        torque = rng.uniform(-3, 3, (10, 2, 6))
        integral = np.square(torque).sum(axis=0) * config.env.physics_dt
        integrals.append(integral)
        history, index, penalty = update(history, index, jp.array(integral))
        window = np.sum(integrals[max(0, tick - 99) :], axis=0)
        expected = np.abs(window[0] - window[1]).sum()
        np.testing.assert_allclose(penalty, expected, atol=2e-6)

    # Alternating legs can balance over time; different joints cannot cancel.
    history, index = jp.zeros((100, 6)), jp.array(0, dtype=jp.int32)
    left = jp.zeros((2, 6)).at[0, 0].set(2)
    right = jp.zeros((2, 6)).at[1, 0].set(2)
    history, index, penalty = update(history, index, left)
    assert float(penalty) == 2
    history, index, penalty = update(history, index, right)
    assert float(penalty) == 0
    for _ in range(98):
        history, index, penalty = update(history, index, jp.zeros((2, 6)))
    assert float(penalty) == 0
    history, index, penalty = update(history, index, jp.zeros((2, 6)))
    assert float(penalty) == 2  # First impulse expired, second still inside window.
    history, index, penalty = update(history, index, jp.zeros((2, 6)))
    assert float(penalty) == 0
    cross_joint = left.at[1, 1].set(2)
    _, _, penalty = update(jp.zeros_like(history), index, cross_joint)
    assert float(penalty) == 4

    env = OP3Env(config.env, build_mjx=args.gpu, batch_capacity=2)
    previous = OP3Env(old.env, build_mjx=False)
    assert env.mapping() == previous.mapping()
    # Unit-gear hinge position actuators: actuator_force is joint torque in Nm.
    ids = np.asarray(env.actuator_indices)
    np.testing.assert_array_equal(env.model.actuator_gear[ids, 0], np.ones(12))
    np.testing.assert_array_equal(env.model.actuator_gear[ids, 1:], np.zeros((12, 5)))
    names = np.array(env.actuator_names)[ids].reshape(2, 6)
    assert all(left[2:] == right[2:] for left, right in zip(*names, strict=True))

    if args.gpu:
        assert jax.default_backend() == "gpu"
        reset = jax.jit(jax.vmap(env.reset))
        state = reset(jax.random.split(jax.random.PRNGKey(0), 2))
        state = jax.vmap(env.set_command)(state, jp.zeros((2, 3)))
        actions = jp.zeros((2, 12)).at[:, 3].set(0.1)
        step = jax.jit(jax.vmap(env.step))

        def reference(data, action):
            data = data.replace(ctrl=env.targets(action))

            def substep(data, _):
                data = mjx.step(env.mjx_model, data)
                return data, data.actuator_force[env.actuator_indices].reshape(2, 6)

            return jax.lax.scan(substep, data, None, length=config.env.action_repeat)

        reference = jax.jit(jax.vmap(reference))
        for tick in range(3):
            expected_data, forces = reference(state.pipeline_state, actions)
            integral = np.square(np.asarray(forces)).sum(axis=1) * config.env.physics_dt
            state = step(state, actions)
            assert np.isfinite(state.reward).all() and not np.asarray(state.done).any()
            assert not np.asarray(state.metrics["physics_overflow"]).any()
            np.testing.assert_allclose(state.pipeline_state.qpos, expected_data.qpos, atol=1e-6)
            np.testing.assert_allclose(
                state.info["torque_history"][:, tick], integral[:, 0] - integral[:, 1], atol=2e-6
            )
            raw = np.abs(np.asarray(state.info["torque_history"]).sum(axis=1)).sum(axis=1)
            np.testing.assert_allclose(
                state.metrics["reward/torque_symmetry"],
                raw * config.env.reward.torque_symmetry * env.dt,
                atol=1e-7,
            )
        assert (raw > 0).all()  # Accumulates even at zero command.
        changed = jax.vmap(env.set_command)(state, jp.ones((2, 3)) * 0.1)
        np.testing.assert_array_equal(changed.info["torque_history"], state.info["torque_history"])

        # A physical fall and a time-limit truncation both erase all torque history.
        wrapped = wrap_for_training(env, episode_length=2)
        batch = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(9), 2))
        wrapped_step = jax.jit(wrapped.step)
        batch = wrapped_step(batch, actions)
        batch = batch.replace(
            pipeline_state=batch.pipeline_state.replace(
                qpos=batch.pipeline_state.qpos.at[0, 2]
                .set(0.5)
                .at[0, 3:7]
                .set(jp.array([0, 1, 0, 0]))
            )
        )
        batch = wrapped_step(batch, actions)
        np.testing.assert_array_equal(batch.done, [1, 1])
        np.testing.assert_array_equal(batch.info["truncation"], [0, 1])
        np.testing.assert_array_equal(batch.info["torque_history"], np.zeros((2, 100, 6)))
        np.testing.assert_array_equal(batch.info["torque_history_index"], [0, 0])
    print(f"Passed torque symmetry checks; device={jax.devices()[0]}, gpu_physics={args.gpu}")


if __name__ == "__main__":
    main()
