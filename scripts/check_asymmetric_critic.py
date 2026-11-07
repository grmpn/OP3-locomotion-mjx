"""Focused critic warm-start, actor isolation, foot-state and reset regressions."""

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks

from op3_locomotion.checkpoints import load_export, save_export, validate_warm_start
from op3_locomotion.config import Config
from op3_locomotion.env import OP3Env, wrap_for_training
from op3_locomotion.observations import contact_air_time, foot_heights
from op3_locomotion.policy import PolicySimulation, network_factory


def equal_trees(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--source", type=Path, default=Path("outputs/baseline/checkpoints/000007024640")
    )
    args = parser.parse_args()
    if not args.gpu:
        jax.config.update("jax_platforms", "cpu")
    config = Config.load(Path("configs/asymmetric_critic.json"))
    env = OP3Env(config.env, build_mjx=args.gpu, batch_capacity=2)
    old = checkpoint.load(str(args.source.resolve()))
    params = validate_warm_start(args.source, config, env.mapping())
    equal_trees(old[1], params[1])
    for name in ("mean", "std", "summed_variance"):
        np.testing.assert_array_equal(getattr(old[0], name), getattr(params[0], name)["state"])
    assert int(old[0].count) == int(params[0].count)
    before = network_factory(config.ppo)(64, 12, running_statistics.normalize)
    after = network_factory(config.ppo, True)(
        env.observation_size, 12, running_statistics.normalize
    )
    obs = old[0].mean + old[0].std * jax.random.normal(jax.random.PRNGKey(8), (8, 64))
    privileged = jp.concatenate((obs, jp.ones((8, 6))), axis=-1)
    inputs = {"state": obs, "privileged_state": privileged}
    # Resizing a matrix can select different TF32 kernels on GPU. Check the
    # function-preserving conversion with float32 products in this assertion.
    with jax.default_matmul_precision("highest"):
        np.testing.assert_allclose(
            before.value_network.apply(old[0], old[2], obs),
            after.value_network.apply(params[0], params[2], inputs),
            atol=2e-5,
            rtol=2e-6,
        )
    make_policy = jax.jit(networks.make_inference_fn(after)(params, deterministic=True))
    original_policy = jax.jit(networks.make_inference_fn(before)(old, deterministic=True))
    expected = original_policy(obs, jax.random.key(0))[0]
    equal_trees(expected, make_policy(inputs, jax.random.key(0))[0])
    # Even NaN critic-only values cannot reach the actor.
    inputs["privileged_state"] = jp.full((8, 70), jp.nan)
    equal_trees(expected, make_policy(inputs, jax.random.key(0))[0])
    # Preserve the future actor normalization update, too.
    inputs["privileged_state"] = privileged
    next_old = jax.jit(running_statistics.update)(old[0], obs)
    next_new = jax.jit(running_statistics.update)(params[0], inputs)
    for name in ("mean", "std", "summed_variance"):
        np.testing.assert_array_equal(getattr(next_old, name), getattr(next_new, name)["state"])

    data = mujoco.MjData(env.model)
    for index in (0, 1, 2):
        data.qpos[:] = np.asarray(env.reset_qpos[index])
        mujoco.mj_forward(env.model, data)
        geometry = SimpleNamespace(
            geom_xpos=jp.array(data.geom_xpos),
            geom_xmat=jp.array(data.geom_xmat.reshape(-1, 3, 3)),
        )
        height = foot_heights(
            geometry, env.foot_geom_indices, env.foot_geom_sizes, env.floor_height
        )
        reference = [
            min(
                data.geom_xpos[g, 2]
                - np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ env.model.geom_size[g]
                for g in geoms
            )
            for geoms in np.asarray(env.foot_geom_indices)
        ]
        np.testing.assert_allclose(height, reference, atol=1e-7)
        if index:
            assert height[index - 1] > 0.015
    timer = jp.zeros(2)
    for _ in range(10):
        timer = jax.jit(contact_air_time)(timer, jp.array([False, True]), 0.002)
    np.testing.assert_allclose(timer, [0.02, 0], atol=1e-7)
    equal_trees(contact_air_time(timer, jp.array([True, True]), 0.002), jp.zeros(2))

    with tempfile.TemporaryDirectory() as directory:
        export = save_export(Path(directory), 0, params, config, env.mapping())
        policy, loaded, mapping, _ = load_export(export)
        assert loaded == config and mapping["observation_size"] == env.observation_size
        equal_trees(policy(obs, jax.random.key(0))[0], expected)
        equal_trees(policy(inputs, jax.random.key(0))[0], expected)
        equal_trees(validate_warm_start(export, config, mapping), params)
        sim = PolicySimulation(env, policy)
        for _ in range(10):
            sim.step([0, 0, 0])
        assert np.isfinite(sim.data.qpos).all() and not sim.done

    if args.gpu:
        assert jax.devices()[0].platform == "gpu"
        symmetric = OP3Env(replace(config.env, asymmetric_critic=False), batch_capacity=2)
        state = jax.jit(env.reset)(jax.random.key(0), 1)
        reference = jax.jit(symmetric.reset)(jax.random.key(0), 1)
        state = env.set_command(state, jp.zeros(3))
        reference = symmetric.set_command(reference, jp.zeros(3))
        step, reference_step = jax.jit(env.step), jax.jit(symmetric.step)
        saw_unsupported_time = False
        for _ in range(10):
            state = step(state, jp.zeros(12))
            reference = reference_step(reference, jp.zeros(12))
            equal_trees(state.obs["state"], reference.obs)
            np.testing.assert_allclose(state.reward, reference.reward, atol=1e-7, rtol=2e-6)
            for name in state.metrics:
                np.testing.assert_allclose(
                    state.metrics[name], reference.metrics[name], atol=1e-7, rtol=2e-6
                )
            assert all(np.isfinite(x).all() for x in jax.tree.leaves(state.obs))
            equal_trees(state.obs["privileged_state"][68:], env.foot_contact(state.pipeline_state))
            equal_trees(state.info["feet_air_time"], jp.zeros(2))
            saw_unsupported_time |= bool(jp.any(state.info["critic_air_time"] > 0))
        assert saw_unsupported_time, "Critic timers must accrue even under a zero command"
        changed = env.set_command(state, jp.array([0.1, 0, 0]))
        equal_trees(changed.info["critic_air_time"], state.info["critic_air_time"])
        wrapped = wrap_for_training(env, episode_length=2)
        # Brax's EpisodeWrapper expects legacy (..., 2) PRNG keys, as PPO uses.
        batch = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(9), 2))
        wrapped_step = jax.jit(wrapped.step)
        batch = wrapped_step(batch, jp.zeros((2, 12)))
        batch = batch.replace(info={**batch.info, "critic_air_time": jp.ones((2, 2))})
        batch = wrapped_step(batch, jp.zeros((2, 12)))
        equal_trees(batch.info["truncation"], jp.ones(2))
        equal_trees(batch.info["critic_air_time"], jp.zeros((2, 2)))
        equal_trees(batch.obs["privileged_state"][:, 66:68], jp.zeros((2, 2)))
        equal_trees(
            batch.obs,
            jax.vmap(env.observation)(
                batch.pipeline_state,
                batch.info["command"],
                batch.info["previous_action"],
                batch.info["critic_air_time"],
            ),
        )
    print(
        json.dumps(
            {
                "device": str(jax.devices()[0]),
                "actor_preserved": True,
                "critic_initial_value_preserved": True,
                "export_reload": True,
                "gpu_reward_parity_and_reset": args.gpu,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
