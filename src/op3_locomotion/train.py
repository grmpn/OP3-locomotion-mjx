"""GPU-only standard Brax PPO entry point."""

import argparse
import json
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np
from brax.training.agents.ppo import train as ppo

from op3_locomotion.checkpoints import save_export, validate_warm_start, versions
from op3_locomotion.config import Config
from op3_locomotion.env import OP3Env, wrap_for_training
from op3_locomotion.policy import network_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--asymmetric-critic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--command-sampling", choices=("uniform", "balanced"))
    parser.add_argument("--foot-slip-weight", type=float)
    parser.add_argument("--foot-air-time-weight", type=float)
    parser.add_argument(
        "--standing-posture-weight", type=float, help="Additional zero-command penalty"
    )
    parser.add_argument("--lift-reset-probability", type=float)
    parser.add_argument("--collision-maxhullvert", type=int)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Warm-start weights/normalization; optimizer, RNG and step count restart",
    )
    for name in (
        "num_timesteps",
        "num_envs",
        "batch_size",
        "num_minibatches",
        "seed",
        "num_evals",
        "num_eval_envs",
        "unroll_length",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    args = parser.parse_args()
    config = Config.load(args.config) if args.config else Config()
    updates = {
        name: getattr(args, name)
        for name in asdict(config.ppo)
        if hasattr(args, name) and getattr(args, name) is not None
    }
    config = replace(config, ppo=replace(config.ppo, **updates))
    if args.asymmetric_critic is not None:
        config = replace(config, env=replace(config.env, asymmetric_critic=args.asymmetric_critic))
    if args.command_sampling:
        config = replace(config, env=replace(config.env, command_sampling=args.command_sampling))
    if args.foot_slip_weight is not None:
        config = replace(
            config,
            env=replace(
                config.env, reward=replace(config.env.reward, foot_slip=args.foot_slip_weight)
            ),
        )
    if args.foot_air_time_weight is not None:
        config = replace(
            config,
            env=replace(
                config.env,
                reward=replace(config.env.reward, foot_air_time=args.foot_air_time_weight),
            ),
        )
    if args.collision_maxhullvert is not None:
        config = replace(
            config, env=replace(config.env, collision_maxhullvert=args.collision_maxhullvert)
        )
    if args.standing_posture_weight is not None:
        config = replace(
            config,
            env=replace(
                config.env,
                reward=replace(config.env.reward, standing_posture=args.standing_posture_weight),
            ),
        )
    if args.lift_reset_probability is not None:
        config = replace(
            config, env=replace(config.env, lift_reset_probability=args.lift_reset_probability)
        )
    try:
        devices = jax.devices("gpu")
    except RuntimeError as error:
        raise SystemExit(
            "GPU training required: JAX cannot initialize CUDA. Check driver and uv sync."
        ) from error
    if not devices or jax.default_backend() != "gpu":
        raise SystemExit(
            "GPU training required: remove JAX_PLATFORMS=cpu and check CUDA installation."
        )
    print(f"Training device: {devices[0]} ({devices[0].device_kind})", flush=True)
    output = args.output or Path("outputs") / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config.save(output / "config.json")
    run_versions = versions()
    (output / "run.json").write_text(
        json.dumps(
            {
                "seed": config.ppo.seed,
                "device": str(devices[0]),
                "device_kind": devices[0].device_kind,
                "platform_version": devices[0].client.platform_version,
                "versions": run_versions,
                "initialize_from": str(args.initialize_from),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    env = OP3Env(
        config.env, args.assets, batch_capacity=max(config.ppo.num_envs, config.ppo.num_eval_envs)
    )
    mapping = env.mapping()
    (output / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    params = (
        validate_warm_start(args.initialize_from, config, mapping) if args.initialize_from else None
    )
    start = time.monotonic()

    def progress(step, metrics):
        elapsed = time.monotonic() - start
        record = {
            "steps": int(step),
            "elapsed_seconds": elapsed,
            "overall_steps_per_second": int(step) / max(elapsed, 1e-6),
            **{key: float(np.asarray(value)) for key, value in metrics.items()},
        }
        with (output / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")
        summary = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in record.items()
            if k
            in (
                "steps",
                "elapsed_seconds",
                "training/sps",
                "eval/episode_reward",
                "eval/avg_episode_length",
                "eval/episode_linear_error",
                "eval/episode_yaw_error",
            )
        }
        print(json.dumps(summary), flush=True)
        if any("physics_overflow" in key and value > 0 for key, value in record.items()):
            raise RuntimeError(
                "MJX contact/constraint buffer overflow; increase capacities before training"
            )

    def save(step, make_policy, parameters):
        path = save_export(
            output / "checkpoints", int(step), parameters, config, mapping, run_versions
        )
        print(f"Saved policy: {path}", flush=True)

    settings = asdict(config.ppo)
    settings.pop("hidden_sizes")
    ppo.train(
        environment=env,
        **settings,
        episode_length=config.env.episode_steps,
        action_repeat=1,
        wrap_env_fn=wrap_for_training,
        randomization_fn=None,
        normalize_observations=True,
        reward_scaling=1.0,
        deterministic_eval=True,
        network_factory=network_factory(config.ppo, config.env.asymmetric_critic),
        progress_fn=progress,
        policy_params_fn=save,
        restore_params=params,
        max_devices_per_host=1,
        log_training_metrics=True,
        training_metrics_steps=max(config.ppo.num_envs * 100, 10_000),
    )
    (output / "finished.json").write_text(json.dumps({"elapsed_seconds": time.monotonic() - start}))
    print(f"Finished. Evaluate with: uv run op3-evaluate {output}", flush=True)


if __name__ == "__main__":
    main()
