"""Repeatable command suite, tracking metrics, trajectory, and optional video."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from op3_locomotion.checkpoints import load_export, validate_mapping
from op3_locomotion.env import OP3Env
from op3_locomotion.observations import base_motion
from op3_locomotion.policy import PolicySimulation


def command_suite(bounds):
    vx, vy, yaw = np.asarray(bounds) * 0.6
    return {
        "stand": [(0, 0, 0)],
        "forward": [(vx, 0, 0)],
        "backward": [(-vx, 0, 0)],
        "left": [(0, vy, 0)],
        "right": [(0, -vy, 0)],
        "turn_left": [(0, 0, yaw)],
        "turn_right": [(0, 0, -yaw)],
        "changes": [(0, 0, 0), (vx, 0, 0), (vx, 0, yaw), (0, vy, 0), (0, 0, 0)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--physics", choices=("native", "mjx"), default="native")
    parser.add_argument("--seconds", type=float, default=5.0, help="Seconds per command segment")
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Subset of stand/forward/backward/left/right/turn_left/turn_right/changes",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--nominal", action="store_true", help="Compare fixed standing targets without policy"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.physics == "native":
        jax.config.update("jax_platforms", "cpu")
    policy, config, mapping, path = load_export(args.checkpoint)
    env = OP3Env(config.env, args.assets, build_mjx=args.physics == "mjx")
    validate_mapping(mapping, env.mapping())
    simulation = PolicySimulation(env, policy, args.physics, args.seed, args.nominal)
    output = args.output or path / (
        "nominal-evaluation" if args.nominal else f"evaluation-{args.physics}"
    )
    output.mkdir(parents=True, exist_ok=True)
    suite = command_suite(config.env.command_max)
    if args.cases:
        if set(args.cases) - suite.keys():
            parser.error("Unknown command case")
        suite = {name: suite[name] for name in args.cases}
    steps = round(args.seconds / env.dt)
    if steps < 1:
        parser.error("--seconds must be at least one control step")
    motion = jax.jit(base_motion)
    renderer = mujoco.Renderer(env.model, height=480, width=640) if args.video else None
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.0, 135, -15
    results = {}
    started = time.monotonic()
    try:
        for case, commands in suite.items():
            simulation.reset()
            trace = {
                key: []
                for key in ("command", "velocity", "qpos", "qvel", "action", "feet_z", "contacts")
            }
            writer = None
            if renderer:
                import imageio.v2 as imageio

                writer = imageio.get_writer(output / f"{case}.mp4", fps=round(1 / env.dt / 2))
            try:
                for index in range(steps * len(commands)):
                    command = np.asarray(commands[index // steps])
                    data = simulation.step(command)
                    _, linear, angular = motion(jp.asarray(data.qpos), jp.asarray(data.qvel))
                    velocity = np.array([linear[0], linear[1], angular[2]])
                    trace["command"].append(command)
                    trace["velocity"].append(velocity)
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["action"].append(np.asarray(simulation.previous_action))
                    trace["feet_z"].append(data.xpos[env.foot_bodies, 2].copy())
                    trace["contacts"].append(data.ncon)
                    if renderer and index % 2 == 0:
                        camera.lookat[:] = data.qpos[:3] + np.array([0, 0, 0.08])
                        renderer.update_scene(data, camera)
                        writer.append_data(renderer.render())
                    if simulation.done:
                        break
            finally:
                if writer:
                    writer.close()
            trace = {key: np.asarray(value) for key, value in trace.items()}
            np.savez_compressed(output / f"{case}.npz", **trace)
            error = trace["velocity"] - trace["command"]
            segments = []
            for index, command in enumerate(commands):
                values = trace["velocity"][index * steps : (index + 1) * steps]
                if len(values):
                    # Include transients in overall metrics; report settled portion per segment too.
                    settled = values[min(round(1 / env.dt), len(values) // 2) :]
                    segments.append(
                        {
                            "command": list(command),
                            "seconds": len(values) * env.dt,
                            "mean_velocity": values.mean(axis=0).tolist(),
                            "settled_velocity": settled.mean(axis=0).tolist(),
                        }
                    )
            results[case] = {
                "survived": not simulation.done,
                "seconds": len(error) * env.dt,
                "linear_rmse": float(np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1)))),
                "yaw_rmse": float(np.sqrt(np.mean(error[:, 2] ** 2))),
                "mean_velocity": trace["velocity"].mean(axis=0).tolist(),
                "minimum_height": float(trace["qpos"][:, 2].min()),
                "displacement_xy": (trace["qpos"][-1, :2] - np.asarray(env.nominal[:2])).tolist(),
                "foot_height_range": np.ptp(trace["feet_z"], axis=0).tolist(),
                "maximum_contacts": int(trace["contacts"].max()),
                "segments": segments,
            }
            print(case, json.dumps(results[case]), flush=True)
    finally:
        if renderer:
            renderer.close()
    report = {
        "checkpoint": str(path),
        "physics": args.physics,
        "nominal": args.nominal,
        "seed": args.seed,
        "wall_seconds": time.monotonic() - started,
        "cases": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report and trajectories: {output}")


if __name__ == "__main__":
    main()
