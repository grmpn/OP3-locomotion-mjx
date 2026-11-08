"""MuJoCo renderer with explicit keyboard controls and deterministic policy inference."""

import argparse
import time
from pathlib import Path

import glfw
import jax
import mujoco
import numpy as np

from op3_locomotion.checkpoints import load_export, validate_mapping
from op3_locomotion.env import OP3Env
from op3_locomotion.observations import base_motion
from op3_locomotion.policy import PolicySimulation


def gravity_directions(quaternion, gravity_body, gravity_world):
    """World-space body +Z, up reconstructed from the raw observation, and gravity."""
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=np.float64))
    rotation = rotation.reshape(3, 3)
    directions = np.stack((rotation[:, 2], rotation @ -gravity_body, gravity_world))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def add_gravity_vectors(scene, qpos, gravity_body, gravity_world):
    """Add decorative arrows beside the torso without covering the feet with text."""
    directions = gravity_directions(qpos[3:7], gravity_body, gravity_world)
    # Offset the shared origin so the robot's meshes do not hide the arrows.
    origin = qpos[:3] + np.array([0.0, -0.22, 0.12])

    def connector(start, end, kind, width, color):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.zeros(3), start, np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, kind, width, start, end)
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1

    connector(qpos[:3], origin, mujoco.mjtGeom.mjGEOM_LINE, 1, [0.7, 0.7, 0.7, 1])
    for direction, length, width, color in zip(
        directions,
        (0.24, 0.20, 0.20),
        (0.006, 0.009, 0.009),
        ((0.2, 1.0, 0.2, 1), (0.1, 0.85, 1.0, 1), (1.0, 0.45, 0.1, 1)),
        strict=True,
    ):
        connector(origin, origin + length * direction, mujoco.mjtGeom.mjGEOM_ARROW, width, color)


class KeyboardCommand:
    """Hold to ramp, release to stop; independent of keyboard repeat and rendering."""

    bindings = {
        glfw.KEY_W: (0, 1),
        glfw.KEY_S: (0, -1),
        glfw.KEY_A: (1, 1),
        glfw.KEY_D: (1, -1),
        glfw.KEY_Q: (2, 1),
        glfw.KEY_E: (2, -1),
    }

    def __init__(self, bounds, ramp_seconds=0.5, return_seconds=0.35):
        self.bounds = np.asarray(bounds, dtype=float)
        self.ramp_rate = self.bounds / ramp_seconds
        self.return_rate = self.bounds / return_seconds
        self.value = np.zeros(3)
        self.held = set()
        self.reset_requested = False

    def clear(self):
        self.held.clear()
        self.value[:] = 0

    def advance(self, dt):
        direction = np.zeros(3)
        for key in self.held:
            axis, sign = self.bindings[key]
            direction[axis] += sign
        target = direction * self.bounds
        rate = np.where(direction == 0, self.return_rate, self.ramp_rate)
        self.value[:] = np.clip(
            self.value + np.clip(target - self.value, -rate * dt, rate * dt),
            -self.bounds,
            self.bounds,
        )
        return self.value

    def on_focus(self, window, focused):
        if not focused:
            self.clear()

    def on_key(self, window, key, scancode, action, modifiers):
        if action == glfw.RELEASE:
            self.held.discard(key)
            return
        if action != glfw.PRESS:
            return
        if key in self.bindings:
            self.held.add(key)
        elif key == glfw.KEY_SPACE:
            self.clear()
        elif key == glfw.KEY_R:
            self.clear()
            self.reset_requested = True
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--seconds", type=float, help="Optional viewer smoke-check duration")
    parser.add_argument("--reset-pose", choices=("standing", "left", "right"), default="standing")
    parser.add_argument(
        "--vectors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show body up, policy-derived up and true gravity (default: enabled)",
    )
    args = parser.parse_args()
    jax.config.update("jax_platforms", "cpu")
    policy, config, mapping, _ = load_export(args.checkpoint)
    env = OP3Env(config.env, args.assets, build_mjx=False)
    validate_mapping(mapping, env.mapping())
    simulation = PolicySimulation(env, policy, reset_pose=args.reset_pose)
    # Compile inference before opening the window.
    simulation.step(np.zeros(3))
    simulation.reset()
    # The first three actor inputs use this same function, before normalization.
    policy_gravity = jax.jit(lambda q, v: base_motion(q, v)[0])
    if args.vectors:
        policy_gravity(simulation.data.qpos, simulation.data.qvel).block_until_ready()
    if not glfw.init():
        raise SystemExit("GLFW could not open a display. Use op3-evaluate for headless inference.")
    window = glfw.create_window(960, 720, "OP3 locomotion", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("Could not create an OpenGL viewer window")
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    controls = KeyboardCommand(config.env.command_max)
    glfw.set_key_callback(window, controls.on_key)
    glfw.set_window_focus_callback(window, controls.on_focus)
    scene = mujoco.MjvScene(env.model, maxgeom=2000)
    # Shadow maps and floor reflections are costly on software OpenGL displays.
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.0, 135, -15
    option, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    started = previous_tick = time.monotonic()
    accumulator = env.dt
    print(
        "Hold W/S forward/back | A/D lateral | Q/E yaw; release to stop | "
        "Space stop | R reset | Escape exit",
        flush=True,
    )
    try:
        while not glfw.window_should_close(window):
            tick = time.monotonic()
            if args.seconds is not None and tick - started >= args.seconds:
                break
            # Keep fixed control updates independent of display frame rate.
            # Bound catch-up after a paused/slow window so input stays responsive.
            accumulator = min(accumulator + tick - previous_tick, 0.25)
            previous_tick = tick
            glfw.poll_events()
            if controls.reset_requested:
                simulation.reset()
                controls.reset_requested = False
                accumulator = env.dt
            while accumulator >= env.dt and not simulation.done:
                simulation.step(controls.advance(env.dt))
                accumulator -= env.dt
            if simulation.done:
                accumulator = 0.0
            command_text = (
                f"vx {controls.value[0]:+.2f} m/s   vy {controls.value[1]:+.2f} m/s   "
                f"yaw {controls.value[2]:+.2f} rad/s"
            )
            status = (
                "Fallen: press R to reset"
                if simulation.done
                else "Hold W/S A/D Q/E; release to stop | Space stop | R reset | Esc exit"
            )
            glfw.set_window_title(window, f"OP3 | {command_text}")
            camera.lookat[:] = simulation.data.qpos[:3] + np.array([0, 0, 0.08])
            mujoco.mjv_updateScene(
                env.model,
                simulation.data,
                option,
                perturb,
                camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )
            if args.vectors:
                gravity_body = np.asarray(
                    policy_gravity(simulation.data.qpos, simulation.data.qvel)
                )
                add_gravity_vectors(
                    scene, simulation.data.qpos, gravity_body, env.model.opt.gravity
                )
            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjr_render(viewport, scene, context)
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                command_text,
                status,
                context,
            )
            glfw.swap_buffers(window)
            remaining = env.dt - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
