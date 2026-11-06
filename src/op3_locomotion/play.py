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
from op3_locomotion.policy import PolicySimulation


class KeyboardCommand:
    """Incremental commands; custom GLFW callbacks avoid MuJoCo viewer shortcuts."""

    def __init__(self, bounds):
        self.bounds = np.asarray(bounds)
        self.value = np.zeros(3)
        self.reset_requested = False

    def on_key(self, window, key, scancode, action, modifiers):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        bindings = {
            glfw.KEY_W: (0, 0.05),
            glfw.KEY_S: (0, -0.05),
            glfw.KEY_A: (1, 0.05),
            glfw.KEY_D: (1, -0.05),
            glfw.KEY_Q: (2, 0.2),
            glfw.KEY_E: (2, -0.2),
        }
        if key in bindings:
            axis, amount = bindings[key]
            self.value[axis] += amount
            self.value[:] = np.clip(self.value, -self.bounds, self.bounds)
        elif key == glfw.KEY_SPACE:
            self.value[:] = 0
        elif key == glfw.KEY_R:
            self.value[:] = 0
            self.reset_requested = True
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--seconds", type=float, help="Optional viewer smoke-check duration")
    args = parser.parse_args()
    jax.config.update("jax_platforms", "cpu")
    policy, config, mapping, _ = load_export(args.checkpoint)
    env = OP3Env(config.env, args.assets, build_mjx=False)
    validate_mapping(mapping, env.mapping())
    simulation = PolicySimulation(env, policy)
    # Compile inference before opening the window.
    simulation.step(np.zeros(3))
    simulation.reset()
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
    scene = mujoco.MjvScene(env.model, maxgeom=2000)
    context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.0, 135, -15
    option, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    started = time.monotonic()
    print(
        "W/S forward/back | A/D lateral | Q/E yaw | Space stop | R reset | Escape exit", flush=True
    )
    try:
        while not glfw.window_should_close(window):
            tick = time.monotonic()
            if args.seconds is not None and tick - started >= args.seconds:
                break
            glfw.poll_events()
            if controls.reset_requested:
                simulation.reset()
                controls.reset_requested = False
            if not simulation.done:
                simulation.step(controls.value)
            command_text = (
                f"vx {controls.value[0]:+.2f} m/s   vy {controls.value[1]:+.2f} m/s   "
                f"yaw {controls.value[2]:+.2f} rad/s"
            )
            status = (
                "Fallen: press R to reset"
                if simulation.done
                else "W/S  A/D  Q/E | Space stop | R reset | Esc exit"
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
