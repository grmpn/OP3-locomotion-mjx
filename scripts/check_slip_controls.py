"""Focused regressions for contact slip, collision hulls, and held-key commands."""

from dataclasses import replace

import glfw
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from op3_locomotion.config import EnvConfig
from op3_locomotion.env import OP3Env
from op3_locomotion.play import KeyboardCommand
from op3_locomotion.rewards import foot_slip


def check_keyboard():
    bounds = np.array([0.25, 0.15, 0.8])
    controls = KeyboardCommand(bounds)

    def key(code, action):
        controls.on_key(None, code, 0, action, 0)

    for code, (axis, sign) in controls.bindings.items():
        controls.clear()
        key(code, glfw.PRESS)
        np.testing.assert_array_equal(controls.value, np.zeros(3))
        controls.advance(0.1)
        assert np.isclose(controls.value[axis], sign * bounds[axis] * 0.2)
        key(code, glfw.REPEAT)
        assert np.isclose(controls.value[axis], sign * bounds[axis] * 0.2)
        for _ in range(100):
            controls.advance(0.02)
        assert np.isclose(controls.value[axis], sign * bounds[axis])
        key(code, glfw.RELEASE)
        controls.advance(0.1)
        assert 0 < abs(controls.value[axis]) < bounds[axis]
        for _ in range(18):
            controls.advance(0.02)
        np.testing.assert_array_equal(controls.value, np.zeros(3))

    # Same elapsed control time, independent of call count or keyboard repeat.
    key(glfw.KEY_W, glfw.PRESS)
    for _ in range(10):
        controls.advance(0.02)
    expected = controls.value.copy()
    controls.clear()
    key(glfw.KEY_W, glfw.PRESS)
    controls.advance(0.2)
    np.testing.assert_allclose(controls.value, expected)
    key(glfw.KEY_S, glfw.PRESS)
    controls.advance(0.35)
    np.testing.assert_array_equal(controls.value, np.zeros(3))
    key(glfw.KEY_W, glfw.RELEASE)
    controls.advance(0.1)
    assert controls.value[0] < 0
    key(glfw.KEY_SPACE, glfw.PRESS)
    key(glfw.KEY_S, glfw.REPEAT)
    controls.advance(0.1)
    np.testing.assert_array_equal(controls.value, np.zeros(3))
    key(glfw.KEY_Q, glfw.PRESS)
    controls.advance(0.1)
    controls.on_focus(None, False)
    controls.advance(0.1)
    np.testing.assert_array_equal(controls.value, np.zeros(3))
    key(glfw.KEY_R, glfw.PRESS)
    assert controls.reset_requested and not controls.held


def main():
    jax.config.update("jax_platforms", "cpu")
    check_keyboard()
    env = OP3Env(build_mjx=False)
    m, d = env.model, mujoco.MjData(env.model)
    d.qpos[:] = np.asarray(env.nominal)
    d.ctrl[:] = np.asarray(env.nominal_ctrl)
    for _ in range(100):
        mujoco.mj_step(m, d)

    # An independent contact-Jacobian calculation catches frame/order mistakes,
    # including angular motion that a foot-origin speed proxy would mislabel.
    for linear, angular in ((0.2, 0.0), (0.0, 0.3), (0.1, -0.4)):
        d.qvel[:] = 0
        d.qvel[0], d.qvel[5] = linear, angular
        mujoco.mj_forward(m, d)
        load, weighted = np.zeros(2), np.zeros(2)
        for index, contact in enumerate(d.contact):
            if contact.geom1 != m.geom("floor").id:
                continue
            body = m.geom_bodyid[contact.geom2]
            if body not in env.foot_bodies:
                continue
            foot = env.foot_bodies.index(body)
            force, jac = np.zeros(6), np.zeros((3, m.nv))
            mujoco.mj_contactForce(m, d, index, force)
            mujoco.mj_jac(m, d, jac, None, contact.pos, body)
            velocity = contact.frame.reshape(3, 3) @ jac @ d.qvel
            load[foot] += force[0]
            weighted[foot] += force[0] * np.sum(velocity[1:] ** 2)
        assert (load > 1).all()
        contacts = d.sensordata[np.asarray(env.slip_sensor_indices)].reshape(2, 8, 6)
        np.testing.assert_allclose(
            foot_slip(jp.array(contacts), 1), np.sum(weighted / load), rtol=2e-6
        )
    # Airborne feet incur no slip cost, regardless of velocity.
    d.qpos[2] += 1
    mujoco.mj_forward(m, d)
    contacts = d.sensordata[np.asarray(env.slip_sensor_indices)].reshape(2, 8, 6)
    assert float(foot_slip(jp.array(contacts), 1)) == 0

    for limit in (16, 8):
        reduced = OP3Env(replace(EnvConfig(), collision_maxhullvert=limit), build_mjx=False)
        for field in (
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_iquat",
            "dof_damping",
            "dof_armature",
            "geom_friction",
            "actuator_gainprm",
            "actuator_forcerange",
        ):
            np.testing.assert_array_equal(getattr(reduced.model, field), getattr(m, field))
        boxes = (m.geom_type == mujoco.mjtGeom.mjGEOM_BOX) | (
            m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
        )
        for field in ("geom_size", "geom_pos", "geom_quat"):
            np.testing.assert_array_equal(
                getattr(reduced.model, field)[boxes], getattr(m, field)[boxes]
            )
        np.testing.assert_array_equal(reduced.nominal, env.nominal)
        collision_meshes = set(
            m.geom_dataid[(m.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & (m.geom_contype != 0)]
        )
        for mesh in collision_meshes:
            assert reduced.model.mesh_graph[reduced.model.mesh_graphadr[mesh]] <= limit
    print("Passed: held-key ramp/release/focus/reset, contact-slip frames, airborne gating, hulls")


if __name__ == "__main__":
    main()
