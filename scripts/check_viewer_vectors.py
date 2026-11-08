"""Check gravity-arrow frame/sign conventions against native OP3 kinematics."""

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from op3_locomotion.env import OP3Env
from op3_locomotion.observations import observe
from op3_locomotion.play import gravity_directions


def main():
    jax.config.update("jax_platforms", "cpu")
    env = OP3Env(build_mjx=False)
    data = mujoco.MjData(env.model)
    root_body = int(env.model.jnt_bodyid[0])
    observation = jax.jit(
        lambda q, v: observe(
            q,
            v,
            env.nominal,
            env.qpos_indices,
            env.qvel_indices,
            jp.zeros(3),
            jp.zeros(env.action_size),
        )
    )
    for roll, pitch, yaw in ((0, 0, 0), (25, 0, 0), (0, -30, 0), (20, -25, 110)):
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        for axis, angle in zip(np.eye(3)[::-1], (yaw, pitch, roll), strict=True):
            turn = np.empty(4)
            mujoco.mju_axisAngle2Quat(turn, axis, np.radians(angle))
            combined = np.empty(4)
            mujoco.mju_mulQuat(combined, quaternion, turn)
            quaternion = combined
        data.qpos[:] = env.nominal
        data.qpos[3:7] = quaternion
        mujoco.mj_forward(env.model, data)
        rotation = data.xmat[root_body].reshape(3, 3)
        gravity_body = np.asarray(observation(data.qpos, data.qvel)[:3])
        np.testing.assert_allclose(gravity_body, rotation.T @ [0, 0, -1], atol=1e-7)
        body_up, policy_up, true_gravity = gravity_directions(
            quaternion, gravity_body, env.model.opt.gravity
        )
        np.testing.assert_allclose(body_up, rotation[:, 2], atol=1e-7)
        np.testing.assert_allclose(policy_up, [0, 0, 1], atol=1e-7)
        np.testing.assert_allclose(true_gravity, [0, 0, -1], atol=1e-7)
        if roll or pitch:
            assert np.linalg.norm(body_up - policy_up) > 0.1
        # Detect a mistaken gravity source without changing simulation physics.
        independent_gravity = np.array([0.0, -3.0, -4.0])
        altered = gravity_directions(quaternion, gravity_body, independent_gravity)
        np.testing.assert_allclose(altered[:2], [body_up, policy_up])
        np.testing.assert_allclose(altered[2], independent_gravity / 5)
    print("Passed: actor gravity, roll/pitch/yaw frames, body tilt and independent true gravity")


if __name__ == "__main__":
    main()
