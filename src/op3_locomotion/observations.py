"""Shared observations: MuJoCo wxyz quaternion rotates body vectors to world."""

import jax
import jax.numpy as jp


def world_to_body(quat: jax.Array, vector: jax.Array) -> jax.Array:
    """Rotate a world vector with the conjugate of a unit wxyz quaternion."""
    axis = -quat[1:]
    twice_cross = 2 * jp.cross(axis, vector)
    return vector + quat[0] * twice_cross + jp.cross(axis, twice_cross)


def base_motion(qpos: jax.Array, qvel: jax.Array):
    # MuJoCo freejoint qvel: WORLD translation, BODY angular velocity.
    gravity = world_to_body(qpos[3:7], jp.array([0.0, 0.0, -1.0]))
    linear = world_to_body(qpos[3:7], qvel[:3])
    angular = qvel[3:6]
    return gravity, linear, angular


def observe(qpos, qvel, nominal, qpos_indices, qvel_indices, command, previous_action):
    gravity, linear, angular = base_motion(qpos, qvel)
    return jp.concatenate(
        (
            gravity,
            linear * 2.0,
            angular * 0.25,
            qpos[qpos_indices] - nominal[qpos_indices],
            qvel[qvel_indices] * 0.05,
            command * jp.array([2.0, 2.0, 0.25]),
            previous_action,
        )
    )
