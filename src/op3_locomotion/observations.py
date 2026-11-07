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


def foot_heights(data, foot_geom_indices, foot_geom_sizes, floor_height):
    """Lowest original sole-box point above the flat floor, left then right, m."""
    bottom = data.geom_xpos[foot_geom_indices, 2] - jp.sum(
        jp.abs(data.geom_xmat[foot_geom_indices, 2, :]) * foot_geom_sizes, axis=-1
    )
    return jp.min(bottom, axis=-1) - floor_height


def contact_air_time(previous, supported, dt):
    """Seconds without support, independent of command and reward eligibility."""
    return jp.where(supported, 0.0, previous + dt)


def critic_observation(actor_obs, height, air_time, contact):
    """64 actor values + left/right height (m), air time (s), and support (0/1)."""
    return {
        "state": actor_obs,
        "privileged_state": jp.concatenate(
            (actor_obs, height * 10.0, air_time * 5.0, contact.astype(jp.float32))
        ),
    }
