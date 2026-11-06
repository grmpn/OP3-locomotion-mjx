"""Unscaled named terms; multiply weighted sum by control_dt exactly once."""

from dataclasses import asdict

import jax.numpy as jp

from op3_locomotion.observations import base_motion


def reward_terms(
    data, command, action, previous_action, nominal, qpos_indices, lower, upper, terminated, config
):
    gravity, linear, angular = base_motion(data.qpos, data.qvel)
    positions = data.qpos[qpos_indices]
    return {
        "linear_tracking": jp.exp(
            -jp.sum(jp.square(linear[:2] - command[:2])) / config.sigma_linear**2
        ),
        "yaw_tracking": jp.exp(-jp.square(angular[2] - command[2]) / config.sigma_yaw**2),
        "upright": jp.clip(-gravity[2], 0.0, 1.0),
        "vertical_velocity": jp.square(linear[2]),
        "roll_pitch_velocity": jp.sum(jp.square(angular[:2])),
        "effort": jp.sum(jp.square(data.actuator_force)),
        "action_rate": jp.sum(jp.square(action - previous_action)),
        "joint_limits": jp.sum(
            jp.maximum(lower - positions, 0.0) + jp.maximum(positions - upper, 0.0)
        ),
        "posture": jp.sum(jp.square(positions - nominal[qpos_indices])),
        "termination": terminated.astype(jp.float32),
    }


def weighted_terms(terms, config):
    weights = asdict(config)
    return {
        name: jp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) * weights[name]
        for name, value in terms.items()
    }
