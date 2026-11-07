"""Unscaled named terms; multiply weighted sum by control_dt exactly once."""

from dataclasses import asdict

import jax.numpy as jp

from op3_locomotion.observations import base_motion


def update_air_time(air_time, armed, supported, moving, dt):
    """Pay completed unsupported time once on landing, with no duration target.

    Each foot must first have support under a motion command. Standing clears
    both histories, so initial landings and standing-to-motion changes cannot
    cash in uncommanded air time. Called at the physics rate, not the policy rate.
    """
    eligible = armed & moving
    bonus = jp.sum(jp.where(supported & eligible, air_time, 0.0))
    air_time = jp.where(eligible & ~supported, air_time + dt, 0.0)
    armed = moving & (armed | supported)
    return air_time, armed, bonus


def foot_slip(contacts, minimum_force):
    """Sum per-foot force-weighted squared contact-point tangential speed.

    contacts has shape (2 feet, 8 slots, 6): force xyz then velocity xyz,
    both in the contact frame (x normal, yz tangential). Empty slots are zero.
    """
    load = jp.maximum(contacts[..., 0], 0.0)
    total = jp.sum(load, axis=-1)
    speed_squared = jp.sum(jp.square(contacts[..., 4:6]), axis=-1)
    per_foot = jp.sum(load * speed_squared, axis=-1) / jp.maximum(total, 1e-6)
    return jp.sum(jp.where(total > minimum_force, per_foot, 0.0))


def reward_terms(
    data,
    command,
    action,
    previous_action,
    nominal,
    qpos_indices,
    lower,
    upper,
    terminated,
    config,
    slip=0.0,
    air_time=0.0,
):
    gravity, linear, angular = base_motion(data.qpos, data.qvel)
    positions = data.qpos[qpos_indices]
    posture = jp.sum(jp.square(positions - nominal[qpos_indices]))
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
        "posture": posture,
        "standing_posture": jp.where(jp.all(command == 0), posture, 0.0),
        "foot_slip": jp.asarray(slip),
        "foot_air_time": jp.asarray(air_time),
        "termination": terminated.astype(jp.float32),
    }


def weighted_terms(terms, config):
    weights = asdict(config)
    return {
        name: jp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) * weights[name]
        for name, value in terms.items()
    }
