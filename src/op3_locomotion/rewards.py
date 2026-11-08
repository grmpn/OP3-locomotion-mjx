"""Unscaled named terms; multiply weighted sum by control_dt exactly once."""

from dataclasses import asdict

import jax.numpy as jp
import numpy as np

from op3_locomotion.observations import base_motion


def update_torque_window(history, index, squared_torque_integral):
    """Sliding integral imbalance per mirrored joint, then sum absolute values.

    Input is (left/right, six named leg joints), integrated at physics rate over
    one control interval, in (Nm)^2 s. Store signed differences, not instantaneous
    absolute differences, so alternating left/right effort can balance in time.
    Re-sum the finite buffer to avoid long-lived running-sum roundoff drift.
    """
    difference = squared_torque_integral[0] - squared_torque_integral[1]
    history = history.at[index].set(difference)
    penalty = jp.sum(jp.abs(jp.sum(history, axis=0)))
    return history, (index + 1) % history.shape[0], penalty


def sole_area_samples(positions, sizes):
    """Precompute area-weighted sole samples, without counting overlapping boxes twice.

    The pinned OP3 has two coplanar, axis-aligned boxes per foot. Partition their
    union at rectangle edges, then use 4x4 midpoint quadrature in each tile.
    This NumPy setup runs only during environment construction, never in rollouts.
    """
    all_points, all_weights = [], []
    fractions = (np.arange(4) + 0.5) / 4
    for centers, half_sizes in zip(np.asarray(positions), np.asarray(sizes), strict=True):
        bottoms = centers[:, 2] - half_sizes[:, 2]
        if not np.allclose(bottoms, bottoms[0], atol=1e-9, rtol=0):
            raise ValueError("Contact-area reward requires coplanar sole boxes")
        lower, upper = centers[:, :2] - half_sizes[:, :2], centers[:, :2] + half_sizes[:, :2]
        xs, ys = (np.unique(np.concatenate((lower[:, axis], upper[:, axis]))) for axis in (0, 1))
        points, weights = [], []
        for x0, x1 in zip(xs[:-1], xs[1:], strict=True):
            for y0, y1 in zip(ys[:-1], ys[1:], strict=True):
                center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
                if not np.any(np.all((center > lower) & (center < upper), axis=-1)):
                    continue
                for x in x0 + (x1 - x0) * fractions:
                    for y in y0 + (y1 - y0) * fractions:
                        points.append((x, y, bottoms[0]))
                        weights.append((x1 - x0) * (y1 - y0) / 16)
        all_points.append(points)
        all_weights.append(weights)
    return jp.asarray(all_points), jp.asarray(all_weights)


def foot_contact_area(heights, areas, contacts, sole_up, minimum_force, height_sigma):
    """Smooth near-floor area fraction, averaged over loaded feet; zero in flight.

    Absolute height penalizes penetration too. A side/upside-down foot earns no
    area credit. Averaging avoids an extra bonus just for having two stance feet.
    """
    load = jp.sum(jp.maximum(contacts[..., 0], 0.0), axis=-1)
    touching = load > minimum_force
    fraction = jp.sum(areas * jp.exp(-jp.square(heights / height_sigma)), axis=-1)
    fraction /= jp.sum(areas, axis=-1)
    fraction = jp.clip(fraction, 0.0, 1.0)
    eligible = touching & (sole_up > 0)
    return jp.sum(jp.where(eligible, fraction, 0.0)) / jp.maximum(jp.sum(touching), 1)


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
    contact_area=0.0,
    torque_symmetry=0.0,
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
        "torque_symmetry": jp.asarray(torque_symmetry),
        "action_rate": jp.sum(jp.square(action - previous_action)),
        "joint_limits": jp.sum(
            jp.maximum(lower - positions, 0.0) + jp.maximum(positions - upper, 0.0)
        ),
        "posture": posture,
        "standing_posture": jp.where(jp.all(command == 0), posture, 0.0),
        "foot_slip": jp.asarray(slip),
        "foot_air_time": jp.asarray(air_time),
        "foot_contact_area": jp.asarray(contact_area),
        "termination": terminated.astype(jp.float32),
    }


def weighted_terms(terms, config):
    weights = asdict(config)
    return {
        name: jp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) * weights[name]
        for name, value in terms.items()
    }
