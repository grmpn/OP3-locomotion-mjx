"""Targeted air-time event accounting and legacy-configuration regressions."""

import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from op3_locomotion.config import Config, EnvConfig, RewardConfig
from op3_locomotion.env import OP3Env
from op3_locomotion.evaluate import completed_air_times
from op3_locomotion.rewards import update_air_time


def main():
    jax.config.update("jax_platforms", "cpu")
    update = jax.jit(update_air_time)
    timer = jp.zeros(2)
    armed = jp.zeros(2, dtype=bool)

    def tick(supported, moving=True, dt=0.002):
        nonlocal timer, armed
        timer, armed, bonus = update(timer, armed, jp.array(supported), moving, dt)
        return float(bonus)

    # No reward for the reset's airborne start or its first landing.
    for _ in range(50):
        assert tick([False, False]) == 0
    assert tick([True, True]) == 0
    # Left foot is unsupported for 0.1 s; pay only on landing, exactly once.
    for _ in range(50):
        assert tick([False, True]) == 0
    np.testing.assert_allclose(tick([True, True]), 0.1, atol=1e-6)
    assert tick([True, True]) == 0
    # Independent feet, including touchdown within a control interval.
    tick([False, False])
    np.testing.assert_allclose(tick([True, False]), 0.002, atol=1e-7)
    np.testing.assert_allclose(tick([True, True]), 0.004, atol=1e-7)

    # Stopping clears pending credit; resuming in the air cannot harvest it.
    tick([False, True], dt=0.1)
    assert tick([False, True], moving=False) == 0
    np.testing.assert_array_equal(timer, [0, 0])
    np.testing.assert_array_equal(armed, [False, False])
    for _ in range(20):
        assert tick([False, True]) == 0
    assert tick([True, True]) == 0
    for _ in range(20):
        assert tick([False, False], moving=False) == 0
        assert tick([True, True], moving=False) == 0

    # Same total completed air time yields the same raw reward, irrespective of
    # step count or duration. No hidden offset, target duration or clipping cap.
    def completed_swings(durations):
        nonlocal timer, armed
        timer, armed = jp.zeros(2), jp.zeros(2, dtype=bool)
        tick([True, True])
        total = 0
        for duration in durations:
            tick([False, True], dt=duration)
            total += tick([True, True])
        return total

    np.testing.assert_allclose(completed_swings([0.1, 0.1]), completed_swings([0.2]), atol=1e-7)
    np.testing.assert_allclose(completed_swings([0.05, 0.15, 0.8]), 1.0, atol=1e-7)

    support = np.array([[0, 1], [0, 0], [1, 0], [0, 1], [0, 0], [1, 0], [0, 1]], dtype=bool)
    durations = completed_air_times(support, 0.02)
    np.testing.assert_allclose(durations[0], [0.04])  # initial/final partial swings excluded
    np.testing.assert_allclose(durations[1], [0.04, 0.04])

    # Air time must also work when slip is disabled; both off omit the sensors.
    env = OP3Env(replace(EnvConfig(), reward=RewardConfig(foot_slip=0)), build_mjx=False)
    assert env.model.nsensordata == 96 and env.slip_sensor_indices is not None
    disabled = OP3Env(
        replace(EnvConfig(), reward=RewardConfig(foot_slip=0, foot_air_time=0)), build_mjx=False
    )
    assert disabled.model.nsensordata == 0 and disabled.slip_sensor_indices is None
    assert env.observation_size == disabled.observation_size == 64

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "legacy.json"
        config = asdict(Config())
        for name in ("foot_air_time", "air_time_contact_force", "air_time_command_threshold"):
            config["env"]["reward"].pop(name)
        path.write_text(json.dumps(config))
        assert Config.load(path).env.reward.foot_air_time == 0
    print("Passed: touchdown accounting, no cadence target, standing/reset gating, legacy config")


if __name__ == "__main__":
    main()
