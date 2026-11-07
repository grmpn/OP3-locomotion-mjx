"""Small, serializable settings. Commands use body +x forward, +y left, +z yaw."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

REVISION = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
LEG_JOINTS = tuple(
    f"{side}_{joint}"
    for side in ("l", "r")
    for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ank_pitch", "ank_roll")
)


@dataclass(frozen=True)
class RewardConfig:
    linear_tracking: float = 3.0
    yaw_tracking: float = 1.0
    upright: float = 0.3
    vertical_velocity: float = -0.5
    roll_pitch_velocity: float = -0.05
    effort: float = -0.002
    action_rate: float = -0.02
    joint_limits: float = -1.0
    posture: float = -0.05
    standing_posture: float = 0.0  # additional penalty at exactly zero velocity command
    foot_slip: float = -5.0
    slip_contact_force: float = 1.0  # N, minimum summed floor load per foot
    foot_air_time: float = 2.0
    air_time_contact_force: float = 1.0  # N, supported above this floor load
    air_time_command_threshold: float = 0.05  # fraction of each trained command bound
    termination: float = -2.0
    sigma_linear: float = 0.15  # m/s
    sigma_yaw: float = 0.4  # rad/s


@dataclass(frozen=True)
class EnvConfig:
    physics_backend: str = "warp"
    collision_profile: str = "locomotion"
    collision_maxhullvert: int = -1  # -1 retains upstream hulls; otherwise >=4
    physics_dt: float = 0.002
    control_dt: float = 0.02
    episode_seconds: float = 10.0
    command_seconds: float = 3.0
    command_max: tuple[float, float, float] = (0.25, 0.15, 0.8)
    standing_probability: float = 0.2
    command_sampling: str = "uniform"
    lift_reset_probability: float = 0.0
    lift_reset_knee_flexion: float = 0.45  # rad added to lifted knee; half at hip/ankle
    lift_reset_hip_roll: float = 0.15  # rad, both legs lean toward the supporting foot
    action_scale: float = 0.5  # radians
    min_height: float = 0.18
    min_up_cosine: float = 0.5
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        if self.command_sampling not in ("uniform", "balanced"):
            raise ValueError("command_sampling must be uniform or balanced")
        if self.physics_backend not in ("jax", "warp"):
            raise ValueError("physics_backend must be jax or warp")
        if self.collision_profile not in ("full", "locomotion"):
            raise ValueError("collision_profile must be full or locomotion")
        if self.collision_maxhullvert != -1 and self.collision_maxhullvert < 4:
            raise ValueError("collision_maxhullvert must be -1 or at least 4")
        if self.reward.foot_slip > 0 or self.reward.slip_contact_force <= 0:
            raise ValueError("foot_slip must be nonpositive and slip_contact_force positive")
        if (
            not 0 <= self.reward.foot_air_time < float("inf")
            or not 0 < self.reward.air_time_contact_force < float("inf")
            or not 0 <= self.reward.air_time_command_threshold < 1
        ):
            raise ValueError("Invalid air-time weight, support force or command threshold")
        if self.physics_dt <= 0 or self.control_dt < self.physics_dt:
            raise ValueError("Require 0 < physics_dt <= control_dt")
        if abs(self.action_repeat * self.physics_dt - self.control_dt) > 1e-8:
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        if self.episode_seconds < self.control_dt or self.command_seconds < self.control_dt:
            raise ValueError("Episode and command intervals must be at least one control step")
        if len(self.command_max) != 3 or min(self.command_max) <= 0:
            raise ValueError("command_max must contain three positive bounds")
        if not 0 <= self.standing_probability <= 1 or self.action_scale <= 0:
            raise ValueError("Invalid standing probability or action scale")
        if not -float("inf") < self.reward.standing_posture <= 0:
            raise ValueError("standing_posture must be finite and nonpositive")
        if (
            not 0 <= self.lift_reset_probability <= 1
            or not 0 < self.lift_reset_knee_flexion < float("inf")
            or not 0 <= self.lift_reset_hip_roll < float("inf")
        ):
            raise ValueError("Invalid raised-foot reset probability or joint offsets")
        if (
            self.lift_reset_probability
            and max(self.lift_reset_knee_flexion, self.lift_reset_hip_roll) > self.action_scale
        ):
            raise ValueError("Raised-foot reset offsets must fit the policy action range")

    @property
    def action_repeat(self) -> int:
        return round(self.control_dt / self.physics_dt)

    @property
    def episode_steps(self) -> int:
        return round(self.episode_seconds / self.control_dt)

    @property
    def command_steps(self) -> int:
        return round(self.command_seconds / self.control_dt)


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 0
    num_timesteps: int = 30_000_000
    num_envs: int = 256
    batch_size: int = 128
    num_minibatches: int = 4
    unroll_length: int = 20
    num_updates_per_batch: int = 4
    learning_rate: float = 0.0003
    entropy_cost: float = 0.01
    discounting: float = 0.99
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    num_evals: int = 11
    num_eval_envs: int = 32
    hidden_sizes: tuple[int, ...] = (128, 128, 128)

    def __post_init__(self):
        if (
            min(
                self.num_envs,
                self.batch_size,
                self.num_minibatches,
                self.num_timesteps,
                self.unroll_length,
                self.num_eval_envs,
            )
            <= 0
        ):
            raise ValueError("PPO counts must be positive")
        if self.batch_size * self.num_minibatches % self.num_envs:
            raise ValueError("batch_size * num_minibatches must be divisible by num_envs")


@dataclass(frozen=True)
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Config":
        value = json.loads(path.read_text())
        env = value["env"]
        # Older exports predate these rewards; preserve their original objective.
        env["reward"].setdefault("foot_slip", 0.0)
        env["reward"].setdefault("foot_air_time", 0.0)
        env["reward"] = RewardConfig(**env["reward"])
        env["command_max"] = tuple(env["command_max"])
        ppo = value["ppo"]
        ppo["hidden_sizes"] = tuple(ppo["hidden_sizes"])
        return cls(EnvConfig(**env), PPOConfig(**ppo))
