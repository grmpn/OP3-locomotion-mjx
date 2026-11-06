"""Brax/Orbax policy exports. These intentionally are not training-state resumes."""

import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import jax
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks

from op3_locomotion.config import Config
from op3_locomotion.policy import network_factory


def versions():
    packages = ("jax", "jaxlib", "mujoco", "mujoco-mjx", "brax", "flax", "orbax-checkpoint")
    result = {name: version(name) for name in packages}
    result["python"] = platform.python_version()
    try:
        result["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        result["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"]))
    except subprocess.CalledProcessError:
        result["git_revision"] = "uncommitted"
    return result


def save_export(root: Path, step: int, params, config: Config, mapping: dict):
    root = root.resolve()
    network_config = checkpoint.network_config(
        mapping["observation_size"],
        len(mapping["controlled_joints"]),
        True,
        network_factory(config.ppo),
    )
    checkpoint.save(str(root), step, params, network_config)
    path = root / f"{step:012d}"
    config.save(path / "run_config.json")
    (path / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    (path / "export.json").write_text(
        json.dumps(
            {
                "format": 1,
                "kind": "policy_export",
                "steps": step,
                "contains": [
                    "actor",
                    "critic",
                    "observation_normalization",
                    "network_config",
                    "mapping",
                ],
                "full_training_state": False,
                "versions": versions(),
            },
            indent=2,
        )
        + "\n"
    )
    temporary = root / "latest.tmp"
    temporary.write_text(path.name + "\n")
    temporary.replace(root / "latest")
    return path


def resolve_export(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "checkpoints").is_dir():
        path = path / "checkpoints"
    if (path / "latest").is_file():
        path = path / (path / "latest").read_text().strip()
    if not (path / "export.json").is_file():
        raise FileNotFoundError(f"Not an OP3 policy export: {path}")
    return path


def load_export(path: Path):
    path = resolve_export(path)
    config = Config.load(path / "run_config.json")
    mapping = json.loads((path / "mapping.json").read_text())
    # Brax 0.14.2's generic load_policy cannot deserialize null optional kernel
    # initializers. Recreate our shared factory from the saved typed settings,
    # while using Brax's parameter/normalizer loader unchanged.
    network = network_factory(config.ppo)(
        mapping["observation_size"],
        len(mapping["controlled_joints"]),
        preprocess_observations_fn=running_statistics.normalize,
    )
    policy = jax.jit(
        networks.make_inference_fn(network)(checkpoint.load(str(path)), deterministic=True)
    )
    return policy, config, mapping, path


def validate_mapping(expected: dict, actual: dict):
    # Roundtrip to normalize tuples/arrays; exact pinned model and timing are required.
    if json.loads(json.dumps(actual)) != expected:
        raise ValueError("Policy model/action/observation mapping differs from this environment")


def validate_warm_start(path: Path, config: Config, mapping: dict):
    path = resolve_export(path)
    previous = Config.load(path / "run_config.json")
    if (
        asdict(previous.env) != asdict(config.env)
        or previous.ppo.hidden_sizes != config.ppo.hidden_sizes
    ):
        raise ValueError("Warm start requires identical environment and network settings")
    validate_mapping(json.loads((path / "mapping.json").read_text()), mapping)
    return checkpoint.load(str(path))
