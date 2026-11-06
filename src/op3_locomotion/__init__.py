"""Robotis OP3 locomotion with direct MJX physics and Brax PPO."""

import os

# Leave room for the desktop on the development machine's 8 GB GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
