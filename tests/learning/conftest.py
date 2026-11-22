"""Collect native learning controls only under their exact optional CPU profile."""

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path


DIRECTORY = Path(__file__).parent


def pytest_ignore_collect(collection_path):
    if collection_path.parent != DIRECTORY or collection_path.suffix != ".py":
        return None
    pins = {"torch": "2.15.0+cpu", "torch-geometric": "2.8.1", "osqp": "1.1.3",
            "daqp": "0.10.3", "clarabel": "0.11.1", "piqp": "0.6.4"}
    if os.environ.get("SOFTQP_REQUIRE_LEARNING_TESTS") == "1":
        # Explicit qualification runs fail collection if any import/pin is missing.
        return None
    try:
        return any(version(name) != expected for name, expected in pins.items())
    except PackageNotFoundError:
        return True
