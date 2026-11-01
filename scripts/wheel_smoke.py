"""Verify a clean wheel install, its notices and the complete solver regression suite."""

import importlib.metadata
from pathlib import Path
import subprocess
import sys
import softqp
from packaging.specifiers import SpecifierSet

project = Path(__file__).resolve().parents[1]
installed = Path(softqp.__file__).resolve()
if installed.is_relative_to(project / "src"):
    raise RuntimeError("Wheel smoke imported the source checkout instead of the installed wheel")
if "site-packages" not in installed.parts:
    raise RuntimeError("Wheel smoke requires an isolated site-packages install")
dist = importlib.metadata.distribution("softqp")
assert dist.version == softqp.__version__ == "0.1.0"
assert SpecifierSet(dist.metadata["Requires-Python"]) == SpecifierSet(">=3.12,<3.13")
names = {Path(str(file)).name for file in dist.files}
assert {"LICENSE", "NOTICE", "AMD.txt", "OSQPSlack-MIT.txt"} <= names
assert dist.metadata["License-Expression"] == "Apache-2.0 AND MIT AND BSD-3-Clause"
print("Clean installed wheel, Python support and all notices verified", flush=True)
subprocess.run([sys.executable, "-m", "pytest", str(project / "tests"), "-q"], check=True)
