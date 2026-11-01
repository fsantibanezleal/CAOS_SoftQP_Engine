"""Setuptools build of the pinned OSQP/pybind11 CMake extension."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import pybind11

ROOT = Path(__file__).parent.resolve()


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        output = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        # A pip source checkout can already have a long path. Nesting MSBuild's
        # compiler checks below build_temp can exceed Windows' 260-character limit.
        with tempfile.TemporaryDirectory(prefix="softqp-build-") as temporary:
            self._cmake_extension(ext, output, Path(temporary))

    def _cmake_extension(self, ext, output, build):
        config = "Debug" if self.debug else "Release"
        args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={output}",
            f"-DCMAKE_BUILD_TYPE={config}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            "-DOSQP_EXT_MODULE_NAME=ext_builtin",
            "-DOSQP_ALGEBRA_BACKEND=builtin",
            "-DOSQP_CODEGEN=OFF",
            "-DOSQP_ENABLE_INTERRUPT=OFF",
            "-DOSQP_BUILD_SHARED_LIB=OFF",
        ]
        subprocess.run(
            ["cmake", "-S", str(ROOT / "native/python"), "-B", str(build), *args],
            check=True,
        )
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build),
                "--config",
                config,
                "--parallel",
                os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "2"),
            ],
            check=True,
        )


setup(
    ext_modules=[Extension("softqp.ext_builtin", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
)
