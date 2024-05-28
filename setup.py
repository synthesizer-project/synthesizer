"""Setup file for synthesizer.

Most of the build is defined in pyproject.toml but C extensions are not
supported in pyproject.toml yet. To enable the compilation of the C extensions
we use the legacy setup.py. This is ONLY used for the C extensions.
"""

import logging
import tempfile
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.errors import CompileError

LOG_FILE = "build_log.synth"
SETUPTOOLS_LOG_FILE = "setuptools_output.log"

# Set up custom logging for build_log.synth
custom_logger = logging.getLogger("custom_logger")
custom_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE)
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
custom_logger.addHandler(file_handler)

# Set up logging for setuptools output
setuptools_logger = logging.getLogger("setuptools")
setuptools_logger.setLevel(logging.INFO)
setuptools_file_handler = logging.FileHandler(SETUPTOOLS_LOG_FILE)
setuptools_file_handler.setFormatter(formatter)
setuptools_logger.addHandler(setuptools_file_handler)


def has_flags(compiler, flags):
    with tempfile.NamedTemporaryFile("w", suffix=".c") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=flags)
        except CompileError:
            return False
    return True


class BuildExt(build_ext):
    compile_flags = {
        "unix": [
            "-std=c99",
            "-Wall",
            "-O3",
            "-ffast-math",
            "-I{:s}".format(np.get_include()),
        ]
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        custom_logger.info(f"Compiler type: {ct}")

        opts = self.compile_flags.get(ct, [])
        links = []

        # if has_flags(self.compiler, ["-fopenmp"]):
        #     opts += ["-fopenmp"]
        #     links += ["-lgomp"]
        #     custom_logger.info("OpenMP support: using -fopenmp and -lgomp")
        # elif has_flags(self.compiler, ["-Xpreprocessor",
        # "-fopenmp", "-lomp"]):
        #     opts += ["-Xpreprocessor", "-fopenmp"]
        #     links += ["-lomp"]
        #     custom_logger.info(
        #         "OpenMP support: using -Xpreprocessor -fopenmp -lomp"
        #     )
        # elif has_flags(
        #     self.compiler,
        #     [
        #         "-Xpreprocessor",
        #         "-fopenmp",
        #         "-lomp",
        #         '-I"$(brew --prefix libomp)/include"',
        #         '-L"$(brew --prefix libomp)/lib"',
        #     ],
        # ):
        #     opts += [
        #         "-Xpreprocessor",
        #         "-fopenmp",
        #         '-I"$(brew --prefix libomp)/include"',
        #         '-L"$(brew --prefix libomp)/lib"',
        #     ]
        #     links += ["-lomp"]
        #     custom_logger.info(
        #         "OpenMP support: using Homebrew libomp configuration"
        #     )
        # else:
        #     custom_logger.info(
        #         "OpenMP support not found. Compilation may be slower."
        #     )

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = links
            custom_logger.info(
                f"Compiling {ext.name} with options: {opts} and links: {links}"
            )

        try:
            with open(SETUPTOOLS_LOG_FILE, "a") as setuptools_log:
                with redirect_stdout(setuptools_log), redirect_stderr(
                    setuptools_log
                ):
                    build_ext.build_extensions(self)
            custom_logger.info("Build successful.")
        except CompileError as e:
            custom_logger.error(f"Compilation failed: {e}")
            raise


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        self.run_command("build_ext")
        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        self.run_command("build_ext")
        install.run(self)


# Define the extension source files
src_files = {
    "synthesizer.extensions.integrated_spectra": (
        "src/synthesizer/extensions/integrated_spectra.c"
    ),
    "synthesizer.extensions.particle_spectra": (
        "src/synthesizer/extensions/particle_spectra.c"
    ),
    "synthesizer.imaging.extensions.spectral_cube": (
        "src/synthesizer/imaging/extensions/spectral_cube.c"
    ),
    "synthesizer.imaging.extensions.image": (
        "src/synthesizer/imaging/extensions/image.c"
    ),
    "synthesizer.extensions.sfzh": "src/synthesizer/extensions/sfzh.c",
    "synthesizer.extensions.los": "src/synthesizer/extensions/los.c",
    "synthesizer.extensions.integrated_line": (
        "src/synthesizer/extensions/integrated_line.c"
    ),
    "synthesizer.extensions.particle_line": (
        "src/synthesizer/extensions/particle_line.c"
    ),
}

# Create the extension objects
extensions = [
    Extension(
        path,
        sources=[source],
        include_dirs=[np.get_include()],
        py_limited_api=True,
    )
    for path, source in src_files.items()
]

# Ensure the custom build_ext command is used
setup(
    name="synthesizer",
    version="0.1",
    description="Synthesizer with C extensions",
    ext_modules=extensions,
    cmdclass={
        "build_ext": BuildExt,
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
)
