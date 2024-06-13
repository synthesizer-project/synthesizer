"""Setup file for synthesizer.

Most of the build is defined in pyproject.toml but C extensions are not
supported in pyproject.toml yet. To enable the compilation of the C extensions
we use the legacy setup.py. This is ONLY used for the C extensions.

This script enables the user to override the CFLAGS and LDFLAGS environment
variables to pass custom flags to the compiler and linker. It also enables the
definition of preprocessing flags that can then be used in the C code.

Example:
    To build the C extensions with debugging checks enabled, run the following
    command:

    ```bash
    WITH_DEBUGGING_CHECKS=1 pip install .
    ```

    To build the C extensions with custom compiler flags, run the following
    command:

    ```bash
    CFLAGS="-O3 -march=native" pip install .
    ```
"""

import logging
import os
import sys
import tempfile
from datetime import datetime
from distutils.ccompiler import new_compiler

import numpy as np
from setuptools import Extension, setup
from setuptools.errors import CompileError


def has_flags(compiler, flags):
    """
    Check whether the C compiler allows for a flag to be passed.

    This is tested by compiling a small temporary test program.

    Args:
        compiler
            The loaded C compiler.
        flags (list)
            A list of compiler flags to test the compiler with.

    Returns
        bool
            Success/Failure
    """
    # Attempt to compile a temporary C file
    with tempfile.NamedTemporaryFile("w", suffix=".c") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=flags)
        except CompileError:
            return False
    return True


def create_extension(
    name,
    sources,
    compile_flags=[],
    links=[],
    include_dirs=[],
):
    """
    Create a C extension module.

    Args:
        name: The name of the extension module.
        sources: A list of source files.
    """
    logger.info(
        f"### Creating extension {name} with compile args: "
        f"{compile_flags} and link args: {links}"
    )
    return Extension(
        name,
        sources=sources,
        include_dirs=[np.get_include()] + include_dirs,
        extra_compile_args=compile_flags,
        extra_link_args=links,
    )


# Get environment variables we'll need for optional features and flags
CFLAGS = os.environ.get("CFLAGS", "")
LDFLAGS = os.environ.get("LDFLAGS", "")
WITH_OPENMP = os.environ.get("WITH_OPENMP", "")
WITH_DEBUGGING_CHECKS = "ENABLE_DEBUGGING_CHECKS" in os.environ
RUTHLESS = "RUTHLESS" in os.environ

# Define the log file
LOG_FILE = "build_synth.log"

# Set up logging (this allows us to log messages directly to a file during
# the build)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Include a log message for the start of the build
logger.info("\n")
logger.info("### Building synthesizer C extensions")

# Log the Python version
logger.info(f"### Python version: {sys.version}")

# Log the time and date the build started
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger.info(f"### Build started: {current_time}")

# Tell the user lines starting with '###' are log messages from setup.py
logger.info(
    "### Log messages starting with '###' are from setup.py, "
    "other messages are from the build process."
)

# Log the system platform
logger.info(f"### System platform: {sys.platform}")

# Report the environment variables
logger.info(f"### CFLAGS: {CFLAGS}")
logger.info(f"### LDFLAGS: {LDFLAGS}")
logger.info(f"### WITH_OPENMP: {WITH_OPENMP}")
if WITH_DEBUGGING_CHECKS:
    logger.info(f"### WITH_DEBUGGING_CHECKS: {WITH_DEBUGGING_CHECKS}")

# Create a compiler instance
compiler = new_compiler()

# Determine the platform-specific default compiler and linker flags
if sys.platform == "darwin":  # macOS
    default_compile_flags = [
        "-std=c99",
        "-Wall",
        "-O3",
        "-ffast-math",
        "-g",
    ]
    default_link_args = []
    include_dirs = ["/usr/local/include"]
elif sys.platform == "win32":  # windows
    default_compile_flags = [
        "/std:c99",
        "/Ox",
        "/fp:fast",
    ]
    default_link_args = []
    include_dirs = []
else:  # Unix-like systems (Linux)
    default_compile_flags = [
        "-std=c99",
        "-Wall",
        "-O3",
        "-ffast-math",
        "-g",
    ]
    default_link_args = []
    include_dirs = ["/usr/include"]

# Add OpenMP flags if requested
if len(WITH_OPENMP) > 0:
    # If WITH_OPENMP is a path add it to the includes and link args
    if os.path.exists(WITH_OPENMP):
        include_dirs.append(f"{WITH_OPENMP}/include")
        default_link_args.append(f"-L{WITH_OPENMP}/lib")
    if sys.platform == "darwin":
        default_compile_flags.append("-Xpreprocessor")
        default_compile_flags.append("-fopenmp")
        default_link_args.append("-lomp")
    elif sys.platform == "win32":
        default_compile_flags.append("/openmp")
    else:
        default_compile_flags.append("-fopenmp")
        default_link_args.append("-lgomp")
    default_compile_flags.append("-DWITH_OPENMP")

# If RUTHLESS is set, add all the flags to convert warnings to errors and
# enable all warnings
if RUTHLESS:
    if sys.platform == "darwin":
        default_compile_flags.append("-Werror")
        default_compile_flags.append("-Wall")
        default_compile_flags.append("-Wextra")
    elif sys.platform == "win32":
        default_compile_flags.append("/WX")
        default_compile_flags.append("/Wall")
    else:
        default_compile_flags.append("-Werror")
        default_compile_flags.append("-Wall")
        default_compile_flags.append("-Wextra")

# Get user specified flags
compile_flags = CFLAGS.split()
link_args = LDFLAGS.split()

# If no flags are specified, use the default flags
if len(compile_flags) == 0:
    compile_flags = default_compile_flags
if len(link_args) == 0:
    link_args = default_link_args

# Add preprocessor flags
if WITH_DEBUGGING_CHECKS == "1":
    compile_flags.append("-DWITH_DEBUGGING_CHECKS")


# Define the extension modules
extensions = [
    create_extension(
        "synthesizer.extensions.openmp_check",
        ["src/synthesizer/extensions/openmp_check.c"],
        compile_flags=compile_flags,
        links=link_args,
        include_dirs=include_dirs,
    ),
    create_extension(
        "synthesizer.extensions.integrated_spectra",
        [
            "src/synthesizer/extensions/integrated_spectra.c",
            "src/synthesizer/extensions/weights.c",
            "src/synthesizer/extensions/property_funcs.c",
            "src/synthesizer/extensions/timers.c",
        ],
        compile_flags=compile_flags,
        links=link_args,
        include_dirs=include_dirs,
    ),
    create_extension(
        "synthesizer.extensions.particle_spectra",
        [
            "src/synthesizer/extensions/particle_spectra.c",
            "src/synthesizer/extensions/weights.c",
            "src/synthesizer/extensions/property_funcs.c",
            "src/synthesizer/extensions/timers.c",
        ],
        compile_flags=compile_flags,
        links=link_args,
        include_dirs=include_dirs,
    ),
    create_extension(
        "synthesizer.imaging.extensions.spectral_cube",
        ["src/synthesizer/imaging/extensions/spectral_cube.c"],
        compile_flags=compile_flags,
        links=link_args,
        include_dirs=include_dirs,
    ),
    create_extension(
        "synthesizer.imaging.extensions.image",
        ["src/synthesizer/imaging/extensions/image.c"],
        compile_flags=compile_flags,
        links=link_args,
        include_dirs=include_dirs,
    ),
    create_extension(
        "synthesizer.extensions.sfzh",
        [
            "src/synthesizer/extensions/sfzh.c",
            "src/synthesizer/extensions/weights.c",
            "src/synthesizer/extensions/property_funcs.c",
            "src/synthesizer/extensions/timers.c",
        ],
        compile_flags=compile_flags,
        links=link_args,
        include_dirs=include_dirs,
    ),
    create_extension(
        "synthesizer.extensions.los",
        [
            "src/synthesizer/extensions/los.c",
            "src/synthesizer/extensions/weights.c",
            "src/synthesizer/extensions/property_funcs.c",
        ],
        compile_flags=compile_flags,
        links=link_args,
        include_dirs=include_dirs,
    ),
    create_extension(
        "synthesizer.extensions.integrated_line",
        [
            "src/synthesizer/extensions/integrated_line.c",
            "src/synthesizer/extensions/weights.c",
            "src/synthesizer/extensions/property_funcs.c",
            "src/synthesizer/extensions/timers.c",
        ],
        compile_flags=compile_flags,
        links=link_args,
        include_dirs=include_dirs,
    ),
    create_extension(
        "synthesizer.extensions.particle_line",
        [
            "src/synthesizer/extensions/particle_line.c",
            "src/synthesizer/extensions/weights.c",
            "src/synthesizer/extensions/property_funcs.c",
            "src/synthesizer/extensions/timers.c",
        ],
        compile_flags=compile_flags,
        links=link_args,
        include_dirs=include_dirs,
    ),
]

# Setup configuration
setup(
    ext_modules=extensions,
)
