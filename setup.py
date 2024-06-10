"""Setup file for synthesizer.

Most of the build is defined in pyproject.toml but C extensions are not
supported in pyproject.toml yet. To enable the compilation of the C extensions
we use the legacy setup.py. This is ONLY used for the C extensions.

This script enables the user to overise the CFLAGS and LDFLAGS environment
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


def filter_compiler_flags(compiler, flags):
    """
    Filter compiler flags to remove any that aren't compatible.

    We could use the compiler.has_flag() method to check if a flag is
    supported, but this method is not implemented for all compilers. Instead,
    we compile a simple C program with each flag and check if it compiles.

    We could just let the build fail if a flag is not supported, but this is
    more user-friendly and ensure Synthesizer will still build without placing
    a compilation hurdle in front of an inexperienced user.

    Args:
        compiler: The compiler instance.
        flags: A list of compiler flags to test.
    """
    valid_flags = []
    with tempfile.NamedTemporaryFile("w", suffix=".c") as f:
        f.write("int main() { return 0; }\n")
        for flag in flags:
            try:
                compiler.compile([f.name], extra_postargs=[flag])
                valid_flags.append(flag)
            except CompileError:
                logger.info(f"### Compiler flag {flag} is not supported.")
    return valid_flags


def create_extension(name, sources, compile_flags=[], links=[]):
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
        include_dirs=[np.get_include()],
        extra_compile_args=compile_flags,
        extra_link_args=links,
    )


# Get environment variables well need for optional features and flags
CFLAGS = os.environ.get("CFLAGS", "")
LDFLAGS = os.environ.get("LDFLAGS", "")
WITH_DEBUGGING_CHECKS = os.environ.get("WITH_DEBUGGING_CHECKS", "0")

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

# Determine the platform-specific default compiler and linker flags
if sys.platform == "darwin":  # macOS
    default_compile_flags = ["-std=c99", "-Wall", "-O3", "-ffast-math", "-g"]
    default_link_args = []
elif sys.platform == "win32":  # windows
    default_compile_flags = ["/std:c99", "/Ox", "/fp:fast"]
    default_link_args = []
else:  # Unix-like systems (Linux)
    default_compile_flags = ["-std=c99", "-Wall", "-O3", "-ffast-math", "-g"]
    default_link_args = []

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

# Create a compiler instance
compiler = new_compiler()

# Filter the flags
logger.info("### Testing extra compile args")
compile_flags = filter_compiler_flags(compiler, compile_flags)
logger.info(f"### Valid extra compile args: {compile_flags}")


# Define the extension modules
extensions = [
    create_extension(
        "synthesizer.extensions.integrated_spectra",
        [
            "src/synthesizer/extensions/integrated_spectra.c",
            "src/synthesizer/extensions/weights.c",
            "src/synthesizer/extensions/property_funcs.c",
        ],
        compile_flags=compile_flags,
        links=link_args,
    ),
    create_extension(
        "synthesizer.extensions.particle_spectra",
        [
            "src/synthesizer/extensions/particle_spectra.c",
            "src/synthesizer/extensions/weights.c",
            "src/synthesizer/extensions/property_funcs.c",
        ],
        compile_flags=compile_flags,
        links=link_args,
    ),
    create_extension(
        "synthesizer.imaging.extensions.spectral_cube",
        ["src/synthesizer/imaging/extensions/spectral_cube.c"],
        compile_flags=compile_flags,
        links=link_args,
    ),
    create_extension(
        "synthesizer.imaging.extensions.image",
        ["src/synthesizer/imaging/extensions/image.c"],
        compile_flags=compile_flags,
        links=link_args,
    ),
    create_extension(
        "synthesizer.extensions.sfzh",
        [
            "src/synthesizer/extensions/sfzh.c",
            "src/synthesizer/extensions/weights.c",
            "src/synthesizer/extensions/property_funcs.c",
        ],
        compile_flags=compile_flags,
        links=link_args,
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
    ),
    create_extension(
        "synthesizer.extensions.integrated_line",
        [
            "src/synthesizer/extensions/integrated_line.c",
            "src/synthesizer/extensions/weights.c",
            "src/synthesizer/extensions/property_funcs.c",
        ],
        compile_flags=compile_flags,
        links=link_args,
    ),
    create_extension(
        "synthesizer.extensions.particle_line",
        [
            "src/synthesizer/extensions/particle_line.c",
            "src/synthesizer/extensions/weights.c",
            "src/synthesizer/extensions/property_funcs.c",
        ],
        compile_flags=compile_flags,
        links=link_args,
    ),
]

# Setup configuration
setup(
    ext_modules=extensions,
)
