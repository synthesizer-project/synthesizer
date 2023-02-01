""" Setup file for synthesizer

TODO: ADD GUBBINS, LICENCE, COLLABORATION DETAILS ETC
"""
# from distutils.core import setup, Extension
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

from distutils.errors import CompileError
import numpy as np

# The below is taken from PySPHveiwer. Might need to change and adapt for but
# it's a good starting point.

# First we find out what link/etc. flags we want based on the compiler.


def has_flags(compiler, flags):
    """
    This checks whether our C compiler allows for a flag to be passed,
    by compiling a small test program.
    """
    import tempfile
    from distutils.errors import CompileError

    with tempfile.NamedTemporaryFile("w", suffix=".c") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=flags)
        except CompileError:
            return False
    return True


# Build a build extension class that allows for finer selection of flags.

class BuildExt(build_ext):
    # Never check these; they're always added.
    # Note that we don't support MSVC here.
    compile_flags = {"unix": ["-std=c99", "-w",
                              "-ffast-math", "-I{:s}".format(np.get_include())]}

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.compile_flags.get(ct, [])
        links = []

        # Uncomment below if we use openMP and find a nice way to demonstrate
        # the user how to install it.
        # Will - NOTE: this broke for me with a complaint about -lgomp

        # # Check for the presence of -fopenmp; if it's there we're good to go!
        # if has_flags(self.compiler, ["-fopenmp"]):
        #     # Generic case, this is what GCC accepts
        #     opts += ["-fopenmp"]
        #     links += ["-lgomp"]

        # elif has_flags(self.compiler, ["-Xpreprocessor", "-fopenmp", "-lomp"]):
        #     # Hope that clang accepts this
        #     opts += ["-Xpreprocessor", "-fopenmp", "-lomp"]
        #     links += ["-lomp"]

        # elif has_flags(self.compiler, ["-Xpreprocessor",
        #                                "-fopenmp",
        #                                "-lomp",
        #                                '-I"$(brew --prefix libomp)/include"',
        #                                '-L"$(brew --prefix libomp)/lib"']):
        #     # Case on MacOS where somebody has installed libomp using homebrew
        #     opts += ["-Xpreprocessor",
        #              "-fopenmp",
        #              "-lomp",
        #              '-I"$(brew --prefix libomp)/include"',
        #              '-L"$(brew --prefix libomp)/lib"']

        #     links += ["-lomp"]

        # else:

        #     raise CompileError("Unable to compile C extensions on your machine, as we can't find OpenMP. "
        #                        "If you are on MacOS, try `brew install libomp` and try again. "
        #                        "If you are on Windows, please reach out on the GitHub and we can try "
        #                        "to find a solution.")

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = links

        build_ext.build_extensions(self)


extensions = [
    Extension(path, sources=[source])
    for path, source in {
        "synthesizer/extensions/make_sed":
            "synthesizer/extensions/make_sed.c",
        "synthesizer/extensions/weights":
            "synthesizer/extensions/weights.c",
        "synthesizer/imaging/extensions/sph_kernel_calc":
            "synthesizer/imaging/extensions/sph_kernel_calc.c",
        "synthesizer/imaging/extensions/ckernel_functions":
            "synthesizer/imaging/extensions/ckernel_functions.c",
    }.items()
]

setup(
    name="synthesizer",
    version="0.1.0",
    description="Tools ",
    # WILL NOTE: Lists of authors are not actually allowed by setuptools...
    # There are better package management solutions these days that do,
    # but I haven't deleved into those depths
    author="Chris Lovell, Jussi Kuusisto, Will Roper, Stephen Wilkins",
    author_email="FILL EMAILS, w.roper@sussex.ac.uk",
    url="https://github.com/flaresimulations/synthesizer",
    packages=find_packages(),
    install_requires=[  # Need to actually write the module to know this...
        "numpy>=1.14.5",
        "scipy>=1.7",

    ],
    # extras_require={"plotting": ["matplotlib>=2.2.0", "jupyter"]},
    # setup_requires=["pytest-runner", "flake8"],
    # tests_require=["pytest"],
    entry_points={
        "console_scripts": ["init_bc03=grids.grid_bc03:main",
                            "init_fsps=grids.grid_fsps:main"]
    },
    include_package_data=True,
    include_dirs=[np.get_include()],
    package_data={"synthesizer": ["*.c", "*.txt"]},
    cmdclass=dict(build_ext=BuildExt),
    ext_modules=extensions,

    # Need to populate these properly
    # license="GNU GPL v3",
    # classifiers=[
    #     "Programming Language :: Python :: 3.7",
    #     "Programming Language :: Python :: 3.8",
    #     "Programming Language :: Python :: 3.9",
    #     "Programming Language :: Python :: 3.10",
    #     "Programming Language :: Python :: 3.11",
    #     "Topic :: Utilities",
    # ],
    keywords="galaxy modelling smoothed particle hydrodynamics particles nbody"
    " galaxy formation parametric theory sph cosmology galaxy evolution survey"
    " space telescope SED sed spectral energy distribution stellar population "
    "synthesis",
)
