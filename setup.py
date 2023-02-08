""" Setup file for synthesizer

TODO: ADD GUBBINS, LICENCE, COLLABORATION DETAILS ETC
"""

from setuptools import setup, find_packages

from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy as np

extensions = [
    Extension("weights", ["synthesizer/weights.pyx"], 
        define_macros=[('CYTHON_TRACE', '1')])
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
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions),
    include_dirs = [np.get_include()],
)
