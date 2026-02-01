import numpy as np
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "dmrg._parity",
        ["src/dmrg/_parity.cpp"],
        include_dirs=[np.get_include()],
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
