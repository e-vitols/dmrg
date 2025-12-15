import copy

import numpy as np

from .hamiltonian import Hamiltonian
from .mpo import MpoDriver
from .mps import MpsDriver

# import veloxchem as vlx


class SweepDriver:
    def __init__(self):
        self.nsweeps = 50
