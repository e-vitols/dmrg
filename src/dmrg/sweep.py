import copy

import numpy as np

from .hamiltonian import Hamiltonian
from .mpo import MpoDriver
from .mps import MpsDriver

# import veloxchem as vlx


class SweepDriver(MpsDriver, MpoDriver):
    def __init__(self):
        super().__init__()
        self.nsweeps = 50
