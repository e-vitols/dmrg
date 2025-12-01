import copy

import numpy as np
import veloxchem as vlx

from .hamiltonian import Hamiltonian
from .mpo import MpoDriver
from .mps import MpsDriver


class SweepDriver:
    def __init__(self):
        self.nsweeps = 50
