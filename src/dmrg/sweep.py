import copy

import numpy as np
import veloxchem as vlx

from .hamiltonian import Hamiltonian
from .mpo import MatrixProductOperator
from .mps import MatrixProductState


class Sweep:
    def __init__(self):
        self.nsweeps = 50
