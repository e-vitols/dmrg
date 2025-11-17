import veloxchem as vlx
import numpy as np


class MatrixProductOperator:
    """
    Implements the MatrixProductOperator by importing the orbitals and -- specific, for each operator -- integrals from VeloxChem.
    """

    def __init__(self):
        """
        Initializes the MatrixProductOperator.

        Instance variables:
            - scf_results: The converged SCF results tensor from VeloxChem.
            - operator: Specific operator to construct, if not given, the Hamiltonian is assumed.
        """
        self.scf_results = None
        self.operator = "Ham"

    def construct_mpo(self, scf_results, operator=None):
        """
        Constructs the requested operator.

        :param scf_results:
            The converged SCF results tensor from VeloxChem.
        :param operator:
            The specific operator requested.
        """

        if operator is None:
            operator = self.operator
