import numpy as np

import dmrg


class OperatorDriver:
    def __init__(self, settings):
        self.settings = settings
        self.local_dim = settings.local_dim
        self.nr_sites = settings.nr_sites

    def local_c(self, spin: str, dagger: bool, local_dim=4):
        """
        Local fermionic creation/annihilation operator in the basis
        {|0>, |up>, |down>, |up down>}.

        :param spin:
            The spin of the operator; 'up' or 'down'.
        :param dagger:
            The kind of operator: True -> creation, False -> annihilation
        :param JW:
            Whether to impose fermionic commutation relations with Jordan-Wigner dressing.
        """
        # local_dim = self.local_dim

        mat = np.zeros((local_dim, local_dim))
        jw_mat = self.jordan_wigner_mat()

        if spin == "up":
            mat[1, 0] = 1.0
            mat[3, 2] = 1.0
        elif spin == "down":
            mat[2, 0] = 1.0
            mat[3, 1] = 1.0
            # Impose antisymmetry for same site-spins
            mat = mat @ jw_mat
        else:
            raise ValueError(f"Unknown spin: {spin!r}")

        if not dagger:
            # Hermitian conjguate if annihilation operator
            # (sufficient with transpose since the mat is real)
            mat = mat.T

        return mat

    @staticmethod
    def jordan_wigner_mat(local_dim=4):  # or nr_qbits
        """
        Constructs the matrix representation, in the basis {|0>, |up>, |down>, |up down>}, necessary for imposing fermionic
        anticommutation relations via the Jordan-Wigner transformation.

        :returns:
            The (4,4) dimensional matrix representation as a numpy array.
        """
        if local_dim != 4:
            raise ValueError("Only implemented for local dimension 4")

        return np.diag((1.0, -1.0, -1.0, 1.0))

    def dress_JW(self, site: int, spin: str, op_kind: bool, local_dim=4):
        """
        Return the Jordan-Wigner transformed operator string.
        """
        jw_mat = self.jordan_wigner_mat(local_dim=local_dim)
        identity = np.eye(local_dim, dtype=np.float64)
        L = self.nr_sites

        operator_str = []
        for l in range(L):
            if l < site:
                operator = jw_mat
            elif l == site:
                operator = self.local_c(spin, op_kind)
            else:
                operator = identity
            operator_str.append(operator)
        return operator_str
