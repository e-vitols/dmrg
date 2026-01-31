import numpy as np

# import dmrg


class OperatorDriver:
    """
    Implements the local operators and operator strings, in the local basis {|0>, |up>, |down>, |up down>}.
    """

    def __init__(self, settings):
        self.settings = settings
        self.local_dim = settings.local_dim
        self.nr_sites = settings.nr_sites

    ##### local parts #####

    def local_c(self, spin: str, dagger: bool):
        """
        Local fermionic creation/annihilation operator in the basis
        {|0>, |up>, |down>, |up down>}.

        :param spin:
            The spin of the operator; 'up' or 'down'.
        :param dagger:
            The kind of operator: True -> creation, False -> annihilation

        :return:
            The local creation/annihilation operator.
        """
        d = self.local_dim

        mat = np.zeros((d, d))
        jw_mat = self.parity_local()

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
    def parity_local(local_dim=4):  # or nr_qbits
        """
        Constructs the matrix representation, in the basis {|0>, |up>, |down>, |up down>}, necessary for imposing fermionic
        anticommutation relations via the Jordan-Wigner transformation.

        :returns:
            The (4,4) dimensional matrix representation as a numpy array.
        """
        if local_dim != 4:
            raise ValueError("Only implemented for local dimension 4")

        return np.diag((1.0, -1.0, -1.0, 1.0))

    def dress_JW(self, site: int, spin: str, op_kind: bool):
        """
        Return the Jordan-Wigner transformed operator string.
        """
        P = self.parity_local()
        I = self.identity_local()
        L = self.nr_sites

        operator_str = []
        for l in range(L):
            if l < site:
                operator = P
            elif l == site:
                operator = self.local_c(spin, op_kind)
            else:
                operator = I
            operator_str.append(operator)
        return operator_str

    def zero_local(self):
        d = self.local_dim
        return np.zeros((d, d))

    def identity_local(self):
        d = self.local_dim
        return np.eye(d)

    def number_local(self, spin):
        """
        :param spin:
            'up'/'down' (string)
        """
        cd = self.local_c(spin, True)
        c = self.local_c(spin, False)

        return cd @ c

    def number_total_local(self):
        return self.number_local("up") + self.number_local("down")

    def double_occ_local(self):
        return self.number_local("up") @ self.number_local("down")

    ##### operator strings #####

    def cd_c_opstrings(self, i: int, j: int, spin: str):
        """
        JW-dressed string for c_i^dagger c_j.
        """
        cd_i = self.dress_JW(i, spin, True)
        c_j = self.dress_JW(j, spin, False)
        return [A @ B for A, B in zip(cd_i, c_j)]

    def chem_pot_opstrings(self, coupling=1.0):
        """
        Returns (opstrings, coeffs) for coupling * sum_i (n_up(i) + n_down(i)).
        NOTE this is equal to the total number operator when coupling=1
        """
        L = self.nr_sites
        I = self.identity_local()
        n_tot = self.number_total_local()

        opstrings = []
        coeffs = []

        for i in range(L):
            # identity everywhere
            ops = [I for _ in range(L)]
            # replace at site i
            ops[i] = n_tot
            opstrings.append(ops)
            coeffs.append(coupling)

        return opstrings, coeffs

    def number_total_opstrings(self):
        """
        Returns (opstrings, coeffs) for sum_i (n_up(i) + n_down(i)).
        """
        L = self.nr_sites
        I = self.identity_local()
        n_tot = self.number_total_local()
        coupling = 1.0

        opstrings = []
        coeffs = []

        for i in range(L):
            # identity everywhere
            ops = [I for _ in range(L)]
            # replace at site i
            ops[i] = n_tot
            opstrings.append(ops)
            coeffs.append(coupling)

        return opstrings, coeffs

    def number_opstrings(self, spin):
        """
        Returns (opstrings, coeffs) for sum_i n_up/down(i).
        """
        L = self.nr_sites
        I = self.identity_local()
        n_tot = self.number_local(spin)
        coupling = 1.0

        opstrings = []
        coeffs = []

        for i in range(L):
            # identity everywhere
            ops = [I for _ in range(L)]
            # replace at site i
            ops[i] = n_tot
            opstrings.append(ops)
            coeffs.append(coupling)

        return opstrings, coeffs

    def hubbard_opstrings(self, t=1.0, U=0.0, mu=0.0):
        """
        Returns (opstrings, coeffs) for:
          U * sum_i n_up(i) n_dn(i)  - mu * sum_i (n_up(i)+n_dn(i))
          - t * sum_{<i,i+1>,sigma} (c_i^† c_{i+1} + c_{i+1}^† c_i)

        """
        L = self.nr_sites
        I = self.identity_local()

        n_tot = self.number_total_local()
        n_dbl = self.double_occ_local()

        opstrings = []
        coeffs = []

        # hopping term
        if t != 0.0:
            for i in range(L - 1):
                for spin in ("up", "down"):
                    opstrings.append(self.cd_c_opstrings(i, i + 1, spin))
                    coeffs.append(-t)
                    opstrings.append(self.cd_c_opstrings(i + 1, i, spin))
                    coeffs.append(-t)

        # onsite repulsion U term
        if U != 0.0:
            for i in range(L):
                ops = [I for _ in range(L)]
                ops[i] = n_dbl
                opstrings.append(ops)
                coeffs.append(U)

        # chemical potential term
        if mu != 0.0:
            for i in range(L):
                ops = [I for _ in range(L)]
                ops[i] = n_tot
                opstrings.append(ops)
                coeffs.append(-mu)

        return opstrings, coeffs
