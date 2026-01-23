import copy

import numpy as np

from .hamiltonian import HamiltonianDriver

# import veloxchem as vlx


class MpoDriver(HamiltonianDriver):
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
        # inherit the attributes and methods of HamiltonianDriver
        super().__init__()

        self.scf_results = None
        self.operator = "Ham"
        self.mpo = None

        self.one_elec_ints_ao = None
        self.two_elec_ints_ao = None
        self.nuc_repulsion_energy = None
        self.overlap = None

        self.nr_sites = None
        self.local_dim = None
        self.max_bond_dim = None

        # self.ham = HamiltonianDriver()

    def update_integrals(self, molecule, basis):
        """
        Get/update AO-basis integral attributes of the MPO, imported from VeloxChem.

        :param molecule:
            A molecule-object as defined in VeloxChem.
        :param basis:
            The associated basis-object as defined in VeloxChem.
        """
        S, h, g, V_nuc = self.ham.get_ints(molecule, basis)
        self.one_elec_ints_ao = h
        self.two_elec_ints_ao = g
        self.nuc_repulsion_energy = V_nuc
        self.overlap = S

    def transform_integrals(self, scf_results):
        """
        Transform the AO-basis integrals to MO-basis.

        :param scf_results:
            The converged SCF tensors from a VeloxChem SCF

        :return:
            Returns the transformed one- and two-electron integrals in MO-basis.
        """

        C_alpha = scf_results["C_alpha"]
        h_ij = np.einsum(
            "uv, ui, vj -> ij", self.one_elec_ints_ao, C_alpha, C_alpha, optimize=True
        )
        g_ijkl = np.einsum(
            "uvws, ui, vj, wk, sl -> ijkl",
            self.two_elec_ints_ao,
            C_alpha,
            C_alpha,
            C_alpha,
            C_alpha,
            optimize=True,
        )

        return h_ij, g_ijkl

    def construct_mpo(self, scf_results, operator=None):
        """
        Constructs the requested operator.

        :param scf_results:
            The converged SCF results tensor from VeloxChem.
        :param operator:
            The specific operator requested, if None takes the set self.operator.
        """
        if operator is None:
            operator = self.operator
        L = self.nr_sites
        d = self.local_dim
        # m = self.max_bond_dim

        h_ij, g_ijkl = self.transform_integrals(scf_results)

        # mpo = [for _ in range(L)]

        identity = np.eye(4)
        c_up_D = self.local_c("up", True)
        c_up = self.local_c("up", False)
        c_d_D = self.local_c("down", True)
        c_d = self.local_c("down", False)

        return h_ij, g_ijkl

    def construct_hamiltonian(self, scf_results):
        L = self.nr_sites
        d = self.local_dim
        # m = self.max_bond_dim

        h_ij, g_ijkl = self.transform_integrals(scf_results)

    def dress_JW(self, site: int, spin: str, op_kind: bool, local_dim=4):
        """
        Return the Jordan-Wigner transformed operator string.
        """
        jw_mat = self.jordan_wigner_mat(local_dim=local_dim)
        identity = np.eye(local_dim)
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

    def mpo_converted_JW(self, jw_string):
        mpo = []
        for core in jw_string:
            mpo.append(core[np.newaxis, :, :, np.newaxis])
        return mpo

    def local_c(self, spin: str, dagger: bool, JW=True, local_dim=4):
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
            mat[1, 0] = 1
            mat[3, 2] = 1
        elif spin == "down":
            mat[2, 0] = 1
            mat[3, 1] = 1
            # Impose antisymmetry for same site-spins
            if JW:
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

        return np.diag((1, -1, -1, 1))

    @staticmethod
    def apply_mpo(mpo, mps):
        """
        Apply operator (only local operator)

        :returns:
            The new, transformed MPS
        """
        transf_mps = mps.copy()

        for l in range(len(mps)):
            transf_mps[l] = np.einsum("dD, ldr -> lDr", mpo[l], mps[l])

        return transf_mps

    # @staticmethod
    def construct_twosite_operator(
        self, creat_ind_spin: tuple, annih_ind_spin: tuple, local_dim: int = 4
    ):
        """
        Implements two-site operator (particle-number conserving),
        i.e., annihilation followed by creation.
        """
        # INCORRECT currently
        creat_ind, creat_spin = creat_ind_spin
        annih_ind, annih_spin = annih_ind_spin
        creat_op = self.dress_JW(creat_ind, creat_spin, True)
        annih_op = self.dress_JW(annih_ind, annih_spin, False)
        same_site = False

        if annih_ind == creat_ind:
            same_site = True
            right_op = annih_op
        # operator with odd number of fermions to the left
        else:
            right_ind = max(creat_ind, annih_ind)

            if right_ind == creat_ind:
                right_op = creat_op.copy()
                left_ind = annih_ind
                left_op = annih_op
            else:
                right_op = annih_op.copy()
                left_ind = creat_ind
                left_op = creat_op

        two_site_op = right_op.copy()

        # I_1 x I_2 x ... x \gamma_i F_i x F_{i+1} x ... x F_{j-1} x \gamma_j x ... I_L
        jw_mat = self.jordan_wigner_mat()
        identity = np.eye(local_dim)

        if same_site:
            two_site_op[creat_ind] = creat_op[creat_ind] @ annih_op[creat_ind]
        else:
            for site in range(right_ind):
                if site == left_ind:
                    if left_ind == creat_ind:
                        two_site_op[site] = left_op[site] @ jw_mat
                    else:
                        two_site_op[site] = jw_mat @ left_op[site]
                elif site > left_ind:
                    two_site_op[site] = jw_mat
                elif site < left_ind:
                    two_site_op[site] = identity

        return two_site_op

    def _construct_twosite_operator(self, creat_ind_spin, annih_ind_spin, local_dim=4):
        """
        Implements two-site operator (particle-number conserving),
        i.e., annihilation followed by creation.

        :param creat_ind_spin:
            The site index and spin of the creation operator (tuple).
        :param annih_ind_spin:
            The site index and spin of the annihialtion operator (tuple).
        """
        creat_ind, creat_spin = creat_ind_spin
        annih_ind, annih_spin = annih_ind_spin

        creat_op = self.dress_JW(creat_ind, creat_spin, True, local_dim=local_dim)
        annih_op = self.dress_JW(annih_ind, annih_spin, False, local_dim=local_dim)

        # Overall operator = creat_op * annih_op  (rightmost acts first on a ket)
        return [A @ B for A, B in zip(creat_op, annih_op)]

    def _construct_foursite_operator(
        self,
        creat_ind_spin,
        creat_ind_spin_p,
        annih_ind_spin_p,
        annih_ind_spin,
        local_dim=4,
    ):
        """
        Implements two-site operator (particle-number conserving),
        i.e., annihilation followed by creation.

        :param creat_ind_spin:
            The site index and spin of the creation operator (tuple).
        :param annih_ind_spin:
            The site index and spin of the annihialtion operator (tuple).
        """
        creat_ind, creat_spin = creat_ind_spin
        annih_ind, annih_spin = annih_ind_spin
        creat_ind_p, creat_spin_p = creat_ind_spin_p
        annih_ind_p, annih_spin_p = annih_ind_spin_p

        creat_op = self.dress_JW(creat_ind, creat_spin, True, local_dim=local_dim)
        annih_op = self.dress_JW(annih_ind, annih_spin, False, local_dim=local_dim)
        creat_op_p = self.dress_JW(creat_ind_p, creat_spin_p, True, local_dim=local_dim)
        annih_op_p = self.dress_JW(
            annih_ind_p, annih_spin_p, False, local_dim=local_dim
        )

        # Overall operator = creat_op * annih_op  (rightmost acts first on a ket)
        return [
            A @ B @ C @ D
            for A, B, C, D in zip(creat_op, creat_op_p, annih_op_p, annih_op)
        ]

    def num_op(self, spin: str):
        """
        Implements the number operator for a specific spin sector.
        """
        L = self.nr_sites
        d = self.local_dim

        if spin == "up":
            a_up_dagger = self.local_c("up", True, JW=False)
            a_up = self.local_c("up", False, JW=False)
            n = a_up_dagger @ a_up
        elif spin == "down":
            a_down_dagger = self.local_c("down", True, JW=False)
            a_down = self.local_c("down", False, JW=False)
            n = a_down_dagger @ a_down

        identity = np.eye(d)
        zero = np.zeros((d, d))

        mpo = []

        # Left-most matrix (row-vector)
        W_0 = np.array([n, identity])[:, np.newaxis]
        mpo.append(W_0)

        W_i = np.array([[identity, n], [zero, identity]])
        for i in range(1, L - 1):
            mpo.append(W_i)

        # Right-most matrix (column-vector)
        W_L = np.swapaxes(W_0, 0, 1)
        mpo.append(W_L)

        return mpo

    def id_op(self):
        """
        Implements the identity operator, mainly for debugging purposes.
        """
        L = self.nr_sites
        d = self.local_dim

        identity = np.eye(d)
        zero = np.zeros((d, d))

        mpo = []

        # Left-most matrix (row-vector)
        W_0 = np.array([identity])
        mpo.append(W_0[np.newaxis])

        W_i = np.array([identity])
        for i in range(1, L - 1):
            mpo.append(W_i[np.newaxis])

        # Right-most matrix (column-vector)
        W_L = W_0
        mpo.append(W_L[np.newaxis])

        return mpo

    def one_e_ham(self):
        """
        Implements the one-electron hamiltonian naively.
        """

        nr_sites = self.nr_sites
        local_dim = self.local_dim

        nr_terms = 2 * nr_sites**2

        W = [[] for _ in range(nr_sites)]
        W[0] = np.zeros((nr_terms, 1, local_dim, local_dim))
        for l in range(1, nr_sites - 1):
            W[l] = np.zeros((nr_terms, nr_terms, local_dim, local_dim))
        W[-1] = np.zeros((1, nr_terms, local_dim, local_dim))

        n = 0
        for i in range(nr_sites):
            for j in range(nr_sites):
                for spin in ["up", "down"]:
                    creat_op, annih_op = (i, spin), (j, spin)
                    operator = self._construct_twosite_operator(creat_op, annih_op)
                    W[0][n, 0] = operator[0]
                    for l in range(1, nr_sites - 1):
                        W[l][n, n] = operator[l]
                    W[-1][0, n] = operator[-1]

                    n += 1
        return W
