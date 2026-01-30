import copy

import numpy as np

from .hamiltonian import HamiltonianDriver
from .mps import MpsDriver
from .operators import OperatorDriver

# import veloxchem as vlx


# class MpoDriver(MpsDriver, HamiltonianDriver):
class MpoDriver(MpsDriver):
    """
    Implements the MatrixProductOperator by importing the orbitals and -- specific, for each operator -- integrals from VeloxChem.

    # NOTE MPO structure is W[wL, wR, s_out, s_in]
    """

    def __init__(self, settings):
        """
        Initializes the MatrixProductOperator driver.

        Instance variables:
            - scf_results: The converged SCF results tensor from VeloxChem.
            - operator: Specific operator to construct, if not given, the Hamiltonian is assumed.
        """
        # TODO: create an initalize function requiring: self.local_dim and self.nr_sites

        # self.nr_sites = None
        # self.local_dim = None
        # self.nr_particles = None

        self.settings = settings
        self.ops = OperatorDriver(settings)

        # self.nr_sites = None
        self.nr_sites = settings.nr_sites
        # self.max_bond_dim = None
        self.max_bond_dim = settings.max_bond_dim
        # self.tolerance = 1e-9
        self.tolerance = settings.svd_thr
        # self.local_dim = None
        self.local_dim = settings.local_dim
        self.max_bond_dim = settings.max_bond_dim
        self.nr_particles = settings.nr_particles

    def mpo_converted_JW(self, jw_string):
        mpo = []
        for core in jw_string:
            mpo.append(core[np.newaxis, :, :, np.newaxis])
        return mpo

    @staticmethod
    def apply_local_mpo(mpo, mps):
        """
        Apply operator (only local operator)

        :returns:
            The transformed MPS
        """
        transf_mps = mps.copy()

        for l in range(len(mps)):
            transf_mps[l] = np.einsum("Dd, ldr -> lDr", mpo[l], mps[l])

        return transf_mps

    def _construct_twosite_operator(
        self, creat_ind_spin, annih_ind_spin, local_dim=4, virtual_bonds=False
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

        creat_op = self.ops.dress_JW(creat_ind, creat_spin, True, local_dim=local_dim)
        annih_op = self.ops.dress_JW(annih_ind, annih_spin, False, local_dim=local_dim)

        if virtual_bonds:
            creat_op = [site_op[np.newaxis, np.newaxis] for site_op in creat_op]
            annih_op = [site_op[np.newaxis, np.newaxis] for site_op in annih_op]

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

        creat_op = self.ops.dress_JW(creat_ind, creat_spin, True, local_dim=local_dim)
        annih_op = self.ops.dress_JW(annih_ind, annih_spin, False, local_dim=local_dim)
        creat_op_p = self.ops.dress_JW(
            creat_ind_p, creat_spin_p, True, local_dim=local_dim
        )
        annih_op_p = self.ops.dress_JW(
            annih_ind_p, annih_spin_p, False, local_dim=local_dim
        )

        # Overall operator = creat_op *creat_op * annih_op * annih_op  (rightmost acts first on a ket)
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
            a_up_dagger = self.ops.local_c("up", True)
            a_up = self.ops.local_c("up", False)
            n = a_up_dagger @ a_up
        elif spin == "down":
            a_down_dagger = self.ops.local_c("down", True)
            a_down = self.ops.local_c("down", False)
            n = a_down_dagger @ a_down

        identity = np.eye(d)
        zero = np.zeros((d, d))

        mpo = []

        # Left-most matrix (row-vector)
        W_0 = np.array([identity, n])[np.newaxis]
        mpo.append(W_0)

        W_i = np.array([[identity, n], [zero, identity]])
        for i in range(1, L - 1):
            mpo.append(W_i.copy())

        # Right-most matrix (column-vector)
        # W_L = np.array([identity, n])[:,np.newaxis]
        W_L = np.array([n, identity])[:, np.newaxis]
        mpo.append(W_L)

        return mpo

    def chem_pot_op(self, coupling=1):
        """
        Implements the chemical potential operator.
        """
        L = self.nr_sites
        d = self.local_dim

        # if spin == "up":
        a_up_dagger = self.ops.local_c("up", True)
        a_up = self.ops.local_c("up", False)
        n_up = a_up_dagger @ a_up
        # elif spin == "down":
        a_down_dagger = self.ops.local_c("down", True)
        a_down = self.ops.local_c("down", False)
        n_down = a_down_dagger @ a_down
        n = coupling * (n_up + n_down)

        identity = np.eye(d)
        zero = np.zeros((d, d))

        mpo = []

        # Left-most matrix (row-vector)
        W_0 = np.array([identity, n])[np.newaxis]
        mpo.append(W_0)

        W_i = np.array([[identity, n], [zero, identity]])
        for i in range(1, L - 1):
            mpo.append(W_i.copy())

        # Right-most matrix (column-vector)
        W_L = np.array([n, identity])[:, np.newaxis]
        mpo.append(W_L)

        return mpo

    def id_op(self):
        """
        Implements the identity operator, mainly for debugging purposes.
        """
        L = self.nr_sites
        d = self.local_dim

        identity = np.eye(d)

        mpo = []

        # Left-most matrix (row-vector)
        W_0 = np.array([identity])
        mpo.append(W_0[np.newaxis])

        W_i = np.array([identity])
        for i in range(1, L - 1):
            mpo.append(W_i[np.newaxis])

        # Right-most matrix (column-vector)
        W_L = W_0.copy()
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

    def get_twosite_mpo(self, mpo, center=None):
        """
        Docstring for get_twosite

        :param mpo:
            The full MPO.
        :param center:
            The center. (bool)

        :return:
            The two-site MPO with fused physical indices d**2. (array)
        """
        if center is None:
            center = self.canonical_center

        allowed_center = 0 <= center <= self.nr_sites - 2
        if allowed_center is not True:
            raise ValueError(
                "The bond to center on must be within the number of sites-1!"
            )

        left_center = center
        right_center = center + 1

        tmp = np.einsum(
            "abcd, bBCD -> aBcCdD", mpo[left_center], mpo[right_center], optimize=True
        )
        l, r, d = tmp.shape[0], tmp.shape[1], tmp.shape[2]
        W2 = tmp.reshape(l, r, d**2, d**2)
        return W2

    def old_naive_electronic_hamiltonian(self, t_ij, v_ijkl):
        """
        Constructs the full electronic Hamiltonian as an MPO.

        :param h_ij:
            The one-electron integrals in MO-basis.
        :param v_ijkl:
            The two-electron integrals in MO-basis.
        """
        nr_sites = self.nr_sites
        nr_particles = self.nr_particles
        local_dim = self.local_dim
        nr_terms = 4 * nr_sites**4 + 2 * nr_sites**2

        W_full = [[] for _ in range(nr_sites)]
        W_full[0] = np.zeros((1, nr_terms, local_dim, local_dim))
        for l in range(1, nr_sites - 1):
            W_full[l] = np.zeros((nr_terms, nr_terms, local_dim, local_dim))
        W_full[-1] = np.zeros((nr_terms, 1, local_dim, local_dim))

        n = 0
        for i in range(nr_sites):
            for j in range(nr_sites):
                for spin in ["up", "down"]:
                    creat_op, annih_op = (i, spin), (j, spin)
                    one_elec_coeff = t_ij[i, j]

                    operator = self._construct_twosite_operator(creat_op, annih_op)

                    # Include coefficient by scaling only the first MPO-tensor
                    W_full[0][0, n] = one_elec_coeff * operator[0]
                    for l_index in range(1, nr_sites - 1):
                        W_full[l_index][n, n] = operator[l_index]
                    W_full[-1][n, 0] = operator[-1]

                    n += 1

        for i in range(nr_sites):
            for j in range(nr_sites):
                for k in range(nr_sites):
                    for l in range(nr_sites):
                        for spin in ["up", "down"]:
                            for spin_p in ["up", "down"]:
                                creat_op, annih_op = (i, spin), (j, spin)
                                creat_op_p, annih_op_p = (k, spin_p), (l, spin_p)

                                operator = self._construct_foursite_operator(
                                    creat_op, creat_op_p, annih_op_p, annih_op
                                )
                                two_elec_coeff = v_ijkl[i, j, k, l]

                                # Include coefficient by scaling only the first MPO-tensor
                                coeff = 0.5 * two_elec_coeff
                                # operator[0] *= 0.5 * (one_elec_coeff + two_elec_coeff)

                                W_full[0][0, n] = coeff * operator[0]
                                for l_index in range(1, nr_sites - 1):
                                    W_full[l_index][n, n] = operator[l_index]
                                W_full[-1][n, 0] = operator[-1]

                                n += 1

        self.mpo_bond_dim = n

        return W_full

    def electronic_hamiltonian(self, t_ij, v_ijkl):
        """
        Constructs the full electronic Hamiltonian as an MPO.

        :param h_ij:
            The one-electron integrals in MO-basis.
        :param v_ijkl:
            The two-electron integrals in MO-basis.
        """
        nr_sites = self.nr_sites
        local_dim = self.local_dim
        # NOTE nr_terms of 2-electron terms halved
        nr_terms = 2 * nr_sites**4 + 3 * nr_sites**2

        W_full = [[] for _ in range(nr_sites)]
        W_full[0] = np.zeros((1, nr_terms, local_dim, local_dim))
        for l in range(1, nr_sites - 1):
            W_full[l] = np.zeros((nr_terms, nr_terms, local_dim, local_dim))
        W_full[-1] = np.zeros((nr_terms, 1, local_dim, local_dim))

        n = 0
        for i in range(nr_sites):
            for j in range(nr_sites):
                for spin in ["up", "down"]:
                    creat_op, annih_op = (i, spin), (j, spin)
                    one_elec_coeff = t_ij[i, j]

                    operator = self._construct_twosite_operator(creat_op, annih_op)

                    # Include coefficient by scaling only the first MPO-tensor
                    W_full[0][0, n] = one_elec_coeff * operator[0]
                    for l_index in range(1, nr_sites - 1):
                        W_full[l_index][n, n] = operator[l_index]
                    W_full[-1][n, 0] = operator[-1]

                    n += 1

        for i in range(nr_sites):
            for j in range(nr_sites):
                for k in range(nr_sites):
                    for l in range(nr_sites):
                        for spin in ["up", "down"]:
                            for spin_p in ["up", "down"]:
                                if (k, l, spin_p) < (i, j, spin):
                                    continue
                                creat_op, annih_op = (i, spin), (j, spin)
                                creat_op_p, annih_op_p = (k, spin_p), (l, spin_p)

                                operator = self._construct_foursite_operator(
                                    creat_op, creat_op_p, annih_op_p, annih_op
                                )
                                two_elec_coeff = v_ijkl[i, j, k, l]

                                # Include coefficient by scaling only the first MPO-tensor
                                # coeff = 0.5 * two_elec_coeff
                                coeff = two_elec_coeff

                                W_full[0][0, n] = coeff * operator[0]
                                for l_index in range(1, nr_sites - 1):
                                    W_full[l_index][n, n] = operator[l_index]
                                W_full[-1][n, 0] = operator[-1]

                                n += 1

        self.mpo_bond_dim = n

        return W_full

    def _electronic_hamiltonian(self, t_ij, v_ijkl):
        """
        Constructs the full electronic Hamiltonian as an MPO.

        :param h_ij:
            The one-electron integrals in MO-basis.
        :param v_ijkl:
            The two-electron integrals in MO-basis.
        """
        nr_sites = self.nr_sites
        nr_particles = self.nr_particles
        local_dim = self.local_dim
        nr_terms = 4 * nr_sites**4

        W_full = [[] for _ in range(nr_sites)]
        W_full[0] = np.zeros((1, nr_terms, local_dim, local_dim))
        for l in range(1, nr_sites - 1):
            W_full[l] = np.zeros((nr_terms, nr_terms, local_dim, local_dim))
        W_full[-1] = np.zeros((nr_terms, 1, local_dim, local_dim))

        n = 0
        for i in range(nr_sites):
            for j in range(nr_sites):
                for k in range(nr_sites):
                    for l in range(nr_sites):
                        for spin in ["up", "down"]:
                            for spin_p in ["up", "down"]:
                                creat_op, annih_op = (i, spin), (j, spin)
                                creat_op_p, annih_op_p = (k, spin_p), (l, spin_p)

                                one_elec_coeff = 0
                                if j == l:
                                    one_elec_coeff += t_ij[i, k]
                                if i == k:
                                    one_elec_coeff += t_ij[j, l]
                                one_elec_coeff /= nr_particles - 1

                                operator = self._construct_foursite_operator(
                                    creat_op, creat_op_p, annih_op_p, annih_op
                                )
                                two_elec_coeff = v_ijkl[i, j, k, l]

                                # Include coefficient by scaling only the first MPO-tensor
                                # operator[0] *= 0.5 * (one_elec_coeff + two_elec_coeff)
                                coeff = 0.5 * (one_elec_coeff + two_elec_coeff)

                                W_full[0][0, n] = coeff * operator[0]
                                for l_index in range(1, nr_sites - 1):
                                    W_full[l_index][n, n] = operator[l_index]
                                W_full[-1][n, 0] = operator[-1]

                                n += 1

        self.mpo_bond_dim = n

        return W_full

    def old_apply_effective_matvec(self, mpo, mps=None, center=None, two_site=True):
        """
        Docstring for get_effective_op

        :param mpo:
            The MPO of which we get the effective one. (list of arrays)
        :param center:
            The center at which we get the effective operator, i.e., centered at the bond between site center and center+1. (int)
        :param two_site:
            Whether to generate the effective two-site (True) or one-site (False). (bool)
        """
        if center is None:
            center = self.canonical_center

        left_center = center
        right_center = center + 1

        left_boundary = self.left_boundary(mpo, mps=mps, center=left_center)
        right_boundary = self.right_boundary(mpo, mps=mps, center=right_center)

        if mps is not None:
            P = self.get_twosite(center=center, mps=mps)

        tmp = np.einsum(
            "abcd, bBCD -> aBcCdD", mpo[left_center], mpo[right_center], optimize=True
        )
        l, r, d = tmp.shape[0], tmp.shape[1], tmp.shape[2]
        W2 = tmp.reshape(l, r, d**2, d**2)

        matvec = np.einsum(
            "bcTt, Lbd, dte, ecR -> LTR",
            W2,
            left_boundary,
            P,
            right_boundary,
            optimize=True,
        )

        return matvec

    def _apply_effective_ham_currsite(self, mpo, mps=None, center=None, two_site=True):
        """
        Docstring for get_effective_op
        NOTE: Included for bugfixing purposes
        NOTE: this avoids forming the two-site MPO intermediate.

        :param mpo:
            The MPO of which we get the effective one. (list of arrays)
        :param center:
            The center at which we get the effective operator, i.e., centered at the bond between site center and center+1. (int)
        :param two_site:
            Whether to generate the effective two-site (True) or one-site (False). (bool)
        """
        if center is None:
            center = self.canonical_center

        left_center = center
        right_center = center + 1

        L = self.left_boundary(mpo, mps=mps, center=left_center)
        R = self.right_boundary(mpo, mps=mps, center=right_center)

        if mps is not None:
            P = self.get_twosite(center=center, mps=mps)

        D = mpo[left_center].shape[3]
        E = mpo[right_center].shape[3]
        P = P.reshape(P.shape[0], D, E, P.shape[2])
        # avoid explicit W2 construction: contract mpo[left_center] and mpo[right_center] on the fly instead
        matvec = np.einsum(
            "bmSA, mcTB, Lbd, dABe, ecR -> LSTR",
            mpo[left_center],
            mpo[right_center],
            L,
            P,
            R,
            optimize=True,
        )

        matvec = matvec.reshape(
            matvec.shape[0], matvec.shape[1] * matvec.shape[2], matvec.shape[3]
        )

        return matvec

    @staticmethod
    def add_mpos(mpo1, mpo2):
        """
        Sum two MPOs given as lists of tensors W[i] with shape (l, r, d, d).
        Returns an MPO for (A + B).
        """
        if len(mpo1) != len(mpo2):
            raise ValueError("MPOs are of different lengths")
        L = len(mpo1)
        out = []

        for i in range(L):
            WA, WB = mpo1[i], mpo2[i]
            l1, r1, d1, d2 = WA.shape
            l2, r2, d1, d2 = WB.shape

            dtype = np.result_type(WA.dtype, WB.dtype)

            if i == 0:
                # (1, r1+r2, d, d)
                assert l1 == l2 == 1
                W = np.concatenate(
                    [WA.astype(dtype, copy=False), WB.astype(dtype, copy=False)], axis=1
                )

            elif i == L - 1:
                # (l1+l2, 1, d, d)
                assert r1 == r2 == 1
                W = np.concatenate(
                    [WA.astype(dtype, copy=False), WB.astype(dtype, copy=False)], axis=0
                )

            else:
                # block-diagonal: (l1+l2, r1+r2, d, d)
                W = np.zeros((l1 + l2, r1 + r2, d1, d2), dtype=dtype)
                W[:l1, :r1, :, :] = WA
                W[l1:, r1:, :, :] = WB

            out.append(W)

        return out

    @staticmethod
    def mpo_from_opstrings(opstrings, coeffs):
        """
        opstrings: list of length
        Returns MPO list W[i] with shapes (l,r,d,d).
        """
        T = len(opstrings)
        L = len(opstrings[0])
        d = opstrings[0][0].shape[0]

        W = [None] * L
        W[0] = np.zeros((1, T, d, d))
        for i in range(1, L - 1):
            W[i] = np.zeros((T, T, d, d))
        W[-1] = np.zeros((T, 1, d, d))

        for t, (ops, c) in enumerate(zip(opstrings, coeffs)):
            W[0][0, t] = c * ops[0]
            for i in range(1, L - 1):
                W[i][t, t] = ops[i]
            W[-1][t, 0] = ops[-1]
        return W

    def hubbard_mpo(self, t=1.0, U=0.0, mu=0.0):
        L = self.nr_sites
        d = self.local_dim
        I = np.eye(d)

        # local onsite ops
        cd_up = self.ops.local_c("up", True)  # ,  JW=False)
        c_up = self.ops.local_c("up", False)  # , JW=False)
        cd_dn = self.ops.local_c("down", True)  # ,  JW=False)
        c_dn = self.ops.local_c("down", False)  # , JW=False)

        n_up = cd_up @ c_up
        n_dn = cd_dn @ c_dn
        n = n_up + n_dn
        n_dbl = n_up @ n_dn

        opstrings = []
        coeffs = []

        # onsite: U * sum_i n_up n_dn  - mu * sum_i n
        if U != 0.0:
            for i in range(L):
                ops = [I] * L
                ops[i] = n_dbl
                opstrings.append(ops)
                coeffs.append(U)

        # chemical potential
        if mu != 0.0:
            for i in range(L):
                ops = [I] * L
                ops[i] = n
                opstrings.append(ops)
                coeffs.append(-mu)

        # hopping: -t * sum_{<i,i+1>,σ} (cd_i c_{i+1} + cd_{i+1} c_i)
        if t != 0.0:
            for i in range(L - 1):
                for spin in ["up", "down"]:
                    opstrings.append(
                        self._construct_twosite_operator((i, spin), (i + 1, spin))
                    )
                    coeffs.append(-t)
                    opstrings.append(
                        self._construct_twosite_operator((i + 1, spin), (i, spin))
                    )
                    coeffs.append(-t)

        return self.mpo_from_opstrings(opstrings, coeffs)

    def hubbard_ham(self, t=1.0, U=0.0, mu=0.0):
        """
        Implements the full Hubbard Hamiltonian, including chemical potential.
        """
        L = self.nr_sites
        d = self.local_dim

        I = np.eye(d, dtype=np.float64)
        Z = np.zeros((d, d), dtype=np.float64)

        c_up = self.ops.local_c("up", dagger=False)
        cd_up = self.ops.local_c("up", dagger=True)
        c_dn = self.ops.local_c("down", dagger=False)
        cd_dn = self.ops.local_c("down", dagger=True)

        n_up = cd_up @ c_up
        n_dn = cd_dn @ c_dn
        n = n_up + n_dn
        n_dbl = n_up @ n_dn

        P = I - 2.0 * n + 4.0 * n_dbl

        # Onsite term
        h_loc = (U * n_dbl) - (mu * n)

        # MPO-bond dimension
        D = 6

        W = [None] * L

        # first site: (1,D,d,d)
        W0 = np.zeros((1, D, d, d), dtype=np.float64)
        W0[0, 0] = I
        W0[0, 1] = cd_up @ P
        W0[0, 2] = P @ c_up
        W0[0, 3] = cd_dn @ P
        W0[0, 4] = P @ c_dn
        W0[0, 5] = h_loc
        W[0] = W0

        # middle sites: (D,D,d,d)
        for i in range(1, L - 1):
            Wi = np.zeros((D, D, d, d), dtype=np.float64)
            Wi[0, 0] = I

            # start hopping on this site
            Wi[0, 1] = cd_up @ P
            Wi[0, 2] = P @ c_up
            Wi[0, 3] = cd_dn @ P
            Wi[0, 4] = P @ c_dn

            # local onsite term
            Wi[0, 5] = h_loc

            # close hopping started on previous site
            Wi[1, 5] = -t * c_up
            Wi[2, 5] = -t * cd_up
            Wi[3, 5] = -t * c_dn
            Wi[4, 5] = -t * cd_dn

            # propagate end
            Wi[5, 5] = I

            W[i] = Wi

        WL = np.zeros((D, 1, d, d), dtype=np.float64)

        WL[0, 0] = h_loc

        # close hopping started on site L-2
        WL[1, 0] = -t * c_up
        WL[2, 0] = -t * cd_up
        WL[3, 0] = -t * c_dn
        WL[4, 0] = -t * cd_dn

        # end fianl contributions
        WL[5, 0] = I
        W[-1] = WL

        return W

    def site_occ_op(self, site):
        """
        Get the site occupation number operator.
        """
        cd_up = self.ops.local_c("up", True)
        c_up = self.ops.local_c("up", False)
        n_up = cd_up @ c_up
        cd_down = self.ops.local_c("down", True)
        c_down = self.ops.local_c("down", False)
        n_down = cd_down @ c_down

        occ_site_down = self.id_op()
        occ_site_up = self.id_op()

        occ_site_down[site][:, :] = n_down
        occ_site_up[site][:, :] = n_up

        return self.add_mpos(occ_site_up, occ_site_down)

    def inverse_op(self):
        """ """
        pass

    def one_rdm(self):
        """
        The 1-RDM matrix.
        """
        L = self.nr_sites
        gamma_pq = np.zeros((L, L))
        for p in range(L):
            for q in range(L):
                pq_up = self._construct_twosite_operator(
                    (p, "up"), (q, "up"), virtual_bonds=True
                )
                pq_down = self._construct_twosite_operator(
                    (p, "down"), (q, "down"), virtual_bonds=True
                )
                pq = self.add_mpos(pq_up, pq_down)
                gamma_pq[p, q] = self.get_expectation_value(pq)
        return gamma_pq
