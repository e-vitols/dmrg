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
        self.ops_drv = OperatorDriver(settings)

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
        Constructs a general MPO from operator strings and coefficients, in a 'naive' way.
        Returns MPO list W[i] with shapes (l,r,d,d).
        """
        T = len(opstrings)
        L = len(opstrings[0])
        d = opstrings[0][0].shape[0]

        W = [None for _ in range(L)]
        W[0] = np.zeros((1, T, d, d))
        for i in range(1, L - 1):
            W[i] = np.zeros((T, T, d, d))
        W[-1] = np.zeros((T, 1, d, d))

        for t, (ops, coeff) in enumerate(zip(opstrings, coeffs)):
            W[0][0, t] = coeff * ops[0]
            for i in range(1, L - 1):
                W[i][t, t] = ops[i]
            W[-1][t, 0] = ops[-1]
        return W

    def _construct_twosite_operator(
        self, creat_ind_spin, annih_ind_spin, virtual_bonds=False
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

        creat_op = self.ops_drv.dress_JW(creat_ind, creat_spin, True)
        annih_op = self.ops_drv.dress_JW(annih_ind, annih_spin, False)

        if virtual_bonds:
            creat_op = [site_op[np.newaxis, np.newaxis] for site_op in creat_op]
            annih_op = [site_op[np.newaxis, np.newaxis] for site_op in annih_op]

        # Overall operator = creat_op * annih_op  (rightmost acts first on a ket)
        return [A @ B for A, B in zip(creat_op, annih_op)]

    def _construct_foursite_operator(
        self, creat_ind_spin, creat_ind_spin_p, annih_ind_spin_p, annih_ind_spin
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

        creat_op = self.ops_drv.dress_JW(creat_ind, creat_spin, True)
        annih_op = self.ops_drv.dress_JW(annih_ind, annih_spin, False)
        creat_op_p = self.ops_drv.dress_JW(creat_ind_p, creat_spin_p, True)
        annih_op_p = self.ops_drv.dress_JW(annih_ind_p, annih_spin_p, False)

        # Overall operator = creat_op *creat_op * annih_op * annih_op  (rightmost acts first on a ket)
        return [
            A @ B @ C @ D
            for A, B, C, D in zip(creat_op, creat_op_p, annih_op_p, annih_op)
        ]

    def onsite_sum_mpo(self, operator, coupling=1):
        """
        Implements the general operator for onsite sums, e.g., number operator or chemical pot.
        """
        L = self.nr_sites

        # n = coupling * (self.ops_drv.number_total_local())
        I = self.ops_drv.identity_local()
        zero = self.ops_drv.zero_local()

        mpo = []

        # Left-most matrix (row-vector)
        W_0 = coupling * np.array([I, operator])[np.newaxis]
        mpo.append(W_0)

        W_i = np.array([[I, operator], [zero, I]])
        for i in range(1, L - 1):
            mpo.append(W_i.copy())

        # Right-most matrix (column-vector)
        W_L = np.array([operator, I])[:, np.newaxis]
        mpo.append(W_L)

        return mpo

    def local_mpo(self, operator, site):
        """ """
        mpo = self.identity_mpo()
        mpo[site][:, :] = operator
        return mpo

    def number_mpo(self, spin: str):
        """
        Implements the number operator for a specific spin sector.
        """
        operator = self.ops_drv.number_local(spin)
        return self.onsite_sum_mpo(operator)

    def total_number_mpo(self):
        """
        Get the total number operator.
        """
        up = self.number_mpo("up")
        dn = self.number_mpo("up")
        return self.add_mpos(up, dn)

    def chem_pot_mpo(self, coupling=1):
        """
        Implements the chemical potential operator.
        """
        L = self.nr_sites

        n = coupling * (self.ops_drv.number_total_local())
        I = self.ops_drv.identity_local()
        zero = self.ops_drv.zero_local()

        mpo = []

        # Left-most matrix (row-vector)
        W_0 = np.array([I, n])[np.newaxis]
        mpo.append(W_0)

        W_i = np.array([[I, n], [zero, I]])
        for i in range(1, L - 1):
            mpo.append(W_i.copy())

        # Right-most matrix (column-vector)
        W_L = np.array([n, I])[:, np.newaxis]
        mpo.append(W_L)

        return mpo

    def _chem_pot_mpo(self, coupling=1):
        """
        Implements the chemical potential operator.
        """
        operator = self.ops_drv.number_total_local()
        return self.onsite_sum_mpo(operator, coupling)

    def identity_mpo(self):
        """
        Implements the identity operator, mainly for debugging purposes.
        """
        L = self.nr_sites

        I = self.ops_drv.identity_local()

        mpo = []
        # Left-most matrix (row-vector)
        W_0 = np.array([I])
        mpo.append(W_0[np.newaxis])

        W_i = np.array([I])
        for i in range(1, L - 1):
            mpo.append(W_i[np.newaxis])

        # Right-most matrix (column-vector)
        W_L = np.array([I])
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

    def hubbard_mpo_naive(self, t=1.0, U=0.0, mu=0.0):
        """
        Implements the Hubbard Hamiltonian, including chemical potential, as an MPO built naively.
        NOTE this seems faster than fixed bon ddim for ~few sites
        """
        opstrings, coeffs = self.ops_drv.hubbard_opstrings(t=t, U=U, mu=mu)
        return self.mpo_from_opstrings(opstrings, coeffs)

    def hubbard_mpo(self, t=1.0, U=0.0, mu=0.0):
        """
        Implements the Hubbard Hamiltonian, including chemical potential, as an MPO built with fixed bond dimension -- independent of nr of sites.
        NOTE this seems slower than the naive MPO for ~few sites
        """
        L = self.nr_sites
        d = self.local_dim

        I = self.ops_drv.identity_local()
        c_up = self.ops_drv.local_c("up", False)
        cd_up = self.ops_drv.local_c("up", True)
        c_dn = self.ops_drv.local_c("down", False)
        cd_dn = self.ops_drv.local_c("down", True)
        n = self.ops_drv.number_total_local()
        n_dbl = self.ops_drv.double_occ_local()
        P = self.ops_drv.parity_local()

        # Onsite term
        h_loc = (U * n_dbl) - (mu * n)

        # MPO-bond dimension
        D = 6

        W = [None for _ in range(L)]
        # row vector: (1,D,d,d)
        W0 = np.zeros((1, D, d, d), dtype=np.float64)
        W0[0, 0] = I
        W0[0, 1] = cd_up @ P
        W0[0, 2] = P @ c_up
        W0[0, 3] = cd_dn @ P
        W0[0, 4] = P @ c_dn
        W0[0, 5] = h_loc
        W[0] = W0

        # middle sites/mats: (D,D,d,d)
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

    def site_occ_mpo(self, site):
        """
        Get the site occupation number operator.
        """
        local_up_op = self.ops_drv.number_local("up")
        local_dn_op = self.ops_drv.number_local("down")
        occ_site_up = self.local_mpo(local_up_op, site)
        occ_site_dn = self.local_mpo(local_dn_op, site)

        return self.add_mpos(occ_site_up, occ_site_dn)

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
