import numpy as np


class MpsDriver:
    """
    Implements the MatrixProductState object.
    # NOTE MPS structure is: (chiL, d, chiR)

    Instance variables:
        - nr_sites: The size of the system -- number of sites/orbitals. (int)
        - max_bond_dim: The maximum bond-dimension allowed. (int)
        - tolerance: The tolerance for discarding singular values/Schmidt coefficients. (float)
        - local_dim: Local dimension of the sites (only uniform is allowed), i.e., if local_dim = 4, 4 allowed occupations of the site: 0, up/alpha, down/beta, or alpha+beta. (int)
    """

    def __init__(self, settings):
        """
        Initializes the MatrixProductState object.

        :param settings:
            Sets the instance variables of the class.
        """
        # Settings specifying the
        self.settings = settings

        self.nr_sites = settings.nr_sites
        self.max_bond_dim = settings.max_bond_dim
        self.tolerance = settings.svd_thr  # not yet implemented/interfaced
        self.local_dim = settings.local_dim
        self.N = settings.nr_particles

        self.mps = None
        self.canonical_center = None
        self.schmidt_spectrum = None
        self.discarded_weight = None
        self.dtype = np.float64

        self.q_phys = np.array([0, 1, 1, 2], dtype=int)
        self.g_degen = {0: 1, 1: 2, 2: 1}
        self.q_bonds = None

    def initialize_random_mps(self, complex=False):
        """
        Initializes the MatrixProductState object with random coefficients.

        :return mps:
            Returns a matrix-product state (MPS) with random (complex) coefficients. (list of arrays)
        """
        L = self.nr_sites
        d = self.local_dim
        m = self.max_bond_dim
        if complex:
            self.dtype = np.complex128
        else:
            self.dtype = np.float64

        mps = []

        def chi(i):
            return max(1, min(m, d ** (i + 1), d ** (L - (i + 1))))

        def rand_tensor(m_l, d, m_r):
            if self.dtype == np.complex128:
                return (
                    np.random.randn(m_l, d, m_r) + 1j * np.random.randn(m_l, d, m_r)
                ).astype(np.complex128)
            else:
                return np.random.randn(m_l, d, m_r).astype(np.float64)

        # Left-most matrix (row-vector)
        m_l = 1
        m_r = chi(0)
        mps.append(rand_tensor(m_l, d, m_r))

        # The middle sites (matrices) are built iteratively, taking the shape of the previous sites right dimension for the new left
        for i in range(1, L - 1):
            m_l = mps[-1].shape[2]
            m_r = chi(i)
            mps.append(rand_tensor(m_l, d, m_r))

        # Right-most matrix (column-vector)
        m_l = mps[-1].shape[2]
        m_r = 1
        mps.append(rand_tensor(m_l, d, m_r))

        self.mps = mps

    def initialize_fixed_mps(self, complex=False):
        """
        Initializes the MatrixProductState object with fixed coefficients (0.5+0.5i).

        :return:
            Returns a matrix-product state (MPS) with fixed coefficients. (list of arrays)
        """
        L = self.nr_sites
        d = self.local_dim
        m = self.max_bond_dim
        if complex:
            self.dtype = np.complex128
        else:
            self.dtype = np.float64

        mps = []

        def chi(i):
            return max(1, min(m, d ** (i + 1), d ** (L - (i + 1))))

        # Left-most matrix (row-vector)
        m_l = 1
        m_r = chi(0)
        if complex:
            A = np.full((m_l, d, m_r), 0.5 + 0.5j, dtype=self.dtype)
        else:
            A = np.full((m_l, d, m_r), 0.5, dtype=self.dtype)
        mps.append(A)

        # The middle sites (matrices) are built iteratively, taking the shape of the previous sites right dimension for the new left
        for i in range(1, L - 1):
            m_l = mps[-1].shape[2]
            m_r = chi(i)
            if complex:
                A = np.full((m_l, d, m_r), 0.5 + 0.5j, dtype=self.dtype)
            else:
                A = np.full((m_l, d, m_r), 0.5, dtype=self.dtype)
            mps.append(A)

        # Right-most matrix (column-vector)
        m_l = mps[-1].shape[2]
        m_r = 1
        # A = np.full((m_l, d, m_r), 0.5 + 0.5j, dtype=np.complex128)
        if complex:
            A = np.full((m_l, d, m_r), 0.5 + 0.5j, dtype=self.dtype)
        else:
            A = np.full((m_l, d, m_r), 0.5, dtype=self.dtype)
        mps.append(A)

        self.mps = mps

    def initialize_u1_mps(self, complex=False):
        """
        Initializes the MatrixProductState object in a fixed particle-number sector, as a simple product state.

        :return:
            Returns a product-state matrix-product state (MPS) with fixed particle number. (list of arrays)
        """
        L = self.nr_sites
        d = self.local_dim
        m = self.max_bond_dim
        N = self.N
        if N < 0 or N > 2 * L:
            raise ValueError(f"N must satisfy 0 <= N <= {2*L}")
        if complex:
            self.dtype = np.complex128
        else:
            self.dtype = np.float64

        def chi(i):
            return max(1, min(m, d ** (i + 1), d ** (L - (i + 1))))

        # multiplicty, degenerate single-occ.
        g = self.g_degen  # {0: 1, 1: 2, 2: 1}
        mu = {0: 0, 1: 0, 2: 1, 3: 0}

        # (qL, g, qR)
        mps = []

        q_phys = self.q_phys  # np.array([0, 1, 1, 2], dtype=int)
        q_bonds = []
        q_bonds.append({0: 1})

        N_left = N
        N_curr = 0
        for i in range(L):
            if i == 0:
                m_l = 1
            m_r = chi(i)

            if N_left >= 2:
                state = 3
                N_left -= 2

            elif N_left >= 1:
                state = 1
                N_left -= 1

            else:
                state = 0

            q_L = N_curr
            q_P = int(q_phys[state])
            q_R = q_L + q_P

            block = np.zeros((m_l, g[q_P], m_r), dtype=self.dtype)
            block[0, mu[state], 0] = 1.0
            mps.append({(q_L, q_R): block})

            q_bonds.append({q_R: 1})

            N_curr = q_R
            m_l = m_r

        # self.q_phys = q_phys
        self.q_bonds = q_bonds
        self.mps = mps

    def canonical_form(self, center=None, mps=None, factorization="QR"):
        """
        Puts the MPS into canonical form on site 'center' via repeated singular-value decomposition.

        :param center:
            The site on which the MPS is canoncalized with respect to, also known as orthogonality center. (integer)
        :param mps:
            The MPS on which to operate. If mps=None, default to the internal self.mps, otherwise act on the supplied MPS. (list of arrays)

        :return:
            The MPS in canonical form. (list of arrays)
        """
        if center is None and self.canonical_center is not None:
            center = self.canonical_center

        allowed_center = 0 <= center <= self.nr_sites - 1
        if allowed_center is not True:
            raise ValueError(
                "The site to center on must be within the number of sites!"
            )

        if mps is None:
            self.canonical_center = center
            if self.mps is None:
                raise ValueError("MPS is not initialized!")

            if factorization.upper() == "QR":
                self.mps = self._canonicalize_QR(self.mps, center)
            elif factorization.upper() == "SVD":
                self.mps = self._canonicalize_SVD(self.mps, center)
            self.schmidt_spectrum = self.get_schmidt_spectrum(self.mps, center)
            return self.mps
        else:
            if factorization.upper() == "QR":
                self.mps = self._canonicalize_QR(mps, center)
            elif factorization.upper() == "SVD":
                self.mps = self._canonicalize_SVD(mps, center)
            # return self._canonicalize_QR(mps, center)
            # return self._canonicalize_SVD(mps, center)

    @staticmethod
    def _canonicalize_SVD(mps, center):
        """
        Implements canonicalization of an MPS with SVD-factorization.

        :param mps:
            The matrix-product state object (list of numpy arrays).
        :param center:
            The SITE-based canonical center.

        :return:
            The MPS in canonical form. (list of arrays)
        """

        L = len(mps)  # self.nr_sites
        d = mps[0].shape[1]

        # Left canonicalize up to center site, right sweep
        for l in range(center):
            # Get bond dimension at site l
            m_l, _, m_r = mps[l].shape

            # Reshape the 3-legged tensor at site l into a matrix
            A_l = mps[l].reshape(m_l * d, m_r)

            # Perform SVD on the reshaped 3-legged tensor (matrix A_l)
            U, S, Vh = np.linalg.svd(A_l, full_matrices=False)
            chi = S.shape[0]

            # Replace the old tensor at site l with the left-canonicalized
            mps[l] = U.reshape(m_l, d, chi)

            # Get the renormalized basis right-transformation matrix
            G = np.diag(S) @ Vh

            # Transform to the renormalized/canonical (depending on if r < m_r) basis
            mps_next_site = mps[l + 1].copy()
            mps[l + 1] = np.einsum("lr, rdm -> ldm", G, mps_next_site)

        # Right canonicalize up to center site, left sweep
        for l in range(L - 1, center, -1):
            # Get bond dimension at site l
            m_l, _, m_r = mps[l].shape

            # Reshape the 3-legged tensor at site l into a matrix
            A_l = mps[l].reshape(m_l, d * m_r)

            # Perform SVD on the reshaped 3-legged tensor (matrix A_l)
            U, S, Vh = np.linalg.svd(A_l, full_matrices=False)
            chi = S.shape[0]

            # Replace the old tensor at site l with the right-canonicalized
            mps[l] = Vh.reshape(chi, d, m_r)

            # Get the renormalized basis left-transformation matrix
            G = U @ np.diag(S)

            # Transform into the renormalized/canonical (depending on if r < m_l) basis
            mps_next_site = mps[l - 1].copy()

            mps[l - 1] = np.einsum("mdl, lr -> mdr", mps_next_site, G)

        return mps

    @staticmethod
    def _canonicalize_QR(mps, center):
        """
        Implements canonicalization of an MPS with QR-factorization.

        :param mps:
            The matrix-product state object (list of numpy arrays).
        :param center:
            The SITE-based canonical center.

        :return:
            The MPS in canonical form. (list of arrays)
        """

        L = len(mps)  # self.nr_sites
        d = mps[0].shape[1]

        # Left canonicalize up to center site, right sweep
        for l in range(center):
            # Get bond dimension at site l
            m_l, _, m_r = mps[l].shape

            # Reshape the 3-legged tensor at site l into a matrix
            A_l = mps[l].reshape(m_l * d, m_r)

            # Perform QR factorization on the reshaped 3-legged tensor (matrix A_l)
            Q, R = np.linalg.qr(A_l)
            k = Q.shape[1]

            # Replace the old tensor at site l with the left-canonicalized
            mps[l] = Q.reshape(m_l, d, k)

            # Transform to the renormalized/canonical (depending on if r < m_r) basis
            mps_next_site = mps[l + 1].copy()
            mps[l + 1] = np.einsum("lr, rdm -> ldm", R, mps_next_site)

        # Right canonicalize up to center site, left sweep
        for l in range(L - 1, center, -1):
            # Get bond dimension at site l
            m_l, _, m_r = mps[l].shape

            # Reshape the 3-legged tensor at site l into a matrix
            A_l = mps[l].reshape(m_l, d * m_r)

            # Perform SVD on the reshaped 3-legged tensor (matrix A_l)
            Q, R = np.linalg.qr(A_l.T.conjugate())
            k = Q.shape[1]

            # Replace the old tensor at site l with the right-canonicalized
            mps[l] = Q.T.conjugate().reshape(k, d, m_r)

            # Transform into the renormalized/canonical (depending on if r < m_l) basis
            mps_next_site = mps[l - 1].copy()
            mps[l - 1] = np.einsum("mdl, lr -> mdr", mps_next_site, R.T.conjugate())

        return mps

    def full_norm(self):
        """
        Gets the norm of the MPS by fully contracting through the MPS.

        :return:
            The norm of the MPS (wavefunction). (float)
        """
        env = np.array([[1.0]])
        for A in self.mps:
            env = np.einsum("lL, ldr, LdR -> rR", env, A, A.conjugate())

        return np.sqrt(env.squeeze().real)

    def canonical_norm(self, mps=None):
        """
        Gets the norm of the MPS assuming it is in canonical form.

        :param mps:
            The matrix-product state object (list of numpy arrays).

        :return:
            The norm of the MPS (wavefunction). (float)
        """
        if mps is None:
            mps = self.mps
        if self.canonical_center is None:
            raise ValueError("Requires the MPS in canonical form!")
        # return np.einsum(
        #    "ldr, ldr",
        #    self.mps[self.canonical_center],
        #    self.mps[self.canonical_center].conjugate(),
        # ).real
        A = mps[self.canonical_center]
        return np.sqrt(np.vdot(A, A).real)

    # @staticmethod
    def overlap(self, mps1, mps2):
        """
        Gets the overlap squared of two MPSs: mps1 and mps2.

        :param mps1:
            The ket MPS. (list of arrays)
        :param mps2:
            The ket MPS. (list of arrays)

        :return:
            The overlap between mps1 and mps2. (float)
        """
        # Environment defined to iteratively contract
        env = np.array([[1.0]], dtype=self.dtype)
        if len(mps1) != len(mps2):
            raise ValueError("The MPSs are of different lengths!!")
        for i, _ in enumerate(mps1):
            env = np.einsum("lL, ldr, LdR -> rR", env, mps1[i], mps2[i].conjugate())

        return env.squeeze()

    def normalize(self, mps=None, center=None):
        """
        Normalizese the MPS.

        :param mps:
            The matrix-product state object (list of numpy arrays).

        :return:
            The normalized MPS. (list of arrays)
        """
        if mps is None:
            mps = self.mps
        if center is None:
            center = self.canonical_center

        mps[center] /= self.canonical_norm()
        self.mps[center] = mps[center]

        return self.mps

    def left_boundary(self, mpo, mps=None, center=None, mps2=None):
        """
        Gets the left environment/boundary of the MPS with MPO up to the canonical center (assuming MPS is in canonical form).

        :param mpo:
            The matrix-product operator object (list of numpy arrays).
        :param mps:
            The matrix-product state object (list of numpy arrays).
        :param mps2:
            The matrix-product state object 2, defaults to mps if None. (list of numpy arrays)
        :param center:
            The canonical center, defaults to the set center if None. (int)

        :return:
            Left boundary/environment, representing the effectve operator acting on those sites. (list of arrays)
        """
        if mps is None:
            mps = self.mps

        if center is None:
            # TODO: make this part automatic
            center = self.canonical_center

        if mps2 is None:
            mps2 = mps

        left_boundary = np.array([[[1.0]]])

        for l in range(center):
            N, M, W = mps[l], mps2[l], mpo[l]
            left_boundary = np.einsum(
                "Ldl, vbdc, LvR, Rcr -> lbr",
                N.conjugate(),
                W,
                left_boundary,
                M,
                optimize=True,
            )

        return left_boundary

    def right_boundary(self, mpo, mps=None, center=None, mps2=None):
        """
        Gets the right boundary of the MPS down to the canonical center (assuming MPS is in canonical form).

        :param mpo:
            The matrix-product operator object (list of numpy arrays).
        :param mps:
            The matrix-product state object (list of numpy arrays).
        :param mps2:
            The matrix-product state object 2, defaults to mps if None. (list of numpy arrays)
        :param center:
            The canonical center, defaults to the set center if None. (int)

        :return:
            Right boundary/environment, representing the effectve operator acting on those sites. (list of arrays)
        """
        if mps is None:
            mps = self.mps
        if center is None:
            # TODO: make this part automatic
            center = self.canonical_center

        if mps2 is None:
            mps2 = mps

        right_boundary = np.array([[[1.0]]], dtype=self.dtype)
        for l in range(self.nr_sites - 1, center, -1):
            N, M, W = mps[l], mps2[l], mpo[l]
            right_boundary = np.einsum(
                "ldL, bvcd, LvR, rcR -> lbr",
                M,
                W,
                right_boundary,
                N.conjugate(),
                optimize=True,
            )

        return right_boundary

    def get_expectation_value(self, mpo, center=None):
        """
        Get the expectation value of an operator (given as an MPO).

        :param mpo:
            The operator as an MPO. (list of arrays)
        :param center:
            The canonical center, defaults to the set center if None. (int)

        :return:
            The expectation value of the MPO. (float)
        """

        if center is None:
            center = self.canonical_center

        left_boundary = self.left_boundary(mpo, center=center + 1)
        right_boundary = self.right_boundary(mpo, center=center)

        # TODO: replace einsum
        exp_val = np.einsum("ldr, rdl", left_boundary, right_boundary)
        # exp_val = np.einsum("ldr, ldr", left_boundary, right_boundary)
        if np.abs(exp_val.imag) < 1e-14:
            return exp_val.real
        else:
            return exp_val

    @staticmethod
    def get_schmidt_spectrum(mps, center: int):
        """
        Gets the schmidt spectrum of the Schmidt decomposition at the bond between site center and center+1.

        :param mps:
            The matrix-product state objec. (list of numpy arrays)
        :param center:
            Defines the bond at which the decomposition is performed, defined to be between site center and center+1. (int)

        :return:
            Singular values. (vector/array)
        """
        # care must be taken if thsi method is used outside where it is currently called
        # the center must be the canonical_center of the canonicalized MPS
        m_l, d, m_r = mps[center].shape
        U, S, Vh = np.linalg.svd(mps[center].reshape(m_l * d, m_r), full_matrices=False)
        return S

    def bipartite_entang_entropy(self, center: int, mps=None):
        """
        Gets the entanglement entropy at the bond between site center and center+1, defined as the von Neumann entropy.

        :param center:
            Defines the bond at which the decomposition is performed, defined to be between site center and center+1. (int)

        :return:
            Entanglement entropy at the bond between site i (=center) and site i+1 (=center+1). (float)
        """
        if mps is None:
            mps = self.mps
        self.canonical_form(center)
        m_l, d, m_r = mps[center].shape

        U, S, Vh = np.linalg.svd(mps[center].reshape(m_l * d, m_r), full_matrices=False)
        lam = S**2
        ent_entropy = -1.0 * np.sum(lam * np.log2(lam))
        return ent_entropy

    def single_orbital_entropy(self, i):
        """
        Gets the single orbital entropy of site/orbital i.

        :param i:
            The site/orbital.

        :return:
            The single site/orbital entanglement entropy.
        """
        self.canonical_form(i)
        A = self.mps[i]

        rho = np.einsum("lar, lbr->ab", A, A.conj())
        rho /= np.trace(rho).real

        p = np.linalg.eigvalsh(rho).real
        return -np.sum(p * np.log2(p))

    def get_twosite(self, center=None, mps=None):
        """
        Gets the fused twosite tensor from the given MPS, enabling two-site optimization.

        :param center:
            The center defining the left site of the two-site tensor. (int)
        :param mps:
            The matrix-product state object (list of numpy arrays).

        :return:
            The two-site tensor. (array)
        """
        if mps is None:
            mps = self.mps

        if center is None:
            center = self.canonical_center

        allowed_center = 0 <= center <= self.nr_sites - 2
        if allowed_center is not True:
            raise ValueError(
                "The bond to center on must be within the number of sites-1!"
            )

        left_center = center
        right_center = center + 1
        # mps[center] = {(qL, qR): block}, qR is the middle-charge
        # mps[center+1] = {(qL, qR): block}, qL is the middle-charge
        A_L = mps[left_center]
        A_R = mps[right_center]

        # midL = {keys[1] for keys in A_L}
        # midR = {keys[0] for keys in A_R}

        theta = {}

        # TODO: this can be made more efficient
        for ql in A_L:
            for qr in A_R:
                if ql[1] == qr[0]:
                    qM = ql[1]

                    two_site_block = np.einsum(
                        "ldr, rDR -> ldDR",
                        A_L[ql],
                        A_R[qr],
                        optimize=True,
                    )
                    key = (ql[0], qM, qr[1])

                    # TODO: the if is probably not needed here
                    if key not in theta:
                        theta[key] = two_site_block
                    else:
                        theta[key] += two_site_block

        return theta

    def split_twosite(self, theta, direction, mps=None, center=None):
        """
        Splits a supplied two-site tensor into two one-site tensors.

        :param theta:
            The two-site tensor. (array)
        :param direction:
            The sweep direction. (str)
        :param mps:
            The matrix-product state object (list of numpy arrays).
        :param center:
            The center defining the left site of the two-site tensor. (int)

        :return:
            The new center and the MPS.
        """
        l, d1, d2, r = theta.shape
        new_canonical_center = center
        if center is None:
            new_canonical_center = self.canonical_center
        if mps is None:
            mps = self.mps

        U, S, Vh = np.linalg.svd(theta.reshape(l * d1, d2 * r), full_matrices=False)
        chi_full = S.size

        # truncate:
        chi = min(self.max_bond_dim, chi_full)
        # truncation error/schur complement
        S2 = S**2
        tot = S2.sum()
        disc = S2[chi:].sum()
        self.discarded_weight = disc / tot if tot > 0 else 0.0

        kept = S2[:chi].sum()
        kept_norm = np.sqrt(kept) if kept > 0 else 1.0
        S = S[:chi] / kept_norm
        U = U[:, :chi]
        Vh = Vh[:chi]

        if direction == "right":
            mps[center] = U.reshape(l, d1, chi)
            mps[center + 1] = (np.diag(S) @ Vh).reshape(chi, d2, r)
            new_canonical_center += 1

        elif direction == "left":
            mps[center] = (U @ np.diag(S)).reshape(l, d1, chi)
            mps[center + 1] = Vh.reshape(chi, d2, r)
            # new_canonical_center = max(center - 1, 0)
            new_canonical_center = max(center, 0)

        return new_canonical_center, mps

    def split_twosite_u1(self, theta, direction, mps=None, center=None):
        """
        Splits a supplied two-site tensor into two one-site tensors (blocks).

        :param theta:
            The two-site tensor. (array)
        :param direction:
            The sweep direction. (str)
        :param mps:
            The matrix-product state object (list of numpy arrays).
        :param center:
            The center defining the left site of the two-site tensor. (int)

        :return:
            The new center and the MPS.
        """
        if mps == None:
            mps = self.mps
        g = self.g_degen
        dim_L, dim_R = self.q_bonds[center], self.q_bonds[center + 2]
        # rows and cols correspond to the rows and cols of the supermat
        # indexed by qM, e.g.: rows[4] {2: slice(0, 1, None)} where the second
        # key refers to the left bond charge, qL. the same holds for the cols
        M_qM, rows, cols = self._get_supermat_by_qM(theta, dim_L, dim_R)

        # TODO: there is a memory-saving option by computing the SVD first
        # but computing only the singular-values: np.linalg.svd(a, compute_uv=False)
        # to use for truncation. alternatively, as is done below, store ALL matrices
        # for later truncation. should investigate and compare
        sing_val_info = []

        A_left_new = {}
        A_right_new = {}
        S_qm_full = {}
        for qM in M_qM:
            U, S, Vh = np.linalg.svd(M_qM[qM], full_matrices=False)
            # chi_full = S.size
            S_qm_full[qM] = S
            chi_qM = S.size  # min(self.max_bond_dim, chi_full)

            for k, S_i in enumerate(S):
                sing_val_info.append((S_i, qM, k))

            for qL, row_sl in rows[qM].items():
                dL = dim_L[qL]
                muL = g[qM - qL]

                # (dL*muL, chi_qM)
                U_part = U[row_sl, :]
                if direction == "right":
                    A_left_new[(qL, qM)] = U_part.reshape(dL, muL, chi_qM)
                elif direction == "left":
                    A_left_new[(qL, qM)] = (U_part * S[None, :]).reshape(
                        dL, muL, chi_qM
                    )

            # right blocks from (S*Vh): iterate qR slices
            if direction == "right":
                SVh = S[:, np.newaxis] * Vh
            elif direction == "left":
                SVh = Vh

            for qR, col_sl in cols[qM].items():
                dR = dim_R[qR]
                muR = g[qR - qM]

                # (chi_qM, muR*dR)
                V_part = SVh[:, col_sl]
                A_right_new[(qM, qR)] = V_part.reshape(chi_qM, muR, dR)

        sing_val_info.sort(key=lambda item: item[0], reverse=True)
        kept = sing_val_info[: self.max_bond_dim]

        chi_keep = {qM: 0 for qM in M_qM}
        for sv, qM, _ in kept:
            chi_keep[qM] += 1

        tot = 0.0
        disc = 0.0
        for qM, S in S_qm_full.items():
            S2 = S * S
            tot += S2.sum()
            disc += S2[chi_keep[qM] :].sum()
        self.discarded_weight = disc / tot if tot > 0 else 0.0

        kept_weight = tot - disc
        scale = (1.0 / np.sqrt(kept_weight)) if kept_weight > 0 else 1.0

        A_left_trunc = {}
        A_right_trunc = {}

        for qM in M_qM:
            chi = chi_keep[qM]
            if chi == 0:
                continue

            for qL in rows[qM].keys():
                block = A_left_new[(qL, qM)][:, :, :chi]
                if direction == "left":
                    block = block * scale
                A_left_trunc[(qL, qM)] = block

            for qR in cols[qM].keys():
                block = A_right_new[(qM, qR)][:chi, :, :]
                if direction == "right":
                    block = block * scale
                A_right_trunc[(qM, qR)] = block

        mps[center] = A_left_trunc
        mps[center + 1] = A_right_trunc

        self.q_bonds[center + 1] = {qM: chi_keep[qM] for qM in M_qM if chi_keep[qM] > 0}

        new_canonical_center = center
        if direction == "right":
            new_canonical_center += 1

        return new_canonical_center, mps

    def _get_supermat_by_qM(self, theta, dim_L, dim_R):
        """ """

        # sort by by middle-charge qM
        g = self.g_degen
        by_qM = {}
        for (qL, qM, qR), M in theta.items():
            by_qM.setdefault(qM, {})[(qL, qR)] = M

        M_qM = {}
        row_slices = {}
        col_slices = {}

        for qM, blocks in by_qM.items():
            qLs = sorted({qL for (qL, qR) in blocks.keys()})
            qRs = sorted({qR for (qL, qR) in blocks.keys()})

            # extract the row- and col sizes
            row_sizes = {qL: dim_L[qL] * g[qM - qL] for qL in qLs}
            col_sizes = {qR: g[qR - qM] * dim_R[qR] for qR in qRs}

            # keep track of the slices
            rs = {}
            off = 0
            for qL in qLs:
                rs[qL] = slice(off, off + row_sizes[qL])
                off += row_sizes[qL]
            cs = {}
            off = 0
            for qR in qRs:
                cs[qR] = slice(off, off + col_sizes[qR])
                off += col_sizes[qR]

            row_slices[qM] = rs
            col_slices[qM] = cs

            # construct M_qM
            row_strips = []
            for qL in qLs:
                sub_blocks = []
                for qR in qRs:
                    if (qL, qR) in blocks:
                        B = blocks[(qL, qR)]
                        dL, muL, muR, dR = B.shape
                        sub_blocks.append(B.reshape(dL * muL, muR * dR))
                    else:
                        sub_blocks.append(
                            np.zeros((row_sizes[qL], col_sizes[qR]), dtype=self.dtype)
                        )
                row_strips.append(np.concatenate(sub_blocks, axis=1))

            M_qM[qM] = np.concatenate(row_strips, axis=0)

        return M_qM, row_slices, col_slices
