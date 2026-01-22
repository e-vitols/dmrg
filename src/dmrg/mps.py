import numpy as np


class MpsDriver:
    """
    Implements the MatrixProductState object.

    Instance variables:
        - nr_sites: The size of the system -- number of sites/orbitals.
        - max_bond_dim: The maximum bond-dimension allowed.
        - tolerance: The tolerance for discarding singular values/Schmidt coefficients.
        - local_dim: Local dimension of the sites (only uniform is allowed), i.e., if local_dim = 4,
                     4 allowed occupations of the site: 0, alpha, beta, or alpha+beta.
    """

    def __init__(self):
        """
        Initializes the MatrixProductState object.
        """
        # Settings specifying the
        self.nr_sites = None
        self.max_bond_dim = None
        self.tolerance = 1e-9
        self.local_dim = None

        self.mps = None
        self.canonical_center = None

    def _initialize_random_mps(self):
        """
        Initializes the MatrixProductState object with random coefficients.

        :return mps:
            Returns a matrix-product state (MPS) with random coefficients.
        """
        L = self.nr_sites
        d = self.local_dim
        m = self.max_bond_dim

        mps = []

        # Left-most matrix (row-vector)
        m_l = 1
        m_r = max(1, min(m, d))
        A = np.random.randn(m_l, d, m_r) + 1j * np.random.randn(m_l, d, m_r)
        mps.append(A)

        # The middle sites (matrices) are built iteratively, taking the shape of the previous sites right dimension for the new left
        for i in range(1, L - 1):
            m_l = mps[-1].shape[2]
            m_r = max(1, min(m, d))
            A = np.random.randn(m_l, d, m_r) + 1j * np.random.randn(m_l, d, m_r)
            mps.append(A)

        # Right-most matrix (column-vector)
        m_l = mps[-1].shape[2]
        m_r = 1
        A = np.random.randn(m_l, d, m_r) + 1j * np.random.randn(m_l, d, m_r)
        mps.append(A)

        self.mps = mps

    def full_norm(self):
        """
        Gets the norm of the MPS.
        """
        env = np.array([[1.0]])
        for A in self.mps:
            env = np.einsum("lL, ldr, LdR -> rR", env, A, A.conjugate())

        return np.sqrt(env.squeeze().real)

    def canonical_norm(self):
        """
        Gets the norm of the MPS assuming it is in canonical form.

        :return norm:
            Returns the norm from
        """
        if self.canonical_center is None:
            raise ValueError("Requires the MPS in canonical form!")
        # return np.einsum(
        #    "ldr, ldr",
        #    self.mps[self.canonical_center],
        #    self.mps[self.canonical_center].conjugate(),
        # ).real
        A = self.mps[self.canonical_center]
        return np.sqrt(np.vdot(A, A).real)

    @staticmethod
    def overlap(mps1, mps2):
        """
        Gets the overlap squared of mps1 and mps2.
        """
        env = np.array([[1.0]], dtype=complex)
        if len(mps1) != len(mps2):
            raise ValueError("The MPSs are of different lengths!!")
        for i, _ in enumerate(mps1):
            env = np.einsum("lL, ldr, LdR -> rR", env, mps1[i], mps2[i].conjugate())

        return env.squeeze()

    def normalize(self):
        """
        Returns the normalized MPS, assumed being in canonical form.
        """
        self.mps[self.canonical_center] /= self.canonical_norm()

        return self.mps

    def canonicalize_mps(self, center=None, mps=None):
        """
        Puts the mps into canonical form on site 'center' with repeated singular-value decomposition.

        :param center:
            The site on which the MPS is canoncalized with respect to.
        :param mps:
            The mps on which to operate. If mps=None, i.e., no mps is supplied, default to the internal self.mps, otherwise act on the supplied mps
        """
        if center is None and self.canonical_center is not None:
            center = self.canonical_center

        allowed_center = 0 <= center <= self.nr_sites
        if allowed_center is not True:
            raise ValueError(
                "The site to center on must be within the number of sites!"
            )

        if mps is None:
            if self.mps is None:
                raise ValueError("MPS is not initialized!")
            self.mps = self._canonicalize_mps(self.mps, center)
            self.canonical_center = center
            return self.mps
        else:
            return self._canonicalize_mps(mps, center)

    @staticmethod
    def _canonicalize_mps(mps, center):
        L = len(mps)
        d = mps[0].shape[1]

        # Left canonicalize up to center site, left sweep
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

    def left_boundary(self, mps, mpo, center=None, mps2=None):
        """
        Gets the left boundary of the MPS up to the canonical center (assuming MPS is in canonical form).
        """

        if center is None:
            # TODO: make this part automatic
            center = mps.canonical_center

        if mps2 is None:
            mps2 = mps

        left_boundary = np.array([[[1.0]]])

        for l in range(center):
            N, M, W = mps[l], mps2[l], mpo[l]
            left_boundary = np.einsum(
                "ldL, vbdc, LvR, Rcr -> lbr", N.T.conjugate(), W, left_boundary, M
            )

        return left_boundary

    def right_boundary(self, mps, mpo, center=None, mps2=None):
        """
        Gets the right boundary of the MPS up to the canonical center (assuming MPS is in canonical form).
        """

        if center is None:
            # TODO: make this part automatic
            center = mps.canonical_center

        if mps2 is None:
            mps2 = mps

        right_boundary = np.array([[[1.0]]])
        for l in range(self.nr_sites - 1, center, -1):
            N, M, W = mps[l], mps2[l], mpo[l]
            right_boundary = np.einsum(
                "ldL, bvcd, LvR, Rcr -> lbr", M, W, right_boundary, N.T.conjugate()
            )

        return right_boundary
