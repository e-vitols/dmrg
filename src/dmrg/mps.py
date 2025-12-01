import numpy as np


class MatrixProductState:
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
        # TODO: change class-name to MatrixProducStateConstructor/MpsConstructor
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
        env = np.array([[1.0]], dtype=complex)
        for A in self.mps:
            env = np.einsum("lL, ldr, LdR -> rR", env, A, A.conjugate())

        return env.squeeze().real

    def canonical_norm(self):
        """
        Gets the norm of the MPS assuming canonical form.
        """
        if self.canonical_center is None:
            raise ValueError("Requires the MPS in canonical form!")
        return np.einsum(
            "ldr, LDR",
            self.mps[self.canonical_center],
            self.mps[self.canonical_center].conjugate(),
        )

    def canonicalize_mps(self, center, mps=None):
        """
        Puts the mps into canonical form on site 'center' with repeated singular-value decomposition.

        :param center:
            The site on which the MPS is canoncalized with respect to.
        :param mps:
            The mps on which to operate. If mps=None, i.e., no mps is supplied, default to the internal self.mps, otherwise act on the supplied mps
        """

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

        # Left canonicalize up to center site
        for l in range(center - 1):
            # Get bond dimension at site l
            m_l, _, m_r = mps[l].shape

            # Reshape the 3-legged tensor at site l into a matrix
            A_l = mps[l].reshape(m_l * d, m_r)

            # Perform SVD on the reshaped 3-legged tensor (matrix A_l)
            U, S, Vh = np.linalg.svd(A_l, full_matrices=False)
            r = S.shape[0]

            # Get the renormalized basis right-transformation matrix
            G = np.diag(S) @ Vh

            # Replace the old tensor at site l with the left-canonicalized
            mps[l] = U.reshape(m_l, d, r)

            # Transform into the renormalized/canonical (depending on if r < m_r) basis
            mps_next_site = mps[l + 1].copy()
            mps[l + 1] = np.einsum("lr, rdm -> ldm", G, mps_next_site)

        for l in range(L - 1, center, -1):
            # Get bond dimension at site l
            m_l, _, m_r = mps[l].shape

            # Reshape the 3-legged tensor at site l into a matrix
            A_l = mps[l].reshape(m_l, d * m_r)

            # Perform SVD on the reshaped 3-legged tensor (matrix A_l)
            U, S, Vh = np.linalg.svd(A_l, full_matrices=False)
            r = S.shape[0]

            # Get the renormalized basis left-transformation matrix
            G = U @ np.diag(S)

            # Replace the old tensor at site l with the right-canonicalized
            mps[l] = Vh.reshape(r, d, m_r)

            # Transform into the renormalized/canonical (depending on if r < m_l) basis
            mps_next_site = mps[l - 1].copy()

            mps[l - 1] = np.einsum("mdl, lr -> mdr", mps_next_site, G)

        return mps
