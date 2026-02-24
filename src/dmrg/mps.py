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

        self.mps = None
        self.canonical_center = None
        self.schmidt_spectrum = None
        self.discarded_weight = None

    def initialize_random_mps(self):
        """
        Initializes the MatrixProductState object with random coefficients.

        :return mps:
            Returns a matrix-product state (MPS) with random (complex) coefficients. (list of arrays)
        """
        L = self.nr_sites
        d = self.local_dim
        m = self.max_bond_dim

        mps = []

        def chi(i):
            return max(1, min(m, d ** (i + 1), d ** (L - (i + 1))))

        # Left-most matrix (row-vector)
        m_l = 1
        m_r = chi(0)
        A = np.random.randn(m_l, d, m_r) + 1j * np.random.randn(m_l, d, m_r)
        mps.append(A)

        # The middle sites (matrices) are built iteratively, taking the shape of the previous sites right dimension for the new left
        for i in range(1, L - 1):
            m_l = mps[-1].shape[2]
            m_r = chi(i)
            A = np.random.randn(m_l, d, m_r) + 1j * np.random.randn(m_l, d, m_r)
            mps.append(A.copy())

        # Right-most matrix (column-vector)
        m_l = mps[-1].shape[2]
        m_r = 1
        A = np.random.randn(m_l, d, m_r) + 1j * np.random.randn(m_l, d, m_r)
        mps.append(A)

        self.mps = mps

    def initialize_fixed_mps(self):
        """
        Initializes the MatrixProductState object with fixed coefficients (0.5+0.5i).

        :return:
            Returns a matrix-product state (MPS) with fixed coefficients. (list of arrays)
        """
        L = self.nr_sites
        d = self.local_dim
        m = self.max_bond_dim

        mps = []

        def chi(i):
            return max(1, min(m, d ** (i + 1), d ** (L - (i + 1))))

        # Left-most matrix (row-vector)
        m_l = 1
        m_r = chi(0)
        A = np.full((m_l, d, m_r), 0.5 + 0.5j, dtype=np.complex128)
        mps.append(A)

        # The middle sites (matrices) are built iteratively, taking the shape of the previous sites right dimension for the new left
        for i in range(1, L - 1):
            m_l = mps[-1].shape[2]
            m_r = chi(i)
            A = A = np.full((m_l, d, m_r), 0.5 + 0.5j, dtype=np.complex128)
            mps.append(A)

        # Right-most matrix (column-vector)
        m_l = mps[-1].shape[2]
        m_r = 1
        A = np.full((m_l, d, m_r), 0.5 + 0.5j, dtype=np.complex128)
        mps.append(A)

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

    @staticmethod
    def overlap(mps1, mps2):
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
        env = np.array([[1.0]], dtype=complex)
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

        right_boundary = np.array([[[1.0]]])
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
        two_site_tensor = np.einsum(
            "ldr, rDR -> ldDR",
            mps[left_center],
            mps[right_center],
            optimize=True,
        )
        l, d1, d2, r = two_site_tensor.shape

        return two_site_tensor.reshape(l, d1, d2, r)
        # return two_site_tensor.reshape(l, d1 * d2, r)

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
        # schur_complement = np.sum(S[chi:])
        self.discarded_weight = np.sum(S[chi:] ** 2)
        kept_norm = np.sqrt(np.sum(S[:chi] ** 2))
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
            new_canonical_center = max(center - 1, 0)

        return new_canonical_center, mps
