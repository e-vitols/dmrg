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
        # Settings specifying the
        self.nr_sites = None
        self.max_bond_dim = None
        self.tolerance = 1e-9
        self.local_dim = None

        self.mps = None

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

        #if self.max_bond_dim:
            # mps.shape = (site, local_dim, bond_i, bond_j)
        #self.mps = np.random.rand(
        #    (self.nr_sites, self.local_dim, self.max_bond_dim, self.max_bond_dim)
        #)

        # Left-most matrix (row-vector)
        m_l = 1
        m_r = max(1, min(m, d))
        A = (np.random.randn(m_l, d, m_r)
             + 1j * np.random.randn(m_l, d, m_r))
        mps.append(A)

        # The middle sites (matrices) are built iteratively, taking the shape of the previous sites right dimension for the new left
        for i in range(1, L - 1):
            m_l = mps[-1].shape[2]
            m_r = max(1, min(m, d))
            A = (np.random.randn(m_l, d, m_r)
                + 1j * np.random.randn(m_l, d, m_r))
            mps.append(A)

        # Right-most matrix (column-vector)
        m_l = mps[-1].shape[2]
        m_r = 1
        A = (np.random.randn(m_l, d, m_r)
             + 1j * np.random.randn(m_l, d, m_r))
        mps.append(A)

        self.mps = mps

    def canonicalize_mps(self, center):
        """
        Puts the mps into canonical form on site 'center' with repeated singular-value decomposition.

        :param center:
            The site on which the MPS is canoncalized with respect to.
        """
        
        


        