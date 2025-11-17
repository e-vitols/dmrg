import numpy as np


class MatrixProductState:
    """
    Implements the MatrixProductState object.

    Instance variables:
        - nr_sites: The size of the system -- number of sites/orbitals.
        - max_bond_dim: The maximum bond-dimension allowed.
        - tolerance: The tolerance for discarding singular values/Schmidt coefficients.
        - local_dim: Local dimension of the sites (only uniform is allowed), i.e., if local_dim = 4,
                     4 allowed occupations of the site: 0, alpha, beta, or, alpha+beta.
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
        """
        if self.max_bond_dim:
            # (site, local_dim, bond_i, bond_j)
            self.mps = np.random.rand(
                (self.nr_sites, self.local_dim, self.max_bond_dim, self.max_bond_dim)
            )
