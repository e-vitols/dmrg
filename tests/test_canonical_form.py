import numpy as np
import pytest

import dmrg
from dmrg import mps
from dmrg.mps import MatrixProductState


class TestCanonicalize:
    def test_right_canonicalize(self, local_dim=4, m_bonddim=8, nr_sites=6):
        # tt = mps.MatrixProductState()
        tt = MatrixProductState()
        tt.local_dim = local_dim
        tt.max_bond_dim = m_bonddim
        tt.nr_sites = nr_sites
        tt._initialize_random_mps()
        mps = tt.mps
        tt.canonicalize_mps(6)

        orthonorm = np.einsum("ldr, rdl -> lr", tt.mps[0], tt.mps[0].conjugate())

        assert np.max(np.abs(orthonorm - np.eye(4))) < 1e-10

    def test_left_canonicalize(self, local_dim=4, m_bonddim=8, nr_sites=6):
        # tt = mps.MatrixProductState()
        tt = MatrixProductState()
        tt.local_dim = local_dim
        tt.max_bond_dim = m_bonddim
        tt.nr_sites = nr_sites
        tt._initialize_random_mps()
        mps = tt.mps
        tt.canonicalize_mps(0)

        orthonorm = np.einsum("ldr, rdl -> lr", tt.mps[-1], tt.mps[-1].conjugate())

        assert np.max(np.abs(orthonorm - np.eye(4))) < 1e-10
