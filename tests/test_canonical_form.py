import numpy as np
import pytest

import dmrg
from dmrg.mpo import MpoDriver
from dmrg.mps import MpsDriver


class TestCanonicalize:
    def test_left_canonicalize(self, local_dim=4, m_bonddim=8, nr_sites=6):
        # tt = mps.MatrixProductState()
        tt = MpsDriver()
        tt.local_dim = local_dim
        tt.max_bond_dim = m_bonddim
        tt.nr_sites = nr_sites
        tt._initialize_random_mps()
        mps = tt.mps
        tt.canonicalize_mps(5)

        orthonorm = np.einsum("ldr, ldR -> rR", tt.mps[0], tt.mps[0].conjugate())

        assert np.max(np.abs(orthonorm - np.eye(4))) < 1e-10

    def test_canonicalize(self, local_dim=4, m_bonddim=8, nr_sites=6):
        # tt = mps.MatrixProductState()
        tt = MpsDriver()
        tt.local_dim = local_dim
        tt.max_bond_dim = m_bonddim
        tt.nr_sites = nr_sites
        tt._initialize_random_mps()

        norm_before = tt.full_norm()

        for center in range(nr_sites):
            tt.canonicalize_mps(center)

            norm = tt.full_norm()

            rel_err = abs(norm - norm_before) / norm_before
            assert rel_err < 1e-10

            canonical_norm = tt.canonical_norm()

            rel_err = abs(canonical_norm - norm_before) / norm_before
            assert rel_err < 1e-10

            for i in range(center):
                A = tt.mps[i]
                m_l, d, m_r = A.shape
                left_metric = np.einsum("ldr, ldR -> rR", A.conjugate(), A).real
                assert np.max(np.abs(left_metric - np.eye(m_r))) < 1e-10

            for i in range(center + 1, nr_sites):
                A = tt.mps[i]
                m_l, d, m_r = A.shape
                right_metric = np.einsum("ldr, Ldr -> lL", A, A.conjugate()).real
                assert np.max(np.abs(right_metric - np.eye(m_l))) < 1e-10
