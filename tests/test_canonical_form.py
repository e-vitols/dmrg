import numpy as np
import pytest

import dmrg
from dmrg.mpo import MpoDriver
from dmrg.mps import MpsDriver


class TestCanonicalize:
    def test_left_canonicalize(self, local_dim=4, m_bonddim=8, nr_sites=6):
        mps_drv = MpsDriver()
        mps_drv.local_dim = local_dim
        mps_drv.max_bond_dim = m_bonddim
        mps_drv.nr_sites = nr_sites
        mps_drv._initialize_random_mps()
        mps = mps_drv.mps
        mps_drv.canonicalize_mps(5)

        orthonorm = np.einsum(
            "ldr, ldR -> rR", mps_drv.mps[0], mps_drv.mps[0].conjugate()
        )

        assert np.max(np.abs(orthonorm - np.eye(4))) < 1e-10

    def test_canonicalize(self, local_dim=4, m_bonddim=8, nr_sites=6):
        # tt = mps.MatrixProductState()
        mps_drv = MpsDriver()
        mps_drv.local_dim = local_dim
        mps_drv.max_bond_dim = m_bonddim
        mps_drv.nr_sites = nr_sites
        mps_drv._initialize_random_mps()

        norm_before = mps_drv.full_norm()

        for center in range(nr_sites):
            mps_drv.canonicalize_mps(center)

            norm = mps_drv.full_norm()

            rel_err = abs(norm - norm_before) / norm_before
            assert rel_err < 1e-10

            canonical_norm = mps_drv.canonical_norm()

            rel_err = abs(canonical_norm - norm_before) / norm_before
            assert rel_err < 1e-10

            for i in range(center):
                A = mps_drv.mps[i]
                m_l, d, m_r = A.shape
                left_metric = np.einsum("ldr, ldR -> rR", A.conjugate(), A).real
                assert np.max(np.abs(left_metric - np.eye(m_r))) < 1e-10

            for i in range(center + 1, nr_sites):
                A = mps_drv.mps[i]
                m_l, d, m_r = A.shape
                right_metric = np.einsum("ldr, Ldr -> lL", A, A.conjugate()).real
                assert np.max(np.abs(right_metric - np.eye(m_l))) < 1e-10

    def test_normalized_canonicalize(self, local_dim=4, m_bonddim=8, nr_sites=6):
        mps_drv = MpsDriver()
        mps_drv.local_dim = local_dim
        mps_drv.max_bond_dim = m_bonddim
        mps_drv.nr_sites = nr_sites
        mps_drv._initialize_random_mps()
        mps_drv.canonicalize_mps(2)
        mps_drv.normalize()

        for center in range(nr_sites):
            mps_drv.canonicalize_mps(center)
            mps_drv.normalize()
            norm_before = mps_drv.full_norm()

            schmidt_norm = np.dot(mps_drv.schmidt_spectrum, mps_drv.schmidt_spectrum)

            rel_err = abs(schmidt_norm - norm_before) / norm_before
            assert rel_err < 1e-10
