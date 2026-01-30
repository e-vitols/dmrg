import numpy as np
import pytest

import dmrg


class TestMps:
    def test_mps(self, local_dim=4, m_bonddim=8, nr_sites=6):
        canonical_center = 0

        settings = dmrg.Settings(
            nr_sites=nr_sites, local_dim=local_dim, max_bond_dim=m_bonddim
        )
        mps_drv = dmrg.MpsDriver(settings)

        mps_drv._initialize_fixed_mps()
        mps = mps_drv.mps
        assert len(mps) == nr_sites

        assert mps[0].shape[:2] == (1, local_dim)
        assert mps[-1].shape[1:] == (local_dim, 1)
        assert mps[1].shape[-1] == mps[2].shape[0]

        mps_drv.canonical_center = canonical_center
        mps_drv.canonical_form()
        mps_drv.normalize()
        mps = mps_drv.mps

        assert abs(mps_drv.full_norm() - 1) < 1e-8
        assert abs(mps_drv.canonical_norm() - 1) < 1e-8
        assert abs(mps_drv.overlap(mps, mps) - 1) < 1e-8
