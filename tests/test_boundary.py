import numpy as np
import pytest

import dmrg


class TestBoundary:
    def test_left_right_mixed_boundary(self, local_dim=4, m_bonddim=8, nr_sites=6):
        # TODO: separate into a different test per assertion
        settings = dmrg.Settings(
            nr_sites=nr_sites, local_dim=local_dim, max_bond_dim=m_bonddim
        )
        mpo_drv = dmrg.MpoDriver(settings)
        mps_drv = dmrg.MpsDriver(settings)
        """
        mpo_drv.local_dim = local_dim
        mpo_drv.nr_sites = nr_sites

        mps_drv.local_dim = local_dim
        mps_drv.max_bond_dim = m_bonddim
        mps_drv.nr_sites = nr_sites
        """
        mps_drv._initialize_random_mps()
        mps_drv.canonical_form(5)
        mps_drv.normalize()

        identity_operator = mpo_drv.id_op()

        lb = mps_drv.left_boundary(
            identity_operator, center=mps_drv.canonical_center + 1
        )

        assert np.abs(lb - 1) < 1e-8

        mps_drv.canonical_form(0)
        rb = mps_drv.left_boundary(
            identity_operator, center=mps_drv.canonical_center - 1
        )

        assert np.abs(rb - 1) < 1e-8

        mps_drv.canonical_form(3)

        lb = mps_drv.left_boundary(identity_operator, center=mps_drv.canonical_center)
        rb = mps_drv.right_boundary(
            identity_operator, center=mps_drv.canonical_center - 1
        )

        assert np.abs(np.einsum("ldr, rdl", lb, rb) - 1) < 1e-8

        lb = mps_drv.left_boundary(
            identity_operator, center=mps_drv.canonical_center + 1
        )
        rb = mps_drv.right_boundary(identity_operator, center=mps_drv.canonical_center)

        assert np.abs(np.einsum("ldr, rdl", lb, rb) - 1) < 1e-8
