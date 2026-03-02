import numpy as np
import pytest

import dmrg


class TestOperator:
    def test_chem_pot(self, m_bonddim=8, nr_sites=4, canonical_center=0):
        # TODO: separate into a different test per assertion
        # Feedback: it looks like you fixed the TODO, so the comment can be deleted ;)
        settings = dmrg.Settings(nr_sites=nr_sites, max_bond_dim=m_bonddim)
        mpo_drv = dmrg.MpoDriver(settings)
        mps_drv = dmrg.MpsDriver(settings)

        mps_drv.initialize_fixed_mps()
        mps_drv.canonical_form(canonical_center)
        mps_drv.normalize()

        mpo = mpo_drv.chem_pot_mpo(coupling=1)
        exp_val = mps_drv.get_expectation_value(mpo)

        ref = 4.00000000000000

        assert abs(exp_val - ref) < 1e-6

    # @pytest.mark.slow
    def test_chem_pot(self, m_bonddim=8, nr_sites=4, canonical_center=0):
        # TODO: separate into a different test per assertion
        settings = dmrg.Settings(nr_sites=nr_sites, max_bond_dim=m_bonddim)
        mpo_drv = dmrg.MpoDriver(settings)
        mps_drv = dmrg.MpsDriver(settings)

        mps_drv.initialize_fixed_mps()
        mps_drv.canonical_form(canonical_center)
        mps_drv.normalize()

        mpo = mpo_drv.chem_pot_mpo(coupling=-1)

        sweep_drv = dmrg.SweepDriver(settings, mpo_drv=mpo_drv, mps_drv=mps_drv)

        E0, mps = sweep_drv.compute(mpo)

        assert abs(E0 - 8)
