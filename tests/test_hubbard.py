import numpy as np
import pytest

import dmrg

# from dmrg.mpo import MpoDriver
# from dmrg.mps import MpsDriver
# from dmrg.hamiltonian import Hamiltonian


class TestHubbardHamiltonian:
    def test_hubbard_efficient(self, m_bonddim=4, nr_sites=3):
        """
        Tests Hubbard with the fixed bond dim MPO.
        """
        canonical_center = 0

        settings = dmrg.Settings(
            nr_sites=nr_sites, max_bond_dim=m_bonddim, nr_particles=3
        )
        mpo_drv = dmrg.MpoDriver(settings)
        mps_drv = dmrg.MpsDriver(settings)

        # mps_drv.initialize_random_mps()
        mps_drv.initialize_u1_mps()
        mps_drv.canonical_form(canonical_center)
        mps_drv.normalize()

        mpo = mpo_drv.hubbard_mpo(t=1, U=2, mu=1)
        sweep_drv = dmrg.SweepDriver(settings, mps_drv=mps_drv, mpo_drv=mpo_drv)

        E0, mps = sweep_drv.compute(mpo)

        # reference
        ref = -4.8200893743747875

        assert abs(E0 - ref) < 1e-10

    def test_hubbard_naive(self, m_bonddim=4, nr_sites=3):
        """
        Tests Hubbard with the fixed bond dim MPO.
        """
        canonical_center = 0

        settings = dmrg.Settings(
            nr_sites=nr_sites, max_bond_dim=m_bonddim, nr_particles=3
        )
        mpo_drv = dmrg.MpoDriver(settings)
        mps_drv = dmrg.MpsDriver(settings)

        # mps_drv.initialize_random_mps()
        mps_drv.initialize_u1_mps()
        mps_drv.canonical_form(canonical_center)
        mps_drv.normalize()

        mpo = mpo_drv.hubbard_mpo_naive(t=1, U=2, mu=1)
        sweep_drv = dmrg.SweepDriver(settings, mps_drv=mps_drv, mpo_drv=mpo_drv)

        E0, mps = sweep_drv.compute(mpo)

        # reference
        ref = -4.8200893743747875

        assert abs(E0 - ref) < 1e-10

    @pytest.mark.slow
    def test_hubbard_naive_long(self, m_bonddim=8, nr_sites=4):
        """
        Tests Hubbard with the naiveöly built MPO.
        """
        canonical_center = 0

        settings = dmrg.Settings(
            nr_sites=nr_sites, max_bond_dim=m_bonddim, nr_particles=4
        )
        mpo_drv = dmrg.MpoDriver(settings)
        mps_drv = dmrg.MpsDriver(settings)

        # mps_drv.initialize_random_mps()
        mps_drv.initialize_u1_mps()
        mps_drv.canonical_form(canonical_center)
        mps_drv.normalize()

        mpo = mpo_drv.hubbard_mpo_naive(t=1, U=2, mu=1)

        sweep_drv = dmrg.SweepDriver(settings, mps_drv=mps_drv, mpo_drv=mpo_drv)

        E0, mps = sweep_drv.compute(mpo, random_lanczos=True)

        # reference
        ref = -6.875942809005068

        assert abs(E0 - ref) < 1e-10
