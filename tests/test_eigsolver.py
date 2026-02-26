import numpy as np
import pytest

import dmrg


class TestDensSolver:
    def test_pos_chem_pot(self, m_bonddim=2, nr_sites=4, canonical_center=0):
        # should yield GS "energy" = 0, but the sparse solver has problems with this (converges to the first excited state), hence confirm the hamiltonian- and sweep implementation is consistent by doing dense, full solving
        settings = dmrg.Settings(
            nr_sites=nr_sites, max_bond_dim=m_bonddim, nr_particles=0
        )
        mpo_drv = dmrg.MpoDriver(settings)
        mps_drv = dmrg.MpsDriver(settings)

        # mps_drv.initialize_random_mps()
        mps_drv.initialize_u1_mps()
        mps_drv.canonical_form(canonical_center)
        mps_drv.normalize()

        mpo = mpo_drv.chem_pot_mpo(coupling=2)
        sweep_drv = dmrg.SweepDriver(settings, mps_drv=mps_drv, mpo_drv=mpo_drv)

        E0, _mps = sweep_drv.compute(mpo, allow_bond_growth=True, dense_debug=True)

        ref = 0.0
        assert abs(E0 - ref) < 1e-6
