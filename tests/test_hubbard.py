import numpy as np
import pytest
import veloxchem as vlx

import dmrg

# from dmrg.mpo import MpoDriver
# from dmrg.mps import MpsDriver
# from dmrg.hamiltonian import Hamiltonian


class TestHubbardHamiltonian:
    def test_hubbard(self, local_dim=4, m_bonddim=8, nr_sites=4):
        canonical_center = 0

        mps_drv = dmrg.MpsDriver()
        mps_drv.local_dim = local_dim
        mps_drv.max_bond_dim = m_bonddim
        mps_drv.nr_sites = nr_sites
        mps_drv._initialize_random_mps()
        mps_drv.canonical_form(canonical_center)
        mps_drv.normalize()

        mpo_drv = dmrg.MpoDriver()
        mpo_drv.nr_sites = nr_sites
        mpo_drv.local_dim = local_dim

        mpo = mpo_drv.hubbard_mpo_from_dressed_strings(t=1, U=2, mu=1)
        sweep_drv = dmrg.SweepDriver(mps_drv=mps_drv, mpo_drv=mpo_drv)

        E0, mps = sweep_drv.compute(mpo, mps=mps_drv.mps)

        # reference
        ref = -6.875942809005068

        assert abs(E0 - ref) < 1e-10
