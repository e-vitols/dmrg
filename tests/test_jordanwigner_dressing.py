import numpy as np
import pytest

from dmrg.mpo import MpoDriver
from dmrg.mps import MpsDriver

# import dmrg

# from dmrg.sweep import Sweep


class TestJordanWigner:
    def test_antisymmetry(
        self, local_dim=4, m_bonddim=8, nr_sites=6, site=3, spin="up", creation=True
    ):
        """
        Test the antisymmetry, i.e., that creating/annihilating a fermion twice in the same
        state collapses the MPS.
        """
        tt = MpsDriver()
        tt.local_dim = local_dim
        tt.max_bond_dim = m_bonddim
        tt.nr_sites = nr_sites
        tt._initialize_random_mps()
        tt.canonicalize_mps(0)
        mps = tt.mps

        con_mpo = MpoDriver()
        con_mpo.nr_sites = tt.nr_sites

        mpo = con_mpo.dress_JW(site, spin, creation)

        transf_mps = con_mpo.apply_mpo(mpo, mps)
        second_transf_mps = con_mpo.apply_mpo(mpo, transf_mps)

        tt.mps = second_transf_mps
        norm = tt.full_norm()
        assert norm < 1e-10
