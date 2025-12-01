import numpy as np
import pytest

from dmrg.mpo import MatrixProductOperator
# import dmrg
from dmrg.mps import MatrixProductState

# from dmrg.sweep import Sweep


class TestJordanWigner:
    def test_antisymmetry(
        self, local_dim=4, m_bonddim=8, nr_sites=6, site=3, spin="up", creation=True
    ):
        """
        Test the antisymmetry, i.e., that creating/annihilating a fermion twice in the same
        state collapses the MPS.
        """
        tt = MatrixProductState()
        tt.local_dim = local_dim
        tt.max_bond_dim = m_bonddim
        tt.nr_sites = nr_sites
        tt._initialize_random_mps()
        tt.canonicalize_mps(0)
        mps = tt.mps

        con_mpo = MatrixProductOperator()
        con_mpo.nr_sites = tt.nr_sites

        mpo = con_mpo.dress_JW(site, spin, creation)

        transf_mps = con_mpo.apply_mpo(mps, mpo)
        second_transf_mps = con_mpo.apply_mpo(transf_mps, mpo)

        tt.mps = second_transf_mps
        norm = tt.full_norm()
        assert norm < 1e-10
