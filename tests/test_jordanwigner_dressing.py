import copy

import numpy as np
import pytest

from dmrg.mpo import MpoDriver
from dmrg.mps import MpsDriver

# import dmrg

# from dmrg.sweep import Sweep


class TestJordanWigner:
    def test_antisymmetry_fullnorm_same_site(
        self, local_dim=4, m_bonddim=8, nr_sites=6, site=3, spin="up", creation=True
    ):
        """
        Test the antisymmetry, i.e., that creating/annihilating a fermion twice in the same
        state collapses the MPS.
        """
        mps_drv = MpsDriver()
        mps_drv.local_dim = local_dim
        mps_drv.max_bond_dim = m_bonddim
        mps_drv.nr_sites = nr_sites
        mps_drv._initialize_random_mps()
        mps_drv.canonicalize_mps(0)
        mps = mps_drv.mps

        mpo_drv = MpoDriver()
        mpo_drv.nr_sites = mps_drv.nr_sites

        mpo = mpo_drv.dress_JW(site, spin, creation)

        transf_mps = mpo_drv.apply_mpo(mpo, mps)
        second_transf_mps = mpo_drv.apply_mpo(mpo, transf_mps)

        mps_drv.mps = second_transf_mps
        norm = mps_drv.full_norm()
        assert norm < 1e-10

    def test_antisymmetry_fullnorm_diff_site(
        self, local_dim=4, m_bonddim=8, nr_sites=6, site=3, spin="up", creation=True
    ):
        """
        Test the antisymmetry, i.e., that creating/annihilating a fermion twice in the same
        state collapses the MPS.
        """
        mps_drv = MpsDriver()
        mps_drv.local_dim = local_dim
        mps_drv.max_bond_dim = m_bonddim
        mps_drv.nr_sites = nr_sites
        for site1 in range(nr_sites - 1):
            for site2 in range(site1, nr_sites):
                for spin in ["up", "down"]:
                    for creation in [True, False]:
                        mps_drv._initialize_random_mps()

                        mps1 = mps_drv.mps
                        mps2 = mps1.copy()

                        mpo_drv = MpoDriver()
                        mpo_drv.nr_sites = mps_drv.nr_sites
                        # print(spin, creation, site, site + 1)

                        mpo_i = mpo_drv.dress_JW(site1, spin, creation)
                        mpo_j = mpo_drv.dress_JW(site2, spin, creation)

                        mps_i = mpo_drv.apply_mpo(mpo_i, mps1)
                        mps_ij = mpo_drv.apply_mpo(mpo_j, mps_i)

                        mps_j = mpo_drv.apply_mpo(mpo_j, mps2)
                        mps_ji = mpo_drv.apply_mpo(mpo_i, mps_j)

                        norm_2 = (
                            mps_drv.overlap(mps_ij, mps_ij)
                            + mps_drv.overlap(mps_ij, mps_ji)
                            + mps_drv.overlap(mps_ji, mps_ij)
                            + mps_drv.overlap(mps_ji, mps_ji)
                        )
                        assert norm_2 < 1e-10

    def test_antisymmetry_canonical_norm(
        self,
        local_dim=4,
        m_bonddim=8,
        nr_sites=6,
        site=3,
        spin="up",
        creation=True,
        canonical_center=0,
    ):
        """
        Test the antisymmetry, i.e., that creating/annihilating a fermion twice in the same
        state collapses the MPS.
        """
        # TODO: remove this, or double-check that thiis doesn't make sense
        # for the zero-vector/mps, as then the canonical form is not well-defined
        mps_drv = MpsDriver()
        mps_drv.local_dim = local_dim
        mps_drv.max_bond_dim = m_bonddim
        mps_drv.nr_sites = nr_sites
        mps_drv._initialize_random_mps()
        mps_drv.canonicalize_mps(canonical_center)
        mps = mps_drv.mps

        mpo_drv = MpoDriver()
        mpo_drv.nr_sites = mps_drv.nr_sites

        mpo = mpo_drv.dress_JW(site, spin, creation)

        transf_mps = mpo_drv.apply_mpo(mpo, mps)
        second_transf_mps = mpo_drv.apply_mpo(mpo, transf_mps)
        mps_drv.mps = second_transf_mps

        mps_drv.canonicalize_mps(canonical_center)

        norm = mps_drv.full_norm()
        assert norm < 1e-10
