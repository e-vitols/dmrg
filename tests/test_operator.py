import numpy as np
import pytest

import dmrg
from dmrg.mpo import MpoDriver
from dmrg.mps import MpsDriver


class TestOperator:
    def test_num_and_basic_ops(self, local_dim=4, m_bonddim=8, nr_sites=6):
        # TODO: separate into a different test per assertion
        mps_drv = MpsDriver()
        mps_drv.max_bond_dim = m_bonddim
        mps_drv.local_dim = local_dim
        mps_drv.nr_sites = nr_sites

        mpo_drv = MpoDriver()
        mpo_drv.local_dim = local_dim
        mpo_drv.nr_sites = nr_sites

        fixed_mps = [[] for _ in range(nr_sites)]

        for i in range(nr_sites):
            fixed_mps[i] = np.array([1, 0, 0, 0])[np.newaxis, :, np.newaxis]

        fixed_mps[0] = np.array([0, 1, 0, 0])[np.newaxis, :, np.newaxis]

        mps_drv.mps = fixed_mps
        canonical_center = 2
        mps_drv.canonical_form(canonical_center)
        mps_drv.normalize()

        assert np.abs(mps_drv.canonical_norm() - 1) < 1e-8

        exp_value_spin_up = mps_drv.get_expectation_value(
            mpo_drv.num_op("up"), center=2
        )

        assert np.abs(exp_value_spin_up.imag) < 1e-9
        assert np.abs(exp_value_spin_up.real - 1) < 1e-6

        mpo = mpo_drv.dress_JW(3, "up", True)
        mps_drv.mps = mpo_drv.apply_local_mpo(mpo, mps_drv.mps)

        exp_value_spin_up = mps_drv.get_expectation_value(
            mpo_drv.num_op("up"), center=2
        )
        print(exp_value_spin_up.real)

        assert np.abs(exp_value_spin_up.imag) < 1e-9
        assert np.abs(exp_value_spin_up.real - 2) < 1e-6
