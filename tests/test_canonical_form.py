import numpy as np
import pytest

import dmrg
from dmrg import mps
from dmrg.mps import MatrixProductState


class TestCanonicalize:
    def test_left_canonicalize(self, local_dim=4, m_bonddim=8, nr_sites=6):
        # tt = mps.MatrixProductState()
        tt = MatrixProductState()
        tt.local_dim = local_dim
        tt.max_bond_dim = m_bonddim
        tt.nr_sites = nr_sites
        tt._initialize_random_mps()
        mps = tt.mps
        tt.canonicalize_mps(6)

        orthonorm = np.einsum("ldr, ldR -> rR", tt.mps[0], tt.mps[0].conjugate())

        assert np.max(np.abs(orthonorm - np.eye(4))) < 1e-10

    def test_right_canonicalize(self, local_dim=4, m_bonddim=8, nr_sites=6):
        # tt = mps.MatrixProductState()
        tt = MatrixProductState()
        tt.local_dim = local_dim
        tt.max_bond_dim = m_bonddim
        tt.nr_sites = nr_sites
        tt._initialize_random_mps()
        mps = tt.mps

        env = np.array([[1.0]], dtype=complex)
        for A in mps:
            env = np.einsum("lL, ldr, LdR -> rR", env, A, A.conjugate())
        norm_before = float(env.squeeze().real)

        tt.canonicalize_mps(0)
        mps = tt.mps

        # orthonorm = np.einsum("ldr, rdl -> lr", tt.mps[-1], tt.mps[-1].conjugate())
        orthonorm = np.einsum("ldr, Ldr -> lL", tt.mps[-1], tt.mps[-1].conjugate())

        assert np.max(np.abs(orthonorm - np.eye(4))) < 1e-10

        env = np.array([[1.0]], dtype=complex)

        for A in mps:
            env = np.einsum("lL, ldr, LdR -> rR", env, A, A.conjugate())
        # assert env.squeeze().imag < 1e-12

        norm = float(env.squeeze().real)

        rel_err = abs(norm - norm_before) / norm_before
        assert rel_err < 1e-10
        # assert abs(norm - norm_before) < 1e-4
        canonical_norm = np.einsum("ldr, ldr", mps[0], mps[0].conjugate())
        rel_err = abs(canonical_norm - norm_before) / norm_before

        assert rel_err < 1e-10
