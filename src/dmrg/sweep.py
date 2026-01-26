import copy

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .hamiltonian import HamiltonianDriver
from .mpo import MpoDriver
from .mps import MpsDriver

# import veloxchem as vlx


class SweepDriver:
    # def __init__(self):
    def __init__(self, *, mps_drv=None, mpo_drv=None, **kwargs):
        if mps_drv is not None:
            self.mps = mps_drv
        else:
            MpsDriver(**kwargs)

        if mpo_drv is not None:
            self.mpo = mpo_drv
        else:
            MpoDriver(**kwargs)

        self.nsweeps = 50

    def __getattr__(self, name):
        # try mps first, then mpo
        if hasattr(self.mps, name):
            return getattr(self.mps, name)
        if hasattr(self.mpo, name):
            return getattr(self.mpo, name)
        raise AttributeError(name)

    def apply_eff_ham(self, L, Wl, Wr, R, X):
        """ """
        Y = np.einsum(
            "bmSA, mcTB, Lbd, dABe, ecR -> LSTR",
            Wl,
            Wr,
            L,
            X,
            R,
            optimize=True,
        )

        # Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], Y.shape[3])

        return Y

    def _effective_linop(
        self, mpo, mps, center=None, two_site=True, dtype=np.complex128
    ):
        """
        The mapping/linear operator that applies the
        """

        # dtype is specified since this saves one iteration, as described in scipy documentation
        if center is None:
            center = self.canonical_center
        if not two_site:
            raise NotImplementedError("Only two-site optimization is enabled.")

        left_center = center
        right_center = center + 1

        L = self.left_boundary(mpo, mps=mps, center=left_center)
        R = self.right_boundary(mpo, mps=mps, center=right_center)
        Wl = mpo[left_center]
        Wr = mpo[right_center]

        Dl = mps[left_center].shape[0]
        Dr = mps[right_center].shape[2]
        d1 = Wl.shape[3]
        d2 = Wr.shape[3]
        dd = d1 * d2
        shape = (Dl, d1, d2, Dr)
        n = Dl * dd * Dr

        def _matvec(x):
            X = x.reshape(shape)
            Y = self.apply_eff_ham(L, Wl, Wr, R, X)  # returns (Dl, d1_out, d2_out, Dr)
            # Y = Y.reshape(Dl, Y.shape[1] * Y.shape[2], Dr)
            return Y.reshape(n)

        return LinearOperator((n, n), matvec=_matvec, dtype=dtype), shape

    def solve_local_two_site(self, mpo, mps, center=None, tol=1e-10, maxiter=None):
        Aop, shape = self._effective_linop(mpo, mps, center=center, two_site=True)

        if center is None:
            center = self.canonical_center

        P0 = self.get_twosite(center=center, mps=mps)
        v0 = P0.reshape(-1)

        w, v = eigsh(Aop, k=1, which="SA", v0=v0, tol=tol, maxiter=maxiter)
        Theta_opt = v[:, 0].reshape(shape)  # (Dl, dd, Dr)
        E0 = w[0].real
        return E0, Theta_opt
