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
            self.mps_drv = mps_drv
        else:
            MpsDriver(**kwargs)

        if mpo_drv is not None:
            self.mpo_drv = mpo_drv
        else:
            MpoDriver(**kwargs)

        self.nr_sweeps = 50

    def __getattr__(self, name):
        # try mps first, then mpo
        if hasattr(self.mps_drv, name):
            return getattr(self.mps_drv, name)
        if hasattr(self.mpo_drv, name):
            return getattr(self.mpo_drv, name)
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
        Theta_opt = v[:, 0].reshape(shape)
        E0 = w[0].real
        return E0, Theta_opt

    def compute(
        self,
        mpo,
        mps=None,
        center=0,
        ene_conv_thr=1e-6,
        trunc_conv_thr=1e-8,
        allow_bond_growth=True,
    ):
        """
        Starts with left-to-right sweep
        """
        if mps is None:
            mps = self.mps_drv.mps

        self.canonical_form(center=center, mps=mps)
        mps = self.normalize(mps=mps, center=center)
        self.mps_drv.mps = mps
        nr_bonds = len(mps) - 1

        self.E_0 = 0
        self.converged = False
        for sweep in range(self.nr_sweeps):
            print(f"Sweep nr. {sweep+1}")

            R_trunc_error = np.zeros(nr_bonds, dtype=float)
            # right-sweep
            for cen in range(nr_bonds):
                E, theta = self.solve_local_two_site(mpo, self.mps_drv.mps, center=cen)
                _center, mps = self.mps_drv.split_twosite(theta, "right", center=cen)

                R_trunc_error[cen] = self.mps_drv.discarded_weight
                self.mps_drv.mps = mps
                self.mps_drv.canonical_center = cen
                self.canonical_center = cen

            E_rsweep = self.mps_drv.get_expectation_value(mpo)
            print(
                f"Energy after left sweep : {E_rsweep:.6f} a.u.\n"
                f"Discarded weight: max = {R_trunc_error.max():.3e}, mean = {R_trunc_error.mean():.3e} (worst bond: {int(R_trunc_error.argmax())})\n"
            )

            L_trunc_error = np.zeros(nr_bonds, dtype=float)
            # left-sweep
            for cen in range(nr_bonds - 1, -1, -1):
                E, theta = self.solve_local_two_site(mpo, self.mps_drv.mps, center=cen)
                _center, mps = self.mps_drv.split_twosite(theta, "left", center=cen)

                L_trunc_error[cen] = self.mps_drv.discarded_weight
                self.mps_drv.mps = mps
                self.mps_drv.canonical_center = cen
                self.canonical_center = cen

            L_trunc_max = L_trunc_error.max()
            E_lsweep = self.mps_drv.get_expectation_value(mpo)
            print(
                f"Energy after left sweep : {E_lsweep:.6f} a.u.\n"
                f"Discarded weight: max = {L_trunc_max:.3e}, mean = {L_trunc_error.mean():.3e} (worst bond: {int(L_trunc_error.argmax())})\n"
            )

            if allow_bond_growth and (L_trunc_max > trunc_conv_thr):
                print(
                    f"**OBS** Large truncation error: Maximum bond dimension increased from {self.mps_drv.max_bond_dim} to {self.mps_drv.max_bond_dim+2}"
                )
                self.mps_drv.max_bond_dim += 2
            elif L_trunc_max > trunc_conv_thr:
                print(
                    f"**OBS** Large truncation error! Allowing for bond dimension growth is advised.\n"
                )
                # To allow for convergence with fixed bond dim
                L_trunc_max = 0
            else:
                # To allow for convergence with fixed bond dim
                L_trunc_max = 0

            if abs(self.E_0 - E_lsweep) < ene_conv_thr and (
                L_trunc_max < trunc_conv_thr
            ):
                self.converged = True
                self.E_0 = E_lsweep
                print(
                    f"\n** Converged after {sweep+1} sweeps! **\nGround-state energy = {self.E_0:.6f} a.u.\n"
                )
                return self.E_0, self.mps_drv.mps

            self.E_0 = E_lsweep
            # TODO: add nuclear energy
            # self.total_energy = self.E_0 + self.V_nuc
