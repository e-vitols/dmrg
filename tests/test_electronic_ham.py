import numpy as np
import pytest
import veloxchem as vlx

import dmrg

# from dmrg.mpo import MpoDriver
# from dmrg.mps import MpsDriver
# from dmrg.hamiltonian import Hamiltonian


class TestElectronicHamiltonian:
    def test_h2(self, local_dim=4, m_bonddim=2, nr_sites=2):
        canonical_center = 0

        settings = dmrg.Settings(
            nr_sites=nr_sites, local_dim=local_dim, max_bond_dim=m_bonddim
        )
        mpo_drv = dmrg.MpoDriver(settings)
        mps_drv = dmrg.MpsDriver(settings)

        mps_drv._initialize_random_mps()
        mps_drv.canonical_form(canonical_center)
        mps_drv.normalize()

        mol_xyz = """2

        H 0.000 0.000 0.000
        H 0.000 0.000 0.741
        """

        molecule = vlx.Molecule.read_xyz_string(mol_xyz)
        basis_str = "STO-3G"
        basis = vlx.MolecularBasis.read(molecule, basis_str)

        scf_drv = vlx.ScfRestrictedDriver()
        scf_drv.ostream.mute()
        scf_res = scf_drv.compute(molecule, basis)

        ham_drv = dmrg.HamiltonianDriver()
        h_ij, g_ijkl = ham_drv.get_transformed_integrals(molecule, basis, scf_res)
        V_nuc = ham_drv.nuc_repulsion_energy

        mpo = mpo_drv.electronic_hamiltonian(h_ij, g_ijkl)
        sweep_drv = dmrg.SweepDriver(settings, mps_drv=mps_drv, mpo_drv=mpo_drv)

        E0, mps = sweep_drv.compute(mpo, mps=mps_drv.mps)
        total_ene = E0 + V_nuc

        # reference full-ci
        E_FCI = -1.1372744056952255

        assert abs(total_ene - E_FCI) < 1e-10
