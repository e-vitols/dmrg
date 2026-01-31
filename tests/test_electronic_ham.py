import numpy as np
import pytest
import veloxchem as vlx

import dmrg

# from dmrg.mpo import MpoDriver
# from dmrg.mps import MpsDriver
# from dmrg.hamiltonian import Hamiltonian


class TestElectronicHamiltonian:
    def test_h2_sto3g(self, m_bonddim=2, nr_sites=2):
        canonical_center = 0

        settings = dmrg.Settings(nr_sites=nr_sites, max_bond_dim=m_bonddim)
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

        int_drv = dmrg.IntegralsDriver()
        h_ij, g_ijkl = int_drv.get_transformed_integrals(molecule, basis, scf_res)
        V_nuc = int_drv.nuc_repulsion_energy

        mpo = mpo_drv.electronic_hamiltonian(h_ij, g_ijkl)
        sweep_drv = dmrg.SweepDriver(settings, mps_drv=mps_drv, mpo_drv=mpo_drv)

        E0, mps = sweep_drv.compute(mpo, mps=mps_drv.mps)
        total_ene = E0 + V_nuc

        # reference full-ci
        E_FCI = -1.1372744056952255

        assert abs(total_ene - E_FCI) < 1e-10

        occ0 = mps_drv.get_expectation_value(mpo_drv.site_occ_mpo(0))
        occ1 = mps_drv.get_expectation_value(mpo_drv.site_occ_mpo(1))

        assert abs(1.9745765353554527 - occ0) < 1e-6
        assert abs(0.0254234646445475 - occ1) < 1e-6

    @pytest.mark.slow
    def test_h2_631G(self, m_bonddim=4, nr_sites=4):
        canonical_center = 0

        settings = dmrg.Settings(nr_sites=nr_sites, max_bond_dim=m_bonddim)
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
        basis_str = "6-31G"
        basis = vlx.MolecularBasis.read(molecule, basis_str)

        scf_drv = vlx.ScfRestrictedDriver()
        scf_drv.ostream.mute()
        scf_res = scf_drv.compute(molecule, basis)

        int_drv = dmrg.IntegralsDriver()
        h_ij, g_ijkl = int_drv.get_transformed_integrals(molecule, basis, scf_res)
        V_nuc = int_drv.nuc_repulsion_energy

        mpo = mpo_drv.electronic_hamiltonian(h_ij, g_ijkl)
        sweep_drv = dmrg.SweepDriver(settings, mps_drv=mps_drv, mpo_drv=mpo_drv)

        E0, mps = sweep_drv.compute(mpo)
        total_ene = E0 + V_nuc

        # reference full-ci
        E_FCI = -1.1516800867759682

        assert abs(total_ene - E_FCI) < 1e-10
