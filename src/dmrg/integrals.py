import numpy as np
import veloxchem as vlx


class IntegralsDriver:
    """
    The IntegralsDriver object interfacing the program to VeloxChem for obtaininng integrals and molecular orbital (MO) coefficients.

    NOTE: this class is responsible for mainly interafacing with VLX

    Instance variables:
        - one_elec_ints_ao: The one-electron integrals in the atomic-orbital basis. (arary)
        - two_elec_ints_ao: The two-electron integrals in the atomic-orbital basis. (arary)
        - overlap: The overlap integrals in the atomic-orbital basis. (arary)
        - nuc_repulsion_energy: the nuclear repulsion energy. (float)
    """

    def __init__(self):
        """
        Initializes the IntegralsDriver object.
        """
        self.one_elec_ints_ao = None
        self.two_elec_ints_ao = None
        self.overlap = None
        self.nuc_repulsion_energy = None

    def get_ints(self, molecule, basis):
        """
        Computes the integrals by interfacing to VeloxChem (VLX).

        :param molecule:
            The VLX molecule object.
        :param:
            The VLX basis object.

        :return:
            The overlap integrals (array), core hamiltonian integrals (array), two-electron integrals (array), and the nuclear repulsion energy (float).
        """

        V_nuc = molecule.nuclear_repulsion_energy()
        overlap_drv = vlx.OverlapDriver()
        S = overlap_drv.compute(molecule, basis).to_numpy()

        # kinetic energy
        kinetic_drv = vlx.KineticEnergyDriver()
        T = kinetic_drv.compute(molecule, basis).to_numpy()

        # nuclear attraction
        npot_drv = vlx.NuclearPotentialDriver()
        V = -1.0 * npot_drv.compute(molecule, basis).to_numpy()

        # one-electron Hamiltonian
        h = T + V

        # two-electron Hamiltonian
        fock_drv = vlx.FockDriver()
        g = fock_drv.compute_eri(molecule, basis)

        return S, h, g, V_nuc

    def update_integrals(self, molecule, basis):
        """
        Get/update AO-basis integral attributes of the MPO, imported from VeloxChem.

        :param molecule:
            A molecule-object as defined in VeloxChem.
        :param basis:
            The associated basis-object as defined in VeloxChem.
        """
        S, h, g, V_nuc = self.get_ints(molecule, basis)
        self.one_elec_ints_ao = h
        self.two_elec_ints_ao = g
        self.nuc_repulsion_energy = V_nuc
        self.overlap = S

    @staticmethod
    def permute_integrals(t_ij, v_ijkl, perm):
        """
        Simpler to just permute the MOs
        """
        perm = np.asarray(perm)
        t_p = t_ij[np.ix_(perm, perm)]
        v_p = v_ijkl[np.ix_(perm, perm, perm, perm)]
        return t_p, v_p

    def get_transformed_integrals(self, molecule, basis, scf_results, permute=None):
        """
        Transform the AO-basis integrals to MO-basis.

        :param scf_results:
            The converged SCF tensors from a VeloxChem SCF calculation.

        :return:
            Returns the transformed one- and two-electron integrals in MO-basis.
        """
        # if integrals is None:
        self.update_integrals(molecule, basis)

        C_alpha = scf_results["C_alpha"]
        if permute is not None:
            C_alpha = C_alpha[:, permute]

        h_ij = np.einsum(
            "uv, ui, vj -> ij", self.one_elec_ints_ao, C_alpha, C_alpha, optimize=True
        )
        g_ijkl = np.einsum(
            "uvws, ui, vj, wk, sl -> ijkl",
            self.two_elec_ints_ao,
            C_alpha,
            C_alpha,
            C_alpha,
            C_alpha,
            optimize=True,
        )

        return h_ij, g_ijkl
