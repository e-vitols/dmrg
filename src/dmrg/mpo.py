import numpy as np
import veloxchem as vlx

from .hamiltonian import Hamiltonian


class MatrixProductOperator:
    """
    Implements the MatrixProductOperator by importing the orbitals and -- specific, for each operator -- integrals from VeloxChem.
    """

    def __init__(self):
        """
        Initializes the MatrixProductOperator.

        Instance variables:
            - scf_results: The converged SCF results tensor from VeloxChem.
            - operator: Specific operator to construct, if not given, the Hamiltonian is assumed.
        """
        # TODO: change class-name to MatrixProducOperatorConstructor/MpoConstructor
        self.scf_results = None
        self.operator = "Ham"

        self.one_elec_ints_ao = None
        self.two_elec_ints_ao = None
        self.nuc_repulsion_energy = None
        self.overlap = None

        self.nr_sites = None
        self.local_dim = None
        self.max_bond_dim = None

        self.ham = Hamiltonian()

    def update_integrals(self, molecule, basis):
        """
        Get/update AO-basis integral attributes of the MPO, imported from VeloxChem.

        :param molecule:
            A molecule-object as defined in VeloxChem.
        :param basis:
            The associated basis-object as defined in VeloxChem.
        """
        S, h, g, V_nuc = self.ham.get_ints(molecule, basis)
        self.one_elec_ints_ao = h
        self.two_elec_ints_ao = g
        self.nuc_repulsion_energy = V_nuc
        self.overlap = S

    def transform_integrals(self, scf_results):
        """
        Transform the AO-basis integrals to MO-basis.

        :param scf_results:
            The converged SCF tensors from a VeloxChem SCF

        :return:
            Returns the transformed one- and two-electron integrals in MO-basis.
        """

        C_alpha = scf_results["C_alpha"]
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

    def construct_mpo(self, scf_results, operator=None):
        """
        Constructs the requested operator.

        :param scf_results:
            The converged SCF results tensor from VeloxChem.
        :param operator:
            The specific operator requested, if None takes the set self.operator.
        """
        if operator is None:
            operator = self.operator
        L = self.nr_sites
        d = self.local_dim
        # m = self.max_bond_dim

        h_ij, g_ijkl = self.transform_integrals(scf_results)

        # mpo = [for _ in range(L)]

        identity = np.eye(4)
        cup_D = self.local_c("up", True)
        cup = self.local_c("up", False)
        cd_D = self.local_c("down", True)
        cd = self.local_c("down", False)

        return h_ij, g_ijkl

    def construct_hamiltonian(self, scf_results):
        L = self.nr_sites
        d = self.local_dim
        # m = self.max_bond_dim

        h_ij, g_ijkl = self.transform_integrals(scf_results)

    def dress_JW(self, p, spin, op_kind, local_dim=4):
        """
        Return the Jordan-Wigner transformed operator string.
        """
        jw_mat = self.jordan_wigner_mat(local_dim=local_dim)
        identity = np.eye(local_dim)
        L = self.nr_sites

        operator_str = []
        for l in range(L):
            if l < p:
                operator = jw_mat
            elif l == p:
                operator = self.local_c(spin, op_kind)
            else:
                operator = identity
            operator_str.append(operator)
        return operator_str

    def mpo_converted_JW(self, jw_string):
        _mpo = []
        for core in jw_string:
            _mpo.append(core[np.newaxis, :, :, np.newaxis])
        return _mpo

    # @staticmethod
    def local_c(self, spin: str, dagger: bool, dim: int = 4):
        """
        Local fermionic creation/annihilation operator in the basis
        {|0>, |up>, |down>, |up down>}.

        :param spin:
            The spin of the operator; 'up' or 'down'.
        :param dagger:
            The kind of operator: True -> creation, False -> annihilation
        """
        mat = np.zeros((dim, dim))
        jw_mat = self.jordan_wigner_mat()

        if spin == "up":
            mat[1, 0] = 1
            mat[3, 2] = 1
        elif spin == "down":
            mat[2, 0] = 1
            mat[3, 1] = 1
            # Impose anticommutation for same site-spins
            mat = mat @ jw_mat
        else:
            raise ValueError(f"Unknown spin: {spin!r}")

        if not dagger:
            mat = mat.T

        return mat

    @staticmethod
    def jordan_wigner_mat(local_dim=4):  # or nr_qbits
        """
        Constructs the matrix representation, in the basis {|0>, |up>, |down>, |up down>}, necessary for imposing fermionic
        anticommutation relations via the Jordan-Wigner transformation.

        :returns:
            The (4,4) dimensional matrix representation as a numpy array.
        """
        if local_dim != 4:
            raise ValueError("Only implemented for local dimension 4")

        return np.diag((1, -1, -1, 1))

    @staticmethod
    def apply_mpo(mpo, mps):
        transf_mps = mps.copy()

        for l in range(len(mps)):
            transf_mps[l] = np.einsum("dD, ldr -> lDr", mpo[l], mps[l])

        return transf_mps
