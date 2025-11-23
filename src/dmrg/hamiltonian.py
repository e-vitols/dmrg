import veloxchem as vlx
import numpy as np


class Hamiltonian:

    def __init__(self):
        self.orbitals = None

    def get_ints(self, molecule, basis):
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
