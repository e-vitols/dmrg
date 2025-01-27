import numpy as np
np.set_printoptions(suppress=True, precision=3)

"""
The spin operator /vec{S}_i at site i, represents the spin at site i.
In the full Hillbert space of a spin-system of length L, the z-spin
component at site 2 is represented by:
S_z(2) = I /kron S_z /kron I /kron ... (L)

The interaction of spins at site i and j is then given 
by /vec{S}_i /cdot /vec{S}_j = S_x(i)S_x(j) + S_y(i)S_y(j) + S_z(i)S_z(j),
which in the full basis similar to before is:
"""

def spin_operators():
    """
    Returns the Sx, Sy, Sz operators for a spin-1/2, each as a 2x2 NumPy array.
    S = 1/2 * sigma (Pauli matrices).
    """
    # Pauli matrices
    sigma_x = np.array([[0, 1],
                        [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j],
                        [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0],
                        [0, -1]], dtype=complex)
    
    S_x = 0.5 * sigma_x
    S_y = 0.5 * sigma_y
    S_z = 0.5 * sigma_z
    return S_x, S_y, S_z

def raiselower_operators():
    S_x, S_y, S_z = spin_operators()
    
    S_p = S_x + 1j * S_y
    S_m = S_p.conj().T

    return S_p, S_m


class SpinChain:
    def __init__(self, L, J=1):
        self.L = L
        self.J = J

    def sys_block(self):
        pass
    
    def interaction(self, S_i, S_j):
        #I = np.eye(2)
        """
        S_i = [S_z, S_plus, S_minus]
        """

        return np.kron(S_i[0], S_j[0]) + 0.5 * (np.kron(S_i[1],S_j[2]) + np.kron(S_i[2],S_j[1]))
    
    def hamiltonian(self):
        #H = 
        _, __, S_z = spin_operators()
        S_p, S_m = raiselower_operators()
        S = np.array([S_z, S_p, S_m])

        H = np.zeros((2**self.L, 2**self.L), dtype=complex)
        H_local = self.interaction(S, S)

        for i in range(self.L - 1):
            H_full = np.eye(2**i, dtype=complex)
            H_full = np.kron(H_full, H_local)
            H_full = np.kron(H_full, np.eye(2**(self.L - i - 2), dtype=complex))
            H += H_full
        return H
    
    def exact_diagonalization(self, H):
        eig_vals, eig_vecs = np.linalg.eigh(H)
        return eig_vals, eig_vecs



if __name__ == '__main__':
    spin_sys = SpinChain(2)
    H = spin_sys.hamiltonian()
    eig_vals, eig_vecs = spin_sys.exact_diagonalization(H)
    print(f'Hamiltonian : \n{H}')
    print(f'Eigenvals: \n{eig_vals}')
    print(f'Eigenvectors: \n{eig_vecs}')