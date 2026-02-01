from dataclasses import dataclass

# To have consistennt settings across classes that use these


@dataclass
class Settings:
    """
    DMRG and system settings.
    """

    nr_sites: int
    max_bond_dim: int

    local_dim: int = 4
    nr_particles: int = 0
    svd_thr: float = 1e-9
    eig_thr: float = 1e-10
    nr_sweeps: int = 50
    allow_bond_growth: bool = True
    bond_growth_step: int = 2
