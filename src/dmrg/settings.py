from dataclasses import dataclass

import numpy as np

# To have consistennt settings across classes that use these


@dataclass
class Settings:
    """
    DMRG and system settings.
    """

    nr_sites: int
    local_dim: int

    max_bond_dim: int
    svd_thr: float = 1e-9
    eig_thr: float = 1e-10
    nr_sweeps: int = 50
    allow_bond_growth: bool = True
    bond_growth_step: int = 2
