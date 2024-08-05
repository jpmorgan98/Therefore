import h5py
import numpy as np

import mcdc


# =============================================================================
# Materials
# =============================================================================

# Load material data
lib = h5py.File("c5g7_xs.h5", "r")


# Setter
def set_mat(mat):
    
    return mcdc.material(
        capture=mat["capture"][:],
        scatter=mat["scatter"][:],
        #fission=mat["fission"][:],
        #nu_p=mat["nu_p"][:],
        #nu_d=mat["nu_d"][:],
        #chi_p=mat["chi_p"][:],
        #chi_d=mat["chi_d"][:],
        speed=mat["speed"],
        #decay=mat["decay"],
    )




mat_uo2 = set_mat(lib["uo2"])  # Fuel: UO2
mat_mox43 = set_mat(lib["mox43"])  # Fuel: MOX 4.3%
mat_mox7 = set_mat(lib["mox7"])  # Fuel: MOX 7.0%
mat_mox87 = set_mat(lib["mox87"])  # Fuel: MOX 8.7%
mat_gt = set_mat(lib["gt"])  # Guide tube
mat_fc = set_mat(lib["fc"])  # Fission chamber
mat_cr = set_mat(lib["cr"])  # Control rod
mat_mod = set_mat(lib["mod"])  # Moderator



# Set surfaces
s1 = mcdc.surface("plane-z", z=0.0, bc="vacuum")
s2 = mcdc.surface("plane-z", z=2.5)
s3 = mcdc.surface("plane-z", z=3.5)
s4 = mcdc.surface("plane-z", z=6.0)
s5 = mcdc.surface("plane-z", z=7.0)
s6 = mcdc.surface("plane-z", z=9.5)
s7 = mcdc.surface("plane-z", z=10.5)
s8 = mcdc.surface("plane-z", z=13, bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], mat_mod)
mcdc.cell([+s2, -s3], mat_uo2)
mcdc.cell([+s3, -s4], mat_mod)
mcdc.cell([+s4, -s5], mat_cr)
mcdc.cell([+s5, -s6], mat_mod)
mcdc.cell([+s6, -s7], mat_uo2)
mcdc.cell([+s7, -s8], mat_mod)


energy = np.ones(7)

mcdc.source(z=[2.5, 3.5], time=[0.0, 1.0], isotropic=True, energy=energy)
mcdc.source(z=[9.5, 10.5], time=[0.0, 1.0], isotropic=True, energy=energy)

mcdc.tally(scores=["flux"], t=np.logspace(0,2, 20), z=np.linspace(0.0, 10, 61), g="all")

mcdc.setting(N_particle=1e6)
mcdc.run()

