import numpy as np
import mcdc

max_time = 5
dt = .1
N_time = int(max_time/dt)

# =============================================================================
# MC/DC setup
# =============================================================================

# Set materials
m = mcdc.material(capture = np.array([0.25]),
                scatter = np.array([[0.1825]]),
                fission = np.array([0.0]),
                nu_p    = np.array([0.0]),
                speed   = np.array([1.0]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=0, bc="vacuum")
s2 = mcdc.surface('plane-x', x=10,  bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m)
mcdc.source(point=[0.0, 0.0, 0.0], time=np.array([0,max_time]), isotropic=True)

# Setting
mcdc.setting(N_particle=1E7)

# =============================================================================
# Running it
# =============================================================================

    
mcdc.tally(scores=['flux'], x=np.linspace(0, 10, 201), t=np.linspace(0, max_time, N_time+1))

mcdc.run()

