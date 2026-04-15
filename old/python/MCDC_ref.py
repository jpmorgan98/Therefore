import numpy as np
import mcdc

import h5py

# =============================================================================
# MC/DC setup
# =============================================================================

L = 1
xsec = 1
ratio = 0.0
scattering_xsec = xsec*ratio

dt = np.array([.1, 0.075, 0.05, 0.025, 0.0125, 0.01])
max_time = 0.5

# Set materials
m = mcdc.material(capture = np.array([xsec]),
                scatter = np.array([[0]]),
                fission = np.array([0.0]),
                nu_p    = np.array([0.0]),
                speed   = np.array([1.0]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=-1E10, bc="vacuum")
s2 = mcdc.surface('plane-x', x=1E10,  bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m)

mcdc.source(point=[1E-9,0.0,0.0], time=np.array([0,1]), isotropic=True)

# Tally
N_time = int(max_time/dt[0])
mcdc.tally(scores=['flux', 'flux-t'], x=np.linspace(0, 1, 201), t=np.linspace(0, max_time, N_time+1)) #np.arange(0, 0.4, dt)

# Setting
mcdc.setting(N_particle=1E6)

for i in range(dt.size):
    N_time = int(max_time/dt[i])

    print()
    print('======================================================================================')
    print('Time step {0}/{1}: {2}'.format(i, dt.size, dt[i]))
    print('======================================================================================')
    print()

    N_time = int(max_time/dt[i])
    
    mcdc.tally(scores=['flux', 'flux-t'], x=np.linspace(0, 1, 201), t=np.linspace(0, max_time, N_time+1))
    mcdc.run()

    with h5py.File('output.h5', 'r') as f:
        sfRef = f['tally/flux/mean'][:]
        sfRef_sd   = f['tally/flux/sdev'][:]
        sfReft = f['tally/flux-t/mean'][:]
        sfReft_sd = f['tally/flux-t/sdev'][:]
        t     = f['tally/grid/t'][:]
        
    
    sfRef = np.transpose(sfRef)

    np.savez('output_mcdc_dt_{0}'.format(dt[i]), sfRef=sfRef, sfRef_sd=sfRef_sd, sfReft=sfReft, sfReft_sd=sfReft_sd)