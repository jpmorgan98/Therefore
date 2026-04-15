import matplotlib.pyplot as plt
import h5py
import numpy as np

from reference import reference


# Load results
with h5py.File("output.h5", "r") as f:
    z = f["tally/grid/z"][:]
    dz = z[1:] - z[:-1]
    z_mid = 0.5 * (z[:-1] + z[1:])

    psi = f["tally/flux/mean"][:]
    psi_sd = f["tally/flux/sdev"][:]

phi=psi
phi_sd=psi_sd


# Flux - spatial average
plt.plot(z_mid, phi[0,:], "-b", label="MC")
print(phi[2,:])
#plt.fill_between(z_mid, phi[0,:] - phi_sd[0,:], phi[0,:] + phi_sd[0,:], alpha=0.2, color="b")
plt.xlabel(r"$z$, cm")
plt.ylabel("Flux")
plt.ylim([0.06, 0.16])
plt.grid()
plt.legend()
plt.title(r"$\bar{\phi}_i$")
plt.show()

