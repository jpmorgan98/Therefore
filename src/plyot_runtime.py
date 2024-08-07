import numpy as np
import matplotlib.pyplot as plt

with np.load('runtimes.npz') as data:
    OCI = data['OCI']
    Sweep = data['Sweep']
    dx = data['dx']

# throwing out first run (spool up and repeated)
dx = dx[1:]
OCI = OCI[1:,:]
Sweep = Sweep[1:,:]

mfp = dx*0.45468

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(mfp, OCI[:,0], 'k')
axs[0, 0].plot(mfp, Sweep[:,0], 'b-*')
axs[0, 0].set_title(r'$S_{4}$')
axs[0, 0].set_yscale('log')

axs[0, 1].plot(mfp, OCI[:,1], 'k')
axs[0, 1].plot(mfp, Sweep[:,1], 'b-*')
axs[0, 1].set_title(r'$S_{8}$')
axs[0, 1].set_yscale('log')

axs[1, 0].plot(mfp, OCI[:,2], 'k')
axs[1, 0].plot(mfp, Sweep[:,2], 'b-*')
axs[1, 0].set_title(r'$S_{16}$')
axs[1, 0].set_yscale('log')

axs[1, 1].plot(mfp, OCI[:,3], 'k')
axs[1, 1].plot(mfp, Sweep[:,3], 'b-*')
axs[1, 1].set_yscale('log')
axs[1, 1].set_title(r'$S_{32}$')


for ax in axs.flat:
    ax.set(xlabel=r'mfp [$\Sigma_2 * \Delta x$]', ylabel='wall-clock runtime [s]')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.savefig("runtimes.png")
plt.close()

# Speedup

speedup = Sweep/OCI
fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(mfp, speedup[:,0], 'r-^')
axs[0, 0].set_title(r'$S_{4}$')
#axs[0, 0].set_yscale('log')

axs[0, 1].plot(mfp, speedup[:,1], 'r-^')
axs[0, 1].set_title(r'$S_{8}$')
#axs[0, 1].set_yscale('log')

axs[1, 0].plot(mfp, speedup[:,2], 'r^-')
axs[1, 0].set_title(r'$S_{16}$')
#axs[1, 0].set_yscale('log')

axs[1, 1].plot(mfp, speedup[:,3], 'r-^')
#axs[1, 1].set_yscale('log')
axs[1, 1].set_title(r'$S_{32}$')


for ax in axs.flat:
    ax.set(xlabel=r'mfp [$\Sigma_2 * \Delta x$]', ylabel='speedup [s]')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.savefig("speedup.png")

