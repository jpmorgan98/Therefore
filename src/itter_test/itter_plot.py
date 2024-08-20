import numpy as np
import matplotlib.pyplot as plt

dt = np.array([10, 5, 1.0, .5, .1, .05, .01, .005, .001])
dx = np.array([.01, .1, 1])
itter_OCI = np.array([[4631, 2541, 667, 469, 150, 83, 18, 11, 5],
                      [509, 282, 74, 53, 18, 11, 5, 4, 3],
                      [61, 34, 12, 9, 5, 4, 3, 3, 3]])

itter_Sweep = np.array([[95, 58, 20, 14, 8, 6, 5, 4, 3],
                        [95, 58, 20, 14, 8, 6, 5, 4, 3],
                        [95, 58, 20, 14, 8, 6, 5, 4, 3]])

# a zeroth sweep doesnt make since for counters
itter_OCI += 1
itter_Sweep += 1

itter_time_oci = np.array([1.6801e-03, 4.0198e-04, 2.4335e-04])
itter_time_si = np.array([3.2381e-01, 3.1652e-02, 3.2620e-03])


fig, axs = plt.subplots(3, 1, constrained_layout=True)
axs[0].plot(dt, itter_OCI[0,:], 'k-^')
axs[0].plot(dt, itter_Sweep[0,:], 'b-*')
axs[0].set_title(r'$\Delta x=0.01$')
axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].grid()

axs[1].plot(dt, itter_OCI[1,:], 'k-^')
axs[1].plot(dt, itter_Sweep[1,:], 'b-*')
axs[1].set_title(r'$\Delta x=0.1$')
axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[1].grid()

axs[2].plot(dt, itter_OCI[2,:], 'k-^')
axs[2].plot(dt, itter_Sweep[2,:], 'b-*')
axs[2].set_title(r'$\Delta x=1$')
axs[2].set_yscale('log')
axs[2].set_xscale('log')
axs[2].grid()


for ax in axs.flat:
    ax.set(xlabel=r'$\Delta t$', ylabel='iterations')

plt.savefig('itter.pdf')