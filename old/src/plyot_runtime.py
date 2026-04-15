import numpy as np
import matplotlib.pyplot as plt

with np.load('runtimes_mt.npz') as data:
    OCI = data['OCI']
    Sweep = data['Sweep']
    dx = data['dx']
    dt = data['dt']
    angles = data['angles']


with np.load('runtimes_iter.npz') as data:
    OCI_iter = data['OCI']
    Sweep_iter = data['Sweep']
    #dx = data['dx']
    #dt = data['dt']
    #angles = data['angles']

print(dt.size)

# throwing out first run (spool up and repeated)
dx = dx[1:]

OCI = OCI[:,:,1:]
Sweep = Sweep[:,:,1:]
speedup = Sweep/OCI

OCI_iter = OCI_iter[:,:,1:]
Sweep_iter = Sweep_iter[:,:,1:]

mfp = dx*0.45468


color_si = '#ca0020'
color_oci = '#404040'
color_speedup = '#7b3294'

for i in range(2):
    fig, axs = plt.subplots(4, 2, constrained_layout=True)
    for j in range (4):
        axs[0+j, 0].plot(mfp, OCI[i,j,:], '^-', color= color_oci, linewidth=2.5)
        axs[0+j, 0].plot(mfp, Sweep[i,j,:], '--*', color=color_si, linewidth=2.5)
        axs[0+j, 0].set_yscale('log')
        axs[0+j, 0].set_xscale('log')
        axs[0+j, 0].set_ylabel('S{0} \n runtime [s]'.format(angles[j]))

        axs[0+j, 1].plot(mfp, speedup[i,j,:], '-.', color=color_speedup, linewidth=2.5)
        #axs[0+j, 1].set_yscale('log')
        axs[0+j, 1].set_xscale('log')
        axs[0+j, 1].set_ylabel('speedup')

        #axs[0+j, 2].plot(mfp, OCI_iter[i,j,:], '-', color= color_oci, linewidth=2.5)
        #axs[0+j, 2].plot(mfp, Sweep_iter[i,j,:], '--', color=color_si, linewidth=2.5)
        #axs[0+j, 2].set_yscale('log')
        #axs[0+j, 2].set_xscale('log')
        #axs[0+j, 2].set_ylabel('iteration'.format(angles[j]))

        axs[j,0].grid()
        axs[j,1].grid()
        #axs[j,2].grid()

        #if (j<3):
        #    axs[j,0].set_xticks([])
        #    axs[j,1].set_xticks([])
            #axs[j,2].set_xticks([])


    axs[3,0].set(xlabel=r'$\delta$ [$\Sigma_2\Delta x$]')
    axs[3,1].set(xlabel=r'$\delta$ [$\Sigma_2\Delta x$]')
    #axs[3,2].set(xlabel=r'$\delta$ [$\Sigma_2\Delta x$]')

    if (i==0):
        axs[0, 0].text(5e-3, 1e-2, 'OCI', style='italic',) #bbox={'facecolor': color_oci, 'alpha': 0.5, 'pad': 1})
        axs[0, 0].text(1e0, 1e0, 'SI', style='italic',)# bbox={'facecolor': color_si, 'alpha': 0.5, 'pad': 1})
    else:
        axs[0, 0].text(5e-3, 1e-2, 'OCI', style='italic',) #bbox={'facecolor': color_oci, 'alpha': 0.5, 'pad': 1})
        axs[0, 0].text(1e0, 1e-1, 'SI', style='italic',)# bbox={'facecolor': color_si, 'alpha': 0.5, 'pad': 1})

    plt.savefig("runtimes_{}.pdf".format(dt[i]))
    #plt.clf()