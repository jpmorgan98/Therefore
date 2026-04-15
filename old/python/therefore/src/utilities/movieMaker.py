import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from .azurv1 import azurv1_spav

def MovieMaker(OCI_soultion, SI_soultion, AZURV1_soultion, sim_perams, FLP=False):
    '''You like Qinton Ternintino? Damn, we didn't measure this in feet
    '''
    
    dt = sim_perams['dt']
    offset = sim_perams['offset']
    L = sim_perams['L']
    scat_ratio = sim_perams['ratio']

    N_mesh = OCI_soultion.shape[0]
    N_time = OCI_soultion.shape[1]
    
    t = np.linspace(0, 10, N_time+1)
    x = np.linspace(0, L, int(N_mesh))
    dx = x[1]-x[0]
    
    t_gif = 10 #how long the gif should lasts
    frames_per = int(N_time/t_gif)


    fig,ax = plt.subplots() #plt.figure(figsize=(6,4)) 
    
    ax.grid()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'Flux')
    ax.set_title(r'$\bar{\phi}_{k,j}$ with $\Delta t=$')#{{{0}}}$'.format(dt))#{0}$, $\Delta x={1}$, and $c={2}$'.format(dt, dx, 0.9))
    
    line1, = ax.plot(x, OCI_soultion[:,0], '-k',label="Therefore OCI")
    line2, = ax.plot(x, SI_soultion[:,0],'-r',label="Therefore SI")
    line3, = ax.plot(x, AZURV1_soultion[0,:], '--b',label="AZURV1")
    text   = ax.text(offset, scat_ratio, '', transform=ax.transAxes)
    ax.legend()
    plt.ylim([0, np.max(OCI_soultion[:,0])]) #, OCI_soultion[:,0], AZURV1_soultion[:,0]
    #plt.xlim([4.9, 5.1])
    
    def animate(k):
        line1.set_ydata(OCI_soultion[:,k])
        line2.set_ydata(SI_soultion[:,k])
        line3.set_ydata(AZURV1_soultion[k,:])
        text.set_text(r'$t \in [%.1f,%.1f]$ s'%(t[k],t[k]))
        print('Figure production percent done: {0}'.format(int(k/N_time)*100), end = "\r")
        return line2, line2, line3, #, text
    print()
    print()
    
    simulation = animation.FuncAnimation(fig, animate, frames=N_time)
    plt.show()
    
    writervideo = animation.PillowWriter(fps=frames_per)
    simulation.save('slab_reactor.gif') #saveit!


#ax.fill_between(x_mid,phi[k,:]-phi_sd[k,:],phi[k,:]+phi_sd[k,:],alpha=0.2,color='b')
#text.set_text(r'$t \in [%.1f,%.1f]$ s'%(t[k],t[k+1]))

