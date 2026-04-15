import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

npzfile = np.load('output_nonscat.npz')

sfMB  = npzfile['sfMB']
sfEuler = npzfile['sfEuler']
x = npzfile['x']
dt = npzfile['dt']
#sfRefan = npzfile['sfRefAn']
N_time = 50

#npzfile2 = np.load('output_beam_scat1000000.npz')

#sfRef = npzfile2['sfReft']
#print(sfRef.shape)
#xref = np.linspace(0,5, sfRef.shape[1])

#sfRef *= 27
#for i in range(sfRef.shape[0]):
#    sfRef[i,:] /= max(sfRef[i,:] )

fig,ax = plt.subplots() #plt.figure(figsize=(6,4))
    
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\phi$')
ax.set_title('Scalar Flux (ϕ)')

line1, = ax.plot(x, sfMB[:,0], '-k',label="MB-SCB")
line2, = ax.plot(x, sfEuler[:,0], '-r',label="BE-SCB")
#line3, = ax.plot(xref, sfRef[0,:], '--g*',label="Ref-MCDC")
#line4, = ax.plot(x, sfRefan[:,0], '--b^', label='Analitical')
text   = ax.text(8.0,0.75,'') #, transform=ax.transAxes
ax.legend()
plt.ylim(-0.2, 1.5) #, OCI_soultion[:,0], AZURV1_soultion[:,0]

def animate(k):
    line1.set_ydata(sfMB[:,k])
    line2.set_ydata(sfEuler[:,k])
    #line3.set_ydata(sfRef[k, :])
    #line4.set_ydata(sfRefan[:,k])
    #ax.set_title(f'Scalar Flux (ϕ) at t=%.1f'.format(dt*k)) #$\bar{\phi}_{k,j}$ with 
    text.set_text(r'$t \in [%.1f,%.1f]$ s'%(dt*k,dt*(k+1)))
    #print('Figure production percent done: {0}'.format(int(k/N_time)*100), end = "\r")
    return line1, line2

simulation = animation.FuncAnimation(fig, animate, frames=50)
#plt.show()

writervideo = animation.PillowWriter(fps=250)
simulation.save('both.gif') #saveit!