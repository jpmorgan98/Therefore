from tkinter import X
import numpy as np
#np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import therefore
from timeit import default_timer as timer

#import mcdc
import numpy as np
#import h5py

def t2p(time):
    return(int((time/max_time)*N_time))


# =============================================================================
# Therefore setup
# =============================================================================

data_type = np.float64

L = 10
dx = 0.1/2
N_mesh = int(L/dx)
xsec = 2
ratio = 0.95
scattering_xsec = xsec*ratio
source_mat = 0.1
N_angle = 64

v = 4

dt = 0.1
max_time = 0.2

N_time = int(max_time/dt)
N_ans = 2*N_mesh
dx_mesh = dx*np.ones(N_mesh, data_type)
xsec_mesh = xsec*np.ones(N_mesh, data_type)
xsec_scatter_mesh = scattering_xsec*np.ones(N_mesh, data_type)
source_mesh = source_mat*np.ones([N_mesh], data_type)

psi_in = source_mat / (xsec*(1-ratio)/2)
#print(psi_in)

[angles_gq, weights_gq] = np.polynomial.legendre.leggauss(N_angle)

IC = 0

inital_scalar_flux = IC*np.ones(2*N_mesh)

inital_angular_flux = np.zeros([N_angle, N_ans], data_type)
total_weight = sum(weights_gq)
for i in range(N_angle):
    inital_angular_flux[i, :] = inital_scalar_flux / total_weight

sim_perams = {'data_type': data_type,
              'N_angles': N_angle,
              'L': L,
              'N_mesh': N_mesh,
              'boundary_condition_left':  'vacuum',
              'boundary_condition_right': 'vacuum',
              'left_in_mag': 0,
              'right_in_mag': 0.3,
              'left_in_angle': 0,
              'right_in_angle': 0,
              'max loops': 10000,
              'velocity': v,
              'dt': dt,
              'max time': max_time,
              'N_time': N_time,
              'offset': 0,
              #'ratio': ratio,
              'tolerance': 1e-9,
              'print': True}

'''
start = timer()
print('OCI MB SCB Single big gpu')
[sfMB, current, spec_rads, time] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'Big') #OCI_MB_GPU
end = timer()
print(end - start)
'''

start = timer()
print('SI MB SCB Single big gpu')
[sfMBSi_gpu, current, spec_rads, time] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'SI_MB_GPU') #OCI_MB_GPU
end = timer()
print(end - start)



'''
start = timer()
print('OCI MB SCB Small GPU')
[sfMB_badGpu, current, spec_rads] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'OCI_MB_GPU')
end = timer()
print(end - start)
'''

'''
start = timer()
print('OCI MB SCB CPU')
[sfMB_trad, current, spec_rads] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'OCI_MB')
end = timer()
print(end - start)
'''


'''
start = timer()
print('SI MB SCB')
[sfMBSi, current, spec_rads] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'SI_MB')
end = timer()
print(end - start)
'''

'''
start = timer()
print('SI BE SCB')
[sfEuler, current, spec_rads, loops] = therefore.euler(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'SI')
end = timer()
print(end - start)
'''

'''
[sfSS, current2, spec_rad2, source_converged, loops] = therefore.OCI(sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh)
[sfSS2, current2, spec_rad2, source_converged, loops] = therefore.SourceItteration(sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh)
'''
'''
for i in range(sfMB.shape[0]):
    for j in range(sfMB.shape[1]):
        #print(i)
        #print(j)
        #print()
        k = sfMB[i,j] == sfMB_trad[i,j]

        if k == False:
            print('fuck at {0}, {1}'.format(i,j))
'''

#print(np.allclose(sfMB, sfMB_trad, atol=1e-9))
#np.set_printoptions(linewidth=np.inf)
#print(sfMB)
#print(sfMB_trad)


x = np.linspace(0, L, int(N_mesh*2))

fig,ax = plt.subplots()
    
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\phi$')
ax.set_title('Inf med (ϕ)')

import matplotlib.animation as animation

line1, = ax.plot(x, sfMB[:,0], '-k',label="MB-OCI-Big")
#line2, = ax.plot(x, sfMB_trad[:,0], '-b',label="MB-OCI-Small")
#line3, = ax.plot(x, sfEuler[:,0], '-g',label="BE-SI")
#line4, = ax.plot(x, sfMBSi[:,0], '-r',label="MB-SI")
line5, = ax.plot(x, sfMBSi_gpu[:,0], '-y',label="MB-SI-Big")
#line6, = ax.plot(x, sfSS, '--k^',label="SS - OCI")
#line7, = ax.plot(x, sfSS2, '--g*',label="SS - SI")

text   = ax.text(.75 ,0.75,'') 
ax.legend()
#lim = np.max((sfMB_trad, sfMBSi, sfMBSi_gpu))*1.25
lim = np.max((sfMB, sfMBSi_gpu))*1.25
plt.ylim(-0.02, lim)

def animate(k):
    line1.set_ydata(sfMB[:,k])
    #line2.set_ydata(sfMB_trad[:,k])
    #line3.set_ydata(sfEuler[:,k])
    #line4.set_ydata(sfMBSi[:,k])
    line5.set_ydata(sfMBSi_gpu[:,k])
    #line6.set_ydata(sfSS)
    #line6.set_ydata(sfSS2)

    text.set_text(r'$t \in [%.1f,%.1f]$ s'%(dt*k,dt*(k+1)))
 #   return line1, line2, line3

simulation = animation.FuncAnimation(fig, animate, frames=N_time)

writervideo = animation.PillowWriter(fps=250)
simulation.save('test_mb.gif') #saveit!