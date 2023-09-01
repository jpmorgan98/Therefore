"""
Created on Sun May 15 20:34:03 2022

@author: joannamorgan
"""

import numpy as np
import matplotlib.pyplot as plt
import therefore
from timeit import default_timer as timer


# >>>>> problem setup


#from tabulate import tabulate
dt = 0.1
max_time = 0.5
N_time = int(max_time/dt)

v = 4

N_angle = 64


def flatLinePlot(x, y, pl):
    for i in range(y.size):
        xx = x[i:i+2]
        yy = [y[i], y[i]]
        plt.plot(xx, yy, pl)

data_type = np.float64

#mesh builder for Reed's Problem
region_id = np.array([1,2,3,4,5], int)
region_widths = np.array([2,2,1,1,2], int)
region_bounds = np.array([2,4,5,6,8], float)
sigma_s = np.array([.99, .9, 0, 0, 0], data_type)
sigma_t = np.array([1, 1, 0, 5, 50], data_type)
Source = np.array([0, 1, 0, 0, 50], data_type)
dx = .1/sigma_t
dx[2] = .25  #fix a nan
dx[0] = .25
dx[1] = .25

print(dx)
N_region = np.array(region_widths/dx, int)

N_mesh: int = sum(N_region)
N_ans: int = (N_mesh*2)

xsec_mesh = np.empty(2*N_mesh, data_type)
xsec_scatter_mesh = np.empty(N_mesh, data_type)
dx_mesh = np.empty(N_mesh, data_type)
source_mesh = np.empty(N_mesh, data_type)
region_id_mesh = np.empty(N_mesh, data_type)
region_id_mesh_2 = np.empty(N_mesh*2, data_type)

#build the mesh
for i in range(region_widths.size):
    LB = sum(N_region[:i])
    RB = sum(N_region[:i+1])
    xsec_mesh[LB:RB] = sigma_t[i]
    xsec_scatter_mesh[LB:RB] = sigma_s[i]
    dx_mesh[LB:RB] = dx[i]
    source_mesh[LB:RB] = Source[i]
    region_id_mesh[LB:RB] = region_id[i]

for i in range(N_mesh):
    region_id_mesh_2[2*i] = region_id_mesh[i]
    region_id_mesh_2[2*i+1] = region_id_mesh[i]

x = np.zeros(N_mesh*2)
for i in range(N_mesh):
    x[2*i] = sum(dx_mesh[:i])
    x[2*i+1] = sum(dx_mesh[:i+1])

np.savez('x.npz', x=x)

#x[-1] = 8.00001
L = 8.0
    
#set sim peramweters
sim_perams = {'data_type': data_type,
              'N_angles': N_angle,
              'L': L,
              'N_mesh': N_mesh,
              'boundary_condition_left': 'vacuum',
              'boundary_condition_right': 'reflecting',
              'left_in_mag': 0,
              'right_in_mag': 0,
              'left_in_angle': 0,
              'right_in_angle': 0,
              'max loops': 10000,
              'velocity': v,
              'dt': dt,
              'max time': max_time,
              'N_time': N_time,
              'offset': 0,
              'tolerance': 1e-9,
              'print': True}

# Initial Condition
[angles_gq, weights_gq] = np.polynomial.legendre.leggauss(N_angle)
inital_scalar_flux = np.zeros(2*N_mesh)
inital_angular_flux = np.zeros([N_angle, N_ans], data_type)
total_weight = sum(weights_gq)
for i in range(N_angle):
    inital_angular_flux[i, :] = inital_scalar_flux / total_weight



# >>>>> problem running

'''
start = timer()
print('OCI MB SCB Single big gpu')
[sfMBSparse, current, spec_rads, loops] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'Big') 
end = timer()
print(end - start)


start = timer()
print('SI MB SCB Single big gpu')
[sfMBSiBig, current, spec_rads, loops] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'SI_MB_GPU') 
end = timer()
print(end - start)
'''

'''
start = timer()
print('SI BE SCB')
[sfEulerSI, current, spec_rads, loops] = therefore.euler(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'SI')
end = timer()
print(end - start)

start = timer()
print('SI BE SCB')
[sfEulerOCI, current, spec_rads, loops] = therefore.euler(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'OCI')
end = timer()
print(end - start)


[sfSS, current2, spec_rad2, source_converged, loops] = therefore.OCI(sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh)

#launch some problems
'''


start = timer()
print('SI MB SCB')
[sfMBSi, current, spec_rads, loops] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'SI_MB')
end = timer()
print(end - start)


start = timer()
print('OCI MB SCB CPU')
[sfMB_trad, current, spec_rads, loops] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'OCI_MB')
end = timer()
print(end - start)


'''
np.savez('TD_Reeds.npz', sfMBSparse=sfMBSparse, sfEulerSI=sfEulerSI,  sfEulerOCI=sfEulerOCI, sfSS=sfSS, sfMBSiBig=sfMBSiBig, sfMBSi=sfMBSi, sfMB_trad=sfMB_trad)



# >>>> problem output

fig,ax = plt.subplots()
    
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\phi$')
ax.set_title('Scalar Flux (ϕ)')

import matplotlib.animation as animation

line1, = ax.plot(x, sfMBSparse[:,0], '-k',label="MB-OCI-Big")
#line2, = ax.plot(x, sfMB_trad[:,0], '-r',label="MB-OCI-Small")
line3, = ax.plot(x, sfEuler[:,0], '-g',label="BE-SI")
line4, = ax.plot(x, sfMBSi[:,0], '-b',label="MB-SI")
line5, = ax.plot(x, sfMBSiBig[:,0], '-y',label="MB-SI-GPU")
line6, = ax.plot(x, sfSS, '-p',label="SS")
text   = ax.text(8.0,0.75,'') 
ax.legend()
plt.ylim(-0.2, 8.2)

def animate(k):
    line1.set_ydata(sfMBSparse[:,k])
    #line2.set_ydata(sfMB_trad[:,k])
    line3.set_ydata(sfEuler[:,k])
    line4.set_ydata(sfMBSi[:,k])
    line5.set_ydata(sfMBSiBig[:,k])
    line6.set_ydata(sfSS)
    text.set_text(r'$t \in [%.1f,%.1f]$ s'%(dt*k,dt*(k+1)))

simulation = animation.FuncAnimation(fig, animate, frames=N_time)

writervideo = animation.PillowWriter(fps=250)
simulation.save('td_reeds.gif') #saveit!

'''