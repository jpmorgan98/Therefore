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

L1 = 2
dx1 = 0.002
N_mesh1 = int(L1/dx1)
xsec1 = 1
ratio1 = 0.99
scattering_xsec1 = xsec1*ratio1
source_mat1 = 50

L2 = 1
dx2 = 0.002
N_mesh2 = int(L2/dx2)
xsec2 = 5
ratio2 = 0
scattering_xsec2 = xsec2*ratio2
source_mat2 = 0


N_mesh = N_mesh1 + N_mesh2
L = L1 + L2

N_angle = 2

v = 1

BCl = 0.5

dt = 0.1
max_time = 2

N_time = int(max_time/dt)

N_ans = 2*N_mesh

dx_mesh = dx2*np.ones(N_mesh2, data_type)
dx_mesh = np.append(dx_mesh, (dx1*np.ones(N_mesh1, data_type)))

xsec_mesh = xsec2*np.ones(N_mesh2, data_type)
xsec_mesh = np.append(xsec_mesh, xsec1*np.ones(N_mesh1, data_type))
xsec_scatter_mesh = scattering_xsec2*np.ones(N_mesh2, data_type)
xsec_scatter_mesh = np.append(xsec_scatter_mesh, scattering_xsec1*np.ones(N_mesh1, data_type))

source_mesh = source_mat2*np.ones([N_mesh2], data_type)
source_mesh = np.append(source_mesh, source_mat1*np.ones([N_mesh1], data_type))
#psi_in = source_mat / (xsec*(1-ratio)/2)
#print(psi_in)

[angles_gq, weights_gq] = np.polynomial.legendre.leggauss(N_angle)

#setup = np.linspace(0, np.pi, 2*N_mesh)
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
              'boundary_condition_right': 'reflecting',
              'left_in_mag': 0.3,
              'right_in_mag': .3,
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

x = np.zeros(N_mesh*2)
for i in range(N_mesh):
    x[2*i] = sum(dx_mesh[:i])
    x[2*i+1] = sum(dx_mesh[:i+1])

'''
start = timer()
print('OCI MB SCB Single big gpu')
[sfMB, current, spec_rads] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'Big') #OCI_MB_GPU
end = timer()
print(end - start)
'''

start = timer()
print('SI MB SCB Single big gpu')
[sfMBSi_gpu, current, spec_rads] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'SI_MB_GPU') #OCI_MB_GPU
end = timer()
print(end - start)

'''
start = timer()
print('OCI MB SCB Small GPU')
[sfMB_badGpu, current, spec_rads] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'OCI_MB_GPU')
end = timer()
print(end - start)



start = timer()
print('OCI MB SCB CPU')
[sfMB_trad, current, spec_rads] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'OCI_MB')
end = timer()
print(end - start)

'''


start = timer()
print('SI MB SCB')
[sfMBSi, current, spec_rads] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'SI_MB')
end = timer()
print(end - start)



start = timer()
print('SI BE SCB')
[sfEuler, current, spec_rads, loops] = therefore.euler(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'SI')
end = timer()
print(end - start)



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


#x = np.linspace(0, L, int(N_mesh*2))

fig,ax = plt.subplots()
    
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\phi$')
ax.set_title('Scalar Flux (ϕ)')

import matplotlib.animation as animation

#line1, = ax.plot(x, sfMB[:,0], '-k',label="MB-OCI-Big")
#line2, = ax.plot(x, sfMB_trad[:,0], '-r',label="MB-OCI-Small")
line3, = ax.plot(x, sfEuler[:,0], '-g*',label="BE-SI")
line4, = ax.plot(x, sfMBSi[:,0], '--b',label="MB-SI")
line5, = ax.plot(x, sfMBSi_gpu[:,0], '-y',label="MB-SI-Big")

text   = ax.text(8.0,0.75,'') 
ax.legend()
lim = np.max((sfEuler, sfMBSi, sfMBSi_gpu))
plt.ylim(-0.2, 200)

def animate(k):
    #line1.set_ydata(sfMB[:,k])
    #line2.set_ydata(sfMB_trad[:,k])
    line3.set_ydata(sfEuler[:,k])
    line4.set_ydata(sfMBSi[:,k])
    line5.set_ydata(sfMBSi_gpu[:,k])

    text.set_text(r'$t \in [%.1f,%.1f]$ s'%(dt*k,dt*(k+1)))
 #   return line1, line2, line3

simulation = animation.FuncAnimation(fig, animate, frames=N_time)

writervideo = animation.PillowWriter(fps=250)
simulation.save('trans_test.gif') #saveit!