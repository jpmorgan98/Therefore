"""
Created on Sun May 15 20:34:03 2022
@author: jacksonmorgan
"""

import numpy as np
import matplotlib.pyplot as plt
import therefore

def analiticalSoultion(t, xsec, inital_flux, source, vel):
    angular_flux = inital_flux*np.exp(-xsec*vel*t) + (source/xsec)*(1-np.exp(-xsec*vel*t))
    
    return(angular_flux)
    


make_soultion = True
run_sim = True

#\Psi (t) = \Psi(0) e^(-\sigma*v*t) + Q/(\sigma)*(1-np.exp**(-\sigma*v*t))

def flatLinePlot(x, y, dat):
    for i in range(y.size):
        xx = x[i:i+2]
        yy = [y[i], y[i]]
        plt.plot(xx, yy, dat)

data_type = np.float64

L = 10
dx = 0.01
N_mesh = int(L/dx)
xsec = 10
ratio = 0.999
scattering_xsec = xsec*ratio
source_mat = 0
#source_a = 2
N_angle = 64

dt = 0.01
max_time = 0.3
inital_offset = 0.1 #from Avurv1
N_time = int(max_time/dt)
N_ans = 2*N_mesh

dx_mesh = dx*np.ones(N_mesh, data_type)
xsec_mesh = xsec*np.ones(N_mesh, data_type)
xsec_scatter_mesh = scattering_xsec*np.ones(N_mesh, data_type)
source_mesh = source_mat*np.ones([N_mesh], data_type)

psi_in = source_mat / (xsec*(1-ratio)/2)
#print(psi_in)

#setup = np.linspace(0, np.pi, 2*N_mesh)
inital_angular_flux = np.zeros([N_angle, 2*N_mesh])
in_mid = np.ones(N_angle)

xm = np.linspace(-L/2,L/2, N_ans+1)
inital_scalar_flux = therefore.azurv1_spav(xm, ratio, inital_offset)


[angles_gq, weights_gq] = np.polynomial.legendre.leggauss(N_angle)



assert(inital_scalar_flux.size == N_ans)

inital_angular_flux = np.zeros([N_angle, N_ans], data_type)
total_weight = sum(weights_gq)
for i in range(N_angle):
    inital_angular_flux[i, :] = inital_scalar_flux / total_weight



#inital_angular_flux[:,N_mesh] = in_mid
#for i in range(int(.48*N_mesh*2), int(.52*N_mesh*2), 1):
#    inital_angular_flux[:,i] = in_mid

#inital_angular_flux = np.array([[np.sin(setup)],[np.sin(setup)]]).reshape(2,200) #[:,:N_mesh]

#in_an_flux = 1
#inital_angular_flux = in_an_flux * np.ones([N_angle, 2*N_mesh])

sim_perams = {'data_type': data_type,
              'N_angles': N_angle,
              'L': L,
              'N_mesh': N_mesh,
              'boundary_condition_left':  'vacuum',
              'boundary_condition_right': 'vacuum',
              'left_in_mag': 10,
              'right_in_mag': 10,
              'left_in_angle': .3,
              'right_in_angle': 0,
              'max loops': 10000,
              'velocity': 1,
              'dt': dt,
              'max time': max_time,
              'N_time': N_time,
              'offset': inital_offset,
              'ratio': ratio}

theta = 1 #for discrete diamond
if run_sim:
    [scalar_flux, current, spec_rads] = therefore.TimeLoop(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, theta, 'SI')
    [scalar_flux2, current, spec_rads] = therefore.TimeLoop(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, theta, 'OCI')

N_ans = int(N_mesh * 2)
x = np.linspace(0, L, N_ans)
if run_sim:
    np.savez('outputs.npz', SI = scalar_flux, OCI = scalar_flux2, x = x, L = L)


if make_soultion == True:
    x_eval = np.linspace(-L/2, L/2, N_ans+1)
    azurv1_timeSpace = np.zeros([N_time+1, N_ans])
    
    time = inital_offset
    for i in range(N_time+1):
        azurv1_timeSpace[i,:] = therefore.azurv1_spav(x_eval, ratio, time)
        time += dt

    #plt.figure(1)
    #plt.plot(x, azurv1_timeSpace[:,1])
    #plt.plot(x, azurv1_space)
    #plt.show()
    #assert(azurv1_timeSpace.shape[0] == N_ans)
    #assert(azurv1_timeSpace.shape[1] == N_time)

    np.savez('azurv1.npz', azurv1 = azurv1_timeSpace)




#therefore.MovieMaker(scalar_flux, scalar_flux2, sim_perams)


'''
plt.figure(4)
plt.plot(x, scalar_flux[:,10] , '-k', label='SI')
plt.plot(x, scalar_flux2[:,10], '-r', label='OCI')
plt.plot(x, therefore.azurv1_spav(x_eval, ratio, inital_offset+dt*10),'--k', label='AVURV1')
plt.plot(x, inital_scalar_flux, '-b', label='Initial Condition')
plt.xlabel('Distance [cm]')
plt.ylabel('Scalar Flux [units of scalar flux]')
plt.title('First time step of transient methods')
plt.xlim(4.75,5.25)
plt.legend()
plt.savefig('SF_test.png')
'''