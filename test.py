"""
Created on Sun May 15 20:34:03 2022
@author: jacksonmorgan
"""

import numpy as np
import matplotlib.pyplot as plt
import therefore


def flatLinePlot(x, y):
    for i in range(y.size):
        xx = x[i:i+2]
        yy = [y[i], y[i]]
        plt.plot(xx, yy, '-k')

data_type = np.float64

L = 10
dx = 1
xsec = 10
ratio = 0.9999
scattering_xsec = xsec*ratio
source_mat = 0
source_a = 2
N_mesh = int(L/dx)

dx_mesh = dx*np.ones(N_mesh, data_type)
xsec_mesh = xsec*np.ones(N_mesh, data_type)
xsec_scatter_mesh = scattering_xsec*np.ones(N_mesh, data_type)
source_mesh = source_mat*np.ones(N_mesh, data_type)

psi_in = source_mat / (xsec*(1-ratio)/2)
print(psi_in)

sim_perams = {'data_type': data_type,
              'N_angles': 2,
              'L': L,
              'N_mesh': N_mesh,
              'boundary_condition_left':  'incident_iso',
              'boundary_condition_right': 'incident_iso',
              'left_in_mag': 10,
              'right_in_mag': 10,
              'left_in_angle': .3,
              'right_in_angle': 0,
              'max loops': 10000}

[scalar_flux, current] = therefore.OCI(sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh)

print(scalar_flux)

f=1
X = np.linspace(0, L, int(N_mesh*2+1))
plt.figure(f)
flatLinePlot(X, scalar_flux)
plt.title('Infinte Med')
plt.xlabel('Distance')
plt.ylabel('Scalar Flux')
plt.show()

f+=1
plt.figure(f)
plt.title('Infinte Med')
plt.xlabel('Distance')
plt.ylabel('Current')
plt.ylim([-1,1])
flatLinePlot(X, current)
plt.show()
#launch source itterations #SourceItteration
