import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import therefore
from timeit import default_timer as timer

def t2p(time):
    return(int((time/max_time)*N_time))


# =============================================================================
# Therefore setup
# =============================================================================

data_type = np.float64

L = 1
dx = 0.1
N_mesh = int(L/dx)
xsec = 1
ratio = 0.25
scattering_xsec = xsec*ratio
source_mat = 1
N_angle = 2

v = 1

dt = 0.5
max_time = .5

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
              'right_in_mag': 0,
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


start = timer()
print('OCI MB SCB Single big gpu')
[sfMB, a, current, spec_rads, time] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'Big') #OCI_MB_GPU
end = timer()
print(end - start)

'''
print(sfMB.shape)
x = np.linspace(0,1,sfMB.shape[0])

plt.figure()
plt.plot(x,sfMB[:,1])
plt.show()'''

a1 = np.array((0.237754,0.240899,0.407043,0.412316,0.0316977,0.133876,0.0230203,0.161521,0.240945,0.24363,0.411953,0.415708,0.153816,0.205816,0.182534,0.27507,0.243921,0.246868,0.415361,0.417586,0.217102,0.239376,0.290803,0.346401,0.247236,0.250615,0.416903,0.416706,0.245023,0.251558,0.356787,0.386743,0.250766,0.253874,0.415128,0.410203,0.253898,0.253263,0.392918,0.406896,0.253263,0.253898,0.406896,0.392918,0.253874,0.250766,0.410203,0.415128,0.251558,0.245023,0.386743,0.356787,0.250615,0.247236,0.416706,0.416903,0.239376,0.217102,0.346401,0.290803,0.246868,0.243921,0.417586,0.415361,0.205816,0.153816,0.27507,0.182534,0.24363,0.240945,0.415708,0.411953,0.133876,0.0316977,0.161521,0.0230203,0.240899,0.237754,0.412316,0.407043))


b = np.zeros((2,20))
c = np.zeros((2,20))

b1 = np.zeros((2,20))
c1 = np.zeros((2,20))

for i in range(10):
    b[0, 2*i] = a[8*i]
    b[0, 2*i+1] = a[8*i+1]
    c[0, 2*i] = a[8*i+2]
    c[0, 2*i+1] = a[8*i+3]
    b[1, 2*i] = a[8*i+4]
    b[1, 2*i+1] = a[8*i+5]
    c[1, 2*i] = a[8*i+6]
    c[1, 2*i+1] = a[8*i+7]

    b1[0, 2*i] = a1[8*i]
    b1[0, 2*i+1] = a1[8*i+1]
    c1[0, 2*i] = a1[8*i+2]
    c1[0, 2*i+1] = a1[8*i+3]
    b1[1, 2*i] = a1[8*i+4]
    b1[1, 2*i+1] = a1[8*i+5]
    c1[1, 2*i] = a1[8*i+6]
    c1[1, 2*i+1] = a1[8*i+7]

x = np.linspace(0, 1, b.shape[1])

plt.figure()
plt.plot(x,b[0,:], x,b[1,:])
plt.plot(x,c[0,:], x,c[1,:])
plt.show()

plt.figure()
plt.plot(x,b[1,:],'r-^', x,b1[1,:],'k-*')
plt.show()