from tkinter import X
import numpy as np
#np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import therefore

#import mcdc
import numpy as np
import h5py

def t2p(time):
    return(int((time/max_time)*N_time))


# =============================================================================
# Therefore setup
# =============================================================================

data_type = np.float64

L = 10
dx = 0.01
N_mesh = int(L/dx)
xsec = 0.25
ratio = 0.75
scattering_xsec = xsec*ratio
source_mat = 0
N_angle = 2

BCl = 1.0


ratio = np.linspace(0, .99, 75)

#source = 0
source = np.linspace(0, 10, 75)

mfp = xsec*dx

xs = ratio.size
ys = source.size


no_converge_oci = np.zeros([xs, ys])
spec_rad_oci = np.zeros([xs, ys])
no_converge_si = np.zeros([xs, ys])
spec_rad_si = np.zeros([xs, ys])

total_runs = xs * ys

time_oci = 0
time_si = 0

dt = 0.1
max_time = 4.0

N_time = int(max_time/dt)

N_ans = 2*N_mesh

dx_mesh = dx*np.ones(N_mesh, data_type)
xsec_mesh = xsec*np.ones(N_mesh, data_type)
xsec_scatter_mesh = scattering_xsec*np.ones(N_mesh, data_type)
source_mesh = source_mat*np.ones([N_mesh], data_type)

psi_in = source_mat / (xsec*(1-ratio)/2)
#print(psi_in)

[angles_gq, weights_gq] = np.polynomial.legendre.leggauss(N_angle)

#setup = np.linspace(0, np.pi, 2*N_mesh)
inital_scalar_flux = np.zeros(2*N_mesh)

inital_angular_flux = np.zeros([N_angle, N_ans], data_type)
total_weight = sum(weights_gq)
for i in range(N_angle):
    inital_angular_flux[i, :] = inital_scalar_flux / total_weight

sim_perams = {'data_type': data_type,
              'N_angles': N_angle,
              'L': L,
              'N_mesh': N_mesh,
              'boundary_condition_left':  'incident_iso',
              'boundary_condition_right': 'vacuum',
              'left_in_mag': BCl,
              'right_in_mag': 10,
              'left_in_angle': .3,
              'right_in_angle': 0,
              'max loops': 10000,
              'velocity': 1.0,
              'dt': dt,
              'max time': max_time,
              'N_time': N_time,
              'offset': 0,
              'ratio': ratio,
              'tolerance': 1e-9,
              'print': False}


#[sfMBSi, current, spec_rads] = therefore.euler(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'SI')
#[sfMB, current, spec_rads] = therefore.euler(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'OCI')

[scalar_flux, current, spec_rad, conver, counter] = therefore.OCI(sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh)
[scalar_flux2, current2, spec_rad2, conver2, countersi] = therefore.SourceItteration(sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh)

print(counter)
print(spec_rad)
print(countersi)
print(spec_rad2)

dt_test = np.array([2, 1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005]) #, 0.001, 0.0001

l_si = []
l_oci = []
specrad_si = []
specrad_oci = []

for i in range(dt_test.size):
    print('dt cycle: {0}'.format(dt_test[i]))

    sim_perams['dt'] = dt_test[i]
    sim_perams['N_time'] = int(max_time/dt_test[i])

    [sfMBSi, current, spec_rads_si, loops_si] = therefore.euler(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'SI')

    [sfMB, current, spec_rads_oci, loops_oci] = therefore.euler(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'OCI')

    l_si.append(loops_si)
    l_oci.append(loops_oci)

    specrad_si.append(max(spec_rads_si))
    specrad_oci.append(max(spec_rads_oci))

print(l_si)
print(l_oci)
print(specrad_si)
print(specrad_oci)

np.savez('euler_loops', loops_si=l_si, loops_oci=l_oci, specrad_si=specrad_si, specrad_oci=specrad_oci, dt_test=dt_test)


plt.figure(1)
plt.loglog(dt_test, l_si, '-*k', label='SI')
plt.loglog(dt_test, l_oci, '--r', label='OCI')
#plt.xscale('log') 
plt.xlabel(r'$\Delta t$')
plt.ylabel('Itterations')
plt.legend()
plt.savefig('euler_loops.png', dpi=600)

plt.figure(2)
plt.loglog(dt_test, specrad_si, '-*r', label='SI')
plt.loglog(dt_test, specrad_oci, '--k', label='OCI')
#plt.xscale('log') 
plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$\rho$')
plt.legend()
plt.savefig('euler_rad.png', dpi=600)

#x = np.linspace(0, L, int(N_mesh*2))
