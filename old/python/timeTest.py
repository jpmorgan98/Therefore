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
dx = 0.1
N_mesh = int(L/dx)
xsec = 0.25
ratio = 0
scattering_xsec = xsec*ratio
source_mat = 0
N_angle = 4

dt = 2
max_time = 20

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
              'left_in_mag': 0.5,
              'right_in_mag': 10,
              'left_in_angle': .3,
              'right_in_angle': 0,
              'max loops': 1000,
              'velocity': 1,
              'dt': dt,
              'max time': max_time,
              'N_time': N_time,
              'offset': 0,
              'ratio': ratio,
              'tolerance': 1e-9}

'''
# =============================================================================
# MC/DC setup
# =============================================================================

# Set materials
m = mcdc.material(capture = np.array([xsec]),
                scatter = np.array([[0]]),
                fission = np.array([0.0]),
                nu_p    = np.array([0.0]),
                speed   = np.array([1.0]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=-1E10, bc="vacuum")
s2 = mcdc.surface('plane-x', x=1E10,  bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m)
mcdc.source(x=[1.0, 1.0, 1.0], time=np.array([0,1]), isotropic=True)

# Setting
mcdc.setting(N_particle=1E7)

# =============================================================================
# Running it
# =============================================================================

    
mcdc.tally(scores=['flux'], x=np.linspace(0, 1, 201), t=np.linspace(0, max_time, N_time+1))
mcdc.run()
'''

[sfMB, current, spec_rads] = therefore.multiBalance(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 'OCI_MB')
[sfEuler, current, spec_rads] = therefore.euler(inital_angular_flux, sim_perams, dx_mesh, xsec_mesh, xsec_scatter_mesh, source_mesh, 1, 'OCI')

ss_xRef_ex = np.array([0, 0.25, 0.5,
    0.75,
    1,
    1.25,
    1.5,
    1.75,
    2,
    2.25,
    2.5,
    2.75,
    3,
    3.25,
    3.5,
    3.75,
    4,
    4.25,
    4.5,
    4.75,
    5,
    5.25,
    5.5,
    5.75,
    6,
    6.25,
    6.5,
    6.75,
    7,
    7.25,
    7.5,
    7.75,
    8,
    8.25,
    8.5,
    8.75,
    9,
    9.25,
    9.5,
    9.75,
    10])


ss_sfRef_ex = np.array([0.299999998,
    0.268779374,
    0.241460362,
    0.217481669,
    0.196369686,
    0.177724223,
    0.161206604,
    0.146529743,
    0.133449844,
    0.121759478,
    0.111281796,
    0.101865694,
    0.093381763,
    0.085718903,
    0.078781493,
    0.072487008,
    0.066764032,
    0.061550582,
    0.056792705,
    0.052443293,
    0.048461094,
    0.044809867,
    0.041457677,
    0.038376299,
    0.035540704,
    0.032928639,
    0.030520253,
    0.028297791,
    0.02624533,
    0.024348547,
    0.022594531,
    0.020971611,
    0.019469216,
    0.018077747,
    0.016788474,
    0.015593439,
    0.014485377,
    0.013457643,
    0.01250415,
    0.011619316,
    0.010798014])


with h5py.File('output.h5', 'r') as f:
    sfRef = f['tally/flux/mean'][:]
    t     = f['tally/grid/t'][:]
sfRef = np.transpose(sfRef)

for j in range(sfMB.shape[1]):
    sfEuler[:,j] = sfEuler[:,j]/max(sfEuler[:,j])
    sfMB[:,j] = sfMB[:,j]/max(sfMB[:,j])

for j in range(sfRef.shape[1]):
    sfRef[:,j] = sfRef[:,j]/max(sfRef[:,j])

ss_sfRef_ex /= max(ss_sfRef_ex)



v=1
# analitic soultion
'''
def psi(x,t):
    s = x + v*t
    return(0.5*np.exp((-xsec*s)/(1+angles_gq[1])))

x = np.linspace(0, L, int(N_mesh*2))
fig, axs = plt.subplots(N_time-1)
for i in range(1,N_time):
    axs[i-1].plot(x, sfEuler[:,i], label='euler')
    axs[i-1].plot(x, sfMB[:,i], label='mb')
    axs[i-1].plot(x, sfRef[:,i], label='ref')
    axs[i-1].set_title('time {0}'.format(i*dt))
plt.tight_layout()

for ax in axs.flat:
    ax.label_outer()

'''
x = np.linspace(0, L, int(N_mesh*2))
plt.figure(1)
plt.plot(x, sfEuler[:,-1], label='euler, t=20')
plt.plot(x, sfMB[:,-1], label='mb, t=20')
plt.plot(x, sfRef[:,-1], label='MCDC, t=20')
plt.plot(ss_xRef_ex, ss_sfRef_ex, label='SS excel')
#plt.plot(x, ss_sfMB, label='SS OCI')
plt.title('Steady State (t=20[s]) v=1, sig=0.25, N=100, c=0')
plt.legend()

plt.savefig('timeTest_ss.png')

'''
x = np.linspace(0, L, int(N_mesh*2))
zippy = np.zeros(N_mesh*2)
plt.figure(4)
#plt.plot(x, ang_flux[:,0], label='MB 0')
#plt.plot(x, ang_flux[:,1], label='MB 1')
#plt.plot(x, ang_flux[:,2], label='MB 2')
#plt.plot(x, ang_flux[:,3], label='MB 3')
plt.plot(x, scalar_flux[:,0], label='Euler 0')
plt.plot(x, scalar_flux[:,1], label='Euler 1')
plt.plot(x, scalar_flux[:,2], label='Euler 2')
plt.plot(x, scalar_flux[:,3], label='Euler 3')
#plt.plot(x, zippy, '-r')
plt.xlabel('Distance [cm]')
plt.ylabel('Scalar Flux [units of scalar flux]')
plt.title('Scalar Flux through time')
plt.legend()
plt.show()'''
