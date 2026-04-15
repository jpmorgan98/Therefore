import numpy as np
import matplotlib.pyplot as plt
import therefore.src.multiBalance.scb_oci_mb as t
#from .scb_oci_mb import OCIMBRun
np.set_printoptions(linewidth=np.inf)

xsec = 0.25
scattering_ratio = 0
xsec_scattering = xsec*scattering_ratio

printer = False
printer_TS = False

dx = 0.5
L = 1
N = int(L/dx)
N_mesh = int(2*N)
Q = 0

dt = 0.1
max_time = 0.5 #dt*(N_time-1)
N_time = int(max_time/dt)

v = 1.0

#BCs incident iso
BCl = np.array([0,1])
BCr = np.array([0,0])

angular_flux      = np.zeros([2, N_mesh])
angular_flux_next = np.zeros([2, N_mesh])
angular_flux_midstep = np.zeros([2, N_mesh])
angular_flux_last = np.zeros([2, N_mesh])

angular_flux_final = np.zeros([2, int(N_mesh), N_time])

mu1 = -0.57735
mu2 = 0.57735
mu = np.array([mu1, mu2])

w1 = 1
w2 = 1
weight = np.array([w1, w2])

w = np.array([w1, w2])

N_angle = 2

tol = 1e-9
error = 1
max_itter = 100000

manaz = dx*xsec_scattering/4
gamma = xsec*dx/2

dtype = np.double

final_angular_flux_solution = np.zeros([N_time, N_angle, N_mesh])
final_angular_flux_midstep_solution = np.zeros([N_time, N_angle, N_mesh])

source = np.zeros([N_angle, N_mesh], dtype=dtype)
xsec_m = xsec*np.ones(N_mesh, dtype=dtype)
xsec_scat_m = xsec_scattering*np.ones(N_mesh, dtype=dtype)
dx_m = dx*np.ones(N_mesh, dtype=dtype)

# the zeroth stored solution is the initial condition
for k in range(1, N_time, 1):

    if (printer_TS):
        print()
        print("========================================")
        print("next time step!")
        print("========================================")
        print()

    # iterating on these till convergence
    angular_flux      = np.zeros([2, N_mesh]) 
    angular_flux_last = np.zeros([2, N_mesh])   # last refers to last iteration
    angular_flux_midstep = np.zeros([2, N_mesh])
    angular_flux_midstep_last = np.zeros([2, N_mesh])   # last refers to last iteration

    #initial guesses?
    itter = 0
    error_eos = 1
    while error_eos > tol and max_itter > itter:

        [angular_flux, angular_flux_midstep] = t.OCIMBRun(final_angular_flux_midstep_solution[k-1:,:], 
                                                        angular_flux_last, angular_flux_midstep_last, source, xsec_m, xsec_scat_m, 
                                                        dx_m, dt, v, mu, weight, BCl, 0)
        #print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        #print(itter)
        #print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        # TODO: Error
        if itter > 2:
            error_eos = np.linalg.norm(angular_flux_midstep - angular_flux_midstep_last, ord=2)
            error_mos = np.linalg.norm(angular_flux - angular_flux_last, ord=2)

        final_angular_flux_solution[k, :, :] = angular_flux
        final_angular_flux_midstep_solution[k, :, :] = angular_flux_midstep
            
        angular_flux_last = angular_flux 
        angular_flux_midstep_last = angular_flux_midstep

        itter += 1 

    #print(itter)

final_scalar_flux = np.zeros([N_time, N_mesh])
for i in range(N_time):
    for j in range(N_mesh):
        final_scalar_flux[i,j] = final_angular_flux_midstep_solution[i,0,j] + final_angular_flux_midstep_solution[i,1,j]


'''
f=1
X = np.linspace(0, L, int(N_mesh))
plt.figure(f)
plt.plot(X, final_angular_flux_solution[1, 1,:],  '--*g', label='0')
plt.plot(X, final_angular_flux_midstep_solution[1, 1,:],  '-*g',  label='0 + 1/2')
plt.plot(X, final_angular_flux_solution[2, 1,:],  '--*k', label='1')
plt.plot(X, final_angular_flux_midstep_solution[2, 1,:],  '-*k',  label='1 + 1/2')
plt.plot(X, final_angular_flux_solution[3, 1,:],  '--*r', label='2')
plt.plot(X, final_angular_flux_midstep_solution[3, 1,:],  '-*r',  label='2 + 1/2')
plt.plot(X, final_scalar_flux[-1,:])
#plt.plot(X, final_angular_flux_midstep_solution[-1, 1,:],  '-*b',  label='3 + 1/2')
#plt.plot(X, scalar_flux2[0,:], '-r',  label='SI 1')
#plt.plot(X, scalar_flux2[1,:], '--r', label='SI 2')
plt.title('Test Ang Flux: Positive ordinant')
plt.xlabel('Distance')
plt.ylabel('Angular Flux')
plt.legend()
#plt.show()
plt.savefig('Test Angular flux')

import scipy.special as sc
def phi_(x,t):
    v=1
    if x > v*t:
        return 0.0
    else:
        return 1.0/BCl * (xsec*x*(sc.exp1(xsec*v*t) - sc.exp1(xsec*x)) + \
                        np.e**(-xsec*x) - x/(v*t)*np.e**(-xsec*v*t))


def psi_(x, t):
    v=2
    if x> v*t:
        return 0.0
    else:
        return 1/BCl*np.exp(-xsec * x / mu2)

def analitical(x, t):
    y = np.zeros(x.shape)
    for i in range(x.size):
        y[i] = psi_(x[i],t)
    return y

import matplotlib.animation as animation

fig,ax = plt.subplots() #plt.figure(figsize=(6,4))
    
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\psi$')
ax.set_title('Angular Flux (ψ)')

line1, = ax.plot(X, final_scalar_flux[0,:], '-k',label="MB-SCB")
line2, = ax.plot(X, analitical(X,0), '--*g',label="Ref")
text   = ax.text(8.0,0.75,'') #, transform=ax.transAxes
ax.legend()
plt.ylim(-0.2, 1.2*BCl) #, OCI_soultion[:,0], AZURV1_soultion[:,0]

def animate(k):
    line1.set_ydata(final_scalar_flux[k,:])
    line2.set_ydata(analitical(X,k*dt))
    #ax.set_title(f'Scalar Flux (ϕ) at t=%.1f'.format(dt*k)) #$\bar{\phi}_{k,j}$ with 
    text.set_text(r'$t \in [%.1f,%.1f]$ s'%(dt*k,dt*(k+1)))
    #print('Figure production percent done: {0}'.format(int(k/N_time)*100), end = "\r")
    return line1, line2,

simulation = animation.FuncAnimation(fig, animate, frames=N_time)
#plt.show()

writervideo = animation.PillowWriter(fps=1000)
simulation.save('transport_into_slab.gif') #saveit!'''