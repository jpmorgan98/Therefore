import numpy as np
import matplotlib.pyplot as plt

printer = False

xsec = .25
scattering_ratio = 0
xsec_scattering = xsec*scattering_ratio

dx = 0.01
L = 10
N = int(L/dx)
N_mesh = 2*N
S = 0

dt = 0.1
t_max = 17
N_time = int(t_max/dt)
v = 1

#BCs incident iso
BCl = 1
BCr = 0

mu1 = -1 #-0.57735
mu2 = 1 #0.57735
[angles, weights] = np.polynomial.legendre.leggauss(N_angles)

tol = 1e-6
max_itter = 100000


manaz = dx*xsec_scattering/4
gamma = xsec*dx/2


final_angular_flux_solution = np.zeros([N_time, 2, N_mesh])
theta = 1

angular_flux      = np.zeros([2, int(N_mesh)])
angular_flux_next = np.zeros([2, int(N_mesh)])
angular_flux_last = np.zeros([2, int(N_mesh)])

for k in range(N_time):

    xsec_t = xsec + (1/(v* theta* dt)) # time augmented scattering crossection

    if k > 1: #source from last time step
        S_t = S + final_angular_flux_solution[k-1,:,:]/(v* theta* dt)
    else:
        S_t = S + np.zeros([2, int(N_mesh)])

    error = 1
    itter = 0

    manaz = dx*xsec_scattering/4
    gamma = xsec_t*dx/2


    print()
    print("========================================")
    print("time step!")
    print("========================================")
    print()

    while error > tol and max_itter > itter:

        if printer:
            print()
            print("========================================")
            print("next cycle")
            print("========================================")
            print()

        # TODO: OCI
        for i in range(N):
            i_l = int(i*2)
            i_r = int(i*2)+1

            A = np.zeros([4,4])
            b = np.zeros([4,1])

            A = np.array([[-mu1/2 - w1*manaz + gamma, mu1/2,                    -w2*manaz,                 0],
                        [-mu1/2,                    -mu1/2 - w1*manaz + gamma,  0,                       -w2*manaz],
                        [-w1*manaz,                 0,                         mu2/2 + gamma - w2*manaz, mu2/2],
                        [0,                         -w1*manaz,                 -mu2/2,                   mu2/2 + gamma - w2*manaz]])

            if i == 0: #left bc
                b = np.array([[dx/2*S_t[0,i_l]],
                              [dx/2*S_t[0,i_r] - mu1 * angular_flux[0, i*2+2]],
                              [dx/2*S_t[1,i_l] + mu2 * BCl],
                              [dx/2*S_t[1,i_r]]])
            elif i == N-1: #right bc
                b = np.array([[dx/2*S_t[0,i_l]],
                              [dx/2*S_t[0,i_r] - mu1 * BCr],
                              [dx/2*S_t[1,i_l] + mu2 * angular_flux[1, i*2-1]],
                              [dx/2*S_t[1,i_r]]])
            else: #mid communication
                b = np.array([[dx/2*S_t[0,i_l]],
                              [dx/2*S_t[0,i_r] - mu1 * angular_flux[0, i*2+2]],
                              [dx/2*S_t[1,i_l] + mu2 * angular_flux[1, i*2-1]],
                              [dx/2*S_t[1,i_r]]])
            

            angular_flux_next[:,2*i:2*i+2] = np.linalg.solve(A,b).reshape(-1,2)


            if printer:
                print("Large cell %d".format(i))
                print(b)
                print()
                print(A)
                print()
                print(angular_flux_next[:,2*i:2*i+2])
                print()

            if max_itter-2 < itter:
                print(">>>>WARNING<<<<<")
                print("     max itter hit")
                print("     {0}".format(itter))

        itter += 1 

        # TODO: Error
        if itter > 2:
            error = np.linalg.norm(angular_flux_next - angular_flux, ord=2)

        angular_flux_last = angular_flux
        angular_flux = angular_flux_next

    final_angular_flux_solution[k,:,:] = angular_flux

    print(itter)

final_scalar_flux = np.zeros([N_time, N_mesh])
for i in range(N_time):
    for j in range(N_mesh):
        final_scalar_flux[i,j] = final_angular_flux_solution[i,0,j] + final_angular_flux_solution[i,1,j]


f=1
X = np.linspace(0, L, int(N_mesh))
plt.figure(f)
plt.plot(X, angular_flux[0,:],  '-*k',  label='OCI 1')
plt.plot(X, angular_flux[1,:],  '--*k', label='OCI 2')
#plt.plot(X, scalar_flux2[0,:], '-r',  label='SI 1')
#plt.plot(X, scalar_flux2[1,:], '--r', label='SI 2')
plt.title('Test Flux')
plt.xlabel('Distance')
plt.ylabel('Angular Flux')
plt.show()
#plt.savefig('Test Angular flux')

#


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

writervideo = animation.PillowWriter(fps=250)
simulation.save('transport_into_slab_scb_be.gif') #saveit!