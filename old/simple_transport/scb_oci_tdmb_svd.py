import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=np.inf)

def A_neg(dx, v, dt, mu, xsec_total):
    gamma = (dx*xsec_total)/2
    timer = dx/(v*dt)
    timer2 = dx/(2*v*dt)
    a = mu/2

    A_n = np.array([[-a + gamma, a,          timer2,            0],
                    [-a,         -a + gamma, 0,                 timer2],
                    [-timer,     0,          timer - a + gamma, a],
                    [0,          -timer,     -a,                timer -a + gamma]])
    
    return(A_n)



def A_pos(dx, v, dt, mu, xsec_total):
    gamma = (dx*xsec_total)/2
    timer = dx/(v*dt)
    timer2 = dx/(2*v*dt)
    a = mu/2

    A_p = np.array([[a + gamma, a,         timer2,            0],
                    [-a,        a + gamma, 0,                 timer2],
                    [-timer,    0,         timer + a + gamma, a],
                    [0,         -timer,    -a,                timer +a + gamma]])

    return(A_p)



def c_neg(dx, v, dt, mu, Ql, Qr, Q_halfNext_L, Q_halfNext_R, psi_halfLast_L, psi_halfLast_R, psi_rightBound, psi_halfNext_rightBound):
    timer2 = dx/(v*dt*2)

    c_n = np.array([[dx/1*Ql + timer2*psi_halfLast_L],
                    [dx/1*Qr + timer2*psi_halfLast_R - mu* psi_rightBound],
                    [dx/1*Q_halfNext_L],
                    [dx/1*Q_halfNext_R - mu*psi_halfNext_rightBound]])

    return(c_n)



def c_pos(dx, v, dt, mu, Ql, Qr, Q_halfNext_L, Q_halfNext_R, psi_halfLast_L, psi_halfLast_R, psi_leftBound, psi_halfNext_leftBound):
    timer2 = dx/(v*dt*2)

    c_p = np.array([[dx/1*Ql + timer2*psi_halfLast_L + mu * psi_leftBound],
                    [dx/1*Qr + timer2*psi_halfLast_R],
                    [dx/1*Q_halfNext_L + mu*psi_halfNext_leftBound],
                    [dx/1*Q_halfNext_R]])
    
    return(c_p)



def scatter_source(dx, xsec_scattering, N, w):
    S = np.zeros([4*N,4*N])
    beta = dx*xsec_scattering/4

    for i in range(N):
        for j in range(N):
            S[i*4, j*4]     = beta*w[j]
            S[i*4+1, j*4+1] = beta*w[j]
            S[i*4+2, j*4+2] = beta*w[j]
            S[i*4+3, j*4+3] = beta*w[j]
    return(S)





def DMD_est(matvec, x0, K = 10):
   
    N = x0.size
    #if x.size != b.size:
    #    x = b.copy()
    #x_new = x*0
    #x_orig = x.copy()
    #x_0 = x.copy()

    x_old = x0
    
    #perform K iterations of matvec
    Yplus = np.zeros((N,K-1))
    Yminus = np.zeros((N,K-1))
   
    for k in range(K):
        x_new = matvec(x_old)
        if (k < K-1):
            Yminus[:,k] = x_new
            x_old = x_new.copy()
        if (k>0):
            Yplus[:,k-1] = x_new
           
    #compute svd
    [u,s,v] = np.linalg.svd(Yminus,full_matrices=True)
    #find the non-zero singular values
    if (x0.size > 1) and (s[(1-np.cumsum(s)/np.sum(s)) > 1.e-12].size >= 1):
        spos = s[(1-np.cumsum(s)/np.sum(s)) > 1.e-12].copy()
    else:
        spos = s.copy()
    #create diagonal matrix
    mat_size = np.min([K,len(spos)])
    S = np.zeros((mat_size,mat_size))
   
    #select the u and v that correspond with the nonzero singular values
    unew = 1.0*u[:,0:mat_size]
    vnew = 1.0*v[0:mat_size,:]
    #S will be the inverse of the singular value diagonal matrix
    S[np.diag_indices(mat_size)] = 1/spos

    #the approximate A operator is Ut A U = Ut Y+ V S
    part1 = np.dot(np.matrix(unew).getH(),Yplus)
    part2 = np.dot(part1,np.matrix(vnew).getH())
    Atilde = np.dot(part2,np.matrix(S).getH())

    return Atilde


xsec = 1
scattering_ratio = 0.9
xsec_scattering = xsec*scattering_ratio

printer = False
printer_TS = False

mfp = 0.25

dx = mfp/xsec
L = 100
N = int(L/dx)
print(N)
N_mesh = int(2*N)
Q = 0

N_angles = 8

dt = 0.1
max_time = 1 #dt*(N_time-1)
N_time = 2

v = 5

#BCs incident iso
BCl = 0
BCr = 0

angular_flux      = np.zeros([2, N_mesh])
angular_flux_next = np.zeros([2, N_mesh])
angular_flux_midstep = np.zeros([2, N_mesh])
angular_flux_last = np.zeros([2, N_mesh])

angular_flux_final = np.zeros([2, int(N_mesh), N_time])


N_angle = N_angles
[angles, weights] = np.polynomial.legendre.leggauss(N_angles)
#mu1 = -0.57735
#mu2 = 0.57735

#w1 = 1
#w2 = 1
#w = np.array([w1, w2])

tol = 1e-13
error = 1
max_itter = int(1e4)


Q_sweep = np.zeros()

final_angular_flux_solution = np.zeros([N_time, N_angle, N_mesh])
final_angular_flux_midstep_solution = np.zeros([N_time, N_angle, N_mesh])

def soa_transport(aflux_last):
    aflux_next = np.zeros_like(aflux_last)

    for i in range(N_mesh):
        for m in range(N_angle):
            if angles[m] > 0:
                aflux_next[0] = BCl + Q_sweep[0]
            elif angles[m] < 0:
                aflux_next[0] = BCr + Q_sweep[0]
    error = aflux_next - aflux_last
    aflux_next = aflux_last + error

    return(aflux_next)


def oci_iteration(aflux_last):

    aflux = np.zeros(4*N_mesh*N_angles)
    # OCI
    for i in range(N):
        #print('>>>>cell {0}<<<<'.format(i))

        A = np.zeros([N_angles*4,N_angles*4])
        c = np.zeros([4*N_angles,1])

        for m in range(N_angles):
            psi_halfLast_L = 0#final_angular_flux_midstep_solution[k-1, m, i_l] # known
            psi_halfLast_R = 0#final_angular_flux_midstep_solution[k-1, m, i_r] # known
            
            if angles[m] < 0:
                if i == N-1:
                    psi_rightBound          = BCr
                    psi_halfNext_rightBound = BCr
                else:
                    psi_rightBound          = aflux_last[4*N_angles*(i+1) + 4*m + 0]
                    psi_halfNext_rightBound = aflux_last[4*N_angles*(i+1) + 4*m + 2] 

                A_small = A_neg(dx, v, dt, angles[m], xsec)
                c_small = c_neg(dx, v, dt, angles[m], Q, Q, Q, Q, psi_halfLast_L, psi_halfLast_R, psi_rightBound, psi_halfNext_rightBound)

            elif angles[m] > 0:
                if i == 0:
                    psi_leftBound           = BCl
                    psi_halfNext_leftBound  = BCl
                else:
                    psi_leftBound           = aflux_last[4*N_angles*(i-1) + 4*m + 1]
                    psi_halfNext_leftBound  = aflux_last[4*N_angles*(i-1) + 4*m + 3]

                A_small = A_pos(dx, v, dt, angles[m], xsec)
                c_small = c_pos(dx, v, dt, angles[m], Q, Q, Q, Q, psi_halfLast_L, psi_halfLast_R, psi_leftBound, psi_halfNext_leftBound)

            else:
                print('>>>>>Error')

            A[m*4:(m+1)*4, m*4:(m+1)*4] = A_small
            c[m*4:(m+1)*4] = c_small

        S = scatter_source(dx, xsec_scattering, N_angle, weights)

        A = A - S

        aflux_raw_small = np.linalg.solve(A,c)

        for p in range(N_angles):
            aflux[4*N_angles*(i) + 4*p + 0] = aflux_raw_small[4*p + 0]
            aflux[4*N_angles*(i) + 4*p + 1] = aflux_raw_small[4*p + 1]
            aflux[4*N_angles*(i) + 4*p + 2] = aflux_raw_small[4*p + 2]
            aflux[4*N_angles*(i) + 4*p + 3] = aflux_raw_small[4*p + 3]

    return(aflux)



start_iteration = np.array([0, 5,10,20,30, 40, 80])
K = 5


for r in range(start_iteration.size):
    aflux_0 = np.random.random(N_mesh*N_angles*4)

    for t in range(start_iteration[r]):
        aflux_0 = oci_iteration(aflux_0).copy()

    Atilde = DMD_est(oci_iteration, aflux_0, K = K)

    eig, eig_vec = np.linalg.eig(Atilde)
    max_eigval = eig[np.abs(eig).argmax()]

    plt.figure(1)
    plt.plot(eig.real, eig.imag, 'k.')
    plt.plot(max_eigval.real, max_eigval.imag, 'rX', markersize=15, label=r'$\rho$ {:.4f} ({:.3f})'.format(np.abs(max_eigval), max_eigval))
    plt.title(r"λ from DMD k={} (iterations [{}, {}]) ($δ={}$, S$_{}$, $c={}$, $Δt={}$)".format(K, start_iteration[r], K+start_iteration[r]+1, mfp, N_angles, scattering_ratio, dt))
    plt.xlabel(r"$λ_r$")
    plt.ylabel(r"$λ_i$")
    plt.legend()
    #plt.show()
    plt.savefig("dmd_eig_{}.png".format(start_iteration[r]))
    plt.clf()

"""
K = np.array([5, 10, 15, 20, 40, 80])

for t in range(K.size):
    print(K[t])

    Atilde = DMD_est(oci_iteration, aflux_0, K = K[t])


    eig, eig_vec = np.linalg.eig(Atilde)
    max_eigval = eig[np.abs(eig).argmax()]

    plt.figure(1)
    plt.plot(eig.real, eig.imag, 'k.')
    plt.plot(max_eigval.real, max_eigval.imag, 'rX', markersize=15, label=r'$\rho$ {:.4f} ({:.3f})'.format(np.abs(max_eigval), max_eigval))
    plt.title(r"λ from DMD (k={}) ($δ={}$, S$_{}$, $c={}$, $Δt={}$)".format(K[t], mfp, N_angles, scattering_ratio, dt))
    plt.xlabel(r"$λ_r$")
    plt.ylabel(r"$λ_i$")
    plt.legend()
    plt.savefig("dmd_eig_{}.png".format(K[t]))
    plt.clf()

"""
'''
# the zeroth stored solution is the initial condition
for k in range(1, N_time, 1):

    if (printer_TS):
        print()
        print("========================================")
        print("next time step: {0}".format(k))
        print("========================================")
        print()
        print(final_angular_flux_solution[k-1,:,:])

    #initial guesses?
    itter = 0
    error = 1
    error_last = 1
    converged = False

    aflux_last = np.random.random(N_mesh*N_angles*4)

    spec_rad = np.zeros(1)

    aflux_list = np.zeros(1)

    while not converged:

        aflux = oci_iteration(aflux_last)

        if itter > 1:
            error = np.linalg.norm(aflux-0, 2)

        spec_rad = np.append(spec_rad,error/error_last)

        if error<tol*(1-spec_rad[-1]):
            converged = True

        aflux_last = aflux.copy()
        error_last = error

        print("l {}, ρ {}, error {}".format(itter, spec_rad[-1], error))

        itter += 1

'''

#aflux_list = aflux_list[1:]
#spec_rad = spec_rad[1:]

#K = itter-1

#plt.figure()
#plt.plot(x, spec_rad)
#plt.ylim(-0.1,1)
#plt.show()



'''
final_scalar_flux = np.zeros([N_time, N_mesh])

for i in range(N_time):
    for j in range(N_mesh):
        for m in range(N_angle):
            final_scalar_flux[i,j] += (weights[m] * final_angular_flux_midstep_solution[i,m,j])

X = np.linspace(0, L, int(N_mesh))

X = np.linspace(0, L, int(N_mesh))


f=1
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
    if x > v*t:
        return 0.0
    else:
        return 1.0/BCl * (xsec*x*(sc.exp1(xsec*v*t) - sc.exp1(xsec*x)) + \
                        np.e**(-xsec*x) - x/(v*t)*np.e**(-xsec*v*t))

mu2 = 0.57
def psi_(x, t):
    if x> v*t*mu2:
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
#line2, = ax.plot(X, analitical(X,0), '--*g',label="Ref")
text   = ax.text(8.0,0.75,'') #, transform=ax.transAxes
ax.legend()
plt.ylim(-0.2, 1.5) #, OCI_soultion[:,0], AZURV1_soultion[:,0]

def animate(k):
    line1.set_ydata(final_scalar_flux[k,:])
    #line2.set_ydata(analitical(X,k*dt))
    #ax.set_title(f'Scalar Flux (ϕ) at t=%.1f'.format(dt*k)) #$\bar{\phi}_{k,j}$ with 
    text.set_text(r'$t \in [%.1f,%.1f]$ s'%(dt*k,dt*(k+1)))
    #print('Figure production percent done: {0}'.format(int(k/N_time)*100), end = "\r")
    return line1, #line2,

simulation = animation.FuncAnimation(fig, animate, frames=N_time)
#plt.show()

writervideo = animation.PillowWriter(fps=1000)
simulation.save('transport_into_slab.gif') #saveit!
'''