import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=np.inf)

"""Thomson's Rule for First-Time Telescope Makers: It is faster to make a 
four-inch mirror then a six-inch mirror than to make a six-inch mirror."""

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
    S = np.zeros([2*N,2*N])
    beta = dx*xsec_scattering/4
    for i in range(N):
        for j in range(N):
            S[2*i,2*j] = beta * w[i]
            S[2*i+1,2*j+1] = beta * w[i]
    return(S)



xsec = 0.25
scattering_ratio = 0 #0.5
xsec_scattering = xsec*scattering_ratio

printer = False
printer_TS = False

dx = 0.01
L = 10
N = int(L/dx)
N_mesh = int(2*N)
Q = 0
S = Q

dt = 0.1
max_time = 7 #dt*(N_time-1)
N_time = int(max_time/dt)

v = 2

#BCs incident iso
BCl = 1
BCr = 0

angular_flux      = np.zeros([2, N_mesh])
angular_flux_next = np.zeros([2, N_mesh])
angular_flux_midstep = np.zeros([2, N_mesh])
angular_flux_last = np.zeros([2, N_mesh])

angular_flux_final = np.zeros([2, int(N_mesh), N_time])

mu1 = -0.57735
mu2 = 0.57735

w1 = 1
w2 = 1
w = np.array([w1, w2])

N_angle = 2

tol = 1e-9
error = 1
max_itter = 100000

manaz = dx*xsec_scattering/4
gamma = xsec*dx/2

final_angular_flux_solution = np.zeros([N_time, N_angle, N_mesh])
final_angular_flux_midstep_solution = np.zeros([N_time, N_angle, N_mesh])


def run_mb(final_angular_flux_midstep_solution):
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
            #print(itter)
            if itter == max_itter:
                print('Crap: {0}'.format(k))

            if (printer):
                print()
                print("========================================")
                print("next cycle: {0}".format(itter))
                print("========================================")
                print()

            # TODO: OCI
            for i in range(N):
                i_l = int(2*i)
                i_r = int(2*i+1)

                A = np.zeros([8,8])
                c = np.zeros([8,1])
                
                A_small = A_neg(dx, v, dt, mu1, xsec)
                S_small = scatter_source(dx, xsec_scattering, N_angle, w)
                #print('>>> S_small <<<')
                #print(S_small)
                #print()

                assert ((A_small.size == S_small.size))

                A[:4, :4] = A_small - S_small

                A_small = A_pos(dx, v, dt, mu2, xsec)
                S_small = scatter_source(dx, xsec_scattering, N_angle, w)

                assert ((A_small.size == S_small.size))

                A[4:, 4:] = A_small - S_small

                psi_halfLast_L = final_angular_flux_midstep_solution[k-1, :, i_l] # known
                psi_halfLast_R = final_angular_flux_midstep_solution[k-1, :, i_r] # known

                # boundary conditions
                if i == 0:  #left
                    psi_rightBound = angular_flux_last[0, i_r+1] # iterating on (unknown)
                    psi_leftBound =  BCl # known

                    psi_halfNext_rightBound = angular_flux_midstep_last[0, i_r+1] # iterating on (unknown)
                    psi_halfNext_leftBound  = BCl # known

                elif i == N-1: #right
                    psi_rightBound = BCr # known
                    psi_leftBound =  angular_flux_last[1, i_l-1] # iterating on (unknown)

                    psi_halfNext_rightBound = BCr # known
                    psi_halfNext_leftBound  = angular_flux_midstep_last[1, i_l-1] # iterating on (unknown)

                else: #middles
                    psi_rightBound = angular_flux_last[0, i_r+1] # iterating on (unknown)
                    psi_leftBound =  angular_flux_last[1, i_l-1] # iterating on (unknown)

                    psi_halfNext_rightBound = angular_flux_midstep_last[0, i_r+1] # iterating on (unknown)
                    psi_halfNext_leftBound  = angular_flux_midstep_last[1, i_l-1] # iterating on (unknown)

                #       c_neg(dx, v, dt, mu, Ql, Qr, Q_halfNext_L, Q_halfNext_R, psi_halfLast_L, psi_halfLast_R, psi_rightBound, psi_halfNext_rightBound)
                c[:4] = c_neg(dx, v, dt, mu1, Q, Q, Q, Q, psi_halfLast_L[0], psi_halfLast_R[0], psi_rightBound, psi_halfNext_rightBound)
                c[4:] = c_pos(dx, v, dt, mu2, Q, Q, Q, Q, psi_halfLast_L[1], psi_halfLast_R[1], psi_leftBound, psi_halfNext_leftBound)

                if (printer):
                    print("Large cell {0}".format(i))
                    print('>>> psi_right bound (BC) <<<')
                    print(psi_rightBound)
                    print()
                    print('>>> c vector <<<')
                    print(c)
                    print()
                    print('>>> full A mat <<<')
                    print(A)
                    print()

                angular_flux_raw = np.linalg.solve(A,c)

                # resorting into proper locations in solution vectors
                for p in range(N_angle):
                    angular_flux[p,i_l]         = angular_flux_raw[4*p]
                    angular_flux[p,i_r]         = angular_flux_raw[4*p+1]
                    
                    angular_flux_midstep[p,i_l] = angular_flux_raw[4*p+2]
                    angular_flux_midstep[p,i_r] = angular_flux_raw[4*p+3]

                if (printer):
                    print('>>> raw solution <<<')
                    print(angular_flux_raw)
                    print()
                    print('>>> angular flux eos reorganized <<<')
                    print(angular_flux)
                    print()
                    print('>>> angular flux mid step reorganized <<<')
                    print(angular_flux_midstep)
                    print()


                itter += 1 
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


def run_euler():
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


        #print()
        #print("========================================")
        #print("time step!")
        #print("========================================")
        #print()

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

    final_scalar_flux = np.zeros([N_time, N_mesh])

    for i in range(N_time):
        for j in range(N_mesh):
            final_scalar_flux[i,j] = final_angular_flux_solution[i,0,j] + final_angular_flux_solution[i,1,j]

    return(final_scalar_flux)


run_mb(final_angular_flux_midstep_solution)
print('mb_done')
final_scalar_flux_e = run_euler()

final_scalar_flux = np.zeros([N_time, N_mesh])
for i in range(N_time):
    for j in range(N_mesh):
        final_scalar_flux[i,j] = final_angular_flux_midstep_solution[i,0,j] + final_angular_flux_midstep_solution[i,1,j]

f=1
X = np.linspace(0, L, int(N_mesh))

import scipy.special as sc
def phi_(x,t):
    if x > v*t:
        return 0.0
    else:
        return 1.0/BCl * (xsec*x*(sc.exp1(xsec*v*t) - sc.exp1(xsec*x)) + \
                        np.e**(-xsec*x) - x/(v*t)*np.e**(-xsec*v*t))


def psi_(x, t):
    if x> v*t*mu2:
        return 0.0
    else:
        return 1/BCl*np.exp(-xsec * x / mu2)

def analitical(x, t):
    print(mu2)
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
line2, = ax.plot(X, final_scalar_flux_e[0,:], '-r',label="BE-SCB")
line3, = ax.plot(X, analitical(X,0), '--*g',label="Ref")
text   = ax.text(8.0,0.75,'') #, transform=ax.transAxes
ax.legend()
plt.ylim(-0.2, 1.2*BCl) #, OCI_soultion[:,0], AZURV1_soultion[:,0]

def animate(k):
    line1.set_ydata(final_scalar_flux[k,:])
    line2.set_ydata(final_scalar_flux_e[k,:])
    line3.set_ydata(analitical(X,k*dt))
    #ax.set_title(f'Scalar Flux (ϕ) at t=%.1f'.format(dt*k)) #$\bar{\phi}_{k,j}$ with 
    text.set_text(r'$t \in [%.1f,%.1f]$ s'%(dt*k,dt*(k+1)))
    #print('Figure production percent done: {0}'.format(int(k/N_time)*100), end = "\r")
    return line1, line2, line3

simulation = animation.FuncAnimation(fig, animate, frames=N_time)
#plt.show()

writervideo = animation.PillowWriter(fps=250)
simulation.save('both.gif') #saveit!