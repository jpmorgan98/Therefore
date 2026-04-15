import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import numba as nb

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

#N_cell = 100
N_angle = 8
dx = 1.0

L = 25
N_cell = int(L/dx)

N = 2*N_cell*N_angle

sigma = 1
sigma_s = 0*sigma
sigma_a = sigma - sigma_s

D = 1/(3*sigma)

tol = 1e-9
max_it = int(500)

SET_palmer_acc = True
printer = True

[angles, weights] = np.polynomial.legendre.leggauss(N_angle)

af_l = 0
af_r = 0

sf_l = 0
sf_r = 0

second_l = 0
second_r = 0

source = 0

#@nb.jit
def Sbuild():
    S = np.zeros((2*N_angle,2*N_angle))
    #for m in range(N_angle):

    beta = sigma_s * dx / 4

    for p in range(N_angle):
        for j in range(N_angle):
            S[p*2,   j*2]   = beta*weights[j]
            S[p*2+1, j*2+1] = beta*weights[j]

    return(S)

#@nb.jit
def Ablock(mu):
    return(
        np.array(((np.abs(mu)/2 + sigma*dx/2, -mu/2),
                  (mu/2, np.abs(mu)/2 + sigma*dx/2)))
    )

#@nb.jit
def buildA():
    A = np.zeros((2*N_angle, 2*N_angle))
    for a in range(N_angle):
        A[a*2:(a+1)*2,a*2:(a+1)*2] = Ablock(angles[a])

    S = Sbuild()

    A.shape == S.shape

    A = A-S

    return(A)

#@nb.jit
def b_neg(psi_rightBound, mu):
    b_n = np.array((0,-mu*psi_rightBound))
    return(b_n)

#@nb.jit
def b_pos(psi_leftBound, mu):
    b_p = np.array((mu*psi_leftBound, 0))
    return(b_p)

#@nb.jit
def buildb(aflux_last, i):

    b = np.zeros(2*N_angle)

    for m in range(N_angle):
        if angles[m]>0:
            if (i==0):
                af_lb = 0
            else:
                af_lb = aflux_last[2*N_angle*(i-1)+1]

            b[m*2:(m+1)*2] = b_pos(af_lb, angles[m])

        elif angles[m]<0:
            if (i==N_cell-1):
                af_rb = 0
            else:
                af_rb = aflux_last[2*N_angle*(i+1)+0]

            b[m*2:(m+1)*2] = b_neg(af_rb, angles[m])

    return(b)

#@nb.jit
def af_vec2mat(af_vec):
    # so I can use np.sum to compute the accelerations

    af_mat = np.zeros((N_angle, 2*N_cell))

    for j in range(N_cell):
        for m in range(N_angle):
            af_mat[m, 2*j] = af_vec[j*2*N_angle + m*2]
            af_mat[m, 2*j + 1] = af_vec[j*2*N_angle + m*2 + 1]
    return(af_mat)

#@nb.jit
def compute_moments(af):

    N_mom = 2*N_cell

    zeroth = np.zeros(N_mom)
    first = np.zeros(N_mom)
    second = np.zeros(N_mom)

    for j in range(N_cell):
        for m in range(N_angle):
            zeroth[2*j] += weights[m] * af[j*N_angle*2 + m*2]
            first[2*j]  += weights[m] * angles[m] * af[j*N_angle*2 + m*2]
            second[2*j] += weights[m] * .5*(3*angles[m]**2 - 1) * af[j*N_angle*2 + m*2]

            zeroth[2*j+1] += weights[m] * af[j*N_angle*2 + m*2 +1]
            first[2*j+1]  += weights[m] * angles[m] * af[j*N_angle*2 + m*2 +1]
            second[2*j+1] += weights[m] * .5*(3*(angles[m]**2) - 1) * af[j*N_angle*2 + m*2 + 1]

    
    return(zeroth, first, second)

#
# @nb.jit
def build_diff_tri_diag():
    A = np.zeros((2*N_cell, 2*N_cell))
    delta = 1/2
    gamma = 1/4

    for i in range(N_cell):
        A[2*i,2*i]   = (1-delta)*D/dx + gamma + sigma_a*dx/2
        A[2*i,2*i+1] = -(1-delta)*D/dx

        A[2*i+1,2*i]   = -(1-delta)*D/dx 
        A[2*i+1,2*i+1] = (1-delta)*D/dx + gamma + sigma_a*dx/2

        if (i==N_cell-1): #right bound (no right of cell info, moved to b)
            A[2*i,2*(i-1)]   = -delta*D/dx
            A[2*i,2*(i-1)+1] = delta*D/dx - gamma
        elif (i==0): #left bound (no left of cell info, moved to b)
            A[2*i+1,2*(i+1)]   = -gamma+delta*D/dx
            A[2*i+1,2*(i+1)+1] = -delta*D/dx
        else: #interior cell
            #cell j-1 info
            A[2*i,2*(i-1)]   = -delta*D/dx
            A[2*i,2*(i-1)+1] = delta*D/dx - gamma
            # cell j+1 info=
            A[2*i+1,2*(i+1)]   = -gamma+delta*D/dx
            A[2*i+1,2*(i+1)+1] = -delta*D/dx

    return(A)

#@nb.jit
def palmer_acc(aflux):

    # compute angular moments
    zeroth, first, second = compute_moments(aflux)
    #second = np.zeros(2*N_cell)

    #print("aflux", aflux)
    #print("second", second)
    
    D = 1/(3*sigma)
    delta = 0.5
    gamma = 0.25

    # using simple equations from Palmer notes Eq (30) and (31)
    # 2nd mom modified 4 step DSA solve

    b_vec = np.zeros(2*N_cell)
    for i in range(N_cell):

        if i == 0: # left hand bound, using bcs
            b_l = dx/4*source + (1-delta)*4*D/dx * (second[2*i+1]-second[2*i]) - (1-delta)*4*D/dx * (second_l-second_l) - -delta*D/dx * (sf_l) - (delta*D/dx - gamma) * (sf_l)
            b_r = dx/4*source + (1-delta)*4*D/dx * (second[2*i+1]-second[2*i]) - (1-delta)*4*D/dx * (second[2*(i+1)+1]-second[2*(i+1)])
        elif i==N_cell-1: #right hand bcs
            b_l = dx/4*source + (1-delta)*4*D/dx * (second[2*i+1]-second[2*i]) - (1-delta)*4*D/dx * (second[2*(i-1)+1]-second[2*(i-1)])
            b_r = dx/4*source + (1-delta)*4*D/dx * (second[2*i+1]-second[2*i]) - (1-delta)*4*D/dx * (second_r-second_r) + (gamma+delta*D/dx + delta*D/dx)*sf_r
        else: # interior cell
            b_l = dx/4*source + (1-delta)*4*D/dx * (second[2*i+1]-second[2*i]) - (1-delta)*4*D/dx * (second[2*(i-1)+1]-second[2*(i-1)])
            b_r = dx/4*source + (1-delta)*4*D/dx * (second[2*i+1]-second[2*i]) - (1-delta)*4*D/dx * (second[2*(i+1)+1]-second[2*(i+1)])

        b_vec[2*i]   = b_l
        b_vec[2*i+1] = b_r
    
    #print(b_vec)
    
    diff = build_diff_tri_diag()

    #print(diff)
    #print(b_vec)

    sf_new = np.linalg.solve(diff, b_vec)

    #print(sf_new)

    # computing new current (first angular moment) (palmer eq 30/31)

    first_new = np.zeros_like(first)

    for i in range(N_cell):
        first_new[2*i]   = -1/(3*dx)*(sf_new[2*i+1]-sf_new[2*i]) - 4/(3*dx)*(second[2*i+1]-second[2*i])
        first_new[2*i+1] = -1/(3*dx)*(sf_new[2*i+1]-sf_new[2*i]) - 4/(3*dx)*(second[2*i+1]-second[2*i])

    # compute new updates with l+1/2 af, and new zeroth and first via yavuz clever update 

    aflux_new = np.zeros_like(aflux)

    #first_new[:] = np.zeros_like(zeroth)
    #sf_new[:]    = np.zeros_like(zeroth)

    #first_new[:] = first[:]
    #sf_new[:]    = zeroth[2*i]

    for i in range(N_cell):
        for m in range(N_angle):
            af_i = 2*i*N_angle + 2*m
            aflux_new[af_i]   = aflux[af_i]   + (1/2)*((sf_new[2*i]   - zeroth[2*i])   + 3*angles[m]*(first_new[2*i]   - first[2*i]))
            aflux_new[af_i+1] = aflux[af_i+1] + (1/2)*((sf_new[2*i+1] - zeroth[2*i+1]) + 3*angles[m]*(first_new[2*i+1] - first[2*i+1]))


    return(aflux_new)

#@nb.jit
def transport():

    aflux_last = np.random.random(N)
    aflux_last = np.ones(N)
    aflux = np.zeros(N)

    converged = False

    error = 1
    error_last = 1

    l = 0

    while (not converged):

        #oci loop
        for i in range(N_cell):

            A = buildA()
            b = buildb(aflux_last, i)

            aflux_cell = np.linalg.solve(A,b)

            for m in range(N_angle):
                aflux[i*N_angle*2 + m*2]   = aflux_cell[m*2]
                aflux[i*N_angle*2 + m*2+1] = aflux_cell[m*2+1]

        if (SET_palmer_acc):
            aflux = palmer_acc(aflux)
        
        error = np.linalg.norm(aflux-0, ord=2)
        spec_rad = error/error_last

        if l>1:
            if (error<tol*(1-spec_rad)):
                converged = True
        if l>max_it:
            converged = True
            print("warning: didn't converge after max iter")

        if (printer):
            print("l ",l," error ", error, " ρ ", spec_rad)
            #print("l {}, error {}, ρ {}".format(l, error, spec_rad))

        error_last = error
        aflux_last[:] = aflux[:]
        l += 1

    #print(aflux)

    return(spec_rad, l)


if __name__ == '__main__':
    #printer = True
    L = 10

   # mfp = 1.5
    
    sigma = 1
    sigma_s = sigma*.7
    sigma_a = sigma-sigma_s

    D = 1/(3*sigma)

    dx = .1
    N_cell = int(L/dx)
    N = 2*N_angle*N_cell

    SET_palmer_acc = True

    transport()
    
    exit()

    #SET_palmer_acc = True
    #printer = False

    N_mfp = 5
    N_c = 10

    mfp_range = np.logspace(-1,1,N_mfp) #0.01, 0.025, 0.05, 0.075, 
    c_range = np.linspace(0,1,N_c)

    print(mfp_range, c_range)

    spec_rad_pacc = np.zeros((N_mfp, N_c))
    spec_rad_oci = np.zeros((N_mfp, N_c))

    for k in range(N_mfp):
        for h in range(N_c):

            dx = mfp_range[k]/sigma
            N_cell = int(L/dx)
            N = 2*N_angle*int(L/dx)

            sigma_s = sigma*c_range[h]
            sigma_a = sigma-sigma_s
            D = 1/(3*sigma)

            SET_palmer_acc = False
            spec_rad_oci[k,h], oci_i = transport()

            SET_palmer_acc = True
            spec_rad_pacc[k,h], sosa_i = transport()

            if spec_rad_pacc[k,h] > 1:
                spec_rad_pacc[k,h] = 1

            print("δ={}, c={}, OCI took {}({}), Palmer Acc took {}({})".format(mfp_range[k], c_range[h], oci_i, spec_rad_oci[k,h], sosa_i, spec_rad_pacc[k,h] ))

    c, mfp = np.meshgrid(c_range, mfp_range)

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    surf = ax1.contourf(mfp, c, spec_rad_pacc, levels=100, cmap=cm.viridis, antialiased=False)
    fig.colorbar(surf)
    
    surf2 = ax2.contourf(mfp, c, spec_rad_oci, levels=100, cmap=cm.viridis, antialiased=False)
    fig.colorbar(surf2)

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    ax1.set_xlabel(r"$\delta$")
    ax1.set_ylabel(r"$c$")
    ax2.set_xlabel(r"$\delta$")

    ax1.text(0e0, .1, 'Palmer Acc', color='w', style='italic',)# bbox={'facecolor': color_oci, 'alpha': 0.5, 'pad': 5})
    ax2.text(3e0, .1, 'OCI', color='w', style='italic',)# bbox={'facecolor': color_si, 'alpha': 0.5, 'pad': 5})

    plt.gcf().set_size_inches(6.5, 3)
    ax2.label_outer()
    fig.tight_layout()
    plt.show()