import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import math


np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

N_angle = 8

[angles, weights] = np.polynomial.legendre.leggauss(N_angle)

#angles = angles[N_angle:]
#weights = weights[N_angle:]

N_lam = 25
lam = np.pi*np.linspace(0,2,N_lam)

dx = 1
sigma = 4
sigmas = .5

i = complex(0,1)

def Rblockpos(mu, l):
    return(
        np.array([[mu/2+sigma*dx/2, -mu/2],
                  [mu/2-mu*np.exp(-i*sigma*lam[l]*dx), mu/2+sigma*dx/2]])
    )
def Rblockneg(mu, l):
    return(
        np.array([[-mu/2 + sigma*dx/2, -mu/2+mu*np.exp(i*sigma*lam[l]*dx)],
                  [mu/2, -mu/2 + sigma*dx/2]])
    )
def Rbuild(l):
    A = np.zeros((2*N_angle, 2*N_angle),dtype=np.complex_)
    for a in range(N_angle):
        if angles[a] > 0:
            A[a*2:(a+1)*2,a*2:(a+1)*2] = Rblockpos(angles[a], l)
        elif (angles[a] < 0):
            A[a*2:(a+1)*2,a*2:(a+1)*2] = Rblockneg(angles[a], l)
    return(A)



def Sbuild():
    
    S = np.zeros((2*N_angle,2*N_angle)).astype(np.complex_)
    #for m in range(N_angle):

    beta = sigmas * dx / 4

    for p in range(N_angle):
        for j in range(N_angle):
            S[p*2,   j*2]   = beta*weights[j]
            S[p*2+1, j*2+1] = beta*weights[j]

    return(S)



def eig_val():

    eig_lam = np.zeros(1).astype(np.complex_)

    S = Sbuild()

    for i in range(N_lam):
        R = Rbuild(i)
        Rinv = np.linalg.inv(R)
        T =  np.matmul(Rinv, S)

        eig_val, stand_eig_mat = np.linalg.eig(T)

        eig_lam = np.append(eig_lam, eig_val)

    #plt.plot(eig_lam.real, eig_lam.imag, 'b.') 
    #plt.ylabel('Imaginary') 
    #plt.xlabel('Real') 
    #plt.show()

    return(np.max(np.abs(eig_lam)))


if __name__ == '__main__':
    mfp = .01
    scat = 1.0
    dx = mfp/sigma
    sigmas = scat*sigma
    print( eig_val() )

    #exit()
    N_mfp = 25
    N_c = 30

    mfp_range = np.linspace(.1,20,N_mfp)
    c_range = np.linspace(0,1,N_c)

    spec_rad = np.zeros([N_mfp, N_c])

    itter = 0
    for y in range(mfp_range.size):
        for u in range(c_range.size):
            dx = mfp_range[y]/sigma
            sigmas = c_range[u]*sigma
            spec_rad[y,u] = eig_val()

    c, mfp = np.meshgrid(c_range, mfp_range)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(mfp, c, spec_rad, cmap=cm.viridis,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(0, 1.0) #np.max(spec_rad)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel("mfp")
    ax.set_ylabel("c")
    ax.set_zlabel(r"$\rho$")
    ax.set_title(r"$\rho$ for $\lambda \in [0,2\pi]$ (at {0} points), in $S_{1}$".format(N_lam, N_angle))
    plt.show()