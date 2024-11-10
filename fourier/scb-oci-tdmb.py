import numpy as np
import matplotlib.pyplot as plt
import math

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

N_angle = 8

[angles, weights] = np.polynomial.legendre.leggauss(N_angle*2)

# snagging only the positive angles
angles = angles[N_angle:]
weights = weights[N_angle:]

dx = .1
dt = 1000
v = 5
sigma = 4.0
sigma_s = sigma*.5

N_l = 10
lam = np.pi*np.linspace(0,2,N_l)

i = complex(0,1)

def Rblock(a):
    return(np.array([[angles[a]/2 + dx/2*sigma, -angles[a]/2, dx/(2*v*dt), 0],
                     [angles[a]/2, angles[a]/2 + dx/2*sigma, 0, dx/(2*v*dt)],
                     [-dx/(v*dt), 0, dx/(v*dt)+angles[a]/2+dx/2*sigma, -angles[a]/2],
                     [0, -dx/(v*dt), angles[a]/2, dx/(v*dt) + angles[a]/2 + dx/2*sigma]])
    )

def RBuild():
    A = np.zeros((4*N_angle, 4*N_angle),dtype=np.complex_)
    for a in range(N_angle):
        A[a*4:(a+1)*4,a*4:(a+1)*4] = Rblock(a)
    return(A)

def Eblock(a,l):
    return(np.array([[0,0,0,0],
                     [angles[a]*np.exp(-i*lam[l]*sigma*dx),0,0,0],
                     [0,0,0,0],
                     [0,0,angles[a]*np.exp(-i*lam[l]*sigma*dx),0]],dtype=np.complex_)
    )

def EBuild(l):
    A = np.zeros((4*N_angle, 4*N_angle),dtype=np.complex_)
    for a in range(N_angle):
        A[a*4:(a+1)*4,a*4:(a+1)*4] = Eblock(a,l)
    return(A)

def Sbuild():
    N = N_angle
    S = np.zeros((4*N_angle, 4*N_angle),dtype=np.complex_)
    beta = dx*sigma_s/4

    for p in range(N):
        for j in range(N):
            S[p*4,   j*4]   = beta*weights[j]
            S[p*4+1, j*4+1] = beta*weights[j]
            S[p*4+2, j*4+2] = beta*weights[j]
            S[p*4+3, j*4+3] = beta*weights[j]
    return(S)

# Print iterations progress
def printProgressBar (iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 1, length = 25, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def compute():
    spec_rad = np.zeros(N_l)
    S = Sbuild()
    R = RBuild()

    RmSinv = np.linalg.inv(R-S)

    for l in range(N_l):
        E = EBuild(l)
        T = np.matmul(RmSinv, E)
        eig_val, stand_eig_mat = np.linalg.eig(T)
        spec_rad[l] = np.max(np.abs(eig_val))

    return(np.max(spec_rad))

if __name__ == '__main__':

    N_mfp = 100
    N_c = 50
    mfp_range = np.linspace(.1,10,N_mfp)
    c_range = np.linspace(0,1,N_c)

    spec_rad = np.zeros([N_mfp,N_c])

    itter = 0
    for y in range(mfp_range.size):
        for u in range(c_range.size):
            dx = mfp_range[y]/sigma
            sigma_s = sigma*c_range[u]
            spec_rad[y,u] = compute()

            itter += 1

            printProgressBar(itter, mfp_range.size*c_range.size)

    
    #print(spec_rad)

    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    c, mfp = np.meshgrid(c_range, mfp_range)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(mfp, c, spec_rad, cmap=cm.viridis,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, np.max(spec_rad))
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel("mfp")
    ax.set_ylabel("c")
    ax.set_zlabel(r"$\rho$")
    ax.set_title(r"$\rho$ for $\Delta t$={0}, $v$={1}, $\sigma$={2}, $\lambda \in [0,2\pi]$ (at {3} points), and $+\mu$ in $S_{4}$".format(dt, v, sigma, N_l, 2*N_angle))
    plt.show()