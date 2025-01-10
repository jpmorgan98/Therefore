import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import math

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

N_angle = 4

[angles, weights] = np.polynomial.legendre.leggauss(N_angle)

dx = .1
dt = 100
v = 5
sigma = 4.0
sigma_s = sigma*.5

N_l = 25
lam = np.pi*np.linspace(0,2,N_l)

i = complex(0,1)

tol = 1e-13

def Rblockpos(a):
    return(np.array([[angles[a]/2 + dx/2*sigma, -angles[a]/2, dx/(2*v*dt), 0],
                     [angles[a]/2, angles[a]/2 + dx/2*sigma, 0, dx/(2*v*dt)],
                     [-dx/(v*dt), 0, dx/(v*dt)+angles[a]/2+dx/2*sigma, -angles[a]/2],
                     [0, -dx/(v*dt), angles[a]/2, dx/(v*dt) + angles[a]/2 + dx/2*sigma]])
    )

def Rblockneg(a):
    return(np.array([[-angles[a]/2 + dx/2*sigma, -angles[a]/2, dx/(2*v*dt), 0],
                     [angles[a]/2, -angles[a]/2 + dx/2*sigma, 0, dx/(2*v*dt)],
                     [-dx/(v*dt), 0, dx/(v*dt)-angles[a]/2+dx/2*sigma, -angles[a]/2],
                     [0, -dx/(v*dt), angles[a]/2, dx/(v*dt) - angles[a]/2 + dx/2*sigma]])
    )

def RBuild():
    A = np.zeros((4*N_angle, 4*N_angle),dtype=np.complex_)
    for a in range(N_angle):
        if   angles[a] > 0:
            A[a*4:(a+1)*4,a*4:(a+1)*4] = Rblockpos(a)
        elif angles[a] < 0:
            A[a*4:(a+1)*4,a*4:(a+1)*4] = Rblockneg(a)
    return(A)



def Eblockpos(a,l):
    return(np.array([[0,0,0,0],
                     [angles[a]*np.exp(-i*lam[l]*sigma*dx),0,0,0],
                     [0,0,0,0],
                     [0,0,angles[a]*np.exp(-i*lam[l]*sigma*dx),0]],dtype=np.complex_)
    )
def Eblockneg(a,l):
    return(np.array([[0,-angles[a]*np.exp(i*lam[l]*sigma*dx),0,0],
                     [0,0,0,0],
                     [0,0,0,-angles[a]*np.exp(i*lam[l]*sigma*dx)],
                     [0,0,0,0]],dtype=np.complex_)
    )

def EBuild(l):
    A = np.zeros((4*N_angle, 4*N_angle),dtype=np.complex_)
    for a in range(N_angle):
        if   angles[a] > 0:
            A[a*4:(a+1)*4,a*4:(a+1)*4] = Eblockpos(a,l)
        elif angles[a] < 0:
            A[a*4:(a+1)*4,a*4:(a+1)*4] = Eblockneg(a,l)
    return(A)



def Gblockpos(a):
    return(np.array([[angles[a]/2, -angles[a]/2, dx/(2*v*dt), 0],
                     [angles[a]/2-angles[a], angles[a]/2, 0, dx/(2*v*dt)],
                     [-dx/(v*dt), 0, dx/(v*dt)+angles[a]/2, -angles[a]/2],
                     [0, -dx/(v*dt), angles[a]/2-angles[a], dx/(v*dt) + angles[a]/2]])
    )

def Gblockneg(a):
    return(np.array([[-angles[a]/2, -angles[a]/2+angles[a], dx/(2*v*dt), 0],
                     [angles[a]/2, -angles[a]/2, 0, dx/(2*v*dt)],
                     [-dx/(v*dt), 0, dx/(v*dt)-angles[a]/2, -angles[a]/2+angles[a]],
                     [0, -dx/(v*dt), angles[a]/2, dx/(v*dt) - angles[a]/2]])
    )

def GBuild():
    A = np.zeros((4*N_angle, 4*N_angle),dtype=np.complex_)
    for a in range(N_angle):
        if   angles[a] > 0:
            A[a*4:(a+1)*4,a*4:(a+1)*4] = Gblockpos(a)
        elif angles[a] < 0:
            A[a*4:(a+1)*4,a*4:(a+1)*4] = Gblockneg(a)
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
    spec_rad_complex = np.zeros(N_l, dtype=np.complex_)
    S = Sbuild()
    R = RBuild()

    RmSinv = np.linalg.inv(R-S)

    for l in range(N_l):
        E = EBuild(l)
        T = np.matmul(RmSinv, E)

        eig_val, stand_eig_mat = np.linalg.eig(T)
        eig_val_abs = np.abs(eig_val)
        spec_rad_complex[l] = eig_val[eig_val_abs.argmax()]

    spec_rad_abs = np.abs(spec_rad_complex)

    return(np.max(np.abs(spec_rad_complex)))



def computeTSA():
    spec_rad = np.zeros(N_l)
    spec_rad_complex = np.zeros(N_l, dtype=np.complex_)
    S = Sbuild()
    R = RBuild()

    RmSinv = np.linalg.inv(R-S)

    for l in range(N_l):
        E = EBuild(l)
        T = np.matmul(RmSinv, E)

        L = R-E
        beta = 1
        TSA = np.linalg.inv(L-(1-beta)*S)
        Td = np.add (T,   np.matmul( np.matmul( TSA, E ), (T-np.identity(4*N_angle))) )
        eig_val, stand_eig_mat = np.linalg.eig(Td)

        #eig_val, stand_eig_mat = np.linalg.eig(T)
        eig_val_abs = np.abs(eig_val)
        spec_rad_complex[l] = eig_val[eig_val_abs.argmax()]

    spec_rad_abs = np.abs(spec_rad_complex)

    return(np.max(np.abs(spec_rad_complex)))


def computeSOA():
    spec_rad = np.zeros(N_l)
    spec_rad_complex = np.zeros(N_l, dtype=np.complex_)
    S = Sbuild()
    R = RBuild()

    RmSinv = np.linalg.inv(R-S)

    for l in range(N_l):
        E = EBuild(l)
        T = np.matmul(RmSinv, E)

        G = GBuild()

        SOA = np.linalg.inv(G)
        Td = np.add (T,   np.matmul( np.matmul( SOA, E ), (T-np.identity(4*N_angle))) )
        eig_val, stand_eig_mat = np.linalg.eig(Td)

        #eig_val, stand_eig_mat = np.linalg.eig(T)
        eig_val_abs = np.abs(eig_val)
        spec_rad_complex[l] = eig_val[eig_val_abs.argmax()]

    spec_rad_abs = np.abs(spec_rad_complex)

    return(np.max(np.abs(spec_rad_complex)))

def computeSI():
    spec_rad = np.zeros(N_l)
    spec_rad_complex = np.zeros(N_l, dtype=np.complex_)
    S = Sbuild()
    R = RBuild()

    for l in range(N_l):
        E = EBuild(l)
        RmEinv = np.linalg.inv(R-E)
        T = np.matmul(RmEinv, S)
        eig_val, stand_eig_mat = np.linalg.eig(T)
        eig_val_abs = np.abs(eig_val)
        spec_rad_complex[l] = eig_val[eig_val_abs.argmax()]

    spec_rad_abs = np.abs(spec_rad_complex)

    return(np.max(np.abs(spec_rad_complex)))


def plot_const_dt_surf(dt_val):

    dt = dt_val
    N_mfp = 50
    N_c = 50

    mfp_range = np.linspace(.1,10,N_mfp)
    c_range = np.linspace(0,1,N_c)

    spec_rad = np.zeros([N_mfp, N_c])

    itter = 0
    for y in range(mfp_range.size):
        for u in range(c_range.size):
            dx = mfp_range[y]/sigma
            sigma_s = sigma*c_range[u]
            spec_rad[y,u] = compute()

            itter += 1
            printProgressBar(itter, mfp_range.size*c_range.size)
    print("")
    # plotting 

    c, mfp = np.meshgrid(c_range, mfp_range)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(mfp, c, spec_rad, cmap=cm.viridis,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 1.0) #np.max(spec_rad)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel("mfp")
    ax.set_ylabel("c")
    ax.set_zlabel(r"$\rho$")
    ax.set_title(r"$\rho$ for $\Delta t$={0}, $v$={1}, $\sigma$={2}, $\lambda \in [0,2\pi]$ (at {3} points), in $S_{4}$".format(dt, v, sigma, N_l, N_angle))
    plt.show()


#def plot_dt():


if __name__ == '__main__':
    
    dt = .1
    N_mfp = 25
    N_c = 25

    mfp_range = np.linspace(.1,10,N_mfp)
    c_range = np.linspace(0,1,N_c)

    spec_rad = np.zeros([N_mfp, N_c])
    spec_rad_tsa = np.zeros_like(spec_rad)
    spec_rad_soa = np.zeros_like(spec_rad)
    spec_rad_si = np.zeros_like(spec_rad)


    itter = 0
    for y in range(mfp_range.size):
        for u in range(c_range.size):
            dx = mfp_range[y]/sigma
            sigma_s = sigma*c_range[u]
            spec_rad[y,u] = compute()
            spec_rad_tsa[y,u] = computeTSA()
            spec_rad_soa[y,u] = computeSOA()
            spec_rad_si[y,u] = computeSI()

            itter += 1
            printProgressBar(itter, mfp_range.size*c_range.size)
    print("")
    # plotting 


    c, mfp = np.meshgrid(c_range, mfp_range)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

    surf = ax1.contourf(mfp, c, spec_rad, levels=100, vmin=0, vmax=1, cmap=cm.viridis, antialiased=False)
    fig.colorbar(surf)
    
    surf2 = ax2.contourf(mfp, c, spec_rad_tsa, levels=100, vmin=0, vmax=1, cmap=cm.viridis, antialiased=False)
    fig.colorbar(surf2)
    
    surf3 = ax3.contourf(mfp, c, spec_rad_soa, levels=100, cmap=cm.viridis, antialiased=False)
    fig.colorbar(surf3)

    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')

    ax1.set_xlabel(r"$\delta$")
    ax1.set_ylabel(r"$c$")

    ax2.set_xlabel(r"$\delta$")


    ax1.text(3e0, .1, 'OCI', color='w', style='italic',)# bbox={'facecolor': color_oci, 'alpha': 0.5, 'pad': 5})
    
    ax2.text(3e0, .1, 'TSA', color='w', style='italic',)# bbox={'facecolor': color_si, 'alpha': 0.5, 'pad': 5})

    ax3.text(1e0, .1, r'$μ_m \frac{dψ}{dx}=0$', color='w', style='italic',)# bbox={'facecolor': color_si, 'alpha': 0.5, 'pad': 5})

    plt.gcf().set_size_inches(6.5, 3)

    ax2.label_outer()
    
    fig.tight_layout()

    #ax3.set_xlabel(r"$\delta$")
    #ax1.set_zlabel(r"$\rho$")
    #ax1.set_title(r"$\rho$ for $\lambda \in [0,2\pi]$ (at {0} points), in $S_{1}$".format(N_lam, N_angle))
    plt.show()

    #compute()

    """

    N_mfp = 3
    N_c = 4
    N_dt = 15

    mfp_range = np.array((10, 1, .1))
    c_range = np.array((0.5, 0.75, 0.99))
    dt_range = np.logspace(-3,1,N_dt)

    spec_rad = np.zeros([N_dt])
    spec_rad_si = np.zeros([N_dt])
    itter_pred = np.zeros((N_dt))

    line_format = ['-.', '-', '--', '-*']
    line_color = ['#d95f02', '#7570b3']
    line_color2= ['#af8dc3', '#7fbf7b']
    color_si = '#ca0020'
    color_oci = '#404040'
    line_format_ev = ['-.', '-', '--', '-*']
    itter = 0

    fig, ax = plt.subplots(N_mfp)

    with np.load('spec_rad_numerical_3_avg.npz') as data:
        spec_rad_eval = data['spec_rad']
    with np.load('itter_count_3_nocond.npz') as data:
        itter_eval = data['itter']

    itter_eval -= 1

    for y in range(mfp_range.size):
        for u in range(c_range.size):
            dx = mfp_range[y]/sigma
            sigma_s = sigma*c_range[u]

            for k in range(dt_range.size):
                dt = dt_range[k]
                spec_rad[k] = compute()
                spec_rad_si[k] = computeSI()
                itter_pred[k] = math.log(tol)/math.log(spec_rad[k])
                itter += 1

            if y == 0:
                ax[y].plot(dt_range, spec_rad, line_format_ev[u], color=color_oci, linewidth=2.5, label="c={}".format(c_range[u]))
                
            else:
                ax[y].plot(dt_range, spec_rad, line_format_ev[u], linewidth=2.5, color=color_oci,)
            
            ax[y].plot(dt_range, spec_rad_si, line_format[u], linewidth=2.5, color=color_si,)
            #ax[y].plot(dt_range, spec_rad_eval[:,u,y], line_format_ev[u])
            #ax[y].plot(dt_range, itter_eval[:,u,y], line_format_ev[u])

        #if y == 0:
        #    plt.legend() #ax[y].set_title(r" $v$={1}, $\sigma$={2}, $\lambda \in [0,2\pi]$ (at {3} points), in $S_{4}$ \n mfp = {0}".format(mfp_range[y], v, sigma, N_l, N_angle))
        #else: 
        ax[y].set_title(r"$\delta =$ {0}".format(mfp_range[y]))

        ax[y].set_xlabel(r"$\Delta t$ ")
        ax[y].set_ylabel(r"$\rho$")
        ax[y].set_xscale("log")
        #ax[y].set_yscale("log")
        ax[y].set_ylim((-.1, 1.1))
        ax[y].grid()

    #ax[0].annotate(r'$c=${}'.format(c_range[3]), xy=(3, 0.85))
    #ax[0].annotate(r'$c=${}'.format(c_range[2]), xy=(3, 0.4))
    #ax[0].annotate(r'$c=${}'.format(c_range[1]), xy=(3, 0.15))
    #ax[0].annotate(r'$c=${}'.format(c_range[0]), xy=(3, -0.075))

    for axs in ax.flat:
        axs.label_outer()
    
    fig.tight_layout()

    ax[0].legend()

    ax[2].text(5e-3, .6, 'OCI', style='italic', )#bbox={'facecolor': color_oci, 'alpha': 0.5, 'pad': 5})
    
    ax[2].text(5e-1, .25, 'SI', style='italic', )#bbox={'facecolor': color_si, 'alpha': 0.5, 'pad': 5})

    plt.gcf().set_size_inches(6.5, 6.5)

    plt.savefig("spec_rad_over_dt.pdf")
    #plt.show()
    #print("")
    # plotting 

    """