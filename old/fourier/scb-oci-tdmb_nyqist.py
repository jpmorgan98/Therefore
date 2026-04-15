import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import math


np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

N_angle = 8

[angles, weights] = np.polynomial.legendre.leggauss(N_angle)

#angles = np.array([-0.25,0.25])

dx = .1
dt = 250
v = 5
sigma = 1.0
sigma_s = sigma*.5

N_l = 500
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

    eig_vals = np.zeros((1), dtype=np.complex_)

    for l in range(N_l):
        E = EBuild(l)
        T = np.matmul(RmSinv, E)
        
        eig_val, stand_eig_mat = np.linalg.eig(T)

        eig_vals = np.append(eig_vals, eig_val, axis=0)

    # plot the complex numbers 
    max_eigval = eig_vals[np.abs(eig_vals).argmax()]

    print(max_eigval)

    plt.plot(eig_vals.real, eig_vals.imag, 'k.') 
    plt.plot(max_eigval.real, max_eigval.imag, 'rX', markersize=10, label=r'$\rho$')
    plt.ylabel('Imaginary') 
    plt.xlabel('Real') 
    #plt.title(r"Eigenvalues of OCI-SCB-TDMB ($\Delta t =${}, $\delta =${}, $c=${})".format(dt, dx*sigma, sigma_s/sigma))
    plt.grid()
    plt.legend()
    #plt.show()
    plt.savefig("eig_plot.pdf")
    #plt.savefig("eigplots/c{}/mfp{}dt{}c{}.png".format(sigma_s/sigma, dx*sigma, dt, sigma_s/sigma))
    plt.clf()

    #mags = np.abs(eig_vals)

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(eig_vals.real, eig_vals.imag, mags)
    #plt.show()

    #x = np.linspace(np.min(eig_vals.real), np.max(eig_vals.real), 20)
    #y = np.linspace(np.min(eig_vals.imag), np.max(eig_vals.imag), 20)

    #heatmap, blah, blah = np.histogram2d(x, y, weights=mags)

    #plt.clf()
    #plt.imshow(heatmap)
    #plt.show()

    #from matplotlib import cm
    #f = plt.figure()

    #ax = f.add_subplot(111)
    #ax.pcolormesh(eig_vals.real,eig_vals.imag, mags, cmap = cm.viridis)
    #f.show()

    #print(np.max(np.abs(eig_vals)))

    return(np.max(np.abs(eig_vals)))
#def plot_dt():


def computeSI():
    spec_rad = np.zeros(N_l)
    spec_rad_complex = np.zeros(N_l, dtype=np.complex_)
    S = Sbuild()
    R = RBuild()

    eig_vals = np.zeros((1), dtype=np.complex_)

    for l in range(N_l):
        E = EBuild(l)
        RmEinv = np.linalg.inv(R-E)
        T = np.matmul(RmEinv, S)
        eig_val, stand_eig_mat = np.linalg.eig(T)
        eig_vals = np.append(eig_vals, eig_val, axis=0)

    max_eigval = eig_vals[np.abs(eig_vals).argmax()]

    plt.plot(eig_vals.real, eig_vals.imag, 'k.') 
    plt.plot(max_eigval.real, max_eigval.imag, 'rX', markersize=15, label=r'$\rho$')
    plt.ylabel('Imaginary') 
    plt.xlabel('Real') 
    plt.title(r"Eigenvalues of SI-SCB-TDMB ($\Delta t =${}, $\delta =${}, $c=${})".format(dt, dx*sigma, sigma_s/sigma))
    plt.legend()
    plt.savefig("eigplots/si/c{}/mfp{}dt{}c{}.png".format(sigma_s/sigma, dx*sigma, dt, sigma_s/sigma))
    plt.clf()

    return(np.max(np.abs(eig_vals)))


if __name__ == '__main__':

    sigma_s = 0.9*sigma
    dt = .1
    dx = .25

    spec = compute()

    exit()

    dt_range = np.array([100, 10, 1, 0.1])
    mfp = np.array([0.01, 1.0, 10])
    c_range = np.array([0, 0.5, 1.0])
    c = 1
    sigma_s = c*sigma

    for k in range (c_range.size):
        for m in range(dt_range.size):
            for j in range(mfp.size):
                sigma_s = c_range[k]*sigma
                dt = dt_range[m]
                dx = mfp[j]/sigma

                spec = compute()
                spec_si = computeSI()

                print("c {}, dt {}, mfp {}, spec rad {} and {}".format(c_range[k], dt, mfp[j], spec, spec_si))

    # plotting 
