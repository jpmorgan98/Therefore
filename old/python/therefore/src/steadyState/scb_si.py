import numpy as np
import numba as nb

#Simple Corner balence sweep
@nb.jit(nopython=True, parallel=True, cache=False, nogil=False, fastmath=False)
def SCBRun(Q, xsec, dx, mu, BCl, BCr, N_mesh):
    '''Return angular flux
    
    Runs a parallelized and jit compiled simple corner balance source 
    (aka Richardson aka Fixed Point) itteration. Launches a parallelized job over 
    the number of angles as each angular flux is calculated independently of one another
    
    Returns 2D Numpy array of angular flux (size: [N_angles, N_cells])
    '''
    angular_flux = np.zeros((mu.size, 2*N_mesh), np.float64)
    
    for angle in nb.prange(mu.size):
        if mu[angle] < 0: #goin back
            for i in range(N_mesh-1, -1, -1):
                #check bound
                if i == N_mesh-1:
                    psi_ph = BCr[angle] 
                else:
                    psi_ph = angular_flux[angle, 2*(i+1)]
                
                [angular_flux[angle, 2*i], angular_flux[angle, 2*i+1]]  = SCBKernel_negative(Q[angle, 2*i], Q[angle, 2*i+1], psi_ph, xsec[i], dx[i], mu[angle]) 
                
        else: #goin forward
            for i in range(N_mesh):
                #check bound
                if i == 0:
                    psi_mh = BCl[angle]
                else:
                    psi_mh = angular_flux[angle, (2*i)-1]
                
                [angular_flux[angle, 2*i], angular_flux[angle, 2*i+1]] = SCBKernel_positive(Q[angle, 2*i], Q[angle, 2*i+1], psi_mh, xsec[i], dx[i], mu[angle])
                
    return(angular_flux)

# negative mu!
@nb.jit(nopython=True, cache=False, fastmath=False)
def SCBKernel_negative(Q_l, Q_r, psi_ph, xsec, dx, mu):
    '''SCB going from right to the left (mu<0)'''
    
    #mannaz = xsec*dx/2 - mu/2
    
    A = np.array([[xsec*dx/2 - mu/2, mu/2],
                  [-mu/2,            xsec*dx/2 - mu/2]])
    
    b = np.array([[dx/2*Q_l],
                  [dx/2*Q_r - mu*psi_ph]])
    
    [psi_l, psi_r] = np.linalg.solve(A,b)
    
    return(psi_l[0], psi_r[0])

# positive mu
@nb.jit(nopython=True, cache=False, fastmath=False)
def SCBKernel_positive(Q_l, Q_r, psi_mh, xsec, dx, mu):
    '''SCB going from the left to the right (mu>0)'''
    
    #mannaz = mu/2 + (xsec*dx)/2
    
    A = np.array([[(xsec*dx)/2 + mu/2, mu/2],
                  [-mu/2,              (xsec*dx)/2 + mu/2]])
    
    b = np.array([[dx*Q_l/2  + mu*psi_mh],
                  [dx*Q_r/2]])
    
    [psi_l, psi_r] = np.linalg.solve(A,b)
    
    return(psi_l[0], psi_r[0])


