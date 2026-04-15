import numpy as np
import numba as nb

def OCIRun_Diamond(angular_flux, source, xsec, xsec_scatter, dx, mu, weight, BCl, BCr):
    
    n_mesh = int(dx.size)
    angular_flux_next = np.zeros_like(angular_flux)
    
    for i in range(n_mesh):
        #print(i)
        #print()
        
        bound_ang_flux = np.zeros([mu.size, 2])
        
        if i == n_mesh-1: 
            bound_ang_flux[:,0] = angular_flux[:,-2]
            bound_ang_flux[:,1] = BCr
        elif i == 0:
            bound_ang_flux[:,0] = BCl
            bound_ang_flux[:,1] = angular_flux[:,1]
        else:
            bound_ang_flux[:,0] = angular_flux[:,i-1]
            bound_ang_flux[:,1] = angular_flux[:,i+1]
            
        angular_flux_cell = Diamond_OneCellInv_Cell(bound_ang_flux, source[i], xsec[i], xsec_scatter[i], dx[i], mu, weight)
        
        angular_flux_next[:,i] = angular_flux_cell
        
    #print(angular_flux_next)
    #print(angular_flux_next.shape)
    
    return(angular_flux_next)    
        
    
def Diamond_OneCellInv_Cell(angular_flux, source, xsec, xsec_scatter, dx, mu, weight):
    
    n_angle = mu.size
    
    A = np.zeros([n_angle, n_angle])
    b = np.zeros([n_angle])
    
    '''
    print('================================================================')
    print(xsec*dx/2)
    print(mu[0]/2)
    print(mu[1]/2)
    print()
    print(angular_flux)
    print()
    print()
    '''
    alpha = xsec*dx/2
    #build matrix
    #lefts [k,i], [row, col] so we are filling in col by col
    for k in range (0, n_angle):
        A[k,k] = xsec
        
        #scalar flux
        for i in range (0, n_angle):
            A[k,i] += -xsec_scatter*weight[i]
            
    for i in range (n_angle):
        b[i]   = source  +  (mu[i]/dx) * (angular_flux[i, 1] - angular_flux[i, 0])
    '''
    print('A mat')
    print()
    print(A)
    print()
    print(b)
    print()
    '''
    next_angflux = np.linalg.solve(A, b)
    #print(next_angflux)
    '''
    returned = np.zeros([2,2])
    returned[0,0] = next_angflux[0]
    returned[0,1] = next_angflux[1]
    returned[1,0] = next_angflux[2]
    returned[1,1] = next_angflux[3]
    '''
    
    return(next_angflux)
    
    
    
if __name__ == '__main__':
    
    
    angular_flux  = np.array([[1,1],[1,1]])
    xsec = 1
    xsec_scatter = 0
    dx = 2
    mu = np.array([1,1])
    weights = np.array([1,1])
    N_mesh = 0
    source = 1
    
    SCB_OneCellInv_Cell(angular_flux, source, xsec, xsec_scatter, dx, mu, weights)
