import numpy as np
#from .matrix import A_pos, A_neg, c_neg, c_pos, scatter_source, A_neg_nomat, A_pos_nomat
from .matrix import A_pos, A_neg, c_neg, c_pos, scatter_source
import therefore.src.utilities as utl
import numba as nb
#import cupyx.scipy.sparse.linalg as gpuLinalg
import scipy.sparse.linalg as cpuLinalg
#import cupyx.scipy.sparse as spMat
#import cupy as cu
np.set_printoptions(linewidth=np.inf)
from scipy.sparse import csr_matrix, lil_matrix
import betterspy

#@profile
def OCIMBTimeStepBig(sim_perams, angular_flux_previous, angular_flux_midstep_previous, source_mesh, xsec_mesh, xsec_scatter_mesh, dx_mesh, angles, weights):

    velocity = sim_perams['velocity']
    dt = sim_perams['dt']
    N_time = sim_perams['N_time']
    N_angles = sim_perams['N_angles']
    N_mesh = sim_perams['N_mesh']
    data_type = sim_perams['data_type']
    max_it = sim_perams['max loops']
    tol = sim_perams['tolerance']

    source_converged: bool = False
    source_counter: int = 0
    no_convergence: bool = False
    spec_rad = 0
    
    N_ans = int(2*N_mesh)

    angular_flux = np.zeros([N_angles, N_ans], data_type)
    angular_flux_midstep = np.zeros([N_angles, N_ans], data_type)

    #angular_flux_last = np.zeros([N_angles, N_ans], data_type) 
    #angular_flux_mid_last = np.zeros([N_angles, N_ans], data_type)

    angular_flux_last = angular_flux_previous ##
    angular_flux_mid_last = angular_flux_midstep_previous # #

    scalar_flux = np.zeros(N_ans, data_type)
    scalar_flux_last = np.zeros(N_ans, data_type)
    scalar_flux_next = np.zeros(N_ans, data_type)

    A = BuildHer(xsec_mesh, xsec_scatter_mesh, dx_mesh, dt, velocity, angles, weights)
    print(A)
    print()

    A = csr_matrix(A)
    #A_gpu = spMat.csr_matrix(A) 

    np.set_printoptions(linewidth=np.inf)

    # LAST -- refers to last itteration
    # PREVIOUS -- refers to previous time step
    while source_converged == False:

        BCl = utl.BoundaryCondition(sim_perams['boundary_condition_left'],   0, N_mesh, angular_flux=angular_flux, incident_flux_mag=sim_perams['left_in_mag'],  angle=sim_perams['left_in_angle'],  angles=angles)
        BCr = utl.BoundaryCondition(sim_perams['boundary_condition_right'], -1, N_mesh, angular_flux=angular_flux, incident_flux_mag=sim_perams['right_in_mag'], angle=sim_perams['right_in_angle'], angles=angles)
        
        c = BuildC(angular_flux_midstep_previous, angular_flux_last, angular_flux_mid_last, source_mesh, dx_mesh, dt, velocity, angles, BCl, BCr)
        print(c)
        #print()
        #c_gpu = cu.asarray(c)

        #angular_flux_raw = runBig(A, c)
        #[angular_flux_raw_gpu, info] = gpuLinalg.gmres(A_gpu, c_gpu)
        [angular_flux_raw, info] = cpuLinalg.gmres(A, c)
        print(angular_flux_raw)
        #angular_flux_raw = cu.asnumpy(angular_flux_raw_gpu.get())

        [angular_flux, angular_flux_midstep] = reset(angular_flux_raw, [N_angles, N_mesh])

        #calculate current
        current = utl.Current(angular_flux, weights, angles)
        
        #calculate scalar flux for next itteration
        scalar_flux_next = utl.ScalarFlux(angular_flux, weights)
        
        if source_counter > 2:
            #check for convergence
            error_eos = np.linalg.norm(angular_flux_midstep - angular_flux_mid_last, ord=2)
            error_mos = np.linalg.norm(angular_flux - angular_flux_last, ord=2)

            if error_mos < tol and error_eos < tol:
                source_converged = True

            spec_rad = np.linalg.norm(scalar_flux_next - scalar_flux, ord=2) / np.linalg.norm((scalar_flux - scalar_flux_last), ord=2)

        if source_counter > max_it:
            print('Error source not converged after max iterations')
            print()
            source_converged = True
            no_convergence = True

        angular_flux_last = angular_flux
        angular_flux_mid_last = angular_flux_midstep
        
        scalar_flux_last = scalar_flux
        scalar_flux = scalar_flux_next
        
        source_counter  += 1
        
    print(angular_flux_raw)

    return(angular_flux, angular_flux_midstep, angular_flux_raw, current, spec_rad, source_counter, source_converged)


#@nb.njit
def BuildHer(xsec, xsec_scatter, dx, dt, v, mu, weight):

    N_mesh = dx.size
    sizer = mu.size*4
    N_angle = mu.size

    A_uge = np.zeros((4*N_angle*N_mesh, 4*N_angle*N_mesh))

    for i in range(N_mesh):
        
        A = np.zeros((sizer,sizer))
        for m in range(N_angle):
            if mu[m] < 0:
                A_small = A_neg(dx[i], v, dt, mu[m], xsec[i])
                            
            elif mu[m] > 0:
                A_small = A_pos(dx[i], v, dt, mu[m], xsec[i])

            A[m*4:(m+1)*4, m*4:(m+1)*4] = A_small

        S = scatter_source(dx[i], xsec_scatter[i], N_angle, weight)
        A = A - S

        #helper index values
        Ba = 4*N_angle*i
        Bb = 4*N_angle*(i+1)

        A_uge[Ba:Bb, Ba:Bb] = A

    return(A_uge)

@nb.njit#(nopython=True, parallel=False, cache=True, nogil=True, fastmath=True)
def BuildC(angular_flux_mid_previous, angular_flux_last, angular_flux_midstep_last, source, dx, dt, v, mu, BCl, BCr):
    N_mesh = dx.size
    sizer = mu.size*4
    N_angle = mu.size

    c_uge = np.zeros((4*N_angle*N_mesh, 1))

    for i in range(N_mesh):
        
        i_l: int = int(2*i)
        i_r: int = int(2*i+1)

        Q = source[:,i_l:i_r+1]/2
        #angle space time
        
        c = np.zeros((sizer,1))

        # getting really sloppy with the indiceis
        for m in range(N_angle):
            psi_halfLast_L = angular_flux_mid_previous[m, i_l] # known all angles from the last time step in the left cell
            psi_halfLast_R = angular_flux_mid_previous[m, i_r] # known

            if mu[m] < 0:
                if i == N_mesh-1:
                    #print('Right bound')
                    #print(BCr)
                    psi_rightBound          = BCr[m]
                    psi_halfNext_rightBound = BCr[m]
                else:
                    psi_rightBound          = angular_flux_last[m, i_r+1]
                    psi_halfNext_rightBound = angular_flux_midstep_last[m, i_r+1] 
                
                c_small = c_neg(dx[i], v, dt, mu[m], Q[m,0], Q[m,1], Q[m,0], Q[m,1], psi_halfLast_L, psi_halfLast_R, psi_rightBound, psi_halfNext_rightBound)
                            
            elif mu[m] > 0:
                if i == 0:
                    psi_leftBound           = BCl[m]
                    psi_halfNext_leftBound  = BCl[m]
                else:
                    psi_leftBound           = angular_flux_last[m, i_l-1]
                    psi_halfNext_leftBound  = angular_flux_midstep_last[m, i_l-1]

                c_small = c_pos(dx[i], v, dt, mu[m], Q[m,0], Q[m,1], Q[m,0], Q[m,1], psi_halfLast_L, psi_halfLast_R, psi_leftBound, psi_halfNext_leftBound)

            c[m*4:(m+1)*4] = c_small

        #helper index values
        Ba = 4*N_angle*i
        Bb = 4*N_angle*(i+1)

        c_uge[Ba:Bb] = c

    return(c_uge)

#@nb.njit
def runBig(A, c):

    angular_flux_raw = cpuLinalg.spsolve(A, c)
    #angular_flux_raw = cu.asnumpy(angular_flux_raw_gpu.get())

    return(angular_flux_raw)



@nb.njit#(nopython=True, parallel=False, cache=True, nogil=True, fastmath=True)
def reset(angular_flux_raw, size):
    N_angle = size[0]
    N_mesh = size[1]

    angular_flux = np.zeros((N_angle, N_mesh*2), np.float64)
    angular_flux_midstep = np.zeros((N_angle, N_mesh*2), np.float64)

    for p in range(N_mesh):
        for m in range(N_angle):
            # number of elements preceeding desired anwser set
            raw_index = int(4*m + 4*p*N_angle)

            angular_flux[m,2*p]           = angular_flux_raw[raw_index]
            angular_flux[m,2*p+1]         = angular_flux_raw[raw_index+1]
            
            angular_flux_midstep[m,2*p]   = angular_flux_raw[raw_index+2]
            angular_flux_midstep[m,2*p+1] = angular_flux_raw[raw_index+3]

    return(angular_flux, angular_flux_midstep)





def BuildHer_Check(xsec, xsec_scatter, dx, dt, v, mu, weight):

    N_mesh = dx.size
    sizer = mu.size*4
    N_angle = mu.size

    A_uge = np.zeros((4*N_angle*N_mesh, 4*N_angle*N_mesh))

    for i in range(N_mesh):
        
        A = np.zeros((sizer,sizer))
        for m in range(N_angle):
            if mu[m] < 0:
                A_small = A_neg(dx[i], v, dt, mu[m], xsec[i])
                            
            elif mu[m] > 0:
                A_small = A_pos(dx[i], v, dt, mu[m], xsec[i])

            A[m*4:(m+1)*4, m*4:(m+1)*4] = A_small

        S = scatter_source(dx[i], xsec_scatter[i], N_angle, weight)
        A = A - S

        #helper index values
        Ba = 4*N_angle*i
        Bb = 4*N_angle*(i+1)

        A_uge[Ba:Bb, Ba:Bb] = A

    return(A_uge)




if __name__ == '__main__':
    

    N_mesh = 5#int(1e4)

    xsec = .5*np.ones(N_mesh)
    xsec_scatter = 2*np.ones(N_mesh)
    dx = .75*np.ones(N_mesh)
    dt = .5
    v = 1
    weight = np.array([1,1])
    mu = np.array([1,1])

    A = BuildHer(xsec, xsec_scatter, dx, dt, v, mu, weight)

    betterspy.show(A)
    betterspy.write_png("out.png", A)


