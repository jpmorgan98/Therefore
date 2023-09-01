import numpy as np
from .matrix import A_pos, A_neg, c_neg, c_pos, scatter_source
import therefore.src.utilities as utl
import numba as nb
np.set_printoptions(linewidth=np.inf)


def OCIMBTimeStep(sim_perams, angular_flux_previous, angular_flux_mid_previous, source_mesh, xsec_mesh, xsec_scatter_mesh, dx_mesh, angles, weights):

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
    angular_flux_mid = np.zeros([N_angles, N_ans], data_type)

    #angular_flux_last = np.zeros([N_angles, N_ans], data_type)
    #angular_flux_mid_last = np.zeros([N_angles, N_ans], data_type)

    angular_flux_last = angular_flux_previous #np.zeros([N_angles, N_ans], data_type)
    angular_flux_mid_last = angular_flux_mid_previous #np.zeros([N_angles, N_ans], data_type)

    scalar_flux = np.zeros(N_ans, data_type)
    scalar_flux_last = np.zeros(N_ans, data_type)
    scalar_flux_next = np.zeros(N_ans, data_type)

    #print('Next TS')
    while source_converged == False:
        
        BCl = utl.BoundaryCondition(sim_perams['boundary_condition_left'],   0, N_mesh, angular_flux=angular_flux, incident_flux_mag=sim_perams['left_in_mag'],  angle=sim_perams['left_in_angle'],  angles=angles)
        BCr = utl.BoundaryCondition(sim_perams['boundary_condition_right'], -1, N_mesh, angular_flux=angular_flux, incident_flux_mag=sim_perams['right_in_mag'], angle=sim_perams['right_in_angle'], angles=angles)
        
        [angular_flux, angular_flux_mid] = OCIMBRun(angular_flux_mid_previous, angular_flux_last, angular_flux_mid_last, source_mesh, xsec_mesh, xsec_scatter_mesh, dx_mesh, dt, velocity, angles, weights, BCl, BCr)

        #calculate current
        current = utl.Current(angular_flux, weights, angles)
        
        #calculate scalar flux for next itteration
        scalar_flux_next = utl.ScalarFlux(angular_flux, weights)
        
        if source_counter > 2:
            #check for convergence
            error_eos = np.linalg.norm(angular_flux_mid - angular_flux_mid_last, ord=2)
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
        angular_flux_mid_last = angular_flux_mid
        
        scalar_flux_last = scalar_flux
        scalar_flux = scalar_flux_next
        
        source_counter  += 1

    return(angular_flux, angular_flux_mid, current, spec_rad, source_counter, source_converged)

# last refers to iteration
# prev refers to previous time step
@nb.jit(nopython=True, parallel=True, cache=True, nogil=True, fastmath=True)
def OCIMBRun(angular_flux_mid_previous, angular_flux_last, angular_flux_midstep_last, source, xsec, xsec_scatter, dx, dt, v, mu, weight, BCl, BCr):
    
    N_mesh = dx.size
    N_ans = (N_mesh*2)
    sizer = mu.size*4
    N_angle = mu.size
    half = int(mu.size/2)

    angular_flux = np.zeros_like(angular_flux_last)
    angular_flux_midstep = np.zeros_like(angular_flux_midstep_last) #(N_angle, N_ans), dtype=np.float64)

    for i in nb.prange(N_mesh):
        
        i_l: int = int(2*i)
        i_r: int = int(2*i+1)

        Q = source[:,i_l:i_r+1]/2
        #angle space time
        
        A = np.zeros((sizer,sizer))
        c = np.zeros((sizer,1))

        # getting really sloppy with the indiceis
        for m in range(N_angle):
            psi_halfLast_L = angular_flux_mid_previous[m, i_l] # known all angles from the last time step in the left cell
            psi_halfLast_R = angular_flux_mid_previous[m, i_r] # known

            if mu[m] < 0:
                if i == N_mesh-1:
                    psi_rightBound          = BCr[m]
                    psi_halfNext_rightBound = BCr[m]
                else:
                    psi_rightBound          = angular_flux_last[m, i_r+1]
                    psi_halfNext_rightBound = angular_flux_midstep_last[m, i_r+1] 
                
                A_small = A_neg(dx[i], v, dt, mu[m], xsec[i])
                c_small = c_neg(dx[i], v, dt, mu[m], Q[m,0], Q[m,1], Q[m,0], Q[m,1], psi_halfLast_L, psi_halfLast_R, psi_rightBound, psi_halfNext_rightBound)
                            
            elif mu[m] > 0:
                if i == 0:
                    psi_leftBound           = BCl[m]
                    psi_halfNext_leftBound  = BCl[m]
                else:
                    psi_leftBound           = angular_flux_last[m, i_l-1]
                    psi_halfNext_leftBound  = angular_flux_midstep_last[m, i_l-1]

                A_small = A_pos(dx[i], v, dt, mu[m], xsec[i])
                c_small = c_pos(dx[i], v, dt, mu[m], Q[m,0], Q[m,1], Q[m,0], Q[m,1], psi_halfLast_L, psi_halfLast_R, psi_leftBound, psi_halfNext_leftBound)

            A[m*4:(m+1)*4, m*4:(m+1)*4] = A_small
            c[m*4:(m+1)*4] = c_small

        S = scatter_source(dx[i], xsec_scatter[i], N_angle, weight)

        A = A - S

        angular_flux_raw = np.linalg.solve(A, c)


        for m in range(N_angle):
            angular_flux[m,i_l]         = angular_flux_raw[4*m,0]
            angular_flux[m,i_r]         = angular_flux_raw[4*m+1,0]
            
            angular_flux_midstep[m,i_l] = angular_flux_raw[4*m+2,0]
            angular_flux_midstep[m,i_r] = angular_flux_raw[4*m+3,0]

    return(angular_flux, angular_flux_midstep)


def neg_flux_fixup(next_angflux):
    for i in range(next_angflux.shape[0]):
        for k in range(next_angflux.shape[1]):
            if next_angflux[i,k] < 0:
                next_angflux[i,k] = 0