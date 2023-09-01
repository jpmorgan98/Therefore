from logging.handlers import QueueListener
import numpy as np
from .matrix import A_pos, A_neg, b_neg, b_pos, scatter_source
import therefore.src.utilities as utl
import numba as nb
np.set_printoptions(threshold=9999999)
np.set_printoptions(threshold=9999999)


def SIMBTimeStep(sim_perams, angular_flux_previous, angular_flux_mid_previous, source_mesh, xsec_mesh, xsec_scatter_mesh, dx_mesh, angles, weights):

    velocity = sim_perams['velocity']
    dt = sim_perams['dt']
    N_time = sim_perams['N_time']
    N_angles = sim_perams['N_angles']
    N_mesh = sim_perams['N_mesh']
    data_type = sim_perams['data_type']
    max_it = sim_perams['max loops']
    tol = sim_perams['tolerance']
    printer = sim_perams['print']

    source_converged: bool = False
    source_counter: int = 0
    no_convergence: bool = False
    spec_rad = 0
    
    N_ans = int(2*N_mesh)
    angular_flux = np.zeros([N_angles, N_ans], data_type)
    angular_flux_mid = np.zeros([N_angles, N_ans], data_type)

    #angular_flux_last = np.zeros([N_angles, N_ans], data_type)
    #angular_flux_mid_last = np.zeros([N_angles, N_ans], data_type)

    scalar_flux = np.zeros(N_ans, data_type)
    scalar_flux_mid = np.zeros(N_ans, data_type)

    scalar_flux_last = np.zeros(N_ans, data_type)
    scalar_flux_next = np.zeros(N_ans, data_type)

    scalar_flux_mid_next = np.zeros(N_ans, data_type)

    while source_converged == False:

        BCl = utl.BoundaryCondition(sim_perams['boundary_condition_left'],   0, N_mesh, angular_flux=angular_flux, incident_flux_mag=sim_perams['left_in_mag'],  angle=sim_perams['left_in_angle'],  angles=angles)
        BCr = utl.BoundaryCondition(sim_perams['boundary_condition_right'], -1, N_mesh, angular_flux=angular_flux, incident_flux_mag=sim_perams['right_in_mag'], angle=sim_perams['right_in_angle'], angles=angles)

        [angular_flux, angular_flux_mid] = Itteration(angular_flux_mid_previous, scalar_flux, scalar_flux_mid_next, source_mesh, xsec_mesh, xsec_scatter_mesh, dx_mesh, dt, velocity, angles, BCl, BCr)

        #calculate current
        current = utl.Current(angular_flux, weights, angles)
        
        #calculate scalar flux for next itteration
        scalar_flux_next = utl.ScalarFlux(angular_flux, weights)
        scalar_flux_mid_next = utl.ScalarFlux(angular_flux_mid, weights)

        if source_counter > 2:
            #check for convergence
            error_eos = np.linalg.norm(scalar_flux_mid_next - scalar_flux_mid, ord=2)
            error_mos = np.linalg.norm(scalar_flux_next - scalar_flux, ord=2)

            if error_eos < tol and error_mos < tol:
                source_converged = True

            spec_rad = np.linalg.norm(scalar_flux_next - scalar_flux, ord=2) / np.linalg.norm((scalar_flux - scalar_flux_last), ord=2)

        if source_counter > max_it:
            print('Error source not converged after max iterations')
            print()
            source_converged = True
            no_convergence = True
        
        scalar_flux_last = scalar_flux
        scalar_flux = scalar_flux_next

        scalar_flux_mid = scalar_flux_mid_next

        source_counter += 1

    return(angular_flux, angular_flux_mid, current, spec_rad, source_counter, source_converged)

@nb.jit(nopython=True, parallel=True, cache=True, nogil=True, fastmath=True)
def Itteration(angular_flux_previous, scalar_flux, scalar_flux_halfNext, Q, xsec, xsec_scatter, dx, dt, v, mu, BCl, BCr):
    N_angle = mu.size
    N_mesh = dx.size

    sizer: int = 4

    half = int(mu.size/2)

    angular_flux_next = np.zeros_like(angular_flux_previous)
    angular_flux_mid_next = np.zeros_like(angular_flux_previous)

    for angle in nb.prange(N_angle):
        if mu[angle] < 0:
            for i in range(N_mesh-1, -1, -1):
                i_l: int = int(2*i)
                i_r: int = int(2*i+1)

                psi_halfLast_L = angular_flux_previous[angle, i_l]
                psi_halfLast_R = angular_flux_previous[angle, i_r]

                Ql = Q[angle, i_l]
                Qr = Q[angle, i_r]

                A = np.zeros((sizer,sizer))
                b = np.zeros((sizer,1))

                if i == N_mesh-1:
                    psi_rightBound          = BCr[angle]
                    psi_halfNext_rightBound = BCr[angle]
                else:
                    psi_rightBound          = angular_flux_next[angle, i_r+1]
                    psi_halfNext_rightBound = angular_flux_mid_next[angle, i_r+1]

                A = A_neg(dx[i], v, dt, mu[angle], xsec[i])
                b = b_neg(dx[i], v, dt, mu[angle], Ql, Qr, Ql, Qr, psi_halfLast_L, psi_halfLast_R, psi_rightBound, psi_halfNext_rightBound, xsec_scatter[i], scalar_flux[i_l], scalar_flux[i_r], scalar_flux_halfNext[i_l], scalar_flux_halfNext[i_r])
                
                psi_raw = np.linalg.solve(A, b)
                
                angular_flux_next[angle, i_l] = psi_raw[0,0]
                angular_flux_next[angle, i_r] = psi_raw[1,0]
                angular_flux_mid_next[angle, i_l] = psi_raw[2,0]
                angular_flux_mid_next[angle, i_r] = psi_raw[3,0]

        elif mu[angle] > 0:
            for i in range(N_mesh):

                i_l: int = int(2*i)
                i_r: int = int(2*i+1)

                psi_halfLast_L = angular_flux_previous[angle, i_l]
                psi_halfLast_R = angular_flux_previous[angle, i_r]

                Ql = Q[angle, i_l]
                Qr = Q[angle, i_r]

                A = np.zeros((sizer,sizer))
                b = np.zeros((sizer,1))

                #print('POSITIVE')
                if i == 0:
                    psi_leftBound           = BCl[angle]
                    psi_halfNext_leftBound  = BCl[angle]
                else:
                    psi_leftBound           = angular_flux_next[angle, i_l-1]
                    psi_halfNext_leftBound  = angular_flux_mid_next[angle, i_l-1]

                A = A_pos(dx[i], v, dt, mu[angle], xsec[i])
                b = b_pos(dx[i], v, dt, mu[angle], Ql, Qr, Ql, Qr, psi_halfLast_L, psi_halfLast_R, psi_leftBound, psi_halfNext_leftBound, xsec_scatter[i], scalar_flux[i_l], scalar_flux[i_r], scalar_flux_halfNext[i_l], scalar_flux_halfNext[i_r])
            
                psi_raw = np.linalg.solve(A, b)
                
                angular_flux_next[angle, i_l] = psi_raw[0,0]
                angular_flux_next[angle, i_r] = psi_raw[1,0]
                angular_flux_mid_next[angle, i_l] = psi_raw[2,0]
                angular_flux_mid_next[angle, i_r] = psi_raw[3,0]

            #print('A')
            #print(A)
            #print('b')
            #print(b)
            #print('Raw')
            #print(psi_raw)
            #print('Resorted')
            #print(angular_flux_next)
            #print(angular_flux_mid_next)

    return(angular_flux_next, angular_flux_mid_next)




#def cell():