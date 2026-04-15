from logging.handlers import QueueListener
import numpy as np
from .matrix import A_pos, A_neg, b_neg, b_pos, b_pos_interior, b_neg_interior
import therefore.src.utilities as utl
import numba as nb

#import cupyx.scipy.sparse.linalg as gpuLinalg
import scipy.sparse.linalg as cpuLinalg
import scipy.sparse as cpuSpMat
#import cupyx.scipy.sparse as spMat
#import cupy as cu
from scipy.sparse import csr_matrix, lil_matrix
import betterspy

np.set_printoptions(threshold=9999999)



#@profile
def SIMBTimeStepBig(sim_perams, angular_flux_previous, angular_flux_mid_previous, source_mesh, xsec_mesh, xsec_scatter_mesh, dx_mesh, angles, weights):

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

    #source_mesh *= 2

    angular_flux = np.zeros([N_angles, N_ans], data_type)
    angular_flux_mid = np.zeros([N_angles, N_ans], data_type)

    #angular_flux_last = np.zeros([N_angles, N_ans], data_type)
    #angular_flux_mid_last = np.zeros([N_angles, N_ans], data_type)

    angular_flux_last = angular_flux_previous ##
    angular_flux_mid_last = angular_flux_mid_previous # #

    scalar_flux = np.zeros(N_ans, data_type)
    scalar_flux_mid = np.zeros(N_ans, data_type)

    scalar_flux_last = np.zeros(N_ans, data_type)
    scalar_flux_next = np.zeros(N_ans, data_type)

    scalar_flux_mid_next = np.zeros(N_ans, data_type)
 
    A = buildHer(xsec_mesh, dx_mesh, dt, velocity, angles)
    print(A)
    A = csr_matrix(A)
    #A_gpu = cpuSpMat.csr_matrix(A)
    #A = A.todense()

    while source_converged == False:

        BCl = utl.BoundaryCondition(sim_perams['boundary_condition_left'],   0, N_mesh, angular_flux=angular_flux, incident_flux_mag=sim_perams['left_in_mag'],  angle=sim_perams['left_in_angle'],  angles=angles)
        BCr = utl.BoundaryCondition(sim_perams['boundary_condition_right'], -1, N_mesh, angular_flux=angular_flux, incident_flux_mag=sim_perams['right_in_mag'], angle=sim_perams['right_in_angle'], angles=angles)

        b = buildHim(angular_flux_mid_previous, scalar_flux, scalar_flux_mid_next, source_mesh, xsec_scatter_mesh, dx_mesh, dt, velocity, angles, BCl, BCr)
        print(b)
        #b_gpu = cu.asarray(b)
        
        #angular_flux_raw = np.linalg.solve(A, b)
        angular_flux_raw = cpuLinalg.spsolve(A,b)
        #[angular_flux_raw_gpu, info] = gpuLinalg.gmres(A_gpu, b_gpu)
        #angular_flux_raw = cu.asnumpy(angular_flux_raw_gpu.get())
        

        [angular_flux, angular_flux_mid] = resort(angular_flux_raw, angles, N_mesh)

        #print(A.todense())
        #print(b)
        #print()
        #print(angular_flux_raw)
        #print()
        #print(angular_flux)
        #print(angular_flux_mid)


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

        angular_flux_last = angular_flux
        angular_flux_mid_last = angular_flux_mid
        
        scalar_flux_last = scalar_flux
        scalar_flux = scalar_flux_next

        scalar_flux_mid = scalar_flux_mid_next

        source_counter += 1

    return(angular_flux, angular_flux_mid, current, spec_rad, source_counter, source_converged)



def buildHer(xsec, dx, dt, v, mu):
    N_angle = mu.size
    N_mesh = dx.size

    sizer: int = 4 * N_mesh

    A_uge = lil_matrix((4*N_angle*N_mesh, 4*N_angle*N_mesh))

    for j in range(N_angle):

        A_quadrature = lil_matrix((sizer, sizer))#lil_matrix


        for i in range(N_mesh):
            # forming the lower triangular mats 
            # cols of our A_quadrature matrix are filled with the same sub mats
            # forming lower triangular for positive sweeps and reflected 

            start = i+1
            stop = N_mesh
            step = 1
            streaming = np.zeros((4,4))
            
            if   mu[j] < 0: # negaitive
                A = A_neg(dx[i], v, dt, mu[j], xsec[i])
                

                # dense lil ladies on the diagonal
                row_left:  int = 4*(N_mesh - i) - 4
                row_right: int = 4*(N_mesh - i+1) - 4
                col_top:   int = 4*(N_mesh - i) - 4
                col_bot:   int = 4*(N_mesh - i+1) - 4

                A_quadrature[row_left:row_right, col_top:col_bot] = A

                #print(row_left)
                #print(row_right)
                #print(col_top)
                #print(col_bot)
                
                if i > 0:
                    streaming[1,0] += mu[j]
                    streaming[3,2] += mu[j]

                    row_left:  int = 4*(N_mesh - i)
                    row_right: int = 4*(N_mesh - i+1)
                    col_top:   int = 4*(N_mesh - i -1)
                    col_bot:   int = 4*(N_mesh - i)

                    #print('Streaking')
                    #print(row_left)
                    #print(row_right)
                    #print(col_top)
                    #print(col_bot)

                    A_quadrature[row_left:row_right, col_top:col_bot] = streaming

            elif mu[j] > 0: # positive
                A = A_pos(dx[i], v, dt, mu[j], xsec[i])
                streaming[0,1] -= mu[j]
                streaming[2,3] -= mu[j]
            
                # dense lil ladies on the diagonal
                row_left:  int = 4*i
                row_right: int = 4*(i+1)
                col_top:   int = 4*i
                col_bot:   int = 4*(i+1)

                A_quadrature[row_left:row_right, col_top:col_bot] = A

                if i < N_mesh - 1:
                    row_left:  int = 4*(i+1)
                    row_right: int = 4*(i+2)
                    col_top:   int = 4*(i)
                    col_bot:   int = 4*(i+1)
                    A_quadrature[row_left:row_right, col_top:col_bot] = streaming

            

        Ba = 4*N_mesh*j
        Bb = 4*N_mesh*(j+1)

        A_uge[Ba:Bb, Ba:Bb] = A_quadrature

    return(A_uge)

@nb.njit
def buildHim(angular_flux_previous, scalar_flux, scalar_flux_halfNext, Q, xsec_scatter, dx, dt, v, mu, BCl, BCr):
    N_angle = mu.size
    N_mesh = dx.size

    sizer: int = 4 * N_angle

    b_big = np.zeros((4*N_angle*N_mesh))

    for angle in nb.prange(N_angle):
        for i in range(N_mesh):
            i_l: int = int(2*i) # left index of convencnence
            i_r: int = int(2*i+1)

            psi_halfLast_L = angular_flux_previous[angle, i_l]
            psi_halfLast_R = angular_flux_previous[angle, i_r]

            Ql = 2*Q[angle, i_l]
            Qr = 2*Q[angle, i_r]

            if mu[angle] < 0:
                if i == N_mesh-1:
                    b = b_neg(dx[i], v, dt, mu[angle], Ql, Qr, Ql, Qr, psi_halfLast_L, psi_halfLast_R, BCr[angle], BCr[angle], xsec_scatter[i], scalar_flux[i_l], scalar_flux[i_r], scalar_flux_halfNext[i_l], scalar_flux_halfNext[i_r])
                else:
                    b = b_neg_interior(dx[i], v, dt, Ql, Qr, Ql, Qr, psi_halfLast_L, psi_halfLast_R, xsec_scatter[i], scalar_flux[i_l], scalar_flux[i_r], scalar_flux_halfNext[i_l], scalar_flux_halfNext[i_r])
                
                Bup = angle*N_mesh*4 + 4*(N_mesh-i) - 4
                Bbo = angle*N_mesh*4 + 4*((N_mesh-i)+1) - 4

            elif mu[angle] > 0:
                if i == 0: #BCl[angle]
                    b = b_pos(dx[i], v, dt, mu[angle], Ql, Qr, Ql, Qr, psi_halfLast_L, psi_halfLast_R, BCl[angle], BCl[angle], xsec_scatter[i], scalar_flux[i_l], scalar_flux[i_r], scalar_flux_halfNext[i_l], scalar_flux_halfNext[i_r])
                else:
                    b = b_pos_interior(dx[i], v, dt, Ql, Qr, Ql, Qr, psi_halfLast_L, psi_halfLast_R, xsec_scatter[i], scalar_flux[i_l], scalar_flux[i_r], scalar_flux_halfNext[i_l], scalar_flux_halfNext[i_r])

                Bup = angle*N_mesh*4 + 4*i
                Bbo = angle*N_mesh*4 + 4*(i+1)


            b_big[Bup:Bbo] = b[:,0]

    return(b_big)

#@nb.njit
def resort(angular_flux_raw, angles, N_mesh):
    N_angles = angles.size

    af = np.zeros((N_angles, N_mesh*2))
    af_mid = np.zeros((N_angles, N_mesh*2))

    for i in range(N_angles):
        start_LU = int(i*N_mesh*4)
        if angles[i] > 0:
            
            for j in range(N_mesh):

                x = start_LU + j*4
                af[i,2*j]           = angular_flux_raw[x]
                af[i,2*j+1]         = angular_flux_raw[x+1]
                
                af_mid[i,2*j]   = angular_flux_raw[x+2]
                af_mid[i,2*j+1] = angular_flux_raw[x+3]


        elif angles[i] < 0:
            for j in range(N_mesh):
                x = start_LU + (N_mesh - j)*4 - 4 ### THIS IS Right?

                af[i,2*j]           = angular_flux_raw[x]
                af[i,2*j+1]         = angular_flux_raw[x+1]
                
                af_mid[i,2*j]   = angular_flux_raw[x+2]
                af_mid[i,2*j+1] = angular_flux_raw[x+3]

    return(af, af_mid)


if __name__ == '__main__':
    

    N_mesh = int(3)

    xsec = .5*np.ones(N_mesh)
    xsec[1] = 2
    xsec_scatter = 0*np.ones(N_mesh)
    dx = .75*np.ones(N_mesh)
    dt = .5
    v = 1
    weight = np.array([1,1])
    mu = np.array([-1, 1])

    A = buildHer(xsec, dx, dt, v, mu)

    np.set_printoptions(edgeitems=10, linewidth=400)
    print(A.todense())

    #betterspy.show(A)
    #betterspy.write_png("out.png", A)


#def cell():