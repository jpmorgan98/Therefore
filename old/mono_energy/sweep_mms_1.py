import numpy as np
import matplotlib.pyplot as plt
from mms_auto2 import Q1, af1
import matplotlib.animation as animation
import math
from scipy.integrate import quad
import numba as nb

PI = np.pi

"""
The intention of this file is to be as self sufficient as possible when running the method of manufactured solutions
It does this via sweeping, OCI can then be verified by converging to source iterations to within a convergence tol
"""

class problem_space:
    # physical space
    N_cells = None
    N_angles = None
    N_time = None
    N_group = None
    N_mat = None #order of the system (number of linear equations)
    N_rm  = None#size of the whole ass mat
    time_val = None
    Length = None
    dt = None
    dx = None
    t_max = None
    material_source = None
    velocity = None
    L = None
    t_init = None
    time_conv_loop = None
    av_time_per_itter = None
    times = None

    weights = None
    angles = None

    af_last = None

    # computational
    convergence_tolerance = 1e-9
    initialize_from_previous = None
    max_iteration = None

    SIZE_cellBlocks = None
    ELEM_cellBlocks = None
    SIZE_groupBlocks = None
    SIZE_angleBlocks = None
    ELEM_sf = None



class cell:
    cell_id = None
    region_id = None
    N_angle = None
    x_left = None
    x = None
    x_right = None
    xsec_scatter = None # the within group scattering cross section
    xsec_total = None # the total cross section of
    v = None # the velocity of the particles in energy
    xsec_g2g_scatter = None # the group to group scattering terms in each cell
    material_source = None # the actual material source
    
    Q = None
    # Q stores non-homogenous terms in rm-form of dims [4 x N_groups] for normal problems
    # or for MMS: [4 x N_groups x N_angles] where within a group
        # 0   left half cell space integrated, time averaged
        # 1   right half cell space integrated, time averaged
        # 2   left half cell space integrated, time edge
        # 3   right half cell space integrated, time edge

    dx = None
    dt = None

def sort(af_unsorted, ps):
    af = np.zeros((2, ps.N_groups, ps.N_angles, 2*ps.N_cells))
    sf = np.zeros((2, ps.N_groups, 2*ps.N_cells))
    for g in range(ps.N_groups):
        for i in range(ps.N_cells):
            for m in range(ps.N_angles):
                helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*m)

                af[0, g, m, 2*i  ] = af_unsorted[helper + 0]
                af[0, g, m, 2*i+1] = af_unsorted[helper + 1]
                af[1, g, m, 2*i  ] = af_unsorted[helper + 2]
                af[1, g, m, 2*i+1] = af_unsorted[helper + 3]

                sf[0, g, 2*i  ] += ps.weights[m] * af_unsorted[helper + 0]
                sf[0, g, 2*i+1] += ps.weights[m] * af_unsorted[helper + 1]
                sf[1, g, 2*i  ] += ps.weights[m] * af_unsorted[helper + 2]
                sf[1, g, 2*i+1] += ps.weights[m] * af_unsorted[helper + 3]

    return(af, sf)

def recomb(sf, ps):
    sf_recomb = np.zeros((2, ps.N_groups, ps.N_cells))
    for k in range(ps.N_cells):
        sf_recomb[0, 0, k] = sf[0,0,k*2] + sf[0,0,k*2+1] / 2
        sf_recomb[1, 0, k] = sf[1,0,k*2] + sf[1,0,k*2+1] / 2

    return(sf_recomb)


def sort_sf(sf_unsorted, ps):
    sf = np.zeros((2, ps.N_groups, 2*ps.N_cells))
    for g in range(ps.N_groups):
        for i in range(ps.N_cells):
                #helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*m)
                index_sf = (i*4) + (g*4*ps.N_cells)

                sf[0, g, 2*i  ] = sf_unsorted[index_sf + 0]
                sf[0, g, 2*i+1] = sf_unsorted[index_sf + 1]
                sf[1, g, 2*i  ] = sf_unsorted[index_sf + 2]
                sf[1, g, 2*i+1] = sf_unsorted[index_sf + 3]

    return(sf)


# math.sin(x*PI)
# y = (-x**4 + x + 1)
# dy/dx = 1-4*x**3
def af(x,t,mu):
    return (1-mu**2)*math.sin(x*PI)*math.exp(-t)

def sf(x, t):
    return (4/3)*math.exp(-t)*math.sin(x*PI)

def Q(x, v, sigma, sigma_s, t, mu):
    #return 2/v + 2*mu + 2*sigma*(x+t+mu+1) - 2*sigma_s*(x+t+1)
    daf_dt = -math.sin(x*PI)*math.exp(-t)*(1-mu**2) #sin(πx)(μ2−1)e−t
    daf_dx = math.exp(-t)*(1-mu**2)*PI*math.cos(x*PI) #PI*math.cos(x*PI)
    sf_v = sf(x, t)
    af_v = af(x,t,mu)
    return 2/v*daf_dt + 2*mu*daf_dx + 2*sigma*af_v - sigma_s*sf_v

def Q_spaceint_L(t, v, sigma, sigma_s, x, deltax, mu): 
    left_edge = x - deltax/2
    b = quad(Q, left_edge, x, args=(v, sigma, sigma_s, t, mu))[0]
    # symbolic math
    #b = ( deltax * (sigma * (v * (4 * x + 4 * t + 4) + 4 * v * mu - v * deltax) + sigma_s * (v * (-4 * x - 4 * t - 4) + v * deltax) + 4 * v * mu + 4)) / (4 * v)
    
    # my calcs
    #k = 2/v + 2*mu + 2*sigma*(t+mu+1) - 2*sigma_s*(t+1)
    #c = k*x+(sigma-sigma_s)*x**2 - k*(x-deltax/2) - (sigma-sigma_s)*(x-deltax/2)**2
    #assert(math.isclose(b,c))
    b *= (2/deltax)
    return(b)


def Q_spaceint_R(t, v, sigma, sigma_s, x, deltax, mu):
    right_edge = x + deltax/2
    b = quad(Q, x, right_edge, args=(v, sigma, sigma_s, t, mu))[0]
    
    # symbolic math
    #b = -(deltax * (sigma_s * (v * (4 * x + 4 * t + 4) + v * deltax) + sigma * (v * (-4 * x - 4 * t - 4) - 4 * v * mu - v * deltax) - 4 * v * mu - 4)) / (4 * v)
    
    # my calcs
    #k = 2/v + 2*mu + 2*sigma*(t+mu+1) - 2*sigma_s*(t+1)
    #c = k*(x+deltax/2) + (sigma-sigma_s)*(x+deltax/2)**2 - k*(x) - (sigma-sigma_s)*x**2
    #assert(math.isclose(b,c))
    b *= (2/deltax)
    return(b)

def Q1_man(v, sigma, sigma_s, x, deltax, t, deltat, mu):
    Q = np.zeros(4)

    time_edge_past = t-deltat/2
    time_edge_future = t+deltat/2

    Q[0] = quad(Q_spaceint_L, time_edge_past, time_edge_future, args=(v, sigma, sigma_s, x, deltax, mu) )[0] / deltat
    Q[1] = quad(Q_spaceint_R, time_edge_past, time_edge_future, args=(v, sigma, sigma_s, x, deltax, mu) )[0] / deltat
    Q[2] = Q_spaceint_L(time_edge_future, v, sigma, sigma_s, x, deltax, mu)
    Q[3] = Q_spaceint_R(time_edge_future, v, sigma, sigma_s, x, deltax, mu)

    return(Q)

def af_spaceint_L(t, x, deltax, mu):
    #return (2/deltax) * (deltax * (4 * x + 4 * mu - deltax + 4 * t + 4)) / 8
    left_edge = x - deltax/2
    return (2/deltax) * quad(af, left_edge, x, args=(t,mu))[0]

def af_spaceint_R(t, x, deltax, mu):
    #return (2/deltax) * (deltax * (4 * x + 4 * mu + deltax + 4 * t + 4)) / 8
    right_edge = x + deltax/2
    return (2/deltax)* quad(af, x, right_edge, args=(t,mu))[0]

def af_man(x, deltax, t, deltat, mu):
    af = np.zeros(4)

    time_edge_past = t-deltat/2
    time_edge_future = t+deltat/2

    #print(x, deltax, time_edge_past, mu, time_edge_future)

    af[0] = quad(af_spaceint_L, time_edge_past, time_edge_future, args=(x, deltax, mu) )[0] / deltat
    af[1] = quad(af_spaceint_R, time_edge_past, time_edge_future, args=(x, deltax, mu) )[0] / deltat
    af[2] = af_spaceint_L(time_edge_future, x, deltax, mu)
    af[3] = af_spaceint_R(time_edge_future, x, deltax, mu)

    return(af)

def sf_spaceint_L(t, x, deltax):
    #return (2/deltax) * (deltax * (4 * x - deltax + 4 * t + 4)) / 4
    left_edge = x - deltax/2
    return (2/deltax) * quad(sf, left_edge, x, args=(t))[0]

def sf_spaceint_R(t, x, deltax):
    #return (2/deltax) * (deltax * (4 * x + deltax + 4 * t + 4)) / 4
    right_edge = x + deltax/2
    return (2/deltax) * quad(sf, x, right_edge, args=(t))[0]

def sf_man(x, deltax, t, deltat):
    sf = np.zeros(4)

    time_edge_past = t-deltat/2
    time_edge_future = t+deltat/2

    sf[0] = quad(sf_spaceint_L, time_edge_past, time_edge_future, args=(x, deltax) )[0] / deltat
    sf[1] = quad(sf_spaceint_R, time_edge_past, time_edge_future, args=(x, deltax) )[0] / deltat
    sf[2] = sf_spaceint_L(time_edge_future, x, deltax)
    sf[3] = sf_spaceint_R(time_edge_future, x, deltax)

    return(sf)


def evaluateMMSsf(cells, ps, x):
    sf = np.zeros((4*ps.N_cells*ps.N_groups))
    for i in range(ps.N_cells):
        sf_helper = (i*4)

        temp = sf_man( x[i], cells[i].dx, ps.time_val, ps.dt )

        sf[sf_helper+0] = temp[0]
        sf[sf_helper+1] = temp[1]
        sf[sf_helper+2] = temp[2]
        sf[sf_helper+3] = temp[3]

    return(sf)


def A_neg(cell, mu):
    gamma = (cell.dx*cell.xsec_total)/2
    timer = cell.dx/(cell.v*cell.dt)
    timer2 = cell.dx/(2*cell.v*cell.dt)
    a = mu/2

    A_n = np.array([[-a + gamma, a,          timer2,            0],
                    [-a,         -a + gamma, 0,                 timer2],
                    [-timer,     0,          timer - a + gamma, a],
                    [0,          -timer,     -a,                timer -a + gamma]])
    return(A_n)



def A_pos(cell, mu):
    gamma = (cell.dx*cell.xsec_total)/2
    timer = cell.dx/(cell.v*cell.dt)
    timer2 = cell.dx/(2*cell.v*cell.dt)
    a = mu/2

    A_p = np.array([[a + gamma, a,         timer2,            0],
                    [-a,        a + gamma, 0,                 timer2],
                    [-timer,    0,         timer + a + gamma, a],
                    [0,         -timer,    -a,                timer +a + gamma]])

    return(A_p)



def c_pos(cell, mu, sf, Q, offset, offset_af, af_hl_L, af_hl_R, af_LB, af_hn_LB):
    timer2 = cell.dx/(2*cell.v * cell.dt)

    c_pos = np.array([cell.dx/4*(cell.xsec_scatter*sf[offset+0]) + cell.dx/4*Q[offset_af+0] + timer2*af_hl_L + mu*af_LB,
                      cell.dx/4*(cell.xsec_scatter*sf[offset+1]) + cell.dx/4*Q[offset_af+1] + timer2*af_hl_R,
                      cell.dx/4*(cell.xsec_scatter*sf[offset+2]) + cell.dx/4*Q[offset_af+2] + mu*af_hn_LB,
                      cell.dx/4*(cell.xsec_scatter*sf[offset+3]) + cell.dx/4*Q[offset_af+3]])
    return(c_pos)



def c_neg(cell, mu, sf, Q, offset, offset_af, af_hl_L, af_hl_R, af_RB, af_hn_RB):
    timer2 = cell.dx/(2*cell.v * cell.dt)

    c_neg = np.array([cell.dx/4*(cell.xsec_scatter*sf[offset+0]) + cell.dx/4*Q[offset_af+0] + timer2*af_hl_L,
                      cell.dx/4*(cell.xsec_scatter*sf[offset+1]) + cell.dx/4*Q[offset_af+1] + timer2*af_hl_R - mu*af_RB,
                      cell.dx/4*(cell.xsec_scatter*sf[offset+2]) + cell.dx/4*Q[offset_af+2] ,
                      cell.dx/4*(cell.xsec_scatter*sf[offset+3]) + cell.dx/4*Q[offset_af+3] - mu*af_hn_RB])
    return(c_neg)



def sweep(af_last, af_prev, sf, Q, cells, ps, x_mid_cell):
    for j in range(ps.N_angles):
        if (ps.angles[j] < 0): # negative sweep
            for i in range(ps.N_cells-1, -1, -1):

                helper =  (i*(ps.SIZE_cellBlocks) + 4*j)
                index_sf = (i*4)

                af_hl_L = af_prev[helper + 2]
                af_hl_R = af_prev[helper + 3]

                if ( i == ps.N_cells - 1 ): #RHS
                    temp = af_man( x_mid_cell[i]+cells[i].dx, cells[i].dx, ps.time_val, ps.dt, ps.angles[j] )
                    af_RB    = temp[0]
                    af_hn_RB = temp[2]
                else:
                    af_RB    = af_last[((i+1)*(ps.SIZE_cellBlocks) + 4*j) + 0]
                    af_hn_RB = af_last[((i+1)*(ps.SIZE_cellBlocks) + 4*j) + 2]

                A = A_neg(cells[i], ps.angles[j])
                c = c_neg(cells[i], ps.angles[j], sf, Q, index_sf, helper, af_hl_L, af_hl_R, af_RB, af_hn_RB)
                x = np.linalg.solve(A, c)

                af_last[helper+0] = x[0]
                af_last[helper+1] = x[1]
                af_last[helper+2] = x[2]
                af_last[helper+3] = x[3]

        elif (ps.angles[j] > 0): # positive sweep
            for i in range (ps.N_cells):

                helper =  (i*(ps.SIZE_cellBlocks) + 4*j)
                index_sf = (i*4) 
                
                af_hl_L = af_prev[helper + 2]
                af_hl_R = af_prev[helper + 3]

                if (i == 0): #LHS boundary condition
                    temp = af_man( -cells[i].dx/2, cells[i].dx, ps.time_val, ps.dt, ps.angles[j] )
                    af_LB    = temp[1]
                    af_hn_LB = temp[3]
                else:
                    af_LB     = af_last[((i-1)*(ps.SIZE_cellBlocks) + 4*j) + 1]
                    af_hn_LB  = af_last[((i-1)*(ps.SIZE_cellBlocks) + 4*j) + 3]

                A = A_pos(cells[i], ps.angles[j])
                c = c_pos(cells[i], ps.angles[j], sf, Q, index_sf, helper, af_hl_L, af_hl_R, af_LB, af_hn_LB)
                
                x = np.linalg.solve(A, c)

                af_last[helper+0] = x[0]
                af_last[helper+1] = x[1]
                af_last[helper+2] = x[2]
                af_last[helper+3] = x[3]

def computeSF(af, sf, ps):
    #zeroing out the SF which will be accumulated
    sf *= 0
    for i in range(ps.N_cells): 
        for j in range(ps.N_angles): 
            sf_index = (i*4)
            af_index = (i*(ps.SIZE_cellBlocks) + 4*j)

            sf[sf_index+0] += ps.weights[j] * af[af_index+0]
            sf[sf_index+1] += ps.weights[j] * af[af_index+1]
            sf[sf_index+2] += ps.weights[j] * af[af_index+2]
            sf[sf_index+3] += ps.weights[j] * af[af_index+3]



def convergenceLoop(af_new, af_previous, cells, ps, mms_source, x_mid_cell):

    converged = False
    itter = 0
    error = 1.0
    error_n1 = 0.5
    error_n2 = 0.5
    spec_rad = 0
    af_last = np.zeros(ps.N_mat)
    sf_new = np.zeros(ps.ELEM_sf)
    sf_last = np.zeros(ps.ELEM_sf)

    computeSF( af_previous, sf_new, ps )

    Q = np.zeros(4*ps.N_cells*ps.N_angles)
    Q[:] = mms_source[:]
    
    while not converged:
        # sweep
        sweep( af_new, af_previous, sf_new, Q, cells, ps, x_mid_cell )

        #print(af_new)
        # compute scalar fluxes
        computeSF( af_new, sf_new, ps )

        # compute the L2 norm between the last and current iteration
        error = np.linalg.norm(sf_new - sf_last) / np.linalg.norm(sf_last) #np.max(np.abs(sf_new-sf_last)/max)

        # compute spectral radius
        spec_rad = pow( pow(error+error_n1,2), .5) / pow(pow(error_n1+error_n2, 2), 0.5 )

        if (itter > 2):
            if ( error < ps.convergence_tolerance *(1-spec_rad)):
                converged = True
            if ( (error == error_n1) ):
                print( ">>> Sweep solutions where exactly the same within double precision" )
                converged = True 

        if (itter >= ps.max_iteration):
            print( ">>>WARNING: Computation did not converge after ", ps.max_iteration, " iterations <<<" )
            print( "       itter: " )
            print( "       error: ")
            print( "")
            converged = True

        np.copyto(af_last, af_new)
        np.copyto(sf_last, sf_new)

        error_n2 = error_n1
        error_n1 = error

        # CYCLE PRINTING
        cycle_print_flag = 0 # for printing headers

        if (itter != 0):
            cycle_print_flag = 1
        
        t = 0

        if (cycle_print_flag == 0):
            print( ">>>CYCLE INFO FOR TIME STEP: ", ps.time_val,"<<<" )
            print("cycle   error         error_n1      error_n2     spec_rad   cycle_time\n")
            print("===================================================================================\n")
            cycle_print_flag = 1
        print("%3d      %1.4e    %1.4e    %1.4e   %1.4e \n" % (itter, error, error_n1, error_n2, spec_rad, ) )

        itter +=1




def setMMSource(cells, ps, t, x):
    mms_source = np.zeros(4*ps.N_cells*ps.N_angles)

    for i in range(ps.N_cells): 
        for m in range(ps.N_angles):

            sf_helper = (i*4)
            af_helper =  (i*(ps.SIZE_cellBlocks) + 4*m)

            sigma_s  = cells[i].xsec_scatter
            temp = Q1_man(cells[i].v, cells[i].xsec_total, sigma_s, x[i], cells[i].dx, ps.time_val, ps.dt, ps.angles[m])

            mms_source[af_helper + 0] = temp[0]
            mms_source[af_helper + 1] = temp[1]
            mms_source[af_helper + 2] = temp[2]
            mms_source[af_helper + 3] = temp[3]
    
    return(mms_source)

def MMSInitialCond(af, cells, ps, x):
    print(">>>Generating an initial condition for first time step t={0}".format(ps.t_init))

    for i in range(ps.N_cells):
        for m in range(ps.N_angles):

            helper = (i*(ps.SIZE_cellBlocks) + 4*m)
            temp = af_man( x[i], cells[i].dx, ps.t_init, ps.dt, ps.angles[m] )

            af[helper+0] = temp[0]
            af[helper+1] = temp[1]
            af[helper+2] = temp[2]
            af[helper+3] = temp[3]

def evaluateMMSaf(cells, ps, x):
    af = np.zeros((ps.N_mat))
    for i in range(ps.N_cells):
        for m in range(ps.N_angles):

            helper = (i*(ps.SIZE_cellBlocks) + 4*m)
            temp = af_man( x[i], cells[i].dx, ps.time_val, ps.dt, ps.angles[m] )
            
            af[helper+0] = temp[0]
            af[helper+1] = temp[1]
            af[helper+2] = temp[2]
            af[helper+3] = temp[3]

    return(af)




def timeLoop(af_previous, cells, ps, x):

    ps.time_val = ps.times[0]

    af_solution = np.zeros( ps.N_mat )

    MMSInitialCond(af_previous, cells, ps, x)

    data_computed = []
    data_analytic = []
    data_analytic_sf = []

    temp, temp_sf = sort(af_previous, ps)
    temp_a_sf = evaluateMMSsf(cells, ps, x)
    temp_a_sf = sort_sf(temp_a_sf, ps)

    data_computed.append(temp_sf)
    data_analytic.append(temp_a_sf)
    data_analytic_sf.append(temp_a_sf)
    
    for t in range(ps.N_time):

        ps.time_val = ps.times[t+1]

        material_source = setMMSource(cells, ps, t, x)
        

        convergenceLoop(af_solution, af_previous, cells, ps, material_source, x)

        af_previous[:] = af_solution[:]

        temp2 = evaluateMMSaf(cells, ps, x)
        temp_a_sf = evaluateMMSsf(cells, ps, x)

        temp2, temp2_sf = sort(temp2, ps)
        temp, temp_sf = sort(af_solution, ps)

        #print(temp.shape)
        #print(temp)

        temp_a_sf = sort_sf(temp_a_sf, ps)

        data_computed.append(temp_sf)
        data_analytic.append(temp2_sf)
        data_analytic_sf.append(temp_a_sf)

    return(data_computed, data_analytic)





def MMSsweep(dx, dt):
    # testing function
    
    # problem definition
    # eventually from an input deck
    t_init = 0.0
    #t_final = 11.0

    N_time = 1 #int( (t_final-t_init)/dt )
    times = np.linspace(t_init, t_init+dt*(N_time), N_time+1)
    print(times)

    v = .25
    #xsec_total = np.array ((1.5454, 0.04568))
    xsec_total = np.array([0.5])
    #xsec_scatter = np.array((0.61789, 0.072534))
    xsec_scatter = np.array([0.1])
    #vector<double> xsec_scatter = {0,0}
    #double ds = 0.0
    material_source = np.array((0,0,0,0)) # isotropic, g1 time_edge g1 time_avg, g2 time_edge, g2 time_avg

    Length = 1.0
    IC_homo = 0

    N_cells = int(Length/dx) #10
    N_angles = 32
    N_groups = 1

    # 4 = N_subspace (2) * N_subtime (2)
    N_mat = 4 * N_cells * N_angles * N_groups

    # N_cm is the size of the row major vector
    N_rm = N_mat*N_mat

    [angles, weights] = np.polynomial.legendre.leggauss(N_angles)

    ps = problem_space
    ps.L = Length
    ps.dt = dt
    ps.dx = dx
    ps.time_val = t_init
    ps.t_init = t_init
    #ps.ds = ds
    ps.N_angles = N_angles
    ps.N_cells = N_cells
    ps.N_groups = N_groups
    ps.N_time = N_time
    ps.times = times
    ps.N_rm = N_rm
    ps.N_mat = N_mat
    ps.angles = angles
    ps.weights = weights
    ps.initialize_from_previous = False
    ps.max_iteration = int(1e3)
    # 0 for vac 1 for reflecting 3 for mms
    # size of the cell blocks in all groups and angle
    ps.SIZE_cellBlocks = ps.N_angles*ps.N_groups*4
    ps.ELEM_cellBlocks = ps.SIZE_cellBlocks*ps.SIZE_cellBlocks
    # size of the group blocks in all angle within a cell
    ps.SIZE_groupBlocks = ps.N_angles*4
    # size of the angle blocks within a group and angle
    ps.SIZE_angleBlocks = 4
    ps.ELEM_sf = ps.N_cells*4
    # size of the scalar flux solutions
    cells = []

    x = np.zeros(N_cells)

    for i in range (N_cells): 
        cellCon = cell()
        cellCon.cell_id = i
        if (i == 0 ):
            cellCon.x_left = 0
        else:
            cellCon.x_left = cells[len(cells)-1].x_left+cells[len(cells)-1].dx
        cellCon.xsec_scatter = xsec_scatter[0]
        cellCon.x = dx/2 + dx*i
        x[i] = dx/2 + dx*i
        cellCon.xsec_total = xsec_total[0]
        cellCon.dx = dx
        cellCon.v = v
        cellCon.dt = dt
        cellCon.material_source = material_source
        cells.append(cellCon)

    af_previous = np.zeros(N_mat)

    x_plot = np.zeros(ps.N_cells*2)
    for i in range(ps.N_cells): 
        x_plot[i*2] =  cells[i].x_left
        x_plot[i*2+1] = cells[i].x_left + cells[i].dx/2

    data_computed, data_analytic = timeLoop(af_previous, cells, ps, x)
    
    return(data_computed, data_analytic, x_plot)




if __name__ == '__main__':
    dx_vals = np.array([0.001, .5, .25, .1, .05, .025, .01, .005]) #0.001, 0.0001 #, .001, .0001
    dt_vals = np.array([100, 50, 10, 5, 1, .5, .25, .1, .05, .01, .0001, .00001, .000001])
    error_avg = np.zeros(dx_vals.size-1)
    error_edge = np.zeros(dx_vals.size-1)
    
    ti = 0
    for ei in range(dx_vals.size): #dx_vals.size
        dt_spec = .1
        print()
        print(">>>>>>>>>simulation at dx {0} and dt {1}".format(dx_vals[ei], dt_vals[ti]))
        print()

        if (ei==0):
            [data_computed_big, data_analytic_big, x_big] = MMSsweep(dx_vals[ei], dt_spec) # dt_vals[.05]
            N_size = int(x_big.size)
            half_big = int(N_size/2)
            x = x_big
        else:
            [data_computed, data_analytic, x] = MMSsweep(dx_vals[ei], dt_spec) # dt_vals[.05]

        N_size = int(x.size)
        error_avg_vec = np.zeros(N_size)
        error_edge_vec = np.zeros(N_size)
        half = int(N_size/2)

        #for r in range (N_size):
        #    #print(data_computed[-1][0,0,r] - data_analytic[-1][0,0,r])
        #    error_avg_vec[r] = np.abs(data_computed[-1][0,0,r] - data_analytic[-1][0,0,r])
        #    error_edge_vec[r] = np.abs(data_computed[-1][1,0,r] - data_analytic[-1][1,0,r])
        if (ei>0):
            error_avg[ei-1]  = np.abs(data_computed[-1][0,0,half] - data_computed_big[-1][0,0,half_big])#np.linalg.norm(data_computed[-1][0,0,:] - data_analytic[-1][0,0,:]) #/ np.linalg.norm(data_analytic[-1][0,0:])
            error_edge[ei-1] = np.abs(data_computed[-1][1,0,half] - data_computed_big[-1][1,0,half_big]) #np.linalg.norm(data_computed[-1][1,0,:] - data_analytic[-1][1,0,:]) #/ np.linalg.norm(data_analytic[-1][1,0:])
            print(error_avg[ei-1], np.linalg.norm(data_computed[-1][0,0,:] - data_analytic[-1][0,0,:],ord=2))
        #rint(error_avg_vec)
        N_time = 2

    dx_vals = dx_vals[1:]
    plt.figure(1)
    O1 =  dx_vals
    O2 = (dx_vals)**2
    plt.loglog(dx_vals, error_avg, '--*k', label="avg")
    plt.loglog(dx_vals, error_edge, '--^k', label="edge")
    #plt.loglog(dt_vals, O1, '-b', label="O1")
    #plt.loglog(dt_vals, O2, '-r', label="O2")
    plt.legend()
    plt.xlabel(r"$\Delta x$")
    plt.ylabel(r"$||\phi-\phi_r||_2$")
    plt.ylim(top=1)
    plt.grid()
    #plt.title(r'Error after first time step: $\Delta t=$1e-3, $L=1$, $v=0.25$, $\sigma = 0.5$, $\sigma_s = 0.1$, $S_{32}$ ***FANCY NEW MMS****')
    plt.savefig("convergance_rate.pdf")
    plt.show()

    #### PLOTTING
    #print(data_computed.shape)
    max1 = np.max(data_computed_big)
    max2 = np.max(data_analytic_big)
    max = 1.5*max(max1,max2)

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    ax1 = axs[0]
    ax2 = axs[1]
    #ax3 = axs[1,0]
    #ax4 = axs[1,1]

    line1a, = ax1.plot(x_big, data_computed_big [0] [0, 0, :], 'k-*', label=r"sweep")
    #line2a, = ax1.plot(x, data_computed [0] [0, 0, :], label=r"sweep $\mu=+$")
    line3a, = ax1.plot(x_big, data_analytic_big [0] [0, 0, :], 'r', label=r"mms")
    #line4a, = ax1.plot(x, data_analytic [0] [0, 0, :], label=r"mms $\mu=+$")
    ax1.set_title(r"Cell integrated time scalar angular fluxes ($S_{32}$)")
    text1 = ax1.text(0.02, .01, "", transform=ax1.transAxes)
    ax1.set_xlabel("distance [cm]")
    ax1.set_ylabel(r"$\phi$")
    ax1.set_ylim(0, max)
    ax1.legend()

    line1b, = ax2.plot(x_big, data_computed_big [0] [1, 0, :], 'k-*',label=r"sweep")
    #line2b, = ax2.plot(x, data_computed [0] [1, 0, 1, :], label=r"sweep $\mu=+$")
    line3b, = ax2.plot(x_big, data_analytic_big [0] [1, 0, :], 'r', label=r"mms")
    #line4b, = ax2.plot(x, data_analytic [0] [1, 0, 1, :], label=r"mms $\mu=+$")
    ax2.set_title(r"Cell integrated time edge scalar fluxes ($S_{32}$)")
    text2 = ax2.text(0.02, .01, "", transform=ax2.transAxes)
    ax2.set_xlabel("distance [cm]")
    ax2.set_ylabel(r"$\phi$")
    ax2.set_ylim(0, max)
    ax2.legend()

    # line1c, = ax3.plot(x, data_computed [0] [0, 1, 0, :], label=r"sweep $\mu=-$")
    # line2c, = ax3.plot(x, data_computed [0] [0, 1, 1, :], label=r"sweep $\mu=+$")
    # line3c, = ax3.plot(x, data_analytic [0] [0, 1, 0, :], label=r"mms $\mu=-$")
    # line4c, = ax3.plot(x, data_analytic [0] [0, 1, 1, :], label=r"mms $\mu=+$")
    # ax3.set_title(r"G2 ell integrated time average angular fluxes ($S_2$)")
    # text3 = ax3.text(0.02, .01, "", transform=ax3.transAxes)
    # ax3.set_xlabel("distance [cm]")
    # ax3.set_ylabel(r"$\psi$")
    # ax3.set_ylim(0, max)
    # ax3.legend()
    
    # line1d, = ax4.plot(x, data_computed[0] [1, 1, 0, :], label=r"sweep $\mu=-$")
    # line2d, = ax4.plot(x, data_computed[0] [1, 1, 1, :], label=r"sweep $\mu=+$")
    # line3d, = ax4.plot(x, data_analytic[0] [1, 1, 0, :], label=r"mms $\mu=-$")
    # line4d, = ax4.plot(x, data_analytic[0] [1, 1, 1, :], label=r"mms $\mu=+$")
    # ax4.set_title(r"G2 Cell integrated time edge angular fluxes ($S_2$)")
    # text4 = ax4.text(0.02, .01, "", transform=ax4.transAxes)
    # ax4.set_xlabel("distance [cm]")
    # ax4.set_ylabel(r"$\psi$")
    # ax4.set_ylim(0, max)
    # ax4.legend()

    #times_plt = .5*times

    def update(frame):
        # for each frame, update the data stored on each artist.
        # update the line plot:
        line1a.set_ydata(data_computed_big [frame] [0, 0, :])
        #line2a.set_ydata(data_computed [frame] [0, 0, 1, :])
        line3a.set_ydata(data_analytic_big [frame] [0, 0, :])
        #line4a.set_ydata(data_analytic [frame] [0, 0, 1, :])

        line1b.set_ydata(data_computed_big [frame] [1, 0, :])
        #line2b.set_ydata(data_computed [frame] [1, 0, 1, :])
        line3b.set_ydata(data_analytic_big [frame] [1, 0, :])
        #line4b.set_ydata(data_analytic [frame] [1, 0, 1, :])

        # line1c.set_ydata(data_computed [frame] [0, 1, 0, :])
        # line2c.set_ydata(data_computed [frame] [0, 1, 1, :])
        # line3c.set_ydata(data_analytic [frame] [0, 1, 0, :])
        # line4c.set_ydata(data_analytic [frame] [0, 1, 1, :])

        # line1d.set_ydata(data_computed [frame] [1, 1, 0, :])
        # line2d.set_ydata(data_computed [frame] [1, 1, 1, :])
        # line3d.set_ydata(data_analytic [frame] [1, 1, 0, :])
        # line4d.set_ydata(data_analytic [frame] [1, 1, 1, :])

        #text1.set_text(r"$t \in [%.1f,%.1f]$ s" % (times_plt[frame], times_plt[frame + 1]))
        #text2.set_text(r"$t = %.1f$ s" % (times[frame]))
        # text3.set_text(r"$t \in [%.1f,%.1f]$ s" % (times_plt[frame], times_plt[frame + 1]))
        # text4.set_text(r"$t = %.1f$ s" % (times[frame]))

        return ()

    ani = animation.FuncAnimation(fig=fig, func=update, frames=N_time) #, interval=500
    plt.show()
    ani.save('mms.gif') #plt.save_fig('mms.gif')