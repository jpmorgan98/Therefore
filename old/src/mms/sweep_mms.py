import numpy as np
import matplotlib.pyplot as plt
from mms_auto2 import Q1, Q2, af1, af2
import matplotlib.animation as animation

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

                sf[0, g, 2*i  ] += weights[m] * af_unsorted[helper + 0]
                sf[0, g, 2*i+1] += weights[m] * af_unsorted[helper + 1]
                sf[1, g, 2*i  ] += weights[m] * af_unsorted[helper + 2]
                sf[1, g, 2*i+1] += weights[m] * af_unsorted[helper + 3]

    return(af, sf)


def A_neg(cell, mu, g):
    gamma = (cell.dx*cell.xsec_total[g])/2
    timer = cell.dx/(cell.v[g]*cell.dt)
    timer2 = cell.dx/(2*cell.v[g]*cell.dt)
    a = mu/2

    A_n = np.array([[-a + gamma, a,          timer2,            0],
                    [-a,         -a + gamma, 0,                 timer2],
                    [-timer,     0,          timer - a + gamma, a],
                    [0,          -timer,     -a,                timer -a + gamma]])
    
    return(A_n)



def A_pos(cell, mu, g):
    gamma = (cell.dx*cell.xsec_total[g])/2
    timer = cell.dx/(cell.v[g]*cell.dt)
    timer2 = cell.dx/(2*cell.v[g]*cell.dt)
    a = mu/2

    A_p = np.array([[a + gamma, a,         timer2,            0],
                    [-a,        a + gamma, 0,                 timer2],
                    [-timer,    0,         timer + a + gamma, a],
                    [0,         -timer,    -a,                timer +a + gamma]])

    return(A_p)

def c_pos(cell, group, mu, sf, Q, offset, af_hl_L, af_hl_R, af_LB, af_hn_LB):
    timer2 = cell.dx/(2*cell.v[group] * cell.dt)

    c_pos = np.array([cell.dx/4*(cell.xsec_scatter[group]*sf[offset+0] + Q[offset+0]) + timer2*af_hl_L + mu*af_LB,
                      cell.dx/4*(cell.xsec_scatter[group]*sf[offset+1] + Q[offset+1]) + timer2*af_hl_R,
                      cell.dx/4*(cell.xsec_scatter[group]*sf[offset+2] + Q[offset+2]) + mu*af_hn_LB,
                      cell.dx/4*(cell.xsec_scatter[group]*sf[offset+3] + Q[offset+3])])
    
    #print("c_pos ", Q[0+offset], Q[1+offset], Q[2+offset], Q[3+offset]) Q[2+offset]

    return(c_pos)

def c_neg(cell, group, mu, sf, Q, offset, af_hl_L, af_hl_R, af_RB, af_hn_RB):
    timer2 = cell.dx/(2*cell.v[group] * cell.dt)

    c_neg = np.array([cell.dx/4*(cell.xsec_scatter[group]*sf[offset+0] + Q[offset+0]) + timer2*af_hl_L,
                      cell.dx/4*(cell.xsec_scatter[group]*sf[offset+1] + Q[offset+1]) + timer2*af_hl_R - mu*af_RB,
                      cell.dx/4*(cell.xsec_scatter[group]*sf[offset+2] + Q[offset+2]) ,
                      cell.dx/4*(cell.xsec_scatter[group]*sf[offset+3] + Q[offset+3]) - mu*af_hn_RB])

    #print("c_neg ", Q[0+offset], Q[1+offset], Q[2+offset], Q[3+offset])

    return(c_neg)




def sweep(af_last, af_prev, sf, Q, cells, ps):
    for j in range(ps.N_angles):
        for g in range(2):
            if (ps.angles[j] < 0): # negative sweep
                for i in range(ps.N_cells-1, -1, -1): #(int i=ps.N_cells i-- > 0) # looping backwards with an unsigned counter

                    helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j)
                    index_sf = (i*4) + (g*4*ps.N_cells)

                    af_hl_L = af_prev[helper + 2]
                    af_hl_R = af_prev[helper + 3]

                    if ( i == ps.N_cells - 1 ): #RHS
                        #std::cout << "BC right" << std::endl
                        if (g==0):
                            temp = af1(ps.angles[j], ps.time_val, ps.dt, cells[i].x+cells[i].dx, cells[i].dx)
                        elif(g==1):
                            temp = af2(ps.angles[j], ps.time_val, ps.dt, cells[i].x+cells[i].dx, cells[i].dx)
                        else:
                            print( "MMS is only 2 group (negative sweep bc)" )
                        
                        af_RB    = temp[0]
                        af_hn_RB = temp[2] # BCr[angle] sweep is giving us the same anwser in a time step for time averaged and time edge values

                    else:
                        af_RB    = af_last[((i+1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 0]
                        af_hn_RB = af_last[((i+1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 2]

                    A = A_neg(cells[i], ps.angles[j], g)
                    #A = row2colSq(A_rm)
                    #  c_neg(cell cell, int group, double mu, int angle, std::vector<double> sf, int offset, double af_hl_L, double af_hl_R, double af_RB, double af_hn_RB){
                    c = c_neg(cells[i], g, ps.angles[j], sf, Q, index_sf, af_hl_L, af_hl_R, af_RB, af_hn_RB)

                    x = np.linalg.solve(A, c)

                    af_last[helper+0] = x[0]
                    af_last[helper+1] = x[1]
                    af_last[helper+2] = x[2]
                    af_last[helper+3] = x[3]

            elif (ps.angles[j] > 0): # positive sweep
                for i in range (ps.N_cells):
                    helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j)
                    index_sf = (i*4) + (g*4*ps.N_cells) # location in the sf vector of where we are
                    
                    # index corresponding to this position last time step in af
                    af_hl_L = af_prev[helper + 2]
                    af_hl_R = af_prev[helper + 3]

                    if (i == 0): #LHS boundary condition
                        if (g==0):
                            #print(cells[i].x-cells[i].dx) np.array([0.01186138, 0.00792425, 0.00792425, 0.00794925])#
                            temp = af1(ps.angles[j], ps.time_val, ps.dt, cells[i].x-cells[i].dx, cells[i].dx)
                        elif(g==1):
                            temp = af2(ps.angles[j], ps.time_val, ps.dt, cells[i].x-cells[i].dx, cells[i].dx)
                        else :
                            print( "MMS is only 2 group (negative sweep bc)" )
                        #print(temp)
                        af_LB    = temp[1]
                        af_hn_LB = temp[3]
                    else:
                        af_LB     = af_last[((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 1]
                        af_hn_LB  = af_last[((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 3]

                    #if (i ==0):
                    #    if (g==0):
                    #        print(af_hn_LB)
                    A = A_pos(cells[i], ps.angles[j], g)
                    c = c_pos(cells[i], g, ps.angles[j], sf, Q, index_sf, af_hl_L, af_hl_R, af_LB, af_hn_LB)
                    
                    x = np.linalg.solve(A, c)

                    #print_vec_sd(c)

                    #std::cout << "x" << std::endl
                    #print_vec_sd( c )

                    af_last[helper+0] = x[0]
                    af_last[helper+1] = x[1]
                    af_last[helper+2] = x[2]
                    af_last[helper+3] = x[3]


def computeSF(af, sf, ps):
    # a reduction over angle

    #zeroing out the SF which will be accumulated
    sf *= 0

    for i in range(ps.N_cells): #(int i=0 i<ps.N_cells ++i)
        for g in range(ps.N_groups):#( int g=0 g<ps.N_groups ++g)
            for j in range(ps.N_angles): #( int j=0 j<ps.N_angles ++j)
                sf_index = (i*4) + (g*4*ps.N_cells)
                af_index = (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j)

                sf[sf_index+0] += ps.weights[j] * af[af_index+0]
                sf[sf_index+1] += ps.weights[j] * af[af_index+1]
                sf[sf_index+2] += ps.weights[j] * af[af_index+2]
                sf[sf_index+3] += ps.weights[j] * af[af_index+3]

def compute_g2g( cells, sf, ps, material_source, Q ):
    # Energy is communicated by fuddleing around with the source term in the cell component
    # NOTE: Source is a scalar flux in transport sweeps and is angular flux in OCI! (I don't think this is right)
    

    # reset the Q term to be isotropic material source
    # material_source does not have L R average components so its 2*N_groups
    for c in range(ps.N_cells): #c=0 c<ps.N_cells ++c)
        for g in range (ps.N_groups): #(int g=0 g<ps.N_groups ++g)
            index_sf = (c*4) + (g*4*ps.N_cells)
            Q[index_sf+0] = material_source[index_sf+0]
            Q[index_sf+1] = material_source[index_sf+1]
            Q[index_sf+2] = material_source[index_sf+2]
            Q[index_sf+3] = material_source[index_sf+3]

    # First two for loops are over all group to group scattering matrix
    # these are mostly reduction commands, should use that when heading to GPU if needing to offload

    # Scattering looks like row major allined std::vector<doubles> 
    # g->g'
    #     _       g'       _
    #    | 0->0  0->1  0->2 |  fastest
    #  g | 1->0  1->1  1->2 |     |
    #    | 2->0  2->1  2->2 |     \/
    #    -                 -   slowest
    #  Thus the diagnol is the within group scttering

    for i in range(ps.N_groups):#(int i=0 i<ps.N_groups ++i){ # g
        for j in range(ps.N_groups):#(int j=0 j<ps.N_groups ++j){ # g'

            if (i != j):
                for c in range(ps.N_cells):#(int c=0 c<ps.N_cells ++c){ # across cells

                    index_sf = (c*4) + (j*4*ps.N_cells)

                    Q[index_sf+0] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[index_sf+0]
                    Q[index_sf+1] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[index_sf+1]
                    Q[index_sf+2] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[index_sf+2]
                    Q[index_sf+3] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[index_sf+3]
            else:
                for c in range(ps.N_cells): #(int c=0 c<ps.N_cells ++c)
                    if (cells[c].xsec_g2g_scatter[j+i*ps.N_groups] != 0):
                        print(">>>> warning a g2g scatter xsec is non-zero for a group to group")



def convergenceLoop(af_new, af_previous, cells, ps, material_source):

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

    Q = np.zeros(4*ps.N_cells*ps.N_groups)

    while not converged:

        # communicate energy!
        compute_g2g( cells, sf_new, ps, material_source, Q )
        
        #print(af_new)

        # sweep
        assert ((Q==material_source).all)
        sweep( af_new, af_previous, sf_new, Q, cells, ps )

        #print(af_new)

        # compute scalar fluxes
        computeSF( af_new, sf_new, ps )

        # compute the L2 norm between the last and current iteration
        max = np.max(np.abs((sf_new, sf_last)))
        error = np.max(np.abs(sf_new-sf_last)/max)

        # compute spectral radius
        spec_rad = pow( pow(error+error_n1,2), .5) / pow(pow(error_n1+error_n2, 2), 0.5 )

        # too allow for an error & spectral radius computation we need at least three cycles (indexing from zero)
        if (itter > 2):
            # if relative error between the last and just down iteration end the time step
            # including false solution protection!!!!
            if ( error < ps.convergence_tolerance *(1-spec_rad)):
                converged = True
            if ( (sf_last == sf_new).all ):
                print( ">>> Sweep solutions where exactly the same within double precision" )
                converged = True 

        if (itter >= ps.max_iteration):
            print( ">>>WARNING: Computation did not converge after ", ps.max_iteration, " iterations <<<" )
            print( "       itter: " )
            print( "       error: ")
            print( "")
            converged = True
        

        #std::cout << "af_new" << std::endl
        #print_vec_sd(af_new)
        #std::cout << "sf_new" << std::endl
        #print_vec_sd(sf_new)

        af_last = af_new
        sf_last = sf_new

        error_n2 = error_n1
        error_n1 = error


        # CYCLE PRINTING
        cycle_print_flag = 0 # for printing headers

        if (itter != 0):
            cycle_print_flag = 1
        
        t = 0

        if (cycle_print_flag == 0):
            print( ">>>CYCLE INFO FOR TIME STEP: ", t,"<<<" )
            print("cycle   error         error_n1      error_n2     spec_rad   cycle_time\n")
            print("===================================================================================\n")
            cycle_print_flag = 1
        print("%3d      %1.4e    %1.4e    %1.4e   %1.4e \n" % (itter, error, error_n1, error_n2, spec_rad, ) )

        itter +=1
        
    # 2, ps.N_groups, ps.N_angles, 2*ps.N_cells

    x = np.zeros(ps.N_cells*2)
    for i in range(ps.N_cells): #(int i=0 i<cells.size() i++){
        x[i*2] =  cells[i].x_left
        x[i*2+1] = cells[i].x_left + cells[i].dx/2
    
    temp2 = evaluateMMSaf(cells, ps, ps.time_val)
    print(ps.time_val)
    temp2, temp_2_sf = sort(temp2, ps)
    temp, temp_sf = sort(af_new, ps)

    Q_plotting = np.zeros((2, ps.N_groups, ps.N_cells*2))
    for iq in range(ps.N_cells):
        for gq in range(ps.N_groups):
            sf_helper = (iq*4) + (gq*4*ps.N_cells)
            Q_plotting[0, gq, iq*2]   = Q[sf_helper + 0]
            Q_plotting[0, gq, iq*2+1] = Q[sf_helper + 1]
            Q_plotting[1, gq, iq*2]   = Q[sf_helper + 2]
            Q_plotting[1, gq, iq*2+1] = Q[sf_helper + 3]


    #fig, axs = plt.subplots(2, 2, constrained_layout=True)
    #axs[1, 0].plot(x, temp [0, 0, 0, :], label=r"sweep $\mu=-$")
    #axs[1, 0].plot(x, temp [0, 0, 1, :], label=r"sweep $\mu=+$")
    #axs[1, 0].plot(x, temp2[0, 0, 0, :], label=r"mms $\mu=-$")
    #axs[1, 0].plot(x, temp2[0, 0, 1, :], label=r"mms $\mu=+$")
    #axs[1, 0].set_title(r"G1 Cell integrated time average angular fluxes ($S_2$)")
    #axs[1, 0].set_xlabel("distance [cm]")
    #axs[1, 0].set_ylabel(r"$\psi$")
    #axs[1, 0].legend()
#
    #axs[0, 0].plot(x, temp [1, 0, 0, :], label=r"sweep $\mu=-$")
    #axs[0, 0].plot(x, temp [1, 0, 1, :], label=r"sweep $\mu=+$")
    #axs[0, 0].plot(x, temp2[1, 0, 0, :], label=r"mms $\mu=-$")
    #axs[0, 0].plot(x, temp2[1, 0, 1, :], label=r"mms $\mu=+$")
    #axs[0, 0].set_title(r"G1 Cell integrated time edge angular fluxes ($S_2$)")
    #axs[0, 0].set_xlabel("distance [cm]")
    #axs[0, 0].set_ylabel(r"$\psi$")
    #axs[0, 0].legend()
#
    #axs[1, 1].plot(x, temp [0, 1, 0, :], label=r"sweep $\mu=-$")
    #axs[1, 1].plot(x, temp [0, 1, 1, :], label=r"sweep $\mu=+$")
    #axs[1, 1].plot(x, temp2[0, 1, 0, :], label=r"mms $\mu=-$")
    #axs[1, 1].plot(x, temp2[0, 1, 1, :], label=r"mms $\mu=+$")
    #axs[1, 1].set_title(r"G2 ell integrated time average angular fluxes ($S_2$)")
    #axs[1, 1].set_xlabel("distance [cm]")
    #axs[1, 1].set_ylabel(r"$\psi$")
    #axs[1, 1].legend()
    #
    #axs[0, 1].plot(x, temp [1, 1, 0, :], label=r"sweep $\mu=-$")
    #axs[0, 1].plot(x, temp [1, 1, 1, :], label=r"sweep $\mu=+$")
    #axs[0, 1].plot(x, temp2[1, 1, 0, :], label=r"mms $\mu=-$")
    #axs[0, 1].plot(x, temp2[1, 1, 1, :], label=r"mms $\mu=+$")
    #axs[0, 1].set_title(r"G2 Cell integrated time edge angular fluxes ($S_2$)")
    #axs[0, 1].set_xlabel("distance [cm]")
    #axs[0, 1].set_ylabel(r"$\psi$")
    #axs[0, 1].legend()
    #plt.show()

    #plt.figure(3)
    #plt.plot(x, temp[0, 1, 0, :], label=r"sweep av $\mu=+$")
    #plt.plot(x, temp[1, 1, 0, :], label=r"sweep edge $\mu=+$")
    #plt.plot(x, temp[0, 1, 1, :], label=r"sweep av $\mu=+$")
    #plt.plot(x, temp[1, 1, 1, :], label=r"sweep edge $\mu=+$")
#
    #plt.plot(x, temp[0, 0, 0, :], label=r"sweep av $\mu=+$ g1")
    #plt.plot(x, temp[1, 0, 0, :], label=r"sweep edge $\mu=+ g1$")
    #plt.plot(x, temp[0, 0, 1, :], label=r"sweep av $\mu=+ g1$")
    #plt.plot(x, temp[1, 0, 1, :], label=r"sweep edge $\mu=+ g1$")
    #plt.legend()
    #plt.title("sweep found plotted for whole system")
    #plt.show()
#
    #plt.figure(2)
    #plt.plot(x, Q_plotting[0, 0, :], label=r"time av g1")
    #plt.plot(x, Q_plotting[1, 0, :], label=r"time edge g1")
    #plt.plot(x, Q_plotting[0, 1, :], label=r"time av g2")
    #plt.plot(x, Q_plotting[1, 1, :], label=r"time edge g2")
    #plt.title("Source Q")
    #plt.legend()
    #plt.show()
#


def init_Q(cells, ps):
     Nq_exp = 4*ps.N_groups
     for i in range(ps.N_cells):
         cells[i].Q = np.zeros(Nq_exp, dtype=np.double)

def setMMSource(cells, ps, t):
    material_source = np.zeros(4*ps.N_cells*ps.N_groups)

    for i in range(ps.N_cells): 
        for g in range(2): 
            for m in range(ps.N_angles):

                sf_helper = (i*4) + (g*4*ps.N_cells)

                Sigma_S1  = cells[i].xsec_scatter[0]
                Sigma_S2  = cells[i].xsec_scatter[1]
                Sigma_S12 = cells[i].xsec_g2g_scatter[2]
                Sigma_S21 = cells[i].xsec_g2g_scatter[1]

                if ( g == 0 ):
                    temp = Q1(cells[i].v[0], cells[i].v[1], cells[i].xsec_total[0], cells[i].xsec_total[1], ps.angles[m], Sigma_S1, Sigma_S2, Sigma_S12, Sigma_S21, cells[i].x, cells[i].dx, t, ps.dt )
                elif ( g == 1 ):
                    temp = Q2(cells[i].v[0], cells[i].v[1], cells[i].xsec_total[0], cells[i].xsec_total[1], ps.angles[m], Sigma_S1, Sigma_S2, Sigma_S12, Sigma_S21, cells[i].x, cells[i].dx, t, ps.dt)
                else:
                    print("This MMS Verification is a 2 group problem only")
                
                temp *= ps.weights[m]

                material_source[sf_helper + 0] += temp[0]
                material_source[sf_helper + 1] += temp[1]
                material_source[sf_helper + 2] += temp[2]
                material_source[sf_helper + 3] += temp[3]
    
    return(material_source)



def MMSInitialCond(af, cells, ps):
    print(">>>Generating an initial condition for first time step t={0}".format(ps.t_init))
    for j in range(ps.N_cells): # (int j=0 j<ps.N_cells ++j){
        for g in range(2):
            for m in range(ps.N_angles): # (int m=0 m<ps.N_angles ++m){

                helper = (j*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*m)
                #group 1

                if ( g == 0 ):
                    temp = af1( ps.angles[m], ps.t_init, ps.dt, cells[j].x, cells[j].dx )
                elif ( g == 1 ):
                    temp = af2( ps.angles[m], ps.t_init, ps.dt, cells[j].x, cells[j].dx )
                else:
                    print("This MMS Verification is a 2 group problem only")
                
                af[helper+0] = temp[0]
                af[helper+1] = temp[1]
                af[helper+2] = temp[2]
                af[helper+3] = temp[3]


def evaluateMMSaf(cells, ps, time):
    af = np.zeros((ps.N_mat))
    for j in range(ps.N_cells): # (int j=0 j<ps.N_cells ++j){
        for g in range(2):
            for m in range(ps.N_angles): # (int m=0 m<ps.N_angles ++m){

                helper = (j*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*m)
                #group 1

                if ( g == 0 ):
                    temp = af1( ps.angles[m], time, ps.dt, cells[j].x, cells[j].dx )
                elif ( g == 1 ):
                    temp = af2( ps.angles[m], time, ps.dt, cells[j].x, cells[j].dx )
                else:
                    print("This MMS Verification is a 2 group problem only")
                
                af[helper+0] = temp[0]
                af[helper+1] = temp[1]
                af[helper+2] = temp[2]
                af[helper+3] = temp[3]

    return(af)


def timeLoop(af_previous, cells, ps):

    af_solution = np.zeros( ps.N_mat )

    MMSInitialCond(af_previous, cells, ps)

    data_computed = []
    data_analytic = []
    temp, temp_sf = sort(af_previous, ps)
    data_computed.append(temp)
    data_analytic.append(temp)

# 
    # temp, temp2 = sort(af_previous, ps)
    # print(temp)
    # print(af_previous)
    # plt.plot(x, temp[0, 0, 0, :])
    # plt.plot(x, temp[0, 0, 1, :])
    # plt.plot(x, temp[0, 1, 0, :])
    # plt.plot(x, temp[0, 1, 1, :])
    # plt.plot(x, temp[1, 0, 0, :])
    # plt.plot(x, temp[1, 0, 1, :])
    # plt.plot(x, temp[1, 1, 0, :])
    # plt.plot(x, temp[1, 1, 1, :])
    # plt.show()

    
    for t in range(ps.N_time): # (int t=0 t<ps.N_time ++t){

        ps.time_val = ps.times[t+1]

        # set mms if needed
        material_source = setMMSource(cells, ps, t)

        # run convergence loop
        convergenceLoop(af_solution, af_previous, cells, ps, material_source)

        af_previous = af_solution

        temp2 = evaluateMMSaf(cells, ps, ps.time_val)
        temp2, temp_2_sf = sort(temp2, ps)
        temp, temp_sf = sort(af_solution, ps)

        data_computed.append(temp)
        data_analytic.append(temp2)

    return(data_computed, data_analytic)



if __name__ == '__main__':
    # testing function
    
    # problem definition
    # eventually from an input deck
    dx = .1
    dt = 0.01
    t_init = 0

    N_time = 5
    times = np.linspace(t_init, dt*(N_time), N_time+1)

    v = np.array( (1, .5) )
    #xsec_total = np.array ((1.5454, 0.04568))
    xsec_total = np.array((0,0))
    #xsec_scatter = np.array((0.61789, 0.072534))
    xsec_scatter = np.array((0, 0))
    #vector<double> xsec_scatter = {0,0}
    #double ds = 0.0
    material_source = np.array((0,0,0,0,0,0,0,0)) # isotropic, g1 time_edge g1 time_avg, g2 time_edge, g2 time_avg

    Length = 1
    IC_homo = 0
    
    N_cells = 10 #10
    N_angles = 32
    N_groups = 2

    # 4 = N_subspace (2) * N_subtime (2)
    N_mat = 4 * N_cells * N_angles * N_groups

    # N_cm is the size of the row major vector
    N_rm = N_mat*N_mat

    # homogeneous initial condition vector
    # will be stored as the solution at time=0
    IC = np.zeros(N_mat)
    IC *= IC_homo

    # actual computation below here

    # generate g-l quadrature angles and weights
    #vector<double> weights(N_angles, 0.0)
    #vector<double> angles(N_angles, 0.0)

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
    ps.max_iteration = int(1e4)
    # 0 for vac 1 for reflecting 3 for mms
    # size of the cell blocks in all groups and angle
    ps.SIZE_cellBlocks = ps.N_angles*ps.N_groups*4
    ps.ELEM_cellBlocks = ps.SIZE_cellBlocks*ps.SIZE_cellBlocks
    # size of the group blocks in all angle within a cell
    ps.SIZE_groupBlocks = ps.N_angles*4
    # size of the angle blocks within a group and angle
    ps.SIZE_angleBlocks = 4
    # size of the scalar flux solutions
    ps.ELEM_sf = ps.N_cells*ps.N_groups*4

    cells = []

    for i in range (N_cells): #(int i=0 i<N_cells i++){
        # /*building reeds problem from left to right

        cellCon = cell()
        cellCon.cell_id = i
        if (i == 0 ):
            cellCon.x_left = 0
        else:
            cellCon.x_left = cells[len(cells)-1].x_left+cells[len(cells)-1].dx
        
        cellCon.xsec_scatter = np.array((xsec_scatter[0], xsec_scatter[1]))
        cellCon.x = dx/2 + dx*i
        cellCon.xsec_total = np.array((xsec_total[0], xsec_total[1]))
        cellCon.dx = dx
        cellCon.v = v
        cellCon.dt = dt
        cellCon.material_source = material_source
        cellCon.xsec_g2g_scatter = np.array( (0, 0, 0, 0) )
        #cellCon.xsec_g2g_scatter = vector<double> {0, 0, 0, 0}

        #vector<double> temp (N_angles*N_groups*4, 1.0)
        #for (int p=0 p<temp.size() ++p){temp[p] = Q[0]
        #cellCon.Q = np.ones(4*ps.N_groups)
        #cellCon.N_angle = N_angles

        cells.append(cellCon)

    #print(np.arange(dx/2,(N_cells)*dx, dx))

    init_Q(cells, ps)

    af_previous = np.zeros(N_mat)
    
    

    data_computed, data_analytic = timeLoop(af_previous, cells, ps)

    max1 = np.max(data_computed)
    max2 = np.max(data_analytic)
    max = 1.5*max(max1, max2)


    #### PLOTTING

    x = np.zeros(ps.N_cells*2)
    for i in range(ps.N_cells): #(int i=0 i<cells.size() i++){
        x[i*2] =  cells[i].x_left
        x[i*2+1] = cells[i].x_left + cells[i].dx/2

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[1,0]
    ax4 = axs[1,1]

    print(type(ax1))
    print(type(axs[0,0]))

    line1a, = ax1.plot(x, data_computed [0] [0, 0, 0, :], label=r"sweep $\mu=-$")
    line2a, = ax1.plot(x, data_computed [0] [0, 0, 1, :], label=r"sweep $\mu=+$")
    line3a, = ax1.plot(x, data_analytic [0] [0, 0, 0, :], label=r"mms $\mu=-$")
    line4a, = ax1.plot(x, data_analytic [0] [0, 0, 1, :], label=r"mms $\mu=+$")
    ax1.set_title(r"G1 Cell integrated time average angular fluxes ($S_2$)")
    text1 = ax1.text(0.02, .01, "", transform=ax1.transAxes)
    ax1.set_xlabel("distance [cm]")
    ax1.set_ylabel(r"$\psi$")
    ax1.set_ylim(0, max)
    ax1.legend()

    line1b, = ax2.plot(x, data_computed [0] [1, 0, 0, :], label=r"sweep $\mu=-$")
    line2b, = ax2.plot(x, data_computed [0] [1, 0, 1, :], label=r"sweep $\mu=+$")
    line3b, = ax2.plot(x, data_analytic [0] [1, 0, 0, :], label=r"mms $\mu=-$")
    line4b, = ax2.plot(x, data_analytic [0] [1, 0, 1, :], label=r"mms $\mu=+$")
    ax2.set_title(r"G1 Cell integrated time edge angular fluxes ($S_2$)")
    text2 = ax2.text(0.02, .01, "", transform=ax2.transAxes)
    ax2.set_xlabel("distance [cm]")
    ax2.set_ylabel(r"$\psi$")
    ax2.set_ylim(0, max)
    ax2.legend()

    line1c, = ax3.plot(x, data_computed [0] [0, 1, 0, :], label=r"sweep $\mu=-$")
    line2c, = ax3.plot(x, data_computed [0] [0, 1, 1, :], label=r"sweep $\mu=+$")
    line3c, = ax3.plot(x, data_analytic [0] [0, 1, 0, :], label=r"mms $\mu=-$")
    line4c, = ax3.plot(x, data_analytic [0] [0, 1, 1, :], label=r"mms $\mu=+$")
    ax3.set_title(r"G2 ell integrated time average angular fluxes ($S_2$)")
    text3 = ax3.text(0.02, .01, "", transform=ax3.transAxes)
    ax3.set_xlabel("distance [cm]")
    ax3.set_ylabel(r"$\psi$")
    ax3.set_ylim(0, max)
    ax3.legend()
    
    line1d, = ax4.plot(x, data_computed[0] [1, 1, 0, :], label=r"sweep $\mu=-$")
    line2d, = ax4.plot(x, data_computed[0] [1, 1, 1, :], label=r"sweep $\mu=+$")
    line3d, = ax4.plot(x, data_analytic[0] [1, 1, 0, :], label=r"mms $\mu=-$")
    line4d, = ax4.plot(x, data_analytic[0] [1, 1, 1, :], label=r"mms $\mu=+$")
    ax4.set_title(r"G2 Cell integrated time edge angular fluxes ($S_2$)")
    text4 = ax4.text(0.02, .01, "", transform=ax4.transAxes)
    ax4.set_xlabel("distance [cm]")
    ax4.set_ylabel(r"$\psi$")
    ax4.set_ylim(0, max)
    ax4.legend()

    times_plt = .5*times

    print(times_plt)

    def update(frame):
        # for each frame, update the data stored on each artist.
        # update the line plot:
        line1a.set_ydata(data_computed [frame] [0, 0, 0, :])
        line2a.set_ydata(data_computed [frame] [0, 0, 1, :])
        line3a.set_ydata(data_analytic [frame] [0, 0, 0, :])
        line4a.set_ydata(data_analytic [frame] [0, 0, 1, :])

        line1b.set_ydata(data_computed [frame] [1, 0, 0, :])
        line2b.set_ydata(data_computed [frame] [1, 0, 1, :])
        line3b.set_ydata(data_analytic [frame] [1, 0, 0, :])
        line4b.set_ydata(data_analytic [frame] [1, 0, 1, :])

        line1c.set_ydata(data_computed [frame] [0, 1, 0, :])
        line2c.set_ydata(data_computed [frame] [0, 1, 1, :])
        line3c.set_ydata(data_analytic [frame] [0, 1, 0, :])
        line4c.set_ydata(data_analytic [frame] [0, 1, 1, :])

        line1d.set_ydata(data_computed [frame] [1, 1, 0, :])
        line2d.set_ydata(data_computed [frame] [1, 1, 1, :])
        line3d.set_ydata(data_analytic [frame] [1, 1, 0, :])
        line4d.set_ydata(data_analytic [frame] [1, 1, 1, :])

        text1.set_text(r"$t \in [%.1f,%.1f]$ s" % (times_plt[frame], times_plt[frame + 1]))
        text2.set_text(r"$t = %.1f$ s" % (times[frame]))
        text3.set_text(r"$t \in [%.1f,%.1f]$ s" % (times_plt[frame], times_plt[frame + 1]))
        text4.set_text(r"$t = %.1f$ s" % (times[frame]))

        return ()

    ani = animation.FuncAnimation(fig=fig, func=update, frames=ps.N_time, interval=1000)
    plt.show()