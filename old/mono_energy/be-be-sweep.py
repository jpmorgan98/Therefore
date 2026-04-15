import numpy as np
import matplotlib.pyplot as plt
from mms_auto2 import Q1, af1
import matplotlib.animation as animation
import math

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


def Q1_man( v, sigma, sigma_s, x, t, mu ):
    return( 2/v + 2*mu + 2*sigma*(x+t+mu+1) - 2*sigma_s*(x+t+1) )

def af_man(x, t, mu):
    return x+t+mu+1

def sf_man(sigma_s, x, t):
    return 2*sigma_s*(x+t+1)

def sweep(af_last, Q, xsec_t, cells, ps):
    for j in range(ps.N_angles):
        if (ps.angles[j] < 0): # negative sweep
            for i in range(ps.N_cells-1, -1, -1): #(int i=ps.N_cells i-- > 0) # looping backwards with an unsigned counter
                if ( i == ps.N_cells - 1 ): #RHS
                    af_i = af_man( cells[i].x+cells[i].dx, ps.time_val, ps.angles[j] )
                else:
                    af_i  = af_last[(i+1)*ps.N_angles + j]
                af_ip1 = (af_i * ps.angles[j]/cells[i].dx + Q[i]) / ( ps.angles[j]/cells[i].dx + xsec_t )
                af_last[i*ps.N_angles + j] = af_ip1

        elif (ps.angles[j] > 0): # positive sweep
            for i in range (ps.N_cells):
                if ( i == 0 ): #LHS
                    af_i = af_man( cells[i].x-cells[i].dx, ps.time_val, ps.angles[j] )
                else:
                    af_i  = af_last[(i-1)*ps.N_angles + j]
                af_ip1 = (af_i * ps.angles[j]/cells[i].dx + Q[i]) / ( ps.angles[j]/cells[i].dx + xsec_t )
                af_last[i*ps.N_angles + j] = af_ip1 

def computeSF(af, sf, ps):
    sf *= 0
    for i in range(ps.N_cells):
        for j in range(ps.N_angles):
            sf[i] += ps.weights[j]*af[i*ps.N_angles + j]

def sortAF(af):
    af_sort = np.zeros((ps.N_angles, ps.N_cells))

    for i in range(ps.N_cells):
        for j in range(ps.N_angles):
            af_sort[j,i] = af[i*ps.N_angles+j]

    return(af_sort)

def convergenceLoop(sf_new, sf_previous, cells, ps, material_source):

    converged = False
    itter = 0
    error = 1.0
    error_n1 = 0.5
    error_n2 = 0.5
    spec_rad = 0
    af_last = np.zeros(ps.N_mat)
    af_new = np.zeros(ps.N_mat)
    np.copyto(sf_new, sf_previous) 
    sf_last = np.zeros(ps.ELEM_sf)

    #computeSF( af_previous, sf_new, ps )

    Q = np.zeros(ps.ELEM_sf)

    while not converged:
        
        for i in range(ps.N_cells):
            Q[i] = sf_new[i] + material_source[i] + (sf_previous[i]/(cells[0].v*ps.dt))
        
        xsec_t = cells[0].xsec_total + (1/(cells[0].v*ps.dt))

        # sweep
        sweep( af_new, Q, xsec_t, cells, ps ) 

        print(af_new)

        # compute scalar fluxes
        computeSF( af_new, sf_new, ps )

        # compute the L2 norm between the last and current iteration
        error = np.linalg.norm(sf_new - sf_last) / np.linalg.norm(sf_last)

        # compute spectral radius
        spec_rad = pow( pow(error+error_n1,2), .5) / pow(pow(error_n1+error_n2, 2), 0.5 )

        # too allow for an error & spectral radius computation we need at least three cycles (indexing from zero)
        if (itter > 2):
            # if relative error between the last and just down iteration end the time step
            # including false solution protection!!!!
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
            print( ">>>CYCLE INFO FOR TIME STEP: ", t,"<<<" )
            print("cycle   error         error_n1      error_n2     spec_rad   cycle_time\n")
            print("===================================================================================\n")
            cycle_print_flag = 1
        print("%3d      %1.4e    %1.4e    %1.4e   %1.4e \n" % (itter, error, error_n1, error_n2, spec_rad, ) )

        itter +=1


def setMMSource(cells, ps, t):
    material_source = np.zeros(ps.N_cells)

    for i in range(ps.N_cells): 
        for m in range(ps.N_angles):
            sigma_s  = cells[i].xsec_scatter
            material_source[i] += weights[m]*Q1_man(cells[i].v, cells[i].xsec_total, sigma_s, cells[i].x, ps.time_val, ps.angles[m])
    return(material_source)

def MMSInitialCond(sf, cells, ps):
    print(">>>Generating an initial condition for first time step t={0}".format(ps.t_init))
    for i in range(ps.N_cells):
        for j in range(ps.N_angles):
            sf[i] += ps.weights[j]*af_man(cells[j].x, ps.t_init, ps.angles[j])


def evaluateMMSaf(cells, ps):
    af = np.zeros((ps.N_mat))
    for i in range(ps.N_cells):
        for j in range(ps.N_angles):
            af[i*ps.N_cells + j] = af_man( cells[i].x, ps.time_val, ps.angles[j] )
    return(af)

def evaluateMMSsf(cells, ps):
    sf = np.zeros((ps.N_cells))
    for i in range(ps.N_cells):
        sf[i] = sf_man( cells[i].xsec_scatter, cells[i].x, ps.time_val )
    return(sf)

def timeLoop(sf_previous, cells, ps):
    ps.time_val = ps.times[0]

    sf_solution = np.zeros( ps.ELEM_sf )
    #sf_solution = np.zeros( ps.N_mat )

    #MMSInitialCond(sf_previous, cells, ps)
    sf_previous = evaluateMMSsf(cells,ps)

    #print(sf_previous)
    #print(sf)

    #exit()

    data_computed = []
    data_analytic = []
    
    for t in range(ps.N_time): # (int t=0 t<ps.N_time ++t){

        ps.time_val = ps.times[t+1]

        # set mms if needed
        material_source = setMMSource(cells, ps, t)

        print(material_source)
        exit()

        # run convergence loop
        convergenceLoop(sf_solution, sf_previous, cells, ps, material_source)

        np.copyto(sf_previous, sf_solution)

        temp_mms = evaluateMMSsf(cells, ps)

        data_computed.append(sf_solution)
        data_analytic.append(temp_mms)
        #data_analytic_sf.append(temp_a_sf)

        
        #data_analytic_sf.

    return(data_computed, data_analytic)



if __name__ == '__main__':
    # testing function
    
    # problem definition
    # eventually from an input deck
    dx = .1
    dt = 0.1
    t_init = 0

    N_time = 5
    times = np.linspace(t_init, dt*(N_time), N_time+1)

    v = 1.0
    #xsec_total = np.array ((1.5454, 0.04568))
    xsec_total = 0.5
    #xsec_scatter = np.array((0.61789, 0.072534))
    xsec_scatter = 0.1
    #vector<double> xsec_scatter = {0,0}
    #double ds = 0.0
    material_source = np.array((0,0,0,0)) # isotropic, g1 time_edge g1 time_avg, g2 time_edge, g2 time_avg

    Length = 1
    IC_homo = 0
    
    N_cells = 10 #10
    N_angles = 2
    N_groups = 1

    # 4 = N_subspace (2) * N_subtime (2)
    N_mat = N_cells * N_angles * N_groups

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
    ps.SIZE_cellBlocks = ps.N_angles
    ps.ELEM_cellBlocks = ps.SIZE_cellBlocks*ps.SIZE_cellBlocks
    # size of the group blocks in all angle within a cell
    # size of the angle blocks within a group and angle
    ps.SIZE_angleBlocks = 1
    # size of the scalar flux solutions
    ps.ELEM_sf = ps.N_cells

    cells = []

    for i in range (N_cells):

        cellCon = cell()
        cellCon.cell_id = i
        if (i == 0 ):
            cellCon.x_left = 0
        else:
            cellCon.x_left = cells[len(cells)-1].x_left+cells[len(cells)-1].dx
        
        cellCon.xsec_scatter = xsec_scatter
        cellCon.x = dx/2 + dx*i
        cellCon.xsec_total = xsec_total
        cellCon.dx = dx
        cellCon.v = v
        cellCon.dt = dt
        cellCon.material_source = material_source
        cellCon.xsec_g2g_scatter = np.array( (0, 0, 0, 0) )

        cells.append(cellCon)


    sf_previous = np.zeros(N_cells)
    
    

    data_computed, data_analytic = timeLoop(sf_previous, cells, ps)

    max1 = np.max(data_computed)
    max2 = np.max(data_analytic)
    max = 1.5*max(max1, max2)


    #### PLOTTING

    x = np.zeros(ps.N_cells)
    for i in range(ps.N_cells): #(int i=0 i<cells.size() i++){
        x[i] =  cells[i].x
        #x[i*2+1] = cells[i].x_left + cells[i].dx/2

    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    ax1 = axs

    line1a, = ax1.plot(x, data_computed [0] [:], label=r"sweep")
    #line2a, = ax1.plot(x, data_computed [0] [0, 0, :], label=r"sweep $\mu=+$")
    line3a, = ax1.plot(x, data_analytic [0] [:], label=r"mms")
    #line4a, = ax1.plot(x, data_analytic [0] [0, 0, :], label=r"mms $\mu=+$")
    ax1.set_title(r"G1 Cell integrated time scalar angular fluxes ($S_2$)")
    text1 = ax1.text(0.02, .01, "", transform=ax1.transAxes)
    ax1.set_xlabel("distance [cm]")
    ax1.set_ylabel(r"$\phi$")
    ax1.set_ylim(0, max)
    ax1.legend()

    # line1b, = ax2.plot(x, data_computed [0] [1, 0, :], label=r"sweep")
    # #line2b, = ax2.plot(x, data_computed [0] [1, 0, 1, :], label=r"sweep $\mu=+$")
    # line3b, = ax2.plot(x, data_analytic [0] [1, 0, :], label=r"mms")
    # #line4b, = ax2.plot(x, data_analytic [0] [1, 0, 1, :], label=r"mms $\mu=+$")
    # ax2.set_title(r"G1 Cell integrated time edge scalar fluxes ($S_2$)")
    # text2 = ax2.text(0.02, .01, "", transform=ax2.transAxes)
    # ax2.set_xlabel("distance [cm]")
    # ax2.set_ylabel(r"$\phi$")
    # ax2.set_ylim(0, max)
    # ax2.legend()

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

    times_plt = .5*times

    def update(frame):
        # for each frame, update the data stored on each artist.
        # update the line plot:
        line1a.set_ydata(data_computed [frame] [:])
        #line2a.set_ydata(data_computed [frame] [0, 0, 1, :])
        line3a.set_ydata(data_analytic [frame] [:])
        #line4a.set_ydata(data_analytic [frame] [0, 0, 1, :])

        #line1b.set_ydata(data_computed [frame] [1, 0, :])
        #line2b.set_ydata(data_computed [frame] [1, 0, 1, :])
        #line3b.set_ydata(data_analytic [frame] [1, 0, :])
        #line4b.set_ydata(data_analytic [frame] [1, 0, 1, :])

        # line1c.set_ydata(data_computed [frame] [0, 1, 0, :])
        # line2c.set_ydata(data_computed [frame] [0, 1, 1, :])
        # line3c.set_ydata(data_analytic [frame] [0, 1, 0, :])
        # line4c.set_ydata(data_analytic [frame] [0, 1, 1, :])

        # line1d.set_ydata(data_computed [frame] [1, 1, 0, :])
        # line2d.set_ydata(data_computed [frame] [1, 1, 1, :])
        # line3d.set_ydata(data_analytic [frame] [1, 1, 0, :])
        # line4d.set_ydata(data_analytic [frame] [1, 1, 1, :])

        text1.set_text(r"$t \in [%.1f,%.1f]$ s" % (times_plt[frame], times_plt[frame + 1]))
        text2.set_text(r"$t = %.1f$ s" % (times[frame]))
        # text3.set_text(r"$t \in [%.1f,%.1f]$ s" % (times_plt[frame], times_plt[frame + 1]))
        # text4.set_text(r"$t = %.1f$ s" % (times[frame]))

        return ()

    ani = animation.FuncAnimation(fig=fig, func=update, frames=ps.N_time, interval=500)
    plt.show()
    ani.save('mms.gif') #plt.save_fig('mms.gif')