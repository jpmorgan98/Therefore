import numpy as np
import matplotlib.pyplot as plt


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

def A_neg(dx, v, dt, mu, xsec_total):
    gamma = (dx*xsec_total)/2
    timer = dx/(v*dt)
    timer2 = dx/(2*v*dt)
    a = mu/2

    A_n = np.array([[-a + gamma, a,          timer2,            0],
                    [-a,         -a + gamma, 0,                 timer2],
                    [-timer,     0,          timer - a + gamma, a],
                    [0,          -timer,     -a,                timer -a + gamma]])
    
    return(A_n)



def A_pos(dx, v, dt, mu, xsec_total):
    gamma = (dx*xsec_total)/2
    timer = dx/(v*dt)
    timer2 = dx/(2*v*dt)
    a = mu/2

    A_p = np.array([[a + gamma, a,         timer2,            0],
                    [-a,        a + gamma, 0,                 timer2],
                    [-timer,    0,         timer + a + gamma, a],
                    [0,         -timer,    -a,                timer +a + gamma]])

    return(A_p)

def c_pos(dx, v, dt, mu, Ql, Qr, Q_halfNext_L, Q_halfNext_R, psi_halfLast_L, psi_halfLast_R, psi_leftBound, psi_halfNext_leftBound, xsec_scatter, phi_L, phi_R, phi_halfNext_L, phi_halfNext_R):
    timer2 = dx/(2*v*dt)

    b_p = np.array([[dx/4*(xsec_scatter*phi_L + Ql) + timer2*psi_halfLast_L + mu*psi_leftBound],
                    [dx/4*(xsec_scatter*phi_R + Qr) + timer2*psi_halfLast_R],
                    [dx/4*(xsec_scatter*phi_halfNext_L + Q_halfNext_L) + mu*psi_halfNext_leftBound],
                    [dx/4*(xsec_scatter*phi_halfNext_R + Q_halfNext_R)]])

    return(b_p)

def c_neg(dx, v, dt, mu, Ql, Qr, Q_halfNext_L, Q_halfNext_R, psi_halfLast_L, psi_halfLast_R, psi_rightBound, psi_halfNext_rightBound, xsec_scatter, phi_L, phi_R, phi_halfNext_L, phi_halfNext_R):
    timer2 = dx/(2*v*dt)

    b_n = np.array([[dx/4*(xsec_scatter*phi_L + Ql) + timer2*psi_halfLast_L],
                    [dx/4*(xsec_scatter*phi_R + Qr) + timer2*psi_halfLast_R - mu*psi_rightBound],
                    [dx/4*(xsec_scatter*phi_halfNext_L + Q_halfNext_L) ],
                    [dx/4*(xsec_scatter*phi_halfNext_R + Q_halfNext_R) - mu*psi_halfNext_rightBound]])

    return(b_n)

def sweep(af_last, af_prev, sf, cells, ps):
    for j in range(ps.N_angles):
        for g in range(ps.N_groups):
            if (ps.angles[j] < 0) # negative sweep
                for i in range(ps.N_cells, 0, -1) #(int i=ps.N_cells i-- > 0) # looping backwards with an unsigned counter

                    helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j)
                    index_sf = (i*4) + (g*4*ps.N_cells)

                    af_hl_L = af_prev[helper + 2]
                    af_hl_R = af_prev[helper + 3]

                    if ( i == ps.N_cells - 1 ):
                        #std::cout << "BC right" << std::endl
                        if (g==0):
                            temp = AF_g1(ps.angles[j], ps.time_val, ps.dt, cells[i].x+cells[i].dx, cells[i].dx)
                        elif(g==1):
                            temp = AF_g2(ps.angles[j], ps.time_val, ps.dt, cells[i].x+cells[i].dx, cells[i].dx)
                        else:
                            print( "MMS is only 2 group (negative sweep bc)" )
                        
                        af_RB = temp[0]
                        af_hn_RB = temp[2] # BCr[angle]

                    else:
                        af_RB    = af_last[((i+1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 0]
                        af_hn_RB = af_last[((i+1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 2]

                    A = A_neg(cells[i], ps.angles[j], g)
                    #A = row2colSq(A_rm)
                    #  c_neg(cell cell, int group, double mu, int angle, std::vector<double> sf, int offset, double af_hl_L, double af_hl_R, double af_RB, double af_hn_RB){
                    c = c_neg(cells[i], g, ps.angles[j], j, sf, index_sf, af_hl_L, af_hl_R, af_RB, af_hn_RB)

                    np.linalg.solve(A, c)

                    #print_cm(c)

                    af_last[helper+0] = c[0]
                    af_last[helper+1] = c[1]
                    af_last[helper+2] = c[2]
                    af_last[helper+3] = c[3]

            elif (ps.angles[j] > 0): # positive sweep
                for i in range (ps.N_cells):
                    helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j)
                    index_sf = (i*4) + (g*4*ps.N_cells) # location in the sf vector of where we are

                    # index corresponding to this position last time step in af
                    af_hl_L = af_prev[helper + 2]
                    af_hl_R = af_prev[helper + 3]

                    if (i == 0): #LHS boundary condition
                        if (g==0):
                            temp = AF_g1(ps.angles[j], ps.time_val, ps.dt, cells[i].x-cells[i].dx, cells[i].dx)
                        elif(g==1):
                            temp = AF_g2(ps.angles[j], ps.time_val, ps.dt, cells[i].x-cells[i].dx, cells[i].dx)
                            #print_vec_sd(temp)
                        else :
                            print( "MMS is only 2 group (negative sweep bc)" )
                        af_LB = temp[1]
                        af_hn_LB = temp[3]
                    else:
                        af_LB     = af_last[((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 1]
                        af_hn_LB  = af_last[((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 3]

                    A = A_pos(cells[i], ps.angles[j], g)
                    c = c_pos(cells[i], g, ps.angles[j], j, sf, index_sf, af_hl_L, af_hl_R, af_LB, af_hn_LB)
                    
                    #std::cout << "c" << std::endl
                    #print_vec_sd( c )
                    temp = np.linalg.solve(A, c)

                    #print_vec_sd(c)

                    #std::cout << "x" << std::endl
                    #print_vec_sd( c )

                    af_last[helper+0] = temp[0]
                    af_last[helper+1] = temp[1]
                    af_last[helper+2] = temp[2]
                    af_last[helper+3] = temp[3]


computeSF(std::vector<double> &af, std::vector<double> &sf, problem_space &ps){
    # a reduction over angle

    #zeroing out the SF which will be accumulated
    std::fill(sf.begin(), sf.end(), 0.0)

    for (int i=0 i<ps.N_cells ++i){
        for ( int g=0 g<ps.N_groups ++g){
            for ( int j=0 j<ps.N_angles ++j){
                int sf_index = (i*4) + (g*4*ps.N_cells)
                int af_index = (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j)

                outofbounds_check(af_index, af)
                outofbounds_check(sf_index, sf)

                sf[sf_index+0] += ps.weights[j] * af[af_index+0]
                sf[sf_index+1] += ps.weights[j] * af[af_index+1]
                sf[sf_index+2] += ps.weights[j] * af[af_index+2]
                sf[sf_index+3] += ps.weights[j] * af[af_index+3]
            }
        }
    }
}

def compute_g2g( cells, sf, ps ):
    # Energy is communicated by fuddleing around with the source term in the cell component
    # NOTE: Source is a scalar flux in transport sweeps and is angular flux in OCI! (I don't think this is right)
    

    # reset the Q term to be isotropic material source
    # material_source does not have L R average components so its 2*N_groups
    for c in range(ps.N_cells): #c=0 c<ps.N_cells ++c)
        for g in range (ps.N_groups): #(int g=0 g<ps.N_groups ++g)
            cells[c].Q[4*g+0] = cells[c].material_source[2*g+0]
            cells[c].Q[4*g+1] = cells[c].material_source[2*g+0]
            cells[c].Q[4*g+2] = cells[c].material_source[2*g+1]
            cells[c].Q[4*g+3] = cells[c].material_source[2*g+1]

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

                    cells[c].Q[4*i+0] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[index_sf+0]
                    cells[c].Q[4*i+1] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[index_sf+1]
                    cells[c].Q[4*i+2] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[index_sf+2]
                    cells[c].Q[4*i+3] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[index_sf+3]
            else:
                for c in range(ps.N_cells): #(int c=0 c<ps.N_cells ++c)
                    if (cells[c].xsec_g2g_scatter[j+i*ps.N_groups] != 0):
                        print(">>>> warning a g2g scatter xsec is non-zero for a group to group")


def convergenceLoop(af_new, af_previous, cells, ps):

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

    while not converged:

        # communicate energy!
        compute_g2g( cells, sf_new, ps )
        
        # sweep
        sweep( af_new, af_previous, sf_new, cells, ps )

        # compute scalar fluxes
        computeSF( af_new, sf_new, ps )

        # compute the L2 norm between the last and current iteration
        error = infNorm_error( sf_last, sf_new )

        # compute spectral radius
        spec_rad = pow( pow(error+error_n1,2), .5) / pow(pow(error_n1+error_n2, 2), 0.5 )

        # too allow for an error & spectral radius computation we need at least three cycles (indexing from zero)
        if (itter > 2):
            # if relative error between the last and just down iteration end the time step
            # including false solution protection!!!!
            if ( error < ps.convergence_tolerance *(1-spec_rad)):
                converged = True
            if ( sf_last == sf_new ) { 
                print( ">>> Sweep solutions where exactly the same within double precision" )
                converged = True 

        if (itter >= ps.max_iteration)
            print( ">>>WARNING: Computation did not converge after ", ps.max_iteration, " iterations <<<" )
            print( "       itter: " )
            print( "       error: ")
            print( "")
            converged = True
        }
        

        #std::cout << "af_new" << std::endl
        #print_vec_sd(af_new)
        #std::cout << "sf_new" << std::endl
        #print_vec_sd(sf_new)

        af_last = af_new
        sf_last = sf_new

        error_n2 = error_n1
        error_n1 = error


        # CYCLE PRINTING
        int cycle_print_flag = 0 # for printing headers

        if (itter != 0) 
            cycle_print_flag = 1
        
        int t = 0

        if (cycle_print_flag == 0):
            print( ">>>CYCLE INFO FOR TIME STEP: ", t,"<<<" )
            print("cycle   error         error_n1      error_n2     spec_rad   cycle_time\n")
            print("===================================================================================\n")
            cycle_print_flag = 1
        }
        print("%3d      %1.4e    %1.4e    %1.4e   %1.4e \n", itter, error, error_n1, error_n2, spec_rad )


        itter +=1
        
        #std::cout << "" <<std::endl
        #std::cout << "af last" <<std::endl
        #print_vec_sd(af_new)
        #std::cout << "" <<std::endl
        #std::cout << "af new" <<std::endl
        #print_vec_sd(af_new)


# void check_g2g(std::vector<cell> &cells, problem_space &ps){
# 
#     int N_expected = ps.N_groups-1 * ps.N_groups-1
# 
#     if (N_expected < 0) {N_expected = 2}
# 
#     for (int i=0 i<ps.N_cells ++i){
#         if (N_expected != cells[i].xsec_g2g_scatter.size()){
#             std::cout << ">>> Warning: Size of g2g scattering matrix not correct " << std::endl
#             std::cout << "      in cell: " << i << " expected " << N_expected << " got " << cells[i].xsec_g2g_scatter.size() << std::endl
#         }
#     }
#     
# }


# void init_g2g(std::vector<cell> &cells, problem_space &ps){
# 
#     int N_expected = (ps.N_groups-1) * (ps.N_groups-1)
# 
#     for (int i=0 i<ps.N_cells ++i){
#         cells[i].xsec_g2g_scatter = std::vector<double> (N_expected, 0.0)
#     }
# 
# }

# void init_Q(std::vector<cell> &cells, problem_space &ps){
# 
#     int Nq_exp = 4*ps.N_groups
# 
#     for (int i=0 i<ps.N_cells ++i){
#         cells[i].Q = std::vector<double> (Nq_exp, 0.0)
#     }
# }

# void setMMSsourece(std::vector<cell> &cells, problem_space &ps, double t){
# 
#     for (int j=0 j<ps.N_cells ++j){
#         for (int g=0 g<ps.N_groups ++g){
#         for (int m=0 m<ps.N_angles ++m){
# 
#             double Sigma_S1  = cells[j].xsec_g2g_scatter[0]
#             double Sigma_S2  = cells[j].xsec_g2g_scatter[3]
#             double Sigma_S12 = cells[j].xsec_g2g_scatter[2]
#             double Sigma_S21 = cells[j].xsec_g2g_scatter[1]
#             
#             std::vector<double> temp
# 
#             if ( g == 0 ){
#                 temp = Q1(cells[j].v[0], cells[j].v[1], cells[j].xsec_total[0], cells[j].xsec_total[1], ps.angles[m], Sigma_S1, Sigma_S2, Sigma_S12, Sigma_S21, cells[j].x, cells[j].dx, t, ps.dt )
#             } else if ( g == 1) {
#                 temp = Q2(cells[j].v[0], cells[j].v[1], cells[j].xsec_total[0], cells[j].xsec_total[0], ps.angles[m], Sigma_S1, Sigma_S2, Sigma_S12, Sigma_S21, cells[j].x, cells[j].dx, t, ps.dt)
#             } else {
#                 std::cout<<"This MMS Verification is a 2 group problem only" <<std::endl
#             }
# 
#             #group 1
#             cells[j].material_source[0] += temp[0] * ps.weights[m]
#             cells[j].material_source[1] += temp[1] * ps.weights[m]
#             cells[j].material_source[2] += temp[2] * ps.weights[m]
#             cells[j].material_source[3] += temp[3] * ps.weights[m]
#         }
#         }
#     }
# }

# void MMSInitialCond(std::vector<double> &af, std::vector<cell> &cells, problem_space &ps){
# 
#     for (int j=0 j<ps.N_cells ++j){
#         for (int g=0 g<2 ++g){
#         for (int m=0 m<ps.N_angles ++m){
# 
#             int helper = (j*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*m)
#             #group 1
# 
#             std::vector<double> temp
# 
#             if ( g == 0 ){
#                 temp = AF_g1( ps.angles[j], 0, ps.dt, cells[j].x, cells[j].dx )
#             } else if ( g == 1) {
#                 temp = AF_g2( ps.angles[j], 0, ps.dt, cells[j].x, cells[j].dx )
#             } else{
#                 std::cout<<"This MMS Verification is a 2 group problem only" <<std::endl
#             }
# 
#             af[helper+0] = temp[0]
#             af[helper+1] = temp[1]
#             af[helper+2] = temp[2]
#             af[helper+3] = temp[3]
# 
#         }
#         }
#     }
# }

# void MMSFullSol ( std::vector<cell> &cells, problem_space &ps ){
# 
#     std::vector<double> mms_temp(ps.N_mat)
#     std::vector<double> temp(4)
#     int index_start
# 
#     double time_val = ps.t_init
# 
#     #double mu, double t_k, double Deltat, double x_j, double Deltax
# 
#     for (int tp=0 tp<ps.N_time tp++){
#         for (int ip=0 ip<ps.N_cells ip++){
#             for (int gp=0 gp<2 gp++){
#             #for (int gp=0 gp<ps.N_groups gp++){ #manual override for mms 
#                 for (int jp=0 jp<ps.N_angles jp++){
#                     
# 
#                     if ( gp == 0 ){
#                         temp = AF_g1( ps.angles[jp], time_val, ps.dt, cells[ip].x, cells[ip].dx )
#                     } else if ( gp == 1) {
#                         temp = AF_g2( ps.angles[jp], time_val, ps.dt, cells[ip].x, cells[ip].dx )
#                     } else{
#                         std::cout<<"This MMS Verification is a 2 group problem only" <<std::endl
#                     }
#                     index_start = (ip*(ps.SIZE_cellBlocks) + gp*(ps.SIZE_groupBlocks) + 4*jp)
#                     mms_temp[index_start] = temp[0]
#                     mms_temp[index_start+1] = temp[1]
#                     mms_temp[index_start+2] = temp[2]
#                     mms_temp[index_start+3] = temp[3]
#                 }
#             }
#         }
# 
# 
#         
#         string ext = ".csv"
#         string file_name = "mms_sol"
#         string dt = to_string(tp)
# 
#         file_name = file_name + dt + ext
# 
#         std::ofstream output(file_name)
#         output << "TIME STEP: " << tp << "Unsorted solution vector for mms" << endl
#         output << "N_space: " << ps.N_cells << " N_groups: " << ps.N_groups << " N_angles: " << ps.N_angles << endl
#         for (int i=0 i<mms_temp.size() i++){
#             output << mms_temp[i] << "," << endl
#         }
# 
#         time_val += ps.dt
# 
#     
#     }
# 
#     cout << "time integrated mms solutions published " << endl
# }


#void MMSFullSource ( std::vector<cell> &cells, problem_space &ps ){
#
#    std::vector<double> mms_temp(ps.N_mat)
#    std::vector<double> temp(4)
#    int index_start
#
#    double time_val = ps.t_init
#
#    #double mu, double t_k, double Deltat, double x_j, double Deltax
#
#    for (int tp=0 tp<ps.N_time tp++){
#        for (int ip=0 ip<ps.N_cells ip++){
#            for (int gp=0 gp<2 gp++){
#            #for (int gp=0 gp<ps.N_groups gp++){ #manual override for mms 
#                for (int jp=0 jp<ps.N_angles jp++){
#                    double Sigma_S1  = cells[ip].xsec_g2g_scatter[0]
#                    double Sigma_S2  = cells[ip].xsec_g2g_scatter[3]
#                    double Sigma_S12 = cells[ip].xsec_g2g_scatter[2]
#                    double Sigma_S21 = cells[ip].xsec_g2g_scatter[1]
#
#                    if ( gp == 0 ){
#                        temp =  Q1(cells[ip].v[0], cells[ip].v[1], cells[ip].xsec_total[0], cells[ip].xsec_total[1], ps.angles[jp], Sigma_S1, Sigma_S2, Sigma_S12, Sigma_S21, cells[ip].x, cells[ip].dx, tp, ps.dt )
#                    } else if ( gp == 1) {
#                        temp =  Q2(cells[ip].v[0], cells[ip].v[1], cells[ip].xsec_total[0], cells[ip].xsec_total[1], ps.angles[jp], Sigma_S1, Sigma_S2, Sigma_S12, Sigma_S21, cells[ip].x, cells[ip].dx, tp, ps.dt )
#                    } else{
#                        std::cout<<"This MMS Verification is a 2 group problem only" <<std::endl
#                    }
#                    index_start = (ip*(ps.SIZE_cellBlocks) + gp*(ps.SIZE_groupBlocks) + 4*jp)
#                    mms_temp[index_start] = temp[0]
#                    mms_temp[index_start+1] = temp[1]
#                    mms_temp[index_start+2] = temp[2]
#                    mms_temp[index_start+3] = temp[3]
#                }
#            }
#        }
#
#
#        
#        string ext = ".csv"
#        string file_name = "mms_source"
#        string dt = to_string(tp)
#
#        file_name = file_name + dt + ext
#
#        std::ofstream output(file_name)
#        output << "TIME STEP: " << tp << "Unsorted solution vector for mms" << endl
#        output << "N_space: " << ps.N_cells << " N_groups: " << ps.N_groups << " N_angles: " << ps.N_angles << endl
#        for (int i=0 i<mms_temp.size() i++){
#            output << mms_temp[i] << "," << endl
#        }
#
#        time_val += ps.dt
#
#    
#    }
#
#    cout << "time integrated mms solutions published " << endl
#}



def timeLoop(af_previous, cells, ps){

    af_solution = np.zeros( ps.N_mat )

    MMSInitialCond(af_solution, cells, ps)

    MMSFullSol( cells, ps )
    MMSFullSource( cells, ps )

    #check_g2g(cells, ps)

    #ps.time_val = 0

    for (int t=0 t<ps.N_time ++t){

        # set mms if needed
        setMMSsourece(cells, ps, t)

        # run convergence loop
        convergenceLoop(af_solution,  af_previous, cells, ps)

        # save data
        string ext = ".csv"
        string file_name = "Sweep_afluxUnsorted"
        string dt = to_string(t)

        file_name = file_name + dt + ext

        std::ofstream output(file_name)
        output << "TIME STEP: " << t << "Unsorted solution vector" << endl
        output << "N_space: " << ps.N_cells << " N_groups: " << ps.N_groups << " N_angles: " << ps.N_angles << endl
        for (int i=0 i<af_solution.size() i++){
            output << af_solution[i] << "," << endl
        }

        std::ofstream dist("x.csv")
        dist << "x: " << endl
        for (int i=0 i<cells.size() i++){
            dist << cells[i].x_left << "," << endl
            dist << cells[i].x_left + cells[i].dx/2 << "," <<endl
        }

        cout << "file saved under: " << file_name << endl

        # new previous info
        af_previous = af_solution

        ps.time_val += ps.dt
    }
}





if __name__ == '__main__':
    # testing function
    
    # problem definition
    # eventually from an input deck
    dx = .1
    dt = 0.1
    t_init = 0.25
    v = np.array( (1, 1) )
    xsec_total = np.array ((1.5454, 0.04568))
    #vector<double> xsec_scatter = {0.61789, 0.072534}
    #vector<double> xsec_scatter = {0,0}
    #double ds = 0.0
    material_source = np.array((0,0,0,0,0,0,0,0)) # isotropic, g1 time_edge g1 time_avg, g2 time_edge, g2 time_avg

    Length = .4
    IC_homo = 0
    
    N_cells = 4 #10
    N_angles = 2
    N_time = 1
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

    [weights, angles] = np.polynomial.legendre.leggauss(N_angles)

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
    ps.N_rm = N_rm
    ps.N_mat = N_mat
    ps.angles = angles
    ps.weights = weights
    ps.initialize_from_previous = False
    ps.max_iteration = int(4)
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
            cellCon.x_left = cells[cells.size()-1].x_left+cells[cells.size()-1].dx
        
        #cellCon.xsec_scatter = vector<double> {xsec_scatter[0], xsec_scatter[1]}
        cellCon.xsec_total = np.array((xsec_total[0], xsec_total[1]))
        cellCon.dx = dx
        cellCon.v = v
        cellCon.dt = dt
        cellCon.material_source = material_source
        cellCon.xsec_g2g_scatter = np.array( (0, 0, 0, 0))
        #cellCon.xsec_g2g_scatter = vector<double> {0, 0, 0, 0}

        #vector<double> temp (N_angles*N_groups*4, 1.0)
        #for (int p=0 p<temp.size() ++p){temp[p] = Q[0]}
        #cellCon.Q = temp
        #cellCon.N_angle = N_angles

        cells.append(cellCon)

    init_Q(cells, ps)

    af_previous = np.zeros(N_mat, 0)

    timeLoop(af_previous, cells, ps)

