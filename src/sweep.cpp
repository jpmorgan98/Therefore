#include <iostream>
#include <vector>
#include "util.h" // remove when putting in larger file
#include "base_mats.h"
#include "legendre.h"

//compile commands 
// lockhartt cc sweep.cpp -std=c++20
// g++ -g -L -llapack


extern "C" void dgesv_( int *n, int *nrhs, double  *a, int *lda, int *ipiv, double *b, int *lbd, int *info  );

void Axeb( std::vector<double> &A, std::vector<double> &b){

    int N = b.size(); //defined by the method
    int nrhs = 1;
    std::vector<int> ipiv(b.size());
    int info;
    dgesv_( &N, &nrhs, &A[0], &N, &ipiv[0], &b[0], &N, &info );

}


void sweep(std::vector<double> &af_last, std::vector<double> &af_prev, std::vector<double> &sf, std::vector<cell> &cells, problem_space ps){
    for (int j=0; j<ps.N_angles; ++j){
        for (int g=0; g<ps.N_groups; ++g){

            if (ps.angles[j] < 0){ // negative sweep
                for (int i=ps.N_cells; i-- > 0;){ // looping backwards with an unsigned counter

                    int helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);
                    int index_sf = (i*4) + (g*4*ps.N_cells);

                    // index corresponding to this position last time step
                    outofbounds_check( helper + 2, af_prev );
                    outofbounds_check( helper + 3, af_prev );

                    double af_hl_L = af_prev[helper + 2];
                    double af_hl_R = af_prev[helper + 3];
                    double af_RB;
                    double af_hn_RB;

                    if ( i == ps.N_cells - 1 ){

                        std::vector<double> af_bound1 = AF_cellintegrated_timeedge(ps.angles[j], ps.time_val, ps.dt, cells[i].x+cells[i].dx, cells[i].dx);
                        std::vector<double> af_bound2 = AF_cellintegrated_timeaverage(ps.angles[j], ps.time_val, ps.dt, cells[i].x+cells[i].dx, cells[i].dx);

                        af_RB    = 0; // BCr[angle]
                        af_hn_RB = 0; // BCr[angle]



                    } else {
                        outofbounds_check(((i+1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 0, af_last);
                        outofbounds_check(((i+1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 2, af_last);

                        af_RB    = af_last[((i+1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 0];
                        af_hn_RB = af_last[((i+1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 2];
                    }

                    std::vector<double> A_rm = A_neg_rm(cells[i], ps.angles[j], g);
                    std::vector<double> A = row2colSq(A_rm);
                    //  c_neg(cell cell, int group, double mu, int angle, std::vector<double> sf, int offset, double af_hl_L, double af_hl_R, double af_RB, double af_hn_RB){
                    std::vector<double> c = c_neg(cells[i], g, ps.angles[j], j, sf, index_sf, af_hl_L, af_hl_R, af_RB, af_hn_RB);

                    Axeb(A, c);

                    af_last[helper+0] = c[0];
                    af_last[helper+1] = c[1];
                    af_last[helper+2] = c[2];
                    af_last[helper+3] = c[3];

                }
            } else if (ps.angles[j] > 0) { // positive sweep
                for (int i=0; i<ps.N_cells; ++i){
                    int helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);
                    int index_sf = (i*4) + (g*4*ps.N_cells); // location in the sf vector of where we are

                    outofbounds_check( helper+2, af_prev );
                    outofbounds_check( helper+3, af_prev );

                    // index corresponding to this position last time step in af
                    double af_hl_L = af_prev[helper + 2];
                    double af_hl_R = af_prev[helper + 3];
                    double af_LB;
                    double af_hn_LB;

                    if (i == 0){ //LHS boundary condition
                        af_LB     = 0;//ps.boundary_condition[];
                        af_hn_LB  = 0;//ps.boundary_condition[];
                    } else {
                        outofbounds_check( ((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 1, af_last );
                        outofbounds_check( ((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 3, af_last );

                        af_LB     = af_last[((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 1];
                        af_hn_LB  = af_last[((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 3];
                    }

                    std::vector<double> A_rm = A_pos_rm(cells[i], ps.angles[j], g);
                    std::vector<double> A = row2colSq(A_rm);
                    std::vector<double> c = c_pos(cells[i], g, ps.angles[j], j, sf, index_sf, af_hl_L, af_hl_R, af_LB, af_hn_LB);
                    
                    //std::cout << "c" << std::endl;
                    //print_vec_sd( c );
                    Axeb(A, c);

                    //std::cout << "x" << std::endl;
                    //print_vec_sd( c );

                    af_last[helper+0] = c[0];
                    af_last[helper+1] = c[1];
                    af_last[helper+2] = c[2];
                    af_last[helper+3] = c[3];
                }
            }
        }
    }
}

void quadrature(std::vector<double> &angles, std::vector<double> &weights){

    // infered from size of pre-allocated std::vector
    int N_angles = angles.size();

    // allocation for function
    double weights_d[N_angles];
    double angles_d[N_angles];

    // some super-duper fast function that generates everything but in double arrays
    legendre_compute_glr(N_angles, angles_d, weights_d);

    // converting to std::vectors
    for (int i=0; i<N_angles; i++){
        angles[i] = angles_d[i];
        weights[i] = weights_d[i];
    }

}


void computeSF(std::vector<double> &af, std::vector<double> &sf, problem_space &ps){
    // a reduction over angle

    //zeroing out the SF which will be accumulated
    std::fill(sf.begin(), sf.end(), 0.0);

    for (int i=0; i<ps.N_cells; ++i){
        for ( int g=0; g<ps.N_groups; ++g){
            for ( int j=0; j<ps.N_angles; ++j){
                int sf_index = (i*4) + (g*4*ps.N_cells);
                int af_index = (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);

                outofbounds_check(af_index, af);
                outofbounds_check(sf_index, sf);

                sf[sf_index+0] += ps.weights[j] * af[af_index+0];
                sf[sf_index+1] += ps.weights[j] * af[af_index+1];
                sf[sf_index+2] += ps.weights[j] * af[af_index+2];
                sf[sf_index+3] += ps.weights[j] * af[af_index+3];
            }
        }
    }
}

void compute_g2g(std::vector<cell> &cells, std::vector<double> &sf, problem_space &ps ) {
    /* Energy is communicated by fuddleing around with the source term in the cell component
    NOTE: Source is a scalar flux in transport sweeps and is angular flux in OCI! (I don't think this is right)
    */

    // reset the Q term to be isotropic material source
    // material_source does not have L R average components so its 2*N_groups
    for (int c=0; c<ps.N_cells; ++c){
        for (int g=0; g<ps.N_groups; ++g){

            outofbounds_check(4*g+0, cells[c].Q);
            outofbounds_check(4*g+1, cells[c].Q);
            outofbounds_check(4*g+2, cells[c].Q);
            outofbounds_check(4*g+3, cells[c].Q);

            outofbounds_check(2*g+0, cells[c].material_source);
            outofbounds_check(2*g+1, cells[c].material_source);

            cells[c].Q[4*g+0] = cells[c].material_source[2*g+0];
            cells[c].Q[4*g+1] = cells[c].material_source[2*g+0];
            cells[c].Q[4*g+2] = cells[c].material_source[2*g+1];
            cells[c].Q[4*g+3] = cells[c].material_source[2*g+1];
        }
        
    }

    // First two for loops are over all group to group scattering matrix
    // these are mostly reduction commands, should use that when heading to GPU if needing to offload
    for (int i=0; i<ps.N_groups; ++i){ // g
        for (int j=0; j<ps.N_groups; ++j){ // g'

            if (i != j){
                for (int c=0; c<ps.N_cells; ++c){ // across cells

                    outofbounds_check(4*j+0, cells[c].Q);
                    outofbounds_check(4*j+1, cells[c].Q);
                    outofbounds_check(4*j+2, cells[c].Q);
                    outofbounds_check(4*j+3, cells[c].Q);

                    outofbounds_check(i*ps.N_groups + j, cells[c].xsec_g2g_scatter);

                    outofbounds_check(c*4*ps.N_groups + i*4 + 0, sf);
                    outofbounds_check(c*4*ps.N_groups + i*4 + 1, sf);
                    outofbounds_check(c*4*ps.N_groups + i*4 + 2, sf);
                    outofbounds_check(c*4*ps.N_groups + i*4 + 3, sf);

                    cells[c].Q[4*i+0] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[c*4*ps.N_groups + j*4 + 0];
                    cells[c].Q[4*i+1] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[c*4*ps.N_groups + j*4 + 1];
                    cells[c].Q[4*i+2] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[c*4*ps.N_groups + j*4 + 2];
                    cells[c].Q[4*i+3] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[c*4*ps.N_groups + j*4 + 3];

                }
            } else {
                for (int c=0; c<ps.N_cells; ++c){
                    if (cells[c].xsec_g2g_scatter[j+i*ps.N_groups] != 0){
                        std::cout << ">>>> warning a g2g scatter xsec is non-zero for a group to group" << std::endl;
                    }
                }
            }
        }
    }
}


void convergenceLoop(std::vector<double> &af_new,  std::vector<double> &af_previous, std::vector<cell> &cells, problem_space &ps){

    bool converged = true;
    int itter = 0;
    double error = 1.0;
    double error_n1 = 0.5;
    double error_n2 = 0.5;
    double spec_rad = 0;
    std::vector<double> af_last(ps.N_mat);
    std::vector<double> sf_new(ps.ELEM_sf);
    std::vector<double> sf_last(ps.ELEM_sf);

    computeSF( af_previous, sf_new, ps );

    while (converged){

        // communicate energy!
        compute_g2g( cells, sf_new, ps );
        
        // sweep
        sweep( af_new, af_previous, sf_new, cells, ps );

        // compute scalar fluxes
        computeSF( af_new, sf_new, ps );

        // compute the L2 norm between the last and current iteration
        error = infNorm_error( sf_last, sf_new );

        // compute spectral radius
        spec_rad = pow( pow(error+error_n1,2), .5) / pow(pow(error_n1+error_n2, 2), 0.5 );

        // too allow for an error & spectral radius computation we need at least three cycles (indexing from zero)
        if (itter > 2){
            // if relative error between the last and just down iteration end the time step
            // including false solution protection!!!!
            if ( error < ps.convergence_tolerance *(1-spec_rad)){ converged = false; }
            if ( sf_last == sf_new ) { 
                std::cout << ">>> Sweep solutions where exactly the same within double precision" << std::endl;
                converged = false; 
                } 
        }
        if (itter >= ps.max_iteration){
            cout << ">>>WARNING: Computation did not converge after " << ps.max_iteration << " iterations <<<" << endl;
            cout << "       itter: " << itter << endl;
            cout << "       error: " << error << endl;
            cout << "" << endl;
            converged = false;
        }
        

        //std::cout << "af_new" << std::endl;
        //print_vec_sd(af_new);
        //std::cout << "sf_new" << std::endl;
        //print_vec_sd(sf_new);

        af_last = af_new;
        sf_last = sf_new;

        error_n2 = error_n1;
        error_n1 = error;


        // CYCLE PRINTING
        int cycle_print_flag = 0; // for printing headers

        if (itter != 0) 
            cycle_print_flag = 1;
        
        int t = 0;

        if (cycle_print_flag == 0) {
            cout << ">>>CYCLE INFO FOR TIME STEP: " << t <<"<<<"<< endl;
            printf("cycle   error         error_n1      error_n2     spec_rad   cycle_time\n");
            printf("===================================================================================\n");
            cycle_print_flag = 1;
        }
        printf("%3d      %1.4e    %1.4e    %1.4e   %1.4e \n", itter, error, error_n1, error_n2, spec_rad );


        itter++;
        
        //std::cout << "" <<std::endl;
        //std::cout << "af last" <<std::endl;
        //print_vec_sd(af_new);
        //std::cout << "" <<std::endl;
        //std::cout << "af new" <<std::endl;
        //print_vec_sd(af_new);

    } // end while loop
} // end convergence function


void check_g2g(std::vector<cell> &cells, problem_space &ps){

    int N_expected = ps.N_groups-1 * ps.N_groups-1;

    if (N_expected < 0) {N_expected = 2;}

    for (int i=0; i<ps.N_cells; ++i){
        if (N_expected != cells[i].xsec_g2g_scatter.size()){
            std::cout << ">>> Warning: Size of g2g scattering matrix not correct " << std::endl;
            std::cout << "      in cell: " << i << " expected " << N_expected << " got " << cells[i].xsec_g2g_scatter.size() << std::endl;
        }
    }
    
}


void init_g2g(std::vector<cell> &cells, problem_space &ps){

    int N_expected = (ps.N_groups-1) * (ps.N_groups-1);

    for (int i=0; i<ps.N_cells; ++i){
        cells[i].xsec_g2g_scatter = std::vector<double> (N_expected, 0.0);
    }

}

void init_Q(std::vector<cell> &cells, problem_space &ps){

    int Nq_exp = 4*ps.N_groups;

    for (int i=0; i<ps.N_cells; ++i){
        cells[i].Q = std::vector<double> (Nq_exp, 0.0);
    }
}

void timeLoop(std::vector<double> af_previous, std::vector<cell> &cells, problem_space &ps){

    std::vector<double> af_solution( ps.N_mat );

    //check_g2g(cells, ps);

    for (int t=0; t<ps.N_time; ++t){

        // run convergence loop
        convergenceLoop(af_solution,  af_previous, cells, ps);

        // save data
        string ext = ".csv";
        string file_name = "Sweep_afluxUnsorted";
        string dt = to_string(t);

        file_name = file_name + dt + ext;

        std::ofstream output(file_name);
        output << "TIME STEP: " << t << "Unsorted solution vector" << endl;
        output << "N_space: " << ps.N_cells << " N_groups: " << ps.N_groups << " N_angles: " << ps.N_angles << endl;
        for (int i=0; i<af_solution.size(); i++){
            output << af_solution[i] << "," << endl;
        }

        std::ofstream dist("x.csv");
        dist << "x: " << endl;
        for (int i=0; i<cells.size(); i++){
            dist << cells[i].x_left << "," << endl;
            dist << cells[i].x_left + cells[i].dx/2 << "," <<endl;
        }

        cout << "file saved under: " << file_name << endl;

        // new previous info
        af_previous = af_solution;
    }
}



int main(){
    // testing function

    using namespace std;
    
    // problem definition
    // eventually from an input deck
    double dx = .01;
    double dt = 1.0;
    vector<double> v = {1, 1};
    vector<double> xsec_total = {1.5454, 0.04568};
    vector<double> xsec_scatter = {0.61789, 0.072534};
    //vector<double> xsec_scatter = {0,0};
    //double ds = 0.0;
    //vector<double> material_source = {1, 1, 1, 1}; // isotropic, g1 time_edge g1 time_avg, g2 time_edge, g2 time_avg

    double Length = 1;
    double IC_homo = 0;
    
    int N_cells = 100; //10
    int N_angles = 2;
    int N_time = 1;
    int N_groups = 2;

    // 4 = N_subspace (2) * N_subtime (2)
    int N_mat = 4 * N_cells * N_angles * N_groups;

    // N_cm is the size of the row major vector
    int N_rm = N_mat*N_mat;

    // homogeneous initial condition vector
    // will be stored as the solution at time=0
    vector<double> IC(N_mat, 0.0);
    for (int p=0; p<N_cells*2; p++){IC[p] = IC[p]*IC_homo;}

    // actual computation below here

    // generate g-l quadrature angles and weights
    vector<double> weights(N_angles, 0.0);
    vector<double> angles(N_angles, 0.0);

    quadrature(angles, weights);

    problem_space ps;
    ps.L = Length;
    ps.dt = dt;
    ps.dx = dx;
    //ps.ds = ds;
    ps.N_angles = N_angles;
    ps.N_cells = N_cells;
    ps.N_groups = N_groups;
    ps.N_time = N_time;
    ps.N_rm = N_rm;
    ps.N_mat = N_mat;
    ps.angles = angles;
    ps.weights = weights;
    ps.initialize_from_previous = false;
    ps.max_iteration = int(1e3);
    // 0 for vac 1 for reflecting 3 for mms
    ps.boundary_conditions = {0,0};
    // size of the cell blocks in all groups and angle
    ps.SIZE_cellBlocks = ps.N_angles*ps.N_groups*4;
    ps.ELEM_cellBlocks = ps.SIZE_cellBlocks*ps.SIZE_cellBlocks;
    // size of the group blocks in all angle within a cell
    ps.SIZE_groupBlocks = ps.N_angles*4;
    // size of the angle blocks within a group and angle
    ps.SIZE_angleBlocks = 4;
    // size of the scalar flux solutions
    ps.ELEM_sf = ps.N_cells*ps.N_groups*4;

    vector<cell> cells;

    int region_id = 0;

    for (int i=0; i<N_cells; i++){
        // /*building reeds problem from left to right

        cell cellCon;
        cellCon.cell_id = i;
        if (i == 0 )
            cellCon.x_left = 0;
        else
            cellCon.x_left = cells[cells.size()-1].x_left+cells[cells.size()-1].dx;
        
        cellCon.xsec_scatter = vector<double> {xsec_scatter[0], xsec_scatter[1]};
        cellCon.xsec_total = vector<double> {xsec_total[0], xsec_total[1]};
        cellCon.dx = dx;
        cellCon.v = v;
        cellCon.dt = dt;
        cellCon.material_source = material_source;
        cellCon.xsec_g2g_scatter = vector<double> {0, .38211, .92747, 0};
        //cellCon.xsec_g2g_scatter = vector<double> {0, 0, 0, 0};

        //vector<double> temp (N_angles*N_groups*4, 1.0);
        //for (int p=0; p<temp.size(); ++p){temp[p] = Q[0];}
        //cellCon.Q = temp;
        //cellCon.N_angle = N_angles;

        cells.push_back(cellCon);
        
    }

    init_Q(cells, ps);

    std::vector<double> af_previous(N_mat, 0);

    std::cout << "entering conv loop" <<std::endl;

    timeLoop(af_previous, cells, ps);
    std::cout << "done" << std::endl;

    return(1);
}

