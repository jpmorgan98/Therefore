#include <iostream>
#include <vector>
#include "../util.h" // remove when putting in larger file
#include "../base_mats.h"
#include "../legendre.h"

#include "mms2.h"

//compile commands 
// lockhartt cc sweep.cpp -std=c++20
// g++ -g -L -llapack
// my mac: g++-14 -g -std=c++20 -llapack sweep_mms.cpp


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
                        //std::cout << "BC right" << std::endl;
                        std::vector<double> temp;
                        if (g==0){
                            temp = AF_g1(ps.angles[j], ps.time_val, ps.dt, cells[i].x+cells[i].dx, cells[i].dx);
                        } else if(g==1) {
                            temp = AF_g2(ps.angles[j], ps.time_val, ps.dt, cells[i].x+cells[i].dx, cells[i].dx);
                        } else {
                            std::cout << "MMS is only 2 group (negative sweep bc)" << std::endl;
                        }
                        af_RB = temp[0];
                        af_hn_RB = temp[2]; // BCr[angle]

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

                    //print_cm(c);

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
                        std::vector<double> temp;
                        if (g==0){
                            temp = AF_g1(ps.angles[j], ps.time_val, ps.dt, cells[i].x-cells[i].dx, cells[i].dx);
                        } else if(g==1) {
                            temp = AF_g2(ps.angles[j], ps.time_val, ps.dt, cells[i].x-cells[i].dx, cells[i].dx);
                            //print_vec_sd(temp);
                        } else {
                            std::cout << "MMS is only 2 group (negative sweep bc)" << std::endl;
                        }
                        af_LB = temp[1];
                        af_hn_LB = temp[3];

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

                    //print_vec_sd(c);

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

    // Scattering looks like row major allined std::vector<doubles> 
    // g->g'
    //     _       g'       _
    //    | 0->0  0->1  0->2 |  fastest
    //  g | 1->0  1->1  1->2 |     |
    //    | 2->0  2->1  2->2 |     \/
    //    -                 -   slowest
    //  Thus the diagnol is the within group scttering

    for (int i=0; i<ps.N_groups; ++i){ // g
        for (int j=0; j<ps.N_groups; ++j){ // g'

            if (i != j){
                for (int c=0; c<ps.N_cells; ++c){ // across cells

                    int index_sf = (c*4) + (j*4*ps.N_cells);

                    outofbounds_check(4*j+0, cells[c].Q);
                    outofbounds_check(4*j+1, cells[c].Q);
                    outofbounds_check(4*j+2, cells[c].Q);
                    outofbounds_check(4*j+3, cells[c].Q);

                    outofbounds_check(i*ps.N_groups + j, cells[c].xsec_g2g_scatter);

                    outofbounds_check(c*4*ps.N_groups + i*4 + 0, sf);
                    outofbounds_check(c*4*ps.N_groups + i*4 + 1, sf);
                    outofbounds_check(c*4*ps.N_groups + i*4 + 2, sf);
                    outofbounds_check(c*4*ps.N_groups + i*4 + 3, sf);

                    cells[c].Q[4*i+0] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[index_sf+0];
                    cells[c].Q[4*i+1] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[index_sf+1];
                    cells[c].Q[4*i+2] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[index_sf+2];
                    cells[c].Q[4*i+3] += cells[c].xsec_g2g_scatter[j+i*ps.N_groups] * sf[index_sf+3];

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

        print_vec_sd(af_new);

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

void setMMSsourece(std::vector<cell> &cells, problem_space &ps, double t){

    for (int j=0; j<ps.N_cells; ++j){
        for (int g=0; g<ps.N_groups; ++g){
        for (int m=0; m<ps.N_angles; ++m){

            double Sigma_S1  = cells[j].xsec_g2g_scatter[0];
            double Sigma_S2  = cells[j].xsec_g2g_scatter[3];
            double Sigma_S12 = cells[j].xsec_g2g_scatter[2];
            double Sigma_S21 = cells[j].xsec_g2g_scatter[1];
            
            std::vector<double> temp;

            if ( g == 0 ){
                temp = Q1(cells[j].v[0], cells[j].v[1], cells[j].xsec_total[0], cells[j].xsec_total[1], ps.angles[m], Sigma_S1, Sigma_S2, Sigma_S12, Sigma_S21, cells[j].x, cells[j].dx, t, ps.dt );
            } else if ( g == 1) {
                temp = Q2(cells[j].v[0], cells[j].v[1], cells[j].xsec_total[0], cells[j].xsec_total[0], ps.angles[m], Sigma_S1, Sigma_S2, Sigma_S12, Sigma_S21, cells[j].x, cells[j].dx, t, ps.dt);
            } else {
                std::cout<<"This MMS Verification is a 2 group problem only" <<std::endl;
            }

            //group 1
            cells[j].material_source[0] += temp[0] * ps.weights[m];
            cells[j].material_source[1] += temp[1] * ps.weights[m];
            cells[j].material_source[2] += temp[2] * ps.weights[m];
            cells[j].material_source[3] += temp[3] * ps.weights[m];
        }
        }
    }
}

void MMSInitialCond(std::vector<double> &af, std::vector<cell> &cells, problem_space &ps){

    for (int j=0; j<ps.N_cells; ++j){
        for (int g=0; g<2; ++g){
        for (int m=0; m<ps.N_angles; ++m){

            int helper = (j*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*m);
            //group 1

            std::vector<double> temp;

            if ( g == 0 ){
                temp = AF_g1( ps.angles[m], 0, ps.dt, cells[j].x, cells[j].dx );
            } else if ( g == 1) {
                temp = AF_g2( ps.angles[m], 0, ps.dt, cells[j].x, cells[j].dx );
            } else{
                std::cout<<"This MMS Verification is a 2 group problem only" <<std::endl;
            }

            af[helper+0] = temp[0];
            af[helper+1] = temp[1];
            af[helper+2] = temp[2];
            af[helper+3] = temp[3];

        }
        }
    }

}

void MMSFullSol ( std::vector<cell> &cells, problem_space &ps ){

    std::vector<double> mms_temp(ps.N_mat);
    std::vector<double> temp(4);
    int index_start;

    double time_val = ps.t_init;

    //double mu, double t_k, double Deltat, double x_j, double Deltax

    for (int tp=0; tp<ps.N_time; tp++){
        for (int ip=0; ip<ps.N_cells; ip++){
            for (int gp=0; gp<2; gp++){
            //for (int gp=0; gp<ps.N_groups; gp++){ //manual override for mms 
                for (int jp=0; jp<ps.N_angles; jp++){
                    

                    if ( gp == 0 ){
                        temp = AF_g1( ps.angles[jp], time_val, ps.dt, cells[ip].x, cells[ip].dx );
                    } else if ( gp == 1) {
                        temp = AF_g2( ps.angles[jp], time_val, ps.dt, cells[ip].x, cells[ip].dx );
                    } else{
                        std::cout<<"This MMS Verification is a 2 group problem only" <<std::endl;
                    }
                    index_start = (ip*(ps.SIZE_cellBlocks) + gp*(ps.SIZE_groupBlocks) + 4*jp);
                    mms_temp[index_start] = temp[0];
                    mms_temp[index_start+1] = temp[1];
                    mms_temp[index_start+2] = temp[2];
                    mms_temp[index_start+3] = temp[3];
                }
            }
        }


        
        string ext = ".csv";
        string file_name = "mms_sol";
        string dt = to_string(tp);

        file_name = file_name + dt + ext;

        std::ofstream output(file_name);
        output << "TIME STEP: " << tp << "Unsorted solution vector for mms" << endl;
        output << "N_space: " << ps.N_cells << " N_groups: " << ps.N_groups << " N_angles: " << ps.N_angles << endl;
        for (int i=0; i<mms_temp.size(); i++){
            output << mms_temp[i] << "," << endl;
        }

        time_val += ps.dt;

    
    }

    cout << "time integrated mms solutions published " << endl;
}


void MMSFullSource ( std::vector<cell> &cells, problem_space &ps ){

    std::vector<double> mms_temp(ps.N_mat);
    std::vector<double> temp(4);
    int index_start;

    double time_val = ps.t_init;

    //double mu, double t_k, double Deltat, double x_j, double Deltax

    for (int tp=0; tp<ps.N_time; tp++){
        for (int ip=0; ip<ps.N_cells; ip++){
            for (int gp=0; gp<2; gp++){
            //for (int gp=0; gp<ps.N_groups; gp++){ //manual override for mms 
                for (int jp=0; jp<ps.N_angles; jp++){
                    double Sigma_S1  = cells[ip].xsec_g2g_scatter[0];
                    double Sigma_S2  = cells[ip].xsec_g2g_scatter[3];
                    double Sigma_S12 = cells[ip].xsec_g2g_scatter[2];
                    double Sigma_S21 = cells[ip].xsec_g2g_scatter[1];

                    if ( gp == 0 ){
                        temp =  Q1(cells[ip].v[0], cells[ip].v[1], cells[ip].xsec_total[0], cells[ip].xsec_total[1], ps.angles[jp], Sigma_S1, Sigma_S2, Sigma_S12, Sigma_S21, cells[ip].x, cells[ip].dx, tp, ps.dt );
                    } else if ( gp == 1) {
                        temp =  Q2(cells[ip].v[0], cells[ip].v[1], cells[ip].xsec_total[0], cells[ip].xsec_total[1], ps.angles[jp], Sigma_S1, Sigma_S2, Sigma_S12, Sigma_S21, cells[ip].x, cells[ip].dx, tp, ps.dt );
                    } else{
                        std::cout<<"This MMS Verification is a 2 group problem only" <<std::endl;
                    }
                    index_start = (ip*(ps.SIZE_cellBlocks) + gp*(ps.SIZE_groupBlocks) + 4*jp);
                    mms_temp[index_start] = temp[0];
                    mms_temp[index_start+1] = temp[1];
                    mms_temp[index_start+2] = temp[2];
                    mms_temp[index_start+3] = temp[3];
                }
            }
        }


        
        string ext = ".csv";
        string file_name = "mms_source";
        string dt = to_string(tp);

        file_name = file_name + dt + ext;

        std::ofstream output(file_name);
        output << "TIME STEP: " << tp << "Unsorted solution vector for mms" << endl;
        output << "N_space: " << ps.N_cells << " N_groups: " << ps.N_groups << " N_angles: " << ps.N_angles << endl;
        for (int i=0; i<mms_temp.size(); i++){
            output << mms_temp[i] << "," << endl;
        }

        time_val += ps.dt;

    
    }

    cout << "time integrated mms solutions published " << endl;
}



void timeLoop(std::vector<double> af_previous, std::vector<cell> &cells, problem_space &ps){

    std::vector<double> af_solution( ps.N_mat );

    MMSInitialCond(af_solution, cells, ps);

    MMSFullSol( cells, ps );
    MMSFullSource( cells, ps );

    //check_g2g(cells, ps);

    //ps.time_val = 0;

    for (int t=0; t<ps.N_time; ++t){

        // set mms if needed
        setMMSsourece(cells, ps, t);

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

        ps.time_val += ps.dt;
    }
}





int main(){
    // testing function

    using namespace std;
    
    // problem definition
    // eventually from an input deck
    double dx = .1;
    double dt = 0.1;
    double t_init = 0.25;
    vector<double> v = {1, 1};
    vector<double> xsec_total = {1.5454, 0.04568};
    //vector<double> xsec_scatter = {0.61789, 0.072534};
    //vector<double> xsec_scatter = {0,0};
    //double ds = 0.0;
    vector<double> material_source = {0,0,0,0,0,0,0,0}; // isotropic, g1 time_edge g1 time_avg, g2 time_edge, g2 time_avg

    double Length = .4;
    double IC_homo = 0;
    
    int N_cells = 4; //10
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
    ps.time_val = t_init;
    ps.t_init = t_init;
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
    ps.max_iteration = int(4);
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
        
        //cellCon.xsec_scatter = vector<double> {xsec_scatter[0], xsec_scatter[1]};
        cellCon.xsec_total = vector<double> {xsec_total[0], xsec_total[1]};
        cellCon.dx = dx;
        cellCon.v = v;
        cellCon.dt = dt;
        cellCon.material_source = material_source;
        cellCon.xsec_g2g_scatter = vector<double> {0, 0, 0, 0};
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

