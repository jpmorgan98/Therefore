#include <iostream>
#include <vector>
#include "util.h" // remove when putting in larger file
#include "base_mats.h"
#include "legendre.h"

//compile commands 
// cc sweep.cpp -std=c++20 -L/usr/lib64 -llapack


extern "C" void dgesv_( int *n, int *nrhs, double  *a, int *lda, int *ipiv, double *b, int *lbd, int *info  );

void Axeb( std::vector<double> A, std::vector<double> b){

    int N = b.size(); //defined by the method
    int nrhs = 1;
    std::vector<int> ipiv(b.size());
    int info;
    dgesv_( &N, &nrhs, &A[0], &N, &ipiv[0], &b[0], &N, &info );

}


void sweep(std::vector<double> &af_last, std::vector<double> &af_prev, std::vector<double> sf_new, std::vector<cell> &cells, problem_space ps){
    for (int j=0; j<ps.N_angles; ++j){
        for (int g=0; g<ps.N_groups; ++g){
            if (ps.angles[j] < 0){ // negative sweep
                for (int i=0; i<ps.N_cells; ++i){

                    int helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);

                    // index corresponding to this position last time step
                    double af_hl_L = af_prev[helper + 2];
                    double af_hl_R = af_prev[helper + 3];
                    double af_R;
                    double af_hn_R;
                    if ( i == ps.N_cells - 1 ){
                        af_R    = 0; // BCr[angle]
                        af_hn_R = 0; // BCr[angle]
                    } else {
                        af_R    = af_last[((i+1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j)];
                        af_hn_R = af_last[((i+1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j)];
                    }

                    std::vector<double> A = A_neg_rm(cells[i], ps.angles[j], g);
                    std::vector<double> b = b_neg(cells[i], g, ps.angles[j], j, af_hl_L, af_hl_R, af_R, af_hn_R);

                    Axeb(A, b);

                    af_last[helper+0] = b[0];
                    af_last[helper+1] = b[1];
                    af_last[helper+2] = b[2];
                    af_last[helper+3] = b[3];

                }
            } else if (ps.angles[j] > 0) { // positive sweep
                for (int i=0; i<ps.N_cells; ++i){
                    int helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);

                    // index corresponding to this position last time step
                    double af_hl_L = af_prev[helper + 2];
                    double af_hl_R = af_prev[helper + 3];
                    double af_L;
                    double af_hn_L;

                    if (i == 0){
                        af_L     = 1;//ps.boundary_condition[];
                        af_hn_L  = 1;//ps.boundary_condition[];
                    } else {
                        af_L     = af_last[((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j)];
                        af_hn_L  = af_last[((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 2];
                    }

                    std::vector<double> A = A_pos_rm(cells[i], ps.angles[j], g);
                    std::vector<double> b = b_pos(cells[i], g, ps.angles[j], j, af_hl_L, af_hl_R, af_L, af_hn_L);

                    Axeb(A, b);

                    af_last[helper+0] = b[0];
                    af_last[helper+1] = b[1];
                    af_last[helper+2] = b[2];
                    af_last[helper+3] = b[3];
                }
            }
        }
    }
}

void quadrature(std::vector<double> &angles, std::vector<double> &weights){

    // infred from size of pre-allocated std::vector
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

void compute_scalar_flux(std::vector<double> sf, std::vector<double> af, std::vector<double> weights){
    /* assumed data shape
    */

    int N_ele = sf.size();

    int N_angles = weights.size();

    for (int i=0; i<N_ele; ++i){
        for (int m=0; m<N_angles; ++m){
            sf[i] *= weights[m]*af[i*4*m];
        }
    }


}

void convergenceLoop(std::vector<double> &af_new,  std::vector<double> &af_previous, std::vector<cell> &cells, problem_space &ps){

    bool converged = true;
    int itter = 0;
    double error = 1.0;
    double error_n1 = 1.0;
    double error_n2 = 1.0;
    double spec_rad = 0;
    std::vector<double> af_last(ps.N_mat);
    std::vector<double> sf_new;
    std::vector<double> sf_last;

    compute_scalar_flux( sf_new, af_previous, ps.weights );

    while (converged){
        
        sweep( af_new,  af_previous, sf_new, cells, ps);


        
        // compute the L2 norm between the last and current iteration
        error = infNorm_error( af_last, af_new );

        // compute spectral radius
        spec_rad = pow( pow(error+error_n1,2), .5) / pow(pow(error_n1+error_n2, 2), 0.5);

        // too allow for an error & spectral radius computation we need at least three cycles (indexing from zero)
        if (itter > 2){
            // if relative error between the last and just down iteration end the time step
            // including false solution protection!!!!
            if ( error < ps.convergence_tolerance *(1-spec_rad)){ converged = false; } }

        if (itter >= ps.max_iteration){
            cout << ">>>WARNING: Computation did not converge after " << ps.max_iteration << " iterations <<<" << endl;
            cout << "       itter: " << itter << endl;
            cout << "       error: " << error << endl;
            cout << "" << endl;
            converged = false;
        }
        
        cout << "af_last" << endl;
        print_vec_sd(af_last);
        cout << "af_new" << endl;
        print_vec_sd(af_new);
        af_last = af_new;
        
        

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
        /*
        std::cout << "" <<std::endl;
        std::cout << "af last" <<std::endl;
        print_vec_sd(af_last);
        std::cout << "" <<std::endl;
        std::cout << "af new" <<std::endl;
        print_vec_sd(af_new);*/

    } // end while loop
} // end convergence function

int main(){
    // testing function

    using namespace std;
    
    // problem definition
    // eventually from an input deck
    double dx = .5;
    double dt = 1.0;
    vector<double> v = {1, 4};
    vector<double> xsec_total = {1, 3.0};
    vector<double> xsec_scatter = {.99, .6};
    double ds = 0.0;
    vector<double> Q = {0, 0};

    double Length = 1;
    double IC_homo = 0;
    
    int N_cells = 2; //10
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
    ps.ds = ds;
    ps.N_angles = N_angles;
    ps.N_cells = N_cells;
    ps.N_groups = N_groups;
    ps.N_time = N_time;
    ps.N_rm = N_rm;
    ps.N_mat = N_mat;
    ps.angles = angles;
    ps.weights = weights;
    ps.initialize_from_previous = false;
    ps.max_iteration = int(3);
    // 0 for vac 1 for reflecting 3 for mms
    ps.boundary_conditions = {0,0};
    // size of the cell blocks in all groups and angle
    ps.SIZE_cellBlocks = ps.N_angles*ps.N_groups*4;
    ps.ELEM_cellBlocks = ps.SIZE_cellBlocks*ps.SIZE_cellBlocks;
    // size of the group blocks in all angle within a cell
    ps.SIZE_groupBlocks = ps.N_angles*4;
    // size of the angle blocks within a group and angle
    ps.SIZE_angleBlocks = 4;


    vector<cell> cells;

    int region_id = 0;

    for (int i=0; i<N_cells; i++){
        // /*building reeds problem from left to right

        cell cellCon;
        cellCon.cell_id = i;
        if (i ==0 )
            cellCon.x_left = 0;
        else
            cellCon.x_left = cells[cells.size()-1].x_left+cells[cells.size()-1].dx;
        
        cellCon.xsec_scatter = vector<double> {xsec_scatter[0], xsec_scatter[1]};
        cellCon.xsec_total = vector<double> {xsec_total[0], xsec_total[1]};
        cellCon.dx = dx;
        cellCon.v = v;
        cellCon.dt = dt;

        vector<double> temp (N_angles*N_groups*4, 1.0);
        for (int p=0; p<temp.size(); ++p){temp[p] = Q[0];}
        cellCon.Q = temp;
        cellCon.N_angle = N_angles;

        cells.push_back(cellCon);
    }

    
    std::vector<double> af_previous(N_mat, 0);
    std::vector<double> af_last(N_mat, 0);

    std::cout << "entering conv loop" <<std::endl;

    convergenceLoop( af_last,  af_previous, cells, ps);

    std::cout << "done" << std::endl;

    return(1);
}

