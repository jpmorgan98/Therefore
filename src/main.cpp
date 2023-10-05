/*brief: assembly functions for matrices to compute therefore commands
date: May 23rd 2023
auth: J Piper Morgan (morgajoa@oregonstate.edu)*/

#include <iostream>
#include <vector>

#include "util.h"
//#include "mms.h"
#include "builders.h"
//#include "cusolver_axb.cu"

//#include "H5Cpp.h"
//#include "lapacke.h"
//#include <Eigen/Dense>
//#include <cusparse_v2.h>
//#include <cuda.h>

/* compile notes and prerecs
For LAPACK (not dense)
    In UBUNTU 
        you need these libraries:
            sudo apt-get install libblas-dev checkinstall
            sudo apt-get install libblas-doc checkinstall
            sudo apt-get install liblapacke-dev checkinstall
            sudo apt-get install liblapack-doc checkinstall
        Should be able to configure with: 
            g++ main.cpp -std=c++20 -llapack

    In OSX
        you need:
            brew install gcc
            brew install lapack
        Should be able to compile with: 
            g++-13 main.cpp -std=c++20 -llapack
            
For Trilinos (see Tril.md for insturcations)

For CUDA GPU
    On LASSEN
        modules
            CUDA Toolkit 12.0
            cmake/20

        Uncomment CUDA lines
        to build
            mkdir build
            cd build
            cmake ..
            make
*/
using namespace std;

void eosPrint(ts_solutions state);

// row major to start -> column major for lapack computation
extern "C" void dgesv_( int *n, int *nrhs, double  *a, int *lda, int *ipiv, double *b, int *lbd, int *info  );


// i space, m is angle, k is time, g is energy group

const bool print_mats = false;
const bool debug_print = false;
const bool cycle_print = true;
const bool gpu = false;


class run{

    public:

        problem_space ps;
        vector<cell> cells;
        vector<double> IC;

        vector<double> aflux_last;
        vector<double> aflux_previous;



        int cycle_print_flag = 0; // for printing headers

        int itter;          // iteration counter
        double error;       // error from current iteration
        double error_n1;    // error back one iteration (from last)
        double error_n2;    // error back two iterations
        bool converged;  // converged boolean
        double spec_rad;

        double time = 0;

        // lapack variables!
        int nrhs; // one column in b
        int lda;
        int ldb; // leading b dimention for row major
        int ldb_col; // leading b dim for col major
        std::vector<int> i_piv;  // pivot column vector
        int info;

        // source for the method of manufactured solution
        mms manSource;

        void init_vectors(){
            // vector org angular flux from last iteration
            aflux_last.resize(ps.N_mat);
            // vector org converged angular flux from previous time step
            aflux_previous.resize(ps.N_mat);
            // initializing the inital previous as the IC
            aflux_previous = IC;
        }

        void init_af_timestep(){
            if (ps.initialize_from_previous){
                    // all the angular fluxes start from the previous converged time step
                    aflux_last = aflux_last;
                } else {
                    // all angular fluxes start this time step iteration from 0
                    fill(aflux_last.begin(), aflux_last.end(), 0.0);
                }
        }

        void cycle_print_func(int t){
            if (itter == 0)
                cycle_print_flag = 0;
            else 
                cycle_print_flag = 1;

            if (cycle_print){
                if (cycle_print_flag == 0) {
                    cout << ">>>CYCLE INFO FOR TIME STEP: " << t <<"<<<"<< endl;
                    printf("cycle   error         error_n1      error_n2     spec_rad   cycle_time\n");
                    printf("===================================================================================\n");
                    cycle_print_flag = 1;
                }
                printf("%3d      %1.4e    %1.4e    %1.4e   %1.4e \n", itter, error, error_n1, error_n2, spec_rad );
            }
        }

        void save_eos_data(int t){
                string ext = ".csv";
                string file_name = "afluxUnsorted";
                string dt = to_string(t);

                file_name = file_name + dt + ext;

                std::ofstream output(file_name);
                output << "TIME STEP: " << t << "Unsorted solution vector" << endl;
                output << "N_space: " << ps.N_cells << " N_groups: " << ps.N_groups << " N_angles: " << ps.N_angles << endl;
                for (int i=0; i<aflux_last.size(); i++){
                    output << aflux_last[i] << "," << endl;
                }

                std::ofstream dist("x.csv");
                dist << "x: " << endl;
                for (int i=0; i<cells.size(); i++){
                    dist << cells[i].x_left << "," << endl;
                    dist << cells[i].x_left + cells[i].dx/2 << "," <<endl;
                }


                cout << "file saved under: " << file_name << endl;
        }



        void sourceSource( ){
            vector<double> temp;
            for (int i=0; i<ps.N_cells; ++i){
                //for (int g=0; g<ps.N_groups; g++){ 
                    for (int j=0; j<ps.N_angles; ++j){
                        
                        // group 1
                        temp = manSource.group1source(cells[i].x, cells[i].dx, time,  ps.dt, ps.angles[j]);
                        cells[i].Q[8*j  ] = temp[0];
                        cells[i].Q[8*j+1] = temp[1];
                        cells[i].Q[8*j+2] = temp[2];
                        cells[i].Q[8*j+3] = temp[3];

                        // group 2
                        temp = manSource.group2source(cells[i].x, cells[i].dx, time,  ps.dt, ps.angles[j]);
                        cells[i].Q[4+8*j  ] = temp[0];
                        cells[i].Q[4+8*j+1] = temp[1];
                        cells[i].Q[4+8*j+2] = temp[2];
                        cells[i].Q[4+8*j+3] = temp[3];
                    }
                //}
            }
        }

        void cudaDense_linear_solver(vector<double> &A_copy, vector<double> &b){
            //cuSolver(A_copy, b);
        }

        void linear_solver(vector<double> &A_copy, vector<double> &b){
            if (itter == 0){
                // lapack variables (col major)!
                nrhs = 1; // one column in b
                lda = ps.N_mat;
                ldb = ps.N_mat; // leading b dimention for row major
                ldb_col = ps.N_mat; // leading b dim for col major
                i_piv.resize(ps.N_mat, 0);  // pivot column vector
            }

            // solve Ax=b
            //info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, N_mat, nrhs, &A_copy[0], lda, &i_piv[0], &b[0], ldb );
            dgesv_( &ps.N_mat, &nrhs, &A_copy[0], &lda, &i_piv[0], &b[0], &ldb_col, &info );

            if( info > 0 ) {
                printf( "\n>>>ERROR<<<\n" );
                printf( "The diagonal element of the triangular factor of A,\n" );
                printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
                printf( "the solution could not be computed.\n" );
                exit( 1 );
            }
        }


        void checkSpecRad (){
            if (itter > 9){
                if ( spec_rad > 1.0 ){
                    printf( "\n>>>WARNING<<<\n" );
                    printf( "An unfortunate spectral radius has been detected\n" );
                    printf( "ρ = %1.4e ", spec_rad );
                    printf( "the solution could not be computed\n\n" );
                    //exit( 1 );
                }
            }
        }


        void publish_mms (){

            std::vector<double> mms_temp(ps.N_mat);
            std::vector<double> temp(4);
            int index_start;

            for (int tp=0; tp<ps.N_time; tp++){
                for (int ip=0; ip<ps.N_cells; ip++){
                    //for (int gp=0; gp<ps.N_groups; gp++){ //manual override for mms 
                        for (int jp=0; jp<ps.N_angles; jp++){

                            temp = manSource.group1af(cells[ip].x, cells[ip].dx, ps.dt*tp, ps.dt, ps.angles[jp]);
                            index_start = (ip*(ps.SIZE_cellBlocks) + 0*(ps.SIZE_groupBlocks) + 4*jp);
                            mms_temp[index_start] = temp[0];
                            mms_temp[index_start+1] = temp[1];
                            mms_temp[index_start+2] = temp[2];
                            mms_temp[index_start+3] = temp[3];

                            temp = manSource.group2af(cells[ip].x, cells[ip].dx, ps.dt*tp, ps.dt, ps.angles[jp]);
                            index_start = (ip*(ps.SIZE_cellBlocks) + 1*(ps.SIZE_groupBlocks) + 4*jp);
                            mms_temp[index_start] = temp[0];
                            mms_temp[index_start+1] = temp[1];
                            mms_temp[index_start+2] = temp[2];
                            mms_temp[index_start+3] = temp[3];
                        }
                    //}
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

                
            }

        cout << "time integrated mms solutions published " << endl;
        }

    



        void run_timestep(){

            init_vectors();

            // allocation of the whole ass mat
            vector<double> A(ps.N_rm);

            // generation of the whole ass mat
            A_gen(A, cells, ps);
            vector<double> A_col = row2colSq(A);

            vector<double> b(ps.N_mat);

            // time step loop
            for(int t=0; t<ps.N_time; ++t){ //
                ps.time_val = t;
                time += ps.dt;
                init_af_timestep();

                if ( ps.mms_bool ){
                    sourceSource( );
                }


                vector<double> b(ps.N_mat, 0.0);
                
                // resets
                itter = 0;          // iteration counter
                error = 1;          // error from current iteration 
                error_n1 = 1;       // error back one iteration (from last)
                error_n2 = 1;       // error back two iterations
                converged = true;   // converged boolean

                vector<double> A_copy(ps.N_mat);
                
                while (converged){

                    // lapack requires a copy of data that it uses for row piviot (A after _dgesv != A)
                    A_copy = A_col;
                    ps.assign_boundary(aflux_last);
                    //print_vec_sd(ps.af_right_bound);

                    b_gen(b, aflux_previous, aflux_last, cells, ps);
                    
                    // reminder: last refers to iteration, previous refers to time step

                    //Lapack solver 
                    linear_solver(A_copy, b);
                    
                    // compute the L2 norm between the last and current iteration
                    error = L2Norm( aflux_last, b );

                    // compute spectral radius
                    // np.linalg.norm(scalar_flux_next - scalar_flux, ord=2) / np.linalg.norm((scalar_flux - scalar_flux_last), ord=2)
                    spec_rad = pow( pow(error+error_n1,2), .5) / pow(pow(error_n1+error_n2, 2), 0.5);
                    checkSpecRad( );

                    // too allow for an error & spectral radius computation we need at least three cycles (indexing from zero)
                    if (itter > 2){
                        // if relative error between the last and just down iteration end the time step
                        // including false solution protection!!!!
                        if ( error < ps.convergence_tolerance*(1-spec_rad) ){ converged = false; } }

                    if (itter >= ps.max_iteration){
                        cout << ">>>WARNING: Computation did not converge after " << ps.max_iteration << "iterations<<<" << endl;
                        cout << "       itter: " << itter << endl;
                        cout << "       error: " << error << endl;
                        cout << "" << endl;
                        converged = false;
                    }

                    aflux_last = b;
                    

                    cycle_print_func(t);
                    
                    itter++;

                    error_n2 = error_n1;
                    error_n1 = error;

                } // end convergence loop

                aflux_previous = b;

                save_eos_data(t);

            } // end of time step loop
        }
};

int main(void){

    print_title();

    using namespace std;
    
    // problem definition
    // eventually from an input deck
    double dx = .1;
    double dt = 1.0;
    vector<double> v = {4, 1};
    vector<double> xsec_total = {1, 0.5};
    vector<double> xsec_scatter = {.2, .1};
    double ds = 0.0;
    vector<double> Q = {1, 1};

    double Length = 1;
    double IC_homo = 0;
    
    int N_cells = 10; //10
    int N_angles = 2; 
    int N_time = 5;
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

    // problem space class construction
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
    ps.max_iteration = int(100);
    // 0 for vac 1 for reflecting 3 for mms
    ps.boundary_conditions = {0,0};
    // size of the cell blocks in all groups and angle
    ps.SIZE_cellBlocks = ps.N_angles*ps.N_groups*4;
    // size of the group blocks in all angle within a cell
    ps.SIZE_groupBlocks = ps.N_angles*4;
    // size of the angle blocks within a group and angle
    ps.SIZE_angleBlocks = 4;


    // allocates a zero vector of nessacary size
    ps.initilize_boundary();

    /*
    // =================== REEDS Problem

    // reeds problem mat stuff 
    
    // mono-energetic data
    vector<double> sigma_s_reeds = {.9,  .9,    0,    0,    0};
    vector<double> sigma_t_reeds = {1,    1,    0,    5,    50};
    vector<double> Source_reeds  = {0,    1,    0,    0,    50};
    vector<double> dx_reeds =      {0.25, 0.25, 0.25, 0.02, 0.02};
    vector<int> N_per_region =     {8,    8 ,   4,    50,   100};
    
    // cell construction;
    vector<cell> cells;

    int region_id = 0;
    int N_next = N_per_region[0]-1;

    for (int i=0; i<N_cells; i++){
        // /*building reeds problem from left to right

        if (i==N_next+1){
            region_id ++;
            N_next += N_per_region[region_id];
        }

        cell cellCon;
        cellCon.cell_id = i;
        if (i ==0 )
            cellCon.x_left = 0;
        else
            cellCon.x_left = cells[cells.size()-1].x_left+cells[cells.size()-1].dx;
        
        cellCon.xsec_scatter = vector<double> {sigma_s_reeds[region_id], sigma_s_reeds[region_id]};
        cellCon.xsec_total = vector<double> {sigma_t_reeds[region_id], sigma_t_reeds[region_id]};
        cellCon.dx = dx_reeds[region_id];
        cellCon.v = v;
        cellCon.dt = dt;
        cellCon.Q = vector<double> {Source_reeds[region_id], Source_reeds[region_id], Source_reeds[region_id], Source_reeds[region_id],
                                    0, 0, 0, 0};
        cellCon.region_id = region_id;
        cellCon.N_angle = N_angles;

        cells.push_back(cellCon);
    }
    */

   // ===================
   

    vector<cell> cells;

    for (int i=0; i<N_cells; i++){
        // /*building a single region problem

        cell cellCon;
        cellCon.cell_id = i;
        if (i ==0 )
            cellCon.x_left = 0;
        else
            cellCon.x_left = cells[cells.size()-1].x_left+cells[cells.size()-1].dx;
        cellCon.x = cellCon.x_left + dx/2;
        cellCon.xsec_scatter = xsec_scatter;
        cellCon.xsec_total = xsec_total;
        cellCon.dx = dx;
        cellCon.v = v;
        cellCon.dt = dt;
        vector<double> temp (N_angles*N_groups*4);
        cellCon.Q = temp;
        cellCon.region_id = 1;
        cellCon.N_angle = ps.N_angles;

        cells.push_back(cellCon);

    }

    ps.mms_bool = false;

    // ===================


    // initial condition stored as first element of solution vector
    vector<ts_solutions> solutions;

    ts_solutions sol_con;
    sol_con.aflux = IC;
    sol_con.time = 0.0;
    sol_con.spectral_radius = 0.0;
    sol_con.N_step = 0;
    sol_con.number_iteration = 0;

    solutions.push_back(sol_con);


    run problem;
    problem.ps = ps;
    problem.cells = cells;
    problem.IC = IC;
    //// mms coefficients
    //problem.manSource.A = 1.0;
    //problem.manSource.B = 1.0;
    //problem.manSource.C = 1.0;
    //problem.manSource.D = 1.0;
    //problem.manSource.F = 1.0;
    //problem.manSource.v1 = v[0];
    //problem.manSource.v2 = v[1];
    //problem.manSource.sigma1 = xsec_total[0];
    //problem.manSource.sigma2 = xsec_total[1];
    //problem.manSource.sigmaS1 = xsec_scatter[0];
    //problem.manSource.sigmaS2 = xsec_scatter[1];
    //problem.manSource.sigmaS1_2 = ps.ds;
//
    //ps.manSource = problem.manSource;

    
    problem.run_timestep();

    problem.publish_mms();
    
    return(0);
} // end of main