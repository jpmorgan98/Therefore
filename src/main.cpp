/*brief: assembly functions for matrices to compute therefore commands
date: May 23rd 2023
auth: J Piper Morgan (morgjack@oregonstate.edu)*/

#include <iostream>
#include <vector>

#include "util.h"
#include "builders.h"
//#include "H5Cpp.h"
//#include "lapacke.h"
//#include <Eigen/Dense>
//#include <cusparse_v2.h>
//#include <cuda.h>

/* compile notes and prerecs

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
            
*/

void eosPrint(ts_solutions state);

// row major!!!!!!
extern "C" void dgesv_( int *n, int *nrhs, double  *a, int *lda, int *ipiv, double *b, int *lbd, int *info  );
//extern "C" void LAPACKE_dgesv_( LAPACK_ROW_MAJOR, int *n, int *nrhs, double  *a, int *lda, int *ipiv, double *b, int *lbd, int *info  );
//-llapacke
std::vector<double> row2colSq(std::vector<double> row);

// i space, m is angle, k is time, g is energy group

const bool print_mats = false;
const bool debug_print = false;
const bool cycle_print = true;

int main(void){

    print_title();

    using namespace std;
    
    // problem definition
    // eventually from an input deck
    double dx = 0.05;
    double dt = 1.0;
    vector<double> v = {4};
    vector<double> xsec_total = {1, 0.5};
    vector<double> xsec_scatter = {0.25, 0.1};
    vector<double> Q = {1, 0};
    double Length = 1;
    double IC_homo = 0;
    
    int N_cells = 170; 
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
    ps.dt = dt;
    ps.dx = dx;
    ps.N_angles = N_angles;
    ps.N_cells = N_cells;
    ps.N_groups = N_groups;
    ps.N_time = N_time;
    ps.angles = angles;
    ps.weights = weights;
    ps.initialize_from_previous = false;
    ps.max_iteration = int(1e4);
    ps.boundary_conditions = {0,1};

    // size of the cell blocks in all groups and angle
    ps.SIZE_cellBlocks = ps.N_angles*ps.N_groups*4;
    // size of the group blocks in all angle within a cell
    ps.SIZE_groupBlocks = ps.N_angles*4;
    // size of the angle blocks within a group and angle
    ps.SIZE_angleBlocks = 4;


    // allocates a zero vector of nessacary size
    ps.initilize_boundary();

    // reeds problem mat stuff
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
        /*building reeds problem from left to right*/

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
        cellCon.Q = vector<double> {Source_reeds[region_id], Source_reeds[region_id]};
        cellCon.region_id = region_id;

        cells.push_back(cellCon);
    }

    if (debug_print){
        for (int k=0; k<N_cells; k++){
            cout << cells[k].cell_id << " " << cells[k].x_left  << "   " << cells[k].region_id << "   " << cells[k].dx << "   " << cells[k].Q[0] << "   " << cells[k].xsec_scatter[0] << "   " << cells[k].xsec_total[0] << endl;
        }

        return(0);
    }
    
    // initial condition stored as first element of solution vector
    vector<ts_solutions> solutions;

    ts_solutions sol_con;
    sol_con.aflux = IC;
    sol_con.time = 0.0;
    sol_con.spectral_radius = 0.0;
    sol_con.N_step = 0;
    sol_con.number_iteration = 0;

    solutions.push_back(sol_con);

    // allocation of the whole ass mat
    vector<double> A(N_rm);
    
    // vector org angular flux from last iteration
    vector<double> aflux_last(N_mat, 0.0);
    // vector org converged angular flux from previous time step
    vector<double> aflux_previous(N_mat, 0.0);
    // initializing the inital previous as the IC
    aflux_previous = IC;

    int nrhs = 1; // one column in b
    int lda = N_mat;
    int ldb = 1; // leading b dimention for row major
    int ldb_col = N_mat; // leading b dim for col major
    std::vector<int> i_piv(N_mat, 0);  // pivot column vector
    int info;

    // generation of the whole ass mat
    A_gen(A, cells, ps);
    vector<double> A_col = row2colSq(A);

    if (print_mats){
        print_rm(A);
    }

    vector<double> b(N_mat);

    // time step loop
    for(int t=0; t<N_time; ++t){

        
        
        if (ps.initialize_from_previous){
            // all the angular fluxes start from the previous converged time step
            aflux_last = aflux_last;
        } else {
            // all angular fluxes start this time step iteration from 0
            fill(aflux_last.begin(), aflux_last.end(), 0.0);
        }
        

        vector<double> b(N_mat, 0.0);
        
        int itter = 0;          // iteration counter
        double error = 1;       // error from current iteration
        double error_n1 = 1;    // error back one iteration (from last)
        double error_n2 = 1;    // error back two iterations
        bool converged = true;  // converged boolean
        double spec_rad;

        //vector<double> A_copy;
        vector<double> A_copy(N_mat);

        if (cycle_print){
            cout << ">>>CYCLE INFO FOR TIME STEP: " << t <<"<<<"<< endl;
            printf("cycle   error         error_n1      error_n2     spec_rad   cycle_time\n");
            printf("===================================================================================\n");
        }
        
        while (converged){

            // lapack requires a copy of data that it uses for row piviot (A after _dgesv != A)
            A_copy = A_col;

            b_gen(b, aflux_previous, aflux_last, cells, ps);
            // reminder: last refers to iteration, previous refers to time step
            
            if (print_mats){
                cout << "Cycle: " << itter << endl;
                cout << "RHS" << endl;
                print_vec_sd(b);
            }
            // solve Ax=b
            //info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, N_mat, nrhs, &A_copy[0], lda, &i_piv[0], &b[0], ldb );

            //info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, nrhs, a, lda, ipiv, b, ldb );
            //Lapack solver 
            dgesv_( &N_mat, &nrhs, &A_copy[0], &lda, &i_piv[0], &b[0], &ldb_col, &info );

            if (print_mats){
                cout << "x" <<endl;
                print_vec_sd(b);
            }

            if( info > 0 ) {
                printf( "The diagonal element of the triangular factor of A,\n" );
                printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
                printf( "the solution could not be computed.\n" );
                exit( 1 );
            }

            // compute the relative error between the last and current iteration
            error = infNorm_error(aflux_last, b);

            // compute spectral radius
            spec_rad = abs(error-error_n1) / abs(error_n1 - error_n2);

            // too allow for a error computation we need at least three cycles
            if (itter > 3){
                // if relative error between the last and just down iteration end the time step
                if ( error < ps.convergence_tolerance ){ converged = false; }
            }

            if (itter > ps.max_iteration){
                cout << ">>>WARNING: Computation did not converge after " << ps.max_iteration << "iterations<<<" << endl;
                cout << "       itter: " << itter << endl;
                cout << "       error: " << error << endl;
                cout << "" << endl;
                converged = false;
            }

            aflux_last = b;
            itter++;

            if (cycle_print){
                printf("%3d      %1.4e    %1.4e    %1.4e   %1.4e \n", itter, error, error_n1, error_n2, spec_rad );
            }

            error_n2 = error_n1;
            error_n1 = error;

        } // end convergence loop

        aflux_previous = b;

        string ext = ".csv";
        string file_name = "afluxUnsorted";
        string dt = to_string(t);

        file_name = file_name + dt + ext;

        std::ofstream output(file_name);
        output << "TIME STEP: " << t << "Unsorted solution vector" << endl;
        output << "N_space: " << ps.N_cells << " N_groups: " << ps.N_groups << " N_angles: " << ps.N_angles << endl;
        for (int i=0; i<b.size(); i++){
            output << b[i] << "," << endl;
        }

        std::ofstream dist("x.csv");
        dist << "x: " << endl;
        for (int i=0; i<cells.size(); i++){
            dist << cells[i].x_left << "," << endl;
            dist << cells[i].x_left + cells[i].dx/2 << "," <<endl;
        }


        cout << "file saved under: " << file_name << endl;

        // store solution vector org
        //ts_solutions save_timestep;
        //save_timestep.time = (t+1)*ps.dt;
        //save_timestep.spectral_radius = spec_rad;
        //save_timestep.N_step = t+1;
        //save_timestep.number_iteration = itter;

        // print end of step information
        //eosPrint(save_timestep);
        //print_vec_sd(b);
        //save_timestep.aflux = b;
        //print_vec_sd(save_timestep.aflux);
        //solutions.push_back(save_timestep);

    } // end of time step loop

    //WholeProblem wp = WholeProblem(cells, ps, solutions);
    //wp.PublishUnsorted();
    
    return(0);
} // end of main

/*
class run{
    
    

    public:
        problem_space ps;
        vector<cell> cells;




        // functions

        void run_whole_problem(){

        }
};*/


std::vector<double> row2colSq(std::vector<double> row){
    /*brief */
    
    int SIZE = sqrt(row.size());

    std::vector<double> col(SIZE*SIZE);

    for (int i = 0; i < SIZE; ++i){
        for (int j = 0; j < SIZE; ++j){
            outofbounds_check(i * SIZE + j, col);
            outofbounds_check(j * SIZE + i, row);


            col[ i * SIZE + j ] = row[ j * SIZE + i ];
        }
    }

    return(col);
}

void std2cuda_bsr(problem_space ps){
    int block_size = 4*ps.N_angles*ps.N_groups;
    int number_row_blocks = ps.N_cells;
    int number_col_blocks = ps.N_cells;
    int number_nonzero_blocks = ps.N_cells;

}
/*
void A_gen_c_g(){
    
    breif: assmebles a coeeficant matrix within a given group and cell for all angles
    NOTE: ROW MAJOR FORMAT
    
}
*/

void eosPrint(ts_solutions state){
    using namespace std;
    /* brief: end of time step printer*/

    if (state.N_step == 0){
        // print header
        printf("Time    Step Table\n");
        printf("step    time    error   spectral\n");
        printf("                        radius\n");
        printf("============================================================================\n");
    } 
    
    // print cycle information table
    printf("%3d     %2.3e    %1.4e     %5f\n", state.N_step, state.time, state.final_error, state.spectral_radius);
}