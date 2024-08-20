#include <iostream>
#include <vector>
#include "util.h"
#include "base_mats.h"
#include "legendre.h"
#include <hip/hip_runtime_api.h>
#include "rocsolver.cpp"

const bool OPTIMIZED = true;

//compile commands 
// lockhartt cc sweep.cpp -std=c++20
// g++ -g -L -llapack
// compiling on ROCM device hipcc -g -w -I/opt/rocm/include -L/opt/rocm/lib -L/usr/lib64 -lrocsolver -lrocblas -llapack sweep_gpu.cpp -o xSweep.out


// for profiling:
    // compile with hipcc -g -w -I/opt/rocm/include -L/opt/rocm/lib -L/usr/lib64 -lrocsolver -lrocblas -llapack sweep_gpu.cpp -o xSweep.out
    // run with rocprof --hip-trace ./xSweep.out
    // view on https://ui.perfetto.dev

const bool cycle_print = true;
const bool save_output = false;

extern "C" void dgesv_( int *n, int *nrhs, double  *a, int *lda, int *ipiv, double *b, int *lbd, int *info  );

void Axeb( std::vector<double> &A, std::vector<double> &b){

    int N = b.size(); //defined by the method
    int nrhs = 1;
    std::vector<int> ipiv(b.size());
    int info;
    dgesv_( &N, &nrhs, &A[0], &N, &ipiv[0], &b[0], &N, &info );

}


void sb_Axeb(double *A_sp_cm, vector<double> &b, int N_groups, int N_angles){
    /*breif: Solves individual dense cell matrices in parallel on cpu
    requires -fopenmp to compile.

    A and b store all matrices in col major in a single std:vector as
    offsets from one another*/
    int info;

    int N_elements_cell = 16 * N_groups * N_angles;

    // parallelized over the number of cells
    //#pragma omp parallel for
    for (int i=0; i<N_groups*N_angles; ++i){

        // lapack variables for a single cell (col major!)
        int nrhs = 1; // one column in b
        int lda = 4; // leading A dim for col major
        int ldb_col = 4; // leading b dim for col major
        std::vector<int> ipiv_par(4); // pivot vector
        int Npbj = 4; // size of problem

        

        // solve Ax=b in a cell
        dgesv_( &Npbj, &nrhs, &A_sp_cm[i*16], &lda, &ipiv_par[0], &b[i*4], &ldb_col, &info );

        if( info > 0 ) {
            printf( "\n>>>PBJ LINALG ERROR<<<\n" );
            printf( "The diagonal element of the triangular factor of A,\n" );
            printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
            printf( "the solution could not be computed.\n" );
            exit( 1 );
        }
    }
}
    
    
    


void gpu_sb_nopt_Axeb(double *A_sp_cm, vector<double> &b, int N_groups, int N_angles){
    // solves a strided batched problem but assumes that nothing is prior allocated

    rocblas_int N = 4;           // rows and cols in each household problem
    rocblas_int lda = 4;         // leading dimension of A in each household problem
    rocblas_int ldb = 4;         // leading dimension of B in each household problem
    rocblas_int nrhs = 1;                         // number of nrhs in each household problem
    rocblas_stride strideA = 16;  // stride from start of one matrix to the next (household to the next)
    rocblas_stride strideB = 4;  // stride from start of one rhs to the next
    rocblas_stride strideP = 4;  // stride from start of one pivot to the next
    rocblas_int batch_count = N_angles * N_groups;         // number of matricies (in this case number of cells)

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // defininig pointers to memory on GPU
    double *dA, *db;
    rocblas_int *ipiv, *dinfo;

    // double alloaction of problem
    hipMalloc(&dA, sizeof(double)*strideA*batch_count);         // allocates memory for strided matrix container
    hipMalloc(&db, sizeof(double)*strideB*batch_count);         // allocates memory for strided rhs container

    // integer allocation
    hipMalloc(&ipiv, sizeof(rocblas_int)*strideB*batch_count);  // allocates memory for integer pivot vector in GPU
    hipMalloc(&dinfo, sizeof(rocblas_int)*batch_count);

    hipMemcpy(dA, A_sp_cm, sizeof(double)*strideA*batch_count, hipMemcpyHostToDevice);
    hipMemcpy(db, &b[0], sizeof(double)*strideB*batch_count, hipMemcpyHostToDevice);

    rocsolver_dgesv_strided_batched(handle, N, nrhs, dA, lda, strideA, ipiv, strideP, db, ldb, strideB, dinfo, batch_count);
    //hipDeviceSynchronize();

    hipMemcpy(&b[0], db, sizeof(double)*strideB*batch_count, hipMemcpyDeviceToHost);

    hipFree(ipiv);
    hipFree(dinfo);
    hipFree(dA);
    hipFree(db);
    rocblas_destroy_handle(handle);
    //solve on gpu
}

void gpu_sb_Axeb(double *dA, double *db, int itter, int *ipiv, int *dinfo, int N_groups, int N_angles, rocblas_handle handle){

    rocblas_int N = 4;           // rows and cols in each household problem
    rocblas_int lda = 4;         // leading dimension of A in each household problem
    rocblas_int ldb = 4;         // leading dimension of B in each household problem
    rocblas_int nrhs = 1;                         // number of nrhs in each household problem
    rocblas_stride strideA = 16;  // stride from start of one matrix to the next (household to the next)
    rocblas_stride strideB = 4;  // stride from start of one rhs to the next
    rocblas_stride strideP = 4;  // stride from start of one pivot to the next
    rocblas_int batch_count = N_angles * N_groups;         // number of matricies (in this case number of cells)

    rocsolver_dgesv_strided_batched(handle, N, nrhs, dA, lda, strideA, ipiv, strideP, db, ldb, strideB, dinfo, batch_count);
}


void sweep_normal(std::vector<double> &af_last, std::vector<double> &af_prev, std::vector<double> &sf, std::vector<cell> &cells, problem_space ps){
    for (int j=0; j<ps.N_angles; ++j){
        for (int g=0; g<ps.N_groups; ++g){

            if (ps.angles[j] < 0){ // negative sweep
                for (int i=ps.N_cells; i-- > 0;){ // looping backwards with an unsigned counter

                    // where we are in AF vectors
                    int helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);
                    
                    // SF is stored diffrently then AF
                    int index_sf = (i*4) + (g*4*ps.N_cells);//(i*4*ps.N_groups) + (g*4);

                    // index corresponding to this position last time step
                    outofbounds_check( helper + 2, af_prev );
                    outofbounds_check( helper + 3, af_prev );

                    double af_hl_L = af_prev[helper + 2];
                    double af_hl_R = af_prev[helper + 3];
                    double af_RB;
                    double af_hn_RB;

                    if ( i == ps.N_cells - 1 ){

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

                    std::vector<double> c = c_neg(cells[i], g, ps.angles[j], j, sf, index_sf, af_hl_L, af_hl_R, af_RB, af_hn_RB);

                    Axeb(A, c);

                    af_last[helper+0] = c[0];
                    af_last[helper+1] = c[1];
                    af_last[helper+2] = c[2];
                    af_last[helper+3] = c[3];

                }
            } else if (ps.angles[j] > 0) { // positive sweep
                for (int i=0; i<ps.N_cells; ++i){

                    // location of where we are on the RHS and AF vectors
                    int helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);

                    // location in the sf vector of where we are
                    int index_sf = (i*4) + (g*4*ps.N_cells); 

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

                    Axeb(A, c);

                    af_last[helper+0] = c[0];
                    af_last[helper+1] = c[1];
                    af_last[helper+2] = c[2];
                    af_last[helper+3] = c[3];
                }
            }
        }
    }
}

void build_A_si(int i, int i_n, double *A_cell, std::vector<cell> &cells, problem_space &ps){
    // outputs a set of N_angle*N_group strided batched 4*4 matrices for a given cell

    for ( int g=0; g<ps.N_groups; ++g){
        for (int j=0; j<ps.N_angles; ++j){
            std::vector<double> A_c_g_a;
            if (ps.angles[j] > 0){
                std::vector<double> temp = A_pos_rm(cells[i], ps.angles[j], g);
                 A_c_g_a = row2colSq(temp);
            } else { // negative sweep
                std::vector<double> temp = A_neg_rm(cells[i_n], ps.angles[j], g);
                A_c_g_a = row2colSq(temp);
            }

            int index_start = 4*4*ps.N_angles*g + 16*j;

            for (int r=0; r<16; ++r){
                A_cell[index_start+r] = A_c_g_a[r];
            }
        }
    }
}


void build_A_fullproblem(std::vector<double> &A, std::vector<cell> &cells, problem_space &ps){
    //builds A for sweeps in every angle, cell and group

    int N_elements_cell = 16 * ps.N_groups * ps.N_angles;
    for (int i=0; i<ps.N_cells; ++i){
        int i_n = ps.N_cells-1 - i; // index for the negative sweeps
        build_A_si(i, i_n, &A[N_elements_cell*i], cells, ps);
    }
}


void build_b_si(int i, int i_n, std::vector<double> &b, std::vector<double> &af_last, std::vector<double> &af_prev, std::vector<double> &sf, std::vector<cell> &cells, problem_space &ps){
    for ( int g=0; g<ps.N_groups; ++g ){
        for ( int j=0; j<ps.N_angles; ++j ){

            if (ps.angles[j] < 0){ // negative sweep

                // location of where we are on the RHS and AF vectors
                int helper =  (i_n*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);

                // where we are in the scalar flux vector (tied to cell location)
                int index_sf = (i_n*4) + (g*4*ps.N_cells);

                // local helper for the RHS b vector
                int local_helper = (g*4*ps.N_angles) + 4*j;

                double af_hl_L = af_prev[helper + 2];
                double af_hl_R = af_prev[helper + 3];
                double af_RB;
                double af_hn_RB;

                if ( i_n == ps.N_cells - 1 ){

                    af_RB    = 0; // BCr[angle]
                    af_hn_RB = 0; // BCr[angle]

                } else {
                    outofbounds_check(((i_n+1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 0, af_last);
                    outofbounds_check(((i_n+1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 2, af_last);

                    af_RB    = af_last[((i_n+1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 0];
                    af_hn_RB = af_last[((i_n+1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 2];
                }

                std::vector<double> c = c_neg(cells[i_n], g, ps.angles[j], j, sf, index_sf, af_hl_L, af_hl_R, af_RB, af_hn_RB);

                b[local_helper+0] = c[0];
                b[local_helper+1] = c[1];
                b[local_helper+2] = c[2];
                b[local_helper+3] = c[3];

            } else if (ps.angles[j] > 0) { // positive sweep

                // location of where we are on AF vectors
                int helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);

                // where we are in the scalar flux vector (tied to cell location)
                int index_sf = (i*4) + (g*4*ps.N_cells);

                // local helper for the RHS b vector
                int local_helper = (g*4*ps.N_angles) + 4*j;

                double af_hl_L = af_prev[helper + 2];
                double af_hl_R = af_prev[helper + 3];
                double af_LB;
                double af_hn_LB;

                if ( i == 0 ){

                    af_LB    = 0; // BCr[angle]
                    af_hn_LB = 0; // BCr[angle]

                } else {
                    outofbounds_check( ((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 1, af_last );
                    outofbounds_check( ((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 3, af_last );

                    af_LB     = af_last[((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 1];
                    af_hn_LB  = af_last[((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 3];
                }

                std::vector<double> c = c_pos(cells[i], g, ps.angles[j], j, sf, index_sf, af_hl_L, af_hl_R, af_LB, af_hn_LB);

                b[local_helper+0] = c[0];
                b[local_helper+1] = c[1];
                b[local_helper+2] = c[2];
                b[local_helper+3] = c[3];
            }
        }
    }
}


void build_b_const_win_sweep(std::vector<double> &b, std::vector<double> &af_last, std::vector<double> &af_prev, std::vector<double> &sf, std::vector<cell> &cells, problem_space &ps){
    for ( int i=0; i<ps.N_cells; ++i){
        // index for the negative sweeps
        int i_n = ps.N_cells-1 - i; 
        for ( int g=0; g<ps.N_groups; ++g ){
            for ( int j=0; j<ps.N_angles; ++j ){
                int helper_b = (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);
                if (ps.angles[j] < 0){ // negative sweep

                    // location of where we are on the RHS and AF vectors
                    int helper =  (i_n*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);
                    
                    // where we are in the scalar flux vector (tied to cell location)
                    int index_sf = (i_n*4) + (g*4*ps.N_cells);

                    double af_hl_L = af_prev[helper + 2];
                    double af_hl_R = af_prev[helper + 3];

                    std::vector<double> c = c_neg_const_win_sweep(cells[i_n], g, ps.angles[j], j, sf, index_sf, af_hl_L, af_hl_R);

                    b[helper_b+0] = c[0];
                    b[helper_b+1] = c[1];
                    b[helper_b+2] = c[2];
                    b[helper_b+3] = c[3];

                } else if (ps.angles[j] > 0) { // positive sweep

                    // location of where we are on AF vectors
                    int helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);

                    // where we are in the scalar flux vector (tied to cell location)
                    int index_sf = (i*4) + (g*4*ps.N_cells);

                    double af_hl_L = af_prev[helper + 2];
                    double af_hl_R = af_prev[helper + 3];

                    std::vector<double> c = c_pos_const_win_sweep(cells[i], g, ps.angles[j], j, sf, index_sf, af_hl_L, af_hl_R);

                    b[helper_b+0] = c[0];
                    b[helper_b+1] = c[1];
                    b[helper_b+2] = c[2];
                    b[helper_b+3] = c[3];
                }
            }
        }
    }
}


void cpu_b_var_win_sweep(int i, int i_n, std::vector<double> &b, problem_space &ps){
    for ( int g=0; g<ps.N_groups; ++g ){
        for ( int j=0; j<ps.N_angles; ++j ){

            if (ps.angles[j] < 0){ // negative sweep

                // location of where we are on the RHS and AF vectors
                int helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);

                double af_RB;
                double af_hn_RB;

                if ( i_n == ps.N_cells - 1 ){
                    af_RB    = 0; // BCr[angle]
                    af_hn_RB = 0; // BCr[angle]
                } else {
                    outofbounds_check(((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 0, b);
                    outofbounds_check(((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 2, b);

                    af_RB    = b[((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 0];
                    af_hn_RB = b[((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 2];
                }

                b[helper+1] -= af_RB*ps.angles[j];
                b[helper+3] -= af_hn_RB*ps.angles[j];

            } else if (ps.angles[j] > 0) { // positive sweep

                // location of where we are on AF vectors
                int helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);

                double af_LB;
                double af_hn_LB;

                if ( i == 0 ){

                    af_LB    = 0; // BCr[angle]
                    af_hn_LB = 0; // BCr[angle]

                } else {
                    outofbounds_check( ((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 1, b );
                    outofbounds_check( ((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 3, b );

                    af_LB     = b[((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 1];
                    af_hn_LB  = b[((i-1)*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j) + 3];
                }

                b[helper+0] += af_LB*ps.angles[j];
                b[helper+2] += af_hn_LB*ps.angles[j];
            }
        }
    }
}

__global__ void gpu_b_var_win_sweep(int i, int i_n, double *b, double *angles, int N_cells, int N_angles, int N_groups){
    int j = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    //__syncthreads();

    int SIZE_cellBlocks = N_angles*N_groups*4;
    int SIZE_groupBlocks = N_angles*4;
    //int MAX_SIZE = 4*N_angles*N_groups*N_cells;

    if (j<N_angles){
        //printf("size cell blocks %3d, size group blocks %3d, j %3d, i %3d, i_n %3d\n", SIZE_cellBlocks, SIZE_groupBlocks, j, i, i_n);

        for  (int g=0; g<N_groups; ++g){
            if (angles[j] < 0){ // negative sweep

                // location of where we are on the RHS and AF vectors
                int helper =  (i*(SIZE_cellBlocks) + g*(SIZE_groupBlocks) + 4*j);

                double af_RB;
                double af_hn_RB;

                if ( i_n == N_cells - 1 ){
                    af_RB    = 0; // BCr[angle]
                    af_hn_RB = 0; // BCr[angle]
                } else {

                    af_RB    = b[((i-1)*(SIZE_cellBlocks) + g*(SIZE_groupBlocks) + 4*j) + 0];
                    //printf("b in GPU: %1.4e, index %3d, raw b val 1.4%e\n", af_RB, ((i_n+1)*(SIZE_cellBlocks) + g*(SIZE_groupBlocks) + 4*j) + 0, b[((i_n+1)*(SIZE_cellBlocks) + g*(SIZE_groupBlocks) + 4*j) + 0]);
                    //std::cout << "in gpu" << af_RB <<std::endl;
                    af_hn_RB = b[((i-1)*(SIZE_cellBlocks) + g*(SIZE_groupBlocks) + 4*j) + 2];
                }

                b[helper+1] -= af_RB*angles[j];
                b[helper+3] -= af_hn_RB*angles[j];

            } else if (angles[j] > 0) { // positive sweep

                // location of where we are on AF vectors
                int helper =  (i*(SIZE_cellBlocks) + g*(SIZE_groupBlocks) + 4*j);

                // where we are in the scalar flux vector (tied to cell location)
                int index_sf = (i*4) + (g*4*N_cells);

                // local helper for the RHS b vector
                int local_helper = (g*4*N_angles) + 4*j;

                double af_LB;
                double af_hn_LB;

                if ( i == 0 ){

                    af_LB    = 0; // BCr[angle]
                    af_hn_LB = 0; // BCr[angle]

                } else {

                    af_LB     = b[((i-1)*(SIZE_cellBlocks) + g*(SIZE_groupBlocks) + 4*j) + 1];
                    af_hn_LB  = b[((i-1)*(SIZE_cellBlocks) + g*(SIZE_groupBlocks) + 4*j) + 3];
                }

                b[helper+0] += af_LB*angles[j];
                b[helper+2] += af_hn_LB*angles[j];
            }
        }
    }
}



void resort (int i, int i_n, std::vector<double> &af_last, double *b, std::vector<cell> &cells, problem_space &ps){
    // moving 

    for ( int g=0; g<ps.N_groups; ++g ){
        for ( int m=0; m<ps.N_angles; ++m ){

            int local_helper = (g*4*ps.N_angles) + 4*m;
            int helper;

            if (ps.angles[m] < 0){ // negative sweep values
                helper =  (i_n*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*m);
            } else if (ps.angles[m] > 0) { // positive sweep values
                helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*m);
            }

            af_last[helper + 0] = b[local_helper + 0];
            af_last[helper + 1] = b[local_helper + 1];
            af_last[helper + 2] = b[local_helper + 2];
            af_last[helper + 3] = b[local_helper + 3];
        }
    }
}


void resort_wp (std::vector<double> &b, std::vector<double> &af_last, problem_space &ps){
    for ( int i=0; i<ps.N_cells; ++i){
        int i_n = ps.N_cells-1 - i; // index for the negative sweeps
        for ( int g=0; g<ps.N_groups; ++g ){
            for ( int m=0; m<ps.N_angles; ++m ){

                int helper_b = (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*m);
                int helper;

                if (ps.angles[m] < 0){ // negative sweep values
                    helper =  (i_n*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*m);
                } else if (ps.angles[m] > 0) { // positive sweep values
                    helper =  (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*m);
                }

                af_last[helper + 0] = b[helper_b + 0];
                af_last[helper + 1] = b[helper_b + 1];
                af_last[helper + 2] = b[helper_b + 2];
                af_last[helper + 3] = b[helper_b + 3];
            }
        }
    }
}

void sweep_batched(std::vector<double> &af_last, std::vector<double> &af_prev, std::vector<double> &sf, std::vector<cell> &cells, problem_space &ps){
    // sweep the problem batching all angles and groups within a cell
    // can be executed on a GPU or CPU

    std::vector<double> A (16 * ps.N_groups * ps.N_angles * ps.N_cells);

    int N_elements_cell = 16 * ps.N_groups * ps.N_angles;

    std::vector<double> b_cell (4 * ps.N_groups * ps.N_angles);

    build_A_fullproblem(A, cells, ps);

    for (int i=0; i<ps.N_cells; ++i){
        int i_n = ps.N_cells-1 - i; // index for the negative sweeps
        
        //build b for Cell all group and angle within cell
        build_b_si( i, i_n, b_cell, af_last, af_prev, sf, cells, ps );

        //solve for x in Ax=b
        gpu_sb_nopt_Axeb( &A[N_elements_cell*i], b_cell, ps.N_groups, ps.N_angles );

        //resort
        resort( i, i_n, af_last, &b_cell[0], cells, ps );
    }
}


void sweep_batched_gpu(rocblas_handle handle, int itter, double* dA, double *db, std::vector<double> &af_last, std::vector<double> &af_prev, std::vector<double> &sf, std::vector<cell> &cells, problem_space &ps){
    // do a transport sweep such that the whole thing happens on the GPU

    rocblas_int N = 4;            // rows and cols in each household problem
    rocblas_int lda = 4;          // leading dimension of A in each household problem
    rocblas_int ldb = 4;          // leading dimension of B in each household problem
    rocblas_int nrhs = 1;         // number of nrhs in each household problem
    rocblas_stride strideA = 16;  // stride from start of one matrix to the next (household to the next)
    rocblas_stride strideB = 4;   // stride from start of one rhs to the next
    rocblas_stride strideP = 4;   // stride from start of one pivot to the next
    rocblas_int batch_count = ps.N_angles * ps.N_groups;         // number of matricies (in this case number of cells)

    int N_elements_cell = 16 * ps.N_groups * ps.N_angles;
    int N_mats = ps.N_angles * ps.N_groups;

    double *dangles;
    hipMalloc( &dangles, sizeof(double)*4*N_mats*ps.N_cells);
    hipMemcpy( dangles, &ps.angles[0], sizeof(double)*ps.N_angles, hipMemcpyHostToDevice );

    rocblas_int *ipiv, *dinfo;
    hipMalloc(&ipiv, sizeof(rocblas_int)*4*N_mats);  // allocates memory for integer pivot vector in GPU
    hipMalloc(&dinfo, sizeof(rocblas_int)*N_mats);   // allocates memory for info (if anything goes wrong)

    int threadsperblock = 32;
    int blockspergrid = (ps.N_angles + (threadsperblock - 1)) / threadsperblock;

    // ***WARNING***:
    // b is ogrinised backwards for negative indicies
    // such that each traunch of work is EVERYTHING that's ready to compute
    std::vector<double> b (4 * ps.N_groups * ps.N_angles * ps.N_cells);
    build_b_const_win_sweep(b, af_last, af_prev, sf, cells, ps);
    hipMemcpy( db, &b[0], sizeof(double)*4*N_mats*ps.N_cells, hipMemcpyHostToDevice );

    // everything on a GPU!
    for (int i=0; i<ps.N_cells; ++i){

        int i_n = ps.N_cells-1 - i; 

        // build b for Cell all group and angle within cell
        hipLaunchKernelGGL(gpu_b_var_win_sweep, dim3(blockspergrid), dim3(threadsperblock), 0, 0, 
                                    i, i_n, db, dangles, ps.N_cells, ps.N_angles, ps.N_groups );
        hipDeviceSynchronize();

        if ( OPTIMIZED ){
            if (itter == 0){
                rocsolver_dgesv_strided_batched(handle, N, nrhs, &dA[N_elements_cell*i], lda, strideA, ipiv, strideP, &db[4*N_mats*i], ldb, strideB, dinfo, batch_count);
            } else {
                rocsolver_dgetrs_strided_batched(handle, rocblas_operation_none, N, nrhs, &dA[N_elements_cell*i], lda, strideA, ipiv, strideP, &db[4*N_mats*i], ldb, strideB, batch_count);
            } 
        } else {
            rocsolver_dgesv_strided_batched(handle, N, nrhs, &dA[N_elements_cell*i], lda, strideA, ipiv, strideP, &db[4*N_mats*i], ldb, strideB, dinfo, batch_count);
        }
        hipDeviceSynchronize();
    }

    hipMemcpy( &b[0], db, sizeof(double)*4*N_mats*ps.N_cells, hipMemcpyDeviceToHost );

    resort_wp(b, af_last, ps);

    hipFree(ipiv);
    hipFree(dinfo);
    hipFree(dangles);
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

    // zeroing out the SF which will be accumulated
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

    std::vector<double> sf_new(ps.ELEM_sf);
    std::vector<double> sf_last(ps.ELEM_sf);

    computeSF( af_previous, sf_new, ps );

    //build_A
    std::vector<double> A (16 * ps.N_groups * ps.N_angles * ps.N_cells);
    
    build_A_fullproblem(A, cells, ps);
    double *dA, *db;
    hipMalloc(&dA, sizeof(double)*A.size());         // allocates memory for strided matrix container
    int sizeb = 4*ps.N_angles*ps.N_groups*ps.N_cells;
    hipMalloc(&db, sizeof(double)*sizeb);         // allocates memory for strided rhs container
    hipMemcpy(dA, &A[0], sizeof(double)*A.size(), hipMemcpyHostToDevice);
    

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    Timer timer;

    while (converged){

        Timer timer2;

        // communicate energy!
        compute_g2g( cells, sf_new, ps );

        
        // A is in/out for LAPACK functions
        // If we are not going to use the out in an optmiized fassion we need to communicate
        // A every itteration. If it is optimized we only solve it once then do back sub
        if ( !OPTIMIZED ){
            hipMemcpy(dA, &A[0], sizeof(double)*A.size(), hipMemcpyHostToDevice);
        }

        //af_2 = af_new;
        //sweep_normal( af_new, af_previous, sf_new, cells, ps );
        sweep_batched_gpu( handle, itter, dA, db, af_new, af_previous, sf_new, cells, ps );
        //sweep_batched( af_2, af_previous, sf_new, cells, ps );
        //check_close(af_new, af_2);

        // compute scalar fluxes
        computeSF( af_new, sf_new, ps );

        // compute the inf norm between the last and current iteration
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

        sf_last = sf_new;

        

        if (cycle_print){
            // CYCLE PRINTING
            int cycle_print_flag = 0; // for printing headers

            if (itter != 0) 
                cycle_print_flag = 1;
            
            int t = 0;

            if (cycle_print_flag == 0) {
                cout << ">>>SI CYCLE INFO FOR TIME STEP: " << t <<" for dt: " << ps.dt << "<<<"<< endl;
                printf("cycle   error         error_n1      error_n2     spec_rad     cycle_time\n");
                printf("===================================================================================\n");
                cycle_print_flag = 1;
            }
            printf("%3d      %1.4e    %1.4e    %1.4e   %1.4e   %1.4e \n", itter, error, error_n1, error_n2, spec_rad, timer2.elapsed() );
        }
        
        error_n2 = error_n1;
        error_n1 = error;
        itter++;

    } // end while convergence loop

    //destorying A allocations
    hipFree(dA);
    hipFree(db);
    rocblas_destroy_handle( handle );

    // timer functions
    ps.time_conv_loop = timer.elapsed();
    ps.av_time_per_itter = timer.elapsed()/itter-1;


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

        if (save_output){

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
        }

        // new previous info
        af_previous = af_solution;
    }
}



//int main(){
extern "C"{ double ThereforeSweep ( double dx, double dt, int N_angles ) {
    // testing function

    using namespace std;
    
    // problem definition
    // eventually from an input deck
    //double dx = .05;
    //int N_angles = 16;

    //double dt = 0.1;
    vector<double> v = {1, .5};
    vector<double> xsec_total = {1.5454, 0.45468};
    vector<double> xsec_scatter = {0.61789, 0.072534};
    //vector<double> xsec_scatter = {0,0};
    //double ds = 0.0;
    vector<double> material_source = {1, 1, 1, 1}; // isotropic, g1 time_edge g1 time_avg, g2 time_edge, g2 time_avg

    double Length = 100;
    double IC_homo = 0;
    
    int N_cells = int(Length/dx); //int N_cells = 100; //10
    
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
    ps.max_iteration = int(1e4);
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

    //material_source

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
        cellCon.xsec_g2g_scatter = vector<double> {0, .38211, .92724, 0};
        //cellCon.xsec_g2g_scatter = vector<double> {0, .92747,.38211,  0};
        //cellCon.xsec_g2g_scatter = vector<double> {0, 1, 0, 0};

        //vector<double> temp (N_angles*N_groups*4, 1.0);
        //for (int p=0; p<temp.size(); ++p){temp[p] = Q[0];}
        //cellCon.Q = temp;
        //cellCon.N_angle = N_angles;

        cells.push_back(cellCon);
        
    }

    init_Q(cells, ps);

    std::vector<double> af_previous(N_mat, 0);

    timeLoop(af_previous, cells, ps);

    return(ps.time_conv_loop);
} // end of main
} // end of extern function