#include "util.h"
#include "builders.h"
#include <hip/hip_runtime.h>
#include "rocsolver.cpp"

// row major to start -> column major for lapack computation
extern "C" void dgesv_( int *n, int *nrhs, double  *a, int *lda, int *ipiv, double *b, int *lbd, int *info  );
//double gpuL2norm(rocblas_handle handle, double *v1, double *v2, int n);
//__global__ gpu_b_gen_var_win_iter(double *b, double *aflux_last, double *angles, double *boundary, int *ps );


class run{

    public:

        problem_space ps;
        vector<cell> cells;
        vector<double> IC;

        vector<double> aflux_last;
        vector<double> aflux_previous;


        bool cycle_print = true;
        

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
                    aflux_last = aflux_previous;
                } else {
                    // all angular fluxes start this time step iteration from 0
                    fill(aflux_last.begin(), aflux_last.end(), 0.0);
                }
        }

        void cycle_print_func(int t){
            int cycle_print_flag = 0; // for printing headers

            if (itter != 0) 
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



        void rocDense_linearSolver(vector<double> &A_copy, vector<double> &b){
            //amdGPU_dgesv(A_copy, b);
        }



        void linear_solver(vector<double> &A_copy, vector<double> &b){
            /* DO NOT CALL DEBUGING ONLY
            Solves a single large dense matrix, in this case zeros and all
            */

            // lapack variables for the whole a problem (col major)!
            nrhs = 1; // one column in b
            lda = ps.N_mat;
            ldb = 1; // leading b dimention for row major
            ldb_col = ps.N_mat; // leading b dim for col major
            i_piv.resize(ps.N_mat, 0);  // pivot column vector

            // solve Ax=b
            int Ndgsev = ps.N_mat;
            dgesv_( &Ndgsev, &nrhs, &A_copy[0], &lda, &i_piv[0], &b[0], &ldb_col, &info );

            if( info > 0 ) {
                printf( "\n>>>WHOLE A ERROR<<<\n" );
                printf( "The diagonal element of the triangular factor of A,\n" );
                printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
                printf( "the solution could not be computed.\n" );
                exit( 1 );
            }
        }



        void PBJlinear_solver(vector<double> &A_sp_cm, vector<double> &b){
            /*breif: Solves individual dense cell matrices in parrallel on cpu
            requires -fopenmp to copmile.

            A and b store all matrices in col major in a single std:vector as
            offsets from one another*/

            // parallelized over the number of cells
            #pragma omp parallel for
            for (int i=0; i<ps.N_cells; ++i){

                // lapack variables for a single cell (col major!)
                nrhs = 1; // one column in b
                lda = ps.SIZE_cellBlocks; // leading A dim for col major
                ldb_col = ps.SIZE_cellBlocks; // leading b dim for col major
                std::vector<int> ipiv_par(ps.SIZE_cellBlocks); // pivot vector
                int Npbj = ps.SIZE_cellBlocks; // size of problem

                // solve Ax=b in a cell
                dgesv_( &Npbj, &nrhs, &A_sp_cm[i*ps.ELEM_cellBlocks], &lda, &ipiv_par[0], &b[i*ps.SIZE_cellBlocks], &ldb_col, &info );
            }
            
            
            if( info > 0 ) {
                printf( "\n>>>PBJ LINALG ERROR<<<\n" );
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
            //vector<double> A(ps.N_rm);

            // generation of the whole ass mat
            //A_gen(A, cells, ps);
            //vector<double> A_col = row2colSq(A);

            vector<double> A(ps.ELEM_cellBlocks*ps.N_cells);
            A_gen_sparse(A, cells, ps);

            // time step loop
            for(int t=0; t<ps.N_time; ++t){ //
                ps.time_val = t;
                time += ps.dt;
                init_af_timestep();

                if ( ps.mms_bool ){
                    sourceSource( );
                }

                // resets
                itter = 0;          // iteration counter
                error = 1;          // error from current iteration 
                error_n1 = 1;       // error back one iteration (from last)
                error_n2 = 1;       // error back two iterations
                converged = true;   // converged boolean
                
                vector<double> b_const_cpu(ps.N_mat);
                vector<double> b_const_gpu(ps.N_mat);
                b_gen_const_win_iter(b_const_cpu, aflux_previous, cells, ps);
                b_const_gpu = b_const_cpu;

                //convergenceLoop(A, b_const_cpu, t);

                ConvergenceLoopOptGPU( A, b_const_gpu, t );

                //check_close(b_const_cpu, b_const_gpu);

                aflux_previous = b_const_gpu;

                save_eos_data(t);

            } // end of time step loop
        }

        void convergenceLoop(std::vector<double> &A, std::vector<double> &b_const, int t){

            aflux_last = aflux_previous;

            while (converged){

                // lapack requires a copy of data that it uses for row piviot (A after _dgesv != A)
                std::vector<double> A_copy = A;

                // b is also used up and has to be ressinged
                std::vector<double> b = b_const;
                

                //assing angular flux
                ps.assign_boundary( aflux_last );

                // b has a constant and
                // reminder: last refers to iteration, previous refers to time step
                b_gen_var_win_iter( b, aflux_last, ps );

                //Lapack solvers
                //amdGPU_dgesv_strided_batched(A_sp_copy2, b_copy, ps);
                PBJlinear_solver( A_copy, b );

                //check_close(b, b_copy);
                
                // compute the L2 norm between the last and current iteration
                error = L2Norm( aflux_last, b );

                // compute spectral radius
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

                if (itter == 3){
                    converged = false;
                }

                aflux_last = b;
                
                cycle_print_func(t);
                
                itter++;

                error_n2 = error_n1;
                error_n1 = error;

            } // end convergence loop
        }

        void ConvergenceLoopOptGPU( std::vector<double> &hA, std::vector<double> &hb_const, int t){
            /*intilizes the information for a itteration on gpu
                - allocates memory
                - moves data back and forth
                - destroyes gpu memory manages runtime
            all functions above are unaware of gpu runtime*/
            //

            // perameters
            rocblas_int N = ps.SIZE_cellBlocks;           // ros and cols in each household problem
            rocblas_int lda = ps.SIZE_cellBlocks;         // leading dimension of A in each household problem
            rocblas_int ldb = ps.SIZE_cellBlocks;         // leading dimension of B in each household problem
            rocblas_int nrhs = 1;                         // number of nrhs in each household problem
            rocblas_stride strideA = ps.ELEM_cellBlocks;  // stride from start of one matrix to the next (household to the next)
            rocblas_stride strideB = ps.SIZE_cellBlocks;  // stride from start of one rhs to the next
            rocblas_stride strideP = ps.SIZE_cellBlocks;  // stride from start of one pivot to the next
            rocblas_int batch_count = ps.N_cells;         // number of matricies (in this case number of cells)

            rocblas_handle handle;
            rocblas_create_handle(&handle);

            // preload rocBLAS GEMM kernels (optional)
            // rocblas_initialize();

            std::vector<double> herror (3);
            std::vector<int> probSpace {ps.N_cells, ps.N_groups, ps.N_angles};

            print_vec_sd_int(probSpace);

            // allocate memory on GPU
            double *dA, *db, *dangles, *dboundary, *daflux_last, *db_const;
            rocblas_int *ipiv, *dinfo;
            int *dps;  // without further inreration

            // alloaction of problem
            hipMalloc(&dA, sizeof(double)*strideA*batch_count);         // allocates memory for strided matrix container
            hipMalloc(&db_const, sizeof(double)*strideB*batch_count);         // allocates memory for strided rhs container
            hipMalloc(&db, sizeof(double)*strideB*batch_count);         // allocates memory for strided rhs container
            hipMalloc(&daflux_last, sizeof(double)*strideB*batch_count);
            hipMalloc(&dangles, sizeof(double)*ps.N_angles);
            hipMalloc(&dboundary, sizeof(double)*ps.N_angles*ps.N_groups*2);

            // integers
            hipMalloc(&ipiv, sizeof(rocblas_int)*strideB*batch_count);  // allocates memory for integer pivot vector in GPU
            hipMalloc(&dinfo, sizeof(rocblas_int)*batch_count);
            hipMalloc(&dps, sizeof(int)*3);

            // copy data to GPU
            hipMemcpy(db_const, &hb_const[0], sizeof(double)*strideB*batch_count, hipMemcpyHostToDevice);
            hipMemcpy(daflux_last, &aflux_previous[0], sizeof(double)*ps.N_mat, hipMemcpyHostToDevice);
            hipMemcpy(dangles, &ps.angles[0], sizeof(double)*ps.N_angles, hipMemcpyHostToDevice);
            hipMemcpy(dps, &probSpace[0], sizeof(int)*3, hipMemcpyHostToDevice);

            //hipMemcpy(dps, &probSpace[0], sizeof(double)*strideA*batch_count, hipMemcpyHostToDevice);

            itter = 0;

            int threadsperblock = 256;
            int blockspergrid = (ps.N_mat + (threadsperblock - 1)) / threadsperblock;

            std::vector<double> hb(ps.N_mat);
            std::vector<double> hb_const_check(ps.N_mat);

            // on gpu!
            while (converged){

                hipMemcpy(dA, &hA[0], sizeof(double)*strideA*batch_count, hipMemcpyHostToDevice);
                hipMemcpy(daflux_last, &hb[0], sizeof(double)*strideB*batch_count, hipMemcpyHostToDevice);
                hipMemcpy(db, &hb_const[0], sizeof(double)*strideB*batch_count, hipMemcpyHostToDevice);
                

                //gpu_assign_boundary(dboundary, daflux_last);
                //hipDeviceSynchronize();

                //GPUb_gen_var_win_iter(double *b, double *aflux_last, , double *angles, int *ps)
                hipMemcpy(&hb[0], db, sizeof(double)*ps.N_mat, hipMemcpyDeviceToHost);
                //std::cout << "NEW ITTERATION" << std::endl;
                //std::cout << "" << std::endl;
                //std::cout << "b inital" << std::endl;
                //print_vec_sd(hb);

                int threadsperblock = 256;
                int blockspergrid = (ps.N_cells + (threadsperblock - 1)) / threadsperblock;
                hipLaunchKernelGGL(GPUb_gen_var_win_iter, dim3(blockspergrid), dim3(threadsperblock), 0, 0, 
                                    db, daflux_last, dangles, dps );
                hipDeviceSynchronize();

                //hipMemcpy(&hb_const_check[0], db_const, sizeof(double)*ps.N_mat, hipMemcpyDeviceToHost);

                //std::cout<<"in itteration print" <<std::endl;
                hipMemcpy(&hb[0], db, sizeof(double)*ps.N_mat, hipMemcpyDeviceToHost);
                //std::cout << "b after initilization" << std::endl;
                //print_vec_sd(hb);
                ///print_vec_sd(hb);
                //std::cout << "b_const" << std::endl;
                //print_vec_sd(hb_const);
                //std::cout << "db_const_check" << std::endl;
                //print_vec_sd(hb_const_check);

                rocsolver_dgesv_strided_batched(handle, N, nrhs, dA, lda, strideA, ipiv, strideP, db, ldb, strideB, dinfo, batch_count);
                hipDeviceSynchronize();

                hipMemcpy(&hb[0], db, sizeof(double)*ps.N_mat, hipMemcpyDeviceToHost);

                herror[0] = L2Norm( aflux_last, hb );
                
                error = herror[0];
                error_n2 = error_n1;
                error_n1 = error;

                //print_vec_sd(herror);
                //hipMemcpy(&hb_const_check[0], daflux_last, sizeof(double)*ps.N_mat, hipMemcpyDeviceToHost);

                //warning! daflux_last is in-out
                //herror[0] = gpuL2norm(handle, daflux_last, db, ps.N_mat);
                //hipDeviceSynchronize();
                // what is not understood is the functional argument in the design situation

                // Right I wunderstand but I have not been functionally around to establish an undercurrent for desning of the facilities
                // I think if I narrow out the underlying funciton of the creation I should be able to establish the grea
                //print_vec_sd(herror);

                // on cpu
                spec_rad = pow( pow(herror[0]+herror[1],2), .5) / pow(pow(herror[1]+herror[2], 2), 0.5);

                if (itter > 2){
                    if ( error < ps.convergence_tolerance*(1-spec_rad) ){ converged = false; } }

                if (itter >= ps.max_iteration){
                                cout << ">>>WARNING: Computation did not converge after " << ps.max_iteration << "iterations<<<" << endl;
                                cout << "       itter: " << itter << endl;
                                cout << "       error: " << error << endl;
                                cout << "" << endl;
                                converged = false;
                }

                //if (itter == 3){
                //    converged = false;
                //}

                cycle_print_func(t);

                aflux_last = hb;
                
                itter++;

                herror[2] = herror[1];
                herror[1] = herror[0];

                hipFree(ipiv);
                hipFree(dinfo);
            }

            hipMemcpy(&aflux_previous[0], db, sizeof(double)*ps.N_mat, hipMemcpyDeviceToHost);

            hipFree(dA);
            hipFree(db);
            hipFree(daflux_last);
            hipFree(dangles);
            hipFree(dboundary);
            
        }
};





/*
__global__ gpu_b_gen_var_win_iter(double *b, double *aflux_last, double *angles, double *boundary, int *ps ){
    //brief: builds b

    // helper index
    int index_start;
    int index_start_n1;
    int index_start_p1;

    int i =  blockDim.x;
    int g = blockIdx.x;
    int j = threadIdx.x;

    //for (int i=0; i<ps.N_cells; i++){
    //    for (int g=0; g<ps.N_groups; g++){
    //        for (int j=0; j<ps.N_angles; j++){

    //ps is an int vector holding data about the problem
    // [0] is the total number of elements in b
    // [1] SIZE_cellBlocks
    // [2] SIZE_groupBlocks
    // [3]

    // the first index in the smallest chunk of 4
    int index_start = (i*(ps[1]) +  g*(ps[2]) + 4*j);
    // 4 blocks organized af_l, af_r, af_hn_l, af_hn_r

    if (index_start < ps[0])

        // negative angle
        if (angles[j] < 0){
            if (i == ps.N_cells-1){ // right boundary condition
                int boundary_index = g*4 +j*3;
                b[index_start+1] -= angles[j]*boundary[boundary_index];
                b[index_start+3] -= angles[j]*boundary[boundary_index];

            } else { // pulling information from right to left
                index_start_p1 = index_start + ps[1];

                //outofbounds_check(index_start_p1, aflux_last);
                //outofbounds_check(index_start_p1+2, aflux_last);

                b[index_start+1] -= angles[j]*aflux_last[index_start_p1];
                b[index_start+3] -= angles[j]*aflux_last[index_start_p1+2];
            }

        // positive angles
        } else {
            if (i == 0){ // left boundary condition

                int boundary_index = g*4 +j*3;
                
                b[index_start]    += angles[j]*boundary[boundary_index];
                b[index_start+2]  += angles[j]*boundary[boundary_index];

            } else { // pulling information from left to right
                index_start_n1 = index_start - ps[1];

                b[index_start]    += angles[j]*aflux_last[index_start_n1+1];
                b[index_start+2]  += angles[j]*aflux_last[index_start_n1+3];
            }
        }
    }


*/