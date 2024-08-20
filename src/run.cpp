#include "util.h"
#include "builders.h"
#include <hip/hip_runtime.h>
#include "rocsolver.cpp"
//#include <omp.h>

bool OPTIMIZED = true;

// lapack function! To copmile requires <-Ipath/to/lapack/headers -llapack>
extern "C" void dgesv_( int *n, int *nrhs, double  *a, int *lda, int *ipiv, double *b, int *lbd, int *info  );

class run{

    public:

        problem_space ps;
        vector<cell> cells;
        vector<double> IC;

        vector<double> aflux_last;
        vector<double> aflux_previous;


        bool cycle_print = true;
        bool save_output = true;
        

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
        //mms manSource;

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

        void cycle_print_func( int t, double elapsed ){
            int cycle_print_flag = 0; // for printing headers

            if (itter != 0) 
                cycle_print_flag = 1;

            if (cycle_print){
                if (cycle_print_flag == 0) {
                    cout << ">>>OCI CYCLE INFO FOR TIME STEP: " << t <<" for dt: " << ps.dt << "<<<"<< endl;
                    printf("cycle   error         error_n1      error_n2     spec_rad   cycle_time\n");
                    printf("===================================================================================\n");
                    cycle_print_flag = 1;
                }
                printf("%3d      %1.4e    %1.4e    %1.4e   %1.4e    %1.4e\n", itter, error, error_n1, error_n2, spec_rad, elapsed );
            }
        }

        void save_eos_data(int t){
            string ext = ".csv";
            string file_name = "afluxUnsorted_therefore";
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

        void linear_solver(vector<double> &A_copy, vector<double> &b){
            /* DO j CALL DEBUGING ONLY
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

        void run_timestep(){

            init_vectors();

            vector<double> A(ps.ELEM_cellBlocks*ps.N_cells);
            A_gen_sparse(A, cells, ps);

            // time step loop
            for(int t=0; t<ps.N_time; ++t){ //

                ps.time_val = t;
                time += ps.dt;
                init_af_timestep();

                // resets
                itter = 0;          // iteration counter
                error = 1;          // error from current iteration 
                error_n1 = 1;       // error back one iteration (from last)
                error_n2 = 1;       // error back two iterations
                converged = true;   // converged boolean
                
                vector<double> b_const_cpu( ps.N_mat );
                vector<double> b_const_gpu( ps.N_mat );
                b_gen_const_win_iter( b_const_cpu, aflux_previous, cells, ps );
                b_const_gpu = b_const_cpu;

                //convergenceLoop( A, b_const_cpu, t );

                ConvergenceLoopOptGPU( A, b_const_gpu, t );

                //check_close( b_const_cpu, b_const_gpu );

                aflux_previous = b_const_gpu;

                if ( save_output )
                    save_eos_data(t);

            } // end of time step loop
        }

        void convergenceLoop(std::vector<double> &A, std::vector<double> &b_const, int t){

            aflux_last = aflux_previous;
            converged = false;
            itter = 0;
            error = 1.0;
            error_n1 = 1.0;
            error_n2 = 1.0;

            while (!converged){

                Timer timer2;

                // lapack requires a copy of data that it uses for row piviot (A after _dgesv != A)
                std::vector<double> A_copy = A;

                // b is also used up and has to be ressinged
                std::vector<double> b = b_const;
                //std::vector<double> b (ps.N_mat);
                

                //assing angular flux
                ps.assign_boundary( aflux_last );

                // b has a constant and
                // reminder: last refers to iteration, previous refers to time step
                b_gen_var_win_iter( b, aflux_last, ps );
                //b_gen(b, aflux_previous, aflux_last, cells, ps);


                //Lapack solvers
                //amdGPU_dgesv_strided_batched(A_copy, b, ps);
                PBJlinear_solver( A_copy, b );

                //check_close(b, b_copy);
                
                // compute the L2 norm between the last and current iteration
                error = infNorm_error( aflux_last, b );

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
                    converged = true;
                }

                aflux_last = b;
                
                if ( cycle_print )
                    cycle_print_func( t, timer2.elapsed() );
                
                itter++;

                error_n2 = error_n1;
                error_n1 = error;

                

            } // end convergence loop

        //b_const = aflux_last;
        }

        void ConvergenceLoopOptGPU( std::vector<double> &hA, std::vector<double> &hb_const, int t){
            /*intilizes the information for a itteration on gpu
                - allocates memory
                - moves data back and forth
                - destroyes gpu memory
                - manages runtime
            all functions above are unaware of gpu runtime*/
            //

            // perameters
            rocblas_int N = ps.SIZE_cellBlocks;           // rows and cols in each household problem
            rocblas_int lda = ps.SIZE_cellBlocks;         // leading dimension of A in each household problem
            rocblas_int ldb = ps.SIZE_cellBlocks;         // leading dimension of B in each household problem
            rocblas_int nrhs = 1;                         // number of nrhs in each household problem
            rocblas_stride strideA = ps.ELEM_cellBlocks;  // stride from start of one matrix to the next (household to the next)
            rocblas_stride strideB = ps.SIZE_cellBlocks;  // stride from start of one rhs to the next
            rocblas_stride strideP = ps.SIZE_cellBlocks;  // stride from start of one pivot to the next
            rocblas_int batch_count = ps.N_cells;         // number of matricies (in this case number of cells)

            rocblas_handle handle;
            rocblas_create_handle(&handle);

            // when profiling the funtion 
            // preload rocBLAS GEMM kernels (optional)
            // rocblas_initialize();

            std::vector<double> herror (3);
            std::vector<int> probSpace {ps.N_cells, ps.N_groups, ps.N_angles};

            //print_vec_sd_int(probSpace);

            // defininig pointers to memory on GPU
            double *dA, *db, *dangles, *dboundary, *daflux_last, *db_const;
            rocblas_int *ipiv, *dinfo;
            int *dps;  // without further inreration

            // double alloaction of problem
            hipMalloc(&dA, sizeof(double)*strideA*batch_count);         // allocates memory for strided matrix container
            hipMalloc(&db_const, sizeof(double)*strideB*batch_count);         // allocates memory for strided rhs container
            hipMalloc(&db, sizeof(double)*strideB*batch_count);         // allocates memory for strided rhs container
            hipMalloc(&daflux_last, sizeof(double)*strideB*batch_count);
            hipMalloc(&dangles, sizeof(double)*ps.N_angles);
            hipMalloc(&dboundary, sizeof(double)*ps.N_angles*ps.N_groups*2);

            // integer allocation
            hipMalloc(&ipiv, sizeof(rocblas_int)*strideB*batch_count);  // allocates memory for integer pivot vector in GPU
            hipMalloc(&dinfo, sizeof(rocblas_int)*batch_count);
            hipMalloc(&dps, sizeof(int)*3);

            // copy data to GPU
            hipMemcpy(db_const, &hb_const[0], sizeof(double)*strideB*batch_count, hipMemcpyHostToDevice);
            hipMemcpy(daflux_last, &aflux_previous[0], sizeof(double)*ps.N_mat, hipMemcpyHostToDevice);
            hipMemcpy(dangles, &ps.angles[0], sizeof(double)*ps.N_angles, hipMemcpyHostToDevice);
            hipMemcpy(dps, &probSpace[0], sizeof(int)*3, hipMemcpyHostToDevice);

            itter = 0;

            int threadsperblock = 256;
            int blockspergrid = (ps.N_mat + (threadsperblock - 1)) / threadsperblock;

            std::vector<double> hb(ps.N_mat);
            std::vector<double> hb_const_check(ps.N_mat);
            converged = true;

            hipMemcpy(dA, &hA[0], sizeof(double)*strideA*batch_count, hipMemcpyHostToDevice);

            Timer timer;

            // on gpu!
            while (converged){

                Timer timer2;
                
                hipMemcpy(daflux_last, &hb[0], sizeof(double)*strideB*batch_count, hipMemcpyHostToDevice);
                hipMemcpy(db, &hb_const[0], sizeof(double)*strideB*batch_count, hipMemcpyHostToDevice);

                int threadsperblock = 256;
                int blockspergrid = (ps.N_cells + (threadsperblock - 1)) / threadsperblock;
                hipLaunchKernelGGL(GPUb_gen_var_win_iter, dim3(blockspergrid), dim3(threadsperblock), 0, 0, 
                                    db, daflux_last, dangles, dps );
                hipDeviceSynchronize();
                
                //first iteration we solve for the LU decomp in each cell and solve with back subbing
                // in subsequent iteration we just back substitute as A is already solved for
                
                if ( OPTIMIZED ){
                    //std::cout << "OPTIMIZED" << std::endl;
                    if (itter == 0){
                        rocsolver_dgesv_strided_batched(handle, N, nrhs, dA, lda, strideA, ipiv, strideP, db, ldb, strideB, dinfo, batch_count);
                        hipDeviceSynchronize();
                    } else {
                        //enum rocblas_operation_none;
                        rocsolver_dgetrs_strided_batched(handle, rocblas_operation_none, N, nrhs, dA, lda, strideA, ipiv, strideP, db, ldb, strideB, batch_count);
                        hipDeviceSynchronize();
                    } 
                } else {
                    //std::cout << "NOT OPTIMIZED" << std::endl;
                    hipMemcpy(dA, &hA[0], sizeof(double)*strideA*batch_count, hipMemcpyHostToDevice);
                    rocsolver_dgesv_strided_batched(handle, N, nrhs, dA, lda, strideA, ipiv, strideP, db, ldb, strideB, dinfo, batch_count);
                    hipDeviceSynchronize();
                }

                // warning! daflux_last is in-out!
                error = gpuL2norm(handle, daflux_last, db, ps.N_mat);
                hipDeviceSynchronize();

                // on cpu
                spec_rad = pow( pow(error+error_n1,2), .5) / pow(pow(error_n1+error_n2, 2), 0.5);

                if (itter > 2){
                    if ( error < ps.convergence_tolerance ){ converged = false; } } //*(1-spec_rad)

                if (itter >= ps.max_iteration){
                                cout << ">>>WARNING: Computation did not converge after " << ps.max_iteration << "iterations<<<" << endl;
                                cout << "       itter: " << itter << endl;
                                cout << "       error: " << error << endl;
                                cout << "" << endl;
                                converged = false;
                }

                if (cycle_print)
                    cycle_print_func(t, timer2.elapsed() );

                hipMemcpy(&hb[0], db, sizeof(double)*ps.N_mat, hipMemcpyDeviceToHost);
                aflux_last = hb;
                
                itter++;

                error_n2 = error_n1;
                error_n1 = error;

            }

            ps.time_conv_loop = timer.elapsed();
            ps.av_time_per_itter = timer.elapsed()/itter-1;

            //std::cout << "Time elapsed in OCI transport only: " << timer.elapsed() << " seconds\n";

            hipMemcpy(&aflux_previous[0], db, sizeof(double)*ps.N_mat, hipMemcpyDeviceToHost);

            hipFree(ipiv);
            hipFree(dinfo);
            hipFree(dA);
            hipFree(db);
            hipFree(daflux_last);
            hipFree(dangles);
            hipFree(dboundary);
            hipFree(db_const);
            hipFree(dps);

            rocblas_destroy_handle(handle);
        }
};