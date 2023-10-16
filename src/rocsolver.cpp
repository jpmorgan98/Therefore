#include <iostream>
#include <algorithm> // for std::min
#include <stddef.h>  // for size_t
#include <vector>
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <hip/hip_runtime_api.h> // for hip functions

// compile
// module load rocm
// hipcc rocsolver.cpp -I/opt/rocm/include -lrocsolver -lrocblas
// cc

void amdGPU_dgesv( std::vector<double> &hA, std::vector<double> &hb ) {
    //https://rocsolver.readthedocs.io/en/latest/api/lapack.html#_CPPv415rocsolver_dgesv14rocblas_handleK11rocblas_intK11rocblas_intPdK11rocblas_intP11rocblas_intPdK11rocblas_intP11rocblas_int
    
    size_t n = hb.size();
    size_t A = n*n;
    const rocblas_int N = n;
    const rocblas_int lda = n;
    const rocblas_int ldb = n;
    const rocblas_int nrhs = 1;

    hipStream_t stream;
    hipStreamCreate(&stream);

    rocblas_handle handle;
    rocblas_create_handle(&handle); 

    rocblas_set_stream(handle, stream);

    //rocblas_initialize();

    double *dA;
    double *db;

    // integer pivot array on device
    rocblas_int *ipiv;
    rocblas_int *dinfo;

    // alloaction of problem
    hipMalloc(&dA, sizeof(double)*A); // allocates memory for LHS matrix in GPU
    hipMalloc(&db, sizeof(double)*n);   // allocates memory for RHS vector in GPU
    hipMalloc(&ipiv, sizeof(rocblas_int)*n);   // allocates memory for integer pivot vector in GPU
    hipMalloc(&dinfo, sizeof(rocblas_int));

    // copy data to GPU
    hipMemcpy(dA, &hA[0], sizeof(double)*A, hipMemcpyHostToDevice);
    hipMemcpy(db, &hb[0], sizeof(double)*n, hipMemcpyHostToDevice);

    //rocblas_status rocsolver_dgesv(rocblas_handle handle, const rocblas_int n, const rocblas_int nrhs, double *A, const rocblas_int lda, rocblas_int *ipiv, double *B, const rocblas_int ldb, rocblas_int *info)
    rocblas_status gesv_status = rocsolver_dgesv(handle, n, nrhs, dA, lda, ipiv, db, ldb, dinfo);

    int info = 0;
    hipMemcpyAsync(&info, dinfo, sizeof(rocblas_int), hipMemcpyDeviceToHost, stream);
    if (info != 0){
        std::cout << "ROCSOLVER ERROR: Matrix is singular" << std::endl;
        std::cout << info << std::endl;
    }
    hipDeviceSynchronize();

    // copy the results back to CPU
    hipMemcpy(&hb[0], db, sizeof(double)*n, hipMemcpyDeviceToHost);


    hipFree(dA);                        // de-allocate GPU memory
    hipFree(db);                        // de-allocate GPU memory
    hipFree(ipiv);
}

/*
class problem_space{
        public:
            int SIZE_cellBlocks;
            int N_cells;
            int ELEM_cellBlocks;
    };*/

void amdGPU_dgesv_strided_batched( std::vector<double> &hA, std::vector<double> &hb, problem_space ps) {
    
    /*breif:
        Solves a set of individual matrices striding over a cell.
        Right now everything is allocated and de-allocated on in every itteration
        this will be optimized in future implementaitons
    */
    
    // perameters
    rocblas_int N = ps.SIZE_cellBlocks;           // ros and cols in each household problem
    rocblas_int lda = ps.SIZE_cellBlocks;         // leading dimension of A in each household problem
    rocblas_int ldb = ps.SIZE_cellBlocks;         // leading dimension of B in each household problem
    rocblas_int nrhs = 1;                         // number of nrhs in each household problem
    rocblas_stride strideA = ps.ELEM_cellBlocks;  // stride from start of one matrix to the next (household to the next)
    rocblas_stride strideB = ps.SIZE_cellBlocks;  // stride from start of one rhs to the next
    rocblas_stride strideP = ps.SIZE_cellBlocks;  // stride from start of one pivot to the next
    rocblas_int batch_count = ps.N_cells;         // number of matricies (in this case number of cells)

    // initialization
    //hipStream_t stream;
    //hipStreamCreate(&stream);

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    //rocblas_set_stream(handle, &stream);

    // preload rocBLAS GEMM kernels (optional)
    // rocblas_initialize();

    // allocate memory on GPU
    double *dA, *db;
    rocblas_int *ipiv, *dinfo;

    // alloaction of problem
    hipMalloc(&dA, sizeof(double)*strideA*batch_count);         // allocates memory for strided matrix container
    hipMalloc(&db, sizeof(double)*strideB*batch_count);         // allocates memory for strided rhs container
    hipMalloc(&ipiv, sizeof(rocblas_int)*strideB*batch_count);  // allocates memory for integer pivot vector in GPU
    hipMalloc(&dinfo, sizeof(rocblas_int)*batch_count);

    // copy data to GPU
    hipMemcpy(dA, &hA[0], sizeof(double)*strideA*batch_count, hipMemcpyHostToDevice);
    hipMemcpy(db, &hb[0], sizeof(double)*strideB*batch_count, hipMemcpyHostToDevice);

    // bathed strided soultion to Ax=b via double-gesv
    // rocsolver_dgesv_strided_batched(rocblas_handle handle, const rocblas_int n, const rocblas_int nrhs, 
    //          double *A, const rocblas_int lda, const rocblas_stride strideA, rocblas_int *ipiv, const rocblas_stride strideP, 
    //          double *B, const rocblas_int ldb, const rocblas_stride strideB, rocblas_int *info, const rocblas_int batch_count)
    
    rocsolver_dgesv_strided_batched(handle, N, nrhs, dA, lda, strideA, ipiv, strideP, db, ldb, strideB, dinfo, batch_count);
    
    hipDeviceSynchronize();

    //std::cout << "A" << std::endl;
    std::vector<int> info(batch_count);
    //std::cout << "B" << std::endl;
    hipMemcpy(&info[0], dinfo, sizeof(rocblas_int)*batch_count, hipMemcpyDeviceToHost);
    //std::cout << "C" << std::endl;
    for (int k=0; k<info.size(); ++k){
        if (info[k] != 0){
            std::cout << "ROCSOLVER ERROR: Matrix is singular in batch " << k << std::endl;
            std::cout << info[k] << std::endl;
        }
    }

    // copy the results back to CPU
    hipMemcpy(&hb[0], db, sizeof(double)*strideB*batch_count, hipMemcpyDeviceToHost);

    // clean up
    hipFree(dA);
    hipFree(db);
    hipFree(ipiv);
    hipFree(dinfo);
    //std::cout << "delta" << std::endl;

    rocblas_destroy_handle(handle);
    //std::cout << "gamma" << std::endl;
}



/*

int Test_amdGPU_dgesv_batched(){

    std::vector<double> A = {1, 3 , 2.1, 5, 6.5, 5.5, 8.4, 1.2, 5.9, 1, 3 , 2.1, 5, 6.5, 5.5, 8.4, 1.2, 5.9};
    std::vector<double> b = {4.2, 3.2, 2.3, 4.2, 3.2, 2.3};

    problem_space ps;
    ps.SIZE_cellBlocks = 3;
    ps.N_cells = 2;
    ps.ELEM_cellBlocks = 9;

    amdGPU_dgesv_batched(A, b, ps);

    std::cout <<" should be:      actually was" << std::endl;
    std::cout<<" "<<b[0]<<"    = "<<"-3.71152895"<<std::endl;
    std::cout<<" "<<b[1]<<"    = "<<"2.28223652"<<std::endl;
    std::cout<<" "<<b[2]<<"    = "<<"-0.41662543"<<std::endl;
    std::cout<<" "<<b[3]<<"    = "<<"-3.71152895"<<std::endl;
    std::cout<<" "<<b[4]<<"    = "<<"2.28223652"<<std::endl;
    std::cout<<" "<<b[5]<<"    = "<<"-0.41662543"<<std::endl;

    return(1);
}

int Test_amdGPU_dgesv(){

    std::vector<double> A = {1, 3 , 2.1, 5, 6.5, 5.5, 8.4, 1.2, 5.9};
    std::vector<double> b = {4.2, 3.2, 2.3};

    amdGPU_dgesv(A, b);

    std::cout << "Computed soultion:" << std::endl;
    std::cout<<b[0]<<std::endl;
    std::cout<<b[1]<<std::endl;
    std::cout<<b[2]<<std::endl;
    std::cout << "" << std::endl;
    std::cout << "Exact soultion:" << std::endl;
    std::cout<<"-3.71152895"<<std::endl;
    std::cout<<"2.28223652"<<std::endl;
    std::cout<<"-0.41662543"<<std::endl;

    return( 1 );
}


/*
int main(){
    //hipcc rocsolver.cpp -I/opt/rocm/include -lrocsolver -lrocblas

    Test_amdGPU_dgesv_batched();
    //Test_amdGPU_dgesv();
    return( 1 );
}*/