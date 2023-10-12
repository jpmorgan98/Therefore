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
    hipStreamSynchronize(stream);

    // copy the results back to CPU
    hipMemcpy(&hb[0], db, sizeof(double)*n, hipMemcpyDeviceToHost);

    hipMemcpy(&hb[0], db, sizeof(double)*n, hipMemcpyDeviceToHost);

    hipFree(dA);                        // de-allocate GPU memory
    hipFree(db);                        // de-allocate GPU memory
    hipFree(ipiv);
}

/*
int main(){

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

    return(1);
}
*/