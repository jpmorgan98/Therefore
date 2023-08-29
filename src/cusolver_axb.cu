#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"

using namespace std;

void cuSolver( std::vector<double> &hA, std::vector<double> &hB) {
    /*Adapted from:  https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/gesv/cusolver_irs_expert_cuda-11.cu*/
    
    // Matrix size
    int N = hB.size();
    
    // Allocate soultion vector
    //std::vector<double> hX (N);

    // number of right hand sides
    int nrhs = 1;

    // Use double precision matrix and half precision factorization
    typedef double T;
    // Select appropriate functions for chosen precisions
    auto cusolver_gesv_buffersize = cusolverDnDHgesv_bufferSize;
    auto cusolver_gesv = cusolverDnDDgesv;

    cusolver_int_t lda;
    cusolver_int_t ldb;
    cusolver_int_t ldx;

    cudaStream_t stream;
    cudaEvent_t event_start, event_end;
    cusolverDnHandle_t handle;

    std::cout << "Initializing CUDA..." << std::endl;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_end));
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));

    cout << "n:" << N << endl;
    // matrix on device
    T *dA;
    cusolver_int_t ldda = ALIGN_TO(N * sizeof(T), device_alignment) / sizeof(T);
    cout << "ldda (should be n):  " << ldda << endl;
    // right hand side on device
    T *dB;
    cusolver_int_t lddb = ALIGN_TO(N * sizeof(T), device_alignment) / sizeof(T);
    cout << "lddb (should be n):  " << lddb << endl;
    // solution on device
    T *dX;
    cusolver_int_t lddx = ALIGN_TO(N * sizeof(T), device_alignment) / sizeof(T);
    cout << "lddx (should be n):  " << lddx << endl;

    // pivot sequence on device
    cusolver_int_t *dipiv;
    // info indicator on device
    cusolver_int_t *dinfo;
    // work buffer
    void *dwork;
    // size of work buffer
    size_t dwork_size;
    // number of refinement iterations returned by solver
    cusolver_int_t iter;
    
    std::cout << "Allocating memory on device..." << std::endl;
    // allocate data
    CUDA_CHECK(cudaMalloc(&dA, ldda * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dB, lddb * nrhs * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dX, lddx * nrhs * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dipiv, N * sizeof(cusolver_int_t)));
    CUDA_CHECK(cudaMalloc(&dinfo, sizeof(cusolver_int_t)));

    // copy input data
    CUDA_CHECK(cudaMemcpy(dA, &hA[0], N * N * sizeof(T), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(dB, &hB[0], N * sizeof(T),     cudaMemcpyDefault));

    //cudaMemcpy	(	void * 	dst, const void * 	src, size_t 	count, enum cudaMemcpyKind 	kind	 )	

    // get required device work buffer size
    CUSOLVER_CHECK(cusolver_gesv_buffersize(handle, N, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                            dwork, &dwork_size));
    std::cout << "Workspace is " << dwork_size << " bytes" << std::endl;
    CUDA_CHECK(cudaMalloc(&dwork, dwork_size));

    std::cout << "Solving matrix on device..." << std::endl;
    CUDA_CHECK(cudaEventRecord(event_start, stream));

    /*
    cusolverDnDDgesv_bufferSize(
    cusolverHandle_t                handle,
    int                             n,
    int                             nrhs,
    double                      *   dA,
    int                             ldda,
    int                         *   dipiv,
    double                      *   dB,
    int                             lddb,
    double                      *   dX,
    int                             lddx,
    void                        *   dwork,
    size_t                      *   lwork_bytes);

    rhs = 1; // one column in b
                lda = ps.N_mat;
                ldb = ps.N_mat; // leading b dimention for row major
                ldb_col = ps.N_mat; // leading b dim for col major
                i_piv.resize(ps.N_mat, 0);  // pivot column vector
            }

            // solve Ax=b
            //info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, N_mat, nrhs, &A_copy[0], lda, &i_piv[0], &b[0], ldb );
            dgesv_( &ps.N_mat, &nrhs, &A_copy[0], &lda, &i_piv[0], &b[0], &ldb_col, &info );
    */

    // Actual solve command
    cusolverStatus_t gesv_status = cusolver_gesv(handle, N, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dwork, dwork_size, &iter, dinfo);
    CUSOLVER_CHECK(gesv_status);

    CUDA_CHECK(cudaEventRecord(event_end, stream));
    // check solve status
    int info = 0;
    CUDA_CHECK(
        cudaMemcpyAsync(&info, dinfo, sizeof(cusolver_int_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "Solve info is: " << info << ", iter is: " << iter << std::endl;

    // push data back into hB vector like how normal LAPACK does
    CUDA_CHECK(cudaMemcpy(&hB[0], dX, N * sizeof(T), cudaMemcpyDefault));

    CUDA_CHECK(cudaGetLastError());

    float solve_time = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&solve_time, event_start, event_end));

    std::cout << "Releasing resources..." << std::endl;
    CUDA_CHECK(cudaFree(dwork));
    CUDA_CHECK(cudaFree(dinfo));
    CUDA_CHECK(cudaFree(dipiv));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));

    CUSOLVER_CHECK(cusolverDnDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaEventDestroy(event_end));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "Done!" << std::endl;

}

int main () {

    std::vector<double> A = {1, 3 , 2.1, 5, 6.5, 5.5, 8.4, 1.2, 5.9};
    std::vector<double> b = {4.2, 3.2, 2.3};

    cuSolver(A, b);

    cout<<b[0]<<endl;
    cout<<b[1]<<endl;
    cout<<b[2]<<endl;

    return( 1 );
}

/*
cusolverStatus_t
            cusolverDnDDgesv_bufferSize(
                cusolverHandle_t                handle,
                int                             n,
                int                             nrhs,
                double                      *   dA,
                int                             ldda,
                int                         *   dipiv,
                double                      *   dB,
                int                             lddb,
                double                      *   dX,
                int                             lddx,
                void                        *   dwork,
                size_t                      *   lwork_bytes);
                */