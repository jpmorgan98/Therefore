#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>

using namespace std



int cuSolver( std::vector<double> &dA, std::vector<double> &hB) {
    /*Adapted from:  https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/gesv/cusolver_irs_expert_cuda-11.cu*/
    
    int SIZE_wholeproblem = pow(A.size, 0.5);
    int nrhs = 1;


    bool verbose = false;

    // Matrix size
    const int N = 1024;

    // Numer of right hand sides
    const int nrhs = 1;

    // Use double precision matrix and half precision factorization
    typedef double T;
    // Select appropriate functions for chosen precisions
    auto cusolver_gesv_buffersize = cusolverDnDHgesv_bufferSize;
    auto cusolver_gesv = cusolverDnDHgesv;

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

    // matrix on device
    T *dA;
    cusolver_int_t ldda = ALIGN_TO(N * sizeof(T), device_alignment) / sizeof(T);
    // right hand side on device
    T *dB;
    cusolver_int_t lddb = ALIGN_TO(N * sizeof(T), device_alignment) / sizeof(T);
    // solution on device
    T *dX;
    cusolver_int_t lddx = ALIGN_TO(N * sizeof(T), device_alignment) / sizeof(T);

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
    CUDA_CHECK(cudaMemcpy2D(dA, ldda * sizeof(T), &hA[0], lda * sizeof(T), N * sizeof(T), N, cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy2D(dB, lddb * sizeof(T), &hB[0], ldb * sizeof(T), N * sizeof(T), nrhs, cudaMemcpyDefault));

    //cudaMemcpy	(	void * 	dst, const void * 	src, size_t 	count, enum cudaMemcpyKind 	kind	 )	

    // get required device work buffer size
    CUSOLVER_CHECK(cusolver_gesv_buffersize(handle, N, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx,
                                            dwork, &dwork_size));
    std::cout << "Workspace is " << dwork_size << " bytes" << std::endl;
    CUDA_CHECK(cudaMalloc(&dwork, dwork_size));

    std::cout << "Solving matrix on device..." << std::endl;
    CUDA_CHECK(cudaEventRecord(event_start, stream));

    cusolverStatus_t gesv_status = cusolver_gesv(handle, N, nrhs, dA, ldda, dipiv, dB, lddb, dX,
                                                 lddx, dwork, dwork_size, &iter, dinfo);
    CUSOLVER_CHECK(gesv_status);

    CUDA_CHECK(cudaEventRecord(event_end, stream));
    // check solve status
    int info = 0;
    CUDA_CHECK(
        cudaMemcpyAsync(&info, dinfo, sizeof(cusolver_int_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "Solve info is: " << info << ", iter is: " << iter << std::endl;

    CUDA_CHECK(cudaMemcpy2D(hX, ldx * sizeof(T), dX, lddx * sizeof(T), N * sizeof(T), nrhs,
                            cudaMemcpyDefault));
    if (verbose) {
        std::cout << "X:\n";
        print_matrix(nrhs, N, hX, ldx);
    }

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

    free(hA);
    free(hB);
    free(hX);

    CUSOLVER_CHECK(cusolverDnDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaEventDestroy(event_end));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "Done!" << std::endl;

    return 0;



}

__global__ void square_vector(vector<double> &A_copy, vector<double> &b, ){

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();

    cudaError_t error = cudaMallocArray( &cuArray, &channelDesc, size,1);

    cudaError_t error1= cudaMemcpyToArray(cuArray, 0, 0, (void*)&A_copy[0], size, cudaMemcpyHostToDevice);
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