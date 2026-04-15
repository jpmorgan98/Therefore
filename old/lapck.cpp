#include <iostream>
#include <vector>
//#include "lapacke.h"

extern "C" void dgesv_( int *n, int *nrhs, double  *a, int *lda, int *ipiv, double *b, int *lbd, int *info  );

int main() {
    int SIZE = 3;
    int nrhs = 1; // one column in b
    int lda = SIZE;
    int ldb = 1;
    std::vector<int> i_piv(SIZE, 0);  // pivot column vector
    int info;
    std::vector<double> A(SIZE*SIZE, 0); // sq mat with 0's
    A = {5, 2, 8, 9, 7, 2, 10, 3, 4};
    std::vector<double> b(SIZE);
    b = {22, 13, 17};

    //LAPACK_ROW_MAJORs

    dgesv_( &SIZE, &nrhs, &*A.begin(), &lda, &*i_piv.begin(), &*b.begin(), &ldb, &info );
    //info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, SIZE, nrhs, &A[0], lda, &i_piv[0], &b[0], ldb );


    std::cout << b[0] << std::endl;

    return 0;
}