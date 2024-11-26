using namespace std;

// snagging dgesv from lapack
extern "C" void dgesv_( int *n, int *nrhs, double  *a, int *lda, int *ipiv, double *b, int *lbd, int *info );
// snagging backsub from lapack
extern "C" void dgetrs_( char *COND, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info );

void pbj_ludecomp(vector<double> &A, vector<double> &b, problem_space ps){
    /*breif: Solves individual dense cell matrices in parrallel on cpu
    requires -fopenmp to copmile.

    A and b store all matrices in col major in a single std:vector as
    offsets from one another*/

    // parallelized over the number of cells
    #pragma omp parallel for
    for (int i=0; i<ps.N_cells; ++i){
        // lapack variables for a single cell (col major!)
        int nrhs = 1; // one column in b
        int lda = ps.SIZE_cellBlocks; // leading A dim for col major
        int ldb_col = ps.SIZE_cellBlocks; // leading b dim for col major
        std::vector<int> ipiv_par(ps.SIZE_cellBlocks); // pivot vector
        int Npbj = ps.SIZE_cellBlocks; // size of problem
        int info;

        // solve Ax=b in a cell
        dgesv_( &Npbj, &nrhs, &A[i*ps.ELEM_cellBlocks], &lda, &ipiv_par[0], &b[i*ps.SIZE_cellBlocks], &ldb_col, &info );

        if( info > 0 ) {
            printf( "\n>>>PBJ LINALG ERROR<<<\n" );
            printf( "The diagonal element of the triangular factor of A,\n" );
            printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
            printf( "the solution could not be computed.\n" );
        exit( 1 );
        }
    }
}

void pbj_backsub(vector<double> &A, vector<double> &b, problem_space ps){
    #pragma omp parallel for
    for (int i=0; i<ps.N_cells; ++i){
        // lapack variables for a single cell (col major!)
        int nrhs = 1; // one column in b
        int lda = ps.SIZE_cellBlocks; // leading A dim for col major
        int ldb_col = ps.SIZE_cellBlocks; // leading b dim for col major
        std::vector<int> ipiv_par(ps.SIZE_cellBlocks); // pivot vector
        int Npbj = ps.SIZE_cellBlocks; // size of problem
        int info;

        // solve Ax=b in a cell
        dgetrs_( "N", &Npbj, &nrhs, &A[i*ps.ELEM_cellBlocks], &lda, &ipiv_par[0], &b[i*ps.SIZE_cellBlocks], &ldb_col, &info );

        if( info > 0 ) {
            printf( "\n>>>PBJ LINALG ERROR<<<\n" );
            printf( "The diagonal element of the triangular factor of A,\n" );
            printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
            printf( "the solution could not be computed.\n" );
        exit( 1 );
        }
    }
}