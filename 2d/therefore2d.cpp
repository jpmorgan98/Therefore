#include <iostream>
#include <vector>
#include "../common/util.h"

#include "base_mats.h"
#include "solvers.cpp"

//#include "time_step.cpp"


const bool print_mats = false;
const bool debug_print = false;
const bool cycle_print = true;
const bool print_title_card = false;
const bool save_output = false;

using namespace std;

void d2_quadrature_expantion(int N, vector<double> reduced, vector<double> mu, vector<double> eta, vector<double> weight){
    int N_expanded = (N*(N+2)/2);
    vector<double> expanded;

    for (int i=0; i<N; ++i){
        for (int j=0; j<N; ++j){
            mu[i*N+j]       = reduced[i];
            eta[i*N+j]      = reduced[j];
            weight[i*N+j]  = reduced[N];
        }
    }
}


int main(){

    // TODO: build and initalize problem 
    problem_space ps;

    // build A
    vector<double> A(ps.ELEM_A);
    build_A(A, ps);

    // build b const
    vector<double> b( ps.SIZE_aflux );
    build_b_const(b, ps);

    bool converged = false;
    int iter = 0;

    // iteration parameters
    double spec_rad_est = 1;
    double error;
    double error_n1;
    vector<double> aflux_last( ps.SIZE_aflux );
    
    vector<double> reduced(ps.N_angles);

    // S4
    

    while ( !converged ) {
        //build b var from last iteration
        build_b_var(b, ps);

        //solve Ax=b
        if (iter==0){
            // solve full system first try!
            pbj_ludecomp(A, b, ps);
        } else {
            // back substitute only! (using previous decomp)
            pbj_backsub(A, b, ps);
        }

        // compute difference error
        error = infNorm_error(aflux_last, b);

        // copmute spectral radius if enough iteration counts
        spec_rad_est = error/error_n1;


        // check max iteration
        if (iter>ps.max_iter){
            converged = true;
            cout << "WANRING: Simulation did not converge after max iterations" << endl;
        }

        // check convergence
        if ( ps.tol*(1-spec_rad_est) > error ){
            converged = true;
        }

        // increase iteration
        iter++;
        error_n1 = error;
        aflux_last = b;
    }

    return(1);
}

