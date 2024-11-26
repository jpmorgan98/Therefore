using namespace std;

#include <iostream>
#include <vector>
#include "class.h"
#include "../common/util.h"


void build_A(vector<double> &A, problem_space ps){
    for (int i=0; i<ps.N_cells; ++i){
        A_ijk(A, ps, i);
    }
}

// if needed this can be parallelized!
void A_ijk(vector<double> &A, problem_space ps, int region_id){ 
    //builds a matrix for a given cell of material_id composition

    for (int m=0; m<ps.N_angles; ++m){
        int OFFSET = ps.SIZE_angleBlocks*ps.SIZE_angleBlocks*m;

        double gamma = abs(ps.eta[m]/2) + abs(ps.mu[m]/2) + ps.dx*ps.dy*ps.sigma[region_id];

        vector<double> Am = A_m(gamma, ps, m, region_id);

        copy(Am.begin(), Am.end(), A.begin() + OFFSET);
    }

    S(A, ps, region_id);
}

vector<double> A_m(double gamma, problem_space ps, int m, int region_id){
    double timer =  ps.dx*ps.dy/(4*ps.v[region_id]*ps.dt);
    double timert = ps.dx*ps.dy/(2*ps.v[region_id]*ps.dt);

    vector<double> Am = {
        gamma, ps.mu[m]/2, -ps.eta[m]/2, 0, timer, 0, 0, 0,
        -ps.mu[m]/2, gamma, 0, -ps.eta[m]/2, 0, timer, 0, 0,
        ps.eta[m]/2, 0, gamma, ps.mu[m]/2, 0, 0, timer, 0,
        0, ps.eta[m]/2, -ps.mu[m]/2, gamma, 0, 0, 0, timer,
        -timert, 0, 0, 0, timert + gamma, ps.mu[m]/2, -ps.eta[m]/2, 0,
        0, -timert, 0, 0, -ps.mu[m]/2, timert + gamma, 0, -ps.eta[m]/2,
        0, 0, -timert, 0, ps.eta[m]/2, 0, timert + gamma, ps.mu[m]/2,
        0, 0, 0, -timert, 0, ps.eta[m]/2, -ps.mu[m]/2, timert + gamma,
    };

    vector<double> Amnext = row2colSq(Am);

    return(Amnext);
}

void S(vector<double> &A, problem_space ps, int region_id){
    double beta = ps.dx * ps.dy * ps.sigma_s[region_id]/4;
    
    for (int ca=0; ca<ps.N_angles; ca++){
        for (int ra=0; ra<ps.N_angles; ra++){
            for (int aa=0; aa<ps.N_sub; ++aa)
                A[4*ra+aa + 4*4*ca*ps.N_angles + aa*ps.N_angles*4] -= beta*ps.weights[ra];
        }
    }
}

void b_const(vector<double> b, vector<double> af_previous, problem_space ps, int region_id){
    // building the constant part of b
    // previous refers to previous in time
    // last refers to last iteration

    double timer = ps.dx*ps.dy/(4*ps.v[region_id]*ps.dt);

    int OFFSET ;
    b[OFFSET+0] = timer * af_previous[OFFSET+4] + ps.material_source[region_id+0];
    b[OFFSET+1] = timer * af_previous[OFFSET+5] + ps.material_source[region_id+1];
    b[OFFSET+2] = timer * af_previous[OFFSET+6] + ps.material_source[region_id+2];
    b[OFFSET+3] = timer * af_previous[OFFSET+7] + ps.material_source[region_id+3];

    b[OFFSET+4] = ps.material_source[region_id+4];
    b[OFFSET+5] = ps.material_source[region_id+5];
    b[OFFSET+6] = ps.material_source[region_id+6];
    b[OFFSET+7] = ps.material_source[region_id+7];
}

void build_b_const(vector<double> b, problem_space ps){

}


void build_b_var(vector<double> b, problem_space ps){

}


void b_pp(vector<double> &b, vector<double> &af_last, problem_space ps, cell cur_cell, int m){
    // b^{+\mu, +\eta}
    if (cur_cell.boundary) {
        cur_cell.get_boundary_cond();
    } else {
        b[0] += ps.mu[m] * af_last[ ps.get_af_index(1,m,0,cur_cell.i-1,cur_cell.j) ]; //^{NE}_{m,k,i-1,j}
        b[1] += 0;
        b[2] += ps.mu[m] * af_last[ ps.get_af_index(3,m,0,cur_cell.i-1, cur_cell.j) ] + ps.eta[m] * af_last[ ps.get_af_index(0,m,0,cur_cell.i, cur_cell.j-1) ]; //^{SE}_{m,k,i-1,j} ^{NW}_{m,k,i,j-1
        b[3] += ps.eta[m] * af_last[ ps.get_af_index(1,m,0,cur_cell.i, cur_cell.j-1) ];//^{NE}_{m,k,i,j-1}

        b[4] += ps.mu[m] * af_last[ ps.get_af_index(1,m,1,cur_cell.i-1, cur_cell.j) ];//^{NE}_{m,k+1/2,i-1,j}
        b[5] += 0;
        b[6] += ps.mu[m] * af_last[ ps.get_af_index(3,m,1,cur_cell.i-1, cur_cell.j) ]  + ps.eta[m] * af_last[ ps.get_af_index(1,m,1,cur_cell.i, cur_cell.j-1) ]; //^{SE}_{m,k+1/2,i-1,j} ^{NW}_{m,k+1/2,i,j-1}
        b[7] += ps.eta[m]* af_last[ ps.get_af_index(1,m,1,cur_cell.i, cur_cell.j-1) ]; //^{NE}_{m,k+1/2,i,j-1}
    }
}

void b_pn(vector<double> &b, vector<double> &af_last, problem_space ps, cell cur_cell, int m){
    // b^{+\mu, -\eta}
    if (cur_cell.boundary) {
        cur_cell.get_boundary_cond();
    } else {
        b[0] += ps.mu[m] * af_last[ ps.get_af_index(2,m,1,cur_cell.i-1,cur_cell.j) ]; //^{NE}_{m,k,i-1,j}
        b[1] += 0;
        b[2] += ps.mu[m] * af_last[ ps.get_af_index(4,m,1,cur_cell.i-1, cur_cell.j) ] + ps.eta[m] * af_last[ ps.get_af_index(1,m,1,cur_cell.i, cur_cell.j-1) ]; //^{SE}_{m,k,i-1,j} ^{NW}_{m,k,i,j-1
        b[3] += ps.eta[m] * af_last[ ps.get_af_index(2,m,1,cur_cell.i, cur_cell.j-1) ];//^{NE}_{m,k,i,j-1}

        b[4] += ps.mu[m] * af_last[ ps.get_af_index(2,m,2,cur_cell.i-1, cur_cell.j) ];//^{NE}_{m,k+1/2,i-1,j}
        b[5] += 0;
        b[6] += ps.mu[m] * af_last[ ps.get_af_index(4,m,2,cur_cell.i-1, cur_cell.j) ]  + ps.eta[m] * af_last[ ps.get_af_index(2,m,1,cur_cell.i, cur_cell.j-1) ]; //^{SE}_{m,k+1/2,i-1,j} ^{NW}_{m,k+1/2,i,j-1}
        b[7] += ps.eta[m]* af_last[ ps.get_af_index(2,m,2,cur_cell.i, cur_cell.j-1) ]; //^{NE}_{m,k+1/2,i,j-1}
    }
}

void b_np(vector<double> &b, vector<double> &af_last, problem_space ps, cell cur_cell, int m){
    // b^{-\mu, +\eta}
    if (cur_cell.boundary) {
        cur_cell.get_boundary_cond();
    } else {
        b[0] += ps.mu[m] * af_last[ ps.get_af_index(2,m,1,cur_cell.i-1,cur_cell.j) ]; //^{NE}_{m,k,i-1,j}
        b[1] += 0;
        b[2] += ps.mu[m] * af_last[ ps.get_af_index(4,m,1,cur_cell.i-1, cur_cell.j) ] + ps.eta[m] * af_last[ ps.get_af_index(1,m,1,cur_cell.i, cur_cell.j-1) ]; //^{SE}_{m,k,i-1,j} ^{NW}_{m,k,i,j-1
        b[3] += ps.eta[m] * af_last[ ps.get_af_index(2,m,1,cur_cell.i, cur_cell.j-1) ];//^{NE}_{m,k,i,j-1}

        b[4] += ps.mu[m] * af_last[ ps.get_af_index(2,m,2,cur_cell.i-1, cur_cell.j) ];//^{NE}_{m,k+1/2,i-1,j}
        b[5] += 0;
        b[6] += ps.mu[m] * af_last[ ps.get_af_index(4,m,2,cur_cell.i-1, cur_cell.j) ]  + ps.eta[m] * af_last[ ps.get_af_index(2,m,1,cur_cell.i, cur_cell.j-1) ]; //^{SE}_{m,k+1/2,i-1,j} ^{NW}_{m,k+1/2,i,j-1}
        b[7] += ps.eta[m]* af_last[ ps.get_af_index(2,m,2,cur_cell.i, cur_cell.j-1) ]; //^{NE}_{m,k+1/2,i,j-1}
    }
}

void b_nn(vector<double> &b, vector<double> &af_last, problem_space ps, cell cur_cell, int m){
    // b^{-mu, -\eta}
    if (cur_cell.boundary) {
        cur_cell.get_boundary_cond();
    } else {
        b[0] += ps.mu[m] * af_last[ ps.get_af_index(2,m,1,cur_cell.i-1,cur_cell.j) ]; //^{NE}_{m,k,i-1,j}
        b[1] += 0;
        b[2] += ps.mu[m] * af_last[ ps.get_af_index(4,m,1,cur_cell.i-1, cur_cell.j) ] + ps.eta[m] * af_last[ ps.get_af_index(1,m,1,cur_cell.i, cur_cell.j-1) ]; //^{SE}_{m,k,i-1,j} ^{NW}_{m,k,i,j-1
        b[3] += ps.eta[m] * af_last[ ps.get_af_index(2,m,1,cur_cell.i, cur_cell.j-1) ];//^{NE}_{m,k,i,j-1}

        b[4] += ps.mu[m] * af_last[ ps.get_af_index(2,m,2,cur_cell.i-1, cur_cell.j) ];//^{NE}_{m,k+1/2,i-1,j}
        b[5] += 0;
        b[6] += ps.mu[m] * af_last[ ps.get_af_index(4,m,2,cur_cell.i-1, cur_cell.j) ]  + ps.eta[m] * af_last[ ps.get_af_index(2,m,1,cur_cell.i, cur_cell.j-1) ]; //^{SE}_{m,k+1/2,i-1,j} ^{NW}_{m,k+1/2,i,j-1}
        b[7] += ps.eta[m]* af_last[ ps.get_af_index(2,m,2,cur_cell.i, cur_cell.j-1) ]; //^{NE}_{m,k+1/2,i,j-1}
    }
}