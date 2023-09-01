#include <iostream>
#include <vector>

#include "class.h"

// ROW MAJOR!!!!!!!!!


std::vector<double> A_neg_rm(cell cell, double mu, int group){
    double gamma = (cell.dx*cell.xsec_total[group])/2;
    double timer = cell.dx/(cell.v[group]*cell.dt);
    double timer2 = cell.dx/(2*cell.v[group]*cell.dt);
    double a = mu/2;

    std::vector<double> A_n = {-a + gamma, a,          timer2,            0,
                               -a,        -a + gamma,  0,                 timer2,
                               -timer,     0,          timer - a + gamma, a,
                                0,        -timer,     -a,                 timer -a + gamma};
    
    return(A_n);
    }

std::vector<double> A_pos_rm(cell cell, double mu, int group){
    double gamma = (cell.dx*cell.xsec_total[group])/2;
    double timer = cell.dx/(cell.v[group]*cell.dt);
    double timer2 = cell.dx/(2*cell.v[group]*cell.dt);
    double a = mu/2;

    std::vector<double> A_p = {a + gamma, a,         timer2,            0,
                              -a,         a + gamma, 0,                 timer2,
                              -timer,     0,         timer + a + gamma, a,
                               0,        -timer,    -a,                 timer +a + gamma};

    return(A_p);
    }

std::vector<double> scatter(double dx, double xsec_scatter, std::vector<double> w, int N){
    std::vector<double> S ((4*N*4*N));
    double beta = dx*xsec_scatter/4;
    
    for (int ca=0; ca<N; ca++){
        for (int ra=0; ra<N; ra++){

            S[4*ra+0 + 4*4*ca*N + 0*N*4] = beta*w[ra];
            S[4*ra+1 + 4*4*ca*N + 1*N*4] = beta*w[ra];
            S[4*ra+2 + 4*4*ca*N + 2*N*4] = beta*w[ra];
            S[4*ra+3 + 4*4*ca*N + 3*N*4] = beta*w[ra];
        }
    }

    return(S);
}

std::vector<double> b_pos(cell const &cell, int group, double mu, int angle, double af_hl_L, double af_hl_R, double af_L, double af_hn_L){

    double timer2 = cell.dx/(2*cell.v[group] * cell.dt);

    // explanation on indicies of Q located in cell class
    std::vector<double> b_pos = {cell.dx/4*cell.Q[0 + 4*angle + group*4*cell.N_angle] + timer2*af_hl_L + mu* af_L,
                                 cell.dx/4*cell.Q[1 + 4*angle + group*4*cell.N_angle] + timer2*af_hl_R,
                                 cell.dx/4*cell.Q[2 + 4*angle + group*4*cell.N_angle] + mu*af_hn_L,
                                 cell.dx/4*cell.Q[3 + 4*angle + group*4*cell.N_angle]};

    return(b_pos);

}
//const &
std::vector<double> b_neg(cell const &cell, int group, double mu, int angle, double af_hl_L, double af_hl_R, double af_R, double af_hn_R){

    double timer2 = cell.dx/(2*cell.v[group] * cell.dt);

    std::vector<double> b_neg ={cell.dx/4*cell.Q[0 + 4*angle + group*4*cell.N_angle] + timer2*af_hl_L,
                                cell.dx/4*cell.Q[1 + 4*angle + group*4*cell.N_angle] + timer2*af_hl_R - mu* af_R,
                                cell.dx/4*cell.Q[2 + 4*angle + group*4*cell.N_angle],
                                cell.dx/4*cell.Q[3 + 4*angle + group*4*cell.N_angle] - mu*af_hn_R};

    return(b_neg);
}
