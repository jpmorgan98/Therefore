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

std::vector<double> b_pos(cell cell, int group, double mu, int angle, double af_hl_L, double af_hl_R, double af_L, double af_hn_L){

    double timer2 = cell.dx/(2*cell.v[group] * cell.dt);

    int helper = 4*angle + group*4*cell.N_angle;

    outofbounds_check(0 + helper, cell.Q);
    outofbounds_check(1 + helper, cell.Q);
    outofbounds_check(2 + helper, cell.Q);
    outofbounds_check(3 + helper, cell.Q);

    // explanation on indicies of Q located in cell class
    std::vector<double> b_pos = {cell.dx/4*cell.Q[0 + 4*angle + group*4*cell.N_angle] + timer2*af_hl_L + mu* af_L,
                                 cell.dx/4*cell.Q[1 + 4*angle + group*4*cell.N_angle] + timer2*af_hl_R,
                                 cell.dx/4*cell.Q[2 + 4*angle + group*4*cell.N_angle] + mu*af_hn_L,
                                 cell.dx/4*cell.Q[3 + 4*angle + group*4*cell.N_angle]};

    return(b_pos);

}
//const &
std::vector<double> b_neg(cell &cell, int group, double mu, int angle, double af_hl_L, double af_hl_R, double af_R, double af_hn_R){

    double timer2 = cell.dx/(2*cell.v[group] * cell.dt);

    int helper = 4*angle + group*4*cell.N_angle;

    if (helper > cell.Q.size() ){
        cout<<">>>ERROR: Source out of bounds!<<<"<<endl;
        cout<<angle<<endl;
        cout<<group<<endl;
        cout<<cell.N_angle<<endl;
    }

    outofbounds_check(0 + helper, cell.Q);
    outofbounds_check(1 + helper, cell.Q);
    outofbounds_check(2 + helper, cell.Q);
    outofbounds_check(3 + helper, cell.Q);



    std::vector<double> b_neg ={cell.dx/4*cell.Q[0 + helper] + timer2*af_hl_L,
                                cell.dx/4*cell.Q[1 + helper] + timer2*af_hl_R - mu* af_R,
                                cell.dx/4*cell.Q[2 + helper],
                                cell.dx/4*cell.Q[3 + helper] - mu*af_hn_R};

    return(b_neg);
}



// gpu functions
std::vector<double> b_pos_const_win_itteration( cell cell, int group, int angle, double af_hl_L, double af_hl_R ){

    double timer2 = cell.dx/(2*cell.v[group] * cell.dt);

    int helper = 4*angle + group*4*cell.N_angle;

    outofbounds_check(0 + helper, cell.Q);
    outofbounds_check(1 + helper, cell.Q);
    outofbounds_check(2 + helper, cell.Q);
    outofbounds_check(3 + helper, cell.Q);

    // explanation on indicies of Q located in cells class
    std::vector<double> b_pos = {cell.dx/4*cell.Q[0 + 4*angle + group*4*cell.N_angle] + timer2*af_hl_L,
                                 cell.dx/4*cell.Q[1 + 4*angle + group*4*cell.N_angle] + timer2*af_hl_R,
                                 cell.dx/4*cell.Q[2 + 4*angle + group*4*cell.N_angle],
                                 cell.dx/4*cell.Q[3 + 4*angle + group*4*cell.N_angle]};

    return(b_pos);

}

std::vector<double> b_neg_const_win_itteration( cell &cell, int group, int angle, double af_hl_L, double af_hl_R ){

    double timer2 = cell.dx/(2*cell.v[group] * cell.dt);

    int helper = 4*angle + group*4*cell.N_angle;

    outofbounds_check(0 + helper, cell.Q);
    outofbounds_check(1 + helper, cell.Q);
    outofbounds_check(2 + helper, cell.Q);
    outofbounds_check(3 + helper, cell.Q);

    std::vector<double> b_neg ={cell.dx/4*cell.Q[0 + helper] + timer2*af_hl_L,
                                cell.dx/4*cell.Q[1 + helper] + timer2*af_hl_R,
                                cell.dx/4*cell.Q[2 + helper],
                                cell.dx/4*cell.Q[3 + helper]};

    return(b_neg);
}





void b_pos_var_win_itteration( std::vector<double> &b, double mu, double af_L, double af_hn_L ){

    b[0] += mu*af_L;
    b[2] += mu*af_hn_L;

}

void b_neg_var_win_itteration( std::vector<double> &b, double mu, double af_R, double af_hn_R ){

    b[1] -= mu*af_R;
    b[3] -= mu*af_hn_R;
}

//cells[i], g, ps.angles[j], j, sf, index_sf, af_hl_L, af_hl_R, af_LB, af_hn_LB
std::vector<double> c_pos(cell &cell, int group, double mu, int angle, std::vector<double> &sf, int offset, double af_hl_L, double af_hl_R, double af_LB, double af_hn_LB){
    /*
    hl is previous time step info
    LB is sweep info
    sf is previous scatter source info
    */

    double timer2 = cell.dx/(2*cell.v[group] * cell.dt);

    int helper = group*4;

    outofbounds_check(0 + helper, cell.Q);
    outofbounds_check(1 + helper, cell.Q);
    outofbounds_check(2 + helper, cell.Q);
    outofbounds_check(3 + helper, cell.Q);

    outofbounds_check(offset+0, sf);
    outofbounds_check(offset+1, sf);
    outofbounds_check(offset+2, sf);
    outofbounds_check(offset+3, sf);

    outofbounds_check(group*group, cell.xsec_g2g_scatter);
 
    std::vector<double> c_pos ={cell.dx/4*(cell.xsec_scatter[group]*sf[offset+0] + cell.Q[0+helper]) + timer2*af_hl_L + mu*af_LB,
                                cell.dx/4*(cell.xsec_scatter[group]*sf[offset+1] + cell.Q[1+helper]) + timer2*af_hl_R,
                                cell.dx/4*(cell.xsec_scatter[group]*sf[offset+2] + cell.Q[2+helper]) + mu*af_hn_LB,
                                cell.dx/4*(cell.xsec_scatter[group]*sf[offset+3] + cell.Q[3+helper])};

    return (c_pos);
}

std::vector<double> c_neg(cell &cell, int group, double mu, int angle, std::vector<double> &sf, int offset, double af_hl_L, double af_hl_R, double af_RB, double af_hn_RB){
    /*
    hl is previous time step info
    LB is sweep info
    sf is previous scatter source info for the whole problem

    All other vectors are not funcitonal expectations
    */

    double timer2 = cell.dx/(2*cell.v[group] * cell.dt);

    int helper = group*4;

    //std::cout << "index Q in c_neg" << std::endl;

    outofbounds_check(0 + helper, cell.Q);
    outofbounds_check(1 + helper, cell.Q);
    outofbounds_check(2 + helper, cell.Q);
    outofbounds_check(3 + helper, cell.Q);

    outofbounds_check(offset+0, sf);
    outofbounds_check(offset+1, sf);
    outofbounds_check(offset+2, sf);
    outofbounds_check(offset+3, sf);

    outofbounds_check(group*group, cell.xsec_g2g_scatter);

    std:vector<double> c_neg = {cell.dx/4*(cell.xsec_scatter[group]*sf[offset+0] + cell.Q[0+helper]) + timer2*af_hl_L,
                                cell.dx/4*(cell.xsec_scatter[group]*sf[offset+1] + cell.Q[1+helper]) + timer2*af_hl_R - mu*af_RB,
                                cell.dx/4*(cell.xsec_scatter[group]*sf[offset+2] + cell.Q[2+helper]) ,
                                cell.dx/4*(cell.xsec_scatter[group]*sf[offset+3] + cell.Q[3+helper]) - mu*af_hn_RB};

    return c_neg;
}




std::vector<double> c_pos_const_win_sweep(cell &cell, int group, double mu, int angle, std::vector<double> &sf, int offset, double af_hl_L, double af_hl_R){
    /*
    hl is previous time step info
    LB is sweep info
    sf is previous scatter source info
    */

    double timer2 = cell.dx/(2*cell.v[group] * cell.dt);

    int helper = group*4;

    outofbounds_check(0 + helper, cell.Q);
    outofbounds_check(1 + helper, cell.Q);
    outofbounds_check(2 + helper, cell.Q);
    outofbounds_check(3 + helper, cell.Q);

    outofbounds_check(offset+0, sf);
    outofbounds_check(offset+1, sf);
    outofbounds_check(offset+2, sf);
    outofbounds_check(offset+3, sf);

    outofbounds_check(group*group, cell.xsec_g2g_scatter);
 
    std::vector<double> c_pos ={cell.dx/4*(cell.xsec_scatter[group]*sf[offset+0] + cell.Q[0+helper]) + timer2*af_hl_L,
                                cell.dx/4*(cell.xsec_scatter[group]*sf[offset+1] + cell.Q[1+helper]) + timer2*af_hl_R,
                                cell.dx/4*(cell.xsec_scatter[group]*sf[offset+2] + cell.Q[2+helper]),
                                cell.dx/4*(cell.xsec_scatter[group]*sf[offset+3] + cell.Q[3+helper])};

    return (c_pos);
}

std::vector<double> c_neg_const_win_sweep(cell &cell, int group, double mu, int angle, std::vector<double> &sf, int offset, double af_hl_L, double af_hl_R){
    /*
    hl is previous time step info
    LB is sweep info
    sf is previous scatter source info for the whole problem

    All other vectors are not funcitonal expectations
    */

    double timer2 = cell.dx/(2*cell.v[group] * cell.dt);

    int helper = group*4;

    //std::cout << "index Q in c_neg" << std::endl;

    outofbounds_check(0 + helper, cell.Q);
    outofbounds_check(1 + helper, cell.Q);
    outofbounds_check(2 + helper, cell.Q);
    outofbounds_check(3 + helper, cell.Q);

    outofbounds_check(offset+0, sf);
    outofbounds_check(offset+1, sf);
    outofbounds_check(offset+2, sf);
    outofbounds_check(offset+3, sf);

    outofbounds_check(group*group, cell.xsec_g2g_scatter);

    std:vector<double> c_neg = {cell.dx/4*(cell.xsec_scatter[group]*sf[offset+0] + cell.Q[0+helper]) + timer2*af_hl_L,
                                cell.dx/4*(cell.xsec_scatter[group]*sf[offset+1] + cell.Q[1+helper]) + timer2*af_hl_R,
                                cell.dx/4*(cell.xsec_scatter[group]*sf[offset+2] + cell.Q[2+helper]) ,
                                cell.dx/4*(cell.xsec_scatter[group]*sf[offset+3] + cell.Q[3+helper])};

    return c_neg;
}