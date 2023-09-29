#include <iostream>
#include <vector>
#include "base_mats.h"
#include "legendre.h"

using namespace std;

// NOTE: ROW MAJOR FORMAT

void b_gen(std::vector<double> &b, std::vector<double> &aflux_previous, std::vector<double> &aflux_last, std::vector<cell> cells, problem_space ps);
void A_c_gen(int i, std::vector<double> &A_c, std::vector<cell> cells, problem_space ps);
void A_gen(std::vector<double> &A, std::vector<cell> cells, problem_space ps);
void quadrature(std::vector<double> &angles, std::vector<double> &weights);
void outofbounds_check(int index, std::vector<double> &vec);

void b_gen(std::vector<double> &b, std::vector<double> &aflux_previous, std::vector<double> &aflux_last, std::vector<cell> cells, problem_space ps){
    //brief: builds b

    vector<double> b_small;

    // helper index
    int index_start;
    int index_start_n1;
    int index_start_p1;

    for (int i=0; i<ps.N_cells; i++){

        if (i == 0){
        } else if (i==ps.N_cells-1){
        }
        
        for (int g=0; g<ps.N_groups; g++){

            // angular fluxes from the right bound (lhs of cell at right) last iteration
            double af_rb;
            // angular fluxes from the left bound (rhs if cell at left) last iteration
            double af_lb;
            // angular fluxes from the right bound (lhs of cell at right) k+1/2 last iteration
            double af_hn_rb;
            // angular fluxes from the left bound (rhs of cell at left) k+1/2 last iteration
            double af_hn_lb;
            
            for (int j=0; j<ps.N_angles; j++){

                // the first index in the smallest chunk of 4
                index_start = (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);
                // 4 blocks organized af_l, af_r, af_hn_l, af_hn_r

                // angular flux from the k-1+1/2 from within the cell
                double af_hl_l = aflux_previous[index_start+2];
                double af_hl_r = aflux_previous[index_start+3];

                // negative angle
                if (ps.angles[j] < 0){
                    if (i == ps.N_cells-1){ // right boundary condition
                        
                        af_rb = ps.boundary_condition(1,g,j,0);
                        af_hn_rb = ps.boundary_condition(1,g,j,1);
                    } else { // pulling information from right to left
                        index_start_p1 = index_start + ps.SIZE_cellBlocks;

                        outofbounds_check(index_start_p1, aflux_last);
                        outofbounds_check(index_start_p1+2, aflux_last);

                        af_rb    = aflux_last[index_start_p1];
                        af_hn_rb = aflux_last[index_start_p1+2];

                        //cout << af_rb << endl;
                        //cout << af_hn_rb << endl;

                    }
                    //cell &cell, int group, double mu, int angle, double af_hl_L, double af_hl_R, double af_R, double af_hn_R
                    b_small = b_neg(cells[i], g, ps.angles[j], j, af_hl_l, af_hl_r, af_rb, af_hn_rb);

                // positive angles
                } else {
                    if (i == 0){ // left boundary condition
                        
                        af_lb    = ps.boundary_condition(0,g,j,0);
                        af_hn_lb = ps.boundary_condition(0,g,j,1);

                    } else { // pulling information from left to right
                        index_start_n1 = index_start - ps.SIZE_cellBlocks;

                        outofbounds_check(index_start_n1+1, aflux_last);
                        outofbounds_check(index_start_n1+3, aflux_last);

                        af_lb    = aflux_last[index_start_n1+1];
                        af_hn_lb = aflux_last[index_start_n1+3];

                    }
                    b_small = b_pos(cells[i], g, ps.angles[j], j, af_hl_l, af_hl_r, af_lb, af_hn_lb);
                }

                outofbounds_check(index_start,   b);
                outofbounds_check(index_start+1, b);
                outofbounds_check(index_start+2, b);
                outofbounds_check(index_start+3, b);
                
                b[index_start]   = b_small[0];
                b[index_start+1] = b_small[1];
                b[index_start+2] = b_small[2];
                b[index_start+3] = b_small[3];
            }
        }
    }
}


void A_gen(std::vector<double> &A, std::vector<cell> cells, problem_space ps){ 

    int dimA_c = 4 * ps.N_groups * ps.N_angles;

    for (int i=0; i<ps.N_cells; i++){
        
        vector<double> A_c(dimA_c*dimA_c, 0.0);

        A_c_gen(i, A_c, cells, ps);

        int A_id_start = dimA_c*ps.N_cells * dimA_c*i + dimA_c*i;

        for (int r=0; r<dimA_c; r++){
            for (int c=0; c<dimA_c; c++){
                int id_wp = A_id_start + r * (dimA_c*ps.N_cells) + c ;
                int id_c = dimA_c*r + c;
                A[id_wp] = A_c[id_c];
            }
        }
    }
}


void A_c_gen(int i, std::vector<double> &A_c, std::vector<cell> cells, problem_space ps){
    /*
    brief: assembles a coefficient matrix within all groups and angles in a cell
    NOTE: ROW MAJOR FORMAT
    */

   bool ds_make = true;

   for (int g=0; g<ps.N_groups; g++){

        vector<double> A_c_g(4*ps.N_angles * 4*ps.N_angles);
        vector<double> A_c_g_a(4*4);
        vector<double> S(4*ps.N_angles * 4*ps.N_angles);
        vector<double> DS(4*ps.N_angles * 4*ps.N_angles);

        for (int j=0; j<ps.N_angles; j++){
            if (ps.angles[j] > 0){
                A_c_g_a = A_pos_rm(cells[i], ps.angles[j], g);
            } else {
                A_c_g_a = A_neg_rm(cells[i], ps.angles[j], g);
            }

            // push it into an all angle cellwise fuck me
            for (int r=0; r<4; r++){
                for (int c=0; c<4; c++){
                    // awfully confusing I know
                    // id_acell = (moves us betten angle blocks) + (moves us to diagonal) + 
                    //            (moves us in rows w/in an angle) + (moves us in col w/in an angle)
                    //             the discritization in angle can only be multiplied by a functional expansion
                    int id_acell  = ((4*ps.N_angles * 4*j) + (4*j) + (4*ps.N_angles)*r + (c));
                    int id_ancell = 4*r + c;
                    A_c_g[id_acell] = A_c_g_a[id_ancell]; 

                    
                }
            }
        }

        // within group scattering 
        S = scatter(cells[i].dx, cells[i].xsec_scatter[g], ps.weights, ps.N_angles);

        // down scattering!!!!
        bool ds_flag = false;
        
        if (g==1){
            
            double xsec_ds = ps.ds;
            //down scattering look the same just with an off axis terms
            DS = scatter(cells[i].dx, xsec_ds, ps.weights, ps.N_angles);
            bool ds_flag = true;
        }

        int index_start = 4*g*ps.N_angles * 4*ps.N_angles*ps.N_groups + 4*g*ps.N_angles;
        int Adim_angle = 4*ps.N_angles; 

        for (int r=0; r<Adim_angle; r++){
            for (int c=0; c<Adim_angle; c++){

                int id_group = Adim_angle*r + c;
                int id_c_g = index_start + r*(Adim_angle*ps.N_groups) + c;

                A_c[id_c_g] = A_c_g[id_group] - S[id_group];

                if (g==1){
                    A_c[id_c_g-4*ps.N_angles] -= DS[id_group];
                }
            }
        }
    }
}


void quadrature(std::vector<double> &angles, std::vector<double> &weights){

    // infred from size of pre-allocated std::vector
    int N_angles = angles.size();

    // allocation for function
    double weights_d[N_angles];
    double angles_d[N_angles];

    // some super-duper fast function that generates everything but in double arrays
    legendre_compute_glr(N_angles, angles_d, weights_d);

    // converting to std::vectors
    for (int i=0; i<N_angles; i++){
        angles[i] = angles_d[i];
        weights[i] = weights_d[i];
    }

}




