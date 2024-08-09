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
void b_gen_const_win_iter(std::vector<double> &b, std::vector<double> &aflux_previous, std::vector<cell> cells, problem_space ps);
void b_gen_var_win_iter(std::vector<double> &b, std::vector<double> &aflux_last, problem_space ps);

void b_gen(std::vector<double> &b, std::vector<double> &aflux_previous, std::vector<double> &aflux_last, std::vector<cell> cells, problem_space ps){
    //brief: builds b

    vector<double> b_small;

    // helper index
    int index_start;
    int index_start_n1;
    int index_start_p1;

    for (int i=0; i<ps.N_cells; i++){
        
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



void b_gen_const_win_iter(std::vector<double> &b, std::vector<double> &aflux_previous, std::vector<cell> cells, problem_space ps){
    //brief: builds b

    vector<double> b_small;

    // helper index
    int index_start;
    int index_start_n1;
    int index_start_p1;

    for (int i=0; i<ps.N_cells; i++){
        
        for (int g=0; g<ps.N_groups; g++){
            
            for (int j=0; j<ps.N_angles; j++){

                // the first index in the smallest chunk of 4
                index_start = (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);
                // 4 blocks organized af_l, af_r, af_hn_l, af_hn_r

                // angular flux from the k-1+1/2 from within the cell
                double af_hl_l = aflux_previous[index_start+2];
                double af_hl_r = aflux_previous[index_start+3];

                // negative angle
                if (ps.angles[j] < 0){
                    //cell &cell, int group, double mu, int angle, double af_hl_L, double af_hl_R, double af_R, double af_hn_R
                    b_small = b_neg_const_win_itteration(cells[i], g, j, af_hl_l, af_hl_r);

                // positive angles
                } else {
                     //b_pos_const_win_itteration( cell &cell, int group, int angle, double af_hl_L, double af_hl_R )
                    b_small = b_pos_const_win_itteration(cells[i], g, j, af_hl_l, af_hl_r);
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


void b_gen_const_win_iter_ng2g(std::vector<double> &b, std::vector<double> &aflux_previous, std::vector<cell> cells, problem_space ps){
    //brief: builds b

    vector<double> b_small;

    // helper index
    int index_start;
    int index_start_n1;
    int index_start_p1;

    for (int i=0; i<ps.N_cells; i++){
        
        for (int g=0; g<ps.N_groups; g++){
            
            for (int j=0; j<ps.N_angles; j++){

                // the first index in the smallest chunk of 4
                index_start = (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);
                // 4 blocks organized af_l, af_r, af_hn_l, af_hn_r

                // angular flux from the k-1+1/2 from within the cell
                double af_hl_l = aflux_previous[index_start+2];
                double af_hl_r = aflux_previous[index_start+3];

                // negative angle
                if (ps.angles[j] < 0){
                    //cell &cell, int group, double mu, int angle, double af_hl_L, double af_hl_R, double af_R, double af_hn_R
                    b_small = b_neg_const_win_itteration(cells[i], g, j, af_hl_l, af_hl_r);

                // positive angles
                } else {
                     //b_pos_const_win_itteration( cell &cell, int group, int angle, double af_hl_L, double af_hl_R )
                    b_small = b_pos_const_win_itteration(cells[i], g, j, af_hl_l, af_hl_r);
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


void b_gen_var_win_iter(std::vector<double> &b, std::vector<double> &aflux_last, problem_space ps){
    //brief: builds b

    // helper index
    int index_start;
    int index_start_n1;
    int index_start_p1;

    for (int i=0; i<ps.N_cells; i++){
        for (int g=0; g<ps.N_groups; g++){
            for (int j=0; j<ps.N_angles; j++){

                // the first index in the smallest chunk of 4
                index_start = (i*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*j);
                // 4 blocks organized af_l, af_r, af_hn_l, af_hn_r

                // negative angle
                if (ps.angles[j] < 0){
                    if (i == ps.N_cells-1){ // right boundary condition
                        
                        b[index_start+1] -= ps.angles[j]*ps.boundary_condition(1,g,j,0);
                        b[index_start+3] -= ps.angles[j]*ps.boundary_condition(1,g,j,1);

                    } else { // pulling information from right to left
                        index_start_p1 = index_start + ps.SIZE_cellBlocks;

                        //outofbounds_check(index_start_p1, aflux_last);
                        //outofbounds_check(index_start_p1+2, aflux_last);

                        b[index_start+1] -= ps.angles[j]*aflux_last[index_start_p1];
                        b[index_start+3] -= ps.angles[j]*aflux_last[index_start_p1+2];
                    }

                // positive angles
                } else {
                    if (i == 0){ // left boundary condition
                        
                        b[index_start]    += ps.angles[j]*ps.boundary_condition(0,g,j,0);
                        b[index_start+2]  += ps.angles[j]*ps.boundary_condition(0,g,j,1);

                    } else { // pulling information from left to right
                        index_start_n1 = index_start - ps.SIZE_cellBlocks;

                        //outofbounds_check(index_start_n1+1, aflux_last);
                        //outofbounds_check(index_start_n1+3, aflux_last);

                        b[index_start]    += ps.angles[j]*aflux_last[index_start_n1+1];
                        b[index_start+2]  += ps.angles[j]*aflux_last[index_start_n1+3];
                    }
                }
            }
        }
    }
}






void A_gen_sparse(std::vector<double> &A, std::vector<cell> cells, problem_space ps){
    /*breif: only non zero elements of the array are stored in a single array of total size
    (N_an*N_group*4)**2*N_cells. A cell block soultion is then stored in a column
    major where the leading value is an offset from */
    
    for (int i=0; i<ps.N_cells; i++){
        
        vector<double> A_c_rm(ps.ELEM_cellBlocks, 0.0);
        A_c_gen(i, A_c_rm, cells, ps);
        std::vector<double> A_c_cm = row2colSq(A_c_rm);

        int index_start = i*ps.ELEM_cellBlocks;

        for (int r=0; r<ps.ELEM_cellBlocks; ++r){
            A[index_start+r] = A_c_cm[r];
        }

    }
}


void A_c_gen_ng2g(int i, double* A_c, std::vector<cell> cells, problem_space ps){
    /*
    brief: assembles a coefficient matrix within all groups and angles in a cell
    NOTE: ROW MAJOR FORMAT
    */

   int size_mat = pow(ps.N_angles*4, 2);

    for (int g=0; g<ps.N_groups; g++){

        int index_start = g*size_mat;
        int Adim_angle = 4*ps.N_angles; 

        //vector<double> A_c_g(4*ps.N_angles * 4*ps.N_angles);
        vector<double> A_c_g_a(4*4);
        vector<double> temp(4*4);
        

        for (int j=0; j<ps.N_angles; j++){
            if (ps.angles[j] > 0){
                temp = A_pos_rm(cells[i], ps.angles[j], g);
                A_c_g_a = row2colSq(temp);
            } else {
                temp = A_neg_rm(cells[i], ps.angles[j], g);
                A_c_g_a = row2colSq(temp);
            }
            //std::vector<double> A_c_cm = row2colSq(A_c_rm);
            index_start2 = index_start + j*16*ps.N_angle + j*4;
            // push it into an all angle
            for (int c=0; c<4; ++c){ // moving col major
            for (int r=0; r<16; r++){
                A_c[c + r*4*N_angle + index_start2] = A_c_g_a[c + r*4]; 
            }}
        }

        vector<double> temp2(size_mat);
        vector<double> scatter_mat(size_mat);

        int index_start = 0;
        //int index_start = 4*ps.N_angles*gp + 4*4*ps.N_angles*ps.N_angles*ps.N_groups
        
        temp2 = scatter(cells[i].dx, cells[i].xsec_scatter[g], ps.weights, ps.N_angles);
        scatter_mat = row2colSq(temp2);

        // indexing from Angle blocks within a group to cell blocks of all groups
        for (int r=0; r<size_mat r++){ 
            A_c[index_start + r] -= g2gp_scatter[r];
        }
    }
}

void A_gen_sparse_ng2g(std::vector<double> &A, std::vector<cell> cells, problem_space ps){
    /*breif: only non zero elements of the array are stored in a single array of total size
    (N_an*N_group*4)**2*N_cells. A cell block soultion is then stored in a column
    major where the leading value is an offset from */
    
    for (int i=0; i<ps.N_cells; i++){
        A_c_gen(i, A_c_rm, cells, ps);
        std::vector<double> A_c_cm = row2colSq(A_c_rm);

        int index_start = i*ps.ELEM_cellBlocks;

        for (int r=0; r<ps.ELEM_cellBlocks; ++r){
            A[index_start+r] = A_c_cm[r];
        }
    }

}


void A_gen(std::vector<double> &A, std::vector<cell> cells, problem_space ps){ 
    /*FOR DEBUGING ONLY! 
    Produces whole ass sparse matrix structure for PBJ method zeros and all in
    a dense formulation. Total size (N_an*N_group*4*N_cells)**2*/

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

    for (int g=0; g<ps.N_groups; g++){

        int index_start = 4*g*ps.N_angles *4*ps.N_angles*ps.N_groups + 4*g*ps.N_angles;

        int Adim_angle = 4*ps.N_angles; 

        vector<double> A_c_g(4*ps.N_angles * 4*ps.N_angles);
        vector<double> A_c_g_a(4*4);
        

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

        for (int r=0; r<Adim_angle; r++){
                for (int c=0; c<Adim_angle; c++){
                    int id_c_g = index_start + r*(Adim_angle*ps.N_groups) + c;
                    int id_group = Adim_angle*r + c;

                    A_c[id_c_g] = A_c_g[id_group];
            }
        }
    }

    //print_rm(A_c);

    //down scattering look the same just with an off axis terms

    // Scattering looks like row major allined std::vector<doubles> 
    // g->g'
    //     _       g'       _
    //    | 0->0  0->1  0->2 |  fastest
    //  g | 1->0  1->1  1->2 |     |
    //    | 2->0  2->1  2->2 |     \/
    //    -                 -   slowest
    //  Thus the diagnol is the within group scttering

    for (int g=0; g<ps.N_groups; ++g){ //g problem with this loop!!!
        for (int gp=0; gp<ps.N_groups; ++gp){ //g'

            vector<double> g2gp_scatter(4*ps.N_angles * 4*ps.N_angles);

            int index_start = 4*g*ps.N_angles *4*ps.N_angles*ps.N_groups + 4*gp*ps.N_angles;
            //int index_start = 4*ps.N_angles*gp + 4*4*ps.N_angles*ps.N_angles*ps.N_groups

            int Adim_angle = 4*ps.N_angles; 
            
            g2gp_scatter = scatter(cells[i].dx, cells[i].xsec_scatter[gp + ps.N_groups*g], ps.weights, ps.N_angles);

            // indexing from Angle blocks within a group to cell blocks of all groups
            for (int r=0; r<Adim_angle; r++){ //row
                for (int c=0; c<Adim_angle; c++){ //col 
                    int id_group = Adim_angle*r + c;

                    int id_c_g = index_start + r*(Adim_angle*ps.N_groups) + c;

                    // [putting scattering in the big one]
                    A_c[id_c_g] -= g2gp_scatter[id_group];
                }
            }
        }
    }

    //print_rm(A_c);
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




