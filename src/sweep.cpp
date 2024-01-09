#include <iostream>
#include <vector>
#include "base_mats.h"
#include "legendre.h"


void sweep(std::vector<double> af_last, std::vector<double> af_prev, std::vector<cell> cells, problem_space ps){
    for (int j=0; j<ps.N_angles; ++j){
        for (int g=0; g<ps.N_groups; ++g){

            if (ps.angles[j] < 0){ // negative sweep

                for (int i=0; i<ps.N_cells; ++i){
                    int helper_index;

                    std::vector<double> A_small = A_neg_rm(cells[i], ps.angles[j], g);
                    
                }

            } else if (ps.angles[j] > 0) { // positive sweep
            
                for (int i=0; i<ps.N_cells; ++i){
                    int helper_index;


                    int i_l = int(2*i);
                    int i_r = int(2*i+1);

                    // index corresponding to this position last time step
                    double af_hl_L = angular_flux_previous[];
                    double af_hl_R = angular_flux_previous[];
                    double af_L;
                    double af_hn_L;

                    if (i == 0){
                        af_L     = BCl[];
                        af_hn_L  = BCl[];
                    } else {
                        af_L     = angular_flux_next[];
                        af_hn_L  = angular_flux_mid_next[];
                    }

                    std::vector<double> A_small = A_pos_rm(cells[i], ps.angles[j], g);
                    std::vector<double> b_small = b_pos(cells[i], g, ps.angles[j], j, af_hl_L, af_hl_R, af_L, af_hn_L)

                    
                    
                }

            }
        }
    }
}