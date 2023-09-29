#include "mms.h"
using namespace std;


class cell{
    public:
        int cell_id;
        int region_id;
        int N_angle;
        double x_left;
        double x;
        double x_right;
        std::vector<double> xsec_scatter;
        std::vector<double> xsec_total;
        std::vector<double> v;

        // Q stores non-homogenous terms in rm-form of dims [4 x N_groups x N_angles] where within a group
            // 0   left half cell space integrated, time averaged
            // 1   right half cell space integrated, time averaged
            // 2   left half cell space integrated, time edge
            // 2   right half cell space integrated, time edge
        std::vector<double> Q;

        double dx;
        double dt;
};

class problem_space{
    public:
        // physical space
        int N_cells; 
        int N_angles; 
        int N_time;
        int N_groups;
        int N_mat; 
        int N_rm;
        int time_val;
        double Length;
        double dt;
        double dx;
        double t_max;
        double material_source;
        double velocity;
        double L;

        bool mms_bool;
        mms manSource; // source for the method of manufactured solution
        double ds;

        std::vector<double> weights;
        std::vector<double> angles;

        std::vector<double> af_last;

        // computational
        int hardware_precision;
        double convergence_tolerance = 1e-9;
        bool initialize_from_previous;
        int max_iteration;

        int SIZE_cellBlocks;
        int SIZE_groupBlocks;
        int SIZE_angleBlocks;

        vector<int> boundary_conditions = {0,0};
        vector<double> af_left_bound;
        vector<double> af_right_bound;

        int warn_flag = 0;
        
        double reflectingBC(int side, int group, int angle){
            /*breif: computes a reflecting boundary condition for a given angle and group
                
                assumes that angles are orginized symterically from negative to positive
                 
                the incoming angle is the opsite of what needs to be snet back!
                hence reflecting
            */


            if (side == 0){

                outofbounds_check(group*N_angles + angle, af_left_bound);

                return(af_left_bound[group*N_angles]);

            } else if (side == 1) {

                outofbounds_check(group*N_angles + angle, af_left_bound);

                return(af_right_bound[group*N_angles + angle]);

            } else {

                bound_warn();
                return(0.0);

            }
            
            return(0.5);
        }



        double mms_boundary(int side, int group, int angle, int hn){
            
            if (side==0){
                return(af_left_bound[group*2*N_angles + 2*angle + hn]);
            } else if (side==1) {
                return(af_right_bound[group*2*N_angles + 2*angle + hn]);
            } else {
                bound_warn();
                return(0.0);
            }
        }




        double boundary_condition(int side, int group, int angle, int hn){
            /*breif: computes boundary conditions for a specific group and angle
            side 0 for left, 1 for right*/

            if (boundary_conditions[side] == 0){ //vac
                return(0.0);
            } else if (boundary_conditions[side] == 1){ //reflecting
                return( reflectingBC(side, group, angle) ); // manual alteration for reeds, change back
            } else if (boundary_conditions[side] == 3){ //mms
                return ( mms_boundary(side, group, angle, hn) );
            } else {
                bound_warn();
                return(0.0);
            }
        }

        void initilize_boundary(){
        /*breif: allocating af boundary vectors*/
            int boundary_size = 2*N_angles*N_groups;

            af_left_bound.resize(boundary_size, 0.0);
            af_right_bound.resize(boundary_size,0.0);
        }

    void assign_boundary(std::vector<double> aflux_last){
    // breif assigns the boundary fluxes in approriate containers for next itteration

        int index_left;
        int index_right;

        if (mms_bool){
            std::vector<double> temp;

            for (int j=0; j<N_angles; j++){

                temp = manSource.group1af(0, dx, dt*time_val, dt, angles[j]);

                af_left_bound[0*2*N_angles + 2*j  ] = temp[1];
                af_left_bound[0*2*N_angles + 2*j + 1] = temp[3];
                
                temp = manSource.group1af(L, dx, dt*time_val, dt, angles[j]);

                af_right_bound[0*2*N_angles + 2*j  ] = temp[0];
                af_right_bound[0*2*N_angles + 2*j + 1] = temp[2];
                
                temp = manSource.group2af(0, dx, dt*time_val, dt, angles[j]);

                af_left_bound[1*2*N_angles + 2*j  ] = temp[1];
                af_left_bound[1*2*N_angles + 2*j + 1] = temp[3];

                temp = manSource.group2af(L, dx, dt*time_val, dt, angles[j]);
                
                af_right_bound[1*2*N_angles + 2*j  ] = temp[0];
                af_right_bound[1*2*N_angles + 2*j + 1] = temp[2];
            }

        } else {
            for (int g=0; g<N_groups; g++){
                for (int j=0; j<N_angles; j++){

                    index_left  = g*(SIZE_groupBlocks) + 4*j + 2;
                    index_right = ((N_cells-1)*(SIZE_cellBlocks) + g*(SIZE_groupBlocks) + 4*j) + 3;

                    outofbounds_check(index_right, aflux_last);
                    outofbounds_check(index_left, aflux_last);

                    af_left_bound[g*N_groups + j] = aflux_last[index_left];
                    af_right_bound[g*N_groups +j] = aflux_last[index_right];
                }
            }
        }
    }

    void bound_warn(){
        if (warn_flag = 0){
            std::cout << ">>>>WARNING<<<<<" << endl;
            std::cout << "check boundary condition config" << endl;
            std::cout << "assuming vac" << endl;
            warn_flag = 1;
        }
    }
};

class boundary_condition{
    public:
        char side;
        int cell_id;
        int type;
        double magnitude;
        double dt;
        double dx;
};


class ts_solutions{
    public:
        std::vector<double> aflux;
        double time;
        double spectral_radius;
        double final_error;
        int number_iteration;
        int N_step;
};

class WholeProblem{
    public:
        vector<cell> cells;
        problem_space ps;
        vector<ts_solutions> solutions;

        WholeProblem(vector<cell> c, problem_space p, vector<ts_solutions> sol){
            cells = c;
            ps = p;
            solutions = sol;
        }

        void PublishUnsorted(){
            for (int t=0; t<ps.N_time; t++){
                string ext = ".csv";
                string file_name = "afluxUnsorted";
                string dt = to_string(t);

                file_name = file_name + dt + ext;

                std::ofstream output(file_name);
                output << "TIME STEP: " << t << "Unsorted solution vector" << endl;
                output << "N_space: " << ps.N_cells << " N_groups: " << ps.N_groups << " N_angles: " << ps.N_angles << endl;
                for (int i=0; i<solutions[t].aflux.size(); i++){
                    output << solutions[t].aflux[i] << "," << endl;
                }
            }
        }


};