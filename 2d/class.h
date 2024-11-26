
#include <vector>
using namespace std;

// intention with region is that it is put into "vector<region> regions(N_regions);"
// then regions[region_id]
// this could also be done with 2 level vector of ints tho that would be annoying to put in signatures :(
class region{
    public:
        vector<int> cells; //list of cell_ids in a given region of cells in a given region
};


class cell{
    public:
        int cell_id;
        double x_lc; //SW corner!
        double y_lc;
        double dx;
        double dy;
        int i; // integer location in x
        int j; // integer location in y
        bool boundary=false;
        int boundary_type; //1=vac, 2=dirichlet, 3=reflecting
        int boundary_side; //1=N, 2=E, 3=S, 4=W 

        vector<double> dirichlet_boundary_condition; // the actual boundary condition

        void print_cell_specs(){
            cout << "CELL SPECIFICATIONS" << endl;
            cout << "i= " << i << "  j= "<< j << endl;
            if (boundary){
                cout << "I am " << boundary_type << " boundary cell on edge " << boundary_side << endl;
            }
        }

        vector<double> get_boundary_cond(){
            if (boundary_type==1){
                return(vector<double> (8));
            } else if (boundary_type==2){
                cout << ">>>>WARNING Dirichlet boundary condition not currently supported, switching to vac" << endl;
                return(vector<double> (8));
            } else if (boundary_type==3){
                cout << ">>>>WARNING Reflecting boundary condition not currently supported, switching to vac" << endl;
                return(vector<double> (8));
            } else {
                cout << ">>>>WARNING Boundary condition specified for a boundary cell" << endl;
                return(vector<double> (8));
            }
        }

};




class problem_space{
    public:
        int N_angles;
        int N_i;
        int N_j;
        int N_sub = 8;
        int N_cells;
        int N_time;
        int N_region;

        int SIZE_cellBlocks;
        int SIZE_angleBlocks = 8;
        int ELEM_A;
        int ELEM_cellBlocks;
        int SIZE_aflux;

        vector<double> eta;
        vector<double> mu;
        vector<double> weights;

        // vectors of size N_region
        vector<double> sigma;
        vector<double> sigma_s;
        vector<double> v;
        vector<double> material_source;

        double dx;
        double dy;
        double dt;

        vector<cell> cells;

        double tol = 1e-9;
        int max_iter = int(1e4);



        void initialize_cells(int uN_i, double udx, int uN_j, double udy){



            SIZE_cellBlocks = N_angles*N_sub;
            N_cells = N_j*N_i;
            SIZE_aflux = N_angles*SIZE_angleBlocks*N_cells;
            ELEM_cellBlocks = pow(N_angles*SIZE_angleBlocks, 2);
            ELEM_A = N_cells * ELEM_cellBlocks;

            // CELLS ARE ARRANGED IN PDE SPACE
            for (int i=0; i<N_i; ++i){
                for (int j=0; j<N_j; ++j){
                    cell cellcon;
                    cellcon.cell_id = i + N_i*j;

                    cellcon.cell_id;
                    cellcon.x_lc = i*dx; //SW corner!
                    cellcon.y_lc = j*dy;
                    cellcon.dx = dx;
                    cellcon.dy = dy;
                    cellcon.i = i; // integer location in x
                    cellcon.j = j; // integer location in y
                    

                    if (i==N_i-1){
                        cellcon.boundary=true;
                        cellcon.boundary_type=1; //1=vac, 2=dirichlet, 3=reflecting
                        cellcon.boundary_side=1; //1=N, 2=E, 3=S, 4=W 
                    } else if (i==0){
                        cellcon.boundary=true;
                        cellcon.boundary_type=1; //1=vac, 2=dirichlet, 3=reflecting
                        cellcon.boundary_side=1; //1=N, 2=E, 3=S, 4=W 
                    } else if (j==N_j-1){
                        cellcon.boundary=true;
                        cellcon.boundary_type=1; //1=vac, 2=dirichlet, 3=reflecting
                        cellcon.boundary_side=1; //1=N, 2=E, 3=S, 4=W 
                    } else if (j==0){
                        cellcon.boundary=true;
                        cellcon.boundary_type=1; //1=vac, 2=dirichlet, 3=reflecting
                        cellcon.boundary_side=1; //1=N, 2=E, 3=S, 4=W 
                    } else if (j==0){ // corner cases!
                        cellcon.boundary=true;
                        cellcon.boundary_type=1; //1=vac, 2=dirichlet, 3=reflecting
                        cellcon.boundary_side=1; //1=N, 2=E, 3=S, 4=W 
                    } else if (j==0){
                        cellcon.boundary=true;
                        cellcon.boundary_type=1; //1=vac, 2=dirichlet, 3=reflecting
                        cellcon.boundary_side=1; //1=N, 2=E, 3=S, 4=W 
                    } else if (j==0){
                        cellcon.boundary=true;
                        cellcon.boundary_type=1; //1=vac, 2=dirichlet, 3=reflecting
                        cellcon.boundary_side=1; //1=N, 2=E, 3=S, 4=W 
                    } else if (j==0){
                        cellcon.boundary=true;
                        cellcon.boundary_type=1; //1=vac, 2=dirichlet, 3=reflecting
                        cellcon.boundary_side=1; //1=N, 2=E, 3=S, 4=W 
                    } else {
                        cellcon.boundary=true;
                        cellcon.boundary_type=0;
                    }

                    cells.push_back(cellcon);
                }
            }
        }

        int get_af_index(int quadrant, int m, int i, int j, int k){
            // IN: problem space location get back
            // OUT: location in the global af vector
            // PARAMETERS:
            //      quadrant....NW=0, NE=1, SW=2, SE=3
            //      m...........angle index
            //      i...........x-space index
            //      j...........y-space index
            //      k...........0=time avg value (k), 1=time edge value (k+1/2)

            int index_val = SIZE_cellBlocks*(j*N_i + i*N_j) + m*SIZE_angleBlocks + 4*k + quadrant;

            return(index_val);
        }
};
