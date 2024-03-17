/*brief: assembly functions for matrices to compute therefore commands
date: May 23rd 2023
auth: J Piper Morgan (morgajoa@oregonstate.edu)*/

#include <iostream>
#include <vector>
#include "run.cpp"

//#include "util.h"
//#include "mms.h"
//#include "builders.h"
//#include "cusolver_axb.cu"

//#include "H5Cpp.h"
//#include "lapack.h"
//#include "rocsolver.cpp"
//#include <Eigen/Dense>
//#include <cusparse_v2.h>
//#include <cuda.h>

const bool print_mats = false;
const bool debug_print = false;
const bool gpu = false;



/* compile notes and prerecs
For LAPACK (not dense)
    In UBUNTU 
        you need these libraries:
            sudo apt-get install libblas-dev checkinstall
            sudo apt-get install libblas-doc checkinstall
            sudo apt-get install liblapacke-dev checkinstall
            sudo apt-get install liblapack-doc checkinstall
        Should be able to configure with: 
            g++ main.cpp -std=c++20 -llapack

    In OSX
        you need:
            brew install gcc
            brew install lapack
        Should be able to compile with: 
            g++-13 main.cpp -std=c++20 -llapack
            
For Trilinos (see Tril.md for insturcations)

For CUDA GPU
    On LASSEN
        modules
            CUDA Toolkit 12.0
            cmake/20

        Uncomment CUDA lines
        to build
            mkdir build
            cd build
            cmake ..
            make

For AMD GPU
    on Lockhart (AMD MI200 devlopment machine)
        module load rocm
        cc main.cpp -isystem "/opt/rocm-5.5.1/include" -I/opt/rocm/include -lrocsolver -lrocblas -D__HIP_PLATFORM_AMD__
        hipcc -I/opt/rocm/include -L/opt/rocm/lib -L/usr/lib64 -lrocsolver -lrocblas -llapack main.cpp
    Generally on a system with 
        /opt/rocm/bin/hipcc -I/opt/rocm/include -c example.cpp /opt/rocm/bin/hipcc -o example -L/opt/rocm/lib -lrocsolver -lrocblas example.o


/opt/rocm/llvm/bin/clang++ -isystem "/opt/rocm-5.5.1/include"  --offload-arch=gfx90a -O3 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false  -O3 --hip-link --rtlib=compiler-rt -unwindlib=libgcc  -I/opt/rocm/include -L/opt/rocm/lib -lrocsolver -lrocblas -L/home/joamorga/miniconda3/lib -llapack -x hip main.cpp
gfx803
*/



using namespace std;

void eosPrint(ts_solutions state);

// row major to start -> column major for lapack computation

//

// i space, m is angle, k is time, g is energy group

int main(void){

    print_title();

    using namespace std;
    
    // problem definition
    // eventually from an input deck
    double dx = 01;
    double dt = 1.0;
    vector<double> v = {1, 1};
    vector<double> xsec_total = {1.5454, 0.45468};
    //vector<double> xsec_total = {1, 1};
    vector<double> xsec_scatter = {0.61789, 0.38211, .92724, 0.072534};
    //vector<double> xsec_scatter = {0,0,0,0};
    //double ds = 0.0;
    vector<double> Q = {1, 1, 1, 1, 1, 1, 1, 1};

    double Length = 1;
    double IC_homo = 0;
    
    int N_cells = 100; //10
    int N_angles = 2;
    int N_time = 1;
    int N_groups = 2;

    // 4 = N_subspace (2) * N_subtime (2)
    int N_mat = 4 * N_cells * N_angles * N_groups;

    // N_cm is the size of the row major vector
    int N_rm = N_mat*N_mat;

    // homogeneous initial condition vector
    // will be stored as the solution at time=0
    vector<double> IC(N_mat, 0.0);
    for (int p=0; p<N_cells*2; p++){IC[p] = IC[p]*IC_homo;}

    // actual computation below here

    // generate g-l quadrature angles and weights
    vector<double> weights(N_angles, 0.0);
    vector<double> angles(N_angles, 0.0);

    quadrature(angles, weights);

    // problem space class construction
    problem_space ps;
    ps.L = Length;
    ps.dt = dt;
    ps.dx = dx;
    ps.N_angles = N_angles;
    ps.N_cells = N_cells;
    ps.N_groups = N_groups;
    ps.N_time = N_time;
    ps.N_rm = N_rm;
    ps.N_mat = N_mat;
    ps.angles = angles;
    ps.weights = weights;
    ps.initialize_from_previous = false;
    ps.max_iteration = int(1e4);
    // 0 for vac 1 for reflecting 3 for mms
    ps.boundary_conditions = {0,0};
    // size of the cell blocks in all groups and angle
    ps.SIZE_cellBlocks = ps.N_angles*ps.N_groups*4;
    ps.ELEM_cellBlocks = ps.SIZE_cellBlocks*ps.SIZE_cellBlocks;
    // size of the group blocks in all angle within a cell
    ps.SIZE_groupBlocks = ps.N_angles*4;
    // size of the angle blocks within a group and angle
    ps.SIZE_angleBlocks = 4;


    // allocates a zero vector of nessacary size
    ps.initilize_boundary();
    
    /*
    // =================== REEDS Problem

    // reeds problem mat stuff 
    
    // mono-energetic data
    vector<double> sigma_s_reeds = {.9,  .9,    0,    0,    0};
    vector<double> sigma_t_reeds = {1,    1,    0,    5,    50};
    vector<double> Source_reeds  = {0,    1,    0,    0,    50};
    vector<double> dx_reeds =      {0.25, 0.25, 0.25, 0.02, 0.02};
    vector<int> N_per_region =     {8,    8 ,   4,    50,   100};
    
    // cell construction;
    vector<cell> cells;

    int region_id = 0;
    int N_next = N_per_region[0]-1;

    for (int i=0; i<N_cells; i++){
        // /*building reeds problem from left to right

        if (i==N_next+1){
            region_id ++;
            N_next += N_per_region[region_id];
        }

        cell cellCon;
        cellCon.cell_id = i;
        if (i ==0 )
            cellCon.x_left = 0;
        else
            cellCon.x_left = cells[cells.size()-1].x_left+cells[cells.size()-1].dx;
        
        cellCon.xsec_scatter = vector<double> {sigma_s_reeds[region_id], sigma_s_reeds[region_id]};
        cellCon.xsec_total = vector<double> {sigma_t_reeds[region_id], sigma_t_reeds[region_id]};
        cellCon.dx = dx_reeds[region_id];
        cellCon.v = v;
        cellCon.dt = dt;

        vector<double> temp (N_angles*N_groups*4, 1.0);
        for (int p=0; p<temp.size(); ++p){temp[p] *= Source_reeds[region_id];}
        cellCon.Q = temp;

        //cellCon.Q = vector<double> {Source_reeds[region_id], Source_reeds[region_id], Source_reeds[region_id], Source_reeds[region_id],
        //                            0, 0, 0, 0};
                            
        cellCon.region_id = region_id;
        cellCon.N_angle = N_angles;

        cells.push_back(cellCon);
    }*/

   // ===================
   
    vector<cell> cells;

    for (int i=0; i<N_cells; i++){
        // /*building a single region problem

        cell cellCon;
        cellCon.cell_id = i;
        if (i ==0 )
            cellCon.x_left = 0;
        else
            cellCon.x_left = cells[cells.size()-1].x_left+cells[cells.size()-1].dx;
        cellCon.x = cellCon.x_left + dx/2;
        cellCon.xsec_scatter = xsec_scatter;
        cellCon.xsec_total = xsec_total;
        cellCon.dx = dx;
        cellCon.v = v;
        cellCon.dt = dt;
        vector<double> temp (N_angles*N_groups*4, 1.0);
        for (int p=0; p<temp.size(); ++p){temp[i] *= Q[0];}
        cellCon.Q = temp;
        cellCon.region_id = 1;
        cellCon.N_angle = ps.N_angles;

        cells.push_back(cellCon);

    }
    ps.mms_bool = false;

    // ===================


    // initial condition stored as first element of solution vector
    vector<ts_solutions> solutions;

    ts_solutions sol_con;
    sol_con.aflux = IC;
    sol_con.time = 0.0;
    sol_con.spectral_radius = 0.0;
    sol_con.N_step = 0;
    sol_con.number_iteration = 0;

    solutions.push_back(sol_con);


    run problem;
    problem.ps = ps;
    problem.cells = cells;
    problem.IC = IC;
    //// mms coefficients
    //problem.manSource.A = 1.0;
    //problem.manSource.B = 1.0;
    //problem.manSource.C = 1.0;
    //problem.manSource.D = 1.0;
    //problem.manSource.F = 1.0;
    //problem.manSource.v1 = v[0];
    //problem.manSource.v2 = v[1];
    //problem.manSource.sigma1 = xsec_total[0];
    //problem.manSource.sigma2 = xsec_total[1];
    //problem.manSource.sigmaS1 = xsec_scatter[0];
    //problem.manSource.sigmaS2 = xsec_scatter[1];
    //problem.manSource.sigmaS1_2 = ps.ds;
//
    //ps.manSource = problem.manSource;

    
    problem.run_timestep();

    //problem.publish_mms();
    
    return(0);
} // end of main