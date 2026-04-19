# Therefore


## Compilation

### Inf homo mediuem

OpenMP
```bash
g++-15 -std=c++17 -O3 -fopenmp output.cpp transport2d.cpp example_infinite_medium.cpp -llapack -lblas -DTHEREFORE2D_EXAMPLE_USE_OPENMP -o inf_med.out
```

### TRT

Hot Wall
```bash
g++-15 -std=c++17 -O3 -fopenmp \
  output.cpp transport2d.cpp trt2d.cpp example_trt_hotwall_vacuum_cpu.cpp \
  -llapack -lblas \
  -DTHEREFORE2D_EXAMPLE_USE_OPENMP \
  -o simple_trt_hotwall
```

OpenMP
```bash
g++-15 -std=c++17 -O3 -fopenmp output.cpp transport2d.cpp trt2d.cpp example_trt_lattice_cpu.cpp -llapack -lblas -DTHEREFORE2D_EXAMPLE_USE_OPENMP -o trt_lattice_cpu.out
```

CPU Profiling (mac)
```bash
g++-15 -std=c++17 -O3 -fopenmp -g -I$(brew --prefix gperftools)/include -L$(brew --prefix gperftools)/lib -lprofiler output.cpp transport2d.cpp trt2d.cpp example_trt_lattice_cpu.cpp -llapack -lblas -DTHEREFORE2D_EXAMPLE_USE_OPENMP -o trt_lattice_cpu.out
```

ROCm
```bash
hipcc -std=c++17 -O3 --offload-arch=gfx942 -DTHEREFORE2D_ENABLE_ROCM -DTHEREFORE2D_EXAMPLE_USE_ROCM -I. example_infinite_medium_rocm.cpp transport2d.cpp transport2d_rocm.cpp output.cpp -lrocsolver -lrocblas -llapack -lblas -o example_infinite_medium_rocm
```


## Running

For CPU control OpenMP threads and BLAS threads

```bash
export OMP_NUM_THREADS=16 # however many ro run on
export OPEN_BLAS_NUM_THREADS=1 # BLAS only 1!
```