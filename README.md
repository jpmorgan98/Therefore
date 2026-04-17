# Therefore

hipcc -std=c++17 -O3 --offload-arch=gfx942 -DTHEREFORE2D_ENABLE_ROCM -DTHEREFORE2D_EXAMPLE_USE_ROCM -I. example_infinite_medium_rocm.cpp transport2d.cpp transport2d_rocm.cpp output.cpp -lrocsolver -lrocblas -llapack -lblas -o example_infinite_medium_rocm
