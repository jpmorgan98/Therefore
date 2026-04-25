# =============================================================================
# therefore2d — Makefile
#
# Usage:
#   make                    # CPU builds only (no OpenMP, no ROCm)
#   make USE_OPENMP=1       # CPU builds with OpenMP
#   make USE_ROCM=1         # also compile ROCm variants (requires hipcc)
#   make USE_OPENMP=1 USE_ROCM=1
#   make clean
#
# All binaries land in build/
# All run-time output lands in results/
# =============================================================================

# ---- Toolchain --------------------------------------------------------------

CXX      = icpx
HIPCC    ?= hipcc

# ---- Compiler flags ---------------------------------------------------------

CXXSTD   := -std=c++17
WARN     := -Wall -Wextra -Wpedantic
OPT      := -O3 -march=native -g
INCLUDES :=          # add -I paths here if headers live elsewhere

CXXFLAGS := $(CXXSTD) $(WARN) $(OPT) $(INCLUDES)

# LAPACK / BLAS — adjust for your platform.
# On most Linux distros: -llapack -lblas
# On macOS (Accelerate): -framework Accelerate
LAPACK_LIBS ?= -llapack -lblas

# ---- Optional features ------------------------------------------------------

ifeq ($(USE_OPENMP),1)
    CXXFLAGS += -qopenmp
    OMP_DEFS  := -DTHEREFORE2D_EXAMPLE_USE_OPENMP
else
    OMP_DEFS  :=
endif

ifeq ($(USE_ROCM),1)
    ROCM_DEFS := -DTHEREFORE2D_ENABLE_ROCM -DTHEREFORE2D_EXAMPLE_USE_ROCM
    ROCM_LIBS := -lrocblas -lrocsolver -lamdhip64
    HIPFLAGS  := $(CXXSTD) $(OPT) $(INCLUDES) $(ROCM_DEFS) $(OMP_DEFS)
else
    ROCM_DEFS :=
    ROCM_LIBS :=
endif

# ---- Directories ------------------------------------------------------------

BUILDDIR := build
RESDIR   := results
$(shell mkdir -p $(BUILDDIR) $(RESDIR))

# ---- Core library object files (CPU) ----------------------------------------
# These are shared by every example that does NOT use the ROCm back-end.

CORE_SRCS := transport2d.cpp trt2d.cpp output.cpp anderson.cpp
CORE_OBJS := $(patsubst %.cpp,$(BUILDDIR)/%.o,$(CORE_SRCS))

$(BUILDDIR)/%.o: %.cpp
    
	$(CXX) $(CXXFLAGS) $(OMP_DEFS) -c $< -o $@

# ---- CPU example binaries ---------------------------------------------------

CPU_EXAMPLES := \
    example_transport \
    example_run_transport \
    example_infinite_medium \
    example_trt_lattice_cpu \
    example_trt_hotwall_vacuum_cpu

.PHONY: all clean $(CPU_EXAMPLES)

all: $(CPU_EXAMPLES)

define BUILD_CPU_EXAMPLE
$(BUILDDIR)/$(1): $(BUILDDIR)/$(1).o $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) $(OMP_DEFS) $$^ $(LAPACK_LIBS) -o $$@

$(BUILDDIR)/$(1).o: $(1).cpp
	$(CXX) $(CXXFLAGS) $(OMP_DEFS) -c $$< -o $$@

$(1): $(BUILDDIR)/$(1)
endef

$(foreach ex,$(CPU_EXAMPLES),$(eval $(call BUILD_CPU_EXAMPLE,$(ex))))

# ---- ROCm variants (only built when USE_ROCM=1) -----------------------------

ifeq ($(USE_ROCM),1)

# ROCm core: transport2d_rocm.cpp is a HIP source compiled with hipcc.
# output.cpp and trt2d.cpp are plain C++ but still need the ROCm defines so
# that TrtState2D / TransportOutputFiles2D remain consistent.

ROCM_CORE_SRCS := transport2d.cpp trt2d.cpp output.cpp gmres.cpp
ROCM_CORE_OBJS := $(patsubst %.cpp,$(BUILDDIR)/rocm_%.o,$(ROCM_CORE_SRCS))

$(BUILDDIR)/rocm_%.o: %.cpp
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

$(BUILDDIR)/transport2d_rocm.o: transport2d_rocm.cpp
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

ROCM_EXAMPLES := \
    example_infinite_medium_rocm

define BUILD_ROCM_EXAMPLE
$(BUILDDIR)/$(1): $(BUILDDIR)/$(1)_rocm_main.o $(ROCM_CORE_OBJS) $(BUILDDIR)/transport2d_rocm.o
	$(HIPCC) $(HIPFLAGS) $$^ $(LAPACK_LIBS) $(ROCM_LIBS) -o $$@

$(BUILDDIR)/$(1)_rocm_main.o: $(1).cpp
	$(HIPCC) $(HIPFLAGS) -c $$< -o $$@

$(1): $(BUILDDIR)/$(1)
endef

$(foreach ex,$(ROCM_EXAMPLES),$(eval $(call BUILD_ROCM_EXAMPLE,$(ex))))

all: $(ROCM_EXAMPLES)

endif   # USE_ROCM

# ---- Profiler support (optional, via gperftools) ----------------------------
# If libprofiler is installed, link example_trt_lattice_cpu against it.
# Otherwise the no-op stubs in the source file are used and nothing is needed.
#
# To enable: make USE_PROFILER=1
ifeq ($(USE_PROFILER),1)
$(BUILDDIR)/example_trt_lattice_cpu: $(BUILDDIR)/example_trt_lattice_cpu.o $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) $(OMP_DEFS) $^ $(LAPACK_LIBS) -lprofiler -o $@
endif

# ---- Phony targets ----------------------------------------------------------

clean:
	rm -rf $(BUILDDIR)

.PHONY: all clean $(CPU_EXAMPLES) $(ROCM_EXAMPLES)
