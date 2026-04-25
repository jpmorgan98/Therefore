/// example_trt_hotwall_vacuum_cpu.cpp
///
/// Grey TRT Marshak wave: hot wall (left) driving radiation into cold foam.
///
/// Uses a CONSTANT grey absorption opacity and analytical grey
/// Stefan-Boltzmann Planck functions B(T) = c*a*T^4/(4*pi).
///
/// Default parameters are tuned for a quick run (~1-2 min on one core):
///   kNx=kNy=20, kSN=2 (4 dirs), kSigmaGrey=10 /cm, kDt=1e-12 s.
///
/// For a higher-quality run, increase kNx/kNy/kSN and tighten kNonlinearTol.

#include "anderson.hpp"
#include "output.hpp"
#include "transport2d.hpp"
#include "trt2d.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {
using namespace therefore2d;

// ---------------------------------------------------------------------------
// Problem parameters
// ---------------------------------------------------------------------------
constexpr int    kNx  = 40;   // cells in x
constexpr int    kNy  = 40;   // cells in y
constexpr int    kSN  = 4;    // Sn order: 4 → 8 dirs (use 2 for quick test)
constexpr int    kNumTimeSteps      = 50;
constexpr int    kMaxNonlinearIters = 300;
constexpr int    kMaxTransportIters = 100;
constexpr double kDt           = 1.0e-12;  // s
constexpr double kTransportTol = 1.0e-10;
constexpr double kNonlinearTol = 1.0e-4;   // looser: Picard converges slowly for alpha>0.5
constexpr double kTfloor       = 1.0e-3;   // keV
constexpr double kTcold        = 1.0e-3;   // keV  (initial)
constexpr double kThot         = 1.0;      // keV  (hot wall)
constexpr double kLx           = 1.0;      // cm
constexpr double kLy           = 1.0;      // cm
constexpr double kRho          = 0.2;      // g/cm^3  (Brunner foam)
constexpr double kCv           = 2.41213e14; // erg/(g keV)

// ---- Constant grey opacity -----------------------------------------------
//   sigma =  1 /cm → 1 mfp across domain  (thin, fast wave)
//   sigma = 10 /cm → 10 mfps              (moderate, default)
//   sigma = 40 /cm → 40 mfps              (thick, slow wave)
// Note: larger sigma → larger alpha(T) → more NL iterations needed.
constexpr double kSigma = 10.0;  // /cm


// ---- Anderson mixing acceleration ---
// Set kUseAnderson = true to enable; compare with false to benchmark.
constexpr bool   kUseAnderson  = true;
constexpr int    kAndersonM    = 5;     // history window (3-5 optimal)
constexpr double kAndersonDamp = 0.8;   // 0.8 empirically better than 1.0

// ---------------------------------------------------------------------------
// Grey Stefan-Boltzmann helpers  (no numerical integration needed)
// ---------------------------------------------------------------------------
inline double grey_B(double T) {
    const double t = std::max(T, kTfloor);
    return kTrtSpeedOfLight * kTrtRadiationConstant * t*t*t*t / (4.0 * M_PI);
}
inline double grey_dBdT(double T) {
    const double t = std::max(T, kTfloor);
    return kTrtSpeedOfLight * kTrtRadiationConstant * t*t*t / M_PI;
}
inline double grey_alpha(double T, double dt) {
    const double d = 4.0*M_PI*kSigma*grey_dBdT(T);
    return d / (kRho*kCv/dt + d);
}

// ---------------------------------------------------------------------------
// Boundary and fill
// ---------------------------------------------------------------------------
void set_hotwall_vacuum_bc(Problem2D& p, double psi_left) {
    p.boundary.west.assign(p.ny*p.groups*p.num_dirs()*4, 0.0);
    p.boundary.east.clear(); p.boundary.south.clear(); p.boundary.north.clear();
    for (int j=0;j<p.ny;++j) for (int g=0;g<p.groups;++g)
        for (int d=0;d<p.num_dirs();++d) {
            if (p.directions[d].mu<=0.0) continue;
            const int off=face_offset_west_east(p,j,g,d,0);
            for (int k=0;k<4;++k) p.boundary.west[off+k]=psi_left;
        }
}

void fill_cells(SolverState2D& state, const std::vector<double>& Tlag) {
    Problem2D& p=state.problem;
    const double dx=p.Lx/p.nx, dy=p.Ly/p.ny;
    set_hotwall_vacuum_bc(p, grey_B(kThot));
    for (int j=0;j<p.ny;++j) for (int i=0;i<p.nx;++i) {
        const int cell=cell_id(i,j,p.nx);
        Cell2D& c=state.cells[cell];
        const double T=std::max(Tlag[cell],kTfloor);
        c.x_left=i*dx; c.y_bottom=j*dy; c.dx=dx; c.dy=dy; c.dt=p.time_step;
        c.velocity.assign(1,kTrtSpeedOfLight);
        c.sigma_t.assign(1,0.0); c.sigma_s.assign(1,0.0);
        c.source.assign(p.cell_block_size(),0.0);
        const double B=grey_B(T), dB=grey_dBdT(T);
        const double alpha=grey_alpha(T,p.time_step);
        c.sigma_t[0]=kSigma;
        c.sigma_s[0]=alpha*kSigma;
        const double q=(1.0-alpha)*kSigma*B;
        for (int d=0;d<p.num_dirs();++d) {
            const int off=local_angle_group_offset(p,0,d,0);
            for (int dof=0;dof<kDofsPerAngleGroup2D;++dof) c.source[off+dof]=q;
        }
    }
}

// ---------------------------------------------------------------------------
// Temperature update (linearised implicit energy equation)
// ---------------------------------------------------------------------------
std::vector<double> update_T(const SolverState2D& state,
                              const std::vector<double>& T_old,
                              const std::vector<double>& Tlag,
                              const std::vector<double>& phi) {
    const Problem2D& p=state.problem;
    std::vector<double> next(p.num_cells(),kTfloor);
    for (int cell=0;cell<p.num_cells();++cell) {
        const double T=std::max(Tlag[cell],kTfloor);
        const double B=grey_B(T), dB=grey_dBdT(T);
        double lhs=kRho*kCv/p.time_step;
        double rhs=lhs*T_old[cell];
        lhs+=4.0*M_PI*kSigma*dB;
        rhs+=4.0*M_PI*kSigma*(phi[cell]-B+dB*T);
        next[cell]=std::max(kTfloor, rhs/lhs);
    }
    return next;
}

std::vector<double> make_Trad(const SolverState2D& state) {
    const Problem2D& p=state.problem;
    std::vector<double> v(p.num_cells(),kTfloor);
    for (int c=0;c<p.num_cells();++c) {
        const double phi=cell_centered_scalar_flux(state,state.flux_previous,c,0);
        const double ur=4.0*M_PI*phi/kTrtSpeedOfLight;
        if (ur>0.0) v[c]=std::max(kTfloor, std::pow(ur/kTrtRadiationConstant,0.25));
    }
    return v;
}

double max_rel(const std::vector<double>& a, const std::vector<double>& b) {
    double e=0.0;
    for (std::size_t i=0;i<a.size();++i) {
        const double d=std::max(1e-14,std::max(std::abs(a[i]),std::abs(b[i])));
        e=std::max(e,std::abs(a[i]-b[i])/d);
    }
    return e;
}
} // namespace

int main(int argc, char** argv) {
    using namespace therefore2d;
#ifdef THEREFORE2D_EXAMPLE_USE_OPENMP
    const bool use_openmp=true;
#else
    const bool use_openmp=false;
#endif
    int nsteps=kNumTimeSteps;
    if (argc>1) nsteps=std::stoi(argv[1]);

    SolverState2D state; Problem2D& p=state.problem;
    p.nx=kNx; p.ny=kNy; p.Lx=kLx; p.Ly=kLy;
    p.groups=1; p.max_iters=kMaxTransportIters;
    p.num_time_steps=nsteps; p.time_step=kDt;
    p.convergence_tol=kTransportTol;
    p.initialize_from_previous=true; p.reuse_factorization=false;
    p.directions=make_level_symmetric_quadrature_2d(kSN);
    state.cells.assign(p.num_cells(),Cell2D{});

    std::vector<double> T(p.num_cells(),kTcold);
    fill_cells(state,T);
    initialize_state(state,std::vector<double>(p.total_unknowns(),grey_B(kTcold)));

    const double t_total=kDt*nsteps;
    const double D=kTrtSpeedOfLight/(3.0*kSigma);
    std::cout<<"Grey Marshak wave  sigma="<<kSigma<<" /cm  mfp="<<1.0/kSigma<<" cm\n"
             <<"  nx="<<kNx<<"  kSN="<<kSN<<"  dirs="<<p.num_dirs()
             <<"  cell_block="<<p.cell_block_size()<<"\n"
             <<"  dt="<<kDt<<" s  nsteps="<<nsteps
             <<"  t_total="<<t_total<<" s\n"
             <<"  alpha(T_hot)="<<grey_alpha(kThot,kDt)
             <<"  alpha(T_cold)="<<grey_alpha(kTcold,kDt)<<"\n"
             <<"  diffusion front: sqrt(D*t)="<<std::sqrt(D*t_total)<<" cm\n";

    const std::string outdir="results/example_trt_hotwall_vacuum";
    const std::string jsonpath=outdir+"/summary.json";
    std::filesystem::create_directories(outdir);
    ParaviewSeriesWriter2D writer(make_rectilinear_grid(state),
        ParaviewSeriesConfig2D{outdir,"trt_hotwall",true});

    CpuLUCache cache;
    std::vector<TrtTimestepStats2D> hist;
    hist.reserve(nsteps);
    double time=0.0;

    for (int step=0;step<nsteps;++step) {
        const std::vector<double> T_old=T;
        std::vector<double> Tlag=T;
        TrtTimestepStats2D rec;

        // Anderson accelerator — fresh per timestep (history resets each step).
        FixedPointAccelerator acc(kAndersonM, kAndersonDamp);

        for (int nl=0;nl<kMaxNonlinearIters;++nl) {
            fill_cells(state,Tlag);
            assemble_cell_matrices(state);
            cache.valid=false;
            build_constant_rhs(state);
            rec.transport_stats=run_one_timestep_cpu(state,cache,use_openmp);

            std::vector<double> phi(p.num_cells());
            for (int c=0;c<p.num_cells();++c)
                phi[c]=cell_centered_scalar_flux(state,state.flux_previous,c,0);

            auto nT=update_T(state,T_old,Tlag,phi);
            // Convergence on the RAW residual (before acceleration)
            rec.max_temperature_change=max_rel(Tlag,nT);
            rec.nonlinear_iterations=nl+1;
            if (kUseAnderson) acc.apply(Tlag,nT);
            Tlag=std::move(nT);
            if (rec.max_temperature_change<kNonlinearTol) break;
        }

        T=Tlag; time+=kDt;
        rec.step=step; rec.time=time;
        hist.push_back(rec);

        std::vector<CellScalarField2D> fields;
        fields.push_back(make_cell_scalar_field("radiation_temperature",make_Trad(state)));
        fields.push_back(make_cell_scalar_field("material_temperature",T));
        writer.write_step(step,time,fields);

        std::cout<<"step "<<step
                 <<"  nl="<<rec.nonlinear_iterations
                 <<(rec.nonlinear_iterations>=kMaxNonlinearIters?"(MAX)":"     ")
                 <<"  dT="<<rec.max_temperature_change
                 <<"  tr="<<rec.transport_stats.iterations
                 <<"  T[0]="<<T[0]
                 <<"  T[nx/4]="<<T[p.nx/4]
                 <<"  T[nx/2]="<<T[p.nx/2]
                 <<"\n";
    }

    TrtState2D dummy; dummy.transport=state;
    dummy.config.dt=kDt; dummy.config.num_time_steps=nsteps;
    dummy.history=hist;
    write_trt_summary_json(jsonpath,dummy,writer.pvd_path());
    std::cout<<"Wrote:\n  "<<writer.pvd_path()<<"\n  "<<jsonpath<<"\n";
    return 0;
}
