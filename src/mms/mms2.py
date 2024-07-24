import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
sym.init_printing(use_unicode=True)
import math

# note: all cross sections (sigma) are understood to be macroscopic
# group 1 is the fast group 2 is the slow
# to evaluate intergrals go to https://www.integral-calculator.com/

if __name__ == '__main__':

    # Independent variables
    x = sym.Symbol('x')
    t = sym.Symbol('t')
    mu = sym.Symbol('mu')

    # Group velocities
    v1 = sym.Symbol('v1')
    v2 = sym.Symbol('v2')

    # Group total cross sections
    Sigma_1 = sym.Symbol('Sigma_1')
    Sigma_2 = sym.Symbol('Sigma_2')

    # Within group scattering cross sections
    Sigma_S1 = sym.Symbol('Sigma_S1')
    Sigma_S2 = sym.Symbol('Sigma_S2')

    # Down scattering (from 1 into 2)
    Sigma_S12 = sym.Symbol('Sigma_S12')

    # Upscattering scattering (from 2 into 1)
    Sigma_S21 = sym.Symbol('Sigma_S21')

    # arbitrary manufactured solution
    # note: technically psi1 = Ax + Bt + Cmu but they are all chosen to be 1

    psi1 = x + t + (mu +1)

    # doing some calc 
    dpsi1_dx = sym.diff(psi1, x)
    dpsi1_dt = sym.diff(psi1, t)
    
    # evaluated manually wrt mu, bounds -1 to 1
    eval_intpsi1 = -2*(x+t+1)

    # group 2
    psi2 = x**2*t + (mu+1)

    # doing some calc 
    dpsi2_dx = sym.diff(psi2, x)
    dpsi2_dt = sym.diff(psi2, t)

    # evaluated manually wrt mu, bounds -1 to 1
    eval_intpsi2 = -2*t*x**2-2

    # NTE for group 1
    L1 = (1/v1)*dpsi1_dt + mu*dpsi1_dx + Sigma_1*psi1
    #    w/in group               up scattering
    R1 = Sigma_S1/2*eval_intpsi1 + Sigma_S21 / 2 * eval_intpsi2
    Q1 = sym.simplify(2*L1-R1)

    # NTE for group 2
    L2 = (1/v2)*dpsi2_dt + mu*dpsi2_dx + Sigma_2*psi2
    #    w/in group               down scattering
    R2 = Sigma_S2/2*eval_intpsi2 - Sigma_S12 / 2 * eval_intpsi1
    Q2 = sym.simplify(2*L2-R2)

    # discretization terms

    x_j = sym.Symbol('x_j')
    Deltax = sym.Symbol('Deltax')

    t_k = sym.Symbol('t_k')
    Deltat = sym.Symbol('Deltat')

    # cell integrated time edge

    Q1_Lint = sym.integrate(Q1, (x, x_j-Deltax/2, x_j))
    Q1_Rint = sym.integrate(Q1, (x, x_j, x_j+Deltax/2))
    Q2_Lint = sym.integrate(Q2, (x, x_j-Deltax/2, x_j))
    Q2_Rint = sym.integrate(Q2, (x, x_j, x_j+Deltax/2))

    #cell integrated time averaged sym.simplify

    Q1_Lint_timeav = Q1_Lint.subs(t, t_k+Deltat/2) + Q1_Lint.subs(t, t_k-Deltat/2) / 2
    Q1_Rint_timeav = Q1_Rint.subs(t, t_k+Deltat/2) + Q1_Rint.subs(t, t_k-Deltat/2) / 2
    Q2_Lint_timeav = Q2_Lint.subs(t, t_k+Deltat/2) + Q2_Lint.subs(t, t_k-Deltat/2) / 2
    Q2_Rint_timeav = Q2_Rint.subs(t, t_k+Deltat/2) + Q2_Rint.subs(t, t_k-Deltat/2) / 2

    print("=========SOURCES=========")
    print()
    print('Continuous function describing group 1 source')
    print(Q1)
    print()
    print(sym.latex(Q1))
    print()
    print('Continuous function describing group 2 source')
    print(Q2)
    print()
    print(sym.latex(Q2))
    print()
    print()
    print("Cell integrated time edge values for export to a function")
    print()
    print(Q1_Lint.subs(t, t_k+Deltat/2))
    print(Q1_Rint.subs(t, t_k+Deltat/2))
    print(Q2_Lint.subs(t, t_k+Deltat/2))
    print(Q2_Rint.subs(t, t_k+Deltat/2))
    print()
    print()
    print("Cell integrated time edge values for Latex")
    print()
    print(sym.latex(Q1_Lint.subs(t, t_k+Deltat/2)))
    print(sym.latex(Q1_Rint.subs(t, t_k+Deltat/2)))
    print(sym.latex(Q2_Lint.subs(t, t_k+Deltat/2)))
    print(sym.latex(Q2_Rint.subs(t, t_k+Deltat/2)))
    print()
    print()
    print("Cell integrated time average values for export to a function")
    print()
    print(Q1_Lint_timeav)
    print(Q1_Rint_timeav)
    print(Q2_Lint_timeav)
    print(Q2_Rint_timeav)
    print()
    print()
    print("Cell integrated time average values for Latex")
    print()
    print(sym.latex(Q1_Lint_timeav))
    print(sym.latex(Q1_Rint_timeav))
    print(sym.latex(Q2_Lint_timeav))
    print(sym.latex(Q2_Rint_timeav))
    print()
    print()


    # Same crap but for Angular flux terms

    # cell integrated time edge
    AF1_Lint = sym.integrate(psi1, (x, x_j-Deltax/2, x_j))
    AF1_Rint = sym.integrate(psi1, (x, x_j, x_j+Deltax/2))
    AF2_Lint = sym.integrate(psi2, (x, x_j-Deltax/2, x_j))
    AF2_Rint = sym.integrate(psi2, (x, x_j, x_j+Deltax/2))

    #cell integrated time averaged
    AF1_Lint_timeav = AF1_Lint.subs(t, t_k+Deltat/2) + AF1_Lint.subs(t, t_k-Deltat/2) / 2
    AF1_Rint_timeav = AF1_Rint.subs(t, t_k+Deltat/2) + AF1_Rint.subs(t, t_k-Deltat/2) / 2
    AF2_Lint_timeav = AF2_Lint.subs(t, t_k+Deltat/2) + AF2_Lint.subs(t, t_k-Deltat/2) / 2
    AF2_Rint_timeav = AF2_Rint.subs(t, t_k+Deltat/2) + AF2_Rint.subs(t, t_k-Deltat/2) / 2

    print("=========ANGULAR FLUXES=========")
    print()
    print()
    print("Cell integrated time edge values for export to a function")
    print()
    print(AF1_Lint.subs(t, t_k+Deltat/2))
    print(AF1_Rint.subs(t, t_k+Deltat/2))
    print(AF2_Lint.subs(t, t_k+Deltat/2))
    print(AF2_Rint.subs(t, t_k+Deltat/2))
    print()
    print()
    print("Cell integrated time edge values for Latex")
    print()
    print(sym.latex(AF1_Lint.subs(t, t_k+Deltat/2)))
    print(sym.latex(AF1_Rint.subs(t, t_k+Deltat/2)))
    print(sym.latex(AF2_Lint.subs(t, t_k+Deltat/2)))
    print(sym.latex(AF2_Rint.subs(t, t_k+Deltat/2)))
    print()
    print()

    print("Cell integrated time average values for export to a function")
    print()
    print(AF1_Lint_timeav)
    print(AF1_Rint_timeav)
    print(AF2_Lint_timeav)
    print(AF2_Rint_timeav)
    print()
    print()
    print("Cell integrated time average values for Latex")
    print()
    print(sym.latex(AF1_Lint_timeav))
    print(sym.latex(AF1_Rint_timeav))
    print(sym.latex(AF2_Lint_timeav))
    print(sym.latex(AF2_Rint_timeav))
    print()
    print()

    #### C++ function output

    mms_out =  open("mms_auto2.h", "w")
    print("#include <iostream>", file=mms_out)
    print("#include <vector>", file=mms_out)
    print("//File auto generated", file=mms_out)
    print("", file=mms_out)

    print("std::vector<double> Q1(double v1, double v2, double Sigma_1, double Sigma_2, double mu, double Sigma_S1, double Sigma_S2, double Sigma_S12, double Sigma_S21, double x_j, double Deltax, double t_k, double Deltat) {", file=mms_out)
    print("", file=mms_out)
    print("std::vector<double> Q1_vals (4);", file=mms_out)
    print("", file=mms_out)
    print("Q1_vals[0] = ", Q1_Lint_timeav, ";", file=mms_out)
    print("Q1_vals[1] = ", Q1_Rint_timeav, ";", file=mms_out)
    print("Q1_vals[2] = ", Q1_Lint.subs(t, t_k+Deltat/2), ";", file=mms_out)
    print("Q1_vals[3] = ", Q1_Rint.subs(t, t_k+Deltat/2), ";", file=mms_out)
    print("", file=mms_out)
    print("return( Q1_vals );", file=mms_out)
    print("}", file=mms_out)
    print("", file=mms_out)
    print("std::vector<double> Q2(double v1, double v2, double Sigma_1, double Sigma_2, double mu, double Sigma_S1, double Sigma_S2, double Sigma_S12, double Sigma_S21, double x_j, double Deltax, double t_k, double Deltat) {", file=mms_out)
    print("", file=mms_out)
    print("std::vector<double> Q2_vals (4);", file=mms_out)
    print("", file=mms_out)
    print("Q2_vals[0] = ", Q2_Lint_timeav, ";", file=mms_out)
    print("Q2_vals[1] = ", Q2_Rint_timeav, ";", file=mms_out)
    print("Q2_vals[2] = ", Q2_Lint.subs(t, t_k+Deltat/2), ";", file=mms_out)
    print("Q2_vals[3] = ", Q2_Rint.subs(t, t_k+Deltat/2), ";", file=mms_out)
    print("", file=mms_out)
    print("return( Q2_vals );", file=mms_out)
    print("}", file=mms_out)
    print("", file=mms_out)
    print("", file=mms_out)

    print("std::vector<double> af1(double mu, double t_k, double Deltat, double x_j, double Deltax) {", file=mms_out)
    print("", file=mms_out)
    print("std::vector<double> af1_vals (4);", file=mms_out)
    print("", file=mms_out)
    print("af1_vals[0] = ", AF1_Lint_timeav, ";", file=mms_out)
    print("af1_vals[1] = ", AF1_Rint_timeav, ";", file=mms_out)
    print("af1_vals[2] = ", AF1_Lint.subs(t, t_k+Deltat/2), ";", file=mms_out)
    print("af1_vals[3] = ", AF1_Rint.subs(t, t_k+Deltat/2), ";", file=mms_out)
    print("", file=mms_out)
    print("return( af1_vals );", file=mms_out)
    print("}", file=mms_out)
    print("", file=mms_out)
    print("std::vector<double> af2(double mu, double t_k, double Deltat, double x_j, double Deltax) {", file=mms_out)
    print("", file=mms_out)
    print("std::vector<double> af2_vals (4);", file=mms_out)
    print("", file=mms_out)
    print("af2_vals[0] = ", AF2_Lint_timeav, ";", file=mms_out)
    print("af2_vals[1] = ", AF2_Rint_timeav, ";", file=mms_out)
    print("af2_vals[2] = ", AF2_Lint.subs(t, t_k+Deltat/2), ";", file=mms_out)
    print("af2_vals[3] = ", AF2_Rint.subs(t, t_k+Deltat/2), ";", file=mms_out)
    print("", file=mms_out)
    print("return( af2_vals );", file=mms_out)
    print("}", file=mms_out)
    print("", file=mms_out)
    mms_out.close



    AF1_Lint_timeav_func = sym.lambdify((x_j, Deltax, t_k, Deltat, mu),  AF1_Lint_timeav, "numpy")
    AF1_Rint_timeav_func = sym.lambdify((x_j, Deltax, t_k, Deltat, mu),  AF1_Rint_timeav, "numpy")
    AF2_Lint_timeav_func = sym.lambdify((x_j, Deltax, t_k, Deltat, mu),  AF2_Lint_timeav, "numpy")
    AF2_Rint_timeav_func = sym.lambdify((x_j, Deltax, t_k, Deltat, mu),  AF2_Rint_timeav, "numpy")

    AF1_Lint_func = sym.lambdify((x_j, Deltax, t_k, Deltat, mu),  AF1_Lint.subs(t, t_k+Deltat/2), "numpy")
    AF1_Rint_func = sym.lambdify((x_j, Deltax, t_k, Deltat, mu),  AF1_Rint.subs(t, t_k+Deltat/2), "numpy")
    AF2_Lint_func = sym.lambdify((x_j, Deltax, t_k, Deltat, mu),  AF2_Lint.subs(t, t_k+Deltat/2), "numpy")
    AF2_Rint_func = sym.lambdify((x_j, Deltax, t_k, Deltat, mu),  AF2_Rint.subs(t, t_k+Deltat/2), "numpy")

    a = 1/math.sqrt(3)
    N_angle = 2

    [angles, weights] = np.polynomial.legendre.leggauss(N_angle)
    #angles = np.array((-.57735, .57735))
    N_cell = 10
    dx = .2
    space_center = np.linspace(0,N_cell*dx, N_cell)
    space_plot = np.linspace(0,N_cell*dx, 2*N_cell)
    cell_step = 3/N_cell
    N_time = 1
    dt = .3
    time = np.linspace (0,dt*N_time, N_time)
    
    time_step = 5/N_time

    af = np.zeros((2*N_time, 2, N_angle, 2*N_cell))

    af_long = np.zeros((2*N_time * 2 * N_angle * 2*N_cell))

    sf = np.zeros((2*N_time, 2, 2*N_cell))

    for t in range(N_time):
        for i in range(N_cell):
            for m in range(N_angle):

                #print(AF1_Lint_timeav_func(space_center[i], cell_step, time[t], time_step, angles[m]))
                af[2*t, 0, m, 2*i  ] = AF1_Lint_timeav_func(space_center[i], cell_step, time[t], time_step, angles[m])
                af[2*t, 0, m, 2*i+1] = AF1_Rint_timeav_func(space_center[i], cell_step, time[t], time_step, angles[m])

                #print(AF1_Lint_func(space_center[i], cell_step, time[t], time_step, angles[m]))
                af[2*t+1, 0, m, 2*i  ] = AF1_Lint_func(space_center[i], cell_step, time[t], time_step, angles[m])
                af[2*t+1, 0, m, 2*i+1] = AF1_Rint_func(space_center[i], cell_step, time[t], time_step, angles[m])

                af[2*t, 1, m, 2*i  ] = AF2_Lint_timeav_func(space_center[i], cell_step, time[t], time_step, angles[m])
                af[2*t, 1, m, 2*i+1] = AF2_Rint_timeav_func(space_center[i], cell_step, time[t], time_step, angles[m])

                af[2*t+1, 1, m, 2*i  ] = AF2_Lint_func(space_center[i], cell_step, time[t], time_step, angles[m])
                af[2*t+1, 1, m, 2*i+1] = AF2_Rint_func(space_center[i], cell_step, time[t], time_step, angles[m])


                sf[2*t, 0, 2*i  ] += weights[m] * AF1_Lint_timeav_func(space_center[i], cell_step, time[t], time_step, angles[m])
                sf[2*t, 0, 2*i+1] += weights[m] * AF1_Rint_timeav_func(space_center[i], cell_step, time[t], time_step, angles[m])

                #print(AF1_Lint_func(space_center[i], cell_step, time[t], time_step, angles[m]))
                sf[2*t+1, 0, 2*i  ] += weights[m] * AF1_Lint_func(space_center[i], cell_step, time[t], time_step, angles[m])
                sf[2*t+1, 0, 2*i+1] = weights[m] * AF1_Rint_func(space_center[i], cell_step, time[t], time_step, angles[m])

                sf[2*t, 1, 2*i  ] += weights[m] * AF2_Lint_timeav_func(space_center[i], cell_step, time[t], time_step, angles[m])
                sf[2*t, 1, 2*i+1] += weights[m] * AF2_Rint_timeav_func(space_center[i], cell_step, time[t], time_step, angles[m])

                sf[2*t+1, 1, 2*i  ] += weights[m] * AF2_Lint_func(space_center[i], cell_step, time[t], time_step, angles[m])
                sf[2*t+1, 1, 2*i+1] += weights[m] * AF2_Rint_func(space_center[i], cell_step, time[t], time_step, angles[m])

                for g in range(2):
                    helper = 4*i*2*N_angle + 4*g*N_angle + 4*m
                    if g==0:
                        af_long[helper+0] = AF1_Lint_timeav_func(space_center[i], cell_step, time[t], time_step, angles[m])
                        af_long[helper+1] = AF1_Rint_timeav_func(space_center[i], cell_step, time[t], time_step, angles[m])
                        af_long[helper+2] = AF1_Lint_func(space_center[i], cell_step, time[t], time_step, angles[m])
                        af_long[helper+3] = AF1_Rint_func(space_center[i], cell_step, time[t], time_step, angles[m])
                    elif g==1:
                        af_long[helper+0] = AF2_Lint_timeav_func(space_center[i], cell_step, time[t], time_step, angles[m])
                        af_long[helper+1] = AF2_Rint_timeav_func(space_center[i], cell_step, time[t], time_step, angles[m])
                        af_long[helper+2] = AF2_Lint_func(space_center[i], cell_step, time[t], time_step, angles[m])
                        af_long[helper+3] = AF2_Rint_func(space_center[i], cell_step, time[t], time_step, angles[m])

    # Flux - average
    fig = plt.figure()
    plt.plot(space_plot, sf[0,0,:])
    plt.plot(space_plot, sf[0,1,:])
    plt.show()

    print (af_long)

    print(angles)