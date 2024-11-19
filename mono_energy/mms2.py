import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
sym.init_printing(use_unicode=True)
import math

PRINT_TO_TERMINAL = False
PRINT_PYTHON = True

# note: all cross sections (sigma) are understood to be macroscopic
# group 1 is the fast group 2 is the slow
# to evaluate intergrals go to https://www.integral-calculator.com/

def sort(af_unsorted, N_cells, N_groups, N_angles):
    af = np.zeros((2, N_groups, N_angles, 2*N_cells))
    sf = np.zeros((2, N_groups, 2*N_cells))
    for g in range(N_groups):
        for i in range(N_cells):
            for m in range(N_angles):
                helper =  (i*(N_groups*N_angles*4) + g*(4*N_angles) + 4*m)

                af[0, g, m, 2*i  ] = af_unsorted[helper + 0]
                af[0, g, m, 2*i+1] = af_unsorted[helper + 1]
                af[1, g, m, 2*i  ] = af_unsorted[helper + 2]
                af[1, g, m, 2*i+1] = af_unsorted[helper + 3]

                sf[0, g, 2*i  ] += weights[m] * af_unsorted[helper + 0]
                sf[0, g, 2*i+1] += weights[m] * af_unsorted[helper + 1]
                sf[1, g, 2*i  ] += weights[m] * af_unsorted[helper + 2]
                sf[1, g, 2*i+1] += weights[m] * af_unsorted[helper + 3]

    return(af)


if __name__ == '__main__':

    # Independent variables
    x = sym.Symbol('x')
    t = sym.Symbol('t')
    mu = sym.Symbol('mu')

    # Group velocities
    v = sym.Symbol('v')

    # Group total cross sections
    sigma = sym.Symbol('sigma')

    # Within group scattering cross sections
    sigma_s = sym.Symbol('sigma_s')

    # arbitrary manufactured solution
    # note: technically psi1 = Ax + Bt + Cmu but they are all chosen to be 1

    psi1 = x + t + (mu + 1) # group 1

    # doing some calc g1
    dpsi1_dx = sym.diff(psi1, x)
    dpsi1_dt = sym.diff(psi1, t)
    
    # evaluated manually wrt mu, bounds -1 to 1
    eval_intpsi1 = 2*(x+t+1)

    # NTE for group 1
    L1 = (1/v)*dpsi1_dt + mu*dpsi1_dx + sigma*psi1
    #    w/in group               up scattering
    R1 = sigma_s/2*eval_intpsi1
    Q1 = (2*L1-R1)

    print(Q1)


    # discretizatin terms

    x_j = sym.Symbol('x_j')
    deltax = sym.Symbol('deltax')

    t_k = sym.Symbol('t_k')
    Deltat = sym.Symbol('Deltat')

    # cell integrated time edge
    Q1_Lint = sym.integrate(Q1, (x, x-deltax/2, x))
    Q1_Rint = sym.integrate(Q1, (x, x, x+deltax/2))
    #cell integrated time averaged sym.simplify
    Q1_Lint_timeav = Q1_Lint.subs(t, t_k+Deltat/2) + Q1_Lint.subs(t, t_k-Deltat/2) / 2
    Q1_Rint_timeav = Q1_Rint.subs(t, t_k+Deltat/2) + Q1_Rint.subs(t, t_k-Deltat/2) / 2

    # Same crap but for Angular flux terms

    print(Q1_Lint)
    print(Q1_Rint)

    exit()

    # cell integrated time edge
    AF1_Lint = sym.integrate(psi1, (x, x-Deltax/2, x))
    AF1_Rint = sym.integrate(psi1, (x, x, x+Deltax/2))
    #cell integrated time averaged
    AF1_Lint_timeav = AF1_Lint.subs(t, t_k+Deltat/2) + AF1_Lint.subs(t, t_k-Deltat/2) / 2
    AF1_Rint_timeav = AF1_Rint.subs(t, t_k+Deltat/2) + AF1_Rint.subs(t, t_k-Deltat/2) / 2

    print("here")
    print(Q1_Lint)
    print(Q1_Rint)
    
    if (PRINT_TO_TERMINAL):
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
        print()
        print()
        print("Cell integrated time edge values for Latex")
        print()
        print(sym.latex(Q1_Lint.subs(t, t_k+Deltat/2)))
        print(sym.latex(Q1_Rint.subs(t, t_k+Deltat/2)))
        print()
        print()
        print("Cell integrated time average values for export to a function")
        print()
        print(Q1_Lint_timeav)
        print(Q1_Rint_timeav)
        print()
        print()
        print("Cell integrated time average values for Latex")
        print()
        print(sym.latex(Q1_Lint_timeav))
        print(sym.latex(Q1_Rint_timeav))
        print()
        print()

        print("=========ANGULAR FLUXES=========")
        print()
        print()
        print("Cell integrated time edge values for export to a function")
        print()
        print(AF1_Lint.subs(t, t_k+Deltat/2))
        print(AF1_Rint.subs(t, t_k+Deltat/2))
        print()
        print()
        print("Cell integrated time edge values for Latex")
        print()
        print(sym.latex(AF1_Lint.subs(t, t_k+Deltat/2)))
        print(sym.latex(AF1_Rint.subs(t, t_k+Deltat/2)))
        print()
        print()

        print("Cell integrated time average values for export to a function")
        print()
        print(AF1_Lint_timeav)
        print(AF1_Rint_timeav)
        print()
        print()
        print("Cell integrated time average values for Latex")
        print()
        print(sym.latex(AF1_Lint_timeav))
        print(sym.latex(AF1_Rint_timeav))
        print()
        print()

    #### Python function output
    if PRINT_PYTHON:
        mms_out =  open("mms_auto2.py", "w")
        print("import numpy as np", file=mms_out)
        print("", file=mms_out)
        print("# File auto generated", file=mms_out)
        print("", file=mms_out)

        print("def Q1(v1, Sigma_1, Sigma_S1, mu, x_j, Deltax, t_k, Deltat):", file=mms_out)
        print("", file=mms_out)
        print("   Q1_vals = np.zeros(4)", file=mms_out)
        print("", file=mms_out)
        print("   Q1_vals[0] = ", Q1_Lint_timeav, "", file=mms_out)
        print("   Q1_vals[1] = ", Q1_Rint_timeav, "", file=mms_out)
        print("   Q1_vals[2] = ", Q1_Lint.subs(t, t_k+Deltat/2), "", file=mms_out)
        print("   Q1_vals[3] = ", Q1_Rint.subs(t, t_k+Deltat/2), "", file=mms_out)
        print("", file=mms_out)
        print("   return( Q1_vals )", file=mms_out)
        print("", file=mms_out)
        print("", file=mms_out)

        print("def af1(mu, t_k, Deltat, x_j, Deltax):", file=mms_out)
        print("", file=mms_out)
        print("   af1_vals = np.zeros(4)", file=mms_out)
        print("", file=mms_out)
        print("   af1_vals[0] = ", AF1_Lint_timeav, "", file=mms_out)
        print("   af1_vals[1] = ", AF1_Rint_timeav, "", file=mms_out)
        print("   af1_vals[2] = ", AF1_Lint.subs(t, t_k+Deltat/2), "", file=mms_out)
        print("   af1_vals[3] = ", AF1_Rint.subs(t, t_k+Deltat/2), "", file=mms_out)
        print("", file=mms_out)
        print("   return( af1_vals )", file=mms_out)
        print("", file=mms_out)
        print("", file=mms_out)
        mms_out.close

    AF1_Lint_timeav_func = sym.lambdify((x_j, Deltax, t_k, Deltat, mu),  AF1_Lint_timeav, "numpy")
    AF1_Rint_timeav_func = sym.lambdify((x_j, Deltax, t_k, Deltat, mu),  AF1_Rint_timeav, "numpy")

    AF1_Lint_func = sym.lambdify((x_j, Deltax, t_k, Deltat, mu),  AF1_Lint.subs(t, t_k+Deltat/2), "numpy")
    AF1_Rint_func = sym.lambdify((x_j, Deltax, t_k, Deltat, mu),  AF1_Rint.subs(t, t_k+Deltat/2), "numpy")

    Q1_Lint_timeav_func = sym.lambdify((v1, Sigma_1, Sigma_S1, x_j, Deltax, t_k, Deltat, mu),  Q1_Lint_timeav, "numpy")
    Q1_Rint_timeav_func = sym.lambdify((v1, Sigma_1, Sigma_S1, x_j, Deltax, t_k, Deltat, mu),  Q1_Rint_timeav, "numpy")

    Q1_Lint_func = sym.lambdify((v1, Sigma_1, Sigma_S1, x_j, Deltax, t_k, Deltat, mu),  Q1_Lint.subs(t, t_k+Deltat/2), "numpy")
    Q1_Rint_func = sym.lambdify((v1, Sigma_1, Sigma_S1, x_j, Deltax, t_k, Deltat, mu),  Q1_Rint.subs(t, t_k+Deltat/2), "numpy")

    a = 1/math.sqrt(3)
    N_angle = 2

    [angles, weights] = np.polynomial.legendre.leggauss(N_angle)
    #angles = np.array((-.57735, .57735))
    N_cell = 10
    dx = .1
    space_center = np.arange(dx/2,(N_cell)*dx, dx)
    print(space_center)
    space_plot = np.linspace(0,N_cell*dx, 2*N_cell)
    #dx = 3/N_cell
    N_time = 1
    dt = .1
    time = np.arange(0,dt*N_time, dt)

    sigma = 0.5
    sigmas = 0.1
    vel1 = 1.0

    print(time)
    
    time_step = 5/N_time

    af = np.zeros((2*N_time, 2, N_angle, 2*N_cell))
    Q = np.zeros((2*N_time, 1, 2*N_cell))

    af_long = np.zeros((2*N_time * 2 * N_angle * 2*N_cell))

    sf = np.zeros((2*N_time, 2, 2*N_cell))

    for t in range(N_time):
        print(t, time[t])
        for i in range(N_cell):
            for m in range(N_angle):

                #print(AF1_Lint_timeav_func(space_center[i], dx, time[t], time_step, angles[m]))
                af[2*t, 0, m, 2*i  ] = AF1_Lint_timeav_func(space_center[i], dx, time[t], dt, angles[m])
                af[2*t, 0, m, 2*i+1] = AF1_Rint_timeav_func(space_center[i], dx, time[t], dt, angles[m])

                #print(AF1_Lint_func(space_center[i], dx, time[t], time_step, angles[m]))
                af[2*t+1, 0, m, 2*i  ] = AF1_Lint_func(space_center[i], dx, time[t], dt, angles[m])
                af[2*t+1, 0, m, 2*i+1] = AF1_Rint_func(space_center[i], dx, time[t], dt, angles[m])

                Q[2*t, 0, 2*i  ] +=  weights[m]*Q1_Lint_timeav_func(vel1, sigma, sigmas, space_center[i], dx, time[t], dt, angles[m])
                Q[2*t, 0, 2*i+1] += weights[m]*Q1_Rint_timeav_func(vel1, sigma, sigmas, space_center[i], dx, time[t], dt, angles[m])

                #print(AF1_Lint_func(space_center[i], dx, time[t], time_step, angles[m]))
                Q[2*t+1, 0, 2*i  ] += weights[m]*Q1_Lint_func(vel1, sigma, sigmas, space_center[i], dx, time[t], dt, angles[m])
                Q[2*t+1, 0, 2*i+1] += weights[m]*Q1_Rint_func(vel1, sigma, sigmas, space_center[i], dx, time[t], dt, angles[m])


                #Q[2*t, 0, m, 2*i  ]   = (space_center[i], dx, time[t], time_step, angles[m])
                #Q[2*t, 0, m, 2*i+1]   = (space_center[i], dx, time[t], time_step, angles[m])
                #Q[2*t+1, 0, m, 2*i  ] = (space_center[i], dx, time[t], time_step, angles[m])
                #Q[2*t+1, 0, m, 2*i+1] = (space_center[i], dx, time[t], time_step, angles[m])
                #Q[2*t, 1, m, 2*i  ]   = (space_center[i], dx, time[t], time_step, angles[m])
                #Q[2*t, 1, m, 2*i+1]   = (space_center[i], dx, time[t], time_step, angles[m])
                #Q[2*t+1, 1, m, 2*i  ] = (space_center[i], dx, time[t], time_step, angles[m])
                #Q[2*t+1, 1, m, 2*i+1] = (space_center[i], dx, time[t], time_step, angles[m])

                sf[2*t, 0, 2*i  ] += weights[m] * AF1_Lint_timeav_func(space_center[i], dx, time[t], time_step, angles[m])
                sf[2*t, 0, 2*i+1] += weights[m] * AF1_Rint_timeav_func(space_center[i], dx, time[t], time_step, angles[m])

                #print(AF1_Lint_func(space_center[i], dx, time[t], time_step, angles[m]))
                sf[2*t+1, 0, 2*i  ] += weights[m] * AF1_Lint_func(space_center[i], dx, time[t], time_step, angles[m])
                sf[2*t+1, 0, 2*i+1] = weights[m] * AF1_Rint_func(space_center[i], dx, time[t], time_step, angles[m])

                for g in range(2):
                    helper = 4*N_angle*2*N_cell*t + 4*i*2*N_angle + 4*g*N_angle + 4*m
                    #(j*(ps.SIZE_cellBlocks) + g*(ps.SIZE_groupBlocks) + 4*m)
                    if g==0:
                        print(space_center[i], dx, time[t], dt, angles[m])
                        af_long[helper+0] = AF1_Lint_timeav_func(space_center[i], dx, time[t], dt, angles[m])
                        af_long[helper+1] = AF1_Rint_timeav_func(space_center[i], dx, time[t], dt, angles[m])
                        af_long[helper+2] = AF1_Lint_func(space_center[i], dx, time[t], dt, angles[m])
                        af_long[helper+3] = AF1_Rint_func(space_center[i], dx, time[t], dt, angles[m])

    af_n = sort(af_long, N_cell, 2, N_angle)


    print(af)
    print(af.size)
    # Flux - average
    fig = plt.figure()
    plt.plot(space_plot, Q[0, 0, :  ])
    plt.plot(space_plot, Q[1, 0, :  ])

    #plt.plot(space_plot, af[1, 0, 0, :  ])
    #plt.plot(space_plot, af[1, 0, 1, :  ])
    plt.show()

    print(af_long)