import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
sym.init_printing(use_unicode=True)

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

    psi1 = x + t + mu

    # doing some calc 
    dpsi1_dx = sym.diff(psi1, x)
    dpsi1_dt = sym.diff(psi1, t)
    # evaluated manually wrt mu, bounds -1 to 1
    eval_intpsi1 = 2*(x+t)

    # group 2
    psi2 = x**2*t + mu

    # doing some calc 
    dpsi2_dx = sym.diff(psi2, x)
    dpsi2_dt = sym.diff(psi2, t)
    # evaluated manually wrt mu, bounds -1 to 1
    eval_intpsi2 = 2*t*x**2

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