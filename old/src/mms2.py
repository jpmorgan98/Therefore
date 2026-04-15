import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

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
    sigma1 = sym.Symbol('sigma1')
    sigma2 = sym.Symbol('sigma2')

    # Within group scattering cross sections
    sigmaS1 = sym.Symbol('sigmaS1')
    sigmaS2 = sym.Symbol('sigmaS2')

    # Down scattering (from 1 into 2)
    sigmaS1_2 = sym.Symbol('sigmaS1_2')

    # arbitrary manufactured solution
    # note: technically psi1 = Ax + Bt + Cmu but they are all chosen to be 1
    psi1 = x + t + mu

    # doing some calc 
    dpsi1_dx = sym.diff(psi1, x)
    dpsi1_dt = sym.diff(psi1, t)
    # evaluated manually
    eval_intpsi1 = 2*(x+t)

    # group 2
    psi2 = x**2*t + mu

    # doing some calc 
    dpsi2_dx = sym.diff(psi2, x)
    dpsi2_dt = sym.diff(psi2, t)
    # evaluated manually
    eval_intpsi2 = 2*t*x**2

    # NTE for group 1
    L1 = (1/v1)*dpsi1_dt + mu*dpsi1_dx + sigma1*psi1
    #    w/in group               down scattering
    R1 = sigmaS1/2*eval_intpsi1 + sigmaS1_2 / 2 * eval_intpsi1
    Q1 = 2*(L1-R1)

    # NTE for group 2
    L2 = (1/v2)*dpsi2_dt + mu*dpsi2_dx + sigma2*psi2
    #    w/in group               down scattering
    R2 = sigmaS2/2*eval_intpsi2 - sigmaS1_2 / 2 * eval_intpsi2
    Q2 = 2*(L2-R2)


    print('Continuous function describing group 1 source')
    print(Q1)
    print()
    print()
    print('Continuous function describing group 2 source')
    print(Q2)
    print()

    #SCB-TDMB integrated terms
    #cell integrated time edge

    #cell integrated time average