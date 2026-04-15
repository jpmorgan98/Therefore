import numpy as np
import numba as nb

@nb.jit(nopython=True)
def A_neg(dx, v, dt, mu, xsec_total):
    assert(mu < 0)
    gamma = (dx*xsec_total)/2
    timer = dx/(v*dt)
    timer2 = dx/(2*v*dt)
    a = mu/2

    A_n = np.array([[-a + gamma, a,          timer2,            0],
                    [-a,         -a + gamma, 0,                 timer2],
                    [-timer,     0,          timer - a + gamma, a],
                    [0,          -timer,     -a,                timer -a + gamma]])
    
    return(A_n)

@nb.jit(nopython=True)
def A_pos(dx, v, dt, mu, xsec_total):
    assert(mu > 0)
    gamma = (dx*xsec_total)/2
    timer = dx/(v*dt)
    timer2 = dx/(2*v*dt)
    a = mu/2

    A_p = np.array([[a + gamma, a,         timer2,            0],
                    [-a,        a + gamma, 0,                 timer2],
                    [-timer,    0,         timer + a + gamma, a],
                    [0,         -timer,    -a,                timer +a + gamma]])

    return(A_p)



@nb.jit(nopython=True)
def c_neg(dx, v, dt, mu, Ql, Qr, Q_halfNext_L, Q_halfNext_R, psi_halfLast_L, psi_halfLast_R, psi_rightBound, psi_halfNext_rightBound):
    timer2 = dx/(2*v*dt)
    

    c_n = np.array([[dx/1*Ql + timer2*psi_halfLast_L],
                    [dx/1*Qr + timer2*psi_halfLast_R - mu* psi_rightBound],
                    [dx/1*Q_halfNext_L],
                    [dx/1*Q_halfNext_R - mu*psi_halfNext_rightBound]])

    return(c_n)


@nb.jit(nopython=True)
def c_pos(dx, v, dt, mu, Ql, Qr, Q_halfNext_L, Q_halfNext_R, psi_halfLast_L, psi_halfLast_R, psi_leftBound, psi_halfNext_leftBound):
    timer2 = dx/(2*v*dt)

    c_p = np.array([[dx/1*Ql + timer2*psi_halfLast_L + mu * psi_leftBound],
                    [dx/1*Qr + timer2*psi_halfLast_R],
                    [dx/1*Q_halfNext_L + mu*psi_halfNext_leftBound],
                    [dx/1*Q_halfNext_R]])
    
    return(c_p)



@nb.jit(nopython=True)
def b_pos(dx, v, dt, mu, Ql, Qr, Q_halfNext_L, Q_halfNext_R, psi_halfLast_L, psi_halfLast_R, psi_leftBound, psi_halfNext_leftBound, xsec_scatter, phi_L, phi_R, phi_halfNext_L, phi_halfNext_R):
    timer2 = dx/(2*v*dt)

    b_p = np.array([[dx/4*(xsec_scatter*phi_L + Ql) + timer2*psi_halfLast_L + mu*psi_leftBound],
                    [dx/4*(xsec_scatter*phi_R + Qr) + timer2*psi_halfLast_R],
                    [dx/4*(xsec_scatter*phi_halfNext_L + Q_halfNext_L) + mu*psi_halfNext_leftBound],
                    [dx/4*(xsec_scatter*phi_halfNext_R + Q_halfNext_R)]])

    return(b_p)

@nb.jit(nopython=True)
def b_neg(dx, v, dt, mu, Ql, Qr, Q_halfNext_L, Q_halfNext_R, psi_halfLast_L, psi_halfLast_R, psi_rightBound, psi_halfNext_rightBound, xsec_scatter, phi_L, phi_R, phi_halfNext_L, phi_halfNext_R):
    timer2 = dx/(2*v*dt)

    b_n = np.array([[dx/4*(xsec_scatter*phi_L + Ql) + timer2*psi_halfLast_L],
                    [dx/4*(xsec_scatter*phi_R + Qr) + timer2*psi_halfLast_R - mu*psi_rightBound],
                    [dx/4*(xsec_scatter*phi_halfNext_L + Q_halfNext_L) ],
                    [dx/4*(xsec_scatter*phi_halfNext_R + Q_halfNext_R) - mu*psi_halfNext_rightBound]])

    return(b_n)

@nb.jit(nopython=True)
def b_pos_interior(dx, v, dt, Ql, Qr, Q_halfNext_L, Q_halfNext_R, psi_halfLast_L, psi_halfLast_R, xsec_scatter, phi_L, phi_R, phi_halfNext_L, phi_halfNext_R):
    timer2 = dx/(2*v*dt)

    b_p = np.array([[dx/4*(xsec_scatter*phi_L + Ql) + timer2*psi_halfLast_L],
                    [dx/4*(xsec_scatter*phi_R + Qr) + timer2*psi_halfLast_R],
                    [dx/4*(xsec_scatter*phi_halfNext_L + Q_halfNext_L)],
                    [dx/4*(xsec_scatter*phi_halfNext_R + Q_halfNext_R)]])

    return(b_p)

@nb.jit(nopython=True)
def b_neg_interior(dx, v, dt, Ql, Qr, Q_halfNext_L, Q_halfNext_R, psi_halfLast_L, psi_halfLast_R, xsec_scatter, phi_L, phi_R, phi_halfNext_L, phi_halfNext_R):
    timer2 = dx/(2*v*dt)

    b_n = np.array([[dx/4*(xsec_scatter*phi_L + Ql) + timer2*psi_halfLast_L],
                    [dx/4*(xsec_scatter*phi_R + Qr) + timer2*psi_halfLast_R],
                    [dx/4*(xsec_scatter*phi_halfNext_L + Q_halfNext_L) ],
                    [dx/4*(xsec_scatter*phi_halfNext_R + Q_halfNext_R)]])
    return(b_n)

#@nb.jit(nopython=True)
def scatter_source(dx, xsec_scattering, N, w):
    S = np.zeros((4*N,4*N))
    beta = dx*xsec_scattering/4

    for i in range(N):
        for j in range(N):
            S[i*4,   j*4]   = beta*w[j]
            S[i*4+1, j*4+1] = beta*w[j]
            S[i*4+2, j*4+2] = beta*w[j]
            S[i*4+3, j*4+3] = beta*w[j]
    return(S)



if __name__ == '__main__':
    S = scatter_source(1, 4, 2, np.array([-2,-1,1,2]))
    print(S)
