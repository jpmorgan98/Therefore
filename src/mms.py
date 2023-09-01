# method of manufactured solutions scattering cross-section independent

import numpy as np
import matplotlib.pyplot as plt
import math



class mms:
    """breif: 
        All the parameters to implement the method of manufactured solutions for a simple corner balance time dependent multiple balance scheme in 2 groups

        RETURN CODES
            -Qa time edge left half volume integrated term for
            -Qb time edge right half volume integrated term
            -Qc time averaged left half volume integrated term
            -Qd time averaged right half volume integrated term
        """

    def __init__(self,A,B,C,D,F,v):
        """constant terms for the whole of computation"""
        self.A=A
        self.B=B
        self.C=C
        self.D=D
        self.F=F
        self.v=v

    #
    # GROUP 1
    #

    def group1afCONT(self,mu,x,t):
        """the continuous angular flux function for group 1 """
        return ( self.A*math.cos(mu) + self.B*x**2*t ) 

    def group1afUNINT():
        return ( )

    def group1sourceUNINT(self,mu,x,t,sigma):
        """the un-evaluated integral of the source term used to take our continuous source function into the proper democratization for SCB-TDMB"""
        return ( (2*(self.B*sigma*t*self.v+self.B)*x**3)/(3*self.v)+2*self.B*mu*t*x**2+2*sigma*math.acos(mu)*x )

    def group1source(self,x,dx,t,dt,mu,sigma):
        """actually evaluating the integrals using the un-evaluated term and returning them as values"""
        x_m = x-dx/2
        x_i = x
        x_p = x+dx/2

        t_p = t+dt/2
        t_m = t-dt/2

        # evaluating the integral over dx_i-1/2
        Qa = self.group1sourceUNINT(mu,x_i,t_p, sigma) - self.group1sourceUNINT(mu,x_m,t_p,sigma)
        # evaluating the integral over dx_i+1/2
        Qb = self.group1sourceUNINT(mu,x_p,t_p, sigma) - self.group1sourceUNINT(mu,x_i,t_p,sigma)
        # finding the time averaged integral dx_i-1/2 
        Qc = ( Qa + (self.group1sourceUNINT(mu,x_i,t_m, sigma) - self.group1sourceUNINT(mu,x_m,t_m,sigma)) ) / 2
        # finding the time averaged integral dx_i+1/2 
        Qd = ( Qb +  self.group1sourceUNINT(mu,x_p,t_m, sigma) - self.group1sourceUNINT(mu,x_i,t_m,sigma) ) / 2

        return( Qa, Qb, Qc, Qd )
    

    #
    # GROUP 2
    #
    
    def group2afCONT(self,mu,x,t):
        """the continuous angular flux function for group 2 """
        return ( self.C * math.exp( -self.F*x*t ) + self.D*mu**2 ) 

    def group2afUNINT():
        return ( )

    def group2sourceCONT(self, mu,x,t,sigma):
        """continuous source from MMS"""
        return( (2*math.exp(-self.F*t*x)*(self.D*mu*2*sigma*self.v*math.exp(self.F*t*x)-self.C*self.F*x-self.C*(self.F*mu*t-sigma)*self.v))/self.v )

    def group2sourceUNINT(self,mu,x,t,sigma):
        """the un-evaluated integral (dx) of the source term used to take our continuous source function into the proper discretization for SCB-TDMB"""
        return ( (2*((self.C*(self.F*t*x+1)*math.exp(-self.F*t*x))/(self.F*t**2)-(self.C*sigma*self.v*math.exp(-self.F*t*x))/(self.F*t)+self.C*mu*self.v*math.exp(-self.F*t*x)+self.D*mu**2*sigma*self.v*x))/self.v )

    def group2source(self,x,dx,t,dt,mu,sigma):
        """actually evaluating the integrals using the un-evaluated term and returning them as values"""
        x_m = x-dx/2
        x_i = x
        x_p = x+dx/2

        t_p = t+dt/2
        t_m = t-dt/2

        Qa = self.group2sourceUNINT(mu,x_i,t_p, sigma) - self.group2sourceUNINT(mu,x_m,t_p,sigma)
        Qb = self.group2sourceUNINT(mu,x_p,t_p, sigma) - self.group2sourceUNINT(mu,x_i,t_p,sigma)
        Qc = ( Qa + (self.group2sourceUNINT(mu,x_i,t_m, sigma) - self.group2sourceUNINT(mu,x_m,t_m,sigma)) ) / 2
        Qd = ( Qb +  self.group2sourceUNINT(mu,x_p,t_m, sigma) - self.group2sourceUNINT(mu,x_i,t_m,sigma) ) / 2

        return( Qa, Qb, Qc, Qd )
    
    def group2af(self,x,dx,t,dt,mu,sigma):
        """The proper spatial integral and time edge values of the chosen continuous angular flux"""
        return( )
    


if __name__ == "__main__":
    solution = mms(1,1,1,1,1,1)

    N_angles = 4
    N_cells = 100
    N_time = 2

    x = np.linspace(0,5,N_cells)
    xp = np.linspace(0,5,N_cells*2)
    dx = x[1] # assuming we start from zero
    t = np.array([0,1])
    dt = t[1] # assuming we start from zero

    [mus, weights] = np.polynomial.legendre.leggauss(N_angles)

    sigma = 2

    # quadrature order, groups, timestep, spatial cells
    source = np.zeros([N_angles, 2, 2*N_time, 2*N_cells])

    for m in range(N_angles):
        for k in range(N_time):
            for i in range(N_cells):
                temp = solution.group1source(x[i], dx, t[k], dt, mus[m], sigma)
                source[m,0,k*2,i*2] = temp[0]
                source[m,0,k*2,i*2+1] = temp[1]
                source[m,0,k*2+1,i*2] = temp[2]
                source[m,0,k*2+1,i*2+1] = temp[3]

                temp = solution.group2source(x[i], dx, t[k], dt, mus[m], sigma/2)
                source[m,1,k*2,i*2] = temp[0]
                source[m,1,k*2,i*2+1] = temp[1]
                source[m,1,k*2+1,i*2] = temp[2]
                source[m,1,k*2+1,i*2+1] = temp[3]
    
    plt.plot(xp, source[0,0,1,:])
    plt.plot(xp, source[0,1,1,:])
    
    plt.show()