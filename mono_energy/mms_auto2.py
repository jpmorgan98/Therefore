import numpy as np

# File auto generated

def Q1(v1, Sigma_1, Sigma_S1, mu, x_j, Deltax, t_k, Deltat):

   Q1_vals = np.zeros(4)

   Q1_vals[0] =  3*x_j**2*(Sigma_1 - Sigma_S1/2)/2 - 3*(-Deltax/2 + x_j)**2*(Sigma_1 - Sigma_S1/2)/2 + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 - Sigma_S1*v1*(-Deltat/2 + t_k) - Sigma_S1*v1 + 2*mu*v1 + 2)/(2*v1) + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 - Sigma_S1*v1*(Deltat/2 + t_k) - Sigma_S1*v1 + 2*mu*v1 + 2)/v1 - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 - Sigma_S1*v1*(-Deltat/2 + t_k) - Sigma_S1*v1 + 2*mu*v1 + 2)/(2*v1) - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 - Sigma_S1*v1*(Deltat/2 + t_k) - Sigma_S1*v1 + 2*mu*v1 + 2)/v1 
   Q1_vals[1] =  -3*x_j**2*(Sigma_1 - Sigma_S1/2)/2 + 3*(Deltax/2 + x_j)**2*(Sigma_1 - Sigma_S1/2)/2 - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 - Sigma_S1*v1*(-Deltat/2 + t_k) - Sigma_S1*v1 + 2*mu*v1 + 2)/(2*v1) - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 - Sigma_S1*v1*(Deltat/2 + t_k) - Sigma_S1*v1 + 2*mu*v1 + 2)/v1 + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 - Sigma_S1*v1*(-Deltat/2 + t_k) - Sigma_S1*v1 + 2*mu*v1 + 2)/(2*v1) + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 - Sigma_S1*v1*(Deltat/2 + t_k) - Sigma_S1*v1 + 2*mu*v1 + 2)/v1 
   Q1_vals[2] =  x_j**2*(Sigma_1 - Sigma_S1/2) - (-Deltax/2 + x_j)**2*(Sigma_1 - Sigma_S1/2) + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 - Sigma_S1*v1*(Deltat/2 + t_k) - Sigma_S1*v1 + 2*mu*v1 + 2)/v1 - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 - Sigma_S1*v1*(Deltat/2 + t_k) - Sigma_S1*v1 + 2*mu*v1 + 2)/v1 
   Q1_vals[3] =  -x_j**2*(Sigma_1 - Sigma_S1/2) + (Deltax/2 + x_j)**2*(Sigma_1 - Sigma_S1/2) - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 - Sigma_S1*v1*(Deltat/2 + t_k) - Sigma_S1*v1 + 2*mu*v1 + 2)/v1 + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 - Sigma_S1*v1*(Deltat/2 + t_k) - Sigma_S1*v1 + 2*mu*v1 + 2)/v1 

   return( Q1_vals )


def af1(mu, t_k, Deltat, x_j, Deltax):

   af1_vals = np.zeros(4)

   af1_vals[0] =  3*x_j**2/4 + x_j*(-Deltat/2 + mu + t_k + 1)/2 + x_j*(Deltat/2 + mu + t_k + 1) - 3*(-Deltax/2 + x_j)**2/4 - (-Deltax/2 + x_j)*(-Deltat/2 + mu + t_k + 1)/2 - (-Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1) 
   af1_vals[1] =  -3*x_j**2/4 - x_j*(-Deltat/2 + mu + t_k + 1)/2 - x_j*(Deltat/2 + mu + t_k + 1) + 3*(Deltax/2 + x_j)**2/4 + (Deltax/2 + x_j)*(-Deltat/2 + mu + t_k + 1)/2 + (Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1) 
   af1_vals[2] =  x_j**2/2 + x_j*(Deltat/2 + mu + t_k + 1) - (-Deltax/2 + x_j)**2/2 - (-Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1) 
   af1_vals[3] =  -x_j**2/2 - x_j*(Deltat/2 + mu + t_k + 1) + (Deltax/2 + x_j)**2/2 + (Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1) 

   return( af1_vals )


