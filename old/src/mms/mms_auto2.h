#include <iostream>
#include <vector>
//File auto generated

std::vector<double> Q1(double v1, double v2, double Sigma_1, double Sigma_2, double mu, double Sigma_S1, double Sigma_S2, double Sigma_S12, double Sigma_S21, double x_j, double Deltax, double t_k, double Deltat) {

std::vector<double> Q1_vals (4);

Q1_vals[0] =  Sigma_S21*x_j**3*(-Deltat/2 + t_k)/6 + Sigma_S21*x_j**3*(Deltat/2 + t_k)/3 - Sigma_S21*(-Deltat/2 + t_k)*(-Deltax/2 + x_j)**3/6 - Sigma_S21*(Deltat/2 + t_k)*(-Deltax/2 + x_j)**3/3 + 3*x_j**2*(Sigma_1 + Sigma_S1/2)/2 - 3*(-Deltax/2 + x_j)**2*(Sigma_1 + Sigma_S1/2)/2 + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 ;
Q1_vals[1] =  -Sigma_S21*x_j**3*(-Deltat/2 + t_k)/6 - Sigma_S21*x_j**3*(Deltat/2 + t_k)/3 + Sigma_S21*(-Deltat/2 + t_k)*(Deltax/2 + x_j)**3/6 + Sigma_S21*(Deltat/2 + t_k)*(Deltax/2 + x_j)**3/3 - 3*x_j**2*(Sigma_1 + Sigma_S1/2)/2 + 3*(Deltax/2 + x_j)**2*(Sigma_1 + Sigma_S1/2)/2 - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 ;
Q1_vals[2] =  Sigma_S21*x_j**3*(Deltat/2 + t_k)/3 - Sigma_S21*(Deltat/2 + t_k)*(-Deltax/2 + x_j)**3/3 + x_j**2*(Sigma_1 + Sigma_S1/2) - (-Deltax/2 + x_j)**2*(Sigma_1 + Sigma_S1/2) + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 ;
Q1_vals[3] =  -Sigma_S21*x_j**3*(Deltat/2 + t_k)/3 + Sigma_S21*(Deltat/2 + t_k)*(Deltax/2 + x_j)**3/3 - x_j**2*(Sigma_1 + Sigma_S1/2) + (Deltax/2 + x_j)**2*(Sigma_1 + Sigma_S1/2) - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 ;

return( Q1_vals );
}

std::vector<double> Q2(double v1, double v2, double Sigma_1, double Sigma_2, double mu, double Sigma_S1, double Sigma_S2, double Sigma_S12, double Sigma_S21, double x_j, double Deltax, double t_k, double Deltat) {

std::vector<double> Q2_vals (4);

Q2_vals[0] =  x_j**2*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 + x_j**2*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 + x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - (-Deltax/2 + x_j)**2*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 - (-Deltax/2 + x_j)**2*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - (-Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 - (-Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + x_j**3*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) + x_j**3*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) - (-Deltax/2 + x_j)**3*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) - (-Deltax/2 + x_j)**3*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) ;
Q2_vals[1] =  -x_j**2*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 - x_j**2*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 - x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + (Deltax/2 + x_j)**2*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 + (Deltax/2 + x_j)**2*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + (Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 + (Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - x_j**3*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) - x_j**3*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) + (Deltax/2 + x_j)**3*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) + (Deltax/2 + x_j)**3*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) ;
Q2_vals[2] =  x_j**2*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - (-Deltax/2 + x_j)**2*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - (-Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + x_j**3*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) - (-Deltax/2 + x_j)**3*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) ;
Q2_vals[3] =  -x_j**2*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + (Deltax/2 + x_j)**2*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + (Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - x_j**3*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) + (Deltax/2 + x_j)**3*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) ;

return( Q2_vals );
}


std::vector<double> af1(double mu, double t_k, double Deltat, double x_j, double Deltax) {

std::vector<double> af1_vals (4);

af1_vals[0] =  3*x_j**2/4 + x_j*(-Deltat/2 + mu + t_k + 1)/2 + x_j*(Deltat/2 + mu + t_k + 1) - 3*(-Deltax/2 + x_j)**2/4 - (-Deltax/2 + x_j)*(-Deltat/2 + mu + t_k + 1)/2 - (-Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1) ;
af1_vals[1] =  -3*x_j**2/4 - x_j*(-Deltat/2 + mu + t_k + 1)/2 - x_j*(Deltat/2 + mu + t_k + 1) + 3*(Deltax/2 + x_j)**2/4 + (Deltax/2 + x_j)*(-Deltat/2 + mu + t_k + 1)/2 + (Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1) ;
af1_vals[2] =  x_j**2/2 + x_j*(Deltat/2 + mu + t_k + 1) - (-Deltax/2 + x_j)**2/2 - (-Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1) ;
af1_vals[3] =  -x_j**2/2 - x_j*(Deltat/2 + mu + t_k + 1) + (Deltax/2 + x_j)**2/2 + (Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1) ;

return( af1_vals );
}

std::vector<double> af2(double mu, double t_k, double Deltat, double x_j, double Deltax) {

std::vector<double> af2_vals (4);

af2_vals[0] =  x_j**3*(-Deltat/2 + t_k)/6 + x_j**3*(Deltat/2 + t_k)/3 + 3*x_j*(mu + 1)/2 - (-Deltat/2 + t_k)*(-Deltax/2 + x_j)**3/6 - (Deltat/2 + t_k)*(-Deltax/2 + x_j)**3/3 - 3*(-Deltax/2 + x_j)*(mu + 1)/2 ;
af2_vals[1] =  -x_j**3*(-Deltat/2 + t_k)/6 - x_j**3*(Deltat/2 + t_k)/3 - 3*x_j*(mu + 1)/2 + (-Deltat/2 + t_k)*(Deltax/2 + x_j)**3/6 + (Deltat/2 + t_k)*(Deltax/2 + x_j)**3/3 + 3*(Deltax/2 + x_j)*(mu + 1)/2 ;
af2_vals[2] =  x_j**3*(Deltat/2 + t_k)/3 + x_j*(mu + 1) - (Deltat/2 + t_k)*(-Deltax/2 + x_j)**3/3 - (-Deltax/2 + x_j)*(mu + 1) ;
af2_vals[3] =  -x_j**3*(Deltat/2 + t_k)/3 - x_j*(mu + 1) + (Deltat/2 + t_k)*(Deltax/2 + x_j)**3/3 + (Deltax/2 + x_j)*(mu + 1) ;

return( af2_vals );
}

