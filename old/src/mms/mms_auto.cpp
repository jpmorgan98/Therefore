#include <iostream>
#include <vector>
//File auto generated

std::vector<double> Q1(double v1, double v2, double Sigma_1, double Sigma_2, double mu, double Sigma_S1, double Sigma_S2, double Sigma_S12, double Sigma_S21, double x_j, double Deltax, double t_k, double Deltat) {

std::vector<double> Q1_vals (4);

Q1_vals[0] =  Sigma_S21*pow(x_j, 3)*(-Deltat/2 + t_k)/6 + Sigma_S21*pow(x_j, 3)*(Deltat/2 + t_k)/3 - Sigma_S21*(-Deltat/2 + t_k)*pow(-Deltax/2 + x_j, 3)/6 - Sigma_S21*(Deltat/2 + t_k)*pow(-Deltax/2 + x_j, 3)/3 + 3*pow(x_j, 2)*(Sigma_1 + Sigma_S1/2)/2 - 3*pow(-Deltax/2 + x_j, 2)*(Sigma_1 + Sigma_S1/2)/2 + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 ;
Q1_vals[1] =  -Sigma_S21*pow(x_j, 3)*(-Deltat/2 + t_k)/6 - Sigma_S21*pow(x_j, 3)*(Deltat/2 + t_k)/3 + Sigma_S21*(-Deltat/2 + t_k)*pow(Deltax/2 + x_j, 3)/6 + Sigma_S21*(Deltat/2 + t_k)*pow(Deltax/2 + x_j, 3)/3 - 3*pow(x_j, 2)*(Sigma_1 + Sigma_S1/2)/2 + 3*pow(Deltax/2 + x_j, 2)*(Sigma_1 + Sigma_S1/2)/2 - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 ;
Q1_vals[2] =  Sigma_S21*pow(x_j, 3)*(Deltat/2 + t_k)/3 - Sigma_S21*(Deltat/2 + t_k)*pow(-Deltax/2 + x_j, 3)/3 + pow(x_j, 2)*(Sigma_1 + Sigma_S1/2) - pow(-Deltax/2 + x_j, 2)*(Sigma_1 + Sigma_S1/2) + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 ;
Q1_vals[3] =  -Sigma_S21*pow(x_j, 3)*(Deltat/2 + t_k)/3 + Sigma_S21*(Deltat/2 + t_k)*pow(Deltax/2 + x_j, 3)/3 - pow(x_j, 2)*(Sigma_1 + Sigma_S1/2) + pow(Deltax/2 + x_j, 2)*(Sigma_1 + Sigma_S1/2) - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 ;

return( Q1_vals );
}

std::vector<double> Q2(double v1, double v2, double Sigma_1, double Sigma_2, double mu, double Sigma_S1, double Sigma_S2, double Sigma_S12, double Sigma_S21, double x_j, double Deltax, double t_k, double Deltat) {

std::vector<double> Q2_vals (4);

Q2_vals[0] =  pow(x_j, 2)*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 + pow(x_j, 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 + x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - pow(-Deltax/2 + x_j, 2)*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 - pow(-Deltax/2 + x_j, 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - (-Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 - (-Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + pow(x_j, 3)*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) + pow(x_j, 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) - pow(-Deltax/2 + x_j, 3)*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) - pow(-Deltax/2 + x_j, 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) ;
Q2_vals[1] =  -pow(x_j, 2)*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 - pow(x_j, 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 - x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + pow(Deltax/2 + x_j, 2)*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 + pow(Deltax/2 + x_j, 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + (Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 + (Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - pow(x_j, 3)*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) - pow(x_j, 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) + pow(Deltax/2 + x_j, 3)*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) + pow(Deltax/2 + x_j, 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) ;
Q2_vals[2] =  pow(x_j, 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - pow(-Deltax/2 + x_j, 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - (-Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + pow(x_j, 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) - pow(-Deltax/2 + x_j, 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) ;
Q2_vals[3] =  -pow(x_j, 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + pow(Deltax/2 + x_j, 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + (Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - pow(x_j, 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) + pow(Deltax/2 + x_j, 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) ;

return( Q2_vals );
}


std::vector<double> af1(double mu, double t_k, double Deltat, double x_j, double Deltax) {
std::cout << "in af" << std::endl;
std::vector<double> af1_vals (4);

af1_vals[0] =  3*pow(x_j, 2)/4 + x_j*(-Deltat/2 + mu + t_k + 1)/2 + x_j*(Deltat/2 + mu + t_k + 1) - 3*pow(-Deltax/2 + x_j, 2)/4 - (-Deltax/2 + x_j)*(-Deltat/2 + mu + t_k + 1)/2 - (-Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1) ;
af1_vals[1] =  -3*pow(x_j, 2)/4 - x_j*(-Deltat/2 + mu + t_k + 1)/2 - x_j*(Deltat/2 + mu + t_k + 1) + 3*pow(Deltax/2 + x_j, 2)/4 + (Deltax/2 + x_j)*(-Deltat/2 + mu + t_k + 1)/2 + (Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1) ;
af1_vals[2] =  pow(x_j, 2)/2 + x_j*(Deltat/2 + mu + t_k + 1) - pow(-Deltax/2 + x_j, 2)/2 - (-Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1) ;
af1_vals[3] =  -pow(x_j, 2)/2 - x_j*(Deltat/2 + mu + t_k + 1) + pow(Deltax/2 + x_j, 2)/2 + (Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1) ;
for (int p=0; p<af1_vals.size(); ++p) {
    std::cout << af1_vals[p] << std::endl;
    }

return( af1_vals );
}

std::vector<double> af2(double mu, double t_k, double Deltat, double x_j, double Deltax) {
std::cout << "in af2" << std::endl;
std::vector<double> af2_vals (4);

af2_vals[0] =  pow(x_j, 3)*(-Deltat/2 + t_k)/6 + pow(x_j, 3)*(Deltat/2 + t_k)/3 + 3*x_j*(mu + 1)/2 - (-Deltat/2 + t_k)*pow(-Deltax/2 + x_j, 3)/6 - (Deltat/2 + t_k)*pow(-Deltax/2 + x_j, 3)/3 - 3*(-Deltax/2 + x_j)*(mu + 1)/2 ;
af2_vals[1] =  -pow(x_j, 3)*(-Deltat/2 + t_k)/6 - pow(x_j, 3)*(Deltat/2 + t_k)/3 - 3*x_j*(mu + 1)/2 + (-Deltat/2 + t_k)*pow(Deltax/2 + x_j, 3)/6 + (Deltat/2 + t_k)*pow(Deltax/2 + x_j, 3)/3 + 3*(Deltax/2 + x_j)*(mu + 1)/2 ;
af2_vals[2] =  pow(x_j, 3)*(Deltat/2 + t_k)/3 + x_j*(mu + 1) - (Deltat/2 + t_k)*pow(-Deltax/2 + x_j, 3)/3 - (-Deltax/2 + x_j)*(mu + 1) ;
af2_vals[3] =  -pow(x_j, 3)*(Deltat/2 + t_k)/3 - x_j*(mu + 1) + (Deltat/2 + t_k)*pow(Deltax/2 + x_j, 3)/3 + (Deltax/2 + x_j)*(mu + 1) ;

return( af2_vals );
}



int main(){


    int N_space = 10;
    int N_angles = 2;
    int N_time = 1;
    int N_groups = 2;

    std::vector<double> x (N_space);
    std::vector<double> t (N_time);
    double dt = 0.3;
    double dx = 0.2;
    std::vector<double> angles{-0.57735027,  0.57735027};
    std::vector<double> weights{1.0, 1.0};

    std::vector<double> af (4*N_angles*N_space*N_groups);
    std::vector<double> Q (4*N_angles*N_space*N_groups);
    int k = 0;

    for (int i=1;i<N_space; ++i) {x[i] = x[i-1]+dx;}
    for (int i=1;i<N_time; ++i) {t[i] = t[i-1]+dx;}

    std::cout << "to big loop" <<std::endl;

    for (int i=0; i<N_space; ++i){
        for (int g=0; g<N_groups; ++g) {
            for (int m=0; m<N_angles; ++m){

                int helper = 4*i*N_groups*N_angles + 4*g*N_angles + 4*m;

                std::vector<double> temp;
                std::cout << "i="<< i << "  g=" << g << "  m=" << m <<std::endl;
                
                std::cout << angles[m] << std::endl;
                std::cout << t[k] << std::endl;
                std::cout << x[i] << std::endl; 

                if (g==0){
                    temp = af1(angles[m], t[k], dt, x[i], dx);
                } else if (g==1){
                    temp = af2(angles[m], t[k], dt, x[i], dx);
                }

                std::cout << "thru af" << std::endl;
                for (int p=0; p<temp.size(); ++p) std::cout << temp[p] << std::endl;

                af[helper+0] = temp[0];
                af[helper+1] = temp[1];
                af[helper+2] = temp[2];
                af[helper+3] = temp[3];
            }
        }
    }

    for (int i=0; i<af.size(); ++i){
        std::cout << af[i] << std::endl;
    }

    return( 1 );
}