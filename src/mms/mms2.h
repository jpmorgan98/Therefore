
#include <iostream>
#include <vector>
/*
Method of manufactured soultions

*/



//=========SOURCES=========



std::vector<double> Q_continuous(double v1, double v2, double Sigma_1, double Sigma_2, double mu, double t, double x, double Sigma_S1, double Sigma_S2, double Sigma_S12, double Sigma_S21){
    double Q1 = (v1*(2*Sigma_1*(mu + t + x) - Sigma_S1*(t + x) - Sigma_S21*t*pow(x, 2) + 2*mu) + 2)/v1;
    double Q2  = (v2*(2*Sigma_2*(mu + t*pow(x,2)) + Sigma_S12*(t + x) - Sigma_S2*t*pow(x, 2) + 4*mu*t*x) + 2*pow(x, 2))/v2;

    std::vector<double> ret = {Q1, Q2};

    return( ret );
}


// Cell integrated time edge values for export to a function
std::vector<double> Q_cellintegrated_timeedge(double v1, double v2, double Sigma_1, double Sigma_2, double mu, double Sigma_S1, double Sigma_S2, double Sigma_S12, double Sigma_S21, double x_j, double Deltax, double t_k, double Deltat){
    
    std::vector<double> Q_cellintegrated_timeedge_val (4);
    
    Q_cellintegrated_timeedge_val[0] = Sigma_S21*pow(x_j , 3)*(Deltat/2 + t_k)/3 - Sigma_S21*(Deltat/2 + t_k)*pow(-Deltax/2 + x_j , 3)/3 + pow(x_j , 2)*(Sigma_1 + Sigma_S1/2) - pow(-Deltax/2 + x_j , 2)*(Sigma_1 + Sigma_S1/2) + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1;
    Q_cellintegrated_timeedge_val[1] = -Sigma_S21*pow(x_j , 3)*(Deltat/2 + t_k)/3 + Sigma_S21*(Deltat/2 + t_k)*pow(Deltax/2 + x_j , 3)/3 - pow(x_j , 2)*(Sigma_1 + Sigma_S1/2) + pow(Deltax/2 + x_j , 2)*(Sigma_1 + Sigma_S1/2) - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1;
    Q_cellintegrated_timeedge_val[2] = pow(x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - pow(-Deltax/2 + x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - (-Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + pow(x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) - pow(-Deltax/2 + x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2);
    Q_cellintegrated_timeedge_val[3] = -pow(x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + pow(Deltax/2 + x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + (Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - pow(x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) + pow(Deltax/2 + x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2);

    return( Q_cellintegrated_timeedge_val );

}


// Cell integrated time average values for export to a function
std::vector<double> Q_cellintegrated_timeaverage(double v1, double v2, double Sigma_1, double Sigma_2, double mu, double Sigma_S1, double Sigma_S2, double Sigma_S12, double Sigma_S21, double x_j, double Deltax, double t_k, double Deltat){
    

    std::vector<double> Q_cellintegrated_timeaverage_val (4);


    Q_cellintegrated_timeaverage_val[0] = Sigma_S21*pow(x_j , 3)*(-Deltat/2 + t_k)/6 + Sigma_S21*pow(x_j , 3)*(Deltat/2 + t_k)/3 - Sigma_S21*(-Deltat/2 + t_k)*pow(-Deltax/2 + x_j , 3)/6 - Sigma_S21*(Deltat/2 + t_k)*pow(-Deltax/2 + x_j , 3)/3 + 3*pow(x_j , 2)*(Sigma_1 + Sigma_S1/2)/2 - 3*pow(-Deltax/2 + x_j , 2)*(Sigma_1 + Sigma_S1/2)/2 + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1;
    Q_cellintegrated_timeaverage_val[1] = -Sigma_S21*pow(x_j , 3)*(-Deltat/2 + t_k)/6 - Sigma_S21*pow(x_j , 3)*(Deltat/2 + t_k)/3 + Sigma_S21*(-Deltat/2 + t_k)*pow(Deltax/2 + x_j , 3)/6 + Sigma_S21*(Deltat/2 + t_k)*pow(Deltax/2 + x_j , 3)/3 - 3*pow(x_j , 2)*(Sigma_1 + Sigma_S1/2)/2 + 3*pow(Deltax/2 + x_j , 2)*(Sigma_1 + Sigma_S1/2)/2 - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1;
    Q_cellintegrated_timeaverage_val[2] = pow(x_j , 2)*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 + pow(x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 + x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - pow(-Deltax/2 + x_j , 2)*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 - pow(-Deltax/2 + x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - (-Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 - (-Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + pow(x_j , 3)*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) + pow(x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) - pow(-Deltax/2 + x_j , 3)*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) - pow(-Deltax/2 + x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2);
    Q_cellintegrated_timeaverage_val[3] = -pow(x_j , 2)*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 - pow(x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 - x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + pow(Deltax/2 + x_j , 2)*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 + pow(Deltax/2 + x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + (Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 + (Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - pow(x_j , 3)*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) - pow(x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) + pow(Deltax/2 + x_j , 3)*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) + pow(Deltax/2 + x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2);

    return(Q_cellintegrated_timeaverage_val);

}


std::vector<double> Q1(double v1, double v2, double Sigma_1, double Sigma_2, double mu, double Sigma_S1, double Sigma_S2, double Sigma_S12, double Sigma_S21, double x_j, double Deltax, double t_k, double Deltat){
    
    std::vector<double> Q1_vals (4);

    Q1_vals[0] = Sigma_S21*pow(x_j , 3)*(-Deltat/2 + t_k)/6 + Sigma_S21*pow(x_j , 3)*(Deltat/2 + t_k)/3 - Sigma_S21*(-Deltat/2 + t_k)*pow(-Deltax/2 + x_j , 3)/6 - Sigma_S21*(Deltat/2 + t_k)*pow(-Deltax/2 + x_j , 3)/3 + 3*pow(x_j , 2)*(Sigma_1 + Sigma_S1/2)/2 - 3*pow(-Deltax/2 + x_j , 2)*(Sigma_1 + Sigma_S1/2)/2 + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1;
    Q1_vals[1] = -Sigma_S21*pow(x_j , 3)*(-Deltat/2 + t_k)/6 - Sigma_S21*pow(x_j , 3)*(Deltat/2 + t_k)/3 + Sigma_S21*(-Deltat/2 + t_k)*pow(Deltax/2 + x_j , 3)/6 + Sigma_S21*(Deltat/2 + t_k)*pow(Deltax/2 + x_j , 3)/3 - 3*pow(x_j , 2)*(Sigma_1 + Sigma_S1/2)/2 + 3*pow(Deltax/2 + x_j , 2)*(Sigma_1 + Sigma_S1/2)/2 - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(-Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(-Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/(2*v1) + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1;
    Q1_vals[2] = Sigma_S21*pow(x_j , 3)*(Deltat/2 + t_k)/3 - Sigma_S21*(Deltat/2 + t_k)*pow(-Deltax/2 + x_j , 3)/3 + pow(x_j , 2)*(Sigma_1 + Sigma_S1/2) - pow(-Deltax/2 + x_j , 2)*(Sigma_1 + Sigma_S1/2) + x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 - (-Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1;
    Q1_vals[3] = -Sigma_S21*pow(x_j , 3)*(Deltat/2 + t_k)/3 + Sigma_S21*(Deltat/2 + t_k)*pow(Deltax/2 + x_j , 3)/3 - pow(x_j , 2)*(Sigma_1 + Sigma_S1/2) + pow(Deltax/2 + x_j , 2)*(Sigma_1 + Sigma_S1/2) - x_j*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1 + (Deltax/2 + x_j)*(2*Sigma_1*mu*v1 + 2*Sigma_1*v1*(Deltat/2 + t_k) + 2*Sigma_1*v1 + Sigma_S1*v1*(Deltat/2 + t_k) + Sigma_S1*v1 + Sigma_S21*v1 + 2*mu*v1 + 2)/v1;

    return( Q1_vals );

}


// Cell integrated time average values for export to a function
std::vector<double> Q2(double v1, double v2, double Sigma_1, double Sigma_2, double mu, double Sigma_S1, double Sigma_S2, double Sigma_S12, double Sigma_S21, double x_j, double Deltax, double t_k, double Deltat){
    

    std::vector<double> Q2_vals (4);

    Q2_vals[0] = pow(x_j , 2)*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 + pow(x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 + x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - pow(-Deltax/2 + x_j , 2)*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 - pow(-Deltax/2 + x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - (-Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 - (-Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + pow(x_j , 3)*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) + pow(x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) - pow(-Deltax/2 + x_j , 3)*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) - pow(-Deltax/2 + x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2);
    Q2_vals[1] = -pow(x_j , 2)*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 - pow(x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 - x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + pow(Deltax/2 + x_j , 2)*(-Sigma_S12/2 + 2*mu*(-Deltat/2 + t_k))/2 + pow(Deltax/2 + x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + (Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(-Deltat/2 + t_k) - Sigma_S12 + Sigma_S2)/2 + (Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - pow(x_j , 3)*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) - pow(x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) + pow(Deltax/2 + x_j , 3)*(2*Sigma_2*v2*(-Deltat/2 + t_k) + Sigma_S2*v2*(-Deltat/2 + t_k) + 2)/(6*v2) + pow(Deltax/2 + x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2);
    Q2_vals[2] = pow(x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - pow(-Deltax/2 + x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - (-Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + pow(x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) - pow(-Deltax/2 + x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2);
    Q2_vals[3] = -pow(x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) - x_j*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) + pow(Deltax/2 + x_j , 2)*(-Sigma_S12/2 + 2*mu*(Deltat/2 + t_k)) + (Deltax/2 + x_j)*(2*Sigma_2*mu + 2*Sigma_2 - Sigma_S12*(Deltat/2 + t_k) - Sigma_S12 + Sigma_S2) - pow(x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2) + pow(Deltax/2 + x_j , 3)*(2*Sigma_2*v2*(Deltat/2 + t_k) + Sigma_S2*v2*(Deltat/2 + t_k) + 2)/(3*v2);

    return( Q2_vals );

}

// =========ANGULAR FLUXES=========

// 

//Cell integrated time edge values for export to a function

std::vector<double> AF_cellintegrated_timeedge(double mu, double t_k, double Deltat, double x_j, double Deltax){

    std::vector<double> AF_cellintegrated_timeedge_val (4);

    AF_cellintegrated_timeedge_val[0] = pow(x_j , 2)/2 + x_j*(Deltat/2 + mu + t_k + 1) - pow(-Deltax/2 + x_j , 2)/2 - (-Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1);
    AF_cellintegrated_timeedge_val[1] = -pow(x_j , 2)/2 - x_j*(Deltat/2 + mu + t_k + 1) + pow(Deltax/2 + x_j , 2)/2 + (Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1);
    AF_cellintegrated_timeedge_val[2] = pow(x_j , 3)*(Deltat/2 + t_k)/3 + x_j*(mu + 1) - (Deltat/2 + t_k)*pow(-Deltax/2 + x_j , 3)/3 - (-Deltax/2 + x_j)*(mu + 1);
    AF_cellintegrated_timeedge_val[3] = -pow(x_j , 3)*(Deltat/2 + t_k)/3 - x_j*(mu + 1) + (Deltat/2 + t_k)*pow(Deltax/2 + x_j , 3)/3 + (Deltax/2 + x_j)*(mu + 1);

    return( AF_cellintegrated_timeedge_val );
}



// Cell integrated time average values for export to a function

std::vector<double> AF_cellintegrated_timeaverage(double mu, double t_k, double Deltat, double x_j, double Deltax){

    std::vector<double> AF_cellintegrated_timeaverage_val (4);

    AF_cellintegrated_timeaverage_val[0] = 3*pow(x_j , 2)/4 + x_j*(-Deltat/2 + mu + t_k + 1)/2 + x_j*(Deltat/2 + mu + t_k + 1) - 3*pow(-Deltax/2 + x_j , 2)/4 - (-Deltax/2 + x_j)*(-Deltat/2 + mu + t_k + 1)/2 - (-Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1);
    AF_cellintegrated_timeaverage_val[1] = -3*pow(x_j , 2)/4 - x_j*(-Deltat/2 + mu + t_k + 1)/2 - x_j*(Deltat/2 + mu + t_k + 1) + 3*pow(Deltax/2 + x_j , 2)/4 + (Deltax/2 + x_j)*(-Deltat/2 + mu + t_k + 1)/2 + (Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1);
    AF_cellintegrated_timeaverage_val[2] = pow(x_j , 3)*(-Deltat/2 + t_k)/6 + pow(x_j , 3)*(Deltat/2 + t_k)/3 + 3*x_j*(mu + 1)/2 - (-Deltat/2 + t_k)*pow(-Deltax/2 + x_j , 3)/6 - (Deltat/2 + t_k)*pow(-Deltax/2 + x_j , 3)/3 - 3*(-Deltax/2 + x_j)*(mu + 1)/2;
    AF_cellintegrated_timeaverage_val[3] = -pow(x_j , 3)*(-Deltat/2 + t_k)/6 - pow(x_j , 3)*(Deltat/2 + t_k)/3 - 3*x_j*(mu + 1)/2 + (-Deltat/2 + t_k)*pow(Deltax/2 + x_j , 3)/6 + (Deltat/2 + t_k)*pow(Deltax/2 + x_j , 3)/3 + 3*(Deltax/2 + x_j)*(mu + 1)/2;

    return( AF_cellintegrated_timeaverage_val );

}




//Cell integrated time edge values for export to a function

std::vector<double> AF_g1(double mu, double t_k, double Deltat, double x_j, double Deltax){

    std::vector<double> AF_g1_val (4);

    AF_g1_val[0] = 3*pow(x_j , 2)/4 + x_j*(-Deltat/2 + mu + t_k + 1)/2 + x_j*(Deltat/2 + mu + t_k + 1) - 3*pow(-Deltax/2 + x_j , 2)/4 - (-Deltax/2 + x_j)*(-Deltat/2 + mu + t_k + 1)/2 - (-Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1);
    AF_g1_val[1] = -3*pow(x_j , 2)/4 - x_j*(-Deltat/2 + mu + t_k + 1)/2 - x_j*(Deltat/2 + mu + t_k + 1) + 3*pow(Deltax/2 + x_j , 2)/4 + (Deltax/2 + x_j)*(-Deltat/2 + mu + t_k + 1)/2 + (Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1);
    AF_g1_val[2] = pow(x_j , 2)/2 + x_j*(Deltat/2 + mu + t_k + 1) - pow(-Deltax/2 + x_j , 2)/2 - (-Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1);
    AF_g1_val[3] = -pow(x_j , 2)/2 - x_j*(Deltat/2 + mu + t_k + 1) + pow(Deltax/2 + x_j , 2)/2 + (Deltax/2 + x_j)*(Deltat/2 + mu + t_k + 1);

    return( AF_g1_val );
}

// Cell integrated time average values for export to a function

std::vector<double> AF_g2(double mu, double t_k, double Deltat, double x_j, double Deltax){

    std::vector<double> AF_g2_val (4);

    AF_g2_val[0] = pow(x_j , 3)*(-Deltat/2 + t_k)/6 + pow(x_j , 3)*(Deltat/2 + t_k)/3 + 3*x_j*(mu + 1)/2 - (-Deltat/2 + t_k)*pow(-Deltax/2 + x_j , 3)/6 - (Deltat/2 + t_k)*pow(-Deltax/2 + x_j , 3)/3 - 3*(-Deltax/2 + x_j)*(mu + 1)/2;
    AF_g2_val[1] = -pow(x_j , 3)*(-Deltat/2 + t_k)/6 - pow(x_j , 3)*(Deltat/2 + t_k)/3 - 3*x_j*(mu + 1)/2 + (-Deltat/2 + t_k)*pow(Deltax/2 + x_j , 3)/6 + (Deltat/2 + t_k)*pow(Deltax/2 + x_j , 3)/3 + 3*(Deltax/2 + x_j)*(mu + 1)/2;
    AF_g2_val[2] = pow(x_j , 3)*(Deltat/2 + t_k)/3 + x_j*(mu + 1) - (Deltat/2 + t_k)*pow(-Deltax/2 + x_j , 3)/3 - (-Deltax/2 + x_j)*(mu + 1);
    AF_g2_val[3] = -pow(x_j , 3)*(Deltat/2 + t_k)/3 - x_j*(mu + 1) + (Deltat/2 + t_k)*pow(Deltax/2 + x_j , 3)/3 + (Deltax/2 + x_j)*(mu + 1);

    return( AF_g2_val );

}