#include <cmath>
#include <vector>

# define e 2.7182818

class mms{
    public:
        double A;
        double B;
        double C;
        double D;
        double F;

        double v1;
        double v2;

        double sigma1;
        double sigma2;
        double sigmaS1;
        double sigmaS2;
        double sigmaS1_2;

        double group1afCONT(double mu, double x, double t){
            //the continuous angular flux function for group 1 //
            return ( x + t + mu ) ;
        }

        double group1afUNINT(double mu, double x, double t){
            return ( pow(x,2)/2+t*x+mu*x );
        }

        std::vector<double> group1af(double x, double dx, double t, double dt, double mu){
            //actually evaluating the integrals using the un-evaluated term and returning them as values//
            double x_m = x-dx/2;
            double x_i = x;
            double x_p = x+dx/2;

            double t_p = t+dt/2;
            double t_m = t-dt/2;

            double afa = group1afUNINT(mu,x_i,t_p) - group1afUNINT(mu,x_m,t_p);
            double afb = group1afUNINT(mu,x_p,t_p) - group1afUNINT(mu,x_i,t_p);
            double afc = ( afa + (group1afUNINT(mu,x_i,t_m) - group1afUNINT(mu,x_m,t_m)) ) / 2;
            double afd = ( afb +  (group1afUNINT(mu,x_p,t_m) - group1afUNINT(mu,x_i,t_m)) ) / 2;

            //afa = abs(afa);
            //afb = abs(afb);
            //afc = abs(afc);
            //afd = abs(afd);

            return( std::vector<double> {afa, afb, afc, afd} );
        }

        double group1sourceCONT(double mu, double x, double t){
            //the un-evaluated integral of the source term used to take our continuous source function into the proper democratization for SCB-TDMB//
            return ( 2*mu + 2*sigma1*(mu + t + x) - sigmaS1*(2*t + 2*x) - sigmaS1_2*(2*t + 2*x) + 2/v1 );
        }

        double group1sourceUNINT(double mu, double x, double t){
            //the un-evaluated integral of the source term used to take our continuous source function into the proper democratization for SCB-TDMB//
            return ( (x*(v1*((sigma1-sigmaS1_2-sigmaS1)*(x+2*t)+2*mu*(sigma1+1))+2))/v1 );
        }

        std::vector<double> group1source(double x, double dx, double t, double dt, double mu){
            //actually evaluating the integrals using the un-evaluated term and returning them as values//
            double x_m = x-dx/2;
            double x_i = x;
            double x_p = x+dx/2;

            double t_p = t+dt/2;
            double t_m = t-dt/2;

            // evaluating the integral over dx_i-1/2
            double Qa = group1sourceUNINT(mu,x_i,t_p) - group1sourceUNINT(mu,x_m,t_p);
            // evaluating the integral over dx_i+1/2
            double Qb = group1sourceUNINT(mu,x_p,t_p) - group1sourceUNINT(mu,x_i,t_p);
            // finding the time averaged integral dx_i-1/2 
            double Qc = ( Qa + (group1sourceUNINT(mu,x_i,t_m) - group1sourceUNINT(mu,x_m,t_m)) ) / 2;
            // finding the time averaged integral dx_i+1/2 
            double Qd = ( Qb +  (group1sourceUNINT(mu,x_p,t_m) - group1sourceUNINT(mu,x_i,t_m)) ) / 2;

            return( std::vector<double> {Qa, Qb, Qc, Qd} );
        }

        //
        // GROUP 2
        //
        
        double group2afCONT(double mu, double x, double t){
            //the continuous angular flux function for group 2 //
            return ( pow(x,2)*t + mu ) ;
        }

        double group2afUNINT(double mu, double x, double t){
            return ( (t*pow(x,3))/3+mu*x );
        }

        std::vector<double> group2af(double x, double dx, double t, double dt, double mu){
            //actually evaluating the integrals using the un-evaluated term and returning them as values//
            double x_m = x-dx/2;
            double x_i = x;
            double x_p = x+dx/2;

            double t_p = t+dt/2;
            double t_m = t-dt/2;

            double afa = group2afUNINT(mu,x_i,t_p) - group2afUNINT(mu,x_m,t_p);
            double afb = group2afUNINT(mu,x_p,t_p) - group2afUNINT(mu,x_i,t_p);
            double afc = ( afa + (group2afUNINT(mu,x_i,t_m) - group2afUNINT(mu,x_m,t_m)) ) / 2;
            double afd = ( afb +  (group2afUNINT(mu,x_p,t_m) - group2afUNINT(mu,x_i,t_m)) ) / 2;

            //afa = abs(afa);
            //afb = abs(afb);
            //afc = abs(afc);
            //afd = abs(afd);

            return( std::vector<double> {afa, afb, afc, afd} );
        }


        double group2sourceCONT(double mu, double x, double t){
            //continuous source from MMS//
            return( 4*mu*t*x + 2*sigma2*(mu + t*pow(x,2)) + 2*sigmaS1_2*t*pow(x,2) - 2*sigmaS2*t*pow(x,2) + 2*pow(x,2)/v2 ) ;
        }

        double group2sourceUNINT(double mu, double x, double t){
            //the un-evaluated integral (dx) of the source term used to take our continuous source function into the proper discretization for SCB-TDMB//
            return ( (2*((sigma2-sigmaS2+sigmaS1_2)*t*v2+1)*pow(x,3))/(3*v2)+2*mu*t*pow(x,2)+2*mu*sigma2*x );
        }

        std::vector<double> group2source(double x, double dx, double t, double dt, double mu){
            //actually evaluating the integrals using the un-evaluated term and returning them as values//
            double x_m = x-dx/2;
            double x_i = x;
            double x_p = x+dx/2;

            double t_p = t+dt/2;
            double t_m = t-dt/2;

            double Qa = group2sourceUNINT(mu,x_i,t_p) - group2sourceUNINT(mu,x_m,t_p);
            double Qb = group2sourceUNINT(mu,x_p,t_p) - group2sourceUNINT(mu,x_i,t_p);
            double Qc = ( Qa + (group2sourceUNINT(mu,x_i,t_m) - group2sourceUNINT(mu,x_m,t_m)) ) / 2;
            double Qd = ( Qb +  (group2sourceUNINT(mu,x_p,t_m) - group2sourceUNINT(mu,x_i,t_m)) ) / 2;

            return( std::vector<double> {Qa, Qb, Qc, Qd} );
        }
        
        double group2af(double x, double dx, double t, double dt, double mu, double sigma){
            //The proper spatial integral and time edge values of the chosen continuous angular flux//
            return( 1.0 );
        }

};