#include <iostream>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <chrono> // for std::chrono functions

/*brief: generic utility functions*/

void print_rm(std::vector<double> vec){
    using namespace std;

    // Assuming square
    int N = sqrt(vec.size());

    if (N > 100){
        cout << ">>>Warning: Matrix is over 100x100<<<" << endl;
    }

    cout << "Matrix is of size ["<<N<<","<<N<<"]"<<endl;

    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            printf("%5.5f, ", vec[i*N+j]) ;
        }
        printf("\n");
    }
}


void print_cm(std::vector<double> vec){
    using namespace std;

    // Assuming square
    int N = sqrt(vec.size());

    if (N > 100){
        cout << ">>>Warning: Matrix is over 100x100<<<" << endl;
    }

    cout << "Matrix is of size ["<<N<<","<<N<<"]"<<endl;

    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            printf("%5.2f ", vec[j*N+i]);
        }
        printf("\n");
    }
}


void print_cm_sp(std::vector<double> vec, int offset, int N){
    using namespace std;

    cout << "Matrix is of size ["<<N<<","<<N<<"]"<<endl;

    for (int i=offset; i<offset + N; i++){
        for (int j=0; j<N; j++){
            printf("%5.2f ", vec[j*N+i]);
        }
        printf("\n");
    }
}


void print_cm_point_sp(double *vec, int offset, int N){
    using namespace std;

    cout << "Matrix is of size ["<<N<<","<<N<<"]"<<endl;

    for (int i=offset; i<offset + N; i++){
        for (int j=0; j<N; j++){
            printf("%5.2f ", vec[j*N+i]);
        }
        printf("\n");
    }
}


void print_vec(int N, double *vec){
    using namespace std;

    cout << "" << endl;

    cout << "Size of Vector: "<< N << endl;

    for (int i=0; i<N; i++){
        cout << vec[i] << endl;
    }

    cout << "" << endl;
}



void print_vec_sd(std::vector<double> vec){
    using namespace std;

    int N = vec.size();

    cout << "" << endl;

    cout << "Size of Vector: "<< N << endl;

    for (int i=0; i<N; i++){
        cout << vec[i] << endl;
    }

    cout << "" << endl;
}



void print_vec_sd_int(std::vector<int> vec){
    using namespace std;

    int N = vec.size();

    cout << "" << endl;

    cout << "Size of Vector: "<< N << endl;

    for (int i=0; i<N; i++){
        cout << vec[i] << endl;
    }

    cout << "" << endl;
}



void print_title(){
    std::ifstream f("title.txt");

    if (f.is_open())
        std::cout << f.rdbuf();
}



double relativeError(double a, double b, double max){
    return ( abs(a - b) / max);
}



std::vector<double> relativeError_ebe ( std::vector<double> v1, std::vector<double> v2){
    using namespace std;
    /*breif: computing the relative error between two vectors element by element
            using inf norm https://www.netlib.org/lapack/lug/node75.html
            */

    if (v1.size() != v2.size()){
        std::cout << "WARNING: vectors are not of same size --relativeError_ebe" << std::endl;
    }

    // find the maximum element between all vectors
    vector<double> max_vec = {*max_element(v2.begin(), v2.end()), *max_element(v1.begin(), v1.end())};
    double max = *max_element( max_vec.begin(), max_vec.end() );

    // allocate an error vector
    vector<double> error(v1.size());

    // compute the relative error
    for (int i=0; i<v1.size(); i++){
        error[i] = relativeError(v1[i], v2[i], max);
    }

    return(error);
}





double infNorm_error ( std::vector<double> v1, std::vector<double> v2 ){
    /*brief: computing the max relative error between two vectors*/

    using namespace std;

    std::vector<double> error = relativeError_ebe(v1, v2);

    double relError = *max_element(error.begin(), error.end());

    return(relError);

}


double L2Norm ( std::vector<double> v1, std::vector<double> v2 ){
    
    using namespace std;

    int N = v1.size();

    double error1;
    double error2;

    for (int i=0; i<N; ++i){
        error1 += pow( v1[i]-v2[i], 2 );
        error2 += pow( v1[i], 2 );
    }

    error1 = pow(error1, 0.5);
    error2 = pow(error2, 0.5);

    return( error1/error2 ); 

}




inline void outofbounds_check(int index, std::vector<double> &vec){
    /*brief: debug checking to make sure things are not being indexed over or under a vectors size*/
    using namespace std;

    if ( index < 0 ) {
        cout<<">>>>>>>>>>>>ERROR<<<<<<<<<<<<"<<endl;
        cout<<"something was indexed under 0"<<endl;
        cout<<"index: " << index <<" size of vec: "<<vec.size()<<endl;
    } else if ( index >= vec.size() ) {
        cout<<">>>>>>>>>>>>>>>>>>>>ERROR<<<<<<<<<<<<<<<<<<<<"<<endl;
        cout<<"something was indexed over a vectors max size"<<endl;
        cout<<"index: " << index <<" size of vec: "<<vec.size()<<endl;
        exit(1);
    }
}

std::vector<double> row2colSq(std::vector<double> row){
    /*brief */
    
    int SIZE = sqrt(row.size());

    std::vector<double> col(SIZE*SIZE);

    for (int i = 0; i < SIZE; ++i){
        for (int j = 0; j < SIZE; ++j){
            outofbounds_check(i * SIZE + j, col);
            outofbounds_check(j * SIZE + i, row);


            col[ i * SIZE + j ] = row[ j * SIZE + i ];
        }
    }

    return(col);
}


void check_close(std::vector<double> v1, std::vector<double> v2){
    using namespace std;

    int n1 = v1.size();
    int n2 = v2.size();

    double rtol=1e-05;
    double atol=1e-08;

    if (n1 != n2){
        cout<< "check_close: vectors are not of same size" <<endl;
        cout << "n1 " << n1 << "    n2 " << n2 << endl;
    }

    double r1;
    double r2;

    bool checker = true;
    for (int i=0; i<n1; ++i){

        r1 = abs(v1[i] - v2[i]);
        r2 = atol + rtol * abs(v2[i]);

        if ( r1 >= r2 ){
            checker = false;
        }
    }

    if ( checker == false ){
        cout << "ERROR: Two vectors that should be equal arent" << endl;
    } else {
        cout << "Vectors checked and where the same" <<endl;
    }



}


class Timer{
    private:
        // Type aliases to make accessing nested type easier
        using Clock = std::chrono::steady_clock;
        using Second = std::chrono::duration<double, std::ratio<1> >;

        std::chrono::time_point<Clock> m_beg { Clock::now() };

    public:
        void reset()
        {
            m_beg = Clock::now();
        }

        double elapsed() const
        {
            return std::chrono::duration_cast<Second>(Clock::now() - m_beg).count();
        }
};