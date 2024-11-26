#include <iostream>
#include <vector>
#include <algorithm>
#include "../common/util.h"

using namespace std;

int main(){
    vector<double> A(10);

    vector<double> a_in = {0.1, 0.2};

    //cast(A, a_in);

    //cout << a_in.begin() << endl;

    copy(a_in.begin(), a_in.end(), A.begin() + 2);

    for (int i=0; i<10; ++i){
        cout << A[i] << endl;
    }

    return(0);
};