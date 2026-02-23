#include<iostream>
#include<cmath>
#include <vector>
#include "gnuplot-iostream.h"
using namespace std;
/*
We have function f(x) = x^2 + 4x + 4. We want to find the minimum of this function using gradient descent.
The solution through maths is f'(x) = 2x + 4 = 0 => x = -2. We will see if gradient descent can find this minimum.
*/
double f(double X){
    return pow(X,2) + 4*X + 4;
}

/* Here we intend to use the  Central 
Difference Formula for Derivative. That will
eventually be used in the gradient descent 
algorithm. The central difference formula is given by:
f'(x) = (f(x+h) - f(x-h))/(2*h))
The formula for forward difference is:
f'(x) = (f(x+h) - f(x))/h
The formula for backward difference is:
f'(x) = (f(x) - f(x-h))/h
*/
double centralDerivative(double x, double h){
    return (f(x+h) - f(x-h))/(2*h);
}

double forwardDerivative(double x, double h){
    return (f(x+h) - f(x))/h;
}

double backwardDerivative(double x, double h){
    return (f(x) - f(x-h))/h;
}

int main(){
    Gnuplot gp;
    double x = 10.0; // Initial guess
    double lr = 0.01; // learning rate
    double h = 1e-5; // Small step for numerical derivative
    int iterations = 1000;
    vector<pair<double,double>> path;
    double grad;
    int option;
    cout << "Enter the type of numerical differentiation method to use (1 for central difference, 2 for forward difference, 3 for backward difference):   ";
    cin >> option;
    for (int i =0; i < iterations; i++){
        switch(option){
            case 1:
                grad = centralDerivative(x, h);
                break;
            case 2:
                grad = forwardDerivative(x, h);
                break;
            case 3:
                grad = backwardDerivative(x, h);
                break;
            default:
                cout << "Invalid option. Using central difference by default." << endl;
                grad = centralDerivative(x, h);
        }
        
        x = x - lr * grad; // Update x using gradient descent
        path.push_back({x, f(x)});
        if(i %100 ==0){
            cout << "Iteration " << i << ": x = " << x << ", f(x) = " << f(x) << endl;
        }
    }
    vector<pair<double, double>> curve;
    for (double x = 10; x <=10; x+=0.1){
        curve.push_back({x, f(x)});
    }
    gp << "set title 'Gradient Descent Path'\n";
    gp << "plot '-' with lines title 'f(x)', "
      "'-' with points pt 7 title 'Gradient Descent Path'\n";

    gp.send1d(curve);
    gp.send1d(path);
    return 0;
}
