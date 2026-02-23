#include<iostream>
#include<vector>
#include<cmath>
#include "gnuplot-iostream.h"

using namespace std;

/*
z = w^T x + b
p = sigmoid(z)
Loss (Binary Cross Entropy)
L = -1/N * sum[y*log(p) + (1-y) * log(1-p)]

Gradients:
dw = 1/N * X^T (p - y)
db = 1/N * sum( p - y)
*/

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

int main(){
    const int N = 200; // Number of Samples
    const int D = 2;
    const int EPOCHS = 300;
    double lr = 0.2;

    // Feature Matrix X(N x D)
    vector<vector<double>> X(N, vector<double>(D));
    // Labels vector y (N)
    vector <int> y(N);
    // Generate simple linearly separable data
    for (int i = 0; i < N/2; i++){
        X[i][0] = (double) rand()/ RAND_MAX * 2.0 - 3.0; // cluster 1
        X[i][1] = (double) rand()/ RAND_MAX * 2.0;
        y[i] = 0;
    }

    for (int i = N/2; i < N; i++){
        X[i][0] = (double) rand()/ RAND_MAX * 2.0 + 1; // cluster 2
        X[i][1] = (double) rand()/ RAND_MAX * 2.0;
        y[i] = 1;
    }

    // Initialize weights vector w(D)
    vector<double> w(D, 0.0);
    // Initialize bias b
    double b = 0.0;

    double loss = 0.0;
    
    // ====== Training Loop =====//
    for (int epoch = 0; epoch < EPOCHS; epoch++){
        vector <double> dw(D, 0.0); // gradient for weights
        double db = 0.0;    // gradient for bias
        for (int i = 0; i < N; i++){
            /*
            Forward Pass:
            z = w^T x + b
              = w1 * x1 + w2 * x2 + b

            */
            double z = 0.0;
            for (int j =0; j < D; j++){
                z += w[j] * X[i][j];
            }
            z += b;
            // Probability prediction
            double p = sigmoid(z);
            /*
               Binary Cross Entropy Loss:

               L = -[y * log(p) + (1-y) * log(1-p)]

            */
            loss += -(y[i] * log(p + 1e-12) + (1-y[i]) * log(1-p+1e-12));
            /*
            Gradient Calculation:
            dL/dw = (p-y) * x
            dL/db = (p-y)
            */
            for (int j = 0; j < D; j ++){
                dw[j] += (p - y[i]) * X[i][j];

            }
            db +=  (p - y[i]);
        }
        // Average gradients
        for (int j =0; j <D; j++){
            dw[j] /= N;
        }
        db /= N;
        loss /= N;
        /*
        Gradient Descent Update:
        w = w - lr * dw
        b = b - lr * db

        */
        for (int j =0; j < D; j++){
            w[j] -= lr * dw[j];
        }
        b -= lr * db;
        if (epoch % 50 == 0){
            cout << "Epoch :" << epoch << " loss: " << loss << endl;
        }

    }
    // ===== Evaluation ====//
        int correct = 0;
        for (int i =0; i < N; i++){
            double z = 0.0;
            for (int j =0; j < D; j++){
                z += w[j] * X[i][j];
            }
            z += b;
            double p = sigmoid(z);
            int pred = (p >= 0.5) ? 1 : 0;
            if (pred == y[i]){
                correct++;
            }
        }
        cout << "\n Final Accuracy:" << (double)correct/N * 100.0 << "%" << endl;
        // Print learned decision boundary
        cout << "\n Decision boudary equation :\n";
        cout << w[0] << "x1 + " << w[1] << "x2 + " << b << " = 0" << endl;

            // ====== Plotting Section ======
    Gnuplot gp;

    vector<pair<double,double>> class0;
    vector<pair<double,double>> class1;

    for (int i = 0; i < N; i++) {
        if (y[i] == 0)
            class0.push_back({X[i][0], X[i][1]});
        else
            class1.push_back({X[i][0], X[i][1]});
    }

    // Decision boundary line
    vector<pair<double,double>> boundary;
    for (double x = -5; x <= 5; x += 0.1) {
        double y_line = -(w[0]*x + b) / w[1];
        boundary.push_back({x, y_line});
    }

    gp << "set title 'Logistic Regression Result'\n";
    gp << "set xlabel 'x1'\n";
    gp << "set ylabel 'x2'\n";
    gp << "plot '-' with points pointtype 7 title 'Class 0', "
          "'-' with points pointtype 7 title 'Class 1', "
          "'-' with lines linewidth 2 title 'Decision Boundary'\n";

    gp.send1d(class0);
    gp.send1d(class1);
    gp.send1d(boundary);

        


    return 0;

}
