#include <math.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include<algorithm>

using namespace std;

void mse_forward_cpu(float *truth, float *predict, float *error, int n_out) {
    for (int i = 0; i < n_out; i++) {
        *error += (truth[i] - predict[i]) * (truth[i] - predict[i]) / n_out;
    }
}

void mse_backprop_cpu(float *truth, float *predict, float *error, int n_out) {
    for (int i = 0; i < n_out; i++) {
        error[i] = 2 * (predict[i] - truth[i]) / n_out;
    }
}



void linear_forward_cpu(float *in, float *out, float *weights, float *bias, int n_in, int n_out, int batch_size) {
    int in_index, out_index, weights_index;
    // in = (n_in * batch_size), out = (n_out * batch_size), weights = (n_in * n_out)
    // generate outputs for each sample in batch
    for (int sample = 0; sample < batch_size; sample++) {
        // matrix multiply input by weights
        for (int j = 0; j < n_out; j++) {
            out_index = sample * n_out + j;
            out[out_index] = bias[j];  // add bias to output
            // dot product for input with weights for one output node
            for (int i = 0; i < n_in; i++) {
                in_index = sample * n_in + i;
                weights_index = i * n_out + j;
                out[out_index] += in[in_index] * weights[weights_index];
            }
        }
    }
}

void linear_backprop_cpu(float *errors, float *out_errors, float *weights, int n_in, int n_out, int batch_size) {
    int errors_index, out_errors_index, weights_index;

    // Error at prev layer = matmul of weights with error at curr layer
    // errors = (n_out * batch_size), out_errors = (n_in * batch_size), weights = (n_in * n_out)
    for (int sample = 0; sample < batch_size; sample++) {
        for (int i = 0; i < n_in; i++) {
            out_errors_index = sample * n_in + i;
            for (int j = 0; j < n_out; j++) {
                errors_index = sample * n_out + j;
                weights_index = i * n_out + j;
                out_errors[out_errors_index] += weights[weights_index] * errors[errors_index];
            }
        }
    }
}

void linear_update_cpu(float *in, float *errors, float *weights, float *bias, int n_in, int n_out, int batch_size, float lr) {
    int in_index, errors_index, weights_index;
    for (int sample = 0; sample < batch_size; sample++) {
        for (int j = 0; j < n_out; j++) {
            errors_index = sample * n_out + j;
            bias[j] -= lr * errors[errors_index];
            for (int i = 0; i < n_in; i++) {
                in_index = sample * n_in + i;
                weights_index = i * n_out + j;
                weights[weights_index] -= lr * errors[errors_index] * in[in_index];
            }
        }
    }
}

void relu_forward_cpu(float *in, float *out, int n_out) {
    for (int j = 0; j < n_out; j++) {
        if (in[j] > 0) {
            out[j] = in[j];
        } else {
            out[j] = 0;
        }
    }
}

void relu_backprop_cpu(float *in, float *error, float *error_out, int n_out) {
    for (int j = 0; j < n_out; j++) {
        if (in[j] > 0) {
            error_out[j] = error[j];
        } else {
            error_out[j] = 0;
        }
    }
}

void init_zero(float *a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = 0.0f;
    }
}

void set_eq(float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = b[i];
    }
}

void kaiming_init(float *w, int n_in, int n_out) {
    float std = sqrt(2 / (float)n_in);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std);

    for (int i = 0; i < n_in * n_out; i++) {
        w[i] = dist(gen);
    }
}

void fill_array(float *a, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        a[i] = dist(gen);
    }
}


vector<string> split(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

vector<float> operator/(const vector<float> &m2, const float m1) {
    /*  Returns the product of a float and a vectors (elementwise multiplication).
     Inputs:
     m1: float
     m2: vector
     Output: vector, m1 * m2, product of two vectors m1 and m2
     */

    const unsigned long VECTOR_SIZE = m2.size();
    vector<float> product(VECTOR_SIZE);

    for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
        product[i] = m2[i] / m1;
    };

    return product;
}
int main(int argc, const char *argv[]) {
    string line;
    vector<string> line_v;

    cout << "Loading data ...\n";
    vector<float> X_train;
    vector<float> y_train;

    ifstream myfile("train.txt");
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            line_v = split(line, '\t');
            int digit = strtof((line_v[0]).c_str(), 0);
            for (unsigned i = 0; i < 10; ++i) {
                if (i == digit) {
                    y_train.push_back(1.);
                } else {
                    y_train.push_back(0.);
                }
            }

            int size = static_cast<int>(line_v.size());
            for (unsigned i = 1; i < size; ++i) {
                X_train.push_back(strtof((line_v[i]).c_str(), 0));
            }
        }
        X_train = X_train / 255.0;
        myfile.close();
    }

    std::cout << y_train.size() << "\n";

    std::cout << X_train.size() << "\n";

    int batch_size = 100, n_in = 784, n_epochs = 10;

    int n_hidden = 512;
    int n_out = 10;
    // float *input = new float[batch_size * n_in], *target = new float[batch_size * n_out], *output;

    float *input = &X_train[0], *target = &y_train[0], *output;

    float *l1_weights = new float[n_in * n_hidden];
    float *l1_bias = new float[n_hidden];

    float *l2_weights = new float[n_hidden * n_out];
    float *l2_bias = new float[n_out];

    // fill_array(input, batch_size * n_in);
    // fill_array(target, batch_size * n_out);

    kaiming_init(l1_weights, n_in, n_hidden);
    init_zero(l1_bias, n_hidden);

    kaiming_init(l2_weights, n_hidden, n_out);
    init_zero(l2_bias, n_out);

    std::cout << "TRAINING"
              << "\n";

    for (int i = 0; i < n_epochs; i++) {
        std::cout << "EPOCH" << i << "\n";
        for (int batch = 0; batch < 20; batch++) {
            input = &X_train[batch * batch_size * n_in];
            target = &y_train[batch * batch_size * n_out];

            float *hidden_out = new float[n_hidden * batch_size];

            linear_forward_cpu(input, hidden_out, l1_weights, l1_bias, n_in, n_hidden, batch_size);

            float *hidden_activation = new float[n_hidden * batch_size];
            relu_forward_cpu(hidden_out, hidden_activation, n_hidden * batch_size);

            output = new float[batch_size * n_out];
            linear_forward_cpu(hidden_activation, output, l2_weights, l2_bias, n_hidden, n_out, batch_size);

            float error = 0;
            mse_forward_cpu(target, output, &error, batch_size * n_out);
            std::cout << "error: " << error << std::endl;

            float *output_error = new float[batch_size * n_out];
            mse_backprop_cpu(target, output, output_error, batch_size * n_out);

            float *activation_errors = new float[n_hidden * batch_size];
            linear_backprop_cpu(output_error, activation_errors, l2_weights, n_hidden, n_out, batch_size);

            linear_update_cpu(hidden_activation, output_error, l2_weights, l2_bias, n_hidden, n_out, batch_size, 0.01);

            float *l1_error = new float[n_hidden * batch_size];
            relu_backprop_cpu(hidden_out, activation_errors, l1_error, n_hidden);

            linear_update_cpu(input, l1_error, l1_weights, l1_bias, n_in, n_hidden, batch_size, 0.01);
        }
    }

    input = &X_train[25 * batch_size * n_in];
    target = &y_train[25 * batch_size * n_out];

    float *hidden_out = new float[n_hidden * batch_size];

    linear_forward_cpu(input, hidden_out, l1_weights, l1_bias, n_in, n_hidden, batch_size);

    float *hidden_activation = new float[n_hidden * batch_size];
    relu_forward_cpu(hidden_out, hidden_activation, n_hidden * batch_size);

    output = new float[batch_size * n_out];
    linear_forward_cpu(hidden_activation, output, l2_weights, l2_bias, n_hidden, n_out, batch_size);

    float error = 0;
    mse_forward_cpu(target, output, &error, batch_size * n_out);
    std::cout << "error: " << error << std::endl;

    for (int i = 0; i < batch_size; i++) {
        float max_1 = 0;
        int max_1_ind = 0;
        float max_2 = 0;
        int max_2_ind = 0;
        for (int j = 0; j < 10; j++) {
            std::cout << output[i * 10 + j] << " ";

            if (output[i * 10 + j] > max_1) {
                max_1 = output[i * 10 + j];
                max_1_ind = j;
            }
            if (target[i * 10 + j] > max_2) {
                max_2 = target[i * 10 + j];
                max_2_ind = j;
            }
        }
        std::cout << "\n";
        std::cout << max_1_ind << " " << max_2_ind << " "<< max_1<< std::endl;
        std::cout << "\n";

    }


    return 0;
}