#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

float max(float t1, float t2) {
    return t1 < t2 ? t2 : t1;
}

void CE_forward_cpu(float *truth, float *predict, float *error, int n_out, int batch_size) {
    for (int i = 0; i < n_out * batch_size; i++) {
        *error += -truth[i] * log(max(predict[i], 0.0001)) / batch_size;
    }
}

void softmax_CE_backprop_cpu(float *truth, float *predict, float *error, int n_out, int batch_size) {
    for (int i = 0; i < n_out * batch_size; i++) {
        error[i] = predict[i] - truth[i];
    }
}

void softmax_forward_cpu(float *in, float *out, int n_out, int batch_size) {
    float sum_exp;
    for (int sample = 0; sample < batch_size; sample++) {
        sum_exp = 0.0;
        float max_ = -10000;
        for (int j = 0; j < n_out; j++) {
            max_ = max(max_, in[sample * n_out + j]);
        }
        for (int j = 0; j < n_out; j++) {
            float e = exp(in[sample * n_out + j] - max_);
            sum_exp += e;
            out[sample * n_out + j] = e;
        }
        for (int j = 0; j < n_out; j++) {
            out[sample * n_out + j] /= sum_exp;
            if (isnan(out[sample * n_out + j])) {
                std::cout << sum_exp << "," << max_ << "\n";
            }
        }
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
            bias[j] -= lr / batch_size * errors[errors_index];
            for (int i = 0; i < n_in; i++) {
                in_index = sample * n_in + i;
                weights_index = i * n_out + j;
                weights[weights_index] -= lr / batch_size * errors[errors_index] * in[in_index];
            }
        }
    }
}

void relu_forward_cpu(float *in, float *out, int n_out) {
    for (int j = 0; j < n_out; j++) {
        if (in[j] > 0) {
            out[j] = in[j];
        } else {
            out[j] = 0.2 * in[j];
        }
    }
}

void relu_backprop_cpu(float *in, float *error, float *error_out, int n_out) {
    for (int j = 0; j < n_out; j++) {
        if (in[j] > 0) {
            error_out[j] = error[j];
        } else {
            error_out[j] = 0.2 * error[j];
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

void fill_array_category(float *a, int n, int n_out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, n_out - 1);

    for (int i = 0; i < n; i++) {
        int ind = dist(gen);
        for (int j = 0; j < n_out; j++) {
            a[i * n_out + j] = (j == ind) ? 1 : 0;
        }
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

vector<float> operator-(const vector<float> &m1, const float m2) {
    /*  Returns the difference between two vectors.
     Inputs:
     m1: vector
     m2: vector
     Output: vector, m1 - m2, difference between two vectors m1 and m2.
     */

    const unsigned long VECTOR_SIZE = m1.size();
    vector<float> difference(VECTOR_SIZE);

    for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
        difference[i] = m1[i] - m2;
    };

    return difference;
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

int main() {
    int batch_size = 64, n_in = 784, n_epochs = 10;
    float lr = 0.005;
    int n_hidden = 128;
    int n_out = 10;

    string line;
    vector<string> line_v;

    cout << "Loading data ...\n";
    vector<float> x_train;
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
                x_train.push_back(strtof((line_v[i]).c_str(), 0));
            }
        }
        x_train = x_train / 255.0;
        x_train = x_train - 0.1307;
        x_train = x_train / 0.3081;
        myfile.close();
    }

    std::cout << y_train.size() << "\n";

    std::cout << x_train.size() << "\n";

    int data_size = y_train.size() / n_out;

    // int data_size = 10000;
    // float *x_train = new float[data_size * n_in], *y_train = new float[data_size * n_out];
    // fill_array(x_train, data_size * n_in);
    // fill_array_category(y_train, data_size, n_out);

    float *input, *target, *output;


    float *l1_weights = new float[n_in * n_hidden];
    float *l1_bias = new float[n_hidden];

    float *l2_weights = new float[n_hidden * n_out];
    float *l2_bias = new float[n_out];

    //    for (int j = 0; j < batch_size; j++) {
    //        for (int k = 0; k < n_in; k++) {
    //            std::cout << x_train[j * n_in + k] << " ";
    //        }
    //        std::cout << "\n";
    //    }

    kaiming_init(l1_weights, n_in, n_hidden);
    init_zero(l1_bias, n_hidden);

    kaiming_init(l2_weights, n_hidden, n_out);
    init_zero(l2_bias, n_out);

    for (int i = 0; i < n_epochs; i++) {
        std::cout << "EPOCH" << i << "\n";
        float error;
        float *output_error, *l1_error;
        for (int batch = 0; batch < data_size / batch_size -2 ; batch++) {
            input = &x_train[batch * batch_size * n_in];
            target = &y_train[batch * batch_size * n_out];

            // float *hidden_out = new float[n_hidden * batch_size];
            // linear_forward_cpu(input, hidden_out, l1_weights, l1_bias, n_in, n_hidden, batch_size);

            // float *hidden_activation = new float[n_hidden * batch_size];
            // relu_forward_cpu(hidden_out, hidden_activation, n_hidden * batch_size);

            // output = new float[batch_size * n_out];
            // linear_forward_cpu(hidden_activation, output, l2_weights, l2_bias, n_hidden, n_out, batch_size);

            // error=0;
            // mse_forward_cpu(target, output, &error, batch_size * n_out);

            // float *output_error = new float[batch_size * n_out];
            // mse_backprop_cpu(target, output, output_error, batch_size * n_out);

            // float *activation_errors = new float[n_hidden * batch_size];
            // linear_backprop_cpu(output_error, activation_errors, l2_weights, n_hidden, n_out, batch_size);

            // linear_update_cpu(hidden_activation, output_error, l2_weights, l2_bias, n_hidden, n_out, batch_size, 0.1);

            // float *l1_error = new float[n_hidden * batch_size];
            // relu_backprop_cpu(hidden_out, activation_errors, l1_error, n_hidden);

            // linear_update_cpu(input, l1_error, l1_weights, l1_bias, n_in, n_hidden, batch_size, 0.1);

            float *hidden_out = new float[n_hidden * batch_size];
            linear_forward_cpu(input, hidden_out, l1_weights, l1_bias, n_in, n_hidden, batch_size);

            float *hidden_activation = new float[n_hidden * batch_size];
            relu_forward_cpu(hidden_out, hidden_activation, n_hidden * batch_size);

            float *softmax_input = new float[batch_size * n_out];
            linear_forward_cpu(hidden_activation, softmax_input, l2_weights, l2_bias, n_hidden, n_out, batch_size);

            output = new float[batch_size * n_out];
            softmax_forward_cpu(softmax_input, output, n_out, batch_size);

            error = 0;
            CE_forward_cpu(target, output, &error, n_out, batch_size);

            if (isnan(error)) {
                for (int j = 0; j < n_hidden * n_out; j++) {
                    std::cout << l2_weights[j] << " ";
                }
                std::cout << "===========\n\n\n"
                          << std::endl;

                for (int j = 0; j < batch_size * n_out; j++) {
                    std::cout << output_error[j] << " ";
                }
                return 0;
            }

            std::cout << "error: " << error << std::endl;

            output_error = new float[batch_size * n_out];
            softmax_CE_backprop_cpu(target, output, output_error, n_out, batch_size);

            //    for (int j = 0; j < 10; j++) {
            //        std::cout << l2_weights[j] << " ";
            //    }
            float *activation_errors = new float[n_hidden * batch_size];
            linear_backprop_cpu(output_error, activation_errors, l2_weights, n_hidden, n_out, batch_size);

            linear_update_cpu(hidden_activation, output_error, l2_weights, l2_bias, n_hidden, n_out, batch_size, lr);

            //    for(int j =0; j < 20; j++){
            //        std::cout << activation_errors[j] << " ";
            //    }
            l1_error = new float[n_hidden * batch_size];
            relu_backprop_cpu(hidden_out, activation_errors, l1_error, n_hidden * batch_size);

            //    for(int j =0; j < 20; j++){
            //        std::cout << l1_weights[j] << " ";
            //    }

            linear_update_cpu(input, l1_error, l1_weights, l1_bias, n_in, n_hidden, batch_size, lr);
            // for(int j =0; j < batch_size * n_hidden; j++){
            //     std::cout << l1_error[j] << " ";
            // }
        }
        // std::cout << "error: " << error << std::endl;
    }

    input = &x_train[(data_size / batch_size -1) * batch_size * n_in];
    target = &y_train[(data_size / batch_size -1) * batch_size * n_out];

    float *hidden_out = new float[n_hidden * batch_size];

    linear_forward_cpu(input, hidden_out, l1_weights, l1_bias, n_in, n_hidden, batch_size);

    float *hidden_activation = new float[n_hidden * batch_size];
    relu_forward_cpu(hidden_out, hidden_activation, n_hidden * batch_size);

    float *softmax_input = new float[batch_size * n_out];
    linear_forward_cpu(hidden_activation, softmax_input, l2_weights, l2_bias, n_hidden, n_out, batch_size);

    output = new float[batch_size * n_out];
    softmax_forward_cpu(softmax_input, output, n_out, batch_size);

    float error = 0;
    CE_forward_cpu(target, output, &error, n_out, batch_size);

    float accuracy = 0;
    for (int i = 0; i < batch_size; i++) {
        float max_1 = 0;
        int max_1_ind = 0;
        float max_2 = 0;
        int max_2_ind = 0;
        for (int j = 0; j < 10; j++) {
            if (output[i * 10 + j] > max_1) {
                max_1 = output[i * 10 + j];
                max_1_ind = j;
            }
            if (target[i * 10 + j] > max_2) {
                max_2 = target[i * 10 + j];
                max_2_ind = j;
            }
        }
        // std::cout << "\n"
        //           << std::endl;
        if (max_1_ind == max_2_ind) {
            accuracy += 1;
        }
        // else{
        //     std::cout << max_1 << std::endl;
        // }
        // std::cout << "\n";
        // std::cout << max_1_ind << " " << max_2_ind << " " << max_1 << std::endl;
        // std::cout << "\n";
    }
    std::cout << "TEST error: " << error << std::endl;
    std::cout << "TEST accuracy: " << accuracy / batch_size << std::endl;
    return 0;
}
