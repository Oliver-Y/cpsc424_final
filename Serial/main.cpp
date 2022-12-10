#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

void mse_forward_cpu(float *truth, float *predict, float *error, int n_out) {
    for (int i = 0; i < n_out; i++) {
        *error += (truth[i] - predict[i]) * (truth[i] - predict[i]) / n_out;
    }
}

void mse_backprop_cpu(float *truth, float *predict, float *error, int n_out) {
    for (int i = 0; i < n_out; i++) {
        error[i] = 2 * (predict[i] -truth[i]) / n_out;
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

int main() {
    int batch_size = 3200, n_in = 100, n_epochs = 10;

    int n_hidden = 50;
    int n_out = 1;
    float *input = new float[batch_size * n_in], *target = new float[batch_size * n_out], *output;

    float *l1_weights = new float[n_in * n_hidden];
    float *l1_bias = new float[n_hidden];

    float *l2_weights = new float[n_hidden * n_out];
    float *l2_bias = new float[n_out];

    fill_array(input, batch_size * n_in);
    fill_array(target, batch_size * n_out);


    kaiming_init(l1_weights, n_in, n_hidden);
    init_zero(l1_bias, n_hidden);

    kaiming_init(l2_weights, n_hidden, n_out);
    init_zero(l2_bias, n_out);

    for (int i = 0; i < n_epochs; i++) {
        float *hidden_out = new float[n_hidden * batch_size];
        linear_forward_cpu(input, hidden_out, l1_weights, l1_bias, n_in, n_hidden, batch_size);

        float *hidden_activation = new float[n_hidden * batch_size];
        relu_forward_cpu(hidden_out, hidden_activation, n_hidden * batch_size);

        output = new float[batch_size * n_out];
        linear_forward_cpu(hidden_activation, output, l2_weights, l2_bias, n_hidden, n_out, batch_size);

        float error = 0;
        mse_forward_cpu(target, output, &error, n_out);
        std::cout << "error: " << error << std::endl;



        float *output_error = new float[batch_size * n_out];
        mse_backprop_cpu(target, output, output_error, n_out);

        float *activation_errors = new float[n_hidden * batch_size];
        linear_backprop_cpu(output_error, activation_errors, l2_weights, n_hidden, n_out, batch_size);

        linear_update_cpu(hidden_activation, output_error, l2_weights, l2_bias, n_hidden, n_out, batch_size, 0.01);

        float *l1_error = new float[n_hidden * batch_size];
        relu_backprop_cpu(hidden_out, activation_errors, l1_error, n_hidden);

        linear_update_cpu(input, l1_error, l1_weights, l1_bias, n_in, n_hidden, batch_size, 0.01);


    }

    // for (int i = 0; i < batch_size * n_out; i++) {
    //     std::cout << output[i] << ' ';
    // }

    return 0;
}