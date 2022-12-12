#include <chrono>
#include <cmath>
#include <random>

#include "load_mnist.h"

using namespace std;

#define BATCH_SIZE 32

float max(float t1, float t2) {
    return t1 < t2 ? t2 : t1;
}

void CE_forward_cpu(float *truth, float *predict, float *error, int n_out) {
    for (int i = 0; i < n_out * BATCH_SIZE; i++) {
        *error += -truth[i] * log(max(predict[i], 0.0001)) / BATCH_SIZE;
    }
}

void softmax_CE_backprop_cpu(float *truth, float *predict, float *error, int n_out) {
    for (int i = 0; i < n_out * BATCH_SIZE; i++) {
        error[i] = predict[i] - truth[i];
    }
}

void softmax_forward_cpu(float *in, float *out, int n_out) {
    float sum_exp;
    for (int sample = 0; sample < BATCH_SIZE; sample++) {
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
        }
    }
}

void linear_forward_cpu(float *in, float *out, float *weights, float *bias, int n_in, int n_out) {
    int in_index, out_index, weights_index;
    // in = (n_in * BATCH_SIZE), out = (n_out * BATCH_SIZE), weights = (n_in * n_out)
    // generate outputs for each sample in batch
    for (int sample = 0; sample < BATCH_SIZE; sample++) {
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

void linear_backprop_cpu(float *errors, float *out_errors, float *weights, int n_in, int n_out) {
    int errors_index, out_errors_index, weights_index;

    // Error at prev layer = matmul of weights with error at curr layer
    // errors = (n_out * BATCH_SIZE), out_errors = (n_in * BATCH_SIZE), weights = (n_in * n_out)
    for (int sample = 0; sample < BATCH_SIZE; sample++) {
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

void linear_update_cpu(float *in, float *errors, float *weights, float *bias, int n_in, int n_out, float lr) {
    int in_index, errors_index, weights_index;
    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        for (int j = 0; j < n_out; j++) {
            errors_index = sample * n_out + j;
            bias[j] -= lr / BATCH_SIZE * errors[errors_index];
            for (int i = 0; i < n_in; i++) {
                in_index = sample * n_in + i;
                weights_index = i * n_out + j;
                weights[weights_index] -= lr / BATCH_SIZE * errors[errors_index] * in[in_index];
            }
        }
    }
}

void relu_forward_cpu(float *in, float *out, int n_out) {
    for (int j = 0; j < n_out * BATCH_SIZE; j++) {
        if (in[j] > 0) {
            out[j] = in[j];
        } else {
            out[j] = 0.2 * in[j];
        }
    }
}

void relu_backprop_cpu(float *in, float *error, float *error_out, int n_out) {
    for (int j = 0; j < n_out * BATCH_SIZE; j++) {
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

void kaiming_init(float *w, int n_in, int n_out) {
    float std = sqrt(2 / (float)n_in);

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0f, std);

    for (int i = 0; i < n_in * n_out; i++) {
        w[i] = dist(gen);
    }
}

float accuracy(float *output, float *target, int n_out) {
    float acc = 0;
    for (int i = 0; i < BATCH_SIZE; i++) {
        float max_prob = 0;
        int pred_digit = 0;
        int target_digit = 0;
        for (int j = 0; j < n_out; j++) {
            if (output[i * n_out + j] > max_prob) {
                max_prob = output[i * n_out + j];
                pred_digit = j;
            }
            if (target[i * n_out + j] > 0.5) {
                target_digit = j;
            }
        }
        if (pred_digit == target_digit) {
            acc += 1;
        }
    }
    return acc / BATCH_SIZE;
}

int main(int argc, char *argv[]) {

    if (argc != 2)
    {
        printf("Usage: serial <n_hidden_layers> \n");
        exit(-1);
    }

    int n_hidden = atoi(argv[1]);


    int n_in = 784, n_out = 10, n_epochs = 5;
    float lr = (128.0 / n_hidden) * 0.001;
    int data_size;

    vector<float> x_train;
    vector<float> y_train;
    load_mnist(x_train, y_train, &data_size);

    cout << "Data size: " << data_size << endl;
    cout << "Hidden layer size: "<< n_hidden << endl;
    cout << "Batch size: "<< BATCH_SIZE << endl;

    int train_test_split = (int)(0.9 * data_size);

    float *l1_weights = new float[n_in * n_hidden];
    float *l1_bias = new float[n_hidden]();

    float *l2_weights = new float[n_hidden * n_out];
    float *l2_bias = new float[n_out]();

    kaiming_init(l1_weights, n_in, n_hidden);
    kaiming_init(l2_weights, n_hidden, n_out);

    float error;
    float *input, *target, *output;
    float *l1_out, *relu_out, *l2_out;
    float *l2_error, *l1_error, *relu_error;

    float forward_time = 0, backprop_time = 0;
    chrono::steady_clock::time_point begin, end;
    chrono::steady_clock::time_point b, e;

    begin = chrono::steady_clock::now();
    cout << "===TRAINING===" << endl;

    for (int i = 0; i < n_epochs; i++) {
        cout << "Epoch " << i << "\n";
        for (int batch = 0; batch < train_test_split / BATCH_SIZE; batch++) {
            input = &x_train[batch * BATCH_SIZE * n_in];
            target = &y_train[batch * BATCH_SIZE * n_out];

            // FORWARD PROPAGATION STEP

            b = chrono::steady_clock::now();

            l1_out = new float[n_hidden * BATCH_SIZE]();
            linear_forward_cpu(input, l1_out, l1_weights, l1_bias, n_in, n_hidden);

            relu_out = new float[n_hidden * BATCH_SIZE]();
            relu_forward_cpu(l1_out, relu_out, n_hidden);

            l2_out = new float[BATCH_SIZE * n_out]();
            linear_forward_cpu(relu_out, l2_out, l2_weights, l2_bias, n_hidden, n_out);

            output = new float[BATCH_SIZE * n_out]();
            softmax_forward_cpu(l2_out, output, n_out);

            error = 0;
            CE_forward_cpu(target, output, &error, n_out);
            
            e = chrono::steady_clock::now();
            forward_time += (chrono::duration_cast<chrono::microseconds>(e - b).count());


            // BACK PROPAGATION STEP

            b = chrono::steady_clock::now();

            l2_error = new float[BATCH_SIZE * n_out]();
            softmax_CE_backprop_cpu(target, output, l2_error, n_out);

            relu_error = new float[n_hidden * BATCH_SIZE]();
            linear_backprop_cpu(l2_error, relu_error, l2_weights, n_hidden, n_out);

            l1_error = new float[n_hidden * BATCH_SIZE]();
            relu_backprop_cpu(l1_out, relu_error, l1_error, n_hidden);

            linear_update_cpu(relu_out, l2_error, l2_weights, l2_bias, n_hidden, n_out, lr);
            linear_update_cpu(input, l1_error, l1_weights, l1_bias, n_in, n_hidden, lr);

            e = chrono::steady_clock::now();
            backprop_time += (chrono::duration_cast<chrono::microseconds>(e - b).count());

        }
        // cout << "error: " << error << endl;
    }

    cout << "===TRAINING COMPLETE===" << endl;
    end = chrono::steady_clock::now();
    cout << "Training time: " << (chrono::duration_cast<chrono::microseconds>(end - begin).count()) / 1000000.0f << "s" << endl;
    cout << "Forward propagation time: " << forward_time / 1000000.0f << "s" << endl;
    cout << "Backpropagation time: " << backprop_time / 1000000.0f << "s" << endl;

    cout << "===TESTING===" << endl;

    int last_test_batch = data_size / BATCH_SIZE;
    int first_test_batch = train_test_split / BATCH_SIZE;
    float avg_acc = 0.;
    float avg_err = 0.;

    for (int batch = first_test_batch; batch < last_test_batch; batch++) {
        input = &x_train[batch * BATCH_SIZE * n_in];
        target = &y_train[batch * BATCH_SIZE * n_out];

        l1_out = new float[n_hidden * BATCH_SIZE]();
        linear_forward_cpu(input, l1_out, l1_weights, l1_bias, n_in, n_hidden);

        relu_out = new float[n_hidden * BATCH_SIZE]();
        relu_forward_cpu(l1_out, relu_out, n_hidden);

        l2_out = new float[BATCH_SIZE * n_out]();
        linear_forward_cpu(relu_out, l2_out, l2_weights, l2_bias, n_hidden, n_out);

        output = new float[BATCH_SIZE * n_out]();
        softmax_forward_cpu(l2_out, output, n_out);

        error = 0;
        CE_forward_cpu(target, output, &error, n_out);
        avg_err += error;
        avg_acc += accuracy(output, target, n_out);
    }
    avg_err /= (last_test_batch - first_test_batch);
    avg_acc /= (last_test_batch - first_test_batch);

    cout << "TEST avg error: " << avg_err << endl;
    cout << "TEST avg accuracy: " << avg_acc << endl;

    return 0;
}
