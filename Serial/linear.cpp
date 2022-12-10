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
        for (int j = 0; j < n_in; j++) {
            out_errors_index = sample * n_in + j;
            for (int k = 0; k < n_out; k++) {
                errors_index = sample * n_out + k;
                weights_index = j * n_out + k;
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