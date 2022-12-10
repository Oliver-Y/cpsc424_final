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
