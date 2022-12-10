void mse_forward_cpu(float *in, float *out, int n_out) {
    for (int i = 0; i < n_out; i++) {
        out[n_out] += (in[i] - out[i]) * (in[i] - out[i]) / n_out;
    }
}

void mse_backprop_cpu(float *in, float *out, int n_out) {
    for (int i = 0; i < n_out; i++) {
        in[i] = 2 * (in[i] - out[i]) / n_out;
    }
}
