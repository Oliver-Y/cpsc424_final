void mse_forward_cpu(float *truth, float *predict, float *error, int n_out) {
    for (int i = 0; i < n_out; i++) {
        *error += (truth[i] - predict[i]) * (truth[i] - predict[i]) / n_out;
    }
}

void mse_backprop_cpu(float *truth, float *predict, float *error, int n_out) {
    for (int i = 0; i < n_out; i++) {
        error[i] = 2 * (truth[i] - predict[i]) / n_out;
    }
}
