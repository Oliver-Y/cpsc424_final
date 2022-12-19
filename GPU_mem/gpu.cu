#include <chrono>
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cuda.h>
// #include "load_mnist.h"

using namespace std;

#define BATCH_SIZE 32
#define BLOCK_SIZE 32

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
    const unsigned long VECTOR_SIZE = m1.size();
    vector<float> difference(VECTOR_SIZE);

    for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
        difference[i] = m1[i] - m2;
    };

    return difference;
}

vector<float> operator/(const vector<float> &m2, const float m1) {
    const unsigned long VECTOR_SIZE = m2.size();
    vector<float> product(VECTOR_SIZE);

    for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
        product[i] = m2[i] / m1;
    };
    return product;
}

void load_mnist(vector<float> &x_train, vector<float> &y_train, int *data_size) {
    string line;
    vector<string> line_v;

    cout << "Loading data ...\n";

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
    *data_size = y_train.size() / 10;
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

__global__ void softmax_CE_backprop_gpu(float *truth, float *predict, float *error, int n_out){
    //int row = blockDim.x * blockIdx.x + threadIdx.x, col = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x, row = blockDim.y * blockIdx.y + threadIdx.y;
    if((row < BATCH_SIZE) && (col < n_out)){
        error[row * n_out + col] = predict[row * n_out + col] - truth[row * n_out + col]; 
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
            //Want this to be shared
            sum_exp += e;
            out[sample * n_out + j] = e;
        }
        for (int j = 0; j < n_out; j++) {
            out[sample * n_out + j] /= sum_exp;
        }
    }
}

__global__ void softmax_forward_gpu(float *in, float* out, int n_out){
    //int row = blockDim.x * blockIdx.x + threadIdx.x, col = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x, row = blockDim.y * blockIdx.y + threadIdx.y;
    float sum_exp = 0.0; 
    float max_ = -10000; 
    if((row < BATCH_SIZE) && (col < n_out)){
        int out_index = row * n_out + col; 
        int in_index = row * n_out + col; 
        //Everyone loads it into shared memory 
        for(int i = 0; i < n_out; ++i){
            max_ = max(max_, in[row * n_out + i]); 
        }
        for(int i = 0; i < n_out; ++i){
            float e = exp(in[row * n_out + i] - max_);
            sum_exp += e; 
        }
        out[out_index] = exp(in[in_index] - max_);
        out[out_index] /= sum_exp;
    }
}


__global__ void linear_forward_gpu(float *in, float *out, float *weights, float *bias, int n_in, int n_out) {
    //int row = blockDim.x * blockIdx.x + threadIdx.x, col = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x, row = blockDim.y * blockIdx.y + threadIdx.y;
    int in_index, weights_index, out_index;

    if ((row < BATCH_SIZE) && (col < n_out)) {
        out_index = row * n_out + col;
        out[out_index] = bias[col];

        for (int i = 0; i < n_in; i++) {
            in_index = row * n_in + i;
            weights_index = i * n_out + col;
            out[out_index] += in[in_index] * weights[weights_index];
        }
    }
}

__global__ void linear_backprop_gpu(float *errors, float *out_errors, float *weights, int n_in, int n_out) {
    //int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x, row = blockDim.y*blockIdx.y + threadIdx.y;
    int errors_index, out_errors_index, weights_index;

    if ((row < BATCH_SIZE) && (col < n_out)){
        errors_index = row*n_out + col;
        for (int i=0; i<n_in; i++){
            out_errors_index = row*n_in + i;
            weights_index = i*n_out + col;
            atomicAdd(&out_errors[out_errors_index], weights[weights_index]*errors[errors_index]);
        }
    }
}


__global__ void linear_update_gpu(float *in, float *errors, float *weights, float *bias, int n_in, int n_out, float lr) {
    //int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x, row = blockDim.y*blockIdx.y + threadIdx.y;
    int in_index, errors_index, weights_index;

    if ((row < BATCH_SIZE) && (col < n_out)){
        errors_index = row*n_out + col;
        atomicAdd(&bias[col], -lr / BATCH_SIZE *errors[errors_index]);
        for (int i=0; i<n_in; i++){
            in_index = row*n_in + i;
            weights_index = i*n_out + col;
            atomicAdd(&weights[weights_index], -lr / BATCH_SIZE *in[in_index]*errors[errors_index]);
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

__global__ void relu_forward_gpu(float *in, float *out, int n_out){
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < n_out * BATCH_SIZE){
        out[index] = fmaxf(0.2*in[index], in[index]);
    }
}

__global__ void relu_backward_gpu(float *in, float *error, float *error_out, int n_out){
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < n_out * BATCH_SIZE){
        if (in[index] > 0) {
            error_out[index] = error[index];
        } else {
            error_out[index] = 0.2 * error[index];
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

void set_eq(float *a, float *b, int n){
    for (int i=0; i<n; i++){
        a[i] = b[i];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2)
    {
        printf("Usage: serial <n_hidden_layers> \n");
        exit(-1);
    }

    int n_hidden = atoi(argv[1]);
    cudaEvent_t start, stop; 
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

    float *l1_weights, *l1_bias, *l2_weights , *l2_bias;
    float *dev_l1_weights, *dev_l1_bias, *dev_l2_weights, *dev_l2_bias;
    int size_hidden = n_hidden * BATCH_SIZE * sizeof(float); 
    int size_input = n_in * BATCH_SIZE * sizeof(float); 
    int size_output = n_out * BATCH_SIZE * sizeof(float); 
    l1_weights = (float*) malloc(n_in * n_hidden * sizeof(float)); 
    l1_bias = (float*) malloc(n_hidden * sizeof(float)); 
    l2_weights = (float*) malloc(n_out * n_hidden * sizeof(float)); 
    l2_bias = (float*) malloc(n_out * sizeof(float)); 
  //  l1_weights = (float*) calloc(n_in * n_hidden,sizeof(float)); 
  //  l1_bias = (float*) calloc(n_hidden, sizeof(float)); 
  //  l2_weights = (float*) calloc(n_out * n_hidden,sizeof(float)); 
  //  l2_bias = (float*) calloc(n_out,sizeof(float)); 
   // cudaMallocManaged(&l1_weights, n_in * n_hidden*sizeof(float));
   // cudaMallocManaged(&l1_bias, n_hidden*sizeof(float));
   // cudaMallocManaged(&l2_weights, n_out * n_hidden*sizeof(float));
   // cudaMallocManaged(&l2_bias, n_out*sizeof(float));
    int n_block_rows = (BATCH_SIZE-1) / BLOCK_SIZE + 1;
    int l1_block_cols = (n_hidden - 1) / BLOCK_SIZE +1;
    int l2_block_cols = (n_out - 1) / BLOCK_SIZE +1;
    int relu_blocks = (n_hidden * BATCH_SIZE - 1) / BLOCK_SIZE+1;

    dim3 l1_grid(n_block_rows, l1_block_cols);
    dim3 l2_grid(n_block_rows, l2_block_cols);

    dim3 Block(BLOCK_SIZE, BLOCK_SIZE);

    cout << "row: "<< n_block_rows << " l1 cols: " << l1_block_cols << " l2 cols: " << l2_block_cols << endl;
 
    kaiming_init(l1_weights, n_in, n_hidden);
    init_zero(l1_bias, n_hidden);

    kaiming_init(l2_weights, n_hidden, n_out);
    init_zero(l2_bias, n_out);
    //Copy over to the corresponding device memory 
    cudaMalloc((void**) &dev_l1_weights, n_in * n_hidden * sizeof(float)); 
    cudaMalloc((void**) &dev_l1_bias, n_hidden * sizeof(float)); 
    cudaMalloc((void**) &dev_l2_weights, n_out * n_hidden * sizeof(float)); 
    cudaMalloc((void**) &dev_l2_bias, n_out * sizeof(float)); 
    float error;
    float *input, *target, *output;
    float *l1_out, *relu_out, *l2_out;
    float *dev_l1_out, *dev_l2_out, *dev_relu_out;
    float *l2_error, *l1_error, *relu_error;

    //Sizes for layer intialization
    //Layer input and outputs
    l1_out = (float*)calloc(n_hidden * BATCH_SIZE ,sizeof(float));
    l2_out = (float*)calloc(n_out * BATCH_SIZE,sizeof(float)); 
    relu_out = (float*)calloc(n_hidden* BATCH_SIZE,sizeof(float));
    cudaMalloc((void**)&dev_l1_out, size_hidden);
    cudaMalloc((void**)&dev_l2_out, size_output);
    cudaMalloc((void**)&dev_relu_out, size_hidden);
  //  cudaMallocManaged(&l1_out, n_hidden * BATCH_SIZE*sizeof(float));
   // cudaMallocManaged(&l2_out, n_out * BATCH_SIZE*sizeof(float));
   // cudaMallocManaged(&relu_out, n_hidden * BATCH_SIZE*sizeof(float));
    //Input Output target
    cudaMallocManaged(&input, n_in * data_size*sizeof(float));
    cudaMallocManaged(&output, n_out * BATCH_SIZE*sizeof(float));
    cudaMallocManaged(&target, n_out * data_size*sizeof(float));

    //Error
//    cudaMallocManaged(&l1_error, n_hidden * BATCH_SIZE*sizeof(float));
//    cudaMallocManaged(&l2_error, n_out * BATCH_SIZE*sizeof(float));
//    cudaMallocManaged(&relu_error, n_hidden * BATCH_SIZE*sizeof(float));
    cudaMalloc((void**)&l1_error, size_hidden);
    cudaMalloc((void**)&l2_error, size_output);
    cudaMalloc((void**)&relu_error, size_hidden);
    cudaMemcpy(dev_l1_weights, l1_weights, n_in * n_hidden * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_l1_bias, l1_bias, n_hidden * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_l2_weights, l2_weights, n_out * n_hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_l2_bias, l2_bias, n_out * sizeof(float), cudaMemcpyHostToDevice); 

    set_eq(input, &x_train[0], n_in * data_size);
    set_eq(target, &y_train[0], n_out * data_size); 
    //Make sure everything is synced
    cudaDeviceSynchronize();
    float forward_time = 0, backprop_time = 0;
    chrono::steady_clock::time_point begin, end;
    chrono::steady_clock::time_point b, e;

    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    begin = chrono::steady_clock::now();
    cout << "===TRAINING===" << endl;

    for (int i = 0; i < n_epochs; i++) {
        cout << "Epoch " << i << "\n";
        for (int batch = 0; batch < train_test_split / BATCH_SIZE; batch++) {
            float *curr_in = &input[batch * BATCH_SIZE * n_in];
            float *curr_target = &target[batch * BATCH_SIZE * n_out];
            // set_eq(input, &x_train[batch * BATCH_SIZE * n_in], n_in * BATCH_SIZE);
            // set_eq(target, &y_train[batch * BATCH_SIZE * n_out], n_out * BATCH_SIZE); 
//            cudaMemcpy(dev_curr_in, &input[batch * BATCH_SIZE * n_out], n_in * BATCH_SIZE * sizeof(float),cudaMemcpyHostToDevice);
//            cudaMemcpy(dev_curr_in, &input[batch * BATCH_SIZE * n_out], n_in * BATCH_SIZE * sizeof(float),cudaMemcpyHostToDevice);
            cudaMemset(relu_error, 0, n_hidden * BATCH_SIZE*sizeof(float));
            // FORWARD PROPAGATION STEP
            b = chrono::steady_clock::now();
            linear_forward_gpu<<<l1_grid, Block>>>(curr_in,dev_l1_out, dev_l1_weights, dev_l1_bias, n_in, n_hidden);
            relu_forward_gpu<<<relu_blocks, BLOCK_SIZE>>>(dev_l1_out,dev_relu_out, n_hidden);
            linear_forward_gpu<<<l2_grid, Block>>>(dev_relu_out, dev_l2_out, dev_l2_weights, dev_l2_bias, n_hidden, n_out);
            softmax_forward_gpu<<<l2_grid, Block>>>(dev_l2_out, output, n_out);
            cudaDeviceSynchronize();
            e = chrono::steady_clock::now();
            forward_time += (chrono::duration_cast<chrono::microseconds>(e - b).count());
            // BACKPROPAGATION STEP
            b = chrono::steady_clock::now();
            softmax_CE_backprop_gpu<<<l2_grid,Block>>>(curr_target, output, l2_error, n_out);
            linear_backprop_gpu<<<l2_grid, Block>>>(l2_error, relu_error, dev_l2_weights, n_hidden, n_out);
            relu_backward_gpu<<<relu_blocks, BLOCK_SIZE>>>(dev_l1_out, relu_error, l1_error, n_hidden);
            linear_update_gpu<<<l2_grid, Block>>>(dev_relu_out, l2_error, dev_l2_weights, dev_l2_bias, n_hidden, n_out, lr);
            linear_update_gpu<<<l1_grid, Block>>>(curr_in, l1_error, dev_l1_weights, dev_l1_bias, n_in, n_hidden, lr);
            cudaDeviceSynchronize();
            e = chrono::steady_clock::now();
            backprop_time += (chrono::duration_cast<chrono::microseconds>(e - b).count());
            // cout << "error: " << error << endl;
        }
        // cout << "error: " << error << endl;

    }
    float elapsed_time_ms; 
    cout << "===TRAINING COMPLETE===" << endl;
    end = chrono::steady_clock::now();
    cudaEventRecord(stop,0); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&elapsed_time_ms, start, stop); 
    cout << "Training time: " << (chrono::duration_cast<chrono::microseconds>(end - begin).count()) / 1000000.0f << "s" << endl;
    cout << "Forward propagation time: " << forward_time / 1000000.0f << "s" << endl;
    cout << "Backpropagation time: " << backprop_time / 1000000.0f << "s" << endl;

    cout << "===TESTING===" << endl;
    //Copy the dev wiehgts and biases out 
    cout << "Cuda Timing: " << elapsed_time_ms << endl; 

    cudaMemcpy(l1_weights, dev_l1_weights, n_in * n_hidden * sizeof(float), cudaMemcpyDeviceToHost); 
    cudaMemcpy(l1_bias, dev_l1_bias, n_hidden * sizeof(float), cudaMemcpyDeviceToHost); 
    cudaMemcpy(l2_weights, dev_l2_weights, n_out * n_hidden * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(l2_bias, dev_l2_bias, n_out * sizeof(float), cudaMemcpyDeviceToHost); 

    cudaFree(dev_l1_out); 
    cudaFree(dev_l2_out); 
    cudaFree(dev_relu_out); 
    cudaFree(dev_l1_weights); 
    cudaFree(dev_l2_weights); 
    cudaFree(dev_l1_bias); 
    cudaFree(dev_l2_bias); 

    int last_test_batch = data_size / BATCH_SIZE;
    int first_test_batch = train_test_split / BATCH_SIZE;
    float avg_acc = 0.;
    float avg_err = 0.;

    for (int batch = first_test_batch; batch < last_test_batch; batch++) {
        input = &x_train[batch * BATCH_SIZE * n_in];
        target = &y_train[batch * BATCH_SIZE * n_out];

        l1_out = new float[n_hidden * BATCH_SIZE];
        linear_forward_cpu(input, l1_out, l1_weights, l1_bias, n_in, n_hidden);

        relu_out = new float[n_hidden * BATCH_SIZE];
        relu_forward_cpu(l1_out, relu_out, n_hidden);

        l2_out = new float[BATCH_SIZE * n_out];
        linear_forward_cpu(relu_out, l2_out, l2_weights, l2_bias, n_hidden, n_out);

        output = new float[BATCH_SIZE * n_out];
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
