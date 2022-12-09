#include <stdio.h>
#include <stdlib.h> 
#define FP float

//Define a Layer in C++ 
class Layer_CPU{
    public:
        //Probably add Bias later?
        FP *in, *out, *weights;  
        int in_size, out_size; 
    void forward(float *input, float *output); 
    void backward(); 
    void update(); 
}

//There is a weight per output node for each input node. So if we have n input nodes and p output nodes then we have an nxp matrix

// nxp weights


Layer_CPU::Layer_CPU(int input_size, int output_size){
    in_size = input_size; 
    out_size = output_size; 
    //nxp
    weights = malloc(in_size * out_size * sizeof(FP)); 
}


