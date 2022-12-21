# MNIST Digit Classification using a GPU-accelerated Neural Network 

We investigate the performance gains in training time from using GPUs vs CPUs for a simple neural network. This involves implementing serial and parallel versions of the forward propagation and backpropagation algorithms. We evaluate performance by using the MNIST dataset. 

Overall, we were able to achieve speedups of over 2 orders of magnitude. The scale-up factor also seemed relatively linear for smaller models and smaller batch sizes. While performance did take a hit for larger models and larger batch sizes, this could easily be remedied by training for more epochs.
