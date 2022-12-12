# This Makefile assumes the following module files are loaded:
#
# CUDA
#
# This Makefile will only work if executed on a GPU node.
#

NVCC = nvcc

NVCCFLAGS = -O3 -Wno-deprecated-gpu-targets -g -std=c++11

LFLAGS = -lm -Wno-deprecated-gpu-targets -g -std=c++11

CFLAGS = -std=c++11

# Compiler-specific flags (by default, we always use sm_37)
GENCODE_SM37 = -gencode=arch=compute_37,code=\"sm_37,compute_37\"
GENCODE = $(GENCODE_SM37)

.SUFFIXES : .cu .ptx

BINARIES = gpu

gpu: gpu.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $(CFLAGS) $<

.cu.o: 
	$(NVCC) $(GENCODE) $(NVCCFLAGS) -o $@ $(CFLAGS) -c $<

clean:	
	rm -f *.o $(BINARIES)



# CC = g++

# CFLAGS = -std=c++11
 
# serial: serial.o load_mnist.o
# 	$(CC) -o $@ $(CFLAGS) $^

# %.o: %.cpp
# 	$(CC) $(CFLAGS) -c $<

# clean:
# 	rm -f serial *.o
