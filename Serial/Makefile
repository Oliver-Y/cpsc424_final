# This Makefile assumes the following module files are loaded:
#
# CUDA
#
# This Makefile will only work if executed on a GPU node.
#

NVCC = nvcc

NVCCFLAGS = -O3 -Wno-deprecated-gpu-targets -g

LFLAGS = -lm -Wno-deprecated-gpu-targets -g

# Compiler-specific flags (by default, we always use sm_37)
GENCODE_SM37 = -gencode=arch=compute_37,code=\"sm_37,compute_37\"
GENCODE = $(GENCODE_SM37)

.SUFFIXES : .cu .ptx

BINARIES = rectmatmul matmul

rectmatmul: rectmatmul.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<

matmul: matmul.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<

.cu.o:
	$(NVCC) $(GENCODE) $(NVCCFLAGS) -o $@ -c $<

clean:	
	rm -f *.o $(BINARIES)
