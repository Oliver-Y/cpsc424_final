#!/bin/bash

#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=5 
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --job-name=NN_GPU
#SBATCH --output=%x-%j.out
#SBATCH --reservation=cpsc424gpu
#SBATCH -t 15:00
#SBATCH --gres-flags=enforce-binding
#SBATCH --gpus=1

echo "***Purging module files"
echo ""
module purge
echo ""
echo "***Loading CUDA module file"
echo ""
module load CUDA
echo ""
module list

# echo ""
# echo "***Running nvidia-smi"
# echo ""
# nvidia-smi
# echo ""
# echo ""

# echo "***Running deviceQuery"
# /vast/palmer/apps/avx.grace/software/CUDAcore/11.3.1/extras/demo_suite/deviceQuery
# echo ""

echo "***Building gpu"
make clean
make gpu

echo "***Running"


echo ""
./gpu 32
echo ""

echo ""
./gpu 64
echo ""

echo ""
./gpu 128
echo ""

echo ""
./gpu 256





