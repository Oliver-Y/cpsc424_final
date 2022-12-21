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

echo "***Building GPU"
make clean
make gpu

echo "***Running"

echo "===== HIDDEN LAYER 32 ======="
echo ""
./gpu 32
echo ""


echo "===== HIDDEN LAYER 64 ======="
echo ""
./gpu 64
echo ""


echo "===== HIDDEN LAYER 128 ======="

echo ""
./gpu 128
echo ""


echo "===== HIDDEN LAYER 256 ======="
echo ""
./gpu 256
echo ""





