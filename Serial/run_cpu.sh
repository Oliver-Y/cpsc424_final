#!/bin/bash

#SBATCH --job-name=NN_CPU
#SBATCH --output=%x-%j.out
#SBATCH --ntasks=1 --nodes=1 --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=30:00
#SBATCH --reservation=cpsc424

echo "***Building CPU"
make clean
make serial

echo "***Running"

echo ""
./serial 32
echo ""

echo ""
./serial 64
echo ""

echo ""
./serial 128
echo ""

echo ""
./serial 256




