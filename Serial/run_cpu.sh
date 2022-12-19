#!/bin/bash

#SBATCH --job-name=NN_CPU
#SBATCH --output=%x-final%j.out
#SBATCH --ntasks=1 --nodes=1 --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=45:00
#SBATCH --reservation=cpsc424

echo "***Building CPU"
make clean
make serial

echo "***Running"

echo "===== HIDDEN LAYER 32 ======="
echo ""
./serial 32
echo ""

echo ""
./serial 32
echo ""

echo ""
./serial 32
echo ""

echo "===== HIDDEN LAYER 64 ======="
echo ""
./serial 64
echo ""

echo ""
./serial 64
echo ""

echo ""
./serial 64
echo ""

echo "===== HIDDEN LAYER 128 ======="

echo ""
./serial 128
echo ""

echo ""
./serial 128
echo ""

echo ""
./serial 128
echo ""


echo "===== HIDDEN LAYER 256 ======="
echo ""
./serial 256
echo ""

echo ""
./serial 256
echo ""

echo ""
./serial 256
echo ""




