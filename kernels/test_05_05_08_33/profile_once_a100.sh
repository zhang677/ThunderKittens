#!/bin/bash
sudo nvidia-smi -ac 1215,1065

BASE_DIR=profile_results_a100
mkdir -p $BASE_DIR
# export PYTHONPATH=$PYTHONPATH:/home/ubuntu/genghan/NVIDIA-Nsight-Compute-2025.2/extras/python
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/genghan/nsight-compute-2025.1/extras/python
B=$1
H=108
M=$2
tN=$3
D=$4
N=$((tN * 16))
PROBLEM_SHAPE="${B}x${H}x${M}x${N}x${D}"
echo "Running $PROBLEM_SHAPE"
ncu --launch-skip 4 --launch-count 1 --export $BASE_DIR/ncu_report_$PROBLEM_SHAPE -f --set full --target-processes all python run_once.py $B $H $M $N $D
python extract_memory.py $PROBLEM_SHAPE $BASE_DIR/ncu_report_$PROBLEM_SHAPE.ncu-rep $BASE_DIR/memory_$PROBLEM_SHAPE.csv
