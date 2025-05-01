#!/bin/bash
BASE_DIR=profile_results_l40s
mkdir -p $BASE_DIR
# Add /usr/local/cuda/nsight-compute-2024.3.2/extras/python to Python path
export PYTHONPATH=$PYTHONPATH:/usr/local/cuda/nsight-compute-2024.3.2/extras/python
B=17
H=8
M=$1
tN=$2
D=$3
N=$((tN * 16))
PROBLEM_SHAPE="${B}x${H}x${M}x${N}x${D}"
echo "Running $PROBLEM_SHAPE"
ncu --launch-skip 4 --launch-count 1 --export $BASE_DIR/ncu_report_$PROBLEM_SHAPE -f --set full --target-processes all python run_once.py $B $H $M $N $D
python extract_memory.py $PROBLEM_SHAPE $BASE_DIR/ncu_report_$PROBLEM_SHAPE.ncu-rep $BASE_DIR/memory_$PROBLEM_SHAPE.csv