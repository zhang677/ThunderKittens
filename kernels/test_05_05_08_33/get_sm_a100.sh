#!/bin/bash
sudo nvidia-smi -ac 1215,1065

BASE_DIR=profile_results_a100
mkdir -p $BASE_DIR
# export PYTHONPATH=$PYTHONPATH:/home/ubuntu/genghan/NVIDIA-Nsight-Compute-2025.2/extras/python
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/genghan/nsight-compute-2025.1/extras/python
N=256
H=108
SHAPE_FILE=shapes_smem.csv
tail -n +2 "$SHAPE_FILE" | while IFS=',' read -r M D B; do
    PROBLEM_SHAPE="${B}x${H}x${M}x${N}x${D}"
    echo "Running $PROBLEM_SHAPE"
    if [ -f "$BASE_DIR/ncu_report_$PROBLEM_SHAPE.ncu-rep" ]; then
        echo "Report file $BASE_DIR/ncu_report_$PROBLEM_SHAPE.ncu-rep already exists. Skipping..."
        continue
    fi
    ncu --launch-skip 4 --launch-count 1 \
        --export "$BASE_DIR/ncu_report_$PROBLEM_SHAPE" \
        -f --set full --target-processes all \
        python run_once.py $B $H $M $N $D
    
    python extract_block_limits.py $BASE_DIR/ncu_report_$PROBLEM_SHAPE.ncu-rep $BASE_DIR/occupancy_${M}x${D}.csv
done