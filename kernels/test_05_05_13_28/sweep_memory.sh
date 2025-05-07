#!/bin/bash
sudo nvidia-smi -ac 1215,1065

BASE_DIR=profile_results_a100
mkdir -p $BASE_DIR
# export PYTHONPATH=$PYTHONPATH:/home/ubuntu/genghan/NVIDIA-Nsight-Compute-2025.2/extras/python
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/genghan/nsight-compute-2025.1/extras/python
N=6336
B=108
SHAPE_FILE=../test_05_05_08_33/plot_results/merge_spill_latency.csv
# Skip header line and process each row
tail -n +2 "$SHAPE_FILE" | while IFS=',' read -r shape head rest; do
    # Extract M and H from the shape field (format: MxD)
    M=$(echo $shape | cut -d'x' -f1)
    D=$(echo $shape | cut -d'x' -f2)
    H=$head
    PROBLEM_SHAPE="${B}x${H}x${M}x${N}x${D}"
    echo "Running $PROBLEM_SHAPE"
    if [ -f "$BASE_DIR/ncu_report_$PROBLEM_SHAPE.ncu-rep" ]; then
        echo "Report file $BASE_DIR/ncu_report_$PROBLEM_SHAPE.ncu-rep already exists. Skipping..."
    else
        ncu --launch-skip 4 --launch-count 1 \
            --export "$BASE_DIR/ncu_report_$PROBLEM_SHAPE" \
            -f --set full --target-processes all \
            python run_once.py $B $M $N $D
    fi
    
    python extract_memory.py $PROBLEM_SHAPE $BASE_DIR/ncu_report_$PROBLEM_SHAPE.ncu-rep $BASE_DIR/memory_${PROBLEM_SHAPE}.csv
done
