#!/bin/bash
sudo nvidia-smi -ac 1215,1065
BASE_DIR=profile_results_a100
mkdir -p $BASE_DIR

H=108
N=6336
SHAPE_FILE=shapes_min.csv
tail -n +2 "$SHAPE_FILE" | while IFS=',' read -r M D tB; do
    SUMMARY_FILE="$BASE_DIR/output_${H}x${M}x${N}x${D}.csv"
    if [ -f $SUMMARY_FILE ]; then
        echo "Summary file $SUMMARY_FILE already exists. Skipping..."
        continue
    fi
    # Write header to the CSV file
    echo "shape,latency" > $SUMMARY_FILE

    for B in 1 $((tB - 2)) $((tB - 1)) $((tB)) $((tB + 1)) $((tB + 2)); do

        PROBLEM_SHAPE="${B}x${H}x${M}x${N}x${D}"
        echo "Running $PROBLEM_SHAPE"
        ncu --launch-skip 4 --launch-count 1 --export $BASE_DIR/ncu_report -f --set full --target-processes all python run_once.py $B $H $M $N $D

        echo "Extracting metrics"
        ncu --import $BASE_DIR/ncu_report.ncu-rep --kernel-name "attend_ker" --csv --page details --log-file $BASE_DIR/output_$PROBLEM_SHAPE.csv

        echo "Processing metrics"
        python filter_output.py $BASE_DIR/output_$PROBLEM_SHAPE.csv $BASE_DIR/short_$PROBLEM_SHAPE.csv

        # Extract the latency from the short CSV file
        python extract_latency.py $PROBLEM_SHAPE $BASE_DIR/short_$PROBLEM_SHAPE.csv $SUMMARY_FILE

        # Clean up the intermediate files
        rm $BASE_DIR/output_$PROBLEM_SHAPE.csv
    done
done
