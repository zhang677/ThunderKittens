#!/bin/bash

mkdir -p profile_results

B=17
H=8

for M in 16 32 48 64; do
    for D in 64 96 128 160; do
        SUMMARY_FILE="profile_results/output_${B}x${H}x${M}x${D}.csv"
        if [ -f $SUMMARY_FILE ]; then
            echo "Summary file $SUMMARY_FILE already exists. Skipping..."
            continue
        fi
        # Write header to the CSV file
        echo "shape,latency" > $SUMMARY_FILE
        for tN in 16 32 96 160 224 268 332 396; do
            # Calculate N based on the current value of tN
            N=$((tN * 16))
            PROBLEM_SHAPE="${B}x${H}x${M}x${N}x${D}"
            echo "Running $PROBLEM_SHAPE"
            ncu --export profile_results/ncu_report -f --set full --target-processes all python run_once.py $B $H $M $N $D

            echo "Extracting metrics"
            ncu --import profile_results/ncu_report.ncu-rep --kernel-name "attend_ker" --csv --page details --log-file profile_results/output_$PROBLEM_SHAPE.csv

            echo "Processing metrics"
            python filter_output.py profile_results/output_$PROBLEM_SHAPE.csv profile_results/short_$PROBLEM_SHAPE.csv

            # Extract the latency from the short CSV file
            python extract_latency.py $PROBLEM_SHAPE profile_results/short_$PROBLEM_SHAPE.csv $SUMMARY_FILE

            # Clean up the intermediate files
            rm profile_results/output_$PROBLEM_SHAPE.csv
        done
    done
done
