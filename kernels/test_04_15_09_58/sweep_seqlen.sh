#!/bin/bash

mkdir -p profile_results

B="17"
H="8"
M="32"
D="64"
SUMMARY_FILE="profile_results/output.csv"
# Write header to the CSV file
echo "shape,latency" > $SUMMARY_FILE

for tN in 16 32 96 160 224 268 332 396; do
    # Calculate N based on the current value of tN
    N=$((tN * 16))
    PROBLEM_SHAPE="$Bx$Hx$Mx$Nx$D"
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
