import sys
import os
import csv
import pandas as pd

def extract_latency(problem_shape, profile_file, summary_file):
    df = pd.read_csv(profile_file)
    latency_data = df[df['Metric Name'] == 'Elapsed Cycles' and df['Section Name'] == 'GPU Speed Of Light Throughput']
    with open(summary_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([problem_shape, latency_data['Metric Value'].values[0]])

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_latency.py <problem_shape> <profile_file> <summary_file>")
        sys.exit(1)
    
    problem_shape = sys.argv[1]
    profile_file = sys.argv[2]
    summary_file = sys.argv[3]

    extract_latency(problem_shape, profile_file, summary_file)