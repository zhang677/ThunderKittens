import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import sys
import os

def extract_n(shape_str):
    """Extract the N dimension (4th value) from the shape string"""
    parts = shape_str.split('x')
    if len(parts) >= 5:
        return int(parts[3])
    return 0

def load_and_process_file(filename):
    """Load a shape-latency CSV file and extract the N dimension"""
    df = pd.read_csv(filename)
    
    # Extract N dimension from shape string
    df['N'] = df['shape'].apply(extract_n)
    
    # Sort by N dimension
    df = df.sort_values('N')
    
    return df

def plot_latency_curves(files, output_file="latency_plot.png"):
    """Plot latency vs N curves for multiple files"""
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers for different curves
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', 's', '^', 'x']
    
    # min_latency = float('inf')
    # max_latency = 0
    
    for i, file in enumerate(files):
        if i >= 4:  # Limit to 4 files
            print(f"Warning: Only processing the first 4 files. Skipping {file}")
            continue
            
        if not os.path.exists(file):
            print(f"Error: File {file} does not exist")
            continue
            
        # Get the filename without extension for the legend
        label = os.path.splitext(os.path.basename(file))[0].split('_')[1]
        
        # Load and process data
        try:
            df = load_and_process_file(file)
            
            # Update min/max latency
            # min_latency = min(min_latency, df['latency'].min())
            # max_latency = max(max_latency, df['latency'].max())
            
            # Plot the curve
            plt.plot(df['N'], df['latency'], 
                     marker=markers[i], 
                     color=colors[i], 
                     linestyle='-', 
                     linewidth=2, 
                     markersize=8,
                     label=label)
            
            sim_file = file.replace('output', 'simulated')
            df_sim = load_and_process_file(sim_file)
            plt.plot(df_sim['N'], df_sim['latency'],
                        marker=markers[i], 
                        color=colors[i], 
                        linestyle='--', 
                        linewidth=2, 
                        markersize=8,
                        label=f"{label} (simulated)")
            
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Set y-axis limits to the provided values
    # plt.ylim(26660, 1212093)  # Using the min and max values provided
    
    # Add labels and legend
    plt.xlabel('N Dimension', fontsize=14)
    plt.ylabel('Latency (cycles)', fontsize=14)
    plt.title('Latency vs N Dimension', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    # Set y-axis to logarithmic scale
    # plt.yscale('log')
    
    # # Add labels and legend
    # plt.xlabel('N Dimension', fontsize=14)
    # plt.ylabel('Latency (cycles)', fontsize=14)
    # plt.title('Latency vs N Dimension (Log Scale)', fontsize=16)
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend(fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved as {output_file}")


def main():
    """Main function to parse command line arguments and plot data"""
    if len(sys.argv) < 3 or len(sys.argv) > 6:
        print("Usage: python plot_latency.py output.png file1.csv [file2.csv file3.csv file4.csv]")
        sys.exit(1)

    # Get the files from command line arguments
    files = sys.argv[2:]
    
    # Plot the latency curves
    plot_latency_curves(files, output_file=sys.argv[1])

if __name__ == "__main__":
    main()