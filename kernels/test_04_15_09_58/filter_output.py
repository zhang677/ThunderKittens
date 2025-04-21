import pandas as pd
import sys

def filter_csv(input_file, output_file):
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Check if required columns exist
        required_columns = ["Section Name", "Metric Name", "Metric Unit", "Metric Value"]
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Column '{col}' not found in the CSV file.")
                return False
        
        # Keep only the required columns
        filtered_df = df[required_columns]
        
        # Write to the output CSV file
        filtered_df.to_csv(output_file, index=False)
        
        print(f"Successfully filtered CSV and saved to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error processing the CSV file: {e}")
        return False

if __name__ == "__main__":
    # Check if input and output file paths are provided
    if len(sys.argv) != 3:
        print("Usage: python filter_csv.py input_file.csv output_file.csv")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        filter_csv(input_file, output_file)