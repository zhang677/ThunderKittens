import pandas as pd
import matplotlib.pyplot as plt
import re
import sys

if len(sys.argv) != 3:
    print("Usage: python plot_latency.py <csv_file> <figure_file>")
    sys.exit(1)

# Read the CSV file
df = pd.read_csv(sys.argv[1])

# Extract the N value from the shape column using regex
df['N'] = df['shape'].apply(lambda x: int(re.search(r'(\d+)x(\d+)x(\d+)x(\d+)x(\d+)', x).group(4)))

# Extract B, H, M, D values for the title
shape_parts = re.search(r'(\d+)x(\d+)x(\d+)x(\d+)x(\d+)', df['shape'][0])
B, H, M, D = shape_parts.group(1), shape_parts.group(2), shape_parts.group(3), shape_parts.group(5)
title = f"Latency vs N (B={B}, H={H}, M={M}, D={D})"

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df['N'], df['latency'], marker='o', linestyle='-', linewidth=2)
plt.xlabel('N (KV Length)')
plt.ylabel('Latency (cycles)')
plt.title(title)
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig(sys.argv[2], dpi=300)
plt.close()
print(f"Plot created and saved as {sys.argv[2]}")