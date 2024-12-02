import pandas as pd

# Path to the CSV file
file_path = r"..\streams.csv"

# Read the data into a DataFrame without a header
df = pd.read_csv(file_path, header=None)

# Assign column names
df.columns = ["prio", "name", "type", "source", "endpoint", "size", "period", "deadline"]

# Calculate committedInformationRate and committedBurstSize
df['committedInformationRate'] = (df['size'] * 8) / (df['period'] / 1e6) / 1e6  # Convert to Mbps
df['committedBurstSize'] = df['size']  # Assuming burst size is the same as the size

# Group by priority and calculate the sum of committedInformationRate and sum of committedBurstSize
grouped = df.groupby('prio').agg({
    'committedInformationRate': 'sum',
    'committedBurstSize': 'sum'
}).reset_index()

# Ensure all priorities (0 to 7) are included
all_priorities = pd.DataFrame({'prio': range(8)})
grouped = pd.merge(all_priorities, grouped, on='prio', how='left').fillna(0)

# Generate the output format
output = []
for index, row in grouped.iterrows():
    output.append(f"*.Switch*.bridging.streamFilter.ingress.meter[{index}].committedInformationRate = {row['committedInformationRate']:.2f}Mbps")
    output.append(f"*.Switch*.bridging.streamFilter.ingress.meter[{index}].committedBurstSize = {row['committedBurstSize']}B")

# Write the output to a new CSV file
with open(r"output.csv", 'w') as f:
    for line in output:
        f.write(line + '\n')

print("Output written to output.csv")