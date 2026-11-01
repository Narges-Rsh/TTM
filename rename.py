import pandas as pd
from pathlib import Path

# Folder containing all CSV files
input_dir = Path("/home/nrashvan/Token/raw_csv_files")

for csv_file in input_dir.glob("*.csv"):

    print(f"Processing: {csv_file.name}")

    # Read file assuming it currently has NO header
    df = pd.read_csv(csv_file, header=None)

    # Make sure it has exactly 3 columns
    if df.shape[1] != 2:
        print(
            f"Skipped {csv_file.name}: "
            f"found {df.shape[1]} columns instead of 2"
        )
        continue

    # Add column names
    df.columns = [
        "ground_truth",
        "raw_score",
    ]

    # Save back to the SAME CSV file
    df.to_csv(
        csv_file,
        index=False,
    )

    print(f"Updated: {csv_file.name}")

print("Done.")