import os
import pandas as pd
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Merge Sentiment CSVs for train/test/valid")
parser.add_argument('--split', type=str, choices=['train', 'test', 'valid'], required=True,
                    help="Choose which dataset split to process: train, test, or valid")
args = parser.parse_args()

# Root directory path based on split
base_dir = '/home/station_06/Sentimatic/dataset/Tweetsumm sentiment analysis'
root_dir = os.path.join(base_dir, args.split)

# List of subdirectories
subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# Variable to store the shared/common part
common_df = None

# Dictionary to store 'c2 - c1' values for each folder
c2_c1_dict = {}

for subdir in subdirs:
    csv_path = os.path.join(root_dir, subdir, f'{args.split}_SentimentLabel.csv')
    df = pd.read_csv(csv_path)

    # Extract shared columns only once (from the first folder)
    if common_df is None:
        common_df = df[['customer_01', 'agent', 'customer_02']].copy()

    # Store 'c2 - c1' column with the subfolder name as the key
    c2_c1_dict[subdir] = df['c2 - c1'].reset_index(drop=True)

# Create a DataFrame from all 'c2 - c1' values per folder
c2_c1_df = pd.DataFrame(c2_c1_dict)

# Concatenate the shared columns and the 'c2 - c1' values
final_df = pd.concat([common_df, c2_c1_df], axis=1)

# Optional: remove whitespace from column names
final_df.columns = final_df.columns.str.strip()

# Specify the columns to average (adjust if needed)
score_columns = ['o3-mini', 'gpt-4o', 'llama3.2']
available_score_columns = [col for col in score_columns if col in final_df.columns]

# Add a new column 'avg' with the row-wise mean of the selected columns
if available_score_columns:
    final_df['avg'] = final_df[available_score_columns].mean(axis=1)

# Save the result
output_path = os.path.join(root_dir, f'merged_{args.split}.csv')
final_df.to_csv(output_path, index=False)

print(f"✅ Merged file saved to: {output_path}")