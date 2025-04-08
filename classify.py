import os
import pandas as pd
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Split sentiment data into positive and negative based on avg score")
parser.add_argument('--split', type=str, choices=['train', 'test', 'valid'], required=True,
                    help="Choose which dataset split to process: train, test, or valid")
args = parser.parse_args()

# Set base path
base_dir = '/home/station_06/Sentimatic/dataset/Tweetsumm sentiment analysis'
root_dir = os.path.join(base_dir, args.split)

# Construct input file path
input_path = os.path.join(root_dir, f'merged_{args.split}_sentiment.csv')

# Load the merged dataset
df = pd.read_csv(input_path)

# Clean column names (optional, in case of extra spaces)
df.columns = df.columns.str.strip()

# Check if 'avg' column exists
if 'avg' not in df.columns:
    raise ValueError("❌ 'avg' column not found in the dataset. Please run concat.py first.")

# Split into positive and negative datasets
positive_df = df[df['avg'] > 0].reset_index(drop=True)
negative_df = df[df['avg'] < 0].reset_index(drop=True)

# Save the results
positive_path = os.path.join(root_dir, f'positive_{args.split}.csv')
negative_path = os.path.join(root_dir, f'negative_{args.split}.csv')

positive_df.to_csv(positive_path, index=False)
negative_df.to_csv(negative_path, index=False)

print(f"✅ Loaded: {input_path}")
print(f"✅ Positive ({len(positive_df)} rows) saved to: {positive_path}")
print(f"✅ Negative ({len(negative_df)} rows) saved to: {negative_path}")