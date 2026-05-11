
import pandas as pd
import numpy as np
import os

# Define file paths
authentic_48k_path = "C:\\Users\\Shawon\\OneDrive\\文档\\BanglaFakeNewsProject\\BanglaFakeNewsProject\\Authentic-48K.csv"
fake_1k_path = "C:\\Users\\Shawon\\OneDrive\\文档\\BanglaFakeNewsProject\\BanglaFakeNewsProject\\Fake-1K.csv"
labeled_authentic_7k_path = "C:\\Users\\Shawon\\OneDrive\\文档\\BanglaFakeNewsProject\\BanglaFakeNewsProject\\LabeledAuthentic-7K.csv"
labeled_fake_1k_path = "C:\\Users\\Shawon\\OneDrive\\文档\\BanglaFakeNewsProject\\BanglaFakeNewsProject\\LabeledFake-1K.csv"
output_path = "C:\\Users\\Shawon\\OneDrive\\文档\\BanglaFakeNewsProject\\BanglaFakeNewsProject\\merged_data.csv"

# Load the datasets
try:
    auth_48k = pd.read_csv(authentic_48k_path)
    auth_48k['label'] = 0
except FileNotFoundError:
    print(f"Warning: {authentic_48k_path} not found. Skipping.")
    auth_48k = pd.DataFrame()

try:
    fake_1k = pd.read_csv(fake_1k_path)
    fake_1k['label'] = 1
except FileNotFoundError:
    print(f"Warning: {fake_1k_path} not found. Skipping.")
    fake_1k = pd.DataFrame()

try:
    labeled_auth_7k = pd.read_csv(labeled_authentic_7k_path)
except FileNotFoundError:
    print(f"Warning: {labeled_authentic_7k_path} not found. Skipping.")
    labeled_auth_7k = pd.DataFrame()

try:
    labeled_fake_1k = pd.read_csv(labeled_fake_1k_path)
except FileNotFoundError:
    print(f"Warning: {labeled_fake_1k_path} not found. Skipping.")
    labeled_fake_1k = pd.DataFrame()


# Add all CSVs from images/ folder
images_dir = os.path.join(os.path.dirname(__file__), 'images')
image_csvs = []
if os.path.exists(images_dir):
    for fname in os.listdir(images_dir):
        if fname.lower().endswith('.csv'):
            image_csvs.append(os.path.join(images_dir, fname))

image_dfs = []
for path in image_csvs:
    try:
        df = pd.read_csv(path)
        image_dfs.append(df)
        print(f"Loaded {path} with {len(df)} rows.")
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")

# Combine all dataframes
all_dfs = [auth_48k, fake_1k, labeled_auth_7k, labeled_fake_1k] + image_dfs
all_dfs = [df for df in all_dfs if not df.empty]

if not all_dfs:
    print("No dataframes to merge. Exiting.")
else:
    # Get all columns
    all_columns = set()
    for df in all_dfs:
        all_columns.update(df.columns)

    # Align columns
    for df in all_dfs:
        for col in all_columns:
            if col not in df.columns:
                df[col] = np.nan

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Create image_id if it doesn't exist
    if 'image_id' not in merged_df.columns:
        # Use articleID if it exists, otherwise create a dummy one
        if 'articleID' in merged_df.columns:
            merged_df['image_id'] = merged_df['articleID']
        else:
            merged_df['image_id'] = range(len(merged_df))
    
    # Fix data types - convert float columns that should be integers back to int
    # This happens when concat adds NaN to integer columns
    for col in merged_df.columns:
        if merged_df[col].dtype == 'float64':
            # Check if this column was originally an integer (no decimal values when not NaN)
            non_null = merged_df[col].dropna()
            if len(non_null) > 0 and (non_null % 1 == 0).all():
                # Convert to Int64 (nullable integer type to preserve NaN)
                merged_df[col] = merged_df[col].astype('Int64')

    # Save the merged dataframe
    merged_df.to_csv(output_path, index=False)

    print(f"Successfully merged the data into {output_path}")
    print(f"Total rows: {len(merged_df)}")
    print(f"Columns: {list(merged_df.columns)}")
