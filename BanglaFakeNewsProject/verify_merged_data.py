import pandas as pd
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# List all source CSVs to check (relative to BASE_DIR)
source_csvs = [
    os.path.join(BASE_DIR, "Authentic-48K.csv"),
    os.path.join(BASE_DIR, "Fake-1K.csv"),
    os.path.join(BASE_DIR, "LabeledAuthentic-7K.csv"),
    os.path.join(BASE_DIR, "LabeledFake-1K.csv")
]
merged_csv = os.path.join(BASE_DIR, "merged_data.csv")

# Preferred key columns to compare on (most specific to least)
KEY_PRIORITY = [
    ["articleID"],
    ["id"],
    ["news_id"],
    ["text"],
    ["content"],
]

def pick_key_columns(src_cols, merged_cols):
    # Try preferred keys first
    for keys in KEY_PRIORITY:
        if all(k in src_cols and k in merged_cols for k in keys):
            return keys
    # Fallback to intersection of columns
    inter = [c for c in src_cols if c in merged_cols]
    return inter

def normalize_df_cols(df, cols):
    # Coerce to string, fill NaN, and strip spaces for stable matching
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype(str).fillna("").str.strip()
    return df

# Helper: build a set of row hashes from merged_data.csv for the relevant columns
def build_merged_hash_set(common_cols, chunksize=5000):
    hash_set = set()
    for chunk in pd.read_csv(merged_csv, usecols=common_cols, dtype=str, engine="python", chunksize=chunksize):
        chunk = normalize_df_cols(chunk, common_cols)
        for row in chunk.itertuples(index=False, name=None):
            hash_set.add(tuple(row))
    return hash_set

# Get merged row count (optional, for info)
try:
    merged_row_count = sum(1 for _ in open(merged_csv, encoding='utf-8')) - 1  # minus header
except Exception:
    merged_row_count = 'unknown'
print(f"Merged data: {merged_row_count} rows\n")

all_found = True
for src in source_csvs:
    src_name = os.path.basename(src)
    if not os.path.exists(src):
        print(f"[WARNING] {src_name} not found, skipping.")
        continue
    # Read source as strings to align with merged
    df = pd.read_csv(src, dtype=str, engine="python")
    print(f"Checking {src_name}: {len(df)} rows...")
    # Decide which columns to compare
    merged_columns = pd.read_csv(merged_csv, nrows=0, engine="python").columns
    common_cols = pick_key_columns(df.columns, merged_columns)
    if not common_cols:
        print(f"  [ERROR] No common columns between {src_name} and merged_data.csv!")
        all_found = False
        continue
    # Normalize source on those columns
    df = normalize_df_cols(df, common_cols)
    # Build hash set for merged_data.csv for these columns
    merged_hash_set = build_merged_hash_set(common_cols)
    # For each row in src, check if its tuple exists in merged_hash_set
    missing = 0
    for row_tuple in df[common_cols].itertuples(index=False, name=None):
        if row_tuple not in merged_hash_set:
            missing += 1
    if missing == 0:
        print(f"  ✅ All rows from {src_name} are present in merged_data.csv")
    else:
        print(f"  ❌ {missing} rows from {src_name} are missing in merged_data.csv!")
        all_found = False

if all_found:
    print("\n✅ All source CSV rows are present in merged_data.csv!")
else:
    print("\n❌ Some rows are missing from merged_data.csv. See above.")
