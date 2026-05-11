"""
Fix Merged Data - Properly merge all CSV files
"""

import pandas as pd
import os

def fix_merged_data():
    print("🔧 Fixing merged_data.csv...")
    
    # Source files
    source_files = [
        ("Authentic-48K.csv", 0),      # Real news
        ("Fake-1K.csv", 1),           # Fake news  
        ("LabeledAuthentic-7K.csv", 0), # Real news
        ("LabeledFake-1K.csv", 1)     # Fake news
    ]
    
    all_data = []
    
    for filename, label in source_files:
        if os.path.exists(filename):
            print(f"📁 Loading {filename}...")
            df = pd.read_csv(filename)
            
            # Standardize columns
            if 'headline' not in df.columns:
                if 'title' in df.columns:
                    df['headline'] = df['title']
                elif 'news_title' in df.columns:
                    df['headline'] = df['news_title']
                else:
                    df['headline'] = df.iloc[:, 0]  # First column
            
            if 'image_id' not in df.columns:
                if 'id' in df.columns:
                    df['image_id'] = df['id']
                elif 'articleID' in df.columns:
                    df['image_id'] = df['articleID']
                else:
                    df['image_id'] = range(len(df))
            
            # Add label
            df['label'] = label
            
            # Keep only needed columns
            df = df[['headline', 'image_id', 'label']].copy()
            
            # Remove duplicates and NaN
            df = df.dropna().drop_duplicates()
            
            print(f"   ✅ Added {len(df)} rows")
            all_data.append(df)
        else:
            print(f"   ⚠️ {filename} not found")
    
    # Combine all data
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # Final cleanup
        merged_df = merged_df.dropna().drop_duplicates()
        
        # Save
        merged_df.to_csv('merged_data.csv', index=False)
        
        print(f"\n✅ Created merged_data.csv with {len(merged_df)} rows")
        print(f"   Real news: {len(merged_df[merged_df['label'] == 0])}")
        print(f"   Fake news: {len(merged_df[merged_df['label'] == 1])}")
        
        return True
    else:
        print("❌ No source files found")
        return False

if __name__ == "__main__":
    fix_merged_data()