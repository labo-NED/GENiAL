import pandas as pd
import os

# Fix encoding issue for highest_degree.csv
file_path = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/highest_degree.csv'

print("Attempting to load highest_degree.csv with different encodings...")

# Try different encodings
encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

for encoding in encodings:
    try:
        print(f"Trying {encoding}...")
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"✅ Successfully loaded with {encoding} encoding!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
        
        # Save with proper UTF-8 encoding
        output_path = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/highest_degree_fixed.csv'
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ Saved fixed file to: {output_path}")
        break
        
    except UnicodeDecodeError as e:
        print(f"❌ Failed with {encoding}: {e}")
        continue
    except Exception as e:
        print(f"❌ Unexpected error with {encoding}: {e}")
        continue
else:
    print("❌ All encoding attempts failed. The file may be corrupted.")
