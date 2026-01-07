
import pandas as pd
import os

DATA_DIR = "dataset"
FILENAME = "SDMS_Bulk_Data_Download_2026_01_06_02_44_43.xlsx"

def check_sdms():
    filepath = os.path.join(DATA_DIR, FILENAME)
    if not os.path.exists(filepath):
        print(f"File {FILENAME} not found.")
        return

    print(f"Reading {FILENAME}...")
    df = pd.read_excel(filepath)
    
    print(f"Total Columns: {len(df.columns)}")
    
    # Search for keywords
    keywords = ['tax', 'vat', 'cit', 'revenue', 'income']
    found_cols = []
    
    for col in df.columns:
        c_str = str(col).lower()
        if any(k in c_str for k in keywords):
            found_cols.append(col)
            
    if found_cols:
        print("\nFound Potential Tax Columns:")
        for c in found_cols:
            print(f"- {c}")
            # Show sample data
            print(df[c].dropna().head().to_list())
    else:
        print("\nNo direct CIT/VAT columns found in SDMS file.")

if __name__ == "__main__":
    check_sdms()
