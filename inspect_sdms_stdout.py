import pandas as pd
import os

DATASET_DIR = "dataset"
FILENAME = "SDMS_Bulk_Data_Download_2026_01_06_02_44_43.xlsx"
FILEPATH = os.path.join(DATASET_DIR, FILENAME)

def inspect():
    print(f"Inspecting {FILENAME}")
    
    try:
        xl = pd.ExcelFile(FILEPATH)
        print(f"Sheet Names: {xl.sheet_names}\n")
        
        for sheet in xl.sheet_names:
            print(f"--- Sheet: {sheet} ---")
            df = xl.parse(sheet)
            print(f"Columns: {list(df.columns)}")
            print("First 5 rows:")
            print(df.head(5).to_string())
            print("\n")
            
            # Check for Tax/Oil relevant columns
            relevant_cols = [c for c in df.columns if any(k in str(c).lower() for k in ['gdp', 'oil', 'tax', 'vat', 'revenue'])]
            if relevant_cols:
                print(f"RELEVANT COLUMNS FOUND: {relevant_cols}\n")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
