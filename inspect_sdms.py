import pandas as pd
import os

DATASET_DIR = "dataset"
FILENAME = "SDMS_Bulk_Data_Download_2026_01_06_02_44_43.xlsx"
FILEPATH = os.path.join(DATASET_DIR, FILENAME)
OUTPUT_FILE = "inspection_results.txt"

def inspect():
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"Inspecting {FILENAME}\n")
        
        try:
            xl = pd.ExcelFile(FILEPATH)
            f.write(f"Sheet Names: {xl.sheet_names}\n\n")
            
            for sheet in xl.sheet_names:
                f.write(f"--- Sheet: {sheet} ---\n")
                df = xl.parse(sheet)
                f.write(f"Columns: {list(df.columns)}\n")
                f.write("First 5 rows:\n")
                f.write(df.head(5).to_string())
                f.write("\n\n")
                
                # Check for Tax/Oil relevant columns
                relevant_cols = [c for c in df.columns if any(k in str(c).lower() for k in ['gdp', 'oil', 'tax', 'vat', 'revenue'])]
                if relevant_cols:
                    f.write(f"RELEVANT COLUMNS FOUND: {relevant_cols}\n")
                    
        except Exception as e:
            f.write(f"Error: {e}\n")

if __name__ == "__main__":
    inspect()
