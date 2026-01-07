
import pandas as pd
import os

DATA_DIR = "dataset"
FILES_TO_CHECK = [
    "Financial_Data_in_Excel.xlsx",
    "5_Year_Financial_Statement_in_Excel.xlsx"
]

def inspect_file(filename):
    filepath = os.path.join(DATA_DIR, filename)
    print(f"\n{'='*50}")
    print(f"Inspecting: {filename}")
    print(f"{'='*50}")
    
    if not os.path.exists(filepath):
        print("File not found.")
        return

    try:
        # Load Excel file (just the sheet names first to be fast)
        xl = pd.ExcelFile(filepath)
        print(f"Sheet Names: {xl.sheet_names}")
        
        for sheet in xl.sheet_names:
            print(f"\n--- Sheet: {sheet} ---")
            # Read first few rows
            df = pd.read_excel(filepath, sheet_name=sheet, nrows=5)
            print("Columns:", list(df.columns))
            
            # Check for Tax keywords
            cols = [str(c).lower() for c in df.columns]
            keywords = ['cit', 'company income', 'vat', 'value added', 'tax']
            found = [c for c in cols if any(k in c for k in keywords)]
            if found:
                print(f"Make note! Found potential tax columns: {found}")
                
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    for f in FILES_TO_CHECK:
        inspect_file(f)
