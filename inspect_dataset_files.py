import pandas as pd
import os

DATASET_DIR = "dataset"
FILES_TO_INSPECT = [
    "SDMS_Bulk_Data_Download_2026_01_06_02_44_43.xlsx",
    "Crude_Oil_Data_in_Excel.xlsx",
    "Real_GDP_in_Excel.xlsx",
    "Financial_Data_in_Excel.xlsx"
]

def inspect_file(filename):
    filepath = os.path.join(DATASET_DIR, filename)
    print(f"\n{'='*50}")
    print(f"INSPECTING: {filename}")
    print(f"{'='*50}")
    
    if not os.path.exists(filepath):
        print("File not found.")
        return

    try:
        # Load Excel file (get sheet names first)
        xl = pd.ExcelFile(filepath)
        print(f"Sheet Names: {xl.sheet_names}")
        
        # Preview first sheet
        df = xl.parse(xl.sheet_names[0])
        print(f"\nPreview of Sheet '{xl.sheet_names[0]}':")
        print(df.head(5))
        print(f"\nColumns: {list(df.columns)}")
        
        # Search for specific keywords in Financial Data
        if "Financial" in filename or "SDMS" in filename:
            print("\nSearching for Tax Keywords (CIT, VAT) in columns or first column values...")
            keywords = ['vat', 'tax', 'cit', 'company income', 'value added']
            
            # Check columns
            found_cols = [c for c in df.columns if any(k in str(c).lower() for k in keywords)]
            if found_cols:
                print(f"Found Potential Tax Columns: {found_cols}")
                
            # Check first column values (often indicators are row labels in NBS data)
            if not df.empty:
                first_col = df.iloc[:, 0].astype(str).str.lower()
                matches = first_col[first_col.str.contains('|'.join(keywords), na=False)]
                if not matches.empty:
                    print(f"Found Potential Tax Rows in first column:\n{matches.head(10)}")

    except Exception as e:
        print(f"Error inspecting file: {e}")

if __name__ == "__main__":
    for f in FILES_TO_INSPECT:
        inspect_file(f)
