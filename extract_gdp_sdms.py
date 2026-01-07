import pandas as pd
import os

FILENAME = "dataset/SDMS_Bulk_Data_Download_2026_01_06_02_44_43.xlsx"

def extract_gdp():
    try:
        xl = pd.ExcelFile(FILENAME)
        print("SHEETS found:", xl.sheet_names)
        
        # Find GDP sheet
        gdp_sheet = next((s for s in xl.sheet_names if "GDP" in s or "Real" in s), None)
        
        if gdp_sheet:
            print(f"\nExtracting GDP from sheet: {gdp_sheet}")
            df = xl.parse(gdp_sheet)
            print("Columns:", list(df.columns))
            print("First 5 rows:")
            print(df.head(5).to_string())
        else:
            print("No explicit GDP sheet found. Checking all sheets for 'Growth' column...")
            for s in xl.sheet_names:
                df = xl.parse(s)
                if any("growth" in str(c).lower() for c in df.columns):
                    print(f"Found 'Growth' in sheet {s}")
                    print(df.head().to_string())
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_gdp()
