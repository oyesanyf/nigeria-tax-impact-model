import pandas as pd
import os

DATASET_DIR = "dataset"
FILES = [
    "Crude_Oil_Data_in_Excel.xlsx",
    "Financial_Data_in_Excel.xlsx",
    "Real_GDP_in_Excel.xlsx"
]

def inspect_clean():
    for filename in FILES:
        filepath = os.path.join(DATASET_DIR, filename)
        print(f"\n{'#'*30}")
        print(f"FILE: {filename}")
        print(f"{'#'*30}")
        
        try:
            xl = pd.ExcelFile(filepath)
            for sheet in xl.sheet_names:
                print(f"  SHEET: {sheet}")
                df = xl.parse(sheet)
                print(f"    COLUMNS: {list(df.columns.values)}")
                print("    FIRST Row Values:")
                print(f"      {df.iloc[0].values if not df.empty else 'EMPTY'}")
        except Exception as e:
            print(f"    ERROR: {e}")

if __name__ == "__main__":
    inspect_clean()
