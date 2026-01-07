import pandas as pd

FILENAME = "dataset/SDMS_Bulk_Data_Download_2026_01_06_02_44_43.xlsx"

def find_columns():
    df = pd.read_excel(FILENAME)
    print("Total Columns:", len(df.columns))
    
    print("\n--- GDP Columns ---")
    vals = [c for c in df.columns if "GDP" in str(c) or "Domestic" in str(c)]
    for v in vals: print(v)
    
    print("\n--- TAX / REVENUE Columns ---")
    vals = [c for c in df.columns if "Tax" in str(c) or "VAT" in str(c) or "Rev" in str(c) or "Alloc" in str(c)]
    for v in vals: print(v)
    
    print("\n--- DATE Column ---")
    vals = [c for c in df.columns if "Date" in str(c) or "Year" in str(c) or "Time" in str(c)]
    for v in vals: print(v)

if __name__ == "__main__":
    find_columns()
