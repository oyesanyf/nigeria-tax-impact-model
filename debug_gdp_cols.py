import pandas as pd
import os

FILEPATH = "dataset/SDMS_Bulk_Data_Download_2026_01_06_02_44_43.xlsx"

def inspect_columns():
    print(f"Reading {FILEPATH}...")
    try:
        df = pd.read_excel(FILEPATH)
        print(f"Shape: {df.shape}")
        
        print("\nPossible GDP Columns:")
        for col in df.columns:
            if "GDP" in str(col) and "Quarterly" in str(col):
                print(f"\nCOLUMN: {col}")
                print(f"First 10 values: {df[col].head(10).tolist()}")
                print(f"Max: {df[col].max()}, Min: {df[col].min()}")
                
    except Exception as e:
        print(e)
        
if __name__ == "__main__":
    inspect_columns()
