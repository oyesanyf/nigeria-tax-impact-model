import pandas as pd
import os

FILEPATH = r"D:/harfile/taxmodel/dataset/9JAP_Data_Extract_From_World_Development_Indicators/cbfafe9e-37ea-4dae-b849-58ac0512106d_Data.csv"

def inspect():
    try:
        print(f"Reading {FILEPATH}...")
        df = pd.read_csv(FILEPATH)
        print("Columns:", list(df.columns))
        print("First 5 rows:")
        print(df.head().to_string())
        
        # Check unique indicators
        if 'Series Name' in df.columns:
            print("\nUnique Series found:")
            print(df['Series Name'].unique())
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
