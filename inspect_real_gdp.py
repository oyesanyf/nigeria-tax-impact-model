import pandas as pd

FILEPATH = "dataset/Real_GDP_in_Excel.xlsx"

def inspect():
    print(f"Reading {FILEPATH}...")
    try:
        # Read first few rows to see structure
        df = pd.read_excel(FILEPATH)
        print("Columns:", list(df.columns))
        print("Head:")
        print(df.head(10).to_string())
        
        # Check standard "GDP" keywords
        for c in df.columns:
            if "total" in str(c).lower() or "gdp" in str(c).lower():
                print(f"\nPotential Col: {c}")
                print(df[c].head())
                
    except Exception as e:
        print(e)

if __name__ == "__main__":
    inspect()
