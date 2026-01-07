
import pandas_datareader.data as web
import pandas as pd
import datetime

def fetch_wb_tax():
    print("Attempting to fetch Tax Data from World Bank using pandas_datareader...")
    
    # World Bank Indicators
    # 1. Tax revenue (% of GDP): GC.TAX.TOTL.GD.ZS
    # 2. Tax revenue (current LCU): GC.TAX.TOTL.CN
    # 3. Total tax rate (% of commercial profits): IC.TAX.TOTL.CP.ZS
    
    indicators = {
        'GC.TAX.TOTL.GD.ZS': 'Tax_Revenue_Pct_GDP',
        'GC.TAX.TOTL.CN': 'Tax_Revenue_LCU',
        'IC.TAX.TOTL.CP.ZS': 'Total_Tax_Rate_Pct_Profit'
    }
    
    try:
        # Fetch for Nigeria (NGA)
        # Note: pandas_datareader might use 'wb' source.
        # We need to specify country='NG' or similar if supported, or filter after.
        
        from pandas_datareader import wb
        
        df = wb.download(indicator=list(indicators.keys()), country=['NG'], start=2000, end=2024)
        
        # Rename columns
        df = df.rename(columns=indicators)
        print("\nSuccessfully fetched data for Nigeria:")
        print(df)
        
        # Save
        output_file = "dataset/WB_TAX_DATA.csv"
        df.to_csv(output_file)
        print(f"\nSaved to {output_file}")
        
    except ImportError:
        print("Error: 'pandas_datareader' library is not installed.")
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_wb_tax()
