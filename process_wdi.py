import pandas as pd
import numpy as np
import os

# Config
DATA_DIR = r"D:/harfile/taxmodel/dataset/9JAP_Data_Extract_From_World_Development_Indicators"
WDI_FILE = os.path.join(DATA_DIR, "cbfafe9e-37ea-4dae-b849-58ac0512106d_Data.csv")
OUTPUT_FILE = "dataset/wdi_quarterly_interpolated.csv"

def clean_wdi_value(x):
    if str(x) == '..': return np.nan
    return pd.to_numeric(x, errors='coerce')

def process_wdi():
    print(f"Processing WDI Data from {WDI_FILE}...")
    df = pd.read_csv(WDI_FILE)
    
    # 1. Inspect Structure
    # WDI usually has 'Series Name', 'Series Code', 'Country Name', 'Country Code', and then Years like '2000 [YR2000]'
    year_cols = [c for c in df.columns if '[' in c and 'YR' in c]
    print(f"Found {len(year_cols)} Year Columns: {year_cols[0]} ... {year_cols[-1]}")
    
    # 2. Select Useful Indicators
    indicators = {
        'Tax revenue (% of GDP)': 'Tax_Rev_Pct_GDP',
        'Revenue, excluding grants (% of GDP)': 'Total_Rev_Pct_GDP',
        'Inflation, consumer prices (annual %)': 'Inflation_Rate',
        'Mobile cellular subscriptions (per 100 people)': 'Digital_Penetration',
        'Personal remittances, received (current US$)': 'Remittances_USD'
        # Add more if found useful in valid series list
    }
    
    # Filter for Nigeria (usually only one country in extraction but good to be safe)
    if 'Country Name' in df.columns:
        df = df[df['Country Name'] == 'Nigeria']
        
    df_subset = df[df['Series Name'].isin(indicators.keys())].copy()
    
    # 3. Transpose to Wide Format (Time Series)
    # Pivot: Index=Year, Columns=Indicator
    # We need to reshape.
    results = {}
    
    for _, row in df_subset.iterrows():
        series_name = row['Series Name']
        short_name = indicators.get(series_name)
        if not short_name: continue
        
        # Extract values for years
        ts_data = {}
        for yc in year_cols:
            year = int(yc.split(' ')[0])
            val = clean_wdi_value(row[yc])
            ts_data[year] = val
            
        results[short_name] = pd.Series(ts_data)
        
    df_ts_annual = pd.DataFrame(results)
    
    # Filter to 2000+
    df_ts_annual = df_ts_annual[df_ts_annual.index >= 2000].sort_index()
    
    print("\nAnnual Data (Head):")
    print(df_ts_annual.head())
    
    # 4. Upsample to Quarterly
    # Create Quarter Index
    dates = pd.date_range(start=f"{df_ts_annual.index.min()}-01-01", 
                          end=f"{df_ts_annual.index.max()}-12-31", 
                          freq='QE')
    
    # Reindex to Quarterly using the Dates
    # We set the annual value at Dec 31 (Q4) or interpolate linearly?
    # Linear interpolation is standard for annual-to-quarterly macro variables if better proxy missing.
    df_q = df_ts_annual.copy()
    df_q.index = pd.to_datetime(df_q.index, format='%Y') + pd.offsets.YearEnd(0) # Set to Dec 31
    df_q = df_q.resample('QE').asfreq() # Expand index
    df_q = df_q.interpolate(method='linear') # Fill gaps
    
    print("\nInterpolated Quarterly Data (Head):")
    print(df_q.head(8))
    
    df_q.to_csv(OUTPUT_FILE)
    print(f"Saved Quarterly WDI Data to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_wdi()
