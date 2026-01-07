
# ==========================================
# FETCH TAX DATA USING GOOGLE DATA COMMONS
# ==========================================
# Prerequisites:
# 1. Install Python
# 2. Run: pip install datacommons pandas
# ==========================================

import datacommons as dc
import pandas as pd
import os

def main():
    print("--- Nigeria Tax Data Fetcher (Google Data Commons) ---")

    # 0. SETUP API KEY (REQUIRED)
    # You generally need a Google Cloud API Key for Data Commons.
    # Set it as an environment variable 'DC_API_KEY' or paste it below.
    api_key = os.environ.get("DC_API_KEY")
    
    if not api_key:
        print("\n[!] WARNING: No API Key found.")
        print("    Google Data Commons usually requires an API Key.")
        print("    1. Go to https://console.cloud.google.com/apis/credentials")
        print("    2. Create an API Key")
        print("    3. Set Env Var: $env:DC_API_KEY='your_key'")
        print("    OR edit this script to set: api_key = 'your_key'\n")
        # You can uncomment the line below and paste your key to test quickly:
        # api_key = "PASTE_YOUR_GOOGLE_API_KEY_HERE"
        
    if api_key:
        dc.set_api_key(api_key)
        print("✔ API Key set successfully.")

    # 1. Define Location: Nigeria
    # Data Commons uses 'dcid' (Data Commons ID)
    place_dcid = "country/NGA"
    
    # 2. Define Variables (DCIDs)
    # These map to World Bank indicators usually available in Data Commons.
    variables = {
        "WorldBank/GC.TAX.TOTL.GD.ZS": "Tax Revenue (% of GDP)",
        "WorldBank/IC.TAX.TOTL.CP.ZS": "Total Tax Rate (% of Commercial Profits)",
        "WorldBank/GC.TAX.TOTL.CN":    "Tax Revenue (Local Currency Unit)",
        "Amount_EconomicActivity_GrossDomesticProduct_Real": "Real GDP (USD)" # Native DCID example
    }
    
    print(f"Fetching data for: {place_dcid}...")
    print(f"Variables: {list(variables.keys())}")

    try:
        # 3. Fetch Data
        # 'build_multivariate_dataframe' is deprecated/removed in newer versions.
        # We will use 'get_stat_series' for each variable and combine them.
        
        data_frames = []
        
        for var_dcid, var_name in variables.items():
            print(f"Fetching {var_name} ({var_dcid})...")
            try:
                # Returns dict: {"YYYY-MM-DD": value, ...}
                series_data = dc.get_stat_series(place_dcid, var_dcid)
                
                if series_data:
                    # Convert to Series
                    s = pd.Series(series_data, name=var_name)
                    s.index = pd.to_datetime(s.index)
                    data_frames.append(s)
                else:
                    print(f" -> No data found for {var_name}")
            except Exception as e:
                print(f" -> Error fetching {var_name}: {e}")

        if not data_frames:
            print("\n[ERROR] No data retrieved for any variable.")
            return

        # 4. Clean Data
        # Combine all series into one DataFrame (outer join to keep all dates)
        df = pd.concat(data_frames, axis=1)
        
        # Sort by Date
        df = df.sort_index()
        
        print("\n--- RESULTS ---")
        print(df.head(10))
        print(f"\nTotal Rows: {len(df)}")
        
        # Save to CSV
        output_file = "dataset/google_datacommons_tax_data.csv"
        df.to_csv(output_file)
        print(f"\nSaved data to: {output_file}")
        
    except ImportError:
        print("\n[ERROR] 'datacommons' library not found.")
        print("Please run: pip install datacommons")
    except Exception as e:
        print(f"\n[ERROR] Could not fetch data: {e}")

if __name__ == "__main__":
    main()
