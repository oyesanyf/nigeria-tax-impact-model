
# ======================================================
# FETCH ALL MISSING NIGERIA DATA (Data Commons Client)
# ======================================================
# This script fetches all key economic indicators needed for the Tax Model
# that might be missing or incomplete in local files.
#
# Variables:
# 1. Tax Revenue (% GDP)          -> WorldBank/GC.TAX.TOTL.GD.ZS
# 2. Tax Revenue (Local Currency) -> WorldBank/GC.TAX.TOTL.CN
# 3. Inflation (Consumer Prices)  -> sdg/FP_CPI_TOTL_ZG
# 4. Oil Exports (Crude)          -> Annual_Exports_Fuel_CrudeOil
# 5. GDP (Nominal)                -> Amount_EconomicActivity_GrossDomesticProduction_Nominal
# 6. GDP Growth (Annual %)        -> WorldBank/NY.GDP.MKTP.KD.ZG
# 7. Digital/Internet Usage       -> WorldBank/IT.NET.USER.ZS (Individuals using the Internet)
# 8. Remittances (Personal)       -> WorldBank/BX.TRF.PWKR.CD.DT (Personal remittances, received)
# ======================================================

import pandas as pd
import os
import time

def fetch_all_features():
    print("--- Nigeria Economic Data Fetcher (Data Commons) ---")

    # 1. SETUP API KEY
    api_key = os.environ.get("DC_API_KEY")
    if not api_key:
        print("[ERROR] DC_API_KEY environment variable not set.")
        return

    # 2. Import Client
    try:
        from datacommons_client.client import DataCommonsClient
    except ImportError:
        print("[ERROR] 'datacommons-client' library not found.")
        print("Please run: pip install \"datacommons-client[Pandas]\"")
        return
        
    client = DataCommonsClient(api_key=api_key)
    
    # 3. Define Variables
    variables = {
        "Tax_Revenue_Pct_GDP": "WorldBank/GC.TAX.TOTL.GD.ZS",
        "Tax_Revenue_LCU":     "WorldBank/GC.TAX.TOTL.CN",
        "Inflation_Rate":      "sdg/FP_CPI_TOTL_ZG",
        "Oil_Exports_USD":     "Annual_Exports_Fuel_CrudeOil",
        "GDP_Nominal_LCU":     "Amount_EconomicActivity_GrossDomesticProduction_Nominal",
        "GDP_Growth_Pct":      "WorldBank/NY.GDP.MKTP.KD.ZG",
        "Internet_Usage_Pct":  "WorldBank/IT.NET.USER.ZS",
        "Remittances_USD":     "WorldBank/BX.TRF.PWKR.CD.DT"
    }
    
    dcid_list = list(variables.values())
    print(f"Fetching {len(dcid_list)} variables for Nigeria...")
    
    # 4. Fetch
    try:
        response = client.observation.fetch(
            variable_dcids=dcid_list,
            date='all',
            entity_dcids=["country/NGA"]
        )
        
        # 5. Process
        records = response.to_observation_records()
        
        # Pydantic handling
        if hasattr(records, 'model_dump'):
            data = records.model_dump()
        elif hasattr(records, 'dict'):
             data = records.dict()
        else:
             data = records
             
        df_raw = pd.DataFrame(data)
        
        if df_raw.empty:
            print("No data returned.")
            return

        print(f"\nFetched {len(df_raw)} raw records.")
        
        # 6. Pivot/Format for the Model (Date Index, Variables as Columns)
        # The raw data has columns like: 'variable', 'date', 'value'
        # We need to reshape it.
        
        # Map DCIDs back to our friendly names
        # Create a reverse dictionary for mapping
        rev_vars = {v: k for k, v in variables.items()}
        
        df_raw['variable_name'] = df_raw['variable'].map(rev_vars)
        
        # Filter only what we asked for (just in case)
        df_clean = df_raw.dropna(subset=['variable_name'])
        
        # Pivot
        # Index: date, Columns: variable_name, Values: value
        df_pivoted = df_clean.pivot_table(index='date', columns='variable_name', values='value', aggfunc='first')
        
        # Sort index
        df_pivoted.sort_index(inplace=True)
        
        print("\n--- Consolidated Data (First 5 Rows) ---")
        print(df_pivoted.head())
        
        # Save
        output_file = "dataset/DC_ALL_INDICATORS_NGA.csv"
        df_pivoted.to_csv(output_file)
        print(f"\nSaved consolidated data to: {output_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Fetch failed: {e}")

if __name__ == "__main__":
    fetch_all_features()
