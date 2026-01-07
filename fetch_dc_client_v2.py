
# ==========================================
# FETCH TAX DATA USING DATA COMMONS CLIENT
# ==========================================
import pandas as pd
import os
import sys

def fetch_data_client_version():
    print("--- Nigeria Tax Data Fetcher (Client Version) ---")

    # 1. SETUP API KEY
    api_key = os.environ.get("DC_API_KEY")
    if not api_key:
        print("[ERROR] DC_API_KEY environment variable not set.")
        print("Please run: $env:DC_API_KEY='your_key'")
        return

    # 2. Import Client
    try:
        from datacommons_client.client import DataCommonsClient
    except ImportError:
        print("[ERROR] 'datacommons-client' library not found.")
        print("Please run: pip install 'datacommons-client[Pandas]'")
        return

    # 3. Initialize
    client = DataCommonsClient(api_key=api_key)

    # 4. Variables Definition (Same as successful sample code)
    # We use a broad list of variables to see what sticks.
    variables = [
        "Amount_EconomicActivity_GrossDomesticProduction_Nominal", # Your example (Working)
        "WorldBank/GC.TAX.TOTL.GD.ZS",   # Tax Revenue % GDP
        "WorldBank/GC.TAX.TOTL.CN",      # Tax Revenue LCU
        "sdg/FP_CPI_TOTL_ZG"             # Inflation
    ]
    
    entities = ["country/NGA"]
    
    print(f"Fetching {len(variables)} variables for Nigeria...")
    
    try:
        # 5. Fetch
        # Using the exact syntax from your sample
        response = client.observation.fetch(
            variable_dcids=variables,
            date='all',
            entity_dcids=entities
        )
        
        # 6. Process Records
        records = response.to_observation_records()
        
        # Robust conversion to dict (handle Pydantic versions)
        if hasattr(records, 'model_dump'):
            data = records.model_dump()
        elif hasattr(records, 'dict'):
             data = records.dict()
        else:
             data = records
             
        df = pd.DataFrame(data)
        
        if df.empty:
            print("[Warning] No records returned.")
            return

        print(f"\nSuccessfully fetched {len(df)} records!")
        print(df.head())
        
        # Pivot to make it readable (Date x Variable)
        # raw columns: 'variable', 'date', 'value', 'entity'
        if 'variable' in df.columns and 'value' in df.columns:
            df_pivot = df.pivot_table(index='date', columns='variable', values='value', aggfunc='first')
            print("\n--- Summary (First 5 Years) ---")
            print(df_pivot.head())
            
            output_file = "dataset/DC_CLIENT_RESULTS.csv"
            df_pivot.to_csv(output_file)
            print(f"\nSaved to {output_file}")
            
    except Exception as e:
        print(f"\n[ERROR] Fetch failed: {e}")

if __name__ == "__main__":
    fetch_data_client_version()
