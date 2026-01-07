
# ======================================================
# FETCH INFLATION DATA (Data Commons Client)
# Variable: sdg/FP_CPI_TOTL_ZG (Inflation, consumer prices)
# ======================================================
import pandas as pd
import os

def fetch_inflation():
    print("--- Nigeria Inflation Fetcher (DC Client) ---")

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

    # 3. Initialize & Fetch
    print("Requesting Inflation Data (sdg/FP_CPI_TOTL_ZG)...")
    try:
        client = DataCommonsClient(api_key=api_key)
        
        response = client.observation.fetch(
            variable_dcids=["sdg/FP_CPI_TOTL_ZG"],
            date='all',  # Critical: Get full history
            entity_dcids=["country/NGA"]
        )
        
        # 4. Convert to DataFrame
        # Get list of dicts from the response object
        records = response.to_observation_records()
        
        # Depending on version, records might be a list or a Pydantic model
        if hasattr(records, 'model_dump'):
            data = records.model_dump() # Pydantic v2
        elif hasattr(records, 'dict'):
             data = records.dict() # Pydantic v1
        else:
             data = records # It's just a list
             
        df = pd.DataFrame(data)
        
        if not df.empty:
            print(f"\nSuccessfully fetched {len(df)} records.")
            print(df.head())
            
            # Save
            output_file = "dataset/DC_INFLATION_NGA.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")
            
        else:
            print("Response returned no data records.")

    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    fetch_inflation()
