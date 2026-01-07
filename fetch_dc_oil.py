
# ======================================================
# FETCH OIL EXPORT DATA (Data Commons Client)
# Variable: Annual_Exports_Fuel_CrudeOil
# ======================================================
import pandas as pd
import os

def fetch_oil():
    print("--- Nigeria Oil Exports Fetcher (DC Client) ---")

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
    print("Requesting Oil Data (Annual_Exports_Fuel_CrudeOil)...")
    try:
        client = DataCommonsClient(api_key=api_key)
        
        response = client.observation.fetch(
            variable_dcids=["Annual_Exports_Fuel_CrudeOil"],
            date='all',  # Get full history
            entity_dcids=["country/NGA"]
        )
        
        # 4. Convert to DataFrame
        records = response.to_observation_records()
        
        # Pydantic handling
        if hasattr(records, 'model_dump'):
            data = records.model_dump()
        elif hasattr(records, 'dict'):
             data = records.dict()
        else:
             data = records
             
        df = pd.DataFrame(data)
        
        if not df.empty:
            print(f"\nSuccessfully fetched {len(df)} records.")
            print(df.head())
            
            # Save
            output_file = "dataset/DC_OIL_EXPORTS_NGA.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")
            
        else:
            print("Response returned no data records.")

    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    fetch_oil()
