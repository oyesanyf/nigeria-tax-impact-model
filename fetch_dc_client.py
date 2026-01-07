
# ==========================================
# FETCH TAX DATA USING GOOGLE DATA COMMONS (CLIENT VERSION)
# ==========================================
import pandas as pd
import os
import sys

def main():
    print("--- Nigeria Tax Data Fetcher (Data Commons Client) ---")

    # 1. SETUP API KEY
    api_key = os.environ.get("DC_API_KEY")
    if not api_key:
        print("[ERROR] DC_API_KEY environment variable not set.")
        print("Please run: $env:DC_API_KEY='your_key'")
        return

    # 2. Import Client
    try:
        from datacommons import DataCommonsClient
        # Note: The user code shows 'datacommons_client.client', but imports vary by version.
        # We will try the standard import or the specific one.
    except ImportError:
        try:
             # Fallback for newer package structure if installed as datacommons-client
            from datacommons_client.client import DataCommonsClient
        except ImportError:
            print("[ERROR] 'datacommons-client' library not found.")
            print("Please run: pip install 'datacommons-client[Pandas]'")
            return

    # 3. Initialize Client
    print("Initializing Client...")
    try:
        client = DataCommonsClient(api_key=api_key)
    except Exception as e:
         print(f"Error initializing client: {e}")
         return

    # 4. Define Query
    # We want NIGERIA (country/NGA) and Tax/GDP Variables
    entity_dcids = ["country/NGA"]
    
    # Updated List of likely variables
    variable_dcids = [
        "Amount_EconomicActivity_GrossDomesticProduction_Nominal", # GDP Nominal
        "Amount_EconomicActivity_GrossDomesticProduction_Real",    # GDP Real
        "Tax_Revenue_SNA", # Just a guess, let's stick to the user's example working one + WorldBank ones
        # "WorldBank/GC.TAX.TOTL.GD.ZS", # Tax % GDP (Needs to be supported by this API endpoint)
    ]
    
    print(f"Fetching data for: {entity_dcids}")
    print(f"Variables: {variable_dcids}")
    
    try:
        response = client.observation.fetch(
            variable_dcids=variable_dcids,
            entity_dcids=entity_dcids
        )
        
        # 5. Process Response
        # Convert to Pandas DataFrame
        # The user snippet suggests: response.to_observation_records().model_dump()
        # We need to see if .to_observation_records() exists
        
        if hasattr(response, 'to_observation_records'):
            records = response.to_observation_records()
            # If records is an object that has model_dump (Pydantic), use it
            if hasattr(records, 'model_dump'):
                 data = records.model_dump()
            else:
                 # It might be a list of records directly or similar
                 data = records
                 
            df = pd.DataFrame(data)
            
            print("\n--- RESULTS ---")
            print(df.head())
            
            output_file = "dataset/google_datacommons_client_data.csv"
            df.to_csv(output_file, index=False)
            print(f"\nSaved to {output_file}")
            
        else:
            print("\nReceived response (raw):")
            print(response)

    except Exception as e:
        print(f"\n[ERROR] Fetch failed: {e}")

if __name__ == "__main__":
    main()
