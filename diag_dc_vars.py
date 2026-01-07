
# ==========================================
# DIAGNOSTIC: CHECK AVAILABLE STAT VARS
# ==========================================
import datacommons as dc
import os

def check_stat_vars():
    print("--- Diagnostic: Exploring Available Data for Nigeria ---")
    
    # 1. Setup API Key
    api_key = os.environ.get("DC_API_KEY")
    if api_key:
        dc.set_api_key(api_key)
        print("✔ API Key set.")
    else:
        print("[!] No API Key set.")
        return

    place = "country/NGA"
    
    # 2. Variable Discovery
    # Let's ask Data Commons what statistical variables exist for this place.
    # We will search for 'Tax' related keywords in the available variables.
    
    print(f"\nSearching for variables available for {place}...")
    
    try:
        # dc.get_stat_all(place) returns a dictionary of ALL stats for the place.
        # This is heavy but useful for discovery.
        # Recent library versions might split this usage. 
        # Let's try to just check if our target variable exists explicitly.
        
        target = "WorldBank/GC.TAX.TOTL.GD.ZS"
        print(f"Checking {target}...")
        val = dc.get_stat_value(place, target)
        print(f" -> Current Value: {val}")
        
    except Exception as e:
        print(f"Error checking specific stat: {e}")

    try:
        # Try to find variables related to GDP to verify connectivity
        # "Amount_EconomicActivity_GrossDomesticProduction_Nominal_USD" is very common
        common_var = "Amount_EconomicActivity_GrossDomesticProduction_Nominal_USD"
        print(f"\nChecking common variable ({common_var})...")
        series = dc.get_stat_series(place, common_var)
        if series:
            print(f" -> Found {len(series)} data points.")
            print(f" -> Sample: {list(series.items())[:3]}")
        else:
            print(" -> No data found (Empty Series).")
            
    except Exception as e:
        print(f"Error checking common stat: {e}")

if __name__ == "__main__":
    check_stat_vars()
