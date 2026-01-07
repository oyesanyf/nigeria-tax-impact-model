import wbdata
import pandas as pd
import datetime

# Configuration
COUNTRY_CODE = "NG"  # Nigeria
INDICATOR = "FP.CPI.TOTL.ZG"  # Inflation, consumer prices (annual %)
START_DATE = datetime.datetime(2000, 1, 1)
END_DATE = datetime.datetime(2026, 1, 1)
OUTPUT_FILE = "dataset/inflation_wb_data.csv"

def fetch_inflation():
    print(f"Fetching Inflation Data for {COUNTRY_CODE} from World Bank...")
    
    indicators = {INDICATOR: "Inflation_Rate"}
    
    try:
        df = wbdata.get_dataframe(indicators, country=COUNTRY_CODE, date=(START_DATE, END_DATE))
        
        if df.empty:
            print("No data returned!")
            return

        # WB Data returns Index as Date, but often annual '2020', '2019'.
        # Let's inspect index
        print("\nRaw Data Head:")
        print(df.head())
        
        # Sort chronological
        df = df.sort_index()
        
        # Reset index to make date a column
        df = df.reset_index()
        df.rename(columns={'date': 'Year'}, inplace=True)
        
        # Process dates: WB usually gives '2023' as string. 
        # Convert to Dec 31st of that year for timeseries alignment
        df['Date'] = pd.to_datetime(df['Year']) + pd.offsets.YearEnd(0)
        
        # Save raw annual fetch
        df[['Date', 'Inflation_Rate']].to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved {len(df)} rows to {OUTPUT_FILE}")
        print(df.tail())
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_inflation()
