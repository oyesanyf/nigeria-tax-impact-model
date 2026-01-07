import wbdata
import pandas as pd
import datetime

COUNTRY_CODE = "NG"
START_DATE = datetime.datetime(2000, 1, 1)
END_DATE = datetime.datetime(2026, 1, 1)

# Indicators
INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "GDP_Growth",          # Annual %
    "FP.CPI.TOTL.ZG": "Inflation_Rate",         # Annual %
}
OUTPUT_FILE = "dataset/wb_economic_data.csv"

def fetch():
    print("Fetching World Bank Data...")
    try:
        df = wbdata.get_dataframe(INDICATORS, country=COUNTRY_CODE, date=(START_DATE, END_DATE))
        
        if df.empty:
            print("No data returned.")
            return

        df = df.sort_index().reset_index()
        df.rename(columns={'date': 'Year'}, inplace=True)
        
        # Convert Year '2023' to Datetime 2023-12-31
        df['Date'] = pd.to_datetime(df['Year']) + pd.offsets.YearEnd(0)
        
        # Save
        df[['Date', 'GDP_Growth', 'Inflation_Rate']].to_csv(OUTPUT_FILE, index=False)
        print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
        print(df.head())
        print(df.tail())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch()
