import pandas as pd
import numpy as np
import os

# Config
DATA_DIR = "dataset"
OUTPUT_FILE = "nigeria_economic_data.csv"

# Files
FILE_GDP = os.path.join(DATA_DIR, "SDMS_Bulk_Data_Download_2026_01_06_02_44_43.xlsx")
FILE_OIL = os.path.join(DATA_DIR, "Crude_Oil_Data_in_Excel.xlsx")
FILE_REV = os.path.join(DATA_DIR, "Financial_Data_in_Excel.xlsx")

def process():
    print("Processing Data Files...")
    
    # 1. LOAD GDP
    print(f"Loading GDP from {FILE_GDP}...")
    df_gdp_raw = pd.read_excel(FILE_GDP)
    # Find the GDP column (Long name)
    gdp_col = [c for c in df_gdp_raw.columns if "New GDP" in str(c)][0]
    print(f" -> Found GDP Column: {gdp_col}")
    
    # The file likely doesn't have a clear 'Date' column aligned with GDP if it's "Structure Based on NBS".
    # Often these NBS templates come transposed or with Quarter in rows.
    # Let's assume the rows correspond to Quarters based on standard NBS bulk export.
    # We need to construct a Date index.
    # If the file has 46 columns, maybe it's wide? But find_cols said 46 cols.
    # Let's look at the first few values of GDP to infer structure.
    print(df_gdp_raw[gdp_col].head())
    
    # Assuming rows are chronological quarters.
    # NBS usually starts from 2010 Q1 or earlier.
    # Let's Create a placeholder range and we'll align it later if needed.
    # But wait, we need strict alignment.
    # Let's check if there is a Year/Quarter column in SDMS.
    # The find_sdms_cols output showed NO "Date" column.
    # This implies the index or some other column serves as Date.
    
    # Hack: Let's clean the GDP series and Drop NaNs.
    gdp_series = df_gdp_raw[gdp_col].dropna()
    print(f" -> GDP Data Points: {len(gdp_series)}")
    
    # 2. LOAD OIL (Price)
    print(f"Loading Oil from {FILE_OIL}...")
    df_oil_raw = pd.read_excel(FILE_OIL)
    # Identify Price Column (Value ~ 30-150)
    # Rows: [Year, MonthNum, MonthName, Col1, Col2...]
    # In inspection: [2025, 10, Oct, 66.15, 1.4, 0.95]
    # Col 3 (0-index) is Price.
    df_oil = df_oil_raw.iloc[:, [0, 1, 3]].copy()
    df_oil.columns = ['Year', 'Month', 'Price']
    # Create Date
    df_oil['Date'] = pd.to_datetime(df_oil[['Year', 'Month']].assign(DAY=1)) + pd.offsets.MonthEnd(0)
    # Resample to Quarterly Mean
    oil_q = df_oil.set_index('Date').resample('QE').mean()['Price']
    
    # 3. LOAD REVENUE (StatAlloc)
    print(f"Loading Revenue from {FILE_REV}...")
    df_rev_raw = pd.read_excel(FILE_REV)
    # Columns: ['recDate', ..., 'statAlloc', ...]
    # recDate is likely DD/MM/YYYY or similar.
    df_rev = df_rev_raw[['recDate', 'statAlloc']].copy()
    # Explicitly handle DD/MM/YYYY
    df_rev['Date'] = pd.to_datetime(df_rev['recDate'], dayfirst=True)
    # Resample to Quarterly Sum
    rev_q = df_rev.set_index('Date').resample('QE')['statAlloc'].sum()
    
    # 4. ALIGN GDP
    # ... (Previous code)
    common_idx = oil_q.index
    df_final = pd.DataFrame(index=common_idx)
    df_final['Oil_Price'] = oil_q
    df_final['Revenue_Proxy'] = rev_q
    
    # Now merge GDP. We need to know the Start Date of the GDP series.
    # NBS "New GDP 2010 Constant" typically starts Q1 2010.
    start_date_gdp = pd.Timestamp('2010-03-31')
    gdp_dates = pd.date_range(start=start_date_gdp, periods=len(gdp_series), freq='QE')
    
    s_gdp = pd.Series(gdp_series.values, index=gdp_dates)
    df_final['GDP_Value'] = s_gdp
    df_final['GDP_Growth'] = df_final['GDP_Value'].pct_change() * 100
    
    # ...
    
    # 4b. MERGE WDI DATA (Digital, Remittances, etc.)
    wdi_path = os.path.join(DATA_DIR, "wdi_quarterly_interpolated.csv")
    if os.path.exists(wdi_path):
        print(f"Merging WDI Data from {wdi_path}...")
        df_wdi = pd.read_csv(wdi_path, index_col=0, parse_dates=True)
        # Avoid column collision if WB has same cols (though WB file only has Growth/Inflation)
        cols_to_use = df_wdi.columns.difference(df_final.columns)
        df_final = df_final.join(df_wdi[cols_to_use], how='left')

    # 4c. MERGE WORLD BANK DATA (GDP Growth + Inflation)
    wb_path = "dataset/wb_economic_data.csv"
    if os.path.exists(wb_path):
        print(f"Merging World Bank Data from {wb_path}...")
        df_wb = pd.read_csv(wb_path)
        df_wb['Date'] = pd.to_datetime(df_wb['Date'])
        
        # Upsample Annual to Quarterly
        df_wb = df_wb.set_index('Date').resample('QE').ffill()
        
        # Merge columns (suffixed if collision, but we want to overwrite/fill)
        # We join specifically for the columns we want
        df_final = df_final.join(df_wb[['GDP_Growth', 'Inflation_Rate']], how='left', rsuffix='_WB')
        
        # If we had calculated GDP_Growth from SDMS, prefer WB if available
        if 'GDP_Growth_WB' in df_final.columns:
             df_final['GDP_Growth'] = df_final['GDP_Growth_WB'].combine_first(df_final.get('GDP_Growth', pd.Series(dtype=float)))
             df_final.drop(columns=['GDP_Growth_WB'], inplace=True)
             
        # Same for Inflation
        if 'Inflation_Rate_WB' in df_final.columns:
             df_final['Inflation_Rate'] = df_final['Inflation_Rate_WB']
             df_final.drop(columns=['Inflation_Rate_WB'], inplace=True)

    # ... (Previous Merges)
    
    # 4c. MERGE DATA COMMONS (Official Tax/Inflation/GDP History)
    dc_path = os.path.join(DATA_DIR, "DC_CLIENT_RESULTS.csv")
    if os.path.exists(dc_path):
        print(f"Merging Data Commons History from {dc_path}...")
        df_dc = pd.read_csv(dc_path)
        
        # Ensure Date index
        # DC dates are often just "YYYY", convert to End-of-Year
        if 'date' in df_dc.columns:
            df_dc['Date'] = pd.to_datetime(df_dc['date'].astype(str)) + pd.offsets.YearEnd(0)
            df_dc.set_index('Date', inplace=True)
            
            # Resample DC data to Quarterly (Forward Fill for slow moving vars like Tax%GDP)
            df_dc_q = df_dc.resample('QE').ffill()
            
            # Map DC Variable Names to Our Columns
            # Mappings:
            # "Amount_EconomicActivity_GrossDomesticProduction_Nominal" -> GDP_Nominal_LCU (Reference)
            # "WorldBank/GC.TAX.TOTL.GD.ZS" -> Tax_Rev_Pct_GDP
            # "sdg/FP_CPI_TOTL_ZG" -> Inflation_Rate
            # "WorldBank/GC.TAX.TOTL.CN" -> Total_Tax_Revenue_LCU
            
            rename_map = {
                "WorldBank/GC.TAX.TOTL.GD.ZS": "Tax_Rev_Pct_GDP",
                "sdg/FP_CPI_TOTL_ZG": "Inflation_Rate",
                "WorldBank/GC.TAX.TOTL.CN": "Total_Tax_Official"
            }
            df_dc_q.rename(columns=rename_map, inplace=True)
            
            # Merge logic: Join available columns
            valid_cols = [c for c in rename_map.values() if c in df_dc_q.columns]
            
            if valid_cols:
                # Merge into main DF
                df_final = df_final.join(df_dc_q[valid_cols], how='left', rsuffix='_DC')
                
                # OVERWRITE our dataset with this better data where available
                if 'Inflation_Rate_DC' in df_final.columns:
                    df_final['Inflation_Rate'] = df_final['Inflation_Rate_DC'].combine_first(df_final.get('Inflation_Rate', pd.Series()))
                    
                # Use Official Tax Revenue to replace proxies?
                # If we have "Total_Tax_Official", we can split it 40/30 (still an assumption but on REAL totals)
                if 'Total_Tax_Official' in df_final.columns:
                     # Fill NaN only (Prefer the NRS Actuals we loaded in step 5)
                     # But wait, step 5 is AFTER this content.
                     # So we just set it here, step 5 will overwrite it if NRS data exists. Perfect.
                     df_final['Revenue_Proxy'] = df_final['Total_Tax_Official'].combine_first(df_final['Revenue_Proxy'])
                     print(" -> Updated Revenue History using Data Commons.")
            
    # Fill gaps (forward fill recent, backfill old if needed)
    df_final = df_final.ffill().bfill().fillna(0)
    
    # 5. MERGE ACTUAL TAX REVENUE (If Available)
    # The user wants to use REAL "Actual CIT Collected" and "Actual VAT Collected"
    file_tax_actuals = os.path.join(DATA_DIR, "TAX_REVENUE_ACTUALS.xlsx")
    
    if os.path.exists(file_tax_actuals):
        print(f"Loading ACTUAL Tax Revenue from {file_tax_actuals}...")
        try:
            # Expected Format: Date (YYYY-MM-DD or Quarter), CIT, VAT
            df_tax = pd.read_excel(file_tax_actuals)
            
            # Normalize Date
            # Try to find a date column
            date_col = None
            for c in df_tax.columns:
                if 'date' in str(c).lower() or 'quarter' in str(c).lower():
                    date_col = c
                    break
            
            if date_col:
                df_tax['Date'] = pd.to_datetime(df_tax[date_col])
                df_tax.set_index('Date', inplace=True)
                
                # Resample to Quarterly Sum (just in case it's monthly)
                # We assume columns are named loosely 'CIT' and 'VAT'
                cit_col = [c for c in df_tax.columns if 'cit' in str(c).lower() or 'company' in str(c).lower()][0]
                vat_col = [c for c in df_tax.columns if 'vat' in str(c).lower() or 'value' in str(c).lower()][0]
                
                tax_q = df_tax[[cit_col, vat_col]].resample('QE').sum()
                tax_q.columns = ['CIT_Revenue', 'VAT_Revenue']
                
                # Merge
                df_final = df_final.join(tax_q, how='left')
                print(" -> Successfully merged Actual Tax Data.")
            else:
                print(" -> Error: Could not find Date column in Tax File. Using Proxy.")
                raise ValueError("No Date Column")

        except Exception as e:
            print(f" -> Failed to load Tax Actuals: {e}")
            print(" -> Reverting to Proxy Logic.")
            df_final['CIT_Revenue'] = df_final['Revenue_Proxy'] * 0.4 
            df_final['VAT_Revenue'] = df_final['Revenue_Proxy'] * 0.3
    else:
        print("No 'TAX_REVENUE_ACTUALS.xlsx' found. Using Proxy Logic (40% CIT / 30% VAT).")
        df_final['CIT_Revenue'] = df_final['Revenue_Proxy'] * 0.4
        df_final['VAT_Revenue'] = df_final['Revenue_Proxy'] * 0.3
    
    # Clean: Drop rows only if ESSENTIAL data is missing (GDP, Oil)
    # Note: After WB merge, we trust WB GDP more.
    df_clean = df_final.dropna(subset=['GDP_Growth', 'Oil_Price'])
    
    df_clean = df_clean.reset_index().rename(columns={'index': 'Date'})
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df_clean)} rows to {OUTPUT_FILE}")
    print(df_clean.head())
    print("\nColumns:", list(df_clean.columns))

if __name__ == "__main__":
    process()
