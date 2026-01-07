import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
SEED = 42

def generate_and_save_data(output_file='nigeria_economic_data.csv'):
    """
    Generates synthetic historical data (2000-2025) matching the "Master Excel" format.
    Columns: Date, GDP_Growth, Oil_Price, CIT_Revenue, VAT_Revenue
    """
    print(f"Generating synthetic data and saving to {output_file}...")
    dates = pd.date_range(start='2000-01-01', end='2025-12-31', freq='QE')
    n = len(dates)
   
    np.random.seed(SEED)
   
    # 1. Feature: Oil Price (Random Walk)
    oil_price = np.zeros(n)
    oil_price[0] = 30.0
    for i in range(1, n):
        oil_price[i] = oil_price[i-1] + np.random.normal(0, 3)
    oil_price = np.clip(oil_price, 25, 130)

    # 2. Hidden Features: Tax Rates (Used to generate Revenue, but not saved directly per user format)
    # SME Effective Tax Rate (20-30%)
    sme_tax_rate = np.random.uniform(20, 30, n)
    # VAT Efficiency (50-60%)
    vat_efficiency = np.random.uniform(50, 60, n)

    # 3. Generate Proxies for Revenue (Billions Naira)
    # Assume Revenue depends on Oil Price (proxy for economy size) and the Rate
    # CIT Revenue ~ Oil_Price * Tax_Rate * Noise
    cit_revenue = (oil_price * 2) * (sme_tax_rate / 100) * np.random.normal(1, 0.1, n) * 10 
    # VAT Revenue ~ Oil_Price * Efficiency * Noise
    vat_revenue = (oil_price * 1.5) * (vat_efficiency / 100) * np.random.normal(1, 0.05, n) * 10

    # 4. Target: GDP Growth
    gdp_growth = (
        2.5
        + (0.06 * oil_price)
        - (0.2 * sme_tax_rate**1.1)
        + (0.04 * vat_efficiency)
        + np.random.normal(0, 0.5, n)
    )

    df = pd.DataFrame({
        'Date': dates,
        'GDP_Growth': gdp_growth,
        'Oil_Price': oil_price,
        'CIT_Revenue': cit_revenue,
        'VAT_Revenue': vat_revenue
    })
    
    # Set Date as index for CSV saving but often users want Date column in Excel
    df.to_csv(output_file, index=False)
    print(f"Data saved successfully to {output_file}")
    
    print("\nFirst 5 rows (Format for Manual Entry):")
    print(df.head())

if __name__ == "__main__":
    generate_and_save_data()
