
import pandas as pd
import os

def analyze_missing_data():
    print("--- Data Gap Analysis ---")
    
    # 1. Load Main Dataset
    main_file = "nigeria_economic_data.csv"
    if not os.path.exists(main_file):
        print(f"CRITICAL: {main_file} is missing.")
        return
        
    df = pd.read_csv(main_file)
    print(f"Loaded {len(df)} rows from {main_file}")
    
    # 2. Check for Missing Values (NaN)
    print("\nMissing Values per Column:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # 3. Check for Zeroes (Hidden missing data)
    print("\nZero Values per Column (Potential Placeholders):")
    zeroes = (df == 0).sum()
    print(zeroes[zeroes > 0])
    
    # 4. Specific Critical Checks
    # Check if Digital Penetration is flat (placeholder)
    if 'Digital_Penetration' in df.columns:
        unique_vals = df['Digital_Penetration'].nunique()
        if unique_vals < 5:
             print("\n[!] WARNING: 'Digital_Penetration' looks extremely repetitive (Placeholder?).")
    
    # Check if Tax Revenues are still proxies
    # We check if CIT/VAT are exactly 40%/30% of Revenue_Proxy (which would imply proxy logic)
    if 'Revenue_Proxy' in df.columns and 'CIT_Revenue' in df.columns:
        is_proxy = (df['CIT_Revenue'] == df['Revenue_Proxy'] * 0.4).all()
        if is_proxy:
             print("\n[!] NOTICE: 'CIT_Revenue' is 100% Calculated Proxy (Algorithm: Rev * 0.4).")
             print("    It is NOT using real data yet.")

if __name__ == "__main__":
    analyze_missing_data()
