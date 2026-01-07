
import pandas as pd
import os

# Define the manual data from the NRS Dashboard
# Units: Billions of Naira
# Period: 2025 Q1, Q2, Q3

data = {
    'Date': ['2025-03-31', '2025-06-30', '2025-09-30'],
    
    # "Company Income Tax (Non-Oil)" from Dashboard
    # Q1=1983.52, Q2=2782.34, Q3=2964.6
    'CIT_Revenue': [1983.52, 2782.34, 2964.60],
    
    # VAT = "NCS-Import VAT" + "Non-Import VAT"
    # Q1: 1556.96 + 507.00 = 2063.96
    # Q2: 1554.70 + 508.55 = 2063.25
    # Q3: 479.79 + 1803.40 = 2283.19
    'VAT_Revenue': [2063.96, 2063.25, 2283.19],
    
    # Total Tax Revenue for Validation (optional)
    'Total_Tax_Revenue': [6106.19, 8174.10, 8309.81]
}

def create_actuals_file():
    print("Creating 'TAX_REVENUE_ACTUALS.xlsx' from NRS 2025 Dashboard data...")
    df = pd.DataFrame(data)
    
    # Save to dataset folder
    output_path = os.path.join("dataset", "TAX_REVENUE_ACTUALS.xlsx")
    df.to_excel(output_path, index=False)
    
    print(f" -> Saved to {output_path}")
    print("\nData Preview:")
    print(df)

if __name__ == "__main__":
    create_actuals_file()
