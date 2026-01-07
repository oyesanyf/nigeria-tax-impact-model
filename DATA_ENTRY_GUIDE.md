GUIDE TO POPULATING 'nigeria_economic_data.csv'

I cannot automatically download the full 25-year dataset because NBS and CBN do not provide a single machine-readable historical file. You must aggregate it from quarterly reports.

Below is a starting point with RECENT ACTUAL DATA found during my search. 
Copy these values into your CSV to replace the placeholders for these dates.

=======================================================
RECENT KNOWN VALUES (Billions Naira / USD)
=======================================================

DATE        | CIT_Revenue | VAT_Revenue | Oil_Price ($) | GDP_Growth (%)
------------|-------------|-------------|---------------|----------------
2025-06-30  | 2500.0*     | 2060.0      | 70.20         | [Check NBS]
2025-03-31  | 984.6       | 2060.0      | [Check CBN]   | [Check NBS]
2024-12-31  | 1130.0      | 1560.0      | [Check CBN]   | [Check NBS]
2024-09-30  | 1750.0      | 1430.0      | [Check CBN]   | [Check NBS]
2024-06-30  | 2470.0      | 1560.0      | [Check CBN]   | [Check NBS]
2024-03-31  | 984.0       | 1430.0      | [Check CBN]   | [Check NBS]
...
2020-04-01  | ...         | ...         | 7.15 (Low)    | ...

*Note: 2025 CIT Q2 estimate based on trend reports.

=======================================================
DATA SOURCES RECAP
=======================================================
1. GDP Growth:
   - Source: NBS eLibrary (Gross Domestic Product Reports)
   - Look for: Excel tables attached to "GDP Report Qx 20xx"

2. Oil Prices (Bonny Light):
   - Source: Central Bank of Nigeria (CBN) or EIA
   - Action: Average the 3 monthly prices to get the Quarterly price.

3. Tax Revenue (CIT & VAT):
   - Source: NBS Sectoral Distribution Reports
   - Note: Some quarters have huge spikes (e.g., June/July is often high for CIT filing), so don't be alarmed by volatility.

HOW TO UPDATE THE MODEL:
1. Open 'nigeria_economic_data.csv' in Excel.
2. Paste your real collected data into the columns.
3. Save as CSV.
4. Run 'python nigeria_tax_model.py'.
