import pandas as pd
FILEPATH = "dataset/Crude_Oil_Data_in_Excel.xlsx"
try:
    df = pd.read_excel(FILEPATH)
    print("Oil Data Head:")
    print(df.head())
    print("\nOil Stats:")
    # Assuming 'Price' or similar column
    print(df.describe())
except Exception as e:
    print(e)
