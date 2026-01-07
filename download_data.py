import os
import requests

DOWNLOAD_LIST = [
    {
        "url": "https://microdata.nigerianstat.gov.ng/index.php/catalog/147/download/1073",
        "name": "GDP_Expenditure_Income_Approach_Q1-Q2_2024.xlsx"
    },
    {
        "url": "https://microdata.nigerianstat.gov.ng/index.php/catalog/147/download/1158",
        "name": "Q4_GDP_2024.xlsx"
    },
    {
        "url": "https://microdata.nigerianstat.gov.ng/index.php/catalog/147/download/1257",
        "name": "2019-2024_rebased_figures.xlsx"
    },
    {
        "url": "https://microdata.nigerianstat.gov.ng/index.php/catalog/147/download/1259",
        "name": "Q1_2025_GDP_Estimates.xlsx"
    }
]

DATASET_DIR = "dataset"

def download_files():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"Created directory: {DATASET_DIR}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print("Starting downloads...")
    for item in DOWNLOAD_LIST:
        url = item['url']
        filename = item['name']
        filepath = os.path.join(DATASET_DIR, filename)
        
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f" -> Saved to {filepath}")
        except Exception as e:
            print(f" -> Failed to download {filename}: {e}")

if __name__ == "__main__":
    download_files()
