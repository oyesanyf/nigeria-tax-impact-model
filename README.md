# Nigeria Tax Policy Impact Assessment Model

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated econometric modeling system that uses **AutoML**, **Monte Carlo simulation**, and **time series forecasting** to assess the macroeconomic impact of Nigeria's proposed Tax Act 2025 on GDP growth and recession risk.

---

## 🎯 Overview

This project evaluates the potential impact of eliminating Company Income Tax (CIT) for SMEs and enhancing VAT collection efficiency through digitalization. The model:

- **Trains** on 20+ years of historical Nigerian economic data (2000-2025)
- **Forecasts** 2026 oil prices using AutoARIMA
- **Simulates** 1,000+ economic scenarios using Monte Carlo methods
- **Generates** comprehensive HTML reports with AI-powered executive summaries

### Key Features

✅ **Automated Data Pipeline** - Fetches data from Google Data Commons, World Bank, NBS, and CBN  
✅ **AutoML Model Selection** - Uses TPOT (Genetic Programming) to find optimal regression models  
✅ **Stochastic Risk Assessment** - Monte Carlo simulation with configurable iterations  
✅ **Professional Reporting** - Timestamped HTML reports with 6+ embedded visualizations  
✅ **AI-Powered Summaries** - OpenAI GPT-4 generates detailed executive summaries  
✅ **Command-Line Interface** - Flexible flags for different execution modes  

---

## 📊 Sample Output

**Recession Risk Comparison:**
- Old Law: 12.3%
- New Tax Act 2025: 8.7% (↓ 3.6pp)
- Inflation Shock Scenario: 24.1%

**Model Accuracy:** R² = 0.86 (86% variance explained)

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+** (tested on 3.13)
- **Windows/Linux/macOS**
- **API Keys** (optional but recommended):
  - Google Data Commons API Key
  - OpenAI API Key (for AI summaries)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/nigeria-tax-model.git
   cd nigeria-tax-model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set API keys** (optional):
   ```bash
   # Windows PowerShell
   $env:DC_API_KEY="your_google_datacommons_key"
   $env:OPENAI_API_KEY="sk-your_openai_key"
   
   # Linux/macOS
   export DC_API_KEY="your_google_datacommons_key"
   export OPENAI_API_KEY="sk-your_openai_key"
   ```

4. **Run the model:**
   ```bash
   python tax_model.py
   ```

5. **View the report:**
   - Open `reports/tax_impact_report_[timestamp].html` in your browser

---

## 📦 Project Structure

```
nigeria-tax-model/
├── tax_model.py                  # Main orchestrator (run this)
├── nigeria_tax_model.py          # Core modeling logic
├── enhanced_report.py            # Report generation with charts
├── process_dataset.py            # Data processing pipeline
├── create_nrs_data.py            # NRS Dashboard data generator
├── fetch_dc_client_v2.py         # Google Data Commons fetcher
├── train_tax_model.py            # Standalone training script
│
├── dataset/                      # Raw data files
│   ├── SDMS_Bulk_Data_Download_*.xlsx
│   ├── Crude_Oil_Data_in_Excel.xlsx
│   ├── Financial_Data_in_Excel.xlsx
│   ├── TAX_REVENUE_ACTUALS.xlsx
│   ├── DC_CLIENT_RESULTS.csv
│   └── wb_economic_data.csv
│
├── reports/                      # Generated HTML reports
│   └── tax_impact_report_*.html
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── DATA_SOURCES_URLS.md          # Data source documentation
```

---

## 🔧 Usage

### Basic Usage

```bash
python tax_model.py
```

This runs the full pipeline:
1. Fetches latest data from Google Data Commons
2. Generates NRS 2025 actuals
3. Processes and merges all datasets
4. Trains the AutoML model
5. Runs Monte Carlo simulation
6. Generates timestamped HTML report

### Command-Line Flags

```bash
# Show help
python tax_model.py --help

# Skip data fetching (use cached data)
python tax_model.py --skip-fetch

# Skip processing, just regenerate report
python tax_model.py --reports-only

# Custom number of simulations (default: 1000)
python tax_model.py --simulations 5000

# Combine flags
python tax_model.py --skip-fetch --simulations 2000
```

### Standalone Training

To train the model without running the full pipeline:

```bash
python train_tax_model.py
```

This saves the trained model to `tpot_tax_model.pkl`.

---

## 📚 Data Sources

### Primary Sources

1. **National Bureau of Statistics (NBS)**
   - GDP Growth (Quarterly)
   - Real GDP values
   - Source: [NBS eLibrary](https://nigerianstat.gov.ng/elibrary)

2. **Central Bank of Nigeria (CBN)**
   - Crude Oil Prices (Bonny Light)
   - Financial Statistics
   - Source: [CBN Statistical Bulletin](https://www.cbn.gov.ng/documents/statbulletin.asp)

3. **Nigeria Revenue Service (NRS)**
   - 2025 Tax Revenue Actuals (Q1-Q3)
   - CIT and VAT Collections
   - Source: [NRS Dashboard](https://nrs.gov.ng/transparency/revenue-dashboard)

4. **Google Data Commons**
   - Historical Tax Revenue (% GDP)
   - Inflation (CPI)
   - World Bank Indicators
   - Source: [Data Commons API](https://datacommons.org)

### Data Coverage

- **Time Period:** 2000-2025 (quarterly)
- **Observations:** 80+ data points
- **Variables:** 12+ economic indicators

---

## 🧮 Methodology

### 1. Data Processing Pipeline

```
Raw Excel Files (NBS/CBN) 
    ↓
process_dataset.py
    ↓
Merge with Google Data Commons
    ↓
Merge with NRS 2025 Actuals
    ↓
Feature Engineering (Tax Rate Proxies)
    ↓
nigeria_economic_data.csv
```

**Feature Engineering:**
- Derives effective tax rates from revenue/oil price ratios
- Scales to match historical policy ranges (SME Tax ~25%, VAT ~55%)
- Interpolates quarterly data from annual World Bank indicators

### 2. Forecasting (AutoARIMA)

**Objective:** Predict 2026 oil prices without policy intervention

**Method:**
- Automatic (p,d,q) parameter selection
- AIC-based model comparison
- 4-quarter ahead forecast

**Output:** Mean oil price + uncertainty bands

### 3. AutoML Model Training (TPOT)

**Objective:** Learn the relationship between economic variables and GDP growth

**Algorithm:** Tree-based Pipeline Optimization Tool (TPOT)
- Genetic programming to evolve ML pipelines
- Searches across regression algorithms (Linear, SVM, Random Forest, XGBoost, etc.)
- Automatic hyperparameter tuning

**Features:**
- Oil Price
- SME Tax Rate (proxy)
- VAT Recovery Rate (proxy)
- Digital Penetration (% internet users)
- Remittances (USD)
- Inflation Rate (CPI)

**Target:** GDP Growth (%)

**Validation:** 80/20 train-test split, R² score

### 4. Monte Carlo Simulation

**Objective:** Assess recession risk under policy scenarios

**Process:**
1. For each iteration (default: 1,000):
   - Sample random oil price from N(forecast_mean, $10)
   - Predict GDP under **Old Law** (Tax=25%, VAT=55%)
   - Predict GDP under **New Law** (Tax=0%, VAT=95%, Digital+5%)
   - Predict GDP under **Shock** (New Law + Inflation×1.5)
2. Calculate recession probability (GDP < 0%)

**Output:** Distribution of GDP growth for each scenario

### 5. Report Generation

**Charts (Seaborn/Matplotlib):**
1. GDP Growth Distribution (KDE)
2. Recession Risk Comparison (Bar)
3. GDP Growth Range (Box Plot)
4. Historical GDP Trend (Time Series)
5. Oil Price Forecast (Time Series)
6. Feature Correlation Heatmap

**Executive Summary (OpenAI GPT-4):**
- Context and objectives
- Data sources and methodology
- Key findings and policy implications
- Limitations and recommendations

---

## 🔬 Technical Details

### Dependencies

**Core Libraries:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - ML utilities
- `tpot` - AutoML framework
- `pmdarima` - AutoARIMA implementation

**Visualization:**
- `matplotlib` - Base plotting
- `seaborn` - Statistical visualizations

**Data Fetching:**
- `datacommons-client` - Google Data Commons API
- `openai` - GPT-4 API (optional)

**Full list:** See `requirements.txt`

### Model Architecture

The TPOT framework automatically selects the best pipeline. Example output:

```python
Pipeline(
    steps=[
        ('standardscaler', StandardScaler()),
        ('xgbregressor', XGBRegressor(
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100
        ))
    ]
)
```

### Performance

**Training Time:** ~6 minutes (4 generations, population=30)  
**Simulation Time:** ~30 seconds (1,000 iterations)  
**Report Generation:** ~10 seconds  
**Total Runtime:** ~7 minutes (full pipeline)

---

## 📖 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DC_API_KEY` | Optional | Google Data Commons API key for fetching historical data |
| `OPENAI_API_KEY` | Optional | OpenAI API key for AI-generated summaries |
| `N_SIMULATIONS` | No | Override simulation count (set by `--simulations` flag) |

### Model Parameters

Edit `nigeria_tax_model.py`:

```python
SEED = 42                  # Random seed for reproducibility
FORECAST_HORIZON = 4       # Quarters to forecast (2026)
N_SIMULATIONS = 1000       # Monte Carlo iterations
```

Edit `train_tax_model.py`:

```python
GENERATIONS = 4            # TPOT generations (higher = better model)
POP_SIZE = 30              # TPOT population size
```

---

## 🧪 Testing

### Verify Installation

```bash
python -c "import tpot, pmdarima, seaborn; print('All dependencies installed!')"
```

### Test Data Processing

```bash
python process_dataset.py
# Should create: nigeria_economic_data.csv
```

### Test Model Training

```bash
python train_tax_model.py
# Should create: tpot_tax_model.pkl
# Expected R²: 0.80-0.90
```

### Quick Simulation Test

```bash
python tax_model.py --skip-fetch --simulations 100
# Fast test run with cached data
```

---

## 📊 Interpreting Results

### Recession Risk

**Definition:** Probability that GDP growth falls below 0% (negative growth)

**Thresholds:**
- **Low Risk:** < 5%
- **Moderate Risk:** 5-15%
- **High Risk:** > 15%

### Model Accuracy (R²)

**Interpretation:**
- **R² = 0.86** → Model explains 86% of GDP variance
- **Typical Range:** 0.75-0.90 for macroeconomic models
- **Below 0.70:** Consider adding more features or data

### Policy Impact

**Positive Impact:** New Law recession risk < Old Law risk  
**Negative Impact:** New Law recession risk > Old Law risk  
**Magnitude:** Difference in percentage points (pp)

---

## 🛠️ Troubleshooting

### Common Issues

**1. `ModuleNotFoundError: No module named 'tpot'`**
```bash
pip install tpot
```

**2. `TypeError: ... unexpected keyword argument 'verbosity'`**
- Fixed in latest version
- Update: `pip install --upgrade tpot`

**3. `DC_API_KEY not set` warning**
- Optional: Set API key to fetch latest data
- Alternative: Use cached data with `--skip-fetch`

**4. OpenAI summary fails**
- Optional: Set `OPENAI_API_KEY` for AI summaries
- Fallback: Uses detailed template summary

**5. Memory error during simulation**
- Reduce simulations: `--simulations 500`
- Close other applications

### Data Issues

**Missing `nigeria_economic_data.csv`:**
```bash
python process_dataset.py
```

**Outdated data:**
```bash
# Re-fetch from Google Data Commons
python fetch_dc_client_v2.py
python tax_model.py
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/your-feature`
3. **Commit changes:** `git commit -m 'Add your feature'`
4. **Push to branch:** `git push origin feature/your-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black *.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **National Bureau of Statistics (NBS)** - GDP and economic data
- **Central Bank of Nigeria (CBN)** - Oil price and financial statistics
- **Nigeria Revenue Service (NRS)** - 2025 tax revenue actuals
- **Google Data Commons** - Historical economic indicators
- **World Bank** - Development indicators
- **TPOT Team** - AutoML framework
- **pmdarima Team** - AutoARIMA implementation

---

## 📞 Contact

**Project Maintainer:** [Your Name]  
**Email:** your.email@example.com  
**GitHub:** [@yourusername](https://github.com/yourusername)

---

## 📈 Roadmap

### Planned Features

- [ ] **Real-time Dashboard** - Interactive web interface with Streamlit
- [ ] **Scenario Builder** - Custom policy parameter inputs
- [ ] **Quarterly Updates** - Automated data refresh pipeline
- [ ] **Regional Analysis** - State-level impact assessment
- [ ] **Sensitivity Analysis** - Parameter importance ranking
- [ ] **API Endpoint** - REST API for programmatic access
- [ ] **Docker Support** - Containerized deployment

### Version History

**v1.0.0** (2026-01-07)
- Initial release
- AutoML model training
- Monte Carlo simulation
- Enhanced HTML reports
- AI-generated summaries

---

## 🔗 Related Projects

- [TPOT](https://github.com/EpistasisLab/tpot) - AutoML framework
- [pmdarima](https://github.com/alkaline-ml/pmdarima) - AutoARIMA for Python
- [Data Commons](https://datacommons.org) - Google's knowledge graph

---

## 📚 References

1. Olson, R. S., et al. (2016). "TPOT: A Tree-based Pipeline Optimization Tool for Automating Machine Learning." *AutoML Workshop at ICML*.
2. Hyndman, R. J., & Khandakar, Y. (2008). "Automatic time series forecasting: the forecast package for R." *Journal of Statistical Software*.
3. World Bank. (2025). *World Development Indicators*. Retrieved from https://databank.worldbank.org
4. National Bureau of Statistics. (2025). *Nigerian Gross Domestic Product Report*. Abuja, Nigeria.

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
