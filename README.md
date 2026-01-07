# 🇳🇬 Nigeria Tax Policy Impact Assessment System

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange?style=for-the-badge&logo=openai)](https://openai.com/)

> **An advanced macroeconomic engine leveraging AutoML, Monte Carlo simulations, and Time-Series Forecasting to evaluate the impact of Nigeria's fiscal reforms.**

---

## 🏛️ Project Vision
This system is designed to provide policymakers, researchers, and financial analysts with a data-driven toolkit to assess the 2025 Tax Act. It specifically models the elimination of **SME Company Income Tax (CIT)** and the enhancement of **VAT collection efficiency** through digitalization.

---

## 🚀 Key Features

### 🧠 Intelligent Modeling
- **AutoML (TPOT):** Automatically discovers the optimal machine learning pipeline (StandardScaler, RFE, Regressors) using Genetic Programming.
- **Time-Series Aware:** Training respects chronological order using `TimeSeriesSplit` to prevent "look-ahead" bias.
- **AutoARIMA:** Precisely forecasts external factors like Crude Oil prices for the 2026 horizon.

### 🎲 Risk Analysis
- **Monte Carlo Simulations:** Runs 10,000+ stochastic iterations to quantify the probability of economic outcomes.
- **Dynamic Scenarios:** Configure policy parameters (Tax rates, VAT recovery, Inflation shocks) via `scenarios_config.json` or CLI flags.
- **Recession Risk Metrics:** Direct calculation of the probability that GDP growth falls below 0%.

### 📄 Professional Reporting
- **Multi-Format Output:** Automatically generates **Interactive HTML**, **Formal PDF**, and **Editable Word (.docx)** reports.
- **Rich Visualizations:** 6+ high-resolution charts including KDE distributions, Box Plots, and Feature Correlation Heatmaps.
- **AI Executive Summary:** OpenAI-powered analysis providing deep context, methodology, and policy implications.

---

## 🛠️ Quick Start

### 1. Installation
```bash
git clone https://github.com/oyesanyf/nigeria-tax-impact-model.git
cd nigeria-tax-impact-model
pip install -r requirements.txt
```

### 2. Execution
Run the full pipeline with a single command:
```bash
python tax_model.py --simulations 10000
```

### 3. Dynamic Policy Overrides (CLI)
Test ad-hoc policy changes without editing code:
```bash
python tax_model.py --sme-tax 5.0 --vat-recovery 85.0 --inflation-shock 1.2
```

---

## 📦 Project Architecture

```
nigeria-tax-impact-model/
├── tax_model.py            # Main Orchestrator (Full Pipeline)
├── nigeria_tax_model.py    # Core AI Engine & Simulation Logic
├── enhanced_report.py      # Multi-format Report Engine (HTML, PDF, Docx)
├── scenarios_config.json   # Centralized Scenario Definitions
├── models/                 # Cached Pre-trained AI Models (.pkl)
├── reports/                # Generated Timestamped Reports
├── dataset/                # Raw NBS, CBN, and World Bank files
└── docs/                   # Detailed methodology documentation
```

---

## 🧮 Methodology

### Data Handling
The system ingests data from **NBS**, **CBN**, **World Bank**, and **Google Data Commons**. It employs frequency alignment (annual → quarterly) and forward-filling to handle economic data gaps typical in emerging markets.

### The AI Pipeline
1. **Fetch:** Automated ingestion from cloud APIs and local archives.
2. **Process:** Feature engineering of policy proxies (Effective tax rates).
3. **Train:** TPOT search for the best regression pipeline.
4. **Forecast:** ARIMA projection of 2026 environment.
5. **Simulate:** Thousands of "what-if" scenarios.
6. **Report:** Generation of stakeholder-ready documentation.

---

## 📈 Improving Accuracy
The model currently achieves an **R² of ~0.85**. Accuracy can be further enhanced by providing:
- **FIRS Quarterly Returns:** Sector-specific effective tax rates.
- **NIBSS Payments Data:** Real-time digital economy signals.
- **State-Level Revenue:** For regional disaggregation.

---

## 🤝 Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get started.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
