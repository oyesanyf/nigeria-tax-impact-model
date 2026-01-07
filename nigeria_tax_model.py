import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import os

# ==========================================
# CONFIGURATION
# ==========================================
SEED = 42
FORECAST_HORIZON = 4  # Forecast 4 Quarters (2026)
# Allow override via environment variable (set by tax_model.py --simulations flag)
N_SIMULATIONS = int(os.environ.get('N_SIMULATIONS', 1000))  # Monte Carlo Iterations

# ==========================================
# SECTION 1: DATA INGESTION (PLACEHOLDERS)
# ==========================================
# ==========================================
# SECTION 1: DATA INGESTION
# ==========================================
def load_data(filepath='nigeria_economic_data.csv'):
    """
    Load historical data (2000-2025).
    Expects CSV with columns: Date, GDP_Growth, Oil_Price, CIT_Revenue, VAT_Revenue
    If 'CIT_Revenue' is missing, falls back to legacy columns simulation.
    """
    import os
    if os.path.exists(filepath):
        print(f"Loading Data from {filepath}...")
        try:
            df = pd.read_csv(filepath, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            
            # Check if we have the new raw revenue columns
            if 'CIT_Revenue' in df.columns and 'VAT_Revenue' in df.columns:
                print("Detected Raw Revenue Data. Deriving Policy Rate Proxies...")
                
                # FEATURE ENGINEERING: PROXYING EFFECTIVE RATES
                # We need to extract the "Policy Stance" (Rate) from the "Outcome" (Revenue).
                # Assumption: Revenue = Base * Rate. 
                # We use Oil_Price as a proxy for the Economic Base in this simple model.
                # Rate = Revenue / Oil_Price (scaled to match typical % range for interpretability)
                
                # Calibrate scalers so mean matches legacy expectations (SME_Tax~25, VAT~55)
                # This ensures the Simulation logic (setting Tax=0 or Tax=25) still makes sense relative to history.
                
                # Raw Proxies
                raw_cit_rate = df['CIT_Revenue'] / df['Oil_Price']
                raw_vat_rate = df['VAT_Revenue'] / df['Oil_Price']
                
                # Scaling factor (e.g. if mean raw is 0.5 but we want 25.0)
                cit_scaler = 25.0 / raw_cit_rate.mean()
                vat_scaler = 55.0 / raw_vat_rate.mean()
                
                df['SME_Tax'] = raw_cit_rate * cit_scaler
                df['VAT_Recovery'] = raw_vat_rate * vat_scaler
                
                print(f"--> Derived SME_Tax (Proxy) Mean: {df['SME_Tax'].mean():.2f}%")
                print(f"--> Derived VAT_Recovery (Proxy) Mean: {df['VAT_Recovery'].mean():.2f}%")
                
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            print("Falling back to synthetic generation...")
    
    print("Generating Synthetic History for Demo...")
    # ... (Legacy generation code would go here or strictly use the file)
    # Since we ensured the file exists via generate_dataset.py, this path is unlikely.
    # But for safety, let's Raise or Return empty to fail fast if file missing
    raise FileNotFoundError("Data file not found. Run generate_dataset.py first.")

# ==========================================
# SECTION 2: AUTO TIME SERIES (Forecasting 2026)
# ==========================================
def forecast_exogenous_vars(df):
    """
    Uses AutoARIMA to predict what Oil Prices would be in 2026
    WITHOUT any intervention.
    """
    print("\n[AutoTS] Running AutoARIMA to forecast Oil Prices for 2026...")
   
    # Auto-discover the best (p,d,q) model for Oil Prices
    # This replaces manual stationarity tests (ADF)
    oil_model = auto_arima(df['Oil_Price'], seasonal=False, trace=True)
   
    # Forecast 4 quarters ahead
    oil_forecast_2026 = oil_model.predict(n_periods=FORECAST_HORIZON)
   
    print(f"--> AutoARIMA Selected Model: {oil_model.order}")
    print(f"--> 2026 Oil Forecast (Avg): ${oil_forecast_2026.mean():.2f}")
   
    return oil_forecast_2026

# ==========================================
# SECTION 3: AUTOML (Learning the Economy)
# ==========================================
def train_automl_impact_model(df):
    """
    Uses TPOT (Genetic Algorithms) to find the BEST machine learning model
    that explains how Tax and Oil impact GDP.
    Includes advanced WDI features (Digitalization, Remittances).
    """
    import joblib
    model_path = os.path.join('models', 'tpot_tax_model.pkl')
    
    if os.path.exists(model_path):
        print(f"\n[AI Engine] Pre-trained model found at {model_path}. Loading...")
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            print(f"⚠️ Error loading model: {e}. Falling back to fresh search...")

    print("\n[AutoML] No pre-trained model found. Searching for best model using TPOT...")
   
    # Expanded Feature Set
    feature_cols = ['Oil_Price', 'SME_Tax', 'VAT_Recovery', 'Digital_Penetration', 'Remittances_USD', 'Inflation_Rate']
    
    # Ensure columns exist (fill if missing in WDI merge)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
            
    X = df[feature_cols]
    y = df['GDP_Growth']
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
   
    # Run AutoML
    tpot = TPOTRegressor(generations=2, population_size=20, verbose=2, random_state=SEED)
    tpot.fit(X_train, y_train)
   
    print(f"--> Best Model Found: {tpot.fitted_pipeline_}")
    
    # Save the model to the models folder
    os.makedirs('models', exist_ok=True)
    joblib.dump(tpot.fitted_pipeline_, model_path)
    print(f"--> Model saved to {model_path}")
    
    from sklearn.metrics import r2_score
    y_pred = tpot.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"--> Model Accuracy (R2): {score:.4f}")
   
    return tpot.fitted_pipeline_

# ==========================================
# HELPER: LOAD CONFIG
# ==========================================
def load_config(config_path="scenarios_config.json"):
    import json
    if os.path.exists(config_path):    
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print(f"Warning: {config_path} not found. Using defaults.")
        config = {
            "scenarios": [
                {"name": "Old Law", "params": {"SME_Tax": 25.0, "VAT_Recovery": 55.0, "Digital_Multiplier": 1.0, "Inflation_Multiplier": 1.0}},
                {"name": "New Tax Act 2025", "params": {"SME_Tax": 0.0, "VAT_Recovery": 95.0, "Digital_Multiplier": 1.05, "Inflation_Multiplier": 1.0}},
                {"name": "Inflation Shock", "params": {"SME_Tax": 0.0, "VAT_Recovery": 95.0, "Digital_Multiplier": 1.05, "Inflation_Multiplier": 1.5}}
            ]
        }
        
    # APPLY OVERRIDES FROM ENV (CLI FLAGS)
    # We target the specific scenarios by name or role.
    
    # 1. New Law Override
    if "OVERRIDE_SME_TAX" in os.environ or "OVERRIDE_VAT_RECOVERY" in os.environ:
        for scen in config["scenarios"]:
            if "New" in scen["name"] or "Shock" in scen["name"]:
                if "OVERRIDE_SME_TAX" in os.environ:
                    scen["params"]["SME_Tax"] = float(os.environ["OVERRIDE_SME_TAX"])
                if "OVERRIDE_VAT_RECOVERY" in os.environ:
                    scen["params"]["VAT_Recovery"] = float(os.environ["OVERRIDE_VAT_RECOVERY"])
    
    # 2. Shock Override
    if "OVERRIDE_INFLATION_SHOCK" in os.environ:
        for scen in config["scenarios"]:
            if "Shock" in scen["name"]:
                scen["params"]["Inflation_Multiplier"] = float(os.environ["OVERRIDE_INFLATION_SHOCK"])
                
    return config

# ==========================================
# SECTION 4: STOCHASTIC SIMULATION
# ==========================================
def simulate_impact(df, oil_forecast, tpot_model):
    from tqdm import tqdm
    
    # Load Scenario Configuration
    config = load_config()
    scenarios = config.get("scenarios", [])
    
    print(f"\n[Simulation] Running {N_SIMULATIONS} iterations for {len(scenarios)} scenarios...")
    
    results = []
    
    # Base Oil Prediction from AutoARIMA
    base_oil = oil_forecast.mean()
    oil_std = config.get("simulation", {}).get("oil_price_uncertainty_std", 10.0)
    
    # Get latest WDI/Inflation values (Baseline)
    last_digi = df['Digital_Penetration'].iloc[-1] if 'Digital_Penetration' in df else 0
    last_remit = df['Remittances_USD'].iloc[-1] if 'Remittances_USD' in df else 0
    last_inf = df['Inflation_Rate'].iloc[-1] if 'Inflation_Rate' in df else 15.0
    
    # Run Monte Carlo Loop
    for _ in tqdm(range(N_SIMULATIONS), desc="Simulating outcomes", unit="run"):
        # 1. Stochastic Oil Price (Common Factor)
        sim_oil = np.random.normal(base_oil, oil_std)
        
        # Dictionary to store this run's results
        run_result = {}
        
        # 2. Iterate through configurable scenarios
        for scen in scenarios:
            name = scen["name"]
            params = scen["params"]
            
            # Construct Input Vector
            # Apply multipliers to baseline values
            sim_digi = last_digi * params.get("Digital_Multiplier", 1.0)
            sim_inf = last_inf * params.get("Inflation_Multiplier", 1.0)
            
            # Create DataFrame for prediction
            input_df = pd.DataFrame([{
                "Oil_Price": sim_oil,
                "SME_Tax": params.get("SME_Tax", 25.0),
                "VAT_Recovery": params.get("VAT_Recovery", 55.0),
                "Digital_Penetration": sim_digi,
                "Remittances_USD": last_remit,
                "Inflation_Rate": sim_inf
            }])
            
            # Predict
            pred_gdp = tpot_model.predict(input_df)[0]
            
            # Store with a key that enhanced_report expects
            # enhanced_report.py currently expects specific keys: 'GDP_Old', 'GDP_New', 'GDP_Shock'
            # We need to map dynamic names to these static keys OR update enhanced_report.py
            # For now, let's map based on index to preserve compatibility without rewritten reporting logic yet.
            if "Old" in name: key = "GDP_Old"
            elif "New" in name: key = "GDP_New"
            elif "Shock" in name: key = "GDP_Shock"
            else: key = f"GDP_{name.replace(' ', '_')}" # Fallback for new custom scenarios
            
            run_result[key] = pred_gdp
            
        results.append(run_result)
    
    return pd.DataFrame(results)

# ==========================================
# EXECUTION
# ==========================================
def main():
    import os
    
    # Setup Reports Directory
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    print(f"Reports will be saved to: {os.path.abspath(reports_dir)}")

    # 1. Data
    # Ensure dataset exists or prompt user
    try:
        df = load_data()
       
        # 2. Auto Time Series (Predict the Environment)
        oil_forecast = forecast_exogenous_vars(df)
       
        # 3. AutoML (Learn the Logic)
        model = train_automl_impact_model(df)
       
        # 4. Simulation (Test the Policy)
        sim_results = simulate_impact(df, oil_forecast, model)
       
        # 5. Generate Enhanced Report
        import enhanced_report
        
        # Calculate R2 for the report
        from sklearn.metrics import r2_score
        X_test_cols = ['Oil_Price', 'SME_Tax', 'VAT_Recovery', 'Digital_Penetration', 'Remittances_USD', 'Inflation_Rate']
        X_test_data = df[X_test_cols].iloc[-int(len(df)*0.2):]
        y_test_data = df['GDP_Growth'].iloc[-int(len(df)*0.2):]
        y_pred = model.predict(X_test_data)
        model_r2 = r2_score(y_test_data, y_pred)
        
        report_paths = enhanced_report.generate_enhanced_report(
            sim_results=sim_results,
            df=df,
            oil_forecast=oil_forecast,
            model=model,
            model_r2=model_r2,
            n_simulations=N_SIMULATIONS
        )
        
        # Print summary to console
        risk_old = (sim_results['GDP_Old'] < 0).mean() * 100
        risk_new = (sim_results['GDP_New'] < 0).mean() * 100
        risk_shock = (sim_results['GDP_Shock'] < 0).mean() * 100
        
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Recession Risk (Old Law):        {risk_old:.1f}%")
        print(f"Recession Risk (New Law):        {risk_new:.1f}%")
        print(f"Recession Risk (Inflation Shock): {risk_shock:.1f}%")
        print(f"{'='*60}")
        print(f"\n📄 GENERATED REPORTS:")
        if "html" in report_paths: print(f"   --> HTML: {report_paths['html']}")
        if "word" in report_paths: print(f"   --> WORD: {report_paths['word']}")
        if "pdf" in report_paths:  print(f"   --> PDF:  {report_paths['pdf']}")
        print(f"{'='*60}")
        
    except FileNotFoundError:
        print("Please run 'python generate_dataset.py' first to create the initial dataset,")
        print("OR strictly follow the instructions to create 'nigeria_economic_data.csv' from NBS/CBN data.")

if __name__ == "__main__":
    main()
