"""
Nigeria Tax Model - Main Orchestrator
-------------------------------------
This script runs the full end-to-end pipeline:
1. Data Processing (dataset/ -> CSV)
2. AI Environment Setup
3. Model Training & Simulation
4. Report Generation
"""
import sys
import time
import os

def print_status(step, message):
    print(f"\n{'='*60}")
    print(f"STEP {step}: {message}")
    print(f"{'='*60}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Nigeria Tax Model System - Full Orchestrator",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--skip-fetch', action='store_true', 
                        help="Skip attempting to fetch data from Google/NRS APIs.\nUse this if you are offline or want to use existing local data.")
                        
    parser.add_argument('--reports-only', action='store_true',
                        help="Skip data processing and training, just run the existing model/simulation.\nUseful for re-running the view with different plot settings.")
    
    parser.add_argument('--simulations', type=int, default=1000,
                        help="Number of Monte Carlo simulations to run (default: 1000).\nHigher values increase accuracy but take longer.")

    # Policy Override Flags
    parser.add_argument('--sme-tax', type=float, help="Override SME Tax Rate for New Law scenario (default: 0.0)")
    parser.add_argument('--vat-recovery', type=float, help="Override VAT Recovery Rate for New Law scenario (default: 95.0)")
    parser.add_argument('--inflation-shock', type=float, help="Override Inflation Shock Multiplier (default: 1.5)")

    args = parser.parse_args()

    # -----------------------------------------------
    # DYNAMIC CONFIG OVERRIDE
    # -----------------------------------------------
    # If policy flags are set, we need to inject them into the process.
    # We will do this by setting environment variables that nigeria_tax_model.py checks,
    # OR by temporarily updating the JSON config in memory. 
    # Since the model reads from JSON, let's pass these via Environment Variables as overrides.
    
    if args.sme_tax is not None:
        os.environ['OVERRIDE_SME_TAX'] = str(args.sme_tax)
        print(f"🔹 Override: Setting New Law SME Tax to {args.sme_tax}%")
        
    if args.vat_recovery is not None:
        os.environ['OVERRIDE_VAT_RECOVERY'] = str(args.vat_recovery)
        print(f"🔹 Override: Setting New Law VAT Recovery to {args.vat_recovery}%")
        
    if args.inflation_shock is not None:
        os.environ['OVERRIDE_INFLATION_SHOCK'] = str(args.inflation_shock)
        print(f"🔹 Override: Setting Inflation Shock Multiplier to {args.inflation_shock}x")

    print("Starting Nigeria Tax Model System (End-to-End)...")
    
    # -----------------------------------------------
    # STEP 0: Fetch External Data (Google Data Commons)
    # -----------------------------------------------
    if not args.skip_fetch and not args.reports_only:
        print_status(1, "FETCHING DATA FROM GOOGLE DATA COMMONS")
        if os.environ.get("DC_API_KEY"):
            try:
                import fetch_dc_client_v2
                print("Connecting to Data Commons API...")
                fetch_dc_client_v2.fetch_data_client_version()
                print("✔ Remote data fetch complete.")
            except ImportError:
                print("⚠️  Could not import 'fetch_dc_client_v2'. Skipping fetch.")
            except Exception as e:
                print(f"❌ Error fetching data: {e}")
                print("   Continuing with existing local files...")
        else:
            print("⚠️  DC_API_KEY not set. Skipping Google Data Fetch.")
            print("   Using cached data in 'dataset/' if available.")
    else:
        print("Skipping Step 1 (Fetch) as requested.")

    # -----------------------------------------------
    # STEP 1: Generate Manual Data (NRS Dashboard)
    # -----------------------------------------------
    if not args.skip_fetch and not args.reports_only:
        print_status(2, "GENERATING NRS DASHBOARD ACTUALS")
        try:
            import create_nrs_data
            create_nrs_data.create_actuals_file()
            print("✔ NRS 2025 Actuals created.")
        except Exception as e:
            print(f"❌ Error creating NRS data: {e}")

    # -----------------------------------------------
    # STEP 2: Process & Merge Datasets
    # -----------------------------------------------
    if not args.reports_only:
        print_status(3, "MERGING DATASETS (NBS + CBN + Google + NRS)")
        try:
            import process_dataset
            import importlib
            importlib.reload(process_dataset) # Request reload in case file changed
            print("Processing raw files into Master CSV...")
            process_dataset.process()
            print("✔ Data processing complete.")
        except Exception as e:
            print(f"❌ Error processing data: {e}")
            print("⚠️  Attempting to proceed with existing CSV if available...")

    # -----------------------------------------------
    # STEP 3: Initialize AI Engine
    # -----------------------------------------------
    print_status(4, "INITIALIZING AI ENGINE")
    print("Importing Data Science Libraries (this may take a moment)...")
    
    # Pass simulation count to the model via environment variable
    os.environ['N_SIMULATIONS'] = str(args.simulations)
    print(f"Simulation count set to: {args.simulations}")
    
    t0 = time.time()
    
    try:
        import nigeria_tax_model
        print(f"✔ Libraries loaded in {time.time() - t0:.2f}s")
    except ImportError as e:
        print(f"❌ Critical Error: Could not import model script: {e}")
        sys.exit(1)

    # -----------------------------------------------
    # STEP 4: Run Model & Simulation
    # -----------------------------------------------
    print_status(5, "EXECUTING MODEL & SIMULATION")
    if hasattr(nigeria_tax_model, "main"):
        try:
            nigeria_tax_model.main()
            print_status(6, "COMPLETED SUCCESSFULLY")
            print("Check the 'reports' folder for the generated HTML analysis.")
        except Exception as e:
            print(f"❌ Runtime Error during simulation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Error: nigeria_tax_model.py is missing the main() function.")

if __name__ == "__main__":
    main()
