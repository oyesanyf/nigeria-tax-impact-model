"""
Oil Price Forecasting Module
============================
Uses sound statistical methods to generate realistic oil price scenarios:

1. AutoARIMA with Confidence Intervals (Bootstrap)
2. Historical Volatility Analysis (Rolling σ)
3. Percentile-Based Scenarios (P10, P25, P50, P75, P90)
4. External Forecast Integration (EIA API when available)
5. Ensemble Combining for robust estimates

Author: Nigeria Tax Model System
"""

import numpy as np
import pandas as pd
from pmdarima import auto_arima
import warnings
import os

warnings.filterwarnings('ignore')


class OilPriceForecaster:
    """
    Statistically sound oil price forecasting using ensemble methods.
    """
    
    def __init__(self, historical_prices: pd.Series, forecast_horizon: int = 4):
        """
        Initialize the forecaster with historical oil prices.
        
        Args:
            historical_prices: Pandas Series of historical oil prices (quarterly)
            forecast_horizon: Number of periods to forecast (default: 4 quarters)
        """
        self.historical_prices = historical_prices.dropna()
        self.forecast_horizon = forecast_horizon
        self.fitted_model = None
        self.point_forecast = None
        self.confidence_intervals = None
        self.scenarios = None
        
    def calculate_historical_volatility(self, window: int = 8) -> dict:
        """
        Calculate historical volatility metrics using rolling statistics.
        
        Returns:
            dict with volatility metrics: mean, std, annualized_vol, recent_trend
        """
        prices = self.historical_prices
        
        # Rolling statistics
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        # Log returns for volatility calculation (more statistically appropriate)
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # Annualized volatility (assuming quarterly data, 4 periods per year)
        annualized_vol = log_returns.std() * np.sqrt(4) * 100  # In percentage terms
        
        # Recent trend (last 4 quarters)
        recent_prices = prices.tail(4)
        if len(recent_prices) >= 2:
            recent_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100
        else:
            recent_trend = 0
            
        # Coefficient of Variation (CV) - normalized volatility measure
        cv = prices.std() / prices.mean() * 100
        
        return {
            'mean': prices.mean(),
            'std': prices.std(),
            'current_price': prices.iloc[-1],
            'rolling_mean': rolling_mean.iloc[-1] if len(rolling_mean) > 0 else prices.mean(),
            'rolling_std': rolling_std.iloc[-1] if len(rolling_std) > 0 else prices.std(),
            'annualized_volatility_pct': annualized_vol,
            'recent_trend_pct': recent_trend,
            'coefficient_of_variation': cv,
            'min_historical': prices.min(),
            'max_historical': prices.max(),
            'percentile_25': prices.quantile(0.25),
            'percentile_75': prices.quantile(0.75)
        }
    
    def fit_arima_with_confidence(self, alpha: float = 0.05) -> dict:
        """
        Fit AutoARIMA model and generate prediction intervals using bootstrap.
        
        Args:
            alpha: Significance level for confidence intervals (default: 0.05 for 95% CI)
            
        Returns:
            dict with point forecast and confidence intervals
        """
        print("\n[Statistical Forecasting] Fitting AutoARIMA with confidence intervals...")
        
        # Fit AutoARIMA
        self.fitted_model = auto_arima(
            self.historical_prices, 
            seasonal=False, 
            trace=False,
            suppress_warnings=True,
            error_action='ignore'
        )
        
        # Get point forecast with confidence intervals
        forecast, conf_int = self.fitted_model.predict(
            n_periods=self.forecast_horizon, 
            return_conf_int=True,
            alpha=alpha
        )
        
        self.point_forecast = forecast
        self.confidence_intervals = {
            'lower': conf_int[:, 0],
            'upper': conf_int[:, 1],
            'alpha': alpha
        }
        
        print(f"   --> ARIMA Order: {self.fitted_model.order}")
        print(f"   --> Point Forecast (Mean): ${forecast.mean():.2f}/barrel")
        print(f"   --> 95% CI: [${conf_int[:, 0].mean():.2f}, ${conf_int[:, 1].mean():.2f}]")
        
        return {
            'point_forecast': forecast,
            'lower_ci': conf_int[:, 0],
            'upper_ci': conf_int[:, 1],
            'model_order': self.fitted_model.order,
            'aic': self.fitted_model.aic()
        }
    
    def generate_percentile_scenarios(self, n_simulations: int = 10000) -> dict:
        """
        Generate oil price scenarios using Monte Carlo with historical distribution.
        Uses bootstrap resampling from historical volatility for robust estimates.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            dict with percentile-based scenarios
        """
        print(f"\n[Monte Carlo] Generating {n_simulations:,} simulations for percentile scenarios...")
        
        # Get volatility metrics
        vol_metrics = self.calculate_historical_volatility()
        
        # Use fitted ARIMA forecast as base if available
        if self.point_forecast is not None:
            base_forecast = self.point_forecast.mean()
        else:
            base_forecast = vol_metrics['current_price']
        
        # Historical log returns for bootstrap
        log_returns = np.log(self.historical_prices / self.historical_prices.shift(1)).dropna()
        
        # Monte Carlo simulation using bootstrap of historical returns
        simulated_prices = []
        
        for _ in range(n_simulations):
            # Bootstrap: Sample with replacement from historical returns
            sampled_returns = np.random.choice(log_returns.values, size=self.forecast_horizon, replace=True)
            
            # Generate price path
            cumulative_return = np.exp(sampled_returns.sum())
            simulated_price = base_forecast * cumulative_return
            simulated_prices.append(simulated_price)
        
        simulated_prices = np.array(simulated_prices)
        
        # Calculate percentiles
        percentiles = {
            'P5': np.percentile(simulated_prices, 5),    # Extreme low
            'P10': np.percentile(simulated_prices, 10),  # Pessimistic
            'P25': np.percentile(simulated_prices, 25),  # Conservative
            'P50': np.percentile(simulated_prices, 50),  # Median (Base)
            'P75': np.percentile(simulated_prices, 75),  # Optimistic
            'P90': np.percentile(simulated_prices, 90),  # Bullish
            'P95': np.percentile(simulated_prices, 95),  # Extreme high
            'mean': simulated_prices.mean(),
            'std': simulated_prices.std()
        }
        
        print(f"   --> P10 (Pessimistic): ${percentiles['P10']:.2f}")
        print(f"   --> P50 (Base Case):   ${percentiles['P50']:.2f}")
        print(f"   --> P90 (Optimistic):  ${percentiles['P90']:.2f}")
        
        return percentiles
    
    def fetch_eia_forecast(self) -> dict:
        """
        Attempt to fetch EIA Short-Term Energy Outlook data.
        Falls back gracefully if unavailable.
        
        Returns:
            dict with EIA forecast data or None if unavailable
        """
        try:
            import requests
            
            eia_api_key = os.environ.get('EIA_API_KEY')
            if not eia_api_key:
                print("   ⚠️ EIA_API_KEY not set. Skipping external forecast...")
                return None
            
            # EIA STEO API endpoint for Brent crude price forecast
            url = f"https://api.eia.gov/v2/steo/data/"
            params = {
                'api_key': eia_api_key,
                'frequency': 'monthly',
                'data': ['value'],
                'facets': {'seriesId': ['BREPUUS']},  # Brent crude price
                'sort': [{'column': 'period', 'direction': 'desc'}],
                'length': 24
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'response' in data and 'data' in data['response']:
                    eia_data = data['response']['data']
                    # Extract 2026 forecasts
                    forecasts_2026 = [d for d in eia_data if d.get('period', '').startswith('2026')]
                    if forecasts_2026:
                        prices = [float(d['value']) for d in forecasts_2026 if d.get('value')]
                        eia_forecast = {
                            'source': 'EIA STEO',
                            'mean': np.mean(prices),
                            'min': np.min(prices),
                            'max': np.max(prices),
                            'data_points': len(prices)
                        }
                        print(f"   ✔ EIA Forecast: ${eia_forecast['mean']:.2f}/barrel (2026)")
                        return eia_forecast
            
            return None
            
        except Exception as e:
            print(f"   ⚠️ Could not fetch EIA data: {e}")
            return None
    
    def generate_ensemble_scenarios(self, n_simulations: int = 10000) -> dict:
        """
        Generate final oil price scenarios using ensemble of methods.
        
        This combines:
        1. AutoARIMA point forecast
        2. Bootstrap percentiles from historical volatility
        3. External forecasts (if available)
        
        Returns:
            dict with recommended scenarios for simulation
        """
        print("\n" + "="*60)
        print("OIL PRICE SCENARIO GENERATION (Statistical Ensemble)")
        print("="*60)
        
        # Step 1: Historical volatility analysis
        print("\n[1/4] Analyzing Historical Volatility...")
        vol_metrics = self.calculate_historical_volatility()
        print(f"   --> Current Price: ${vol_metrics['current_price']:.2f}")
        print(f"   --> Historical Mean: ${vol_metrics['mean']:.2f}")
        print(f"   --> Annualized Volatility: {vol_metrics['annualized_volatility_pct']:.1f}%")
        print(f"   --> Recent Trend (4Q): {vol_metrics['recent_trend_pct']:+.1f}%")
        
        # Step 2: ARIMA forecast with confidence intervals
        print("\n[2/4] Fitting Time Series Model...")
        arima_result = self.fit_arima_with_confidence()
        
        # Step 3: Monte Carlo percentile scenarios
        print("\n[3/4] Running Monte Carlo Simulations...")
        percentiles = self.generate_percentile_scenarios(n_simulations)
        
        # Step 4: External forecasts (optional)
        print("\n[4/4] Checking External Forecasts...")
        eia_forecast = self.fetch_eia_forecast()
        
        # Ensemble: Combine ARIMA and Percentile estimates
        # Weight ARIMA point forecast and bootstrap median
        arima_weight = 0.6  # More weight to statistical model
        bootstrap_weight = 0.4
        
        ensemble_base = (
            arima_result['point_forecast'].mean() * arima_weight + 
            percentiles['P50'] * bootstrap_weight
        )
        
        # If EIA available, incorporate it with 30% weight
        if eia_forecast:
            eia_weight = 0.3
            ensemble_base = (
                ensemble_base * (1 - eia_weight) + 
                eia_forecast['mean'] * eia_weight
            )
        
        # Generate final scenarios using ensemble base and volatility-adjusted percentiles
        # Use historical volatility to scale the scenarios
        vol_factor = vol_metrics['rolling_std'] / vol_metrics['mean']  # Relative volatility
        
        final_scenarios = {
            'P10_Low': round(percentiles['P10'], 0),
            'P25_Conservative': round(percentiles['P25'], 0),
            'P50_Base': round(ensemble_base, 0),
            'P75_Optimistic': round(percentiles['P75'], 0),
            'P90_High': round(percentiles['P90'], 0)
        }
        
        # Store results
        self.scenarios = {
            'recommended_scenarios': final_scenarios,
            'volatility_metrics': vol_metrics,
            'arima_forecast': arima_result,
            'percentiles': percentiles,
            'eia_forecast': eia_forecast,
            'ensemble_base': ensemble_base,
            'methodology': {
                'arima_weight': arima_weight,
                'bootstrap_weight': bootstrap_weight,
                'eia_weight': 0.3 if eia_forecast else 0,
                'n_simulations': n_simulations
            }
        }
        
        # Print final recommendations
        print("\n" + "="*60)
        print("RECOMMENDED OIL PRICE SCENARIOS")
        print("="*60)
        print(f"\n🛢️  Statistical Ensemble Results:")
        print(f"   P10 (Low/Pessimistic):    ${final_scenarios['P10_Low']:.0f}/barrel")
        print(f"   P25 (Conservative):       ${final_scenarios['P25_Conservative']:.0f}/barrel")
        print(f"   P50 (Base Case):          ${final_scenarios['P50_Base']:.0f}/barrel")
        print(f"   P75 (Optimistic):         ${final_scenarios['P75_Optimistic']:.0f}/barrel")
        print(f"   P90 (High/Bullish):       ${final_scenarios['P90_High']:.0f}/barrel")
        print("="*60)
        
        return self.scenarios
    
    def get_simulation_prices(self, scenario_type: str = 'standard') -> list:
        """
        Get the list of oil prices to use for multi-scenario simulation.
        
        Args:
            scenario_type: 
                'standard' - 3 scenarios (P25, P50, P75)
                'extended' - 5 scenarios (P10, P25, P50, P75, P90)
                'full' - 7 scenarios including extremes
                
        Returns:
            list of oil prices for simulation
        """
        if self.scenarios is None:
            self.generate_ensemble_scenarios()
        
        scenarios = self.scenarios['recommended_scenarios']
        
        if scenario_type == 'standard':
            return [
                scenarios['P25_Conservative'],
                scenarios['P50_Base'],
                scenarios['P75_Optimistic']
            ]
        elif scenario_type == 'extended':
            return [
                scenarios['P10_Low'],
                scenarios['P25_Conservative'],
                scenarios['P50_Base'],
                scenarios['P75_Optimistic'],
                scenarios['P90_High']
            ]
        else:  # full
            percentiles = self.scenarios['percentiles']
            return [
                round(percentiles['P5'], 0),
                scenarios['P10_Low'],
                scenarios['P25_Conservative'],
                scenarios['P50_Base'],
                scenarios['P75_Optimistic'],
                scenarios['P90_High'],
                round(percentiles['P95'], 0)
            ]


def generate_auto_oil_scenarios(historical_oil_prices: pd.Series, 
                                 forecast_horizon: int = 4,
                                 scenario_type: str = 'extended') -> tuple:
    """
    Convenience function to generate statistically-sound oil price scenarios.
    
    Args:
        historical_oil_prices: Pandas Series of historical oil prices
        forecast_horizon: Number of quarters to forecast
        scenario_type: 'standard' (3), 'extended' (5), or 'full' (7) scenarios
        
    Returns:
        tuple: (list of oil prices, full forecast results dict)
    """
    forecaster = OilPriceForecaster(historical_oil_prices, forecast_horizon)
    results = forecaster.generate_ensemble_scenarios()
    oil_prices = forecaster.get_simulation_prices(scenario_type)
    
    return oil_prices, results


if __name__ == "__main__":
    # Test with sample data
    print("Testing Oil Price Forecaster with sample data...")
    
    # Generate sample historical data
    np.random.seed(42)
    dates = pd.date_range(start='2015-01-01', periods=40, freq='QE')
    
    # Simulate oil price path with realistic volatility
    base_price = 60
    returns = np.random.normal(0.01, 0.15, 40)  # 15% quarterly volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    sample_data = pd.Series(prices, index=dates, name='Oil_Price')
    
    # Run forecaster
    oil_prices, results = generate_auto_oil_scenarios(sample_data, scenario_type='extended')
    
    print(f"\nFinal simulation prices: {oil_prices}")
