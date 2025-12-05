"""
Simple Example: Quick Inflation Forecast
Run this script to quickly test Prophet forecasting
"""

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def quick_forecast(country='Afghanistan', forecast_years=3):
    """
    Quick inflation forecast for any country
    
    Parameters:
    -----------
    country : str
        Country name (must match data exactly)
    forecast_years : int
        Number of years to forecast
    """
    
    print(f"\nForecasting inflation for {country}...")
    print("-" * 50)
    
    # Load data
    df = pd.read_csv('data/csv/inflation_pre.csv')
    
    # Prepare data for Prophet
    country_data = df[df['Country'] == country].copy()
    country_data['ds'] = pd.to_datetime(country_data['Year'], format='%Y')
    country_data['y'] = country_data['Avg.Inflation']
    prophet_df = country_data[['ds', 'y']].sort_values('ds')
    
    # Print basic stats
    print(f"Historical period: {prophet_df['ds'].min().year} - {prophet_df['ds'].max().year}")
    print(f"Data points: {len(prophet_df)}")
    print(f"Average inflation: {prophet_df['y'].mean():.2f}%")
    print(f"Min: {prophet_df['y'].min():.2f}% | Max: {prophet_df['y'].max():.2f}%")
    
    # Train model
    model = Prophet(yearly_seasonality=True, interval_width=0.95)
    model.fit(prophet_df)
    
    # Make forecast
    future = model.make_future_dataframe(periods=forecast_years, freq='Y')
    forecast = model.predict(future)
    
    # Print forecast
    print(f"\nForecast for next {forecast_years} years:")
    print("-" * 50)
    future_only = forecast[forecast['ds'] > prophet_df['ds'].max()]
    
    for _, row in future_only.iterrows():
        year = row['ds'].year
        pred = row['yhat']
        lower = row['yhat_lower']
        upper = row['yhat_upper']
        print(f"{year}: {pred:.2f}% (range: {lower:.2f}% to {upper:.2f}%)")
    
    # Create simple plot
    plt.figure(figsize=(12, 6))
    
    # Historical
    plt.plot(prophet_df['ds'], prophet_df['y'], 
             marker='o', linewidth=2, markersize=6, 
             color='#2E86AB', label='Historical')
    
    # Forecast
    plt.plot(future_only['ds'], future_only['yhat'], 
             marker='s', linewidth=2, markersize=6,
             color='#F77F00', label='Forecast', linestyle='--')
    
    # Confidence interval
    plt.fill_between(future_only['ds'], 
                      future_only['yhat_lower'], 
                      future_only['yhat_upper'],
                      alpha=0.3, color='#F77F00')
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    plt.title(f'{country} Inflation Forecast', fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Inflation Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'{country.lower().replace(" ", "_")}_forecast.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {filename}")
    
    return forecast

if __name__ == "__main__":
    # Example usage - modify these values to test different scenarios
    
    # Example 1: Afghanistan (default)
    quick_forecast('Afghanistan', forecast_years=3)
    
    # Example 2: Try other countries (uncomment to test)
    # quick_forecast('Lebanon', forecast_years=5)
    # quick_forecast('Haiti', forecast_years=3)
    # quick_forecast('Sudan', forecast_years=3)
    
    print("\nDone! Check the generated PNG files.")
