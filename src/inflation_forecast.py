"""
Inflation Forecasting using Prophet
Simple and straightforward implementation for time series forecasting
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
FORECAST_YEARS = 3
COUNTRY_TO_FORECAST = 'Afghanistan'  # Change this to forecast different countries

def load_data():
    """Load inflation and food price data"""
    inflation_df_plot = pd.read_csv('data/csv/inflation_pre.csv')
    inflation_df = pd.read_csv('data/csv/clean_inflation.csv')
    foodprice_df = pd.read_csv('data/csv/filtered_foodprice_mean.csv')
    
    print(f"Loaded {len(inflation_df)} inflation records")
    print(f"Loaded {len(foodprice_df)} food price records")
    print(f"Common countries: {len(set(inflation_df['Country']) & set(foodprice_df['Country']))}")
    
    return inflation_df, foodprice_df, inflation_df_plot

def prepare_prophet_data(df, country):
    """Prepare data for Prophet model (requires 'ds' and 'y' columns)"""
    country_data = df[df['Country'] == country].copy()
    
    # Prophet requires datetime column named 'ds' and target column named 'y'
    country_data['ds'] = pd.to_datetime(country_data['Year'], format='%Y')
    country_data['y'] = country_data['Avg.Inflation']
    
    prophet_df = country_data[['ds', 'y']].sort_values('ds').reset_index(drop=True)
    
    print(f"\n{country} data prepared:")
    print(f"  Period: {prophet_df['ds'].min().year} - {prophet_df['ds'].max().year}")
    print(f"  Data points: {len(prophet_df)}")
    print(f"  Inflation range: {prophet_df['y'].min():.2f}% to {prophet_df['y'].max():.2f}%")
    
    return prophet_df

def add_food_price_regressor(inflation_df, foodprice_df, country):
    """Add food price as external regressor to improve forecast accuracy"""
    # Get food price data for the country
    food_data = foodprice_df[foodprice_df['Country'] == country].copy()
    food_data['ds'] = pd.to_datetime(food_data['Year'], format='%Y')
    
    # Use average of open and close price as representative food price
    food_data['avg_food_price'] = (food_data['Open_price'] + food_data['Close_price']) / 2
    
    # Merge with inflation data
    inflation_data = inflation_df[inflation_df['Country'] == country].copy()
    inflation_data['ds'] = pd.to_datetime(inflation_data['Year'], format='%Y')
    inflation_data['y'] = inflation_data['Avg.Inflation']
    
    merged_df = inflation_data[['ds', 'y']].merge(
        food_data[['ds', 'avg_food_price']], 
        on='ds', 
        how='left'
    )
    
    # Forward fill missing food price values
    merged_df['avg_food_price'].fillna(method='ffill', inplace=True)
    merged_df['avg_food_price'].fillna(method='bfill', inplace=True)
    
    return merged_df

def train_prophet_model(df, use_food_price=False):
    """Train Prophet model with optional food price regressor"""
    # Initialize Prophet with yearly seasonality
    model = Prophet(
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,  # Controls trend flexibility (lower = less flexible)
        seasonality_prior_scale=10,     # Controls seasonality strength
        interval_width=0.95             # 95% confidence interval
    )
    
    # Add food price as regressor if available
    if use_food_price and 'avg_food_price' in df.columns:
        model.add_regressor('avg_food_price')
        print("  Food price added as external regressor")
    
    # Fit the model
    model.fit(df)
    print("  Model training completed")
    
    return model

def make_forecast(model, df, periods, use_food_price=False):
    """Generate future predictions"""
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq='Y')
    
    # Add food price regressor for future periods if used
    if use_food_price and 'avg_food_price' in df.columns:
        # For future periods, use the last known food price (simple approach)
        last_food_price = df['avg_food_price'].iloc[-1]
        future = future.merge(df[['ds', 'avg_food_price']], on='ds', how='left')
        future['avg_food_price'].fillna(last_food_price, inplace=True)
    
    # Make predictions
    forecast = model.predict(future)
    
    return forecast

def plot_historical_inflation(df, country):
    """Plot historical inflation data"""
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['ds'], df['y'], marker='o', linewidth=2, markersize=6, color='#2E86AB')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.title(f'Historical Inflation Rate - {country}', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Inflation Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def plot_forecast_results(model, forecast, inflation_df_plot, df, country, forecast_years):
    """Plot forecast results with components"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Forecast with confidence intervals
    ax1 = axes[0]
    
    # Historical data
    ax1.plot(df['ds'], df['y'], marker='o', linewidth=2, 
             markersize=6, color='#2E86AB', label='Historical Data')
    
    # Forecast
    forecast_future = forecast[forecast['ds'] > df['ds'].max()]
    ax1.plot(forecast_future['ds'], forecast_future['yhat'], 
             marker='s', linewidth=2, markersize=6, 
             color='#F77F00', label='Forecast', linestyle='--')
    # Confidence interval
    ax1.fill_between(forecast_future['ds'], 
                      forecast_future['yhat_lower'], 
                      forecast_future['yhat_upper'],
                      alpha=0.3, color='#F77F00', label='95% Confidence Interval')
    
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_title(f'Inflation Forecast - {country} (Next {forecast_years} Years)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Inflation Rate (%)', fontsize=11)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Trend component
    ax2 = axes[1]
    ax2.plot(forecast['ds'], forecast['trend'], linewidth=2, color='#06A77D')
    ax2.set_title('Inflation Trend Component', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Trend (%)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_forecast_summary(forecast, df, forecast_years):
    """Print summary of forecast results"""
    future_forecast = forecast[forecast['ds'] > df['ds'].max()].copy()
    future_forecast['year'] = future_forecast['ds'].dt.year
    
    print("\n" + "="*60)
    print(f"FORECAST SUMMARY (Next {forecast_years} Years)")
    print("="*60)
    
    for _, row in future_forecast.iterrows():
        print(f"Year {row['year']}:")
        print(f"  Predicted Inflation: {row['yhat']:.2f}%")
        print(f"  Confidence Interval: [{row['yhat_lower']:.2f}%, {row['yhat_upper']:.2f}%]")
    
    print("\n" + "-"*60)
    print("INSIGHTS:")
    print("-"*60)
    
    # Calculate average historical inflation
    avg_historical = df['y'].mean()
    avg_forecast = future_forecast['yhat'].mean()
    
    print(f"Average Historical Inflation: {avg_historical:.2f}%")
    print(f"Average Forecasted Inflation: {avg_forecast:.2f}%")
    
    if avg_forecast > avg_historical:
        diff = avg_forecast - avg_historical
        print(f"\nForecast indicates HIGHER inflation ({diff:+.2f}% increase)")
    else:
        diff = avg_historical - avg_forecast
        print(f"\nForecast indicates LOWER inflation ({diff:.2f}% decrease)")
    
    # Trend analysis
    if future_forecast['yhat'].iloc[-1] > future_forecast['yhat'].iloc[0]:
        print("Trend: INCREASING inflation over forecast period")
    else:
        print("Trend: DECREASING inflation over forecast period")
    
    print("="*60 + "\n")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("INFLATION FORECASTING WITH PROPHET")
    print("="*60 + "\n")
    
    # Load data
    inflation_df, foodprice_df, inflation_df_plot = load_data()
    inflation_df_plot['ds'] = pd.to_datetime(inflation_df_plot['Year'], format='%Y')
    inflation_df_plot['y'] = inflation_df_plot['Avg.Inflation'][inflation_df_plot['Country'] == COUNTRY_TO_FORECAST]
    
    # Prepare data for selected country
    print(f"\nForecasting for: {COUNTRY_TO_FORECAST}")
    print("-"*60)
    
    # Try with food price regressor first
    try:
        print("\nAttempting forecast with food price regressor...")
        df_with_regressor = add_food_price_regressor(inflation_df, foodprice_df, COUNTRY_TO_FORECAST)
        model = train_prophet_model(df_with_regressor, use_food_price=True)
        forecast = make_forecast(model, df_with_regressor, FORECAST_YEARS, use_food_price=True)
        df_final = df_with_regressor
    except Exception as e:
        print(f"\nFood price regressor failed: {e}")
        print("Falling back to simple model without regressor...")
        df_final = prepare_prophet_data(inflation_df, COUNTRY_TO_FORECAST)
        model = train_prophet_model(df_final, use_food_price=False)
        forecast = make_forecast(model, df_final, FORECAST_YEARS, use_food_price=False)
    
    # Print forecast summary
    print_forecast_summary(forecast, df_final, FORECAST_YEARS)
    
    # Create visualizations
    print("Generating visualizations...")
    
    # Plot 1: Historical inflation
    plot_historical_inflation(inflation_df_plot, COUNTRY_TO_FORECAST)
    plt.savefig('inflation_historical.png', dpi=300, bbox_inches='tight')
    print("  Saved: inflation_historical.png")
    
    # Plot 2: Forecast results
    plot_forecast_results(model, forecast, inflation_df_plot, df_final, COUNTRY_TO_FORECAST, FORECAST_YEARS)
    plt.savefig('inflation_forecast.png', dpi=300, bbox_inches='tight')
    print("  Saved: inflation_forecast.png")
    
    print("\nForecasting completed successfully!")

if __name__ == "__main__":
    main()
