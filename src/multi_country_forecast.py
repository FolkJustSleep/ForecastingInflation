"""
Multi-Country Inflation Forecasting using Prophet
Compare forecasts across different countries
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings('ignore')

# Configuration
f = open('data/pickle/common_years_and_countries.pickle', 'rb')
common_data = pickle.load(f)
f.close()
FORECAST_YEARS = 3
COUNTRIES_TO_COMPARE = np.asarray(common_data['Country'])
COUNTRIES_TO_COMPARE = np.delete(COUNTRIES_TO_COMPARE, np.where(COUNTRIES_TO_COMPARE == 'Somalia'))

def load_data():
    """Load inflation and food price data"""
    inflation_df = pd.read_csv('data/csv/clean_inflation.csv')
    inflation_df_org = pd.read_csv('data/csv/inflation_pre.csv') 
    foodprice_df = pd.read_csv('data/csv/filtered_foodprice_mean.csv')
    return inflation_df, inflation_df_org, foodprice_df

def prepare_prophet_data(df, country):
    """Prepare data for Prophet model"""
    country_data = df[df['Country'] == country].copy()
    country_data['ds'] = pd.to_datetime(country_data['Year'], format='%Y')
    country_data['y'] = country_data['Avg.Inflation']
    prophet_df = country_data[['ds', 'y']].sort_values('ds').reset_index(drop=True)
    return prophet_df

def train_and_forecast(df, periods):
    """Train Prophet model and make forecast"""
    model = Prophet(
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        interval_width=0.95
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)
    return model, forecast

def plot_multi_country_comparison(inflation_df, inflation_df_org, countries, forecast_years):
    """Compare historical and forecasted inflation across countries"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    colors = ['#2E86AB', '#A23B72', '#F77F00', '#06A77D', '#D62828', '#6A4C93', '#FF6F61', '#1982C4', '#8AC926', '#FFCA3A', '#6A4C93', '#FF595E', '#1982C4', '#FF6F61', '#8AC926', '#1982C4', '#FFCA3A', '#6A4C93']
    
    # Plot 1: Historical comparison
    ax1 = axes[0]
    for i, country in enumerate(countries):
        df = prepare_prophet_data(inflation_df_org, country)
        ax1.plot(df['ds'], df['y'], marker='o', linewidth=2, 
                markersize=5, color=colors[i], label=country, alpha=0.8)
    
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_title('Historical Inflation Comparison Across Countries', 
                  fontsize=16, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Inflation Rate (%)', fontsize=12)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Forecast comparison
    ax2 = axes[1]
    forecast_summary = []
    
    for i, country in enumerate(countries):
        df = prepare_prophet_data(inflation_df, country)
        model, forecast = train_and_forecast(df, forecast_years)
        
        # Plot historical
        ax2.plot(df['ds'], df['y'], linewidth=1.5, color=colors[i], alpha=0.5)
        
        # Plot forecast
        forecast_future = forecast[forecast['ds'] > df['ds'].max()]
        ax2.plot(forecast_future['ds'], forecast_future['yhat'], 
                marker='s', linewidth=2.5, markersize=6, 
                color=colors[i], label=country, linestyle='--')
        
        # Store forecast summary
        avg_forecast = forecast_future['yhat'].mean()
        forecast_summary.append({
            'Country': country,
            'Avg_Forecast': avg_forecast
        })
    
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax2.set_title(f'Inflation Forecast Comparison (Next {forecast_years} Years)', 
                  fontsize=16, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Inflation Rate (%)', fontsize=12)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_country_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: multi_country_comparison.png")
    
    return pd.DataFrame(forecast_summary)

def plot_forecast_rankings(summary_df):
    """Plot countries ranked by forecasted inflation"""
    summary_df = summary_df.sort_values('Avg_Forecast', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(summary_df['Country'], summary_df['Avg_Forecast'], 
                   color=['#06A77D' if x < 10 else '#F77F00' if x < 20 else '#D62828' 
                          for x in summary_df['Avg_Forecast']])
    
    ax.set_xlabel('Average Forecasted Inflation (%)', fontsize=12)
    ax.set_title('Countries Ranked by Forecasted Inflation Rate', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(summary_df.iterrows()):
        ax.text(row['Avg_Forecast'] + 0.5, i, f"{row['Avg_Forecast']:.1f}%", 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('forecast_rankings.png', dpi=300, bbox_inches='tight')
    print("  Saved: forecast_rankings.png")

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("MULTI-COUNTRY INFLATION FORECASTING")
    print("="*70 + "\n")
    
    # Load data
    inflation_df, inflation_df_org, foodprice_df = load_data()
    
    print(f"Analyzing {len(COUNTRIES_TO_COMPARE)} countries:")
    for country in COUNTRIES_TO_COMPARE:
        print(f"  - {country}")
    
    print(f"\nForecast period: {FORECAST_YEARS} years")
    print("\nGenerating visualizations...")
    
    # Multi-country comparison
    summary_df = plot_multi_country_comparison(inflation_df, inflation_df_org,  COUNTRIES_TO_COMPARE, FORECAST_YEARS)
    
    # Rankings
    plot_forecast_rankings(summary_df)
    
    # Print summary table
    print("\n" + "="*70)
    print("FORECAST SUMMARY")
    print("="*70)
    summary_df = summary_df.sort_values('Avg_Forecast', ascending=False)
    print(summary_df.to_string(index=False))
    print("="*70 + "\n")
    
    print("Multi-country forecasting completed successfully!")

if __name__ == "__main__":
    main()
