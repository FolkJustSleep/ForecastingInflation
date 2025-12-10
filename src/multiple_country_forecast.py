"""
Multi-Country Inflation Forecasting using Prophet
Compare forecasts across different countries
"""
import pickle
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
import preparedata_to_forcast as prep
import forecasting as trainmodel
import stationary_test as statest
warnings.filterwarnings('ignore')

# Configuration
FORECAST_YEARS = 3
COUNTRIES_TO_COMPARE = []

def load_data():
    """Load inflation and food price data"""
    inflation_df = pd.read_csv('data/csv/processed/inflation_pre.csv')
    foodprice_df = pd.read_csv('data/csv/processed/filtered_foodprice_mean_1.csv')
    return inflation_df, foodprice_df


def train_and_forecast(df,test_df, periods):
    """Train Prophet model and make forecast"""
    model = Prophet(
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        interval_width=0.95
    )
    model.fit(df)
    model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(test_df)
    return model, forecast

def plot_multi_country_comparison(inflation_df, foodprice_df, countries, forecast_years):
    """Compare historical and forecasted inflation across countries"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Historical comparison
    ax1 = axes[0]
    for i, country in enumerate(countries):
        df = prep.prepare_prophet_data(inflation_df, country)
        ax1.plot(df['ds'], df['y'], marker='o', linewidth=2, 
                markersize=5, label=country, alpha=0.8)
    
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
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax2.set_title(f'Inflation Forecast Comparison (Next {forecast_years} Years)', 
                  fontsize=16, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Inflation Rate (%)', fontsize=12)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    for i, country in enumerate(countries):
        df = prep.prepare_prophet_data(inflation_df, country)
        df_with_regressor = prep.add_food_price_regressor(inflation_df, foodprice_df, country=country)
        # Split data
        train_df, test_df = trainmodel.split_data(df_with_regressor, train_ratio=0.8)
        # made the data stationary
        train_df_log, test_df_log = statest.stationary_data(train_df, test_df)
        model, forecast = train_and_forecast(train_df_log, test_df_log, forecast_years)
        forecast.to_csv(f'results/output/forecast_{country}.csv', index=False)
        pickle.dump(model, open(f'data/model/prophet_model_{country}_forecasting.pkl', 'wb'))
        # Plot historical
        ax2.plot(df['ds'], df['y'], linewidth=1.5, alpha=0.5)
        
        # Plot forecast
        forecast_future = forecast
        ax2.plot(forecast_future['ds'], forecast_future['yhat'], 
                marker='s', linewidth=2.5, markersize=6, 
                label=country, linestyle='--')
        plt.figure(figsize=(12,6))
        plt.plot(train_df['ds'], train_df['y'], label='Train', marker='o')
        plt.plot(test_df['ds'], test_df['y'], label='Actual', marker='o')
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', marker='x')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
        plt.xlabel('Date')
        plt.ylabel('Inflation Rate (%)')
        plt.title(f'Inflation Forecast vs Actuals for {country}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.savefig(f'results/figure/forecast_vs_actuals_{country}.png', dpi=300, bbox_inches='tight')
        print(f"Forecast vs Actuals plot saved to results/figure/forecast_vs_actuals_{country}.png")
        
        # Store forecast summary
        avg_forecast = forecast_future['yhat'].mean()
        print(f"avg_forecast for {country}: {avg_forecast}")
        forecast_summary.append({
            'Country': country,
            'Avg_Forecast': avg_forecast
        })
    
    
    # fig.savefig('results/figure/multi_country_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: results/figure/multi_country_comparison.png")
    
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
    plt.savefig('results/figure/forecast_rankings.png', dpi=300, bbox_inches='tight')
    print("  Saved: results/figure/forecast_rankings.png")

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("MULTI-COUNTRY INFLATION FORECASTING")
    print("="*70 + "\n")
    
    # Load data
    inflation_df, foodprice_df = load_data()
    COUNTRIES_TO_COMPARE = inflation_df['Country'].unique().tolist()
    
    print(f"Analyzing {len(COUNTRIES_TO_COMPARE)} countries:")
    for country in COUNTRIES_TO_COMPARE:
        print(f"  - {country}")
    
    print(f"\nForecast period: {FORECAST_YEARS} years")
    print("\nGenerating visualizations...")
    
    
    # Multi-country comparison
    summary_df = plot_multi_country_comparison(inflation_df, foodprice_df, COUNTRIES_TO_COMPARE, FORECAST_YEARS)
    summary_df.to_csv('results/output/forecast_summary.csv', index=False)
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