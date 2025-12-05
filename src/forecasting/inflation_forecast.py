import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Configuration
INFLATION_DATA_PATH = "data/csv/inflation_pre.csv"
FOODPRICE_DATA_PATH = "data/csv/filtered_foodprice_mean.csv"
FORECAST_PERIODS = 5  # Number of years to forecast


def load_and_prepare_data(country_name):
    """
    Load and prepare data for Prophet model
    Prophet requires columns named 'ds' (date) and 'y' (value)
    """
    df_inflation = pd.read_csv(INFLATION_DATA_PATH)
    country_data = df_inflation[df_inflation['Country'] == country_name].copy()
    
    country_data['ds'] = pd.to_datetime(country_data['Year'].astype(str) + '-01-01')
    country_data['y'] = country_data['Avg.Inflation']
    prophet_data = country_data[['ds', 'y']].sort_values('ds').reset_index(drop=True)
    
    return prophet_data


def load_foodprice_data(country_name):
    """
    Load food price data for additional analysis (regressor)
    """
    df_foodprice = pd.read_csv(FOODPRICE_DATA_PATH)
    country_data = df_foodprice[df_foodprice['Country'] == country_name].copy()
    
    if len(country_data) == 0:
        return None
    
    country_data['ds'] = pd.to_datetime(country_data['Year'].astype(str) + '-01-01')
    country_data = country_data[['ds', 'Open_price', 'Close_price']].sort_values('ds').reset_index(drop=True)
    
    return country_data


def create_and_train_model(data, use_foodprice=False, foodprice_data=None):
    """
    Create and train Prophet model
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    
    if use_foodprice and foodprice_data is not None:
        data_with_regressor = data.merge(foodprice_data, on='ds', how='left')
        data_with_regressor['Close_price'] = data_with_regressor['Close_price'].ffill().bfill()
        
        model.add_regressor('Close_price')
        model.fit(data_with_regressor)
        return model, data_with_regressor
    else:
        model.fit(data)
        return model, data


def make_forecast(model, data, periods=5, use_foodprice=False, foodprice_data=None):
    """
    Make future predictions
    """
    future = model.make_future_dataframe(periods=periods, freq='Y')
    
    if use_foodprice and foodprice_data is not None:
        last_price = foodprice_data['Close_price'].iloc[-1]
        future = future.merge(
            foodprice_data[['ds', 'Close_price']], 
            on='ds', 
            how='left'
        )
        future['Close_price'].fillna(last_price, inplace=True)
    
    forecast = model.predict(future)
    return forecast


def plot_forecast(model, forecast, data, country_name, save_path=None, show_plots=True):
    """
    Visualize the forecast results
    """
    fig1 = model.plot(forecast, figsize=(14, 6))
    ax1 = fig1.gca()
    ax1.set_title(f'Inflation Forecast for {country_name}', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Inflation Rate (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is None:
        save_path = f'forecast_{country_name.replace(" ", "_")}.png'
    
    forecast_path = save_path.replace('.png', '_forecast.png')
    plt.savefig(forecast_path, dpi=300, bbox_inches='tight')
    print(f"✓ Forecast plot saved to: {forecast_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    fig2 = model.plot_components(forecast, figsize=(14, 8))
    plt.tight_layout()
    
    components_path = save_path.replace('.png', '_components.png')
    plt.savefig(components_path, dpi=300, bbox_inches='tight')
    print(f"✓ Components plot saved to: {components_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def print_forecast_summary(forecast, periods=5):
    """
    Print summary of forecast results
    """
    print("\n" + "="*60)
    print("FORECAST SUMMARY")
    print("="*60)
    
    future_forecast = forecast.tail(periods)
    
    print(f"\nPredicted Inflation Rates for the Next {periods} Years:")
    print("-"*60)
    for idx, row in future_forecast.iterrows():
        year = row['ds'].year
        predicted = row['yhat']
        lower = row['yhat_lower']
        upper = row['yhat_upper']
        print(f"Year {year}: {predicted:.2f}% (Range: {lower:.2f}% to {upper:.2f}%)")
    
    print("="*60 + "\n")


def forecast_for_country(country_name, forecast_years=5, use_foodprice=False, show_plots=False):
    """
    Main function to run forecasting for a specific country
    """
    print(f"\n{'='*60}")
    print(f"Starting Inflation Forecast for {country_name}")
    print(f"{'='*60}\n")
    
    print("Loading data...")
    data = load_and_prepare_data(country_name)
    print(f"Loaded {len(data)} years of historical data ({data['ds'].min().year} - {data['ds'].max().year})")
    
    foodprice_data = None
    if use_foodprice:
        foodprice_data = load_foodprice_data(country_name)
        if foodprice_data is not None:
            print(f"Loaded food price data as additional regressor")
        else:
            print(f"Warning: No food price data available for {country_name}")
            use_foodprice = False
    
    print("\nTraining Prophet model...")
    model, data_with_regressors = create_and_train_model(data, use_foodprice, foodprice_data)
    print("Model training completed!")
    
    print(f"\nGenerating {forecast_years}-year forecast...")
    forecast = make_forecast(model, data, periods=forecast_years, use_foodprice=use_foodprice, foodprice_data=foodprice_data)
    
    print_forecast_summary(forecast, periods=forecast_years)
    
    print("Generating visualization...")
    plot_forecast(model, forecast, data, country_name, show_plots=show_plots)
    
    return model, forecast


def forecast_multiple_countries(countries, forecast_years=5):
    """
    Run forecasting for multiple countries and compare results
    """
    results = {}
    
    fig, axes = plt.subplots(len(countries), 1, figsize=(14, 5*len(countries)))
    if len(countries) == 1:
        axes = [axes]
    
    for idx, country in enumerate(countries):
        print(f"\n{'#'*60}")
        print(f"Processing {country} ({idx+1}/{len(countries)})")
        print(f"{'#'*60}")
        
        data = load_and_prepare_data(country)
        model, _ = create_and_train_model(data)
        forecast = make_forecast(model, data, periods=forecast_years)
        
        results[country] = {
            'model': model,
            'forecast': forecast,
            'data': data
        }
        
        model.plot(forecast, ax=axes[idx])
        axes[idx].set_title(f'Inflation Forecast: {country}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Year')
        axes[idx].set_ylabel('Inflation Rate (%)')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    COUNTRY_TO_FORECAST = 'Sudan'
    YEARS_TO_FORECAST = 5
    USE_FOOD_PRICE = True
    SHOW_PLOTS_INTERACTIVELY = True
    
    df = pd.read_csv(INFLATION_DATA_PATH)
    available_countries = df['Country'].unique()
    print("="*60)
    print("Available countries for forecasting:")
    print("="*60)
    for i, country in enumerate(available_countries, 1):
        print(f"{i:2}. {country}")
    
    print("\n" + "="*60)
    print(f"Forecasting inflation for: {COUNTRY_TO_FORECAST}")
    print("="*60)
    
    model, forecast = forecast_for_country(
        country_name=COUNTRY_TO_FORECAST,
        forecast_years=YEARS_TO_FORECAST,
        use_foodprice=USE_FOOD_PRICE,
        show_plots=SHOW_PLOTS_INTERACTIVELY
    )
    
    print("\n" + "="*60)
    print("✓ Forecasting completed!")
    print("="*60)
    if not SHOW_PLOTS_INTERACTIVELY:
        print(f"\nPlots saved to:")
        print(f"  - forecast_{COUNTRY_TO_FORECAST.replace(' ', '_')}_forecast.png")
        print(f"  - forecast_{COUNTRY_TO_FORECAST.replace(' ', '_')}_components.png")