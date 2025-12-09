import pickle
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from statsmodels.tsa.stattools import adfuller # For ADF test
from statsmodels.tsa.statespace.sarimax import SARIMAX

import preparedata_to_forcast as prep

def split_data(df, train_ratio=0.8):
    """Split data into training and testing sets based on the given ratio"""
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df

def train_prophet_model(train_df, test_df, use_food_price=False):
    """Train Prophet model with optional food price regressor"""
    print("Training Prophet Model...")
    model = Prophet(
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,  # Controls trend flexibility (lower = less flexible)
        seasonality_prior_scale=10,     # Controls seasonality strength
        interval_width=0.95             # 95% confidence interval
    )
    print("Prophet model initialized")
    

    # Add food price as regressor if available
    if use_food_price and 'avg_food_price' in train_df.columns:
        model.add_regressor('avg_food_price')
        print("  Food price added as external regressor")
    
    # Fit the model
    model.fit(train_df)
    model.make_future_dataframe(periods=3)
    forecast = model.predict(test_df)
    print("  Forecasting on test data completed")
    print("  Model training completed")
    pickle.dump(model, open(f'data/pickle/prophet_model_forecasting.pkl', 'wb'))
    print("  Model saved to data/pickle/prophet_model_forecasting.pkl")
    return forecast
    
   
def plot_split_data(train_df, test_df): 
    plt.figure(figsize=(12,4))
    plt.plot(train_df, label='Train')
    plt.plot(test_df, label='Valid')
    plt.legend(); plt.xticks(rotation=45); plt.title('Food Price: Train/Valid Split')
    plt.tight_layout();
    plt.savefig('split_data_food_price.png', dpi=300, bbox_inches='tight')
    print("  Train/Test split done. Plot saved to split_data_food_price.png")

def plot_forecast_results(forecast, df):
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
    ax1.set_title(f'Inflation Forecast', 
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
if __name__ == "__main__":
    inflation_df, foodprice_df = prep.load_data()
    df_prophet = prep.prepare_prophet_data(inflation_df)
    df_with_regressor = prep.add_food_price_regressor(inflation_df, foodprice_df)
    print(df_with_regressor.head(2))
    train_df, test_df = split_data(df_with_regressor, train_ratio=0.8)
    # plot_split_inflation(train_df, test_df)
    plot_split_data(train_df, test_df)
    
    # # Train model with food price regressor
    forecast = train_prophet_model(train_df, test_df, use_food_price=True)
    print(forecast.head(3))
    # fig = plot_forecast_results(forecast, train_df)
    # fig.savefig('prophet_forecast_with_foodprice.png', dpi=300, bbox_inches='tight')