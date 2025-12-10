import pickle
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from statsmodels.tsa.stattools import adfuller # For ADF test
from statsmodels.tsa.statespace.sarimax import SARIMAX

import preparedata_to_forcast as prep
import stationary_test as statest

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
    pickle.dump(model, open(f'data/model/prophet_model_forecasting.pkl', 'wb'))
    print("  Model saved to data/model/prophet_model_forecasting.pkl")
    return forecast
    
   
def plot_split_data(train_df, test_df): 
    _, axes = plt.subplots(2, 1, figsize=(16, 12))
    axis = axes[0]
    
    axis.plot(train_df['ds'],train_df['y'], label='Train')
    axis.plot(test_df['ds'], test_df['y'], label='Valid')
    axis.legend(); plt.xticks(rotation=45); plt.title('Inflation Price: Train/Valid Split')
    
    axis2 = axes[1]
    axis2.plot(train_df['ds'],train_df['avg_food_price'], label='Train')
    axis2.plot(test_df['ds'], test_df['avg_food_price'], label='Valid')
    axis2.legend(); plt.xticks(rotation=45); plt.title('Food Price: Train/Valid Split')
    plt.tight_layout();
    plt.savefig('results/split_data.png', dpi=300, bbox_inches='tight')
    print("  Train/Test split done. Plot saved to split_data.png")

if __name__ == "__main__":
    inflation_df, foodprice_df = prep.load_data()
    country_list = inflation_df['Country'].unique().tolist()
    print("Available countries for stationarity test:")
    for country in country_list:
        print(country)
    country = input("Enter country name for forecasting: ")
    df_prophet = prep.prepare_prophet_data(inflation_df, country=country)
    
    # Test stationarity
    p_value = statest.test_adf_stationarity(df_prophet)
    
    df_with_regressor = prep.add_food_price_regressor(inflation_df, foodprice_df, country=country)
    train_df, test_df = split_data(df_with_regressor, train_ratio=0.8)
    plot_split_data(train_df, test_df)
    forecast = pd.DataFrame()
    if p_value > 0.05:
        # made the data stationary
        train_df_log, test_df_log = statest.stationary_data(train_df, test_df)
        #Train model with food price regressor
        forecast = train_prophet_model(train_df_log, test_df_log, use_food_price=True)
        print("Forecasting completed.")
    else:
        # Train model without food price regressor
        forecast = train_prophet_model(train_df, test_df, use_food_price=False)
        print("Forecasting completed.")
        
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
    plt.savefig('results/figure/forecast_vs_actuals.png', dpi=300, bbox_inches='tight')
    print("Forecast vs Actuals plot saved to results/forecast_vs_actuals.png")