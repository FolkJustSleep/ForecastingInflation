import pandas as pd
import matplotlib.pyplot as plt
import preparedata_to_forcast as prep
import stationary_test as statest


from prophet import Prophet
import src.stationary_test as statest
import src.preparedata_to_forcast as prep
import src.train_forecast_model as trainmodel

choice = input("Choose operation - 'Single', 'Multiple': ")
if choice.lower() == 'Single':
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
    train_df, test_df = trainmodel.split_data(df_with_regressor, train_ratio=0.8)
    trainmodel.plot_split_data(train_df, test_df)
    forecast = pd.DataFrame()
    if p_value > 0.05:
        # made the data stationary
        train_df_log, test_df_log = statest.stationary_data(train_df, test_df)
        #Train model with food price regressor
        forecast = trainmodel.train_prophet_model(train_df_log, test_df_log, use_food_price=True)
        print("Forecasting completed.")
    else:
        # Train model without food price regressor
        forecast = trainmodel.train_prophet_model(train_df, test_df, use_food_price=False)
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
    plt.savefig('forecast_vs_actuals.png', dpi=300, bbox_inches='tight')
    print("Forecast vs Actuals plot saved to forecast_vs_actuals.png")
if choice.lower() == 'multiple':
    inflation_df, foodprice_df = prep.load_data()