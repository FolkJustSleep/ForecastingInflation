from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preparedata_to_forcast as prep

def test_adf_stationarity(df):
    # Perform ADF test - clean data first
    y_series = df['y'].dropna()
    y_series = y_series[~np.isinf(y_series)]
    
    if len(y_series) < 4:
        print("Warning: Insufficient data for ADF test after cleaning")
        return None
    
    result = adfuller(y_series)
    
    # Extract p-value
    p_value = result[1]
    print(f"ADF Test p-value for Inflation Data: {p_value}")
    if p_value < 0.05:
        print("Time series is stationary (passed ADF test)")
    else:
        print("Time series is non-stationary (failed ADF test)")
    return p_value

def stationary_data(train_df, test_df):
    # Apply log1p only to numeric columns, preserve datetime columns
    train_df_log = train_df.copy()
    test_df_log = test_df.copy()
    
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    
    # Apply log1p to numeric columns
    for col in numeric_cols:
        train_df_log[col] = np.log1p(train_df[col])
        test_df_log[col] = np.log1p(test_df[col])
    
    # Replace inf/-inf with NaN, then drop rows with NaN in numeric columns
    train_df_log = train_df_log.replace([np.inf, -np.inf], np.nan)
    test_df_log = test_df_log.replace([np.inf, -np.inf], np.nan)
    train_df_log = train_df_log.dropna(subset=numeric_cols)
    test_df_log = test_df_log.dropna(subset=numeric_cols)
    
    fig = plt.figure(figsize=(12,4))
    plt.plot(train_df_log['ds'], train_df_log['y'], label='log1p(train)')
    plt.legend(); plt.xticks(rotation=45); 
    plt.title('log1p transform (train)')
    plt.tight_layout();
    plt.savefig('results/figure/log1p_transform_train.png', dpi=300, bbox_inches='tight')
    return train_df_log, test_df_log

if __name__ == "__main__":
    inflation_df, foodprice_df = prep.load_data()
    country_list = inflation_df['Country'].unique().tolist()
    print("Available countries for stationarity test:")
    for country in country_list:
        print(country)
    country = input("Enter country name for stationarity test: ")
    df_with_regressor = prep.add_food_price_regressor(inflation_df, foodprice_df, country=country)
    prophet_df = prep.prepare_prophet_data(inflation_df, country=country)
    p_value = test_adf_stationarity(prophet_df)
    