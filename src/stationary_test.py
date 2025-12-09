from statsmodels.tsa.stattools import adfuller
import pandas as pd
import preparedata_to_forcast as prep

def test_adf_stationarity(df):
    # Prepare dataa
    prophet_df = prep.prepare_prophet_data(df)
    
    # Perform ADF test
    result = adfuller(prophet_df['y'])
    
    # Extract p-value
    p_value = result[1]
    return p_value

if __name__ == "__main__":
    inflation_df, foodprice_df = prep.load_data()
    df_with_regressor = prep.add_food_price_regressor(inflation_df, foodprice_df)
    p_value = test_adf_stationarity(inflation_df)
    print(f"ADF Test p-value for Inflation Data: {p_value}")
    if p_value < 0.05:
        print("Time series is stationary (passed ADF test)")
    else:
        print("Time series is non-stationary (failed ADF test)")