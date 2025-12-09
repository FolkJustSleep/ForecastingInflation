import pandas as pd

from statsmodels.tsa.stattools import adfuller # For ADF test

def load_data():
    df_Inflation = pd.read_csv('data/csv/inflation_pre.csv')
    foodprice_df = pd.read_csv('data/csv/filtered_foodprice_mean_1.csv') 
    return df_Inflation, foodprice_df

def prepare_prophet_data(df, country):
    data = df[df['Country'] == country].copy()
    data['ds'] = pd.to_datetime(data['Year'], format='%Y')
    if 'Avg.Inflation' not in data.columns:
        data['y'] = data['avg_food_price'] = (df['Open_price'] + df['Close_price']) / 2
    else:
        data['y'] = data['Avg.Inflation'] 
    prophet_df = data[['ds', 'y']].sort_values('ds').reset_index(drop=True)
    return prophet_df

def add_food_price_regressor(inflation_df, foodprice_df, country):
    """Add food price as external regressor to improve forecast accuracy"""
    # Get food price data for the country
    food_data = foodprice_df[foodprice_df['Country'] == country].copy()
    food_data['ds'] = pd.to_datetime(food_data['Year'], format='%Y')
    
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
    merged_df['avg_food_price'].ffill(inplace=True)
    merged_df['avg_food_price'].bfill(inplace=True)
    
    merged_df.to_csv('data/csv/inflation_with_foodprice_regressor.csv', index=False)
    
    return merged_df

if __name__ == "__main__":
    inflation_df, foodprice_df = load_data()
    df_prophet = prepare_prophet_data(inflation_df)
    print("Inflation : ")
    print(df_prophet.head(2))
    print("Food Price : ")
    print(foodprice_df.head(2))