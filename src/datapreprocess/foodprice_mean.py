import pandas as pd

df = pd.read_csv('data/csv/filtered_foodprice.csv')
df_yearly = df.groupby(
    ['Country', 'Year']
)[['Open_price', 'Close_price', 'Highest_price', 'Lowest_price']].mean().reset_index()

df.to_csv("data/csv/filtered_foodprice_mean.csv", index=False)