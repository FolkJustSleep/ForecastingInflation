import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data/csv/filtered_foodprice.csv')

country_list = df['Country'].unique()
filtered_df = pd.DataFrame(columns=['Country', 'Year', 'avg_food_price'])
mean_open_prices = df.groupby(['Country','Year'])['Open_price'].mean().reset_index()
mean_close_prices = df.groupby(['Country','Year'])['Close_price'].mean().reset_index()
mean_high_prices = df.groupby(['Country','Year'])['Highest_price'].mean().reset_index()
mean_low_prices = df.groupby(['Country','Year'])['Lowest_price'].mean().reset_index()

finally_df = mean_open_prices.merge(mean_close_prices, on=['Country','Year'])
finally_df = finally_df.merge(mean_high_prices, on=['Country','Year'])
finally_df = finally_df.merge(mean_low_prices, on=['Country','Year'])

plt.figure(figsize=(12,4))
plt.xlabel('Year')
plt.ylabel('Average Food Price') 
plt.title('Average Food Price Over Years for Different Countries')
plt.legend(country_list)
plt.tight_layout()
# print(finally_df.head(2))
rows = []
for country in country_list:
    country_data = finally_df[finally_df['Country'] == country]
    country_data['avg_food_price'] = (country_data['Highest_price'] + country_data['Lowest_price'] + country_data['Close_price']) / 3 # Average of high, low, close price (HLC)
    rows.append(country_data[['Country', 'Year', 'avg_food_price']])
    plt.plot(country_data['Year'], country_data['avg_food_price'], label=country)
    
filtered_df = pd.concat(rows)
print(filtered_df.head(10))

plt.savefig('average_food_price_over_years.png', dpi=300, bbox_inches='tight')
filtered_df.to_csv('data/csv/filtered_foodprice_mean_1.csv', index=False)