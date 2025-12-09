import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FilteredFoodPrice = pd.read_csv('data/csv/filtered_foodprice_mean_1.csv')

country_list = np.asarray(FilteredFoodPrice['Country'].unique())
foodprice_year = FilteredFoodPrice['Year'].unique()

plt.figure(figsize=(12,6))
for i in range(len(country_list)):
    year = FilteredFoodPrice['Year'].loc[FilteredFoodPrice['Country'] == country_list[i]]
    foodprice_avg = FilteredFoodPrice['avg_food_price'].loc[FilteredFoodPrice['Country'] == country_list[i]]
    plt.plot(year, foodprice_avg, label=country_list[i])


plt.legend()
plt.xlabel('Year')
plt.ylabel('Average Food Price') 
plt.title('Average Food Price Over Years')
plt.tight_layout()
plt.savefig('results/average_food_price_over_years.png', dpi=300, bbox_inches='tight')