import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "data/csv/inflation_pre.csv"
FilteredInflation = pd.read_csv(DATA_DIR)

country_list = np.asarray(FilteredInflation['Country'].unique())

# Afghanistan_inflation_avg = FilteredInflation['Avg.Inflation'].loc[FilteredInflation['Country'] == country_list[0]]

inflation_year = FilteredInflation['Year'].unique()


print(f"Inflation Countries: {country_list}")
# print(f"Inflation Averages: {Afghanistan_inflation_avg}")
plt.figure(figsize=(12,6))
for i in range(len(country_list)):
    inflation_avg = FilteredInflation['Avg.Inflation'].loc[FilteredInflation['Country'] == country_list[i]]
    plt.plot(inflation_year, inflation_avg, label=country_list[i])


plt.legend()
plt.xlabel('Year')
plt.ylabel('Average Inflation') 
plt.title('Average Inflation of Afghanistan Over Years')
plt.tight_layout()
plt.savefig('results/average_inflation_over_years.png', dpi=300, bbox_inches='tight')
