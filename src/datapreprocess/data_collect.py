# import numpy as np 
import pandas as pd
import pickle

DATADIR = 'salary_data.csv'
data = pd.read_csv(DATADIR)

unique_countries = data['country_name'].unique()
# date = data['mp_year'].unique()


country_list = []

for country in unique_countries:
    country_list.append(country)
    
# year_list = []
# for yr in date:
#     year_list.append(int(yr))
    
# year_list.sort()
print(f"Total countries: {len(country_list)}")
print(f"Countries: {country_list}")
# print(f"Total years: {len(year_list)}")
# print(f"Years: {year_list}")

f = open('data.pickle', 'wb')
pickle.dump({'Country': country_list}, f)
f.close()
print("Data saved to data.pickle")