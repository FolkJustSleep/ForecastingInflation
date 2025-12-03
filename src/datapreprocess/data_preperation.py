import pickle
import numpy as np
import pandas as pd

f = open('common_years_and_countries.pickle', 'rb')
common_data = pickle.load(f)
f.close()

selected_countries = np.asarray(common_data['Country'])
selected_years = np.asarray(common_data['Year'])

# print(f"Selected Countries: {selected_countries}")
# print(f"Selected Years: {selected_years}")

df = pd.read_csv('salary_data.csv')

filtered_df = df[df['country_name'].isin(selected_countries)]

filtered_df.rename(columns={'country_name': 'Country'}, inplace=True)

filtered_df.to_csv('filtered_salary_data.csv', index=False)