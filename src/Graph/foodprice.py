import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FilteredFoodPrice = pd.read_csv('filtered_foodprice.csv')

country_list = np.asarray(FilteredFoodPrice['Country'].unique())
Afghanistan_food_price = FilteredFoodPrice['Avg.FoodPrice'].loc[FilteredFoodPrice['Country'] == country_list[0]]
foodprice_year = FilteredFoodPrice['Year'].unique()