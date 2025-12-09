import numpy as np
import pandas as pd


DATA_DIR = "data/csv/inflation_pre.csv"

FilteredInflation = pd.read_csv(DATA_DIR)

years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
df = FilteredInflation[FilteredInflation['Year'].isin(years)]
df.to_csv("data/csv/clean_inflation.csv", index=False)