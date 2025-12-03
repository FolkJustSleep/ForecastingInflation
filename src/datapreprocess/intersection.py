import pickle


FOOD_DIR = 'foodprice.pickle'
f = open(FOOD_DIR, 'rb')
food_data = pickle.load(f)
f.close()

Salary_DIR = 'salary.pickle'
f = open(Salary_DIR, 'rb')
salary_data = pickle.load(f)
f.close()

Inflation_DIR = 'inflation.pickle'
f = open(Inflation_DIR, 'rb')
inflation_data = pickle.load(f)
f.close()

# print("Inflation Data:", inflation_data)
# print("Food Data", food_data['country'])
# print("Salary Data Countries:", salary_data['Country'])


common_countries = set(food_data['country']).intersection(set(salary_data['Country'])).intersection(set(inflation_data['Country']))
print(f"Total common countries: {len(common_countries)}")
print(f"Common Countries: {common_countries}")

common_years = set(food_data['year']).intersection(set(inflation_data['Year']))
print(f"Total common years: {len(common_years)}")
# print(f"Common Years: {common_years}")

Selected_country_data = []
Selected_year_data = []
for country in common_countries:
    Selected_country_data.append(country)
    
for year in common_years:
    Selected_year_data.append(year)
    
Selected_year_data.sort()
print(f"Sorted Common Years: {Selected_year_data}")

f = open('common_years_and_countries.pickle', 'wb')
pickle.dump({'Country': Selected_country_data,'Year': Selected_year_data}, f)
f.close()
print("Common years saved to common_years_and_countries.pickle")

