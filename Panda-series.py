#%%
import pandas as pd
import numpy as np

data = np.array(['a', 'b', 'c', 'd'])
print(data)

s = pd.Series(data)
print(s)

#%%
# Raw data
from pandas import read_csv

csv_data = read_csv('data.csv')
print(csv_data)

# Header names
headernames = ['No', 'N', 'S', 'Date', 'Dept']
csv_header = read_csv('data.csv', names=headernames)
print(csv_header)

# Dimension
print(csv_header.shape)

# Data attributes
print(csv_header.dtypes)

# %%
# Statiscal Summary of Data
from pandas import read_csv
from pandas import set_option

data = read_csv('data.csv')
salaries = data['salary']

set_option('display.width', 100)
set_option('precision', 2)
print(salaries.shape)
print(salaries.describe())

# Skew
print(salaries.skew())

# Class Distribution
countDept = data.groupby('dept').size()
print('\n', countDept)
# %%
