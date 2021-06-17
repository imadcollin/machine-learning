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
