#%%
#Scaling
from pandas import read_csv
from numpy import set_printoptions
from sklearn import preprocessing

names = [
    'preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'
]
data = read_csv('diabetes-india.csv', names=names)
print("\Values:\n", data.values)

data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_rescaled = data_scaler.fit_transform(data)
print("\nScaled data:\n", data_rescaled)

## Normalization
from sklearn.preprocessing import Normalizer

data_norm = Normalizer(norm='l1', copy=True).fit(data)
Data_normed = data_norm.transform(data)
print("\nNormalized data L1:\n", Data_normed)
### Normalized with precision
set_printoptions(precision=2)
print("\nNormalized data precision:\n", Data_normed)

#### Normalization L2
data_norm = Normalizer(norm='l2', copy=False).fit(data)
Data_normed = data_norm.transform(data)
print("\nNormalized data L2:\n", Data_normed)
### Normalized with precision
set_printoptions(precision=2)
print("\nNormalized data precision:\n", Data_normed)
