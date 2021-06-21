#%%
#Scaling
from pandas import read_csv
from numpy import set_printoptions
from sklearn import preprocessing

names = [
    'preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'
]
data = read_csv('diabetes-india.csv', names=names)
print(data.values)

data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_rescaled = data_scaler.fit_transform(data)
print(data_rescaled)