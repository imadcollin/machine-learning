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

# Binarization
from sklearn.preprocessing import Binarizer

binary = Binarizer(threshold=0.5).fit(data)
binary_data = binary.transform(data)
print("\nBinarized data:\n", binary_data)

# Sttandardization
from sklearn.preprocessing import StandardScaler

data_scaler = StandardScaler().fit(data)
rescaled_Data = data_scaler.transform(data)
print("\nStandaarized data:\n", rescaled_Data)

# Encoder
# %%
import numpy as np
from sklearn import preprocessing

labels = ['1', '2', '2', '3', '4', '4', '5', '5', '6']
encoder = preprocessing.LabelEncoder()
encoder.fit(labels)
test_label = ['1', '2', '5']
encode_values = encoder.transform(test_label)
print("\nLabels: ", test_label)
print("\nEncoded Values", list(encode_values))

# Decoder
decode_list = encoder.inverse_transform([0, 1, 4])
print("\nDecoded Values", list(decode_list))

# %%
