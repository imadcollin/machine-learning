#%%
from pandas import read_csv
from matplotlib import pyplot

headers = [
    'preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'
]
data = read_csv('diabetes-india.csv', names=headers)
print(data.head(10))
data.hist()
pyplot.show()

# %%
#Density Plots
from pandas import read_csv
from matplotlib import pyplot

headers = [
    'preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'
]
data = read_csv('diabetes-india.csv', names=headers)
print(data.head(10))

data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
pyplot.show()
# %%
