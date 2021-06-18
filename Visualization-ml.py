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
#Box and Whisker Plots
from pandas import read_csv
from matplotlib import pyplot

headers = [
    'preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'
]
data = read_csv('diabetes-india.csv', names=headers)
print(data.head(10))
data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
pyplot.show()

#Scatter Matrix Plot
from pandas.plotting import scatter_matrix

scatter_matrix(data)
pyplot.show()

#%%
#Correlation Matrix Plot
from pandas import read_csv
from matplotlib import pyplot

headers = [
    'preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'
]
data = read_csv('diabetes-india.csv', names=headers)
print(data.head(10))
correlation = data.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation, vmin=-1, vmax=+1)
fig.colorbar(cax)

ax.set_xticklabels(headers)
ax.set_yticklabels(headers)
pyplot.show()
# %%
