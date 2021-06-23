#%%
# Univariate Selection
import numpy as np
from pandas import read_csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from numpy import set_printoptions

names = [
    'preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'
]
data = read_csv('diabetes-india.csv', names=names)
print("\Values:\n", data[:10])

x = data.values[:, 0:8]
y = data.values[:, 8]
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(x, y)
set_printoptions(precision=2)
print("\nScores\n", fit.scores_)

featured_data = fit.transform(x)
print("\nFeatured Data\n", featured_data[0:10])

# %%
