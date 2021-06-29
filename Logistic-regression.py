#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

iris_data = datasets.load_iris()
x = iris_data.data[:, :2]
y = (iris_data.target != 0) * 1
plt.figure(figsize=(6, 6))
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color='g', label='0')
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color='y', label='1')

plt.legend()
# %%
