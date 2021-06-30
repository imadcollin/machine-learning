#%%
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import data
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
# Multinomial Logistic Regression Model
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.4,
                                                    random_state=1)

digreg = linear_model.LogisticRegression()
digreg.fit(x_train, y_train)
y_pred = digreg.predict(x_test)
print("\n Logistic regrassion model:  ",
      metrics.accuracy_score(y_test, y_pred) * 100)
