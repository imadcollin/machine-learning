#%%
from typing import Sized
import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
print(label_names)
print(feature_names[0])
print('\nfeatue:', features[1])

# Organizing data into training & testing sets
from sklearn.model_selection import train_test_split

train, test, train_labeles, test_labels = train_test_split(features,
                                                           labels,
                                                           test_size=0.40,
                                                           random_state=42)
# Model evaluation
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
model = gnb.fit(train, train_labeles)

pred = gnb.predict(test)
print("\nPred:", pred)

# Finding accuracy
from sklearn.metrics import accuracy_score

a_c = accuracy_score(test_labels, pred)
print("\nAccurecy:", a_c)
# %%
