#%%
import numpy as np
import csv
import pandas as pd
with open("data.csv", 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    data = list(reader)
    data = np.array(data).astype(str)
print(data)

## Reading with pandas
print("-------------\n")
data = pd.read_csv("data.csv")
print(data)

