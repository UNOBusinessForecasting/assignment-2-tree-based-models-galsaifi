# %%
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

train = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv')
test = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv')

# %%
import numpy as np
import patsy as pt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

y = train['meal']
x = train.drop(['meal', 'id', 'DateTime'], axis = 1)

model = DecisionTreeClassifier()

# x, xt, y, yt = train_test_split(x, y, test_size=0.33, random_state=42)

modelFit = model.fit(x,y)

# %%
xt = test.drop(['meal', 'id', 'DateTime'], axis=1)

pred = modelFit.predict(xt)
