# %%
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

train = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv')
test = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv')

# %%
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

y = train['meal']
x = train.drop(['meal', 'id', 'DateTime'], axis = 1)

model = DecisionTreeClassifier(min_samples_leaf=2, random_state = 42)

x, xt, y, yt = train_test_split(x, y, test_size=0.4, random_state=42)

modelFit = model.fit(x,y)

# %%
pred = modelFit.predict(xt)

# %%
print("In-sample accuracy: %s%%" % str(round(100*accuracy_score(y, model.predict(x)), 2)))
print("Out of sample accuracy: %s%%" % str(round(100*accuracy_score(yt, model.predict(xt)), 2)))

# %%
x_test = test.drop(['meal', 'id', 'DateTime'], axis = 1)

pred = modelFit.predict(x_test)
