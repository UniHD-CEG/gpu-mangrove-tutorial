#!/usr/bin/env python3

import sys
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def ape(y_true, y_pred):
    return np.abs(y_pred - y_true)/y_true

def se(y_true, y_pred):
    return (y_true - y_pred)**2

def ae(y_true, y_pred):
    return np.abs(y_true - y_pred)

if(len(sys.argv) != 2):
    print("Usage:\n\t" + sys.argv[0] + " <samples.db>")
    exit(-1)

print("Reading samples..")
conn = sqlite3.Connection(sys.argv[1])
samples = pd.read_sql_query(
    'select * from samples',
    conn, index_col=['bench','app','dataset','name'])

# Samples can be filtered by index
# This example get all entry in which the index 1 (app) is equal to 'kmeans'
# samples[samples.index.get_level_values(1) == 'kmeans']

X = samples[samples.columns[1:-1]]
y = samples['time']

print("Dividing into train and test set..")
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=17)

# Referr to https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor

###### This section could be solved by the students
regr = RandomForestRegressor(n_estimators=100,
                             criterion='mse',
                             max_depth=None,
                             random_state=17)

##### End of Section

print("Train model..")
regr.fit(X_train, y_train)
y_predict = regr.predict(X_test)
print("Predicting:")
results = pd.DataFrame()
results['y_true'] = y_test
results['y_predict'] = regr.predict(X_test)
results['mse'] = se(y_test, results['y_predict'])
results['mae'] = ae(y_test, results['y_predict'])
results['mape'] = ape(y_test, results['y_predict'])
print(results)
print('Mean errors:')
print(results.mean())
