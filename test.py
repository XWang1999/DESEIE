from dso import DeepSymbolicRegressor
import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')
X = np.array(data.iloc[:,0:2])
y = np.array(data.iloc[:,2])
y = y.reshape(16,)

model = DeepSymbolicRegressor("dso/config/config_regression.json") # Alternatively, you can pass in your own config JSON path

model.fit(X, y)


