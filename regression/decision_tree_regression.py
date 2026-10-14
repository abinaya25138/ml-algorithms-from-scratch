"""
Decision tree regression predicting petrol consumption from tax,
income, highway miles, and license-holder ratio (classic
"Petrol Consumption" dataset), evaluated with MAE/MSE/RMSE.

Requires: data/petrol_consumption.csv
(download from Kaggle - search "petrol consumption dataset").
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
dataset = pd.read_csv(DATA_DIR / "petrol_consumption.csv")
print(dataset.head())
print(dataset.describe())

X = dataset.drop("Petrol_Consumption", axis=1)
y = dataset["Petrol_Consumption"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(comparison)

print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
