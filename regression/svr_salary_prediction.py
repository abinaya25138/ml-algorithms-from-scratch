"""
Support Vector Regression (SVR) to predict salary from job position
level, on the classic "Position Salaries" dataset - a good illustration
of SVR fitting a non-linear relationship with a small dataset.

Requires: data/Position_Salaries.csv (included in this repo).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
dataset = pd.read_csv(DATA_DIR / "Position_Salaries.csv")
print(dataset)

X = dataset.iloc[:, 1:-1].values  # Level
y = dataset.iloc[:, -1].values.reshape(-1, 1)  # Salary

# SVR is sensitive to feature scale, so standardize both X and y
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

plt.scatter(X_scaled, y_scaled, color="red")
plt.title("Scaled Level vs. Salary")
plt.xlabel("Level (scaled)")
plt.ylabel("Salary (scaled)")
plt.show()

regressor = SVR(kernel="rbf")
regressor.fit(X_scaled, y_scaled.ravel())

# Predict the salary for level 6.5
pred_scaled = regressor.predict(scaler_X.transform([[6.5]])).reshape(-1, 1)
pred = scaler_y.inverse_transform(pred_scaled)
print("Predicted salary at level 6.5:", pred[0][0])

# Plot the fitted curve back on the original (unscaled) axes
plt.scatter(scaler_X.inverse_transform(X_scaled), scaler_y.inverse_transform(y_scaled), color="red")
plt.plot(
    scaler_X.inverse_transform(X_scaled),
    scaler_y.inverse_transform(regressor.predict(X_scaled).reshape(-1, 1)),
    color="blue",
)
plt.title("Support Vector Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
