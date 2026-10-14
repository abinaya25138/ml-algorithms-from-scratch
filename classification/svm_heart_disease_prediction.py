"""
Heart disease prediction with a linear SVM, on the UCI/Kaggle
"Heart Disease (Cleveland + Hungarian)" dataset.

Requires: data/heart_statlog_cleveland_hungary_final.csv
(download from Kaggle - search "heart statlog cleveland hungary final").
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
df = pd.read_csv(DATA_DIR / "heart_statlog_cleveland_hungary_final.csv")
print("Shape of DataFrame:", df.shape)
print(df.describe())

# Correlation heatmap between all features
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, fmt=".1f")
plt.title("Feature correlation")
plt.show()

X = np.array(df.drop(["target"], axis=1))
y = np.array(df["target"])

# Standardize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42, test_size=0.2
)

clf = svm.SVC(kernel="linear")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
