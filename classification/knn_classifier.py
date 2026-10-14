"""
K-Nearest Neighbors classifier on the classic "Social Network Ads"
dataset (predicting whether a user clicked an ad from age/salary/gender).

Requires: data/Social_Network_Ads.csv
(a well-known ML-course dataset - download from Kaggle).
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
dataset = pd.read_csv(DATA_DIR / "Social_Network_Ads.csv")
X = dataset.iloc[:, [1, 2, 3]].values
y = dataset.iloc[:, -1].values

# Encode the categorical gender column
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature scaling so no single feature dominates the distance metric
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

print("Confusion matrix:\n", cm)
print("Accuracy:", ac)
