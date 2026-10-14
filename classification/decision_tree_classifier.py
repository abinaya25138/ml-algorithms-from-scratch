"""
Decision tree classifier on the "Social Network Ads" dataset, for
comparison against the KNN classifier on the same data.

Requires: data/Social_Network_Ads.csv
"""
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("../data/Social_Network_Ads.csv")
X = dataset.iloc[:, [1, 2, 3]].values
y = dataset.iloc[:, -1].values

le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

print("Confusion matrix:\n", cm)
print("Accuracy:", ac)
