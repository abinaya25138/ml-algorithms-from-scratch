"""
Multiclass classification on the Iris dataset using an SVM wrapped in a
One-vs-Rest strategy (trains one binary classifier per class vs. all
others) - compare against the One-vs-One version on the same data.
"""
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.80, test_size=0.20, random_state=101
)

model = OneVsRestClassifier(SVC())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
