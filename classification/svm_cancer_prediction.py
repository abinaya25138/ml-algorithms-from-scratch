"""
Breast cancer classification with a linear SVM, using scikit-learn's
bundled breast cancer dataset. Reports accuracy/precision/recall, a
confusion matrix, and plots an ROC curve.
"""
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.metrics import RocCurveDisplay, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

cancer = datasets.load_breast_cancer()
print("Features:", cancer.feature_names)
print("Labels:", cancer.target_names)
print("Data shape:", cancer.data.shape)

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=109
)

clf = svm.SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)


def plot_roc_curve(y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    display.figure_.set_size_inches(5, 5)
    plt.plot([0, 1], [0, 1], color="g", linestyle="--")
    plt.title(title)


y_proba = clf.predict_proba(X_test)
plot_roc_curve(y_test, y_proba[:, 1], "ROC curve (predicted probabilities)")
plt.show()
