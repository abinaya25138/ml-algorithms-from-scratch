# ML Algorithms from Scratch

A set of classic machine learning algorithms implemented with scikit-learn, organized by task. Each script is self-contained and runnable on its own.

## Classification

| Script | Algorithm | Dataset |
|---|---|---|
| `svm_cancer_prediction.py` | SVM (linear kernel) | Breast cancer (bundled with scikit-learn) |
| `svm_heart_disease_prediction.py` | SVM (linear kernel) | Heart Disease (Cleveland + Hungarian) |
| `svm_multiclass_one_vs_one.py` | SVM, One-vs-One multiclass strategy | Iris (bundled with scikit-learn) |
| `svm_multiclass_one_vs_rest.py` | SVM, One-vs-Rest multiclass strategy | Iris (bundled with scikit-learn) |
| `knn_classifier.py` | K-Nearest Neighbors | Social Network Ads |
| `decision_tree_classifier.py` | Decision Tree | Social Network Ads |

## Regression

| Script | Algorithm | Dataset |
|---|---|---|
| `svr_salary_prediction.py` | Support Vector Regression | Position Salaries (included in `data/`) |
| `decision_tree_regression.py` | Decision Tree Regression | Petrol Consumption |

## Setup

```bash
pip install -r requirements.txt
```

Datasets bundled with scikit-learn (breast cancer, Iris) and `Position_Salaries.csv` need no setup. For the others, download the named dataset from Kaggle and drop it into `data/`:

- `Social_Network_Ads.csv` — used by `knn_classifier.py` and `decision_tree_classifier.py`
- `heart_statlog_cleveland_hungary_final.csv` — used by `svm_heart_disease_prediction.py`
- `petrol_consumption.csv` — used by `decision_tree_regression.py`

Then run any script directly, e.g.:

```bash
python classification/knn_classifier.py
```

## Tech stack

Python, scikit-learn, pandas, NumPy, Matplotlib, Seaborn

## Why these two groupings

Classification and regression are the two core supervised-learning tasks — grouping by task (rather than one script per repo) makes the comparisons meaningful: KNN vs. Decision Tree on the same classification dataset, One-vs-One vs. One-vs-Rest on the same multiclass problem, SVR vs. Decision Tree on different regression problems.
