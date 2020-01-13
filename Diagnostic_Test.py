import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv

# --- Load Data ---
dataset = pandas.read_csv('Wisconsin Breast Cancer Dataset (Diagnostic).csv')

# --- Data Processing ---
dataset.drop('id', axis=1, inplace=True)
dataset.dropna(axis=0, inplace=True)

Y = dataset[['diagnosis']]
X = dataset[['radius_4ean']]

# --- Data Visualization ---
plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr())
ax = plt.gcf()
plt.ylim((10, 0))
plt.show()

# --- Rescaling Datasets ---
sc = StandardScaler()
X = sc.fit_transform(X)

# --- CSV rows ---
rows = [[
    'Model',
    'Mean Accuracy',
    'Mean F1 Score',
    'Mean Confusion Matrix F-Measure',
    'Mean Absolute Error',
    'Mean Squared Error',
    'Average Accurate Classification',
    'Average Inaccurate Classifications',
    'Total Instances']]

# --- Training models ---
models = [('SVM', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                      decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
                      max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)),
          ('KNN', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                       metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                                       weights='uniform')),
          ('NB', GaussianNB(priors=None, var_smoothing=1e-09))]

results = []
names = []
folds = 7

for name, model in models:
    kfold = model_selection.KFold(n_splits=folds, random_state=folds)
    accuracy = model_selection.cross_val_score(model, X, np.ravel(Y), cv=kfold, scoring='accuracy')
    MAE = model_selection.cross_val_score(model, X, np.ravel(Y), cv=kfold, scoring='neg_mean_absolute_error')
    MSE = model_selection.cross_val_score(model, X, np.ravel(Y), cv=kfold, scoring='neg_mean_squared_error')
    total_instances = len(X)
    correct_instances = total_instances * accuracy
    incorrect_instances = total_instances - correct_instances
    f1 = model_selection.cross_val_score(model, X, np.ravel(Y), cv=kfold, scoring='f1_macro')
    recall = model_selection.cross_val_score(model, X, np.ravel(Y), cv=kfold, scoring='recall_macro')
    precision = model_selection.cross_val_score(model, X, np.ravel(Y), cv=kfold, scoring='precision_macro')
    fmeasure = 2 * ((recall.mean() * precision.mean()) / (recall.mean() + precision.mean()))

    rows.append([
        name,
        "{0:.6f}".format(accuracy.mean()),
        "{0:.6f}".format(f1.mean()),
        "{0:.6f}".format(fmeasure),
        "{0:.6f}".format(MAE.mean() * -1),
        "{0:.6f}".format(MSE.mean() * -1),
        int(round(correct_instances.mean())),
        int(round(incorrect_instances.mean())),
        len(X)
    ])

new_file = open('results (diagnostic).csv', 'w', newline='')
csv_output = csv.writer(new_file)
csv_output.writerows(rows)
