import pandas as pd
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
dataset = pd.read_csv('Wisconsin Breast Cancer Dataset (Original).csv')

# --- Data Visualization ---

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 32)
print(dataset.describe())
'''
plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr(), annot=True, cmap='cubehelix_r')
ax = plt.gcf()
plt.ylim((10, 0))
plt.show()
'''

# --- Data Processing ---
dataset.drop('id', axis=1, inplace=True)
dataset.dropna(axis=0, inplace=True)

Y = dataset[['class']]
X = dataset[['size_uniformity', 'shape_uniformity', 'marginal_adhesion',
             'epithelial_size', 'bare_nucleoli', 'bland_chromatin',
             'normal_nucleoli']]

# --- Rescaling Datasets ---
sc = StandardScaler()
X = sc.fit_transform(X)

# --- CSV rows ---
rows = [[
    'Model',
    'Mean Accuracy',
    'Mean F1 Score',
    'ROC AUC Score',
    'Explained Variance',
    'Mean Absolute Error',
    'Average Accurate Classification',
    'Average Inaccurate Classifications',
    'Total Instances']]

# --- Training models ---
models = [('SVM', SVC(gamma='scale', probability=True)),
          ('KNN', KNeighborsClassifier(metric='minkowski')),
          ('NB', GaussianNB())]

results = []
names = []
folds = 10

for i in range(folds + 1):
    if i >= 2:
        rows.append(['K-Fold: ' + str(i)])
        for name, model in models:
            kfold = model_selection.KFold(n_splits=i, random_state=i)
            accuracy = model_selection.cross_val_score(model, X, np.ravel(Y), cv=kfold, scoring='accuracy')
            MAE = model_selection.cross_val_score(model, X, np.ravel(Y), cv=kfold, scoring='neg_mean_absolute_error')
            explain_variance = model_selection.cross_val_score(model, X, np.ravel(Y), cv=kfold, scoring='explained_variance')
            total_instances = len(X)
            correct_instances = total_instances * accuracy
            incorrect_instances = total_instances - correct_instances
            f1 = model_selection.cross_val_score(model, X, np.ravel(Y), cv=kfold, scoring='f1_macro')
            roc_auc = model_selection.cross_val_score(model, X, np.ravel(Y), cv=kfold, scoring='roc_auc')

            rows.append([
                name,
                "{0:.3f}".format(accuracy.mean()),
                "{0:.3f}".format(f1.mean()),
                "{0:.3f}".format(roc_auc.mean()),
                "{0:.3f}".format(explain_variance.mean()),
                "{0:.3f}".format(MAE.mean() * -1),
                int(round(correct_instances.mean())),
                int(round(incorrect_instances.mean())),
                len(X)
            ])

new_file = open('results (original).csv', 'w', newline='')
csv_output = csv.writer(new_file)
csv_output.writerows(rows)
