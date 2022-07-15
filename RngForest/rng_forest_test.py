import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from RngForest.rng_forest import RandomForest


def accuracy(y_true, y_predicted):
    acc = np.sum(y_true == y_predicted) / len(y_true)
    return acc


data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

clf = RandomForest(n_trees=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy", acc)
