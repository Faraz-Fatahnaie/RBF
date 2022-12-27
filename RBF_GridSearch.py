import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import random
import math


def generateSinPairs(valueRange, N=100):
    X = np.zeros((N, 1))
    Y = np.zeros((N))
    for i in range(N):
        X[i][0] = (np.random.ranf() * valueRange - (0.5 * valueRange)) * math.pi
        # Y[i][0] = X[i][0] * X[i][0] * abs(math.sin(X[i][0]*math.pi))
        Y[i] = math.sin(X[i][0])
        pass
    return X, Y


(train_X, train_Y) = generateSinPairs(2, N=200)
print(np.shape(train_X), np.shape(train_Y))

parameters = {'kernel': ('linear', 'rbf'), 'C': np.random.uniform(4, 10, 3)}
svr = svm.SVR()
clf = GridSearchCV(svr, parameters)
clf.fit(train_X, train_Y)

print(clf.score(train_X, train_Y))
#print(sorted(clf.cv_results_.keys()))

params = clf.get_params()
print(params)
print('Kernel:', params['estimator__kernel'], 'C:', params['estimator__C'])

