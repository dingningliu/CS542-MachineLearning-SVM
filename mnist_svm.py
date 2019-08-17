#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:46:01 2019

@author: liudingning
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import operator
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

"multi-class classification"
mnist_train = pd.read_csv('/Users/liudingning/Desktop/sum/ml542/hw4/ps4_python/mnist_train.csv', header = None)
mnist_test = pd.read_csv('/Users/liudingning/Desktop/sum/ml542/hw4/ps4_python/mnist_test.csv', header = None)

#seperate the feature and target
Y_train = mnist_train[ mnist_train.columns[0]]
X_train = mnist_train[ mnist_train.columns[1:]]
Y_test = mnist_test[ mnist_test.columns[0]]
X_test = mnist_test[ mnist_test.columns[1:]]




"=============one versus one================"
#fitting the data
clf_ovo = svm.SVC(gamma=0.001, decision_function_shape='ovo', kernel='linear')
clf_ovo.fit(X_train,Y_train)

#using the model to predict new values

predict_ovo = clf_ovo.predict(X_test)
accurate_ovo = np.sum(predict_ovo == Y_test)/len(Y_test)
conf_ovo = confusion_matrix(predict_ovo, Y_test)


"=============one versus rest================"
"==linear kernel=="
clf_ovr = svm.LinearSVC()
clf_ovr.fit(X_train,Y_train)
#predict
predict_ovr = clf_ovr.predict(X_test)

#prediction accuracy and confusion matrix
accurate_ovr = np.sum(predict_ovr == Y_test)/len(Y_test)
conf_ovr = confusion_matrix(predict_ovr, Y_test)



