#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from sklearn.metrics import accuracy_score

sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = svm.SVC(kernel='rbf', C=10000.0, gamma='auto')
t0 = time()
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
predx = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
# print "prediction for 10th elem: ", predx[10], " 26th elem: ", predx[26], "50th elem: ", predx[50]
count = 0
for label in predx:
    if(label == 1):
        count = count+1
print "chris has ", count, "events"
accuracy = accuracy_score(labels_test, predx)
print(accuracy)
