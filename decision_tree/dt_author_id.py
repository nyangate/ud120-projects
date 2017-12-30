#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score
# from class_vis import prettyPicture
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
print(len(features_train[0]))
predict = clf.predict(features_test)
accuracy = accuracy_score(labels_test, predict)
print(accuracy)
#########################################################
### your code goes here ###


#########################################################
