#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 01:01:59 2022

@author: alivaliyev
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
#Metrics
from sklearn.metrics import accuracy_score

#Model Select
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

train=pd.read_csv('model_TVAE_for_Titanic.csv')
#Removing the unneeded column
ports = pd.get_dummies(train.Embarked , prefix='Embarked')
train = train.join(ports)
train.drop(['Embarked'], axis=1, inplace=True)
train.Sex = train.Sex.map({'male':0, 'female':1})
y = train.Survived.copy()
X = train.drop(['Survived'], axis=1) 
X.drop(['Cabin'], axis=1, inplace=True) 
X.drop(['Ticket'], axis=1, inplace=True) 
X.drop(['Name'], axis=1, inplace=True) 
X.drop(['PassengerId'], axis=1, inplace=True)
X.Age.fillna(X.Age.median(), inplace=True)

#Applying machine learning techniques

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 5)



# Random Forest:


random_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
random_forest.fit(X_train, y_train)
Y_prediction = random_forest.predict(X_test)
accuracy_rf=round(accuracy_score(y_test,Y_prediction)* 100, 2)


#Logistic regression

model = LogisticRegression(max_iter = 500000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_lr=round(accuracy_score(y_test,y_pred)* 100, 2)


# K Nearest Neighbor

classifier=KNeighborsClassifier()
classifier.fit(X_train,y_train)
Y_pred=classifier.predict(X_test) 
accuracy_knn=round(accuracy_score(y_test,Y_pred)* 100, 2)



#Naive Bayes

classifier=GaussianNB()
classifier.fit(X_train,y_train)
Y_pred=classifier.predict(X_test)
accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)


#SVM

svc = SVC() 
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
accuracy_svc=round(accuracy_score(y_test,Y_pred)* 100, 2)


#accuracy table

results = pd.DataFrame({
    'Model': [ 'KNN', 
              'Logistic Regression', 
              'Random Forest',
              'Naive Bayes',  
              ' Support Vector Machine'],
    "Accuracy_score":[accuracy_knn,
                      accuracy_lr,
                      accuracy_rf,
                      accuracy_nb,
                      accuracy_svc
                     ]})
result_df = results.sort_values(by='Accuracy_score', ascending=False)
result_df = result_df.reset_index(drop=True)
result_df.head(9)