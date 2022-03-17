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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


df=pd.read_csv('synthetic_data_for_adult_dataset_correlated_mode.csv')
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
df['workclass'] = df['workclass'].fillna('X')
df['occupation'] = df['occupation'].fillna('X')
df['native-country'] = df['native-country'].fillna('X')
#Map Sex as a binary column
df['gender'] = df.gender.map({'Male':0, 'Female':1})

#Married can be converted manually to binary columns
#No spouse means Single
df['marital-status'] = df['marital-status'].replace(['Never-married', 'Divorced', 'Separated', 'Widowed'], 'Single')
#Spouse means Married
df['marital-status'] = df['marital-status'].replace(['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married')
#Now I map it to Binary values
df['marital-status'] = df['marital-status'].map({'Married':1, 'Single':0})
df.drop(labels=["workclass","education","occupation","relationship","race","native-country"], axis = 1, inplace = True)

X = df.drop(labels=['income'], axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_lr=round(accuracy_score(y_test, y_pred)* 100, 2)


#K Nearest Neighbor

knn = KNeighborsClassifier(n_neighbors = 20)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test) 
accuracy_knn=round(accuracy_score(y_test,Y_pred)* 100, 2)


#Naive Bayes


gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test) 
accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)


#SVM


linear_svc = LinearSVC() 
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
accuracy_svc=round(accuracy_score(y_test,Y_pred)* 100, 2)


# Random Forest:


random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(X_train, y_train)
Y_prediction = random_forest.predict(X_test)
accuracy_rf=round(accuracy_score(y_test,Y_prediction)* 100, 2)


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