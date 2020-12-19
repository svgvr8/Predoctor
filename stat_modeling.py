# MODELING DATA
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

file_path= "../input/cam.csv"
diab_data = pd.read_csv(file_path)

diab_data.head()
y = diab_data.Outcome
features = ['Pregnancies','Glucose','Bloodpressure','SkinThickness','Insulin','BMI','Diabetes','Age']
X = diab_data[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = LogisticRegression()
my_model.fit(train_X,train_y)
prediction=my_model.predict(val_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,val_y))
my_model=DecisionTreeClassifier()
my_model.fit(train_X,train_y)
prediction=my_model.predict(val_X)
print('The accuracy of the DecisionTree is',metrics.accuracy_score(prediction,val_y))

types=['rbf','linear']
for i in types:
    my_model=svm.SVC(kernel=i)
    my_model.fit(train_X,train_y)
    prediction=my_model.predict(val_X)
    print('Accuracy for SVM kernel=',i,'is',metrics.accuracy_score(prediction,val_y))

a_index=list(range(1,11))
a=pd.Series()
x=[0,1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    my_model=KNeighborsClassifier(n_neighbors=i) 
    my_model.fit(train_X,train_y)
    prediction=my_model.predict(val_X)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,val_y)))
plt.plot(a_index, a)
plt.xticks(x)
plt.show()
print('Accuracies for different values of n are:',a.values)

abc=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','Decision Tree']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),DecisionTreeClassifier()]
for i in models:
    my_model = i
    my_model.fit(train_X,train_y)
    prediction=my_model.predict(val_X)
    abc.append(metrics.accuracy_score(prediction,val_y))
models_dataframe=pd.DataFrame(abc,index=classifiers)   
models_dataframe.columns=['Accuracy']
models_dataframe

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', LogisticRegression()),
])

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:", scores)
print("Average MAE score (across experiments):")
print(scores.mean())

linear_svc=svm.SVC(kernel='linear',C=0.1,gamma=10,probability=True)
lr=LogisticRegression(C=0.1)

ensemble_lin_lr=VotingClassifier(estimators=[('Linear_svm', linear_svc), ('Logistic Regression', lr)], 
                       voting='soft', weights=[2,1]).fit(train_X,train_y)
print('The accuracy for Linear SVM and Logistic Regression is:',ensemble_lin_lr.score(val_X,val_y))


