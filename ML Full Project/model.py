# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

file_path = 'F:/Deploy/diabetes.csv'
data=pd.read_csv(file_path)
data.head()

y = data.Outcome
features = ['Age','Glucose','BMI','DiabetesPedigreeFunction','Pregnancies','SkinThickness','Insulin',]
X = data[features]

X_train,X_test,Y_train,Y_test= train_test_split(X,y,test_size=0.25)
#Accuracy #ConfusionMatrix
clf = LogisticRegression()
clf.fit(X_train,Y_train)
pred= clf.predict(X_test)

pickle.dump(clf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
