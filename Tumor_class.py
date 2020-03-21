
#BI_RADS_assessment	integer	Definitely benign(1) to Highly suggestive of malignancy (5)
#age	integer	patient's age in years
#shape	integer	mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
#margin	integer	mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
#density	integer	mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
#severity	integer	Predictor Class: benign=0 or malignant=1
print('1 refers to severity of tumor as malignant and 0 refers to benign')
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
dataset=pd.read_csv('C:\Program Files\JetBrains\PyCharm Community Edition 2019.3.1\cancer.csv')
x=np.array(dataset.drop('severity',1))
y=np.array(dataset['severity'])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
scale=StandardScaler()
x_train=scale.fit_transform(x_train)
x_test=scale.transform(x_test)
reg=KNeighborsClassifier(n_neighbors=20,p=2)
reg.fit(x_train,y_train)
y_prediction=reg.predict(x_test)
print('The accuracy of this model is :')
print(accuracy_score(y_test,y_prediction))
print('The confusion matrix for this model is:')
print(confusion_matrix(y_test,y_prediction))
print('The precision of this model is:')
print(f1_score(y_test,y_prediction))
bi_rads=int(input('Enter BI_Rads_Assessment'))
age=int(input('Enter age'))
shape=int(input('Enter shape'))
margin=int(input('Enter margin'))
density=int(input('Enter density'))
print('The severity of this tumor is')
print(reg.predict([[bi_rads,age,shape,margin,density]]))


