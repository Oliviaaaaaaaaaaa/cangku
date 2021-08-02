# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 15:17:35 2021

@author: Yuhang
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



path=r'C:\Users\Yuhang\Desktop\python\default of credit card clients.xls'
data=pd.read_excel(path,index_col=0,header=1) 

x=data.iloc[:,1:23]
y=data.iloc[:,-1]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


sample_size=15

x=np.linspace(1,sample_size,sample_size)
score_train=np.zeros(sample_size)
score_test=np.zeros(sample_size)


for i in range(1,sample_size):
    model=MLPClassifier(hidden_layer_sizes=(i,5))
    x[i]=i
    model.fit(x_train,y_train)
    y_predict=model.predict(x_test)
    #print(model.score(x_train,y_train))
    #print(model.score(x_test,y_test))

    
    score_train=model.score(x_train,y_train)
    score_test=model.score(x_test,y_test)

score_train1=score_train[1:499]
score_test1=score_test[1:499]    
x_=x[1:499]

plt.scatter(x_,score_train1,color='pink')
plt.scatter(x_,score_test1,color='maroon')


for i in range(1,sample_size+1):
    model=MLPClassifier(hidden_layer_sizes=(5,i))
    model.fit(x_train,y_train)
    y_predict=model.predict(x_test)
    #print(model.score(x_train,y_train))
    #print(model.score(x_test,y_test))   
    score_train[i]=model.score(x_train,y_train)
    score_test[i]=model.score(x_test,y_test)

score_train4=score_train[1:14]
score_test4=score_test[1:14]    
x_=x[1:14]

plt.scatter(x_,score_train4,color='pink')
plt.scatter(x_,score_test4,color='maroon')
    
    
     

