# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 09:51:27 2021

@author: 14981
"""

#信用卡违约建模分析

#%%线性回归  ~12%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
path=r"C:\Users\14981\Desktop\python\信用卡违约数据\default of credit card clients.xls"
data=pd.read_excel(path,index_col=0,header=1)    
x=data.iloc[:,0:23]
y=data.iloc[:,23]

x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.5)
model=LinearRegression()
model.fit(x_test,y_test)
model.score(x_test,y_test)

#%%逻辑回归  ~78%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

path=r"C:\Users\14981\Desktop\python\信用卡违约数据\default of credit card clients.xls"
data=pd.read_excel(path,index_col=0,header=1)

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.5)

model=LogisticRegression()
model.fit(x_train,y_train)
model.score(x_train,y_train)

print(model.score(x_test, y_test))
print(model.score(x_train, y_train))

#%%KNN ~69%
from sklearn.neighbors import KNeighborsClassifier 
model_1=KNeighborsClassifier(n_neighbors=1) 
model_1.fit(x_train,y_train)
model_1.score(x_test,y_test)

#%%神经网络  ~77%
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
x=data.iloc[:,:23]
y=data.iloc[:,-1]
model_3=MLPClassifier(hidden_layer_sizes=(6,600))
model_3.fit(x,y)

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.5)
model_3.score(x_test,y_test)
model_3.score(x,y)

#%%决策树   ~82%
import sklearn.tree as tree
model=tree.DecisionTreeClassifier(max_depth=4)
model.fit(x_train.values,y_train.values)
model.score(x_test,y_test)
tree.plot_tree(model)

temp=data.corr()
temp.iloc[:,10]

#%%小彩蛋--桃心
import math
def drawHeart():
    t=np.linspace(0,math.pi,1000)
    x=np.sin(t)
    y=np.cos(t)+np.power(x,300/1300)
    plt.plot(x,y,color="red",linewidth=2,label="h")
    plt.plot(-x,y,color="pink",linewidth=2,label="-h")
    plt.xlabel("t")
    plt.ylabel("h")
    plt.ylim(-2,2)
    plt.xlim(-2,2)
    plt.legend()
def drawHeart_1():
    t=np.linspace(0,math.pi,1000)
    x=np.sin(t)
    y=np.cos(t)+np.power(x,2/3)
    plt.plot(x,y,color="red",linewidth=2,label="h")
    plt.plot(-x,y,color="pink",linewidth=2,label="-h")
    plt.xlabel("t")
    plt.ylabel("h")
    plt.ylim(-2,2)
    plt.xlim(-2,2)
    plt.legend()

plt.show()
drawHeart()
drawHeart_1()
