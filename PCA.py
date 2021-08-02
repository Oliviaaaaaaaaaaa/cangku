# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 22:13:33 2021

@author: Yuhang
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA, PCA
from scipy import sparse 



path=r'C:\Users\Yuhang\Desktop\python\default of credit card clients.xls'
data=pd.read_excel(path,index_col=0,header=1) 


x=data.iloc[:,:23]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)

modelPCA=PCA(n_components=23)
modelPCA.fit(x_train,y_train)
print(modelPCA.n_components_)

print(modelPCA.explained_variance_ratio_)
print(modelPCA.score(x_test,y_test))

Xsparse = sparse.csr_matrix(x_test)  # 压缩稀疏矩阵，并非 IPCA 的必要步骤
print(type(Xsparse))  # <class 'scipy.sparse.csr.csr_matrix'>
print(Xsparse.shape)  
modelIPCA = IncrementalPCA(n_components=6, batch_size=200)
modelIPCA.fit(Xsparse)  # 训练模型 modelIPCA

print(modelIPCA.n_components_)  # 返回 PCA 模型保留的主成份个数
# 6
print(modelIPCA.explained_variance_ratio_)  # 返回 PCA 模型各主成份占比

print(sum(modelIPCA.explained_variance_ratio_))  # 返回 PCA 模型各主成份占比

print(modelIPCA.singular_values_) # 返回 PCA 模型各主成份的奇异值

print(modelPCA.score(x_test,y_test))



