# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 10:30:45 2021

@author: 14981
"""

#%%雷达图
import matplotlib.pyplot as plt
import numpy as np
plt.polar(0.333*np.pi,1,'ro',lw=2)
plt.ylim(0,50)
plt.show()
theta=np.array([0.333,0.667,1,1.333,1.667,2])
r=[0.714,0.146,0.023,0.017,0.025,0.055]
plt.polar(theta*np.pi,r,'ro',lw=2)
plt.ylim(0,1)
plt.show()