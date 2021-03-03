# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:57:29 2021

Author: Mario Angarita
"""

import fialko
import numpy as np
import matplotlib.pyplot as plt

#Test for okada.py replicating Figure 28 of dModels report

x0=0
y0=0
z0=1000
P_G=0.01
a=1000
x=np.arange(0,5001,50)
y=2*x

u,v,w=fialko.forward(x0, y0, z0, P_G, a, x, y)

plt.figure()
plt.plot(x,u,c='green',label='ux')
plt.figure()
plt.plot(y,v,c='blue',label='uy')
plt.figure()
plt.plot(y,w,c='red',label='uz')



plt.show()
