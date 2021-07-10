# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:57:29 2021

Author: Mario Angarita
"""

import fialko
import numpy as np
import matplotlib.pyplot as plt

from source import Source
from data import Data
from inverse import Inverse
from mogi import Mogi
from yang import Yang
from visco_shell import ViscoShell
from penny import Penny

#Test for okada.py replicating Figure 28 of dModels report

x0=0
y0=0
z0=1000
P_G=0.01
a=1000
x=np.arange(-5000,5001,100)
y=np.arange(-5000,5001,100)

d = Data()
d.add_locs(x,y)

penny = Penny(d)

uf,vf,wf=fialko.forward(x0, y0, z0, P_G, a, x, y)
up,vp,wp=penny.forward(x0,y0,z0,P_G,a)

plt.figure()
plt.plot(x,uf,c='green',label='uf')
plt.plot(x,up,c='red',label='up')

plt.figure()
plt.plot(y,vf,c='green',label='vf')
plt.plot(y,vp,c='red',label='vp')

plt.figure()
plt.plot(y,wf,c='green',label='wf')
plt.plot(y,wp,c='red',label='wf')





plt.show()
